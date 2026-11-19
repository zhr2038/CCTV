from __future__ import annotations

import ipaddress
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_config
from .database import (
    connect_database,
    get_event_for_preview,
    query_camera_health,
    query_episode_events,
    query_latest_shadow_scores,
    save_event_reviews,
)
from .diary import (
    available_diary_dates,
    build_diary_payload,
    period_bounds,
    query_activity_detail,
    save_activity_review,
)
from .episodes import (
    build_episodes,
    episode_identifier,
    select_informative_review_inbox,
    select_review_inbox,
)
from .pipeline import query_pipeline_progress
from .preview import generate_event_preview
from .visual_database import query_visual_progress
from .visual_search import (
    VisualSearchService,
    search_activity_event_evidence,
    search_keyframes,
)


PREVIEW_NAME_RE = re.compile(r"^event-\d+-[a-f0-9]{12}\.mp4$")


class ReviewRequest(BaseModel):
    event_label: str
    event_ids: list[int] = Field(min_length=1, max_length=1000)
    verdict: Literal["confirmed", "false_positive", "uncertain"]
    note: str = Field(default="", max_length=2000)


class ActivityReviewRequest(BaseModel):
    verdict: Literal["confirmed", "corrected", "rejected", "uncertain"]
    corrected_activity_type: str | None = Field(
        default=None,
        max_length=80,
    )
    corrected_activity_subtype: str | None = Field(
        default=None,
        max_length=80,
    )
    corrected_start_ts: float | None = None
    corrected_end_ts: float | None = None
    note: str = Field(default="", max_length=2000)


def _date_bounds(
    value: str,
    timezone: ZoneInfo,
) -> tuple[datetime, datetime]:
    try:
        start = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="日期格式必须是 YYYY-MM-DD") from exc
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone)
    start = start.astimezone(timezone).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return start, start + timedelta(days=1)


def _available_dates(
    connection,
    *,
    event_label: str,
    timezone: ZoneInfo,
) -> list[str]:
    rows = connection.execute(
        """
        SELECT DISTINCT wall_start_ts
        FROM audio_events
        WHERE event_label = ? AND wall_start_ts IS NOT NULL
        ORDER BY wall_start_ts DESC
        """,
        (event_label,),
    )
    return list(
        dict.fromkeys(
            datetime.fromtimestamp(
                float(row["wall_start_ts"]),
                timezone,
            ).date().isoformat()
            for row in rows
        )
    )


def _available_labels(connection) -> list[dict]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT event_label AS label, COUNT(*) AS events,
                   MAX(wall_start_ts) AS latest_ts
            FROM audio_events
            GROUP BY event_label
            ORDER BY
                CASE WHEN event_label = 'baby_cry' THEN 0 ELSE 1 END,
                event_label
            """
        )
    ]


def _camera_payload(rows, timezone: ZoneInfo) -> list[dict]:
    now = time.time()
    result: list[dict] = []
    for row in rows:
        last_capture = (
            float(row["last_capture_ts"])
            if row["last_capture_ts"] is not None
            else None
        )
        age_hours = (
            max(0.0, (now - last_capture) / 3600)
            if last_capture is not None
            else None
        )
        if age_hours is None or age_hours > 48:
            status = "attention"
        elif age_hours > 24:
            status = "quiet"
        else:
            status = "recent"
        result.append(
            {
                **dict(row),
                "last_capture_iso": (
                    datetime.fromtimestamp(
                        last_capture,
                        timezone,
                    ).isoformat()
                    if last_capture is not None
                    else None
                ),
                "age_hours": age_hours,
                "health_status": status,
            }
        )
    return result


def create_app(
    *,
    database_path: str | Path,
    config_path: str | Path,
    source_root: str | Path | None = None,
    preview_dir: str | Path = ".data/review_clips",
    visual_dir: str | Path = ".data/visual/keyframes",
    ffmpeg: str | None = None,
    allowed_networks: Iterable[str] | None = None,
    models_config_path: str | Path = "config/models.toml",
) -> FastAPI:
    config = load_config(config_path)
    timezone = ZoneInfo(config.timezone)
    database = Path(database_path)
    media_root = config.source_root(
        str(source_root) if source_root is not None else None
    )
    previews = Path(preview_dir)
    visual_root = Path(visual_dir)
    static_root = Path(__file__).with_name("web_static")
    client_networks = tuple(
        ipaddress.ip_network(value, strict=False)
        for value in (allowed_networks or ())
    )
    previews.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    visual_search = VisualSearchService(
        model_cache=database.parent.joinpath("models", "huggingface"),
        database_path=database,
        models_config_path=models_config_path,
    )

    connection = connect_database(database)
    connection.close()

    app = FastAPI(
        title="家庭录像复核",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount(
        "/assets",
        StaticFiles(directory=static_root),
        name="assets",
    )

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        client_host = request.client.host if request.client else ""
        try:
            client_address = ipaddress.ip_address(client_host)
        except ValueError:
            client_address = None
        if client_networks and (
            client_address is None
            or not any(
                client_address in network
                for network in client_networks
            )
        ):
            response = JSONResponse(
                status_code=403,
                content={"detail": "此设备不在允许的家庭局域网内"},
            )
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "media-src 'self' blob:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(static_root / "index.html")

    @app.get("/diary")
    def diary_index():
        return FileResponse(static_root / "diary.html")

    @app.get("/api/diary")
    def diary_dashboard(
        date: str | None = None,
        period: Literal["day", "week", "month"] = "day",
    ):
        connection = connect_database(database)
        try:
            dates = available_diary_dates(
                connection,
                timezone=timezone,
            )
            selected = date or (
                dates[0]
                if dates
                else datetime.now(timezone).date().isoformat()
            )
            try:
                return build_diary_payload(
                    connection,
                    date=selected,
                    period=period,
                    timezone=timezone,
                    feeding_interval_seconds=(
                        config.feeding_interval_minutes * 60
                    ),
                    feeding_episode_merge_seconds=(
                        config.feeding_episode_merge_minutes * 60
                    ),
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc
        finally:
            connection.close()

    @app.get("/api/visual/progress")
    def visual_progress():
        connection = connect_database(database)
        try:
            payload = query_visual_progress(connection)
            payload["search_index"] = visual_search.index_progress(connection)
            return payload
        finally:
            connection.close()

    @app.get("/api/keyframes/{keyframe_id}/image")
    def keyframe_image(keyframe_id: int):
        connection = connect_database(database)
        try:
            row = connection.execute(
                """
                SELECT relative_image_path
                FROM visual_keyframes
                WHERE id = ?
                """,
                (keyframe_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise HTTPException(status_code=404, detail="证据画面不存在")
        relative = Path(str(row["relative_image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise HTTPException(status_code=404, detail="证据画面不存在")
        root = visual_root.resolve()
        path = root.joinpath(relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail="证据画面不存在",
            ) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="证据画面不存在")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/activities/{event_id}")
    def activity_detail(event_id: int):
        connection = connect_database(database)
        try:
            payload = query_activity_detail(connection, event_id)
        finally:
            connection.close()
        if payload is None:
            raise HTTPException(status_code=404, detail="活动事件不存在")
        return payload

    @app.post("/api/activities/{event_id}/review")
    def review_activity(event_id: int, request: ActivityReviewRequest):
        connection = connect_database(database)
        try:
            try:
                save_activity_review(
                    connection,
                    event_id=event_id,
                    verdict=request.verdict,
                    corrected_activity_type=request.corrected_activity_type,
                    corrected_activity_subtype=(
                        request.corrected_activity_subtype
                    ),
                    corrected_start_ts=request.corrected_start_ts,
                    corrected_end_ts=request.corrected_end_ts,
                    note=request.note,
                )
            except LookupError as exc:
                raise HTTPException(
                    status_code=404,
                    detail=str(exc),
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc
            return {"event_id": event_id, "verdict": request.verdict}
        finally:
            connection.close()

    @app.post("/api/activities/{event_id}/preview")
    async def activity_preview(event_id: int):
        connection = connect_database(database)
        try:
            event = query_activity_detail(connection, event_id)
        finally:
            connection.close()
        if event is None or not event.get("evidence"):
            raise HTTPException(status_code=404, detail="活动证据不存在")
        evidence = event["evidence"][0]
        preview_event = {
            "event_id": 1_000_000_000 + event_id,
            "relative_path": evidence["relative_path"],
            "mtime_ns": evidence["mtime_ns"],
            "media_start_seconds": evidence["media_offset_seconds"],
            "media_end_seconds": evidence["media_offset_seconds"] + 1.0,
            "duration_seconds": evidence["media_duration_seconds"],
        }
        try:
            result = await run_in_threadpool(
                generate_event_preview,
                preview_event,
                source_root=media_root,
                output_dir=previews,
                ffmpeg=ffmpeg,
                before_seconds=8.0,
                after_seconds=12.0,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "url": f"/previews/{result.path.name}",
            "cached": result.cached,
            "encoder": result.encoder,
            "source_start_seconds": result.source_start_seconds,
            "duration_seconds": result.duration_seconds,
        }

    @app.get("/api/visual/search")
    async def visual_search_endpoint(
        q: str = Query(min_length=1, max_length=200),
        date: str | None = None,
        period: Literal["day", "week", "month"] = "month",
        camera: list[str] = Query(default=[]),
        limit: int = Query(default=40, ge=1, le=100),
        deep_verify: bool = False,
    ):
        since_ts = None
        until_ts = None
        if date:
            try:
                start, end = period_bounds(date, timezone, period)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc
            since_ts = start.timestamp()
            until_ts = end.timestamp()
        connection = connect_database(database)
        try:
            archive_search_model = visual_search.select_model(connection)
            # Keep the page responsive while the batch Qwen3 verifier owns
            # the GPU.  The compatible SigLIP2 index remains complete and its
            # query encoder runs on CPU; structured Qwen3 events are returned
            # ahead of similarity candidates.
            search_model = visual_search.model_name
            event_results = search_activity_event_evidence(
                connection,
                query=q,
                since_ts=since_ts,
                until_ts=until_ts,
                camera_ids=tuple(dict.fromkeys(camera)),
                limit=min(12, limit),
            )
            indexed_keyframes = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM visual_embeddings
                    WHERE model_name = ?
                    """,
                    (search_model,),
                ).fetchone()[0]
            )
        finally:
            connection.close()
        use_qwen_search = False
        recalled_results: list[dict] = []
        # Known activity queries can be answered directly from events already
        # verified during the batch pipeline. Return them without paying any
        # model cold-start cost.
        if not event_results or deep_verify:
            query_vector = await run_in_threadpool(
                visual_search.encode_query,
                q,
                search_model,
            )
            connection = connect_database(database)
            try:
                recalled_results = search_keyframes(
                    connection,
                    query_vector=query_vector,
                    model_name=search_model,
                    since_ts=since_ts,
                    until_ts=until_ts,
                    camera_ids=tuple(dict.fromkeys(camera)),
                    limit=limit,
                )
            finally:
                connection.close()
        verified_results: list[dict] = []
        if use_qwen_search and deep_verify:
            verified_results = await run_in_threadpool(
                visual_search.verify_results,
                q,
                recalled_results,
                image_root=visual_root,
                max_candidates=min(36, limit * 3),
            )
        verified_ids = {
            int(item["keyframe_id"]) for item in verified_results
        }
        for item in verified_results:
            item["match_source"] = "vector_recall"
            item["verification_status"] = "qwen_verified"
        possible_results = []
        for item in recalled_results:
            if int(item["keyframe_id"]) in verified_ids:
                continue
            possible = dict(item)
            possible["match_source"] = "vector_recall"
            possible["verification_status"] = "possible"
            possible_results.append(possible)

        results: list[dict] = []
        seen_keyframes: set[int] = set()
        for item in [
            *event_results,
            *verified_results,
            *possible_results,
        ]:
            keyframe_id = int(item["keyframe_id"])
            if keyframe_id in seen_keyframes:
                continue
            seen_keyframes.add(keyframe_id)
            results.append(item)
            if len(results) >= limit:
                break
        status_counts = Counter(
            str(item.get("verification_status") or "possible")
            for item in results
        )
        return {
            "query": q,
            "timezone": config.timezone,
            "results": results[:limit],
            "indexed_keyframes": indexed_keyframes,
            "search_model": search_model,
            "archive_search_model": archive_search_model,
            "verified_by_vlm": use_qwen_search and deep_verify,
            "deep_verification_available": False,
            "status_counts": dict(status_counts),
            "notice": (
                "已有结构化事件优先展示；其余结果保留为可能相关，"
                "不会因本地视觉模型无法确认而被静默丢弃。不进行人脸识别。"
                if deep_verify
                else "已有 Qwen3 事件优先展示；相似画面由轻量本地索引召回，"
                "不会与正在运行的批处理争抢显卡。"
            ),
        }

    @app.get("/api/dashboard")
    def dashboard(
        date: str | None = None,
        label: str = "baby_cry",
        camera: str | None = None,
        minimum_confidence: float = Query(default=0.60, ge=0, le=1),
        gap_seconds: float = Query(default=30.0, ge=0, le=600),
        review_mode: Literal["smart", "all"] = "smart",
        daily_quota: int = Query(default=5, ge=1, le=30),
    ):
        connection = connect_database(database)
        try:
            dates = _available_dates(
                connection,
                event_label=label,
                timezone=timezone,
            )
            selected_date = date or (dates[0] if dates else None)
            since_ts = None
            until_ts = None
            if selected_date:
                start, end = _date_bounds(selected_date, timezone)
                since_ts = start.timestamp()
                until_ts = end.timestamp()
            rows = query_episode_events(
                connection,
                event_label=label,
                camera_id=camera,
                since_ts=since_ts,
                until_ts=until_ts,
                minimum_confidence=minimum_confidence,
            )
            all_episodes = build_episodes(rows, gap_seconds=gap_seconds)
            shadow_run, shadow_scores = query_latest_shadow_scores(
                connection,
                event_label=label,
            )
            shadow_active = bool(
                shadow_run
                and abs(
                    float(shadow_run["minimum_confidence"])
                    - minimum_confidence
                )
                < 1e-9
                and abs(float(shadow_run["gap_seconds"]) - gap_seconds)
                < 1e-9
            )
            if shadow_active:
                for episode in all_episodes:
                    shadow = shadow_scores.get(
                        str(episode["episode_id"])
                    )
                    if shadow is None:
                        continue
                    episode.update(
                        {
                            "shadow_score": float(shadow["score"]),
                            "shadow_bucket": str(shadow["bucket"]),
                            "shadow_review_priority": float(
                                shadow["review_priority"]
                            ),
                            "shadow_reason": str(shadow["reason"]),
                            "shadow_run_id": int(shadow["run_id"]),
                        }
                    )
            if review_mode == "smart":
                if shadow_active:
                    episodes = select_informative_review_inbox(
                        all_episodes,
                        shadow_scores=shadow_scores,
                        quota=daily_quota,
                    )
                else:
                    episodes = select_review_inbox(
                        all_episodes,
                        quota=daily_quota,
                    )
            else:
                episodes = [
                    {
                        **episode,
                        "review_lane": "all",
                        "review_reason": "完整候选列表",
                        "review_rank": rank,
                    }
                    for rank, episode in enumerate(all_episodes, start=1)
                ]
            shadow_selection_used = bool(
                review_mode == "smart"
                and shadow_active
                and any("shadow_score" in episode for episode in episodes)
            )
            status_counts = Counter(
                episode["review_status"] for episode in episodes
            )
            all_status_counts = Counter(
                episode["review_status"] for episode in all_episodes
            )
            actionable = (
                status_counts["pending"] + status_counts["mixed"]
            )
            completed = (
                status_counts["confirmed"]
                + status_counts["false_positive"]
                + status_counts["uncertain"]
            )
            archived = max(0, len(all_episodes) - len(episodes))
            latest_scan = connection.execute(
                "SELECT MAX(finished_at) AS value FROM scan_runs"
            ).fetchone()["value"]
            indexed_files = int(
                connection.execute(
                    "SELECT COUNT(*) FROM media_files"
                ).fetchone()[0]
            )
            return {
                "timezone": config.timezone,
                "selected_date": selected_date,
                "available_dates": dates,
                "labels": _available_labels(connection),
                "cameras": _camera_payload(
                    query_camera_health(connection),
                    timezone,
                ),
                "episodes": episodes,
                "summary": {
                    "episodes": len(episodes),
                    "total_episodes": len(all_episodes),
                    "raw_events": len(rows),
                    "pending": status_counts["pending"],
                    "actionable": actionable,
                    "completed": completed,
                    "confirmed": status_counts["confirmed"],
                    "false_positive": status_counts["false_positive"],
                    "uncertain": status_counts["uncertain"],
                    "mixed": status_counts["mixed"],
                    "archived": archived,
                    "total_pending": (
                        all_status_counts["pending"]
                        + all_status_counts["mixed"]
                    ),
                },
                "review_inbox": {
                    "mode": review_mode,
                    "quota": daily_quota if review_mode == "smart" else None,
                    "strategy": (
                        "shadow_active_learning"
                        if shadow_selection_used
                        else "balanced_fixed"
                    ),
                    "selected": len(episodes),
                    "pending": actionable,
                    "completed": completed,
                    "archived": archived,
                    "total_episodes": len(all_episodes),
                    "lane_counts": dict(
                        Counter(
                            episode["review_lane"]
                            for episode in episodes
                        )
                    ),
                },
                "shadow_calibration": {
                    "available": shadow_run is not None,
                    "active_for_selection": shadow_active,
                    "used_for_current_inbox": shadow_selection_used,
                    "decision_mode": "shadow_only",
                    "notice": (
                        "影子评分只选择抽查样本，不会自动写入人工结论"
                    ),
                    "run_id": (
                        int(shadow_run["id"]) if shadow_run else None
                    ),
                    "finished_at_iso": (
                        datetime.fromtimestamp(
                            int(shadow_run["finished_at"]),
                            timezone,
                        ).isoformat()
                        if shadow_run and shadow_run["finished_at"]
                        else None
                    ),
                    "positive_episodes": (
                        int(shadow_run["positive_episodes"])
                        if shadow_run
                        else 0
                    ),
                    "negative_episodes": (
                        int(shadow_run["negative_episodes"])
                        if shadow_run
                        else 0
                    ),
                    "fingerprinted_episodes": (
                        int(shadow_run["fingerprinted_episodes"])
                        if shadow_run
                        else 0
                    ),
                    "validation": (
                        shadow_run["metrics"].get(
                            "leave_one_episode_out",
                            {},
                        )
                        if shadow_run
                        else {}
                    ),
                },
                "scope": {
                    "source_read_only": config.source_read_only,
                    "cloud_upload": config.cloud_upload,
                    "transcription_mode": config.transcription_mode,
                    "minimum_confidence": minimum_confidence,
                    "gap_seconds": gap_seconds,
                    "review_mode": review_mode,
                    "daily_quota": daily_quota,
                    "allowed_networks": [
                        str(network) for network in client_networks
                    ],
                    "indexed_files": indexed_files,
                    "last_scan_iso": (
                        datetime.fromtimestamp(
                            latest_scan,
                            timezone,
                        ).isoformat()
                        if latest_scan is not None
                        else None
                    ),
                },
                "pipeline": query_pipeline_progress(connection),
            }
        finally:
            connection.close()

    @app.get("/api/pipeline")
    def pipeline_progress():
        connection = connect_database(database)
        try:
            return query_pipeline_progress(connection)
        finally:
            connection.close()

    @app.post("/api/episodes/{episode_id}/review")
    def review_episode(episode_id: str, request: ReviewRequest):
        if (
            episode_identifier(request.event_label, request.event_ids)
            != episode_id
        ):
            raise HTTPException(status_code=400, detail="事件段校验失败")
        connection = connect_database(database)
        try:
            placeholders = ",".join("?" for _ in request.event_ids)
            labels = {
                str(row["event_label"])
                for row in connection.execute(
                    f"""
                    SELECT event_label
                    FROM audio_events
                    WHERE id IN ({placeholders})
                    """,
                    request.event_ids,
                )
            }
            if labels != {request.event_label}:
                raise HTTPException(status_code=400, detail="事件标签不一致")
            updated = save_event_reviews(
                connection,
                event_ids=request.event_ids,
                verdict=request.verdict,
                note=request.note,
            )
            return {
                "episode_id": episode_id,
                "verdict": request.verdict,
                "updated_events": updated,
            }
        finally:
            connection.close()

    @app.post("/api/events/{event_id}/preview")
    async def create_preview(event_id: int):
        connection = connect_database(database)
        try:
            event = get_event_for_preview(connection, event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="候选事件不存在")
            event_payload = dict(event)
        finally:
            connection.close()
        try:
            result = await run_in_threadpool(
                generate_event_preview,
                event_payload,
                source_root=media_root,
                output_dir=previews,
                ffmpeg=ffmpeg,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "url": f"/previews/{result.path.name}",
            "cached": result.cached,
            "encoder": result.encoder,
            "source_start_seconds": result.source_start_seconds,
            "duration_seconds": result.duration_seconds,
        }

    @app.get("/previews/{filename}")
    def preview_file(filename: str):
        if not PREVIEW_NAME_RE.match(filename):
            raise HTTPException(status_code=404, detail="预览不存在")
        path = previews / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="预览不存在")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "source_read_only": config.source_read_only,
            "database": str(database.resolve()),
        }

    return app
