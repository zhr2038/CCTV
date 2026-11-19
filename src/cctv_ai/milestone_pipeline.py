from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Mapping, Sequence

from .model_config import ModelsConfig
from .vl_models import VisionAnalysisContext
from .vl_models.tiered import (
    MILESTONE_ACTIVITIES,
    TieredVisionAnalyzer,
    save_confirmed_milestone,
)


MILESTONE_PROMPTS = {
    "sitting": (
        "判断连续画面是否清楚显示宝宝不依靠成人手部支撑，保持坐姿。"
        "仅仅被抱着、靠在成人或床栏上不能算独立坐稳。"
    ),
    "crawling": (
        "判断连续画面是否清楚显示宝宝用手膝或匍匐动作主动向前移动。"
        "单帧趴着或成人移动宝宝不能算爬行。"
    ),
    "standing": (
        "判断连续画面是否清楚显示宝宝双脚承重保持站立姿态。"
        "成人抱起、跪姿或只有上半身可见不能算站立。"
    ),
    "rolling": (
        "判断连续画面是否清楚显示宝宝主动从仰卧翻到俯卧，或从俯卧"
        "翻到仰卧。单帧侧躺不能证明完成翻身。"
    ),
}


def _already_confirmed(
    connection: sqlite3.Connection,
) -> set[str]:
    result: set[str] = set()
    for row in connection.execute(
        """
        SELECT details_json
        FROM activity_events
        WHERE json_type(details_json, '$.growth_milestone') IS NOT NULL
        """
    ):
        try:
            details = json.loads(row["details_json"] or "{}")
            milestone = details.get("growth_milestone", {}).get("kind")
        except (TypeError, ValueError, AttributeError):
            milestone = None
        if milestone:
            result.add(str(milestone))
    return result


def _candidate_events(
    connection: sqlite3.Connection,
    *,
    since_ts: int,
    until_ts: int,
    limit: int,
    high_model: str,
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT
                e.id, e.activity_type, e.start_ts, e.end_ts,
                e.confidence, e.camera_ids_json, e.location
            FROM activity_events e
            WHERE e.start_ts >= ?
              AND e.start_ts < ?
              AND e.activity_type IN ('sitting', 'crawling', 'standing')
              AND e.review_status NOT IN ('rejected', 'uncertain')
              AND NOT (
                  COALESCE(
                      json_extract(
                          e.details_json,
                          '$.growth_milestone_review.high_model'
                      ),
                      ''
                  ) = ?
                  AND COALESCE(
                      json_extract(
                          e.details_json,
                          '$.growth_milestone_review.status'
                      ),
                      ''
                  ) IN (
                      'rejected_by_32b',
                      'candidate_below_threshold',
                      'ordinary_event'
                  )
              )
            ORDER BY e.start_ts, e.id
            LIMIT ?
            """,
            (
                since_ts,
                until_ts,
                high_model,
                max(0, limit),
            ),
        )
    )


def _save_review(
    connection: sqlite3.Connection,
    *,
    activity_event_id: int,
    primary_model: str,
    high_model: str,
    analysis: Mapping[str, Any],
) -> None:
    row = connection.execute(
        "SELECT details_json FROM activity_events WHERE id = ?",
        (activity_event_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Unknown activity event: {activity_event_id}")
    try:
        details = json.loads(row["details_json"] or "{}")
    except (TypeError, ValueError):
        details = {}
    details["growth_milestone_review"] = {
        "primary_model": primary_model,
        "high_model": high_model,
        "status": analysis["status"],
        "milestone": analysis["milestone"],
        "analysis": dict(analysis),
        "reviewed_at": int(time.time()),
    }
    connection.execute(
        """
        UPDATE activity_events
        SET details_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            json.dumps(
                details,
                ensure_ascii=False,
                sort_keys=True,
            ),
            int(time.time()),
            activity_event_id,
        ),
    )


def _evidence(
    connection: sqlite3.Connection,
    event_id: int,
    *,
    image_root: Path,
) -> tuple[list[Path], VisionAnalysisContext] | None:
    rows = list(
        connection.execute(
            """
            SELECT
                k.id, k.wall_ts, k.relative_image_path, k.motion_score,
                c.name AS camera_name
            FROM activity_event_evidence a
            JOIN visual_keyframes k ON k.id = a.keyframe_id
            JOIN media_files m ON m.id = k.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            WHERE a.activity_event_id = ?
            ORDER BY a.evidence_rank
            LIMIT 5
            """,
            (event_id,),
        )
    )
    paths = [
        image_root.joinpath(str(row["relative_image_path"]))
        for row in rows
    ]
    valid_paths = [path for path in paths if path.is_file()]
    if not rows or not valid_paths:
        return None
    keyframe_ids = [int(row["id"]) for row in rows]
    placeholders = ",".join("?" for _ in keyframe_ids)
    detections = [
        dict(row)
        for row in connection.execute(
            f"""
            SELECT keyframe_id, class_name, confidence, x1, y1, x2, y2
            FROM visual_detections
            WHERE keyframe_id IN ({placeholders})
              AND confidence >= 0.18
            ORDER BY confidence DESC
            LIMIT 30
            """,
            keyframe_ids,
        )
    ]
    object_names = tuple(
        dict.fromkeys(str(item["class_name"]) for item in detections)
    )
    motion_values = [
        float(row["motion_score"])
        for row in rows
        if row["motion_score"] is not None
    ]
    motion = (
        "unknown"
        if not motion_values
        else "active" if max(motion_values) >= 0.12
        else "still" if max(motion_values) <= 0.03
        else "low"
    )
    return (
        valid_paths,
        VisionAnalysisContext(
            camera="、".join(
                dict.fromkeys(str(row["camera_name"]) for row in rows)
            ),
            time=datetime.fromtimestamp(float(rows[0]["wall_ts"]))
            .astimezone()
            .isoformat(sep=" ", timespec="seconds"),
            objects=object_names,
            motion=motion,
            detections=tuple(detections),
        ),
    )


def verify_growth_milestones(
    connection: sqlite3.Connection,
    *,
    models: ModelsConfig,
    image_root: str | Path,
    since_ts: int,
    until_ts: int,
    limit: int = 12,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    confirmed_kinds = _already_confirmed(connection)
    candidates = _candidate_events(
        connection,
        since_ts=since_ts,
        until_ts=until_ts,
        limit=limit,
        high_model=models.high_value.model_id,
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "candidates": len(candidates),
        "analyzed": 0,
        "confirmed": 0,
        "results": [],
    }
    if not candidates:
        return payload
    root = Path(image_root).resolve()
    analyzer = TieredVisionAnalyzer(
        models.vision_language,
        models.high_value,
    )
    try:
        for row in candidates:
            activity = str(row["activity_type"])
            milestone = MILESTONE_ACTIVITIES.get(activity)
            if milestone is None or milestone in confirmed_kinds:
                continue
            evidence = _evidence(
                connection,
                int(row["id"]),
                image_root=root,
            )
            if evidence is None:
                continue
            frames, context = evidence
            analysis = analyzer.analyze_video_frames(
                frames,
                MILESTONE_PROMPTS[activity],
                context,
            )
            saved = save_confirmed_milestone(
                connection,
                activity_event_id=int(row["id"]),
                analysis=analysis,
            )
            _save_review(
                connection,
                activity_event_id=int(row["id"]),
                primary_model=models.vision_language.model_id,
                high_model=models.high_value.model_id,
                analysis=analysis.to_dict(),
            )
            connection.commit()
            payload["analyzed"] += 1
            payload["confirmed"] += int(saved)
            payload["results"].append(
                {
                    "activity_event_id": int(row["id"]),
                    "activity": activity,
                    "analysis": analysis.to_dict(),
                    "saved": saved,
                }
            )
            if saved:
                confirmed_kinds.add(milestone)
            if on_progress is not None:
                on_progress(dict(payload))
    finally:
        analyzer.close()
    return payload
