from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .activity_inference import rebuild_activity_events
from .config import load_config
from .database import (
    connect_database,
    enqueue_missing_audio_jobs_for_scope,
    get_queued_jobs,
    get_ready_files,
    mark_job_error,
    mark_job_running,
    save_probe,
    save_time_segments,
)
from .media import probe_media
from .models import MediaProbe
from .pipeline import PipelineLock
from .visual_database import (
    complete_visual_jobs,
    enqueue_visual_jobs_for_scope,
    finish_visual_run,
    query_keyframes_for_media,
    query_visual_progress,
    save_detections,
    save_embeddings_and_observations,
    save_extracted_keyframes,
    start_visual_run,
    update_visual_run,
)
from .visual_extract import EXTRACTOR_VERSION, extract_media_keyframes
from .visual_models import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    PROMPT_VERSION,
    SiglipVisualEncoder,
    YoloDetector,
)
from .visual_watermark import (
    VisualWatermarkRecognizer,
    map_keyframe_watermarks,
)


DEFAULT_CAMERAS = (
    "lumi1.54ef4482e358",
    "lumi3.8e2ce0d05d9c251e",
    "50ec503b63bf",
    "B88880620430",
)


def _retry_sqlite_write(
    operation: Callable[[], Any],
    *,
    attempts: int = 8,
) -> Any:
    """Retry a short SQLite write when another local service briefly owns it."""
    delay_seconds = 0.10
    for attempt in range(attempts):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if (
                "locked" not in message
                and "busy" not in message
            ) or attempt + 1 >= attempts:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 2.0)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class VisualPipelineOptions:
    database_path: Path
    config_path: Path
    source_root: Path
    output_root: Path
    since_ts: int
    until_ts: int
    camera_ids: tuple[str, ...] = DEFAULT_CAMERAS
    ffprobe: str | None = None
    ffmpeg: str | None = None
    hwaccel: str | None = "cuda"
    extractor_version: str = EXTRACTOR_VERSION
    detector_model: str = DEFAULT_DETECTOR_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    model_cache: Path = Path(".data/models/huggingface")
    probe_limit: int = 2048
    probe_workers: int = 24
    extract_limit: int = 2048
    extract_workers: int = 16
    watermark_limit: int = 256
    watermark_workers: int = 4
    watermark_interval_seconds: float = 120.0
    inference_file_limit: int = 2048
    inference_frame_limit: int = 8192
    detector_batch_size: int = 128
    embedding_batch_size: int = 96
    target_width: int = 640
    bedside_interval_seconds: float = 30.0
    kitchen_interval_seconds: float = 20.0
    error_retry_delay_seconds: int = 3600
    lock_path: Path = Path(".data/cctv-visual-pipeline.lock")

    def validate(self) -> None:
        if self.until_ts <= self.since_ts:
            raise ValueError("until_ts must be after since_ts")
        for name in (
            "probe_limit",
            "probe_workers",
            "extract_limit",
            "extract_workers",
            "watermark_limit",
            "watermark_workers",
            "inference_file_limit",
            "inference_frame_limit",
            "detector_batch_size",
            "embedding_batch_size",
            "target_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bedside_interval_seconds <= 0:
            raise ValueError("bedside_interval_seconds must be positive")
        if self.kitchen_interval_seconds <= 0:
            raise ValueError("kitchen_interval_seconds must be positive")
        if self.watermark_interval_seconds <= 0:
            raise ValueError("watermark_interval_seconds must be positive")
        if not self.camera_ids:
            raise ValueError("At least one camera is required")

    def serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


class VisualPipelineRuntime:
    def __init__(self) -> None:
        self._detector: YoloDetector | None = None
        self._detector_key: tuple[str, int] | None = None
        self._encoder: SiglipVisualEncoder | None = None
        self._encoder_key: tuple[str, str] | None = None

    def detector(self, options: VisualPipelineOptions) -> YoloDetector:
        key = (options.detector_model, options.target_width)
        if self._detector is None or self._detector_key != key:
            self._detector = YoloDetector(
                options.detector_model,
                device="0",
                image_size=options.target_width,
            )
            self._detector_key = key
        return self._detector

    def encoder(self, options: VisualPipelineOptions) -> SiglipVisualEncoder:
        key = (options.embedding_model, str(options.model_cache))
        if self._encoder is None or self._encoder_key != key:
            self._encoder = SiglipVisualEncoder(
                options.embedding_model,
                device="cuda",
                local_cache=options.model_cache,
            )
            self._encoder_key = key
        return self._encoder


def _source_path(source_root: Path, relative_path: str) -> Path:
    parts = [
        part
        for part in relative_path.replace("\\", "/").split("/")
        if part
    ]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("Unsafe media path")
    return source_root.joinpath(*parts)


def _recover_visual_jobs(connection, *, stale_seconds: int = 0) -> int:
    now_ts = int(time.time())
    cursor = connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'queued',
            attempts = MAX(0, attempts - 1),
            payload_json = ?,
            updated_at = ?
        WHERE status = 'running'
          AND job_type IN (
              'keyframe_extract', 'object_detection', 'visual_embedding',
              'watermark_time_map'
          )
          AND updated_at <= ?
        """,
        (
            json.dumps(
                {"notice": "Recovered interrupted visual job"},
                ensure_ascii=False,
            ),
            now_ts,
            now_ts - stale_seconds,
        ),
    )
    return int(cursor.rowcount)


def _camera_lookup(config) -> dict[str, Any]:
    return {camera.id: camera for camera in config.cameras}


def _validated_bedside_profile(
    connection,
    options: VisualPipelineOptions,
) -> dict[str, Any] | None:
    """Return a profile only after enough real probes agree.

    Bedside bundles keep MP4 metadata at the far end of a 128 MB file. Many
    concurrent ffprobe calls therefore make a spinning NAS seek between large
    files. Once at least 100 real probes in this exact scope agree on codec,
    dimensions and audio layout, the remaining bundles can use that profile
    and their filename time envelope. Downstream keyframe extraction still
    verifies that every individual file actually decodes.
    """
    total = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM media_files
            WHERE camera_id = 'B88880620430'
              AND status = 'probed'
              AND capture_start_ts >= ?
              AND capture_start_ts < ?
              AND video_codec IS NOT NULL
              AND width IS NOT NULL
              AND height IS NOT NULL
              AND audio_codec IS NOT NULL
            """,
            (options.since_ts, options.until_ts),
        ).fetchone()[0]
    )
    if total < 100:
        return None
    row = connection.execute(
        """
        SELECT
            video_codec,
            width,
            height,
            audio_codec,
            audio_sample_rate,
            audio_channels,
            AVG(fps) AS fps,
            COUNT(*) AS profile_count
        FROM media_files
        WHERE camera_id = 'B88880620430'
          AND status = 'probed'
          AND capture_start_ts >= ?
          AND capture_start_ts < ?
          AND video_codec IS NOT NULL
          AND width IS NOT NULL
          AND height IS NOT NULL
          AND audio_codec IS NOT NULL
        GROUP BY
            video_codec, width, height,
            audio_codec, audio_sample_rate, audio_channels
        ORDER BY profile_count DESC
        LIMIT 1
        """,
        (options.since_ts, options.until_ts),
    ).fetchone()
    if row is None or int(row["profile_count"]) != total:
        return None
    return dict(row)


def _profile_probe(
    row: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> MediaProbe | None:
    start = row["capture_start_ts"]
    end = row["capture_end_hint_ts"]
    if start is None or end is None or float(end) <= float(start):
        return None
    duration = float(end) - float(start)
    size_bytes = int(row["size_bytes"] or 0)
    return MediaProbe(
        duration_seconds=duration,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        size_bytes=size_bytes,
        bit_rate=(
            round(size_bytes * 8 / duration)
            if size_bytes and duration > 0
            else None
        ),
        video_codec=str(profile["video_codec"]),
        width=int(profile["width"]),
        height=int(profile["height"]),
        fps=(
            float(profile["fps"])
            if profile.get("fps") is not None
            else None
        ),
        audio_codec=str(profile["audio_codec"]),
        audio_sample_rate=(
            int(profile["audio_sample_rate"])
            if profile.get("audio_sample_rate") is not None
            else None
        ),
        audio_channels=(
            int(profile["audio_channels"])
            if profile.get("audio_channels") is not None
            else None
        ),
    )


def _prepare_media(
    connection,
    options: VisualPipelineOptions,
    *,
    camera_lookup: Mapping[str, Any],
    audio_enabled: bool,
    audio_event_labels: tuple[str, ...],
) -> dict[str, int]:
    rows = get_ready_files(
        connection,
        options.probe_limit,
        camera_ids=options.camera_ids,
        since_ts=options.since_ts,
        until_ts=options.until_ts,
        newest_first=False,
    )
    stage = {
        "selected": len(rows),
        "completed": 0,
        "failed": 0,
        "profile_inferred": 0,
        "audio_jobs_repaired": 0,
    }
    if not rows:
        enqueue_visual_jobs_for_scope(
            connection,
            camera_ids=options.camera_ids,
            since_ts=options.since_ts,
            until_ts=options.until_ts,
        )
        if audio_enabled:
            stage["audio_jobs_repaired"] = (
                enqueue_missing_audio_jobs_for_scope(
                    connection,
                    camera_ids=options.camera_ids,
                    since_ts=options.since_ts,
                    until_ts=options.until_ts,
                    audio_event_labels=audio_event_labels,
                )
            )
        connection.commit()
        return stage

    bedside_profile = _validated_bedside_profile(connection, options)

    def inspect(row):
        if (
            str(row["camera_id"]) == "B88880620430"
            and bedside_profile is not None
        ):
            inferred = _profile_probe(row, bedside_profile)
            if inferred is not None:
                return inferred, "validated_camera_profile"
        path = _source_path(
            options.source_root,
            str(row["relative_path"]),
        )
        return probe_media(path, ffprobe=options.ffprobe), "ffprobe"

    workers = min(options.probe_workers, len(rows))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="cctv-visual-probe",
    ) as executor:
        futures = {executor.submit(inspect, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                probe, probe_source = future.result()
                save_probe(
                    connection,
                    row,
                    probe,
                    audio_enabled=(
                        audio_enabled
                        and bool(
                            camera_lookup[str(row["camera_id"])].audio_enabled
                        )
                    ),
                    audio_event_labels=audio_event_labels,
                )
                stage["completed"] += 1
                stage["profile_inferred"] += int(
                    probe_source == "validated_camera_profile"
                )
            except Exception as exc:
                # Keep the normal media retry semantics, but do not let one
                # corrupt file stall a month-long visual backfill.
                connection.execute(
                    """
                    UPDATE media_files
                    SET status = 'error',
                        probe_attempts = probe_attempts + 1,
                        last_error = ?
                    WHERE id = ?
                    """,
                    (str(exc)[:2000], int(row["id"])),
                )
                stage["failed"] += 1
            if (stage["completed"] + stage["failed"]) % 100 == 0:
                connection.commit()
    enqueue_visual_jobs_for_scope(
        connection,
        camera_ids=options.camera_ids,
        since_ts=options.since_ts,
        until_ts=options.until_ts,
    )
    if audio_enabled:
        stage["audio_jobs_repaired"] = (
            enqueue_missing_audio_jobs_for_scope(
                connection,
                camera_ids=options.camera_ids,
                since_ts=options.since_ts,
                until_ts=options.until_ts,
                audio_event_labels=audio_event_labels,
            )
        )
    connection.commit()
    return stage


def _extract_keyframes(
    connection,
    options: VisualPipelineOptions,
) -> dict[str, Any]:
    rows = get_queued_jobs(
        connection,
        "keyframe_extract",
        options.extract_limit,
        camera_ids=options.camera_ids,
        since_ts=options.since_ts,
        until_ts=options.until_ts,
        newest_first=False,
        error_retry_delay_seconds=options.error_retry_delay_seconds,
    )
    claimed: list[dict[str, Any]] = []
    for row in rows:
        if mark_job_running(connection, int(row["job_id"])):
            claimed.append(dict(row))
    connection.commit()
    stage: dict[str, Any] = {
        "selected": len(rows),
        "claimed": len(claimed),
        "completed": 0,
        "failed": 0,
        "keyframes": 0,
        "written_bytes": 0,
    }
    if not claimed:
        return stage

    def extract(row):
        return extract_media_keyframes(
            row,
            source_root=options.source_root,
            output_root=options.output_root,
            ffmpeg=options.ffmpeg,
            hwaccel=options.hwaccel,
            target_width=options.target_width,
            bedside_interval_seconds=options.bedside_interval_seconds,
            kitchen_interval_seconds=options.kitchen_interval_seconds,
            extractor_version=options.extractor_version,
        )

    workers = min(options.extract_workers, len(claimed))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="cctv-keyframe",
    ) as executor:
        futures = {executor.submit(extract, row): row for row in claimed}
        for future in as_completed(futures):
            row = futures[future]
            try:
                keyframes, metadata = future.result()
                _retry_sqlite_write(
                    lambda: save_extracted_keyframes(
                        connection,
                        job_id=int(row["job_id"]),
                        media=row,
                        keyframes=keyframes,
                        output_root=options.output_root,
                        extractor_version=options.extractor_version,
                        metadata=metadata,
                    )
                )
                stage["completed"] += 1
                stage["keyframes"] += len(keyframes)
                stage["written_bytes"] += int(metadata["written_bytes"])
            except Exception as exc:
                _retry_sqlite_write(
                    lambda: mark_job_error(
                        connection,
                        int(row["job_id"]),
                        str(exc),
                    )
                )
                stage["failed"] += 1
            if (stage["completed"] + stage["failed"]) % 25 == 0:
                _retry_sqlite_write(connection.commit)
    _retry_sqlite_write(connection.commit)
    return stage


def _jobs_with_keyframes(
    connection,
    options: VisualPipelineOptions,
    job_type: str,
) -> tuple[list[dict[str, Any]], list[Any]]:
    candidates = get_queued_jobs(
        connection,
        job_type,
        options.inference_file_limit,
        camera_ids=options.camera_ids,
        since_ts=options.since_ts,
        until_ts=options.until_ts,
        newest_first=False,
        error_retry_delay_seconds=options.error_retry_delay_seconds,
    )
    selected: list[dict[str, Any]] = []
    frames: list[Any] = []
    frame_total = 0
    for candidate in candidates:
        media_frames = query_keyframes_for_media(
            connection,
            [int(candidate["id"])],
            extractor_version=options.extractor_version,
        )
        if not media_frames:
            continue
        if selected and (
            frame_total + len(media_frames) > options.inference_frame_limit
        ):
            break
        if mark_job_running(connection, int(candidate["job_id"])):
            selected.append(dict(candidate))
            frames.extend(media_frames)
            frame_total += len(media_frames)
        if frame_total >= options.inference_frame_limit:
            break
    connection.commit()
    return selected, frames


def _map_visual_watermarks(
    connection,
    options: VisualPipelineOptions,
    *,
    camera_lookup: Mapping[str, Any],
) -> dict[str, Any]:
    rows = get_queued_jobs(
        connection,
        "watermark_time_map",
        options.watermark_limit,
        camera_id="B88880620430",
        since_ts=options.since_ts,
        until_ts=options.until_ts,
        newest_first=False,
        error_retry_delay_seconds=options.error_retry_delay_seconds,
    )
    claimed: list[tuple[dict[str, Any], list[Any]]] = []
    for row in rows:
        frames = query_keyframes_for_media(
            connection,
            [int(row["id"])],
            extractor_version=options.extractor_version,
        )
        if not frames:
            continue
        if mark_job_running(connection, int(row["job_id"])):
            claimed.append((dict(row), frames))
    connection.commit()
    stage: dict[str, Any] = {
        "selected": len(rows),
        "claimed": len(claimed),
        "completed": 0,
        "failed": 0,
        "frames_selected": 0,
        "recognized": 0,
    }
    if not claimed:
        return stage
    camera = camera_lookup["B88880620430"]
    if camera.watermark_crop is None:
        raise ValueError("Bedside camera watermark crop is not configured")
    local_state = threading.local()
    timezone_name = load_config(options.config_path).timezone

    def analyze(item):
        row, frames = item
        recognizer = getattr(local_state, "recognizer", None)
        if recognizer is None:
            recognizer = VisualWatermarkRecognizer()
            local_state.recognizer = recognizer
        run = map_keyframe_watermarks(
            recognizer,
            frames,
            output_root=options.output_root,
            timezone=timezone_name,
            source_width=int(row["width"] or 2880),
            source_height=int(row["height"] or 1620),
            crop=camera.watermark_crop,
            interval_seconds=options.watermark_interval_seconds,
            expected_start_ts=(
                float(row["capture_start_ts"])
                if row["capture_start_ts"] is not None
                else None
            ),
            expected_end_ts=(
                float(row["capture_end_hint_ts"])
                if row["capture_end_hint_ts"] is not None
                else None
            ),
        )
        return run

    workers = min(options.watermark_workers, len(claimed))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="cctv-visual-watermark",
    ) as executor:
        futures = {
            executor.submit(analyze, item): item
            for item in claimed
        }
        for future in as_completed(futures):
            row, _frames = futures[future]
            try:
                run = future.result()
                metadata = {
                    "frames_seen": run.frames_selected,
                    "valid_observations": run.recognized,
                    "ocr_failures": run.failures,
                    "segments": len(run.segments),
                    "sample_interval_seconds": run.sample_interval_seconds,
                    "minimum_confidence": 0.90,
                    "source": "local_visual_keyframes",
                    "model": run.model,
                }
                save_time_segments(
                    connection,
                    job_id=int(row["job_id"]),
                    media_file_id=int(row["id"]),
                    segments=run.segments,
                    source="watermark_ocr:visual_keyframes",
                    metadata=metadata,
                )
                stage["completed"] += 1
                stage["frames_selected"] += run.frames_selected
                stage["recognized"] += run.recognized
            except Exception as exc:
                mark_job_error(connection, int(row["job_id"]), str(exc))
                stage["failed"] += 1
            if (stage["completed"] + stage["failed"]) % 20 == 0:
                connection.commit()
    connection.commit()
    return stage


def _absolute_keyframe_paths(
    output_root: Path,
    frames: Sequence[Mapping[str, Any]],
) -> list[Path]:
    root = output_root.resolve()
    result: list[Path] = []
    for frame in frames:
        relative = Path(str(frame["relative_image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe keyframe path in database")
        path = root.joinpath(relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Keyframe path escaped output root") from exc
        result.append(path)
    return result


def _focus_boxes(
    connection,
    frames: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {
        int(frame["id"]): [] for frame in frames
    }
    ids = list(grouped)
    for start in range(0, len(ids), 800):
        batch = ids[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        for row in connection.execute(
            f"""
            SELECT
                keyframe_id, class_name, confidence,
                x1, y1, x2, y2
            FROM visual_detections
            WHERE keyframe_id IN ({placeholders})
              AND class_name = 'person'
              AND confidence >= 0.20
            ORDER BY keyframe_id, confidence DESC
            """,
            batch,
        ):
            grouped[int(row["keyframe_id"])].append(dict(row))
    return [grouped[int(frame["id"])] for frame in frames]


def _run_detection(
    connection,
    options: VisualPipelineOptions,
    runtime: VisualPipelineRuntime,
) -> dict[str, Any]:
    jobs, frames = _jobs_with_keyframes(
        connection,
        options,
        "object_detection",
    )
    stage: dict[str, Any] = {
        "jobs": len(jobs),
        "frames": len(frames),
        "detections": 0,
        "completed": 0,
        "failed": 0,
    }
    if not jobs:
        return stage
    try:
        detector = runtime.detector(options)
        paths = _absolute_keyframe_paths(options.output_root, frames)
        detections: dict[int, list[dict[str, Any]]] = {}
        for start in range(0, len(paths), options.detector_batch_size):
            stop = min(len(paths), start + options.detector_batch_size)
            batch_payload = detector.predict(
                paths[start:stop],
                batch_size=options.detector_batch_size,
            )
            for frame, values in zip(
                frames[start:stop],
                batch_payload,
                strict=True,
            ):
                detections[int(frame["id"])] = values
        stage["detections"] = save_detections(
            connection,
            rows_by_keyframe=detections,
            model_name=detector.model_name,
        )
        stage["completed"] = complete_visual_jobs(
            connection,
            job_ids=[int(job["job_id"]) for job in jobs],
            metadata={
                "model": detector.model_name,
                "frames": len(frames),
                "detections": stage["detections"],
            },
        )
        connection.commit()
    except Exception as exc:
        for job in jobs:
            mark_job_error(connection, int(job["job_id"]), str(exc))
        connection.commit()
        stage["failed"] = len(jobs)
        stage["error"] = str(exc)
    return stage


def _run_embeddings(
    connection,
    options: VisualPipelineOptions,
    runtime: VisualPipelineRuntime,
) -> dict[str, Any]:
    jobs, frames = _jobs_with_keyframes(
        connection,
        options,
        "visual_embedding",
    )
    stage: dict[str, Any] = {
        "jobs": len(jobs),
        "frames": len(frames),
        "embeddings": 0,
        "observations": 0,
        "completed": 0,
        "failed": 0,
    }
    if not jobs:
        return stage
    try:
        encoder = runtime.encoder(options)
        paths = _absolute_keyframe_paths(options.output_root, frames)
        boxes = _focus_boxes(connection, frames)
        embedding_rows: dict[int, dict[str, Any]] = {}
        observation_rows: dict[int, list[dict[str, Any]]] = {}
        for start in range(0, len(paths), options.embedding_batch_size):
            stop = min(len(paths), start + options.embedding_batch_size)
            batch_embeddings, batch_observations = encoder.encode(
                paths[start:stop],
                focus_boxes=boxes[start:stop],
            )
            for frame, embedding, observations in zip(
                frames[start:stop],
                batch_embeddings,
                batch_observations,
                strict=True,
            ):
                keyframe_id = int(frame["id"])
                embedding_rows[keyframe_id] = embedding
                observation_rows[keyframe_id] = observations
        (
            stage["embeddings"],
            stage["observations"],
        ) = save_embeddings_and_observations(
            connection,
            embeddings=embedding_rows,
            observations=observation_rows,
            model_name=encoder.model_name,
            prompt_version=encoder.prompt_version,
        )
        stage["completed"] = complete_visual_jobs(
            connection,
            job_ids=[int(job["job_id"]) for job in jobs],
            metadata={
                "model": encoder.model_name,
                "prompt_version": encoder.prompt_version,
                "frames": len(frames),
                "observations": stage["observations"],
            },
        )
        connection.commit()
    except Exception as exc:
        for job in jobs:
            mark_job_error(connection, int(job["job_id"]), str(exc))
        connection.commit()
        stage["failed"] = len(jobs)
        stage["error"] = str(exc)
    return stage


def run_visual_cycle(
    options: VisualPipelineOptions,
    *,
    runtime: VisualPipelineRuntime | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    options.validate()
    runtime = runtime or VisualPipelineRuntime()
    started = time.monotonic()
    with PipelineLock(options.lock_path, label="CCTV visual pipeline"):
        connection = connect_database(options.database_path)
        run_id: int | None = None
        result: dict[str, Any] = {}
        try:
            config = load_config(options.config_path)
            configured = _camera_lookup(config)
            unknown = [
                camera_id
                for camera_id in options.camera_ids
                if camera_id not in configured
            ]
            if unknown:
                raise ValueError(f"Unknown camera ids: {', '.join(unknown)}")
            _recover_visual_jobs(connection)
            run_id = start_visual_run(
                connection,
                since_ts=options.since_ts,
                until_ts=options.until_ts,
                camera_ids=options.camera_ids,
                extractor_version=options.extractor_version,
                detector_model=options.detector_model,
                embedding_model=options.embedding_model,
                prompt_version=PROMPT_VERSION,
                options=options.serializable(),
            )
            connection.commit()

            result["probe"] = _prepare_media(
                connection,
                options,
                camera_lookup=configured,
                audio_enabled=config.audio_enabled,
                audio_event_labels=config.audio_event_labels,
            )
            if progress_callback:
                progress_callback({"run_id": run_id, **result})
            update_visual_run(connection, run_id, progress=result)
            connection.commit()

            result["keyframes"] = _extract_keyframes(
                connection,
                options,
            )
            if progress_callback:
                progress_callback({"run_id": run_id, **result})
            update_visual_run(connection, run_id, progress=result)
            connection.commit()

            result["watermarks"] = _map_visual_watermarks(
                connection,
                options,
                camera_lookup=configured,
            )
            if progress_callback:
                progress_callback({"run_id": run_id, **result})
            update_visual_run(connection, run_id, progress=result)
            connection.commit()

            result["detections"] = _run_detection(
                connection,
                options,
                runtime,
            )
            if progress_callback:
                progress_callback({"run_id": run_id, **result})
            update_visual_run(connection, run_id, progress=result)
            connection.commit()

            result["embeddings"] = _run_embeddings(
                connection,
                options,
                runtime,
            )
            result["duration_seconds"] = round(
                time.monotonic() - started,
                3,
            )
            progress = query_visual_progress(
                connection,
                since_ts=options.since_ts,
                until_ts=options.until_ts,
                camera_ids=options.camera_ids,
            )
            result["scope_progress"] = progress
            pending = sum(
                count
                for key, count in progress["jobs"].items()
                if key.endswith(":queued") or key.endswith(":running")
            )
            failures = sum(
                count
                for key, count in progress["jobs"].items()
                if key.endswith(":error")
            )
            ready = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM media_files
                    WHERE camera_id IN (
                        {','.join('?' for _ in options.camera_ids)}
                    )
                      AND capture_start_ts >= ?
                      AND capture_start_ts < ?
                      AND status = 'ready'
                    """,
                    (
                        *options.camera_ids,
                        options.since_ts,
                        options.until_ts,
                    ),
                ).fetchone()[0]
            )
            if pending == 0 and ready == 0:
                result["activities"] = rebuild_activity_events(
                    connection,
                    since_ts=options.since_ts,
                    until_ts=options.until_ts,
                    camera_ids=options.camera_ids,
                    embedding_model=options.embedding_model,
                    prompt_version=PROMPT_VERSION,
                    generated_by_run_id=run_id,
                )
                connection.commit()
                progress = query_visual_progress(
                    connection,
                    since_ts=options.since_ts,
                    until_ts=options.until_ts,
                    camera_ids=options.camera_ids,
                )
                result["scope_progress"] = progress
                # Another local worker can discover a newly stable file while
                # activity aggregation is running. Re-evaluate completion
                # after that aggregation so this worker does not report a
                # stale 100% and exit with a freshly queued visual job.
                pending = sum(
                    count
                    for key, count in progress["jobs"].items()
                    if key.endswith(":queued")
                    or key.endswith(":running")
                )
                failures = sum(
                    count
                    for key, count in progress["jobs"].items()
                    if key.endswith(":error")
                )
                ready = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM media_files
                        WHERE camera_id IN (
                            {','.join('?' for _ in options.camera_ids)}
                        )
                          AND capture_start_ts >= ?
                          AND capture_start_ts < ?
                          AND status = 'ready'
                        """,
                        (
                            *options.camera_ids,
                            options.since_ts,
                            options.until_ts,
                        ),
                    ).fetchone()[0]
                )
            status = (
                "completed"
                if pending == 0 and failures == 0 and ready == 0
                else "partial"
            )
            finish_visual_run(
                connection,
                run_id,
                status=status,
                result=result,
            )
            connection.commit()
            if progress_callback:
                progress_callback(
                    {"run_id": run_id, "status": status, **result}
                )
            return {"visual_run_id": run_id, "status": status, **result}
        except BaseException as exc:
            if run_id is not None:
                finish_visual_run(
                    connection,
                    run_id,
                    status="failed",
                    result=result,
                    error=f"{type(exc).__name__}: {exc}",
                )
                connection.commit()
            raise
        finally:
            connection.close()


def run_visual_worker(
    options: VisualPipelineOptions,
    *,
    interval_seconds: int = 1,
    max_cycles: int | None = None,
    on_cycle: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be at least 1")
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be positive")
    runtime = VisualPipelineRuntime()
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        try:
            payload = run_visual_cycle(options, runtime=runtime)
        except Exception as exc:
            payload = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        cycles += 1
        payload["worker_cycle"] = cycles
        if on_cycle is not None:
            on_cycle(payload)
        if (
            payload.get("status") == "completed"
            or not _visual_work_remaining(payload)
        ):
            break
        if max_cycles is None or cycles < max_cycles:
            time.sleep(interval_seconds)
    return cycles


def _visual_work_remaining(payload: dict[str, Any]) -> bool:
    """Return whether another immediate worker cycle can make progress."""
    progress = payload.get("scope_progress")
    if not isinstance(progress, dict):
        # A failed cycle without progress details should retain the existing
        # bounded retry behavior.
        return True
    jobs = progress.get("jobs")
    if isinstance(jobs, dict) and any(
        int(count) > 0
        for key, count in jobs.items()
        if str(key).endswith(":queued")
        or str(key).endswith(":running")
    ):
        return True
    return int(progress.get("files_pending_schedule", 0) or 0) > 0
