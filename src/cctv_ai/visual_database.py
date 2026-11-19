from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .visual_extract import ExtractedKeyframe


VISUAL_JOB_TYPES = (
    "keyframe_extract",
    "object_detection",
    "visual_embedding",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def enqueue_visual_jobs_for_scope(
    connection: sqlite3.Connection,
    *,
    camera_ids: Sequence[str],
    since_ts: int,
    until_ts: int,
) -> int:
    if not camera_ids:
        return 0
    placeholders = ",".join("?" for _ in camera_ids)
    now_ts = int(time.time())
    before = connection.total_changes
    for job_type, priority in (
        ("keyframe_extract", 20),
        ("object_detection", 40),
        ("visual_embedding", 50),
    ):
        connection.execute(
            f"""
            INSERT INTO processing_jobs(
                media_file_id, job_type, status, priority,
                payload_json, created_at, updated_at
            )
            SELECT
                m.id, ?, 'queued', ?, '{{}}', ?, ?
            FROM media_files m
            WHERE m.camera_id IN ({placeholders})
              AND m.capture_start_ts >= ?
              AND m.capture_start_ts < ?
              AND m.status = 'probed'
            ON CONFLICT(media_file_id, job_type) DO UPDATE SET
                status = CASE
                    WHEN processing_jobs.status = 'stale' THEN 'queued'
                    ELSE processing_jobs.status
                END,
                priority = excluded.priority
            """,
            (
                job_type,
                priority,
                now_ts,
                now_ts,
                *camera_ids,
                since_ts,
                until_ts,
            ),
        )
    return connection.total_changes - before


def _map_offset(
    segments: Sequence[sqlite3.Row],
    media_seconds: float,
    *,
    maximum_extrapolation_seconds: float = 45.0,
) -> float | None:
    closest: tuple[float, sqlite3.Row] | None = None
    for segment in segments:
        media_start = float(segment["media_start_seconds"])
        media_end = float(
            segment["media_end_seconds"]
            if segment["media_end_seconds"] is not None
            else media_start
        )
        if media_start <= media_seconds <= media_end:
            distance = 0.0
        else:
            distance = min(
                abs(media_seconds - media_start),
                abs(media_seconds - media_end),
            )
        if closest is None or distance < closest[0]:
            closest = (distance, segment)
    if closest is None or closest[0] > maximum_extrapolation_seconds:
        return None
    segment = closest[1]
    media_start = float(segment["media_start_seconds"])
    media_end = float(
        segment["media_end_seconds"]
        if segment["media_end_seconds"] is not None
        else media_start
    )
    wall_start = float(segment["wall_start_ts"])
    wall_end = float(
        segment["wall_end_ts"]
        if segment["wall_end_ts"] is not None
        else wall_start
    )
    slope = (
        (wall_end - wall_start) / (media_end - media_start)
        if media_end > media_start
        else 1.0
    )
    return wall_start + ((media_seconds - media_start) * slope)


def keyframe_wall_time(
    connection: sqlite3.Connection,
    media: Mapping[str, Any],
    offset_seconds: float,
) -> tuple[float | None, str]:
    if not bool(media["requires_watermark_ocr"]):
        return float(media["capture_start_ts"]) + offset_seconds, "capture_clock"
    segments = list(
        connection.execute(
            """
            SELECT media_start_seconds, media_end_seconds,
                   wall_start_ts, wall_end_ts
            FROM time_segments
            WHERE media_file_id = ?
            ORDER BY media_start_seconds
            """,
            (int(media["id"]),),
        )
    )
    mapped = _map_offset(segments, offset_seconds)
    capture_start = media["capture_start_ts"]
    if mapped is not None:
        capture_end = media.get("capture_end_hint_ts")
        plausible_end = (
            float(capture_end)
            if capture_end is not None
            else (
                float(capture_start)
                + float(media.get("duration_seconds") or 0)
                if capture_start is not None
                else None
            )
        )
        if (
            capture_start is None
            or plausible_end is None
            or (
                float(capture_start) - 6 * 3600
                <= mapped
                <= plausible_end + 6 * 3600
            )
        ):
            return mapped, "watermark"
    if capture_start is not None:
        return float(capture_start) + offset_seconds, "estimated"
    return None, "unknown"


def save_extracted_keyframes(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    media: Mapping[str, Any],
    keyframes: Sequence[ExtractedKeyframe],
    output_root: str | Path,
    extractor_version: str,
    metadata: Mapping[str, Any],
) -> list[int]:
    media_file_id = int(media["id"])
    root = Path(output_root).resolve()
    now_ts = int(time.time())
    connection.execute(
        """
        DELETE FROM visual_keyframes
        WHERE media_file_id = ? AND extractor_version = ?
        """,
        (media_file_id, extractor_version),
    )
    inserted_ids: list[int] = []
    for keyframe in keyframes:
        absolute = keyframe.image_path.resolve()
        try:
            relative = absolute.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Keyframe is outside the visual output root") from exc
        wall_ts, wall_source = keyframe_wall_time(
            connection,
            media,
            keyframe.media_offset_seconds,
        )
        cursor = connection.execute(
            """
            INSERT INTO visual_keyframes(
                media_file_id, media_offset_ms, wall_ts, wall_time_source,
                relative_image_path, frame_sha256, width, height,
                brightness, sharpness, motion_score, night_score,
                sample_reason, extractor_version, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                media_file_id,
                round(keyframe.media_offset_seconds * 1000),
                wall_ts,
                wall_source,
                relative,
                keyframe.frame_sha256,
                keyframe.width,
                keyframe.height,
                keyframe.brightness,
                keyframe.sharpness,
                keyframe.motion_score,
                keyframe.night_score,
                keyframe.sample_reason,
                extractor_version,
                now_ts,
            ),
        )
        inserted_ids.append(int(cursor.lastrowid))
    payload = dict(metadata)
    payload["keyframe_ids"] = inserted_ids
    connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'completed', payload_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (_json(payload), now_ts, job_id),
    )
    return inserted_ids


def query_keyframes_for_media(
    connection: sqlite3.Connection,
    media_file_ids: Sequence[int],
    *,
    extractor_version: str | None = None,
) -> list[sqlite3.Row]:
    if not media_file_ids:
        return []
    placeholders = ",".join("?" for _ in media_file_ids)
    conditions = [f"k.media_file_id IN ({placeholders})"]
    parameters: list[Any] = list(media_file_ids)
    if extractor_version:
        conditions.append("k.extractor_version = ?")
        parameters.append(extractor_version)
    return list(
        connection.execute(
            f"""
            SELECT
                k.*,
                m.camera_id,
                m.relative_path AS media_relative_path,
                m.capture_start_ts,
                c.name AS camera_name,
                c.location
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            WHERE {' AND '.join(conditions)}
            ORDER BY k.media_file_id, k.media_offset_ms
            """,
            parameters,
        )
    )


def save_detections(
    connection: sqlite3.Connection,
    *,
    rows_by_keyframe: Mapping[int, Sequence[Mapping[str, Any]]],
    model_name: str,
) -> int:
    if not rows_by_keyframe:
        return 0
    keyframe_ids = list(rows_by_keyframe)
    placeholders = ",".join("?" for _ in keyframe_ids)
    connection.execute(
        f"""
        DELETE FROM visual_detections
        WHERE keyframe_id IN ({placeholders}) AND model_name = ?
        """,
        (*keyframe_ids, model_name),
    )
    now_ts = int(time.time())
    values: list[tuple[Any, ...]] = []
    for keyframe_id, detections in rows_by_keyframe.items():
        for detection in detections:
            values.append(
                (
                    keyframe_id,
                    int(detection["class_id"]),
                    str(detection["class_name"]),
                    float(detection["confidence"]),
                    float(detection["x1"]),
                    float(detection["y1"]),
                    float(detection["x2"]),
                    float(detection["y2"]),
                    model_name,
                    _json(detection.get("attributes", {})),
                    now_ts,
                )
            )
    connection.executemany(
        """
        INSERT INTO visual_detections(
            keyframe_id, class_id, class_name, confidence,
            x1, y1, x2, y2, model_name, attributes_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def save_embeddings_and_observations(
    connection: sqlite3.Connection,
    *,
    embeddings: Mapping[int, Mapping[str, Any]],
    observations: Mapping[int, Sequence[Mapping[str, Any]]],
    model_name: str,
    prompt_version: str,
) -> tuple[int, int]:
    now_ts = int(time.time())
    embedding_count = 0
    observation_count = 0
    for keyframe_id, value in embeddings.items():
        connection.execute(
            """
            INSERT INTO visual_embeddings(
                keyframe_id, model_name, dimensions, dtype,
                vector, vector_norm, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(keyframe_id, model_name) DO UPDATE SET
                dimensions = excluded.dimensions,
                dtype = excluded.dtype,
                vector = excluded.vector,
                vector_norm = excluded.vector_norm,
                created_at = excluded.created_at
            """,
            (
                keyframe_id,
                model_name,
                int(value["dimensions"]),
                str(value["dtype"]),
                value["vector"],
                float(value["vector_norm"]),
                now_ts,
            ),
        )
        embedding_count += 1
    if observations:
        keyframe_ids = list(observations)
        placeholders = ",".join("?" for _ in keyframe_ids)
        connection.execute(
            f"""
            DELETE FROM visual_observations
            WHERE keyframe_id IN ({placeholders})
              AND model_name = ?
              AND prompt_version = ?
            """,
            (*keyframe_ids, model_name, prompt_version),
        )
    values: list[tuple[Any, ...]] = []
    for keyframe_id, frame_observations in observations.items():
        for observation in frame_observations:
            values.append(
                (
                    keyframe_id,
                    str(observation["group"]),
                    str(observation["label"]),
                    float(observation["score"]),
                    model_name,
                    prompt_version,
                    _json(observation.get("details", {})),
                    now_ts,
                )
            )
    connection.executemany(
        """
        INSERT INTO visual_observations(
            keyframe_id, observation_group, label, score,
            model_name, prompt_version, details_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    observation_count = len(values)
    return embedding_count, observation_count


def complete_visual_jobs(
    connection: sqlite3.Connection,
    *,
    job_ids: Iterable[int],
    metadata: Mapping[str, Any],
) -> int:
    selected = list(dict.fromkeys(int(value) for value in job_ids))
    if not selected:
        return 0
    placeholders = ",".join("?" for _ in selected)
    cursor = connection.execute(
        f"""
        UPDATE processing_jobs
        SET status = 'completed', payload_json = ?, updated_at = ?
        WHERE id IN ({placeholders})
        """,
        (_json(dict(metadata)), int(time.time()), *selected),
    )
    return int(cursor.rowcount)


def start_visual_run(
    connection: sqlite3.Connection,
    *,
    since_ts: int,
    until_ts: int,
    camera_ids: Sequence[str],
    extractor_version: str,
    detector_model: str,
    embedding_model: str,
    prompt_version: str,
    options: Mapping[str, Any],
) -> int:
    started_at = int(time.time())
    connection.execute(
        """
        UPDATE visual_analysis_runs
        SET finished_at = ?,
            status = 'cancelled',
            error = COALESCE(
                error,
                'Recovered an interrupted visual-analysis run'
            )
        WHERE status = 'running'
        """,
        (started_at,),
    )
    cursor = connection.execute(
        """
        INSERT INTO visual_analysis_runs(
            started_at, since_ts, until_ts, camera_ids_json,
            extractor_version, detector_model, embedding_model,
            prompt_version, options_json, hostname, process_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            started_at,
            since_ts,
            until_ts,
            _json(list(camera_ids)),
            extractor_version,
            detector_model,
            embedding_model,
            prompt_version,
            _json(dict(options)),
            socket.gethostname(),
            os.getpid(),
        ),
    )
    return int(cursor.lastrowid)


def update_visual_run(
    connection: sqlite3.Connection,
    run_id: int,
    *,
    progress: Mapping[str, Any],
) -> None:
    connection.execute(
        """
        UPDATE visual_analysis_runs
        SET progress_json = ?
        WHERE id = ?
        """,
        (_json(dict(progress)), run_id),
    )


def finish_visual_run(
    connection: sqlite3.Connection,
    run_id: int,
    *,
    status: str,
    result: Mapping[str, Any],
    error: str | None = None,
) -> None:
    if status not in {"completed", "partial", "failed", "cancelled"}:
        raise ValueError(f"Unsupported visual run status: {status}")
    connection.execute(
        """
        UPDATE visual_analysis_runs
        SET finished_at = ?, status = ?, result_json = ?, error = ?
        WHERE id = ?
        """,
        (
            int(time.time()),
            status,
            _json(dict(result)),
            error[:4000] if error else None,
            run_id,
        ),
    )


def query_visual_progress(
    connection: sqlite3.Connection,
    *,
    since_ts: int | None = None,
    until_ts: int | None = None,
    camera_ids: Sequence[str] = (),
) -> dict[str, Any]:
    latest = connection.execute(
        """
        SELECT *
        FROM visual_analysis_runs
        ORDER BY started_at DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is not None and since_ts is None:
        since_ts = int(latest["since_ts"])
        until_ts = int(latest["until_ts"])
        camera_ids = tuple(json.loads(latest["camera_ids_json"]))
    conditions = ["1 = 1"]
    parameters: list[Any] = []
    if since_ts is not None:
        conditions.append("m.capture_start_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("m.capture_start_ts < ?")
        parameters.append(until_ts)
    if camera_ids:
        placeholders = ",".join("?" for _ in camera_ids)
        conditions.append(f"m.camera_id IN ({placeholders})")
        parameters.extend(camera_ids)
    where = " AND ".join(conditions)
    jobs = {
        f"{row['job_type']}:{row['status']}": int(row["count"])
        for row in connection.execute(
            f"""
            SELECT j.job_type, j.status, COUNT(*) AS count
            FROM processing_jobs j
            JOIN media_files m ON m.id = j.media_file_id
            WHERE {where}
              AND j.job_type IN (
                  'keyframe_extract', 'object_detection', 'visual_embedding',
                  'watermark_time_map'
              )
            GROUP BY j.job_type, j.status
            """,
            parameters,
        )
    }
    keyframes = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            WHERE {where}
            """,
            parameters,
        ).fetchone()[0]
    )
    observation_conditions = [where]
    observation_parameters = list(parameters)
    if latest is not None:
        observation_conditions.extend(
            ["o.model_name = ?", "o.prompt_version = ?"]
        )
        observation_parameters.extend(
            [
                str(latest["embedding_model"]),
                str(latest["prompt_version"]),
            ]
        )
    observations = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM visual_observations o
            JOIN visual_keyframes k ON k.id = o.keyframe_id
            JOIN media_files m ON m.id = k.media_file_id
            WHERE {' AND '.join(observation_conditions)}
            """,
            observation_parameters,
        ).fetchone()[0]
    )
    activities = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM activity_events
            WHERE (? IS NULL OR start_ts >= ?)
              AND (? IS NULL OR start_ts < ?)
            """,
            (since_ts, since_ts, until_ts, until_ts),
        ).fetchone()[0]
    )
    latest_payload = None
    if latest is not None:
        latest_payload = {
            "id": int(latest["id"]),
            "status": str(latest["status"]),
            "started_at": int(latest["started_at"]),
            "finished_at": latest["finished_at"],
            "since_ts": int(latest["since_ts"]),
            "until_ts": int(latest["until_ts"]),
            "camera_ids": json.loads(latest["camera_ids_json"]),
            "extractor_version": str(latest["extractor_version"]),
            "detector_model": str(latest["detector_model"]),
            "embedding_model": str(latest["embedding_model"]),
            "prompt_version": str(latest["prompt_version"]),
            "progress": json.loads(latest["progress_json"] or "{}"),
            "result": json.loads(latest["result_json"] or "{}"),
            "error": latest["error"],
        }
    scheduled = max(
        jobs.get("keyframe_extract:completed", 0)
        + jobs.get("keyframe_extract:queued", 0)
        + jobs.get("keyframe_extract:running", 0)
        + jobs.get("keyframe_extract:error", 0),
        0,
    )
    source_total = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM media_files m
            WHERE {where}
            """,
            parameters,
        ).fetchone()[0]
    )
    source_statuses = {
        str(row["status"]): int(row["count"])
        for row in connection.execute(
            f"""
            SELECT m.status, COUNT(*) AS count
            FROM media_files m
            WHERE {where}
            GROUP BY m.status
            """,
            parameters,
        )
    }
    stable_total = (
        source_statuses.get("ready", 0)
        + source_statuses.get("probed", 0)
    )
    unstable_total = source_statuses.get("waiting", 0)
    source_error_total = source_statuses.get("error", 0)
    completed = jobs.get("visual_embedding:completed", 0)
    return {
        "latest_run": latest_payload,
        "jobs": jobs,
        "keyframes": keyframes,
        "observations": observations,
        "activity_events": activities,
        "files_total": source_total,
        "files_stable": stable_total,
        "files_unstable": unstable_total,
        "files_source_error": source_error_total,
        "files_scheduled": scheduled,
        "files_pending_schedule": max(stable_total - scheduled, 0),
        "files_fully_analyzed": completed,
        "progress_percent": (
            100.0
            if stable_total and completed >= stable_total
            else min(
                99.9,
                round(completed * 100.0 / stable_total, 2),
            )
            if stable_total
            else 0.0
        ),
    }
