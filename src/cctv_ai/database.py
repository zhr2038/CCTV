from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .models import CameraConfig, MediaProbe, ParsedCapture


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT NOT NULL,
    kind TEXT NOT NULL,
    relative_root TEXT NOT NULL UNIQUE,
    audio_enabled INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    source_root TEXT NOT NULL,
    since_ts INTEGER,
    files_seen INTEGER NOT NULL DEFAULT 0,
    parse_errors INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS scan_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_run_id INTEGER NOT NULL REFERENCES scan_runs(id),
    camera_id TEXT NOT NULL REFERENCES cameras(id),
    relative_path TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS media_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL REFERENCES cameras(id),
    relative_path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    first_seen_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    stable_observations INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    capture_start_ts INTEGER NOT NULL,
    capture_end_hint_ts INTEGER,
    capture_end_ts INTEGER,
    time_source TEXT NOT NULL,
    requires_watermark_ocr INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL,
    format_name TEXT,
    bit_rate INTEGER,
    video_codec TEXT,
    width INTEGER,
    height INTEGER,
    fps REAL,
    audio_codec TEXT,
    audio_sample_rate INTEGER,
    audio_channels INTEGER,
    probe_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_scan_run_id INTEGER REFERENCES scan_runs(id)
);

CREATE INDEX IF NOT EXISTS media_files_camera_time
    ON media_files(camera_id, capture_start_ts);
CREATE INDEX IF NOT EXISTS media_files_status
    ON media_files(status);

CREATE TABLE IF NOT EXISTS processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_file_id INTEGER NOT NULL REFERENCES media_files(id),
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 100,
    payload_json TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE(media_file_id, job_type)
);

CREATE INDEX IF NOT EXISTS processing_jobs_status
    ON processing_jobs(status, priority, id);

CREATE TABLE IF NOT EXISTS time_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_file_id INTEGER NOT NULL REFERENCES media_files(id),
    media_start_seconds REAL NOT NULL,
    media_end_seconds REAL,
    wall_start_ts INTEGER NOT NULL,
    wall_end_ts INTEGER,
    source TEXT NOT NULL,
    confidence REAL,
    UNIQUE(media_file_id, media_start_seconds)
);

CREATE TABLE IF NOT EXISTS audio_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_file_id INTEGER NOT NULL REFERENCES media_files(id),
    event_label TEXT NOT NULL,
    model_label TEXT NOT NULL,
    model_class_index INTEGER NOT NULL,
    media_start_seconds REAL NOT NULL,
    media_end_seconds REAL NOT NULL,
    wall_start_ts REAL,
    wall_end_ts REAL,
    confidence REAL NOT NULL,
    windows INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(media_file_id, event_label, media_start_seconds, model_name)
);

CREATE INDEX IF NOT EXISTS audio_events_wall_time
    ON audio_events(event_label, wall_start_ts);

CREATE TABLE IF NOT EXISTS audio_event_fingerprints (
    audio_event_id INTEGER PRIMARY KEY
        REFERENCES audio_events(id) ON DELETE CASCADE,
    model_sha256 TEXT NOT NULL,
    vector_length INTEGER NOT NULL,
    mean_scores BLOB NOT NULL,
    max_scores BLOB NOT NULL,
    windows_seen INTEGER NOT NULL,
    media_start_seconds REAL NOT NULL,
    media_end_seconds REAL NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_reviews (
    audio_event_id INTEGER PRIMARY KEY
        REFERENCES audio_events(id) ON DELETE CASCADE,
    verdict TEXT NOT NULL CHECK(
        verdict IN ('confirmed', 'false_positive', 'uncertain')
    ),
    note TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL DEFAULT 'family',
    reviewed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_review_ledger (
    relative_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ms INTEGER NOT NULL,
    event_label TEXT NOT NULL,
    media_start_ms INTEGER NOT NULL,
    media_end_ms INTEGER NOT NULL,
    verdict TEXT NOT NULL CHECK(
        verdict IN ('confirmed', 'false_positive', 'uncertain')
    ),
    note TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL DEFAULT 'family',
    reviewed_at INTEGER NOT NULL,
    PRIMARY KEY(
        relative_path, size_bytes, mtime_ms, event_label,
        media_start_ms, media_end_ms
    )
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    status TEXT NOT NULL DEFAULT 'running' CHECK(
        status IN ('running', 'completed', 'partial', 'failed', 'skipped')
    ),
    mode TEXT NOT NULL,
    source_root TEXT NOT NULL,
    since_ts INTEGER,
    until_ts INTEGER,
    camera_ids_json TEXT NOT NULL,
    options_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    hostname TEXT NOT NULL,
    process_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS pipeline_runs_started
    ON pipeline_runs(started_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS pipeline_worker_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    status TEXT NOT NULL CHECK(
        status IN ('running', 'stopped', 'error')
    ),
    started_at INTEGER,
    heartbeat_at INTEGER NOT NULL,
    hostname TEXT NOT NULL,
    process_id INTEGER NOT NULL,
    interval_seconds INTEGER NOT NULL,
    scope_json TEXT NOT NULL DEFAULT '{}',
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS shadow_calibration_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_label TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    finished_at INTEGER,
    status TEXT NOT NULL CHECK(
        status IN ('running', 'completed', 'insufficient', 'failed')
    ),
    model_name TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    minimum_confidence REAL NOT NULL,
    gap_seconds REAL NOT NULL,
    positive_episodes INTEGER NOT NULL DEFAULT 0,
    negative_episodes INTEGER NOT NULL DEFAULT 0,
    fingerprinted_episodes INTEGER NOT NULL DEFAULT 0,
    scored_episodes INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error TEXT
);

CREATE INDEX IF NOT EXISTS shadow_calibration_runs_latest
    ON shadow_calibration_runs(event_label, status, id DESC);

CREATE TABLE IF NOT EXISTS shadow_episode_scores (
    run_id INTEGER NOT NULL
        REFERENCES shadow_calibration_runs(id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL,
    event_label TEXT NOT NULL,
    start_ts REAL NOT NULL,
    review_status TEXT NOT NULL,
    score REAL NOT NULL,
    margin REAL NOT NULL,
    positive_similarity REAL NOT NULL,
    negative_similarity REAL NOT NULL,
    review_priority REAL NOT NULL,
    bucket TEXT NOT NULL,
    reason TEXT NOT NULL,
    event_ids_json TEXT NOT NULL,
    feature_json TEXT NOT NULL,
    PRIMARY KEY(run_id, episode_id)
);

CREATE INDEX IF NOT EXISTS shadow_episode_scores_episode
    ON shadow_episode_scores(event_label, episode_id, run_id DESC);

CREATE TABLE IF NOT EXISTS visual_analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    status TEXT NOT NULL DEFAULT 'running' CHECK(
        status IN ('running', 'completed', 'partial', 'failed', 'cancelled')
    ),
    since_ts INTEGER NOT NULL,
    until_ts INTEGER NOT NULL,
    camera_ids_json TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    detector_model TEXT,
    embedding_model TEXT,
    prompt_version TEXT,
    options_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    hostname TEXT NOT NULL,
    process_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS visual_analysis_runs_started
    ON visual_analysis_runs(started_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS visual_keyframes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_file_id INTEGER NOT NULL
        REFERENCES media_files(id) ON DELETE CASCADE,
    media_offset_ms INTEGER NOT NULL,
    wall_ts REAL,
    wall_time_source TEXT NOT NULL CHECK(
        wall_time_source IN ('capture_clock', 'watermark', 'estimated', 'unknown')
    ),
    relative_image_path TEXT NOT NULL UNIQUE,
    frame_sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    brightness REAL,
    sharpness REAL,
    motion_score REAL,
    night_score REAL,
    sample_reason TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(media_file_id, media_offset_ms, extractor_version)
);

CREATE INDEX IF NOT EXISTS visual_keyframes_wall_time
    ON visual_keyframes(wall_ts, media_file_id);
CREATE INDEX IF NOT EXISTS visual_keyframes_media_offset
    ON visual_keyframes(media_file_id, media_offset_ms);

CREATE TABLE IF NOT EXISTS visual_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyframe_id INTEGER NOT NULL
        REFERENCES visual_keyframes(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    model_name TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS visual_detections_frame
    ON visual_detections(keyframe_id, class_name, confidence DESC);
CREATE INDEX IF NOT EXISTS visual_detections_class
    ON visual_detections(class_name, confidence DESC);

CREATE TABLE IF NOT EXISTS visual_embeddings (
    keyframe_id INTEGER NOT NULL
        REFERENCES visual_keyframes(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    vector BLOB NOT NULL,
    vector_norm REAL NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(keyframe_id, model_name)
);

CREATE TABLE IF NOT EXISTS visual_observations (
    keyframe_id INTEGER NOT NULL
        REFERENCES visual_keyframes(id) ON DELETE CASCADE,
    observation_group TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    PRIMARY KEY(keyframe_id, label, model_name, prompt_version)
);

CREATE INDEX IF NOT EXISTS visual_observations_label
    ON visual_observations(label, score DESC, keyframe_id);

CREATE TABLE IF NOT EXISTS visual_frame_verifications (
    keyframe_id INTEGER NOT NULL
        REFERENCES visual_keyframes(id) ON DELETE CASCADE,
    task TEXT NOT NULL CHECK(
        task IN ('feeding', 'toy', 'sleep_presence')
    ),
    label TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(
        confidence >= 0.0 AND confidence <= 1.0
    ),
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    PRIMARY KEY(keyframe_id, task, model_name, prompt_version)
);

CREATE INDEX IF NOT EXISTS visual_frame_verifications_task
    ON visual_frame_verifications(
        task, prompt_version, label, confidence DESC, keyframe_id
    );

CREATE TABLE IF NOT EXISTS activity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL UNIQUE,
    activity_type TEXT NOT NULL,
    activity_subtype TEXT,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    camera_ids_json TEXT NOT NULL,
    location TEXT NOT NULL,
    zone TEXT,
    confidence REAL NOT NULL,
    time_precision TEXT NOT NULL CHECK(
        time_precision IN ('exact', 'mapped', 'estimated')
    ),
    method TEXT NOT NULL,
    model_version TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'machine' CHECK(
        review_status IN (
            'machine', 'confirmed', 'corrected', 'rejected', 'uncertain'
        )
    ),
    details_json TEXT NOT NULL DEFAULT '{}',
    generated_by_run_id INTEGER
        REFERENCES visual_analysis_runs(id) ON DELETE SET NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS activity_events_time
    ON activity_events(start_ts, end_ts, activity_type);
CREATE INDEX IF NOT EXISTS activity_events_type
    ON activity_events(activity_type, confidence DESC, start_ts);

CREATE TABLE IF NOT EXISTS activity_event_evidence (
    activity_event_id INTEGER NOT NULL
        REFERENCES activity_events(id) ON DELETE CASCADE,
    keyframe_id INTEGER NOT NULL
        REFERENCES visual_keyframes(id) ON DELETE CASCADE,
    evidence_rank INTEGER NOT NULL,
    score REAL NOT NULL,
    role TEXT NOT NULL DEFAULT 'representative',
    PRIMARY KEY(activity_event_id, keyframe_id)
);

CREATE INDEX IF NOT EXISTS activity_event_evidence_rank
    ON activity_event_evidence(activity_event_id, evidence_rank);

CREATE TABLE IF NOT EXISTS activity_event_reviews (
    activity_event_id INTEGER PRIMARY KEY
        REFERENCES activity_events(id) ON DELETE CASCADE,
    verdict TEXT NOT NULL CHECK(
        verdict IN ('confirmed', 'corrected', 'rejected', 'uncertain')
    ),
    corrected_activity_type TEXT,
    corrected_activity_subtype TEXT,
    corrected_start_ts REAL,
    corrected_end_ts REAL,
    note TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL DEFAULT 'family',
    reviewed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS toy_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_key TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    toy_type TEXT NOT NULL,
    representative_keyframe_id INTEGER
        REFERENCES visual_keyframes(id) ON DELETE SET NULL,
    first_seen_ts REAL,
    last_seen_ts REAL,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    interaction_seconds REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    model_version TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS toy_event_links (
    toy_entity_id INTEGER NOT NULL
        REFERENCES toy_entities(id) ON DELETE CASCADE,
    activity_event_id INTEGER NOT NULL
        REFERENCES activity_events(id) ON DELETE CASCADE,
    confidence REAL NOT NULL,
    PRIMARY KEY(toy_entity_id, activity_event_id)
);
"""


def connect_database(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA)
    _backfill_review_ledger(connection)
    connection.commit()
    return connection


def backup_database(
    path: str | Path,
    *,
    backup_directory: str | Path | None = None,
    label: str = "manual",
    reuse_daily: bool = False,
    now_ts: int | None = None,
) -> Path | None:
    """Create an integrity-checked SQLite online backup.

    A daily automatic backup can be reused so frequent scan cycles do not
    produce duplicate copies. Manual backups always receive a unique
    timestamped filename.
    """
    source = Path(path).resolve()
    if not source.is_file():
        return None
    safe_label = "".join(
        character
        for character in label.strip().lower().replace(" ", "-")
        if character.isalnum() or character in {"-", "_"}
    ).strip("-_")
    if not safe_label:
        raise ValueError("Backup label must contain a letter or number")

    timestamp = int(now_ts if now_ts is not None else time.time())
    stamp_format = "%Y%m%d" if reuse_daily else "%Y%m%d-%H%M%S"
    stamp = datetime.fromtimestamp(timestamp).strftime(stamp_format)
    directory = (
        Path(backup_directory).resolve()
        if backup_directory is not None
        else source.parent.joinpath("backups")
    )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory.joinpath(
        f"{source.stem}.{safe_label}-{stamp}{source.suffix}"
    )
    if reuse_daily and target.is_file():
        existing = sqlite3.connect(target)
        try:
            integrity = existing.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
        finally:
            existing.close()
        if integrity != "ok":
            raise RuntimeError(
                f"Existing daily backup failed integrity check: {target}"
            )
        return target

    counter = 2
    while target.exists():
        target = directory.joinpath(
            f"{source.stem}.{safe_label}-{stamp}-{counter}{source.suffix}"
        )
        counter += 1
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        source_uri = f"{source.as_uri()}?mode=ro"
        source_connection = sqlite3.connect(source_uri, uri=True)
        target_connection = sqlite3.connect(temporary)
        try:
            source_connection.backup(target_connection)
            integrity = target_connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(
                    f"New backup failed integrity check: {integrity}"
                )
        finally:
            target_connection.close()
            source_connection.close()
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def _backfill_review_ledger(connection: sqlite3.Connection) -> None:
    pending = connection.execute(
        """
        SELECT 1
        FROM event_reviews r
        JOIN audio_events a ON a.id = r.audio_event_id
        JOIN media_files m ON m.id = a.media_file_id
        LEFT JOIN event_review_ledger l
          ON l.relative_path = m.relative_path
         AND l.size_bytes = m.size_bytes
         AND l.mtime_ms = CAST(m.mtime_ns / 1000000 AS INTEGER)
         AND l.event_label = a.event_label
         AND l.media_start_ms = CAST(
             ROUND(a.media_start_seconds * 1000) AS INTEGER
         )
         AND l.media_end_ms = CAST(
             ROUND(a.media_end_seconds * 1000) AS INTEGER
         )
        WHERE l.relative_path IS NULL
           OR l.verdict IS NOT r.verdict
           OR l.note IS NOT r.note
           OR l.reviewer IS NOT r.reviewer
           OR l.reviewed_at IS NOT r.reviewed_at
        LIMIT 1
        """
    ).fetchone()
    if pending is None:
        return

    connection.execute(
        """
        INSERT INTO event_review_ledger(
            relative_path, size_bytes, mtime_ms, event_label,
            media_start_ms, media_end_ms, verdict, note,
            reviewer, reviewed_at
        )
        SELECT
            m.relative_path,
            m.size_bytes,
            CAST(m.mtime_ns / 1000000 AS INTEGER),
            a.event_label,
            CAST(ROUND(a.media_start_seconds * 1000) AS INTEGER),
            CAST(ROUND(a.media_end_seconds * 1000) AS INTEGER),
            r.verdict,
            r.note,
            r.reviewer,
            r.reviewed_at
        FROM event_reviews r
        JOIN audio_events a ON a.id = r.audio_event_id
        JOIN media_files m ON m.id = a.media_file_id
        ON CONFLICT(
            relative_path, size_bytes, mtime_ms, event_label,
            media_start_ms, media_end_ms
        ) DO UPDATE SET
            verdict = excluded.verdict,
            note = excluded.note,
            reviewer = excluded.reviewer,
            reviewed_at = excluded.reviewed_at
        """
    )


def _restore_event_reviews(
    connection: sqlite3.Connection,
    media_file_id: int,
) -> int:
    before = connection.total_changes
    connection.execute(
        """
        INSERT INTO event_reviews(
            audio_event_id, verdict, note, reviewer, reviewed_at
        )
        SELECT
            a.id,
            ledger.verdict,
            ledger.note,
            ledger.reviewer,
            ledger.reviewed_at
        FROM audio_events a
        JOIN media_files m ON m.id = a.media_file_id
        JOIN event_review_ledger ledger
          ON ledger.relative_path = m.relative_path
         AND ledger.size_bytes = m.size_bytes
         AND ledger.mtime_ms = CAST(m.mtime_ns / 1000000 AS INTEGER)
         AND ledger.event_label = a.event_label
         AND ledger.media_start_ms =
             CAST(ROUND(a.media_start_seconds * 1000) AS INTEGER)
         AND ledger.media_end_ms =
             CAST(ROUND(a.media_end_seconds * 1000) AS INTEGER)
        WHERE a.media_file_id = ?
        ON CONFLICT(audio_event_id) DO UPDATE SET
            verdict = excluded.verdict,
            note = excluded.note,
            reviewer = excluded.reviewer,
            reviewed_at = excluded.reviewed_at
        """,
        (media_file_id,),
    )
    return connection.total_changes - before


def sync_cameras(
    connection: sqlite3.Connection, cameras: Iterable[CameraConfig]
) -> None:
    connection.executemany(
        """
        INSERT INTO cameras(id, name, location, kind, relative_root, audio_enabled)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            location = excluded.location,
            kind = excluded.kind,
            relative_root = excluded.relative_root,
            audio_enabled = excluded.audio_enabled
        """,
        [
            (
                camera.id,
                camera.name,
                camera.location,
                camera.kind,
                camera.relative_root,
                int(camera.audio_enabled),
            )
            for camera in cameras
        ],
    )


def start_scan(
    connection: sqlite3.Connection,
    source_root: str,
    since_ts: int | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO scan_runs(started_at, source_root, since_ts)
        VALUES (?, ?, ?)
        """,
        (int(time.time()), source_root, since_ts),
    )
    connection.commit()
    return int(cursor.lastrowid)


def finish_scan(
    connection: sqlite3.Connection,
    scan_run_id: int,
    files_seen: int,
    parse_errors: int,
    status: str = "completed",
) -> None:
    connection.execute(
        """
        UPDATE scan_runs
        SET finished_at = ?, files_seen = ?, parse_errors = ?, status = ?
        WHERE id = ?
        """,
        (int(time.time()), files_seen, parse_errors, status, scan_run_id),
    )
    connection.commit()


def start_pipeline_run(
    connection: sqlite3.Connection,
    *,
    mode: str,
    source_root: str,
    camera_ids: Iterable[str],
    since_ts: int | None,
    until_ts: int | None,
    options: dict[str, Any],
    hostname: str,
    process_id: int,
    now_ts: int | None = None,
) -> int:
    timestamp = int(now_ts if now_ts is not None else time.time())
    cursor = connection.execute(
        """
        INSERT INTO pipeline_runs(
            started_at, status, mode, source_root,
            since_ts, until_ts, camera_ids_json,
            options_json, hostname, process_id
        )
        VALUES (?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            mode,
            source_root,
            since_ts,
            until_ts,
            json.dumps(
                list(dict.fromkeys(str(value) for value in camera_ids)),
                ensure_ascii=False,
            ),
            json.dumps(options, ensure_ascii=False, sort_keys=True),
            hostname,
            process_id,
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def finish_pipeline_run(
    connection: sqlite3.Connection,
    pipeline_run_id: int,
    *,
    status: str,
    result: dict[str, Any],
    error: str | None = None,
    now_ts: int | None = None,
) -> None:
    if status not in {"completed", "partial", "failed", "skipped"}:
        raise ValueError(f"Unsupported pipeline status: {status}")
    timestamp = int(now_ts if now_ts is not None else time.time())
    connection.execute(
        """
        UPDATE pipeline_runs
        SET finished_at = ?, status = ?, result_json = ?, error = ?
        WHERE id = ?
        """,
        (
            timestamp,
            status,
            json.dumps(result, ensure_ascii=False, sort_keys=True),
            error[:4000] if error else None,
            pipeline_run_id,
        ),
    )
    connection.commit()


def recover_interrupted_pipeline_runs(
    connection: sqlite3.Connection,
    *,
    now_ts: int | None = None,
) -> int:
    """Close runs left open after their exclusive pipeline process exited."""
    timestamp = int(now_ts if now_ts is not None else time.time())
    cursor = connection.execute(
        """
        UPDATE pipeline_runs
        SET finished_at = ?,
            status = 'failed',
            error = COALESCE(
                NULLIF(error, ''),
                'Recovered interrupted pipeline run'
            )
        WHERE status = 'running'
        """,
        (timestamp,),
    )
    return int(cursor.rowcount)


def get_latest_pipeline_run(
    connection: sqlite3.Connection,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT *
        FROM pipeline_runs
        ORDER BY started_at DESC, id DESC
        LIMIT 1
        """
    ).fetchone()


def update_pipeline_worker_state(
    connection: sqlite3.Connection,
    *,
    status: str,
    hostname: str,
    process_id: int,
    interval_seconds: int,
    scope: dict[str, Any],
    started_at: int | None = None,
    last_error: str | None = None,
    now_ts: int | None = None,
) -> None:
    if status not in {"running", "stopped", "error"}:
        raise ValueError(f"Unsupported worker status: {status}")
    timestamp = int(now_ts if now_ts is not None else time.time())
    connection.execute(
        """
        INSERT INTO pipeline_worker_state(
            id, status, started_at, heartbeat_at, hostname,
            process_id, interval_seconds, scope_json, last_error
        )
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status,
            started_at = COALESCE(
                excluded.started_at,
                pipeline_worker_state.started_at
            ),
            heartbeat_at = excluded.heartbeat_at,
            hostname = excluded.hostname,
            process_id = excluded.process_id,
            interval_seconds = excluded.interval_seconds,
            scope_json = excluded.scope_json,
            last_error = excluded.last_error
        """,
        (
            status,
            started_at,
            timestamp,
            hostname,
            process_id,
            interval_seconds,
            json.dumps(scope, ensure_ascii=False, sort_keys=True),
            last_error[:4000] if last_error else None,
        ),
    )
    connection.commit()


def get_pipeline_worker_state(
    connection: sqlite3.Connection,
) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM pipeline_worker_state WHERE id = 1"
    ).fetchone()


def record_scan_error(
    connection: sqlite3.Connection,
    *,
    scan_run_id: int,
    camera_id: str,
    relative_path: str,
    error: str,
) -> None:
    connection.execute(
        """
        INSERT INTO scan_errors(
            scan_run_id, camera_id, relative_path, error, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            scan_run_id,
            camera_id,
            relative_path,
            error[:2000],
            int(time.time()),
        ),
    )


def observe_file(
    connection: sqlite3.Connection,
    *,
    scan_run_id: int,
    camera_id: str,
    relative_path: str,
    size_bytes: int,
    mtime_ns: int,
    parsed: ParsedCapture,
    now_ts: int,
    stability_age_seconds: int,
    required_observations: int,
) -> str:
    row = connection.execute(
        """
        SELECT id, size_bytes, mtime_ns, stable_observations, status
        FROM media_files
        WHERE relative_path = ?
        """,
        (relative_path,),
    ).fetchone()
    old_enough = (now_ts - (mtime_ns // 1_000_000_000)) >= stability_age_seconds

    if row is None:
        observations = 1
        status = (
            "ready"
            if old_enough and observations >= required_observations
            else "waiting"
        )
        connection.execute(
            """
            INSERT INTO media_files(
                camera_id, relative_path, size_bytes, mtime_ns,
                first_seen_at, last_seen_at, stable_observations, status,
                capture_start_ts, capture_end_hint_ts, time_source,
                requires_watermark_ocr, last_scan_run_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                relative_path,
                size_bytes,
                mtime_ns,
                now_ts,
                now_ts,
                observations,
                status,
                int(parsed.capture_start.timestamp()),
                (
                    int(parsed.capture_end_hint.timestamp())
                    if parsed.capture_end_hint
                    else None
                ),
                parsed.time_source,
                int(parsed.requires_watermark_ocr),
                scan_run_id,
            ),
        )
        return status

    mtime_difference = abs(int(row["mtime_ns"]) - int(mtime_ns))
    changed = (
        row["size_bytes"] != size_bytes
        or mtime_difference > 1_000_000
    )
    if changed:
        observations = 1
        status = "waiting"
        connection.execute(
            """
            UPDATE media_files
            SET camera_id = ?, size_bytes = ?, mtime_ns = ?,
                last_seen_at = ?, stable_observations = ?, status = ?,
                capture_start_ts = ?, capture_end_hint_ts = ?,
                capture_end_ts = NULL, time_source = ?,
                requires_watermark_ocr = ?, duration_seconds = NULL,
                format_name = NULL, bit_rate = NULL, video_codec = NULL,
                width = NULL, height = NULL, fps = NULL,
                audio_codec = NULL, audio_sample_rate = NULL,
                audio_channels = NULL, last_error = NULL,
                last_scan_run_id = ?
            WHERE id = ?
            """,
            (
                camera_id,
                size_bytes,
                mtime_ns,
                now_ts,
                observations,
                status,
                int(parsed.capture_start.timestamp()),
                (
                    int(parsed.capture_end_hint.timestamp())
                    if parsed.capture_end_hint
                    else None
                ),
                parsed.time_source,
                int(parsed.requires_watermark_ocr),
                scan_run_id,
                row["id"],
            ),
        )
        connection.execute(
            """
            UPDATE processing_jobs
            SET status = 'stale', updated_at = ?
            WHERE media_file_id = ? AND status != 'stale'
            """,
            (now_ts, row["id"]),
        )
        connection.execute(
            "DELETE FROM time_segments WHERE media_file_id = ?",
            (row["id"],),
        )
        connection.execute(
            "DELETE FROM audio_events WHERE media_file_id = ?",
            (row["id"],),
        )
        return status

    observations = int(row["stable_observations"]) + 1
    if row["status"] in {"probed", "error"}:
        status = str(row["status"])
    else:
        status = (
            "ready"
            if old_enough and observations >= required_observations
            else "waiting"
        )
    connection.execute(
        """
        UPDATE media_files
        SET last_seen_at = ?, stable_observations = ?, status = ?,
            last_scan_run_id = ?
        WHERE id = ?
        """,
        (now_ts, observations, status, scan_run_id, row["id"]),
    )
    return status


def get_ready_files(
    connection: sqlite3.Connection,
    limit: int,
    camera_id: str | None = None,
    *,
    camera_ids: Iterable[str] | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    newest_first: bool = True,
) -> list[sqlite3.Row]:
    if limit <= 0:
        return []
    selected_cameras = tuple(
        dict.fromkeys(str(value) for value in (camera_ids or ()))
    )
    if camera_id and selected_cameras:
        raise ValueError("Use camera_id or camera_ids, not both")
    conditions = [
        """
        (
            status = 'ready'
            OR (status = 'error' AND probe_attempts < 3)
        )
        """
    ]
    parameters: list[Any] = []
    if camera_id:
        conditions.append("camera_id = ?")
        parameters.append(camera_id)
    elif selected_cameras:
        placeholders = ",".join("?" for _ in selected_cameras)
        conditions.append(f"camera_id IN ({placeholders})")
        parameters.extend(selected_cameras)
    if since_ts is not None:
        conditions.append("capture_start_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("capture_start_ts < ?")
        parameters.append(until_ts)
    parameters.append(limit)
    direction = "DESC" if newest_first else "ASC"
    return list(
        connection.execute(
            f"""
            SELECT *
            FROM media_files
            WHERE {' AND '.join(conditions)}
            ORDER BY capture_start_ts {direction}, id {direction}
            LIMIT ?
            """,
            parameters,
        )
    )


def _enqueue_job(
    connection: sqlite3.Connection,
    media_file_id: int,
    job_type: str,
    priority: int,
    payload: dict[str, Any] | None = None,
) -> None:
    now_ts = int(time.time())
    connection.execute(
        """
        INSERT INTO processing_jobs(
            media_file_id, job_type, status, priority,
            payload_json, created_at, updated_at
        )
        VALUES (?, ?, 'queued', ?, ?, ?, ?)
        ON CONFLICT(media_file_id, job_type) DO UPDATE SET
            status = CASE
                WHEN processing_jobs.status = 'stale' THEN 'queued'
                ELSE processing_jobs.status
            END,
            priority = excluded.priority,
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
        """,
        (
            media_file_id,
            job_type,
            priority,
            json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
            now_ts,
            now_ts,
        ),
    )


def save_probe(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    probe: MediaProbe,
    *,
    audio_enabled: bool,
    audio_event_labels: tuple[str, ...],
) -> None:
    capture_end_ts = None
    if not row["requires_watermark_ocr"] and probe.duration_seconds is not None:
        capture_end_ts = int(row["capture_start_ts"] + probe.duration_seconds)
    connection.execute(
        """
        UPDATE media_files
        SET status = 'probed', duration_seconds = ?, format_name = ?,
            bit_rate = ?, video_codec = ?, width = ?, height = ?, fps = ?,
            audio_codec = ?, audio_sample_rate = ?, audio_channels = ?,
            capture_end_ts = ?, probe_attempts = probe_attempts + 1,
            last_error = NULL
        WHERE id = ?
        """,
        (
            probe.duration_seconds,
            probe.format_name,
            probe.bit_rate,
            probe.video_codec,
            probe.width,
            probe.height,
            probe.fps,
            probe.audio_codec,
            probe.audio_sample_rate,
            probe.audio_channels,
            capture_end_ts,
            row["id"],
        ),
    )

    _enqueue_job(connection, row["id"], "keyframe_extract", 20)
    _enqueue_job(connection, row["id"], "object_detection", 40)
    _enqueue_job(connection, row["id"], "visual_embedding", 50)
    if row["requires_watermark_ocr"]:
        _enqueue_job(connection, row["id"], "watermark_time_map", 10)
    if audio_enabled and probe.audio_codec:
        _enqueue_job(
            connection,
            row["id"],
            "audio_event_detection",
            30,
            {"labels": list(audio_event_labels)},
        )


def enqueue_missing_audio_jobs_for_scope(
    connection: sqlite3.Connection,
    *,
    camera_ids: Iterable[str],
    since_ts: float,
    until_ts: float,
    audio_event_labels: tuple[str, ...],
) -> int:
    """Repair audio jobs missed when another worker probed media first."""
    selected_cameras = tuple(dict.fromkeys(str(value) for value in camera_ids))
    if not selected_cameras or until_ts <= since_ts:
        return 0
    placeholders = ",".join("?" for _ in selected_cameras)
    now_ts = int(time.time())
    payload = json.dumps(
        {"labels": list(audio_event_labels)},
        ensure_ascii=False,
        sort_keys=True,
    )
    cursor = connection.execute(
        f"""
        INSERT OR IGNORE INTO processing_jobs(
            media_file_id, job_type, status, priority, payload_json,
            attempts, created_at, updated_at
        )
        SELECT
            m.id, 'audio_event_detection', 'queued', 30, ?,
            0, ?, ?
        FROM media_files m
        JOIN cameras c ON c.id = m.camera_id
        WHERE m.camera_id IN ({placeholders})
          AND m.capture_start_ts >= ?
          AND m.capture_start_ts < ?
          AND m.status = 'probed'
          AND m.audio_codec IS NOT NULL
          AND c.audio_enabled = 1
        """,
        (
            payload,
            now_ts,
            now_ts,
            *selected_cameras,
            since_ts,
            until_ts,
        ),
    )
    return int(cursor.rowcount)


def save_probe_error(
    connection: sqlite3.Connection, media_file_id: int, error: str
) -> None:
    connection.execute(
        """
        UPDATE media_files
        SET status = 'error', probe_attempts = probe_attempts + 1,
            last_error = ?
        WHERE id = ?
        """,
        (error[:2000], media_file_id),
    )


def get_queued_jobs(
    connection: sqlite3.Connection,
    job_type: str,
    limit: int,
    camera_id: str | None = None,
    require_time_map_for_watermark: bool = False,
    include_completed: bool = False,
    media_file_id: int | None = None,
    audio_event_label: str | None = None,
    *,
    camera_ids: Iterable[str] | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    newest_first: bool = False,
    error_retry_delay_seconds: int = 0,
    now_ts: int | None = None,
) -> list[sqlite3.Row]:
    if limit <= 0:
        return []
    if error_retry_delay_seconds < 0:
        raise ValueError("error_retry_delay_seconds cannot be negative")
    selected_cameras = tuple(
        dict.fromkeys(str(value) for value in (camera_ids or ()))
    )
    if camera_id and selected_cameras:
        raise ValueError("Use camera_id or camera_ids, not both")
    statuses = (
        "('queued', 'error', 'completed')"
        if include_completed
        else "('queued', 'error')"
    )
    conditions = [
        "j.job_type = ?",
        f"j.status IN {statuses}",
        "j.attempts < 3",
        "m.status = 'probed'",
    ]
    parameters: list[Any] = [job_type]
    if camera_id:
        conditions.append("m.camera_id = ?")
        parameters.append(camera_id)
    elif selected_cameras:
        placeholders = ",".join("?" for _ in selected_cameras)
        conditions.append(f"m.camera_id IN ({placeholders})")
        parameters.extend(selected_cameras)
    if media_file_id is not None:
        conditions.append("m.id = ?")
        parameters.append(media_file_id)
    if since_ts is not None:
        conditions.append("m.capture_start_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("m.capture_start_ts < ?")
        parameters.append(until_ts)
    if audio_event_label:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM audio_events a
                WHERE a.media_file_id = m.id
                  AND a.event_label = ?
            )
            """
        )
        parameters.append(audio_event_label)
    if require_time_map_for_watermark:
        conditions.append(
            """
            (
                m.requires_watermark_ocr = 0
                OR EXISTS (
                    SELECT 1
                    FROM time_segments t
                    WHERE t.media_file_id = m.id
                )
            )
            """
        )
    if error_retry_delay_seconds:
        timestamp = int(now_ts if now_ts is not None else time.time())
        conditions.append(
            "(j.status != 'error' OR j.updated_at <= ?)"
        )
        parameters.append(timestamp - error_retry_delay_seconds)
    parameters.append(limit)
    ordering = (
        "m.capture_start_ts DESC, j.priority, j.id"
        if newest_first
        else "j.priority, j.id"
    )
    return list(
        connection.execute(
            f"""
            SELECT
                j.id AS job_id,
                j.job_type,
                j.status AS job_status,
                j.attempts AS job_attempts,
                m.*
            FROM processing_jobs j
            JOIN media_files m ON m.id = j.media_file_id
            WHERE {' AND '.join(conditions)}
            ORDER BY {ordering}
            LIMIT ?
            """,
            parameters,
        )
    )


def mark_job_running(
    connection: sqlite3.Connection,
    job_id: int,
    *,
    include_completed: bool = False,
) -> bool:
    cursor = connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'running', attempts = attempts + 1, updated_at = ?
        WHERE id = ?
          AND (
              status IN ('queued', 'error')
              OR (? = 1 AND status = 'completed')
          )
          AND attempts < 3
        """,
        (int(time.time()), job_id, int(include_completed)),
    )
    return cursor.rowcount == 1


def recover_stale_running_jobs(
    connection: sqlite3.Connection,
    *,
    stale_before_ts: int,
    now_ts: int | None = None,
) -> int:
    timestamp = int(now_ts if now_ts is not None else time.time())
    cursor = connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'queued',
            payload_json = ?,
            updated_at = ?
        WHERE status = 'running'
          AND updated_at < ?
        """,
        (
            json.dumps(
                {
                    "recovery": "Recovered stale running job",
                    "retry": "immediate",
                },
                ensure_ascii=False,
            ),
            timestamp,
            stale_before_ts,
        ),
    )
    return int(cursor.rowcount)


def mark_job_error(
    connection: sqlite3.Connection, job_id: int, error: str
) -> None:
    connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'error', payload_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            json.dumps({"error": error[:2000]}, ensure_ascii=False),
            int(time.time()),
            job_id,
        ),
    )


def has_time_segments(
    connection: sqlite3.Connection,
    media_file_id: int,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM time_segments
        WHERE media_file_id = ?
        LIMIT 1
        """,
        (media_file_id,),
    ).fetchone()
    return row is not None


def save_time_segments(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    media_file_id: int,
    segments: Iterable[Any],
    source: str,
    metadata: dict[str, Any],
) -> None:
    connection.execute(
        "DELETE FROM time_segments WHERE media_file_id = ?",
        (media_file_id,),
    )
    connection.executemany(
        """
        INSERT INTO time_segments(
            media_file_id, media_start_seconds, media_end_seconds,
            wall_start_ts, wall_end_ts, source, confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                media_file_id,
                segment.media_start_seconds,
                segment.media_end_seconds,
                int(segment.wall_start.timestamp()),
                (
                    int(segment.wall_end.timestamp())
                    if segment.wall_end is not None
                    else None
                ),
                source,
                segment.confidence,
            )
            for segment in segments
        ],
    )
    connection.execute(
        """
        UPDATE media_files
        SET capture_start_ts = (
                SELECT MIN(wall_start_ts)
                FROM time_segments
                WHERE media_file_id = ?
            ),
            capture_end_ts = (
                SELECT MAX(COALESCE(wall_end_ts, wall_start_ts))
                FROM time_segments
                WHERE media_file_id = ?
            ),
            time_source = ?
        WHERE id = ?
        """,
        (media_file_id, media_file_id, source, media_file_id),
    )
    backfill_audio_event_times(connection, media_file_id)
    backfill_visual_keyframe_times(connection, media_file_id)
    connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'completed', payload_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            int(time.time()),
            job_id,
        ),
    )


def _map_watermark_offset(
    segments: list[sqlite3.Row],
    media_seconds: float,
    *,
    maximum_extrapolation_seconds: float = 10.0,
) -> tuple[int, float] | None:
    best: tuple[float, sqlite3.Row] | None = None
    for segment in segments:
        start = float(segment["media_start_seconds"])
        end = (
            float(segment["media_end_seconds"])
            if segment["media_end_seconds"] is not None
            else start
        )
        if start <= media_seconds <= end:
            distance = 0.0
        else:
            distance = min(
                abs(media_seconds - start),
                abs(media_seconds - end),
            )
        if best is None or distance < best[0]:
            best = (distance, segment)
    if best is None or best[0] > maximum_extrapolation_seconds:
        return None

    segment = best[1]
    media_start = float(segment["media_start_seconds"])
    media_end = (
        float(segment["media_end_seconds"])
        if segment["media_end_seconds"] is not None
        else media_start
    )
    wall_start = float(segment["wall_start_ts"])
    wall_end = (
        float(segment["wall_end_ts"])
        if segment["wall_end_ts"] is not None
        else wall_start
    )
    if media_end > media_start:
        slope = (wall_end - wall_start) / (media_end - media_start)
    else:
        slope = 1.0
    wall_time = wall_start + ((media_seconds - media_start) * slope)
    return int(segment["id"]), wall_time


def _mapped_event_wall_times(
    segments: list[sqlite3.Row],
    media_start_seconds: float,
    media_end_seconds: float,
) -> tuple[float | None, float | None]:
    mapped_start = _map_watermark_offset(
        segments,
        media_start_seconds,
    )
    mapped_end = _map_watermark_offset(
        segments,
        media_end_seconds,
    )
    if (
        mapped_start is not None
        and mapped_end is not None
        and mapped_start[0] == mapped_end[0]
    ):
        return mapped_start[1], mapped_end[1]
    return None, None


def backfill_audio_event_times(
    connection: sqlite3.Connection,
    media_file_id: int,
) -> None:
    segments = list(
        connection.execute(
            """
            SELECT *
            FROM time_segments
            WHERE media_file_id = ?
            ORDER BY media_start_seconds
            """,
            (media_file_id,),
        )
    )
    if not segments:
        return
    events = list(
        connection.execute(
            """
            SELECT id, media_start_seconds, media_end_seconds
            FROM audio_events
            WHERE media_file_id = ?
            """,
            (media_file_id,),
        )
    )
    for event in events:
        wall_start, wall_end = _mapped_event_wall_times(
            segments,
            float(event["media_start_seconds"]),
            float(event["media_end_seconds"]),
        )
        connection.execute(
            """
            UPDATE audio_events
            SET wall_start_ts = ?, wall_end_ts = ?
            WHERE id = ?
            """,
            (wall_start, wall_end, event["id"]),
        )


def backfill_visual_keyframe_times(
    connection: sqlite3.Connection,
    media_file_id: int,
) -> None:
    segments = list(
        connection.execute(
            """
            SELECT *
            FROM time_segments
            WHERE media_file_id = ?
            ORDER BY media_start_seconds
            """,
            (media_file_id,),
        )
    )
    if not segments:
        return
    keyframes = list(
        connection.execute(
            """
            SELECT id, media_offset_ms
            FROM visual_keyframes
            WHERE media_file_id = ?
            """,
            (media_file_id,),
        )
    )
    for keyframe in keyframes:
        mapped = _map_watermark_offset(
            segments,
            float(keyframe["media_offset_ms"]) / 1000.0,
        )
        if mapped is None:
            continue
        connection.execute(
            """
            UPDATE visual_keyframes
            SET wall_ts = ?, wall_time_source = 'watermark'
            WHERE id = ?
            """,
            (mapped[1], keyframe["id"]),
        )


def save_audio_events(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    media_file_id: int,
    events: Iterable[Any],
    model_name: str,
    metadata: dict[str, Any],
) -> None:
    media = connection.execute(
        """
        SELECT capture_start_ts, requires_watermark_ocr
        FROM media_files
        WHERE id = ?
        """,
        (media_file_id,),
    ).fetchone()
    if media is None:
        raise ValueError(f"Unknown media file id: {media_file_id}")

    segments = list(
        connection.execute(
            """
            SELECT *
            FROM time_segments
            WHERE media_file_id = ?
            ORDER BY media_start_seconds
            """,
            (media_file_id,),
        )
    )
    requires_watermark = bool(media["requires_watermark_ocr"])

    event_rows: list[tuple[Any, ...]] = []
    now_ts = int(time.time())
    for event in events:
        wall_start: float | None
        wall_end: float | None
        if requires_watermark:
            wall_start, wall_end = _mapped_event_wall_times(
                segments,
                float(event.media_start_seconds),
                float(event.media_end_seconds),
            )
        else:
            wall_start = (
                float(media["capture_start_ts"])
                + float(event.media_start_seconds)
            )
            wall_end = (
                float(media["capture_start_ts"])
                + float(event.media_end_seconds)
            )
        event_rows.append(
            (
                media_file_id,
                event.label,
                event.model_label,
                event.model_class_index,
                event.media_start_seconds,
                event.media_end_seconds,
                wall_start,
                wall_end,
                event.confidence,
                event.windows,
                model_name,
                now_ts,
            )
        )

    connection.execute(
        "DELETE FROM audio_events WHERE media_file_id = ?",
        (media_file_id,),
    )
    connection.executemany(
        """
        INSERT INTO audio_events(
            media_file_id, event_label, model_label, model_class_index,
            media_start_seconds, media_end_seconds,
            wall_start_ts, wall_end_ts, confidence, windows,
            model_name, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        event_rows,
    )
    _restore_event_reviews(connection, media_file_id)
    payload = dict(metadata)
    payload["events"] = len(event_rows)
    connection.execute(
        """
        UPDATE processing_jobs
        SET status = 'completed', payload_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            now_ts,
            job_id,
        ),
    )


def query_audio_events(
    connection: sqlite3.Connection,
    *,
    camera_id: str | None = None,
    event_label: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    minimum_confidence: float = 0.0,
    limit: int = 100,
) -> list[sqlite3.Row]:
    conditions = ["a.confidence >= ?"]
    parameters: list[Any] = [minimum_confidence]
    if camera_id:
        conditions.append("m.camera_id = ?")
        parameters.append(camera_id)
    if event_label:
        conditions.append("a.event_label = ?")
        parameters.append(event_label)
    if since_ts is not None:
        conditions.append("a.wall_start_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("a.wall_start_ts < ?")
        parameters.append(until_ts)
    parameters.append(limit)
    return list(
        connection.execute(
            f"""
            SELECT
                a.id AS event_id,
                a.event_label,
                a.model_label,
                a.media_start_seconds,
                a.media_end_seconds,
                a.wall_start_ts,
                a.wall_end_ts,
                a.confidence,
                a.windows,
                a.model_name,
                m.id AS media_file_id,
                m.camera_id,
                m.relative_path,
                m.capture_start_ts,
                m.requires_watermark_ocr,
                c.name AS camera_name,
                c.location
            FROM audio_events a
            JOIN media_files m ON m.id = a.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            WHERE {' AND '.join(conditions)}
            ORDER BY
                COALESCE(
                    a.wall_start_ts,
                    m.capture_start_ts + a.media_start_seconds
                ) DESC,
                a.id DESC
            LIMIT ?
            """,
            parameters,
        )
    )


def query_episode_events(
    connection: sqlite3.Connection,
    *,
    event_label: str | None = None,
    camera_id: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    minimum_confidence: float = 0.0,
) -> list[sqlite3.Row]:
    conditions = [
        "a.wall_start_ts IS NOT NULL",
        "a.confidence >= ?",
    ]
    parameters: list[Any] = [minimum_confidence]
    if event_label:
        conditions.append("a.event_label = ?")
        parameters.append(event_label)
    if camera_id:
        conditions.append("m.camera_id = ?")
        parameters.append(camera_id)
    if since_ts is not None:
        conditions.append("a.wall_start_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("a.wall_start_ts < ?")
        parameters.append(until_ts)
    return list(
        connection.execute(
            f"""
            SELECT
                a.id AS event_id,
                a.event_label,
                a.model_label,
                a.media_start_seconds,
                a.media_end_seconds,
                a.wall_start_ts,
                a.wall_end_ts,
                a.confidence,
                a.windows,
                m.id AS media_file_id,
                m.camera_id,
                m.relative_path,
                m.duration_seconds,
                m.size_bytes,
                m.mtime_ns,
                c.name AS camera_name,
                c.location,
                r.verdict AS review_verdict,
                r.note AS review_note,
                r.reviewer,
                r.reviewed_at
            FROM audio_events a
            JOIN media_files m ON m.id = a.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            LEFT JOIN event_reviews r ON r.audio_event_id = a.id
            WHERE {' AND '.join(conditions)}
            ORDER BY a.wall_start_ts, a.id
            """,
            parameters,
        )
    )


def query_audio_event_fingerprints(
    connection: sqlite3.Connection,
    *,
    event_ids: Iterable[int],
    model_sha256: str,
) -> dict[int, sqlite3.Row]:
    ids = tuple(dict.fromkeys(int(event_id) for event_id in event_ids))
    if not ids:
        return {}
    result: dict[int, sqlite3.Row] = {}
    for offset in range(0, len(ids), 900):
        chunk = ids[offset : offset + 900]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT *
            FROM audio_event_fingerprints
            WHERE model_sha256 = ?
              AND audio_event_id IN ({placeholders})
            """,
            (model_sha256, *chunk),
        )
        result.update(
            (int(row["audio_event_id"]), row)
            for row in rows
        )
    return result


def save_audio_event_fingerprint(
    connection: sqlite3.Connection,
    *,
    audio_event_id: int,
    model_sha256: str,
    vector_length: int,
    mean_scores: bytes,
    max_scores: bytes,
    windows_seen: int,
    media_start_seconds: float,
    media_end_seconds: float,
    created_at: int | None = None,
) -> None:
    if vector_length < 1:
        raise ValueError("vector_length must be positive")
    if windows_seen < 1:
        raise ValueError("windows_seen must be positive")
    expected_bytes = vector_length * 4
    if len(mean_scores) != expected_bytes or len(max_scores) != expected_bytes:
        raise ValueError("Fingerprint byte length does not match vector_length")
    connection.execute(
        """
        INSERT INTO audio_event_fingerprints(
            audio_event_id, model_sha256, vector_length,
            mean_scores, max_scores, windows_seen,
            media_start_seconds, media_end_seconds, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(audio_event_id) DO UPDATE SET
            model_sha256 = excluded.model_sha256,
            vector_length = excluded.vector_length,
            mean_scores = excluded.mean_scores,
            max_scores = excluded.max_scores,
            windows_seen = excluded.windows_seen,
            media_start_seconds = excluded.media_start_seconds,
            media_end_seconds = excluded.media_end_seconds,
            created_at = excluded.created_at
        """,
        (
            int(audio_event_id),
            model_sha256,
            int(vector_length),
            mean_scores,
            max_scores,
            int(windows_seen),
            float(media_start_seconds),
            float(media_end_seconds),
            int(created_at if created_at is not None else time.time()),
        ),
    )


def query_latest_shadow_scores(
    connection: sqlite3.Connection,
    *,
    event_label: str,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    run = connection.execute(
        """
        SELECT *
        FROM shadow_calibration_runs
        WHERE event_label = ? AND status = 'completed'
        ORDER BY id DESC
        LIMIT 1
        """,
        (event_label,),
    ).fetchone()
    if run is None:
        return None, {}
    scores = {
        str(row["episode_id"]): dict(row)
        for row in connection.execute(
            """
            SELECT *
            FROM shadow_episode_scores
            WHERE run_id = ?
            """,
            (int(run["id"]),),
        )
    }
    payload = dict(run)
    payload["metrics"] = json.loads(payload.pop("metrics_json") or "{}")
    return payload, scores


def save_event_reviews(
    connection: sqlite3.Connection,
    *,
    event_ids: Iterable[int],
    verdict: str,
    note: str = "",
    reviewer: str = "family",
) -> int:
    if verdict not in {"confirmed", "false_positive", "uncertain"}:
        raise ValueError(f"Unsupported review verdict: {verdict}")
    ids = tuple(dict.fromkeys(int(event_id) for event_id in event_ids))
    if not ids:
        raise ValueError("At least one event id is required")
    placeholders = ",".join("?" for _ in ids)
    existing = {
        int(row["id"])
        for row in connection.execute(
            f"SELECT id FROM audio_events WHERE id IN ({placeholders})",
            ids,
        )
    }
    if existing != set(ids):
        missing = sorted(set(ids) - existing)
        raise ValueError(f"Unknown audio event ids: {missing}")
    reviewed_at = int(time.time())
    connection.executemany(
        """
        INSERT INTO event_reviews(
            audio_event_id, verdict, note, reviewer, reviewed_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(audio_event_id) DO UPDATE SET
            verdict = excluded.verdict,
            note = excluded.note,
            reviewer = excluded.reviewer,
            reviewed_at = excluded.reviewed_at
        """,
        [
            (
                event_id,
                verdict,
                note[:2000],
                reviewer[:100],
                reviewed_at,
            )
            for event_id in ids
        ],
    )
    connection.execute(
        f"""
        INSERT INTO event_review_ledger(
            relative_path, size_bytes, mtime_ms, event_label,
            media_start_ms, media_end_ms, verdict, note,
            reviewer, reviewed_at
        )
        SELECT
            m.relative_path,
            m.size_bytes,
            CAST(m.mtime_ns / 1000000 AS INTEGER),
            a.event_label,
            CAST(ROUND(a.media_start_seconds * 1000) AS INTEGER),
            CAST(ROUND(a.media_end_seconds * 1000) AS INTEGER),
            r.verdict,
            r.note,
            r.reviewer,
            r.reviewed_at
        FROM event_reviews r
        JOIN audio_events a ON a.id = r.audio_event_id
        JOIN media_files m ON m.id = a.media_file_id
        WHERE r.audio_event_id IN ({placeholders})
        ON CONFLICT(
            relative_path, size_bytes, mtime_ms, event_label,
            media_start_ms, media_end_ms
        ) DO UPDATE SET
            verdict = excluded.verdict,
            note = excluded.note,
            reviewer = excluded.reviewer,
            reviewed_at = excluded.reviewed_at
        """,
        ids,
    )
    connection.commit()
    return len(ids)


def get_event_for_preview(
    connection: sqlite3.Connection,
    event_id: int,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT
            a.id AS event_id,
            a.event_label,
            a.media_start_seconds,
            a.media_end_seconds,
            a.wall_start_ts,
            a.wall_end_ts,
            a.confidence,
            m.id AS media_file_id,
            m.camera_id,
            m.relative_path,
            m.duration_seconds,
            m.size_bytes,
            m.mtime_ns,
            c.name AS camera_name
        FROM audio_events a
        JOIN media_files m ON m.id = a.media_file_id
        JOIN cameras c ON c.id = m.camera_id
        WHERE a.id = ?
        """,
        (event_id,),
    ).fetchone()


def query_camera_health(
    connection: sqlite3.Connection,
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT
                c.id,
                c.name,
                c.location,
                COUNT(m.id) AS files,
                MAX(
                    COALESCE(
                        m.capture_end_ts,
                        m.capture_end_hint_ts,
                        m.capture_start_ts
                    )
                ) AS last_capture_ts,
                SUM(CASE WHEN m.status = 'waiting' THEN 1 ELSE 0 END)
                    AS waiting,
                SUM(CASE WHEN m.status = 'error' THEN 1 ELSE 0 END)
                    AS errors
            FROM cameras c
            LEFT JOIN media_files m ON m.camera_id = c.id
            GROUP BY c.id, c.name, c.location
            ORDER BY c.id
            """
        )
    )


def import_analysis_database(
    connection: sqlite3.Connection,
    source_path: str | Path,
) -> dict[str, int]:
    source_file = Path(source_path).resolve()
    source = sqlite3.connect(
        f"{source_file.as_uri()}?mode=ro",
        uri=True,
    )
    source.row_factory = sqlite3.Row
    media_id_map: dict[int, int] = {}
    counts = {
        "media": 0,
        "jobs": 0,
        "time_segments": 0,
        "audio_events": 0,
    }
    try:
        analyzed_media = list(
            source.execute(
                """
                SELECT *
                FROM media_files
                WHERE status = 'probed'
                """
            )
        )
        for media in analyzed_media:
            target = connection.execute(
                """
                SELECT id
                FROM media_files
                WHERE relative_path = ?
                """,
                (media["relative_path"],),
            ).fetchone()
            if target is None:
                continue
            target_id = int(target["id"])
            source_id = int(media["id"])
            media_id_map[source_id] = target_id
            connection.execute(
                """
                UPDATE media_files
                SET status = 'probed',
                    capture_start_ts = ?,
                    capture_end_hint_ts = ?,
                    capture_end_ts = ?,
                    time_source = ?,
                    requires_watermark_ocr = ?,
                    duration_seconds = ?,
                    format_name = ?,
                    bit_rate = ?,
                    video_codec = ?,
                    width = ?,
                    height = ?,
                    fps = ?,
                    audio_codec = ?,
                    audio_sample_rate = ?,
                    audio_channels = ?,
                    probe_attempts = MAX(probe_attempts, ?),
                    last_error = ?
                WHERE id = ?
                """,
                (
                    media["capture_start_ts"],
                    media["capture_end_hint_ts"],
                    media["capture_end_ts"],
                    media["time_source"],
                    media["requires_watermark_ocr"],
                    media["duration_seconds"],
                    media["format_name"],
                    media["bit_rate"],
                    media["video_codec"],
                    media["width"],
                    media["height"],
                    media["fps"],
                    media["audio_codec"],
                    media["audio_sample_rate"],
                    media["audio_channels"],
                    media["probe_attempts"],
                    media["last_error"],
                    target_id,
                ),
            )
            counts["media"] += 1

        for job in source.execute(
            """
            SELECT *
            FROM processing_jobs
            ORDER BY id
            """
        ):
            target_id = media_id_map.get(int(job["media_file_id"]))
            if target_id is None:
                continue
            connection.execute(
                """
                INSERT INTO processing_jobs(
                    media_file_id, job_type, status, priority,
                    payload_json, attempts, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(media_file_id, job_type) DO UPDATE SET
                    status = excluded.status,
                    priority = excluded.priority,
                    payload_json = excluded.payload_json,
                    attempts = excluded.attempts,
                    updated_at = excluded.updated_at
                """,
                (
                    target_id,
                    job["job_type"],
                    job["status"],
                    job["priority"],
                    job["payload_json"],
                    job["attempts"],
                    job["created_at"],
                    job["updated_at"],
                ),
            )
            counts["jobs"] += 1

        for source_id, target_id in media_id_map.items():
            connection.execute(
                "DELETE FROM time_segments WHERE media_file_id = ?",
                (target_id,),
            )
            for segment in source.execute(
                """
                SELECT *
                FROM time_segments
                WHERE media_file_id = ?
                ORDER BY media_start_seconds
                """,
                (source_id,),
            ):
                connection.execute(
                    """
                    INSERT INTO time_segments(
                        media_file_id, media_start_seconds,
                        media_end_seconds, wall_start_ts, wall_end_ts,
                        source, confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_id,
                        segment["media_start_seconds"],
                        segment["media_end_seconds"],
                        segment["wall_start_ts"],
                        segment["wall_end_ts"],
                        segment["source"],
                        segment["confidence"],
                    ),
                )
                counts["time_segments"] += 1

            connection.execute(
                "DELETE FROM audio_events WHERE media_file_id = ?",
                (target_id,),
            )
            for event in source.execute(
                """
                SELECT *
                FROM audio_events
                WHERE media_file_id = ?
                ORDER BY media_start_seconds, id
                """,
                (source_id,),
            ):
                connection.execute(
                    """
                    INSERT INTO audio_events(
                        media_file_id, event_label, model_label,
                        model_class_index, media_start_seconds,
                        media_end_seconds, wall_start_ts, wall_end_ts,
                        confidence, windows, model_name, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_id,
                        event["event_label"],
                        event["model_label"],
                        event["model_class_index"],
                        event["media_start_seconds"],
                        event["media_end_seconds"],
                        event["wall_start_ts"],
                        event["wall_end_ts"],
                        event["confidence"],
                        event["windows"],
                        event["model_name"],
                        event["created_at"],
                    ),
                )
                counts["audio_events"] += 1
            _restore_event_reviews(connection, target_id)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        source.close()
    return counts


def build_report(connection: sqlite3.Connection) -> dict[str, Any]:
    cameras = [
        dict(row)
        for row in connection.execute(
            """
            SELECT
                c.id,
                c.name,
                c.location,
                c.kind,
                COUNT(m.id) AS files,
                COALESCE(SUM(m.size_bytes), 0) AS bytes,
                MIN(m.capture_start_ts) AS first_capture_ts,
                MAX(COALESCE(m.capture_end_hint_ts, m.capture_start_ts))
                    AS last_capture_ts,
                SUM(CASE WHEN m.status = 'waiting' THEN 1 ELSE 0 END) AS waiting,
                SUM(CASE WHEN m.status = 'ready' THEN 1 ELSE 0 END) AS ready,
                SUM(CASE WHEN m.status = 'probed' THEN 1 ELSE 0 END) AS probed,
                SUM(CASE WHEN m.status = 'error' THEN 1 ELSE 0 END) AS errors
            FROM cameras c
            LEFT JOIN media_files m ON m.camera_id = c.id
            GROUP BY c.id, c.name, c.location, c.kind
            ORDER BY c.id
            """
        )
    ]
    jobs = [
        dict(row)
        for row in connection.execute(
            """
            SELECT job_type, status, COUNT(*) AS jobs
            FROM processing_jobs
            GROUP BY job_type, status
            ORDER BY job_type, status
            """
        )
    ]
    totals = dict(
        connection.execute(
            """
            SELECT COUNT(*) AS files, COALESCE(SUM(size_bytes), 0) AS bytes
            FROM media_files
            """
        ).fetchone()
    )
    audio_events = [
        dict(row)
        for row in connection.execute(
            """
            SELECT event_label, COUNT(*) AS events,
                   MAX(confidence) AS maximum_confidence
            FROM audio_events
            GROUP BY event_label
            ORDER BY event_label
            """
        )
    ]
    return {
        "totals": totals,
        "cameras": cameras,
        "jobs": jobs,
        "audio_events": audio_events,
    }
