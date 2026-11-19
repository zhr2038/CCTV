from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .adapters import iter_camera_files, parse_capture
from .database import (
    connect_database,
    finish_scan,
    observe_file,
    record_scan_error,
    start_scan,
    sync_cameras,
)
from .models import ProjectConfig


@dataclass(frozen=True)
class ScanSummary:
    scan_run_id: int
    files_seen: int
    parse_errors: int
    statuses: dict[str, int]


def scan_archive(
    config: ProjectConfig,
    database_path: str | Path,
    source_root: Path,
    *,
    since: datetime | None = None,
    camera_ids: Iterable[str] | None = None,
    now_ts: int | None = None,
) -> ScanSummary:
    if not config.source_read_only:
        raise ValueError("This indexer requires source_read_only=true")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    selected_ids = tuple(
        dict.fromkeys(str(value) for value in (camera_ids or ()))
    )
    known_ids = {camera.id for camera in config.cameras}
    unknown_ids = sorted(set(selected_ids) - known_ids)
    if unknown_ids:
        raise ValueError(
            "Unknown camera ids: " + ", ".join(unknown_ids)
        )
    selected_cameras = tuple(
        camera
        for camera in config.cameras
        if not selected_ids or camera.id in selected_ids
    )

    timestamp = int(now_ts if now_ts is not None else time.time())
    connection = connect_database(database_path)
    sync_cameras(connection, config.cameras)
    scan_run_id = start_scan(
        connection,
        str(source_root),
        int(since.timestamp()) if since else None,
    )
    files_seen = 0
    parse_errors = 0
    statuses: Counter[str] = Counter()
    try:
        for camera in selected_cameras:
            for path in iter_camera_files(
                source_root, camera, config.timezone, since
            ):
                files_seen += 1
                try:
                    parsed = parse_capture(camera, path, config.timezone)
                    if since and (
                        parsed.capture_end_hint or parsed.capture_start
                    ) < since:
                        continue
                    stat = path.stat()
                    relative_path = path.relative_to(source_root).as_posix()
                    status = observe_file(
                        connection,
                        scan_run_id=scan_run_id,
                        camera_id=camera.id,
                        relative_path=relative_path,
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        parsed=parsed,
                        now_ts=timestamp,
                        stability_age_seconds=config.stability_age_minutes * 60,
                        required_observations=config.stable_observations,
                    )
                    statuses[status] += 1
                except (OSError, ValueError) as exc:
                    parse_errors += 1
                    try:
                        relative_path = path.relative_to(source_root).as_posix()
                    except ValueError:
                        relative_path = str(path)
                    record_scan_error(
                        connection,
                        scan_run_id=scan_run_id,
                        camera_id=camera.id,
                        relative_path=relative_path,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if files_seen % 1000 == 0:
                    connection.commit()
        connection.commit()
        finish_scan(
            connection,
            scan_run_id,
            files_seen=files_seen,
            parse_errors=parse_errors,
        )
    except Exception:
        connection.rollback()
        finish_scan(
            connection,
            scan_run_id,
            files_seen=files_seen,
            parse_errors=parse_errors,
            status="failed",
        )
        raise
    finally:
        connection.close()

    return ScanSummary(
        scan_run_id=scan_run_id,
        files_seen=files_seen,
        parse_errors=parse_errors,
        statuses=dict(statuses),
    )
