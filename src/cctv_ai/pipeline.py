from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from queue import SimpleQueue
from typing import Any, Callable

from .audio_events import YamnetDetector
from .config import load_config
from .database import (
    backup_database,
    connect_database,
    finish_pipeline_run,
    get_latest_pipeline_run,
    get_pipeline_worker_state,
    get_queued_jobs,
    get_ready_files,
    mark_job_error,
    mark_job_running,
    recover_interrupted_pipeline_runs,
    recover_stale_running_jobs,
    save_audio_events,
    save_probe,
    save_probe_error,
    save_time_segments,
    start_pipeline_run,
    sync_cameras,
    update_pipeline_worker_state,
)
from .media import probe_media
from .scanner import scan_archive


class PipelineAlreadyRunning(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineOptions:
    database_path: Path
    config_path: Path
    source_root: str | Path | None
    camera_ids: tuple[str, ...]
    since: datetime | None
    until: datetime | None = None
    scan: bool = True
    probe_limit_per_camera: int = 4
    probe_workers: int = 1
    audio_limit_per_camera: int = 2
    audio_workers: int = 1
    watermark_limit_per_camera: int = 1
    watermark_workers: int = 1
    watermark_event_label: str = "baby_cry"
    ffprobe: str = "ffprobe"
    ffmpeg: str | None = None
    audio_timeout_seconds: float = 180.0
    watermark_hwaccel: str | None = None
    watermark_interval_seconds: float | None = None
    watermark_minimum_confidence: float = 0.90
    stale_job_seconds: int = 2 * 60 * 60
    error_retry_delay_seconds: int = 6 * 60 * 60
    lock_path: Path = Path(".data/cctv-pipeline.lock")
    mode: str = "backfill"

    def validate(self) -> None:
        if not self.camera_ids:
            raise ValueError("At least one camera id is required")
        if self.until is not None and self.since is not None:
            if self.until <= self.since:
                raise ValueError("Pipeline until must be after since")
        for name, value in (
            ("probe_limit_per_camera", self.probe_limit_per_camera),
            ("audio_limit_per_camera", self.audio_limit_per_camera),
            (
                "watermark_limit_per_camera",
                self.watermark_limit_per_camera,
            ),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.stale_job_seconds <= 0:
            raise ValueError("stale_job_seconds must be positive")
        if self.error_retry_delay_seconds < 0:
            raise ValueError(
                "error_retry_delay_seconds cannot be negative"
            )
        if self.audio_timeout_seconds <= 0:
            raise ValueError("audio_timeout_seconds must be positive")
        for name, value in (
            ("probe_workers", self.probe_workers),
            ("audio_workers", self.audio_workers),
            ("watermark_workers", self.watermark_workers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.watermark_interval_seconds is not None
            and self.watermark_interval_seconds <= 0
        ):
            raise ValueError(
                "watermark_interval_seconds must be positive"
            )
        if not 0 < self.watermark_minimum_confidence <= 1:
            raise ValueError(
                "watermark_minimum_confidence must be in (0, 1]"
            )

    @property
    def since_ts(self) -> int | None:
        return int(self.since.timestamp()) if self.since else None

    @property
    def until_ts(self) -> int | None:
        return int(self.until.timestamp()) if self.until else None

    def public_options(self) -> dict[str, Any]:
        return {
            "scan": self.scan,
            "probe_limit_per_camera": self.probe_limit_per_camera,
            "probe_workers": self.probe_workers,
            "audio_limit_per_camera": self.audio_limit_per_camera,
            "audio_workers": self.audio_workers,
            "watermark_limit_per_camera": (
                self.watermark_limit_per_camera
            ),
            "watermark_workers": self.watermark_workers,
            "watermark_event_label": self.watermark_event_label,
            "audio_timeout_seconds": self.audio_timeout_seconds,
            "watermark_hwaccel": self.watermark_hwaccel,
            "watermark_interval_seconds": (
                self.watermark_interval_seconds
            ),
            "watermark_minimum_confidence": (
                self.watermark_minimum_confidence
            ),
            "stale_job_seconds": self.stale_job_seconds,
            "error_retry_delay_seconds": (
                self.error_retry_delay_seconds
            ),
        }

    def scope(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "camera_ids": list(self.camera_ids),
            "since_ts": self.since_ts,
            "until_ts": self.until_ts,
            **self.public_options(),
        }


class PipelineLock:
    def __init__(self, path: str | Path, *, label: str) -> None:
        self.path = Path(path)
        self.label = label
        self._handle = None

    def __enter__(self) -> PipelineLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            handle.close()
            raise PipelineAlreadyRunning(
                f"{self.label} is already running"
            ) from exc

        metadata = {
            "label": self.label,
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
            "locked_at": int(time.time()),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        )
        handle.flush()
        handle.seek(0)
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _source_path(source_root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ValueError(f"Unsafe media path: {relative_path}")
    parsed = PurePosixPath(relative_path)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or ".." in parsed.parts
        or any(":" in part for part in parsed.parts)
    ):
        raise ValueError(f"Unsafe media path: {relative_path}")
    return source_root.joinpath(*parsed.parts)


def _stage() -> dict[str, Any]:
    return {
        "selected": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "events": {},
        "items": [],
    }


def _retry_sqlite_write(
    operation: Callable[[], Any],
    *,
    attempts: int = 8,
) -> Any:
    """Wait out a competing local SQLite writer without losing a batch."""
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


def _add_item(
    stage: dict[str, Any],
    *,
    relative_path: str,
    camera_id: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    if len(stage["items"]) >= 100:
        return
    stage["items"].append(
        {
            "relative_path": relative_path,
            "camera_id": camera_id,
            "status": status,
            **(details or {}),
        }
    )


def _audio_metadata(config, analysis) -> dict[str, Any]:
    return {
        "windows_seen": analysis.windows_seen,
        "raw_detections": analysis.raw_detections,
        "model": analysis.model,
        "model_sha256": analysis.model_sha256,
        "minimum_confidence": analysis.minimum_confidence,
        "merge_gap_seconds": analysis.merge_gap_seconds,
        "labels": list(config.audio_event_labels),
    }


def _run_pipeline_locked(options: PipelineOptions) -> dict[str, Any]:
    options.validate()
    started_monotonic = time.monotonic()
    hostname = socket.gethostname()
    process_id = os.getpid()
    config = load_config(options.config_path)
    if not config.source_read_only:
        raise ValueError("Pipeline requires source_read_only=true")
    if config.cloud_upload:
        raise ValueError("Pipeline requires cloud_upload=false")

    camera_map = {camera.id: camera for camera in config.cameras}
    unknown = sorted(set(options.camera_ids) - set(camera_map))
    if unknown:
        raise ValueError("Unknown camera ids: " + ", ".join(unknown))
    media_root = config.source_root(
        str(options.source_root)
        if options.source_root is not None
        else None
    )
    if not media_root.is_dir():
        raise FileNotFoundError(
            f"Source root does not exist: {media_root}"
        )

    result: dict[str, Any] = {
        "backup": None,
        "scan": None,
        "recovered_interrupted_runs": 0,
        "recovered_stale_jobs": 0,
        "probe": _stage(),
        "audio": _stage(),
        "watermark": _stage(),
    }
    if options.scan:
        backup_path = backup_database(
            options.database_path,
            label="automatic-pre-scan",
            reuse_daily=True,
        )
        result["backup"] = (
            str(backup_path) if backup_path is not None else None
        )
    connection = connect_database(options.database_path)
    try:
        sync_cameras(connection, config.cameras)
        result["recovered_interrupted_runs"] = (
            recover_interrupted_pipeline_runs(connection)
        )
        result["recovered_stale_jobs"] = recover_stale_running_jobs(
            connection,
            stale_before_ts=int(time.time()) - options.stale_job_seconds,
        )
        connection.commit()
        run_id = start_pipeline_run(
            connection,
            mode=options.mode,
            source_root=str(media_root),
            camera_ids=options.camera_ids,
            since_ts=options.since_ts,
            until_ts=options.until_ts,
            options=options.public_options(),
            hostname=hostname,
            process_id=process_id,
        )
    finally:
        connection.close()

    try:
        if options.scan:
            scan = scan_archive(
                config,
                options.database_path,
                media_root,
                since=options.since,
                camera_ids=options.camera_ids,
            )
            result["scan"] = {
                "scan_run_id": scan.scan_run_id,
                "files_seen": scan.files_seen,
                "parse_errors": scan.parse_errors,
                "statuses": scan.statuses,
            }

        connection = connect_database(options.database_path)
        try:
            probe_stage = result["probe"]
            probe_rows: list[tuple[dict[str, Any], Any]] = []
            for camera_id in options.camera_ids:
                rows = get_ready_files(
                    connection,
                    options.probe_limit_per_camera,
                    camera_id,
                    since_ts=options.since_ts,
                    until_ts=options.until_ts,
                    newest_first=True,
                )
                camera = camera_map[camera_id]
                probe_rows.extend(
                    (dict(row), camera) for row in rows
                )
            probe_stage["selected"] = len(probe_rows)

            def analyze_probe(row):
                relative_path = str(row["relative_path"])
                path = _source_path(media_root, relative_path)
                return probe_media(path, options.ffprobe)

            if probe_rows:
                probe_worker_count = min(
                    options.probe_workers,
                    len(probe_rows),
                )
                with ThreadPoolExecutor(
                    max_workers=probe_worker_count,
                    thread_name_prefix="cctv-probe",
                ) as executor:
                    probe_futures = {
                        executor.submit(analyze_probe, row): (
                            row,
                            camera,
                        )
                        for row, camera in probe_rows
                    }
                    for future in as_completed(probe_futures):
                        row, camera = probe_futures[future]
                        relative_path = str(row["relative_path"])
                        camera_id = str(row["camera_id"])
                        try:
                            probe = future.result()
                            _retry_sqlite_write(
                                lambda: save_probe(
                                    connection,
                                    row,
                                    probe,
                                    audio_enabled=(
                                        config.audio_enabled
                                        and camera.audio_enabled
                                    ),
                                    audio_event_labels=(
                                        config.audio_event_labels
                                    ),
                                )
                            )
                            probe_stage["completed"] += 1
                            _add_item(
                                probe_stage,
                                relative_path=relative_path,
                                camera_id=camera_id,
                                status="completed",
                                details={
                                    "duration_seconds": (
                                        probe.duration_seconds
                                    ),
                                    "audio_codec": probe.audio_codec,
                                },
                            )
                        except Exception as exc:
                            _retry_sqlite_write(
                                lambda: save_probe_error(
                                    connection,
                                    int(row["id"]),
                                    str(exc),
                                )
                            )
                            probe_stage["failed"] += 1
                            _add_item(
                                probe_stage,
                                relative_path=relative_path,
                                camera_id=camera_id,
                                status="failed",
                                details={"error": str(exc)[:1000]},
                            )
                        _retry_sqlite_write(connection.commit)

            audio_rows: list[Any] = []
            for camera_id in options.camera_ids:
                rows = get_queued_jobs(
                    connection,
                    "audio_event_detection",
                    options.audio_limit_per_camera,
                    camera_id,
                    since_ts=options.since_ts,
                    until_ts=options.until_ts,
                    newest_first=True,
                    error_retry_delay_seconds=(
                        options.error_retry_delay_seconds
                    ),
                )
                audio_rows.extend(rows)
            audio_stage = result["audio"]
            audio_stage["selected"] = len(audio_rows)
            detectors: list[YamnetDetector] = []
            if audio_rows:
                if not config.audio_enabled:
                    raise ValueError(
                        "Audio analysis is disabled in project config"
                    )
                detector_count = min(
                    options.audio_workers,
                    len(audio_rows),
                )
                for _ in range(detector_count):
                    detectors.append(
                        YamnetDetector(
                            model_path=config.audio_model_path,
                            class_map_path=(
                                config.audio_label_map_path
                            ),
                            event_labels=(
                                config.audio_event_labels
                            ),
                            minimum_confidence=(
                                config.audio_minimum_confidence
                            ),
                            merge_gap_seconds=(
                                config.audio_merge_gap_seconds
                            ),
                        )
                    )

            aggregate_events: Counter[str] = Counter()
            claimed_audio_rows: list[dict[str, Any]] = []
            for row in audio_rows:
                job_id = int(row["job_id"])
                claimed = _retry_sqlite_write(
                    lambda: mark_job_running(connection, job_id)
                )
                if not claimed:
                    audio_stage["skipped"] += 1
                    continue
                _retry_sqlite_write(connection.commit)
                claimed_audio_rows.append(dict(row))

            detector_pool: SimpleQueue = SimpleQueue()
            for detector in detectors:
                detector_pool.put(detector)

            def analyze_audio_row(row):
                detector = detector_pool.get()
                try:
                    path = _source_path(
                        media_root,
                        str(row["relative_path"]),
                    )
                    return detector.analyze(
                        path,
                        ffmpeg=options.ffmpeg,
                        timeout_seconds=(
                            options.audio_timeout_seconds
                        ),
                    )
                finally:
                    detector_pool.put(detector)

            def save_audio_result(row, analysis, error) -> None:
                job_id = int(row["job_id"])
                relative_path = str(row["relative_path"])
                camera_id = str(row["camera_id"])
                if error is None:
                    counts = Counter(
                        event.label for event in analysis.events
                    )
                    aggregate_events.update(counts)
                    _retry_sqlite_write(
                        lambda: save_audio_events(
                            connection,
                            job_id=job_id,
                            media_file_id=int(row["id"]),
                            events=analysis.events,
                            model_name=analysis.model,
                            metadata=_audio_metadata(config, analysis),
                        )
                    )
                    audio_stage["completed"] += 1
                    _add_item(
                        audio_stage,
                        relative_path=relative_path,
                        camera_id=camera_id,
                        status="completed",
                        details={
                            "windows_seen": analysis.windows_seen,
                            "events": dict(sorted(counts.items())),
                        },
                    )
                else:
                    _retry_sqlite_write(
                        lambda: mark_job_error(
                            connection,
                            job_id,
                            str(error),
                        )
                    )
                    audio_stage["failed"] += 1
                    _add_item(
                        audio_stage,
                        relative_path=relative_path,
                        camera_id=camera_id,
                        status="failed",
                        details={"error": str(error)[:1000]},
                    )
                _retry_sqlite_write(connection.commit)

            if claimed_audio_rows:
                worker_count = min(
                    len(detectors),
                    len(claimed_audio_rows),
                )
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="cctv-audio",
                ) as executor:
                    futures = {
                        executor.submit(
                            analyze_audio_row,
                            row,
                        ): row
                        for row in claimed_audio_rows
                    }
                    for future in as_completed(futures):
                        row = futures[future]
                        try:
                            analysis = future.result()
                            error = None
                        except Exception as exc:
                            analysis = None
                            error = exc
                        save_audio_result(row, analysis, error)
            audio_stage["events"] = dict(
                sorted(aggregate_events.items())
            )

            watermark_stage = result["watermark"]
            watermark_rows: list[tuple[Any, Any]] = []
            for camera_id in options.camera_ids:
                camera = camera_map[camera_id]
                if camera.watermark_crop is None:
                    continue
                rows = get_queued_jobs(
                    connection,
                    "watermark_time_map",
                    options.watermark_limit_per_camera,
                    camera_id,
                    audio_event_label=(
                        options.watermark_event_label
                    ),
                    since_ts=options.since_ts,
                    until_ts=options.until_ts,
                    newest_first=True,
                    error_retry_delay_seconds=(
                        options.error_retry_delay_seconds
                    ),
                )
                watermark_rows.extend((row, camera) for row in rows)
            watermark_stage["selected"] = len(watermark_rows)
            if watermark_rows:
                from .watermark_ocr import run_watermark_ocr

            claimed_watermark_rows: list[tuple[dict[str, Any], Any]] = []
            for row, camera in watermark_rows:
                job_id = int(row["job_id"])
                claimed = _retry_sqlite_write(
                    lambda: mark_job_running(connection, job_id)
                )
                if not claimed:
                    watermark_stage["skipped"] += 1
                    continue
                _retry_sqlite_write(connection.commit)
                claimed_watermark_rows.append((dict(row), camera))

            def analyze_watermark(row, camera):
                relative_path = str(row["relative_path"])
                path = _source_path(media_root, relative_path)
                interval = (
                    options.watermark_interval_seconds
                    or camera.watermark_sample_seconds
                )
                selected_hwaccel = options.watermark_hwaccel
                fallback_error = None
                try:
                    run = run_watermark_ocr(
                        path,
                        timezone=config.timezone,
                        crop=camera.watermark_crop,
                        interval_seconds=interval,
                        ffmpeg=options.ffmpeg,
                        hwaccel=selected_hwaccel,
                        minimum_confidence=(
                            options.watermark_minimum_confidence
                        ),
                    )
                except Exception as exc:
                    if not selected_hwaccel:
                        raise
                    fallback_error = str(exc)
                    selected_hwaccel = None
                    run = run_watermark_ocr(
                        path,
                        timezone=config.timezone,
                        crop=camera.watermark_crop,
                        interval_seconds=interval,
                        ffmpeg=options.ffmpeg,
                        hwaccel=None,
                        minimum_confidence=(
                            options.watermark_minimum_confidence
                        ),
                    )
                metadata = {
                    "frames_seen": run.frames_seen,
                    "valid_observations": len(run.observations),
                    "ocr_failures": run.ocr_failures,
                    "segments": len(run.segments),
                    "sample_interval_seconds": interval,
                    "minimum_confidence": (
                        options.watermark_minimum_confidence
                    ),
                    "hwaccel": selected_hwaccel,
                    "hwaccel_fallback_error": fallback_error,
                    "model": run.model,
                }
                return run, metadata

            if claimed_watermark_rows:
                watermark_worker_count = min(
                    options.watermark_workers,
                    len(claimed_watermark_rows),
                )
                with ThreadPoolExecutor(
                    max_workers=watermark_worker_count,
                    thread_name_prefix="cctv-watermark",
                ) as executor:
                    watermark_futures = {
                        executor.submit(
                            analyze_watermark,
                            row,
                            camera,
                        ): (row, camera)
                        for row, camera in claimed_watermark_rows
                    }
                    for future in as_completed(watermark_futures):
                        row, camera = watermark_futures[future]
                        job_id = int(row["job_id"])
                        relative_path = str(row["relative_path"])
                        camera_id = str(row["camera_id"])
                        try:
                            run, metadata = future.result()
                            _retry_sqlite_write(
                                lambda: save_time_segments(
                                    connection,
                                    job_id=job_id,
                                    media_file_id=int(row["id"]),
                                    segments=run.segments,
                                    source="watermark_ocr:rapidocr",
                                    metadata=metadata,
                                )
                            )
                            watermark_stage["completed"] += 1
                            _add_item(
                                watermark_stage,
                                relative_path=relative_path,
                                camera_id=camera_id,
                                status="completed",
                                details=metadata,
                            )
                        except Exception as exc:
                            _retry_sqlite_write(
                                lambda: mark_job_error(
                                    connection,
                                    job_id,
                                    str(exc),
                                )
                            )
                            watermark_stage["failed"] += 1
                            _add_item(
                                watermark_stage,
                                relative_path=relative_path,
                                camera_id=camera_id,
                                status="failed",
                                details={"error": str(exc)[:1000]},
                            )
                        _retry_sqlite_write(connection.commit)

            failures = sum(
                int(result[name]["failed"])
                for name in ("probe", "audio", "watermark")
            )
            scan_errors = int(
                (result["scan"] or {}).get("parse_errors", 0)
            )
            status = (
                "partial"
                if failures > 0 or scan_errors > 0
                else "completed"
            )
            result["duration_seconds"] = round(
                time.monotonic() - started_monotonic,
                3,
            )
            _retry_sqlite_write(
                lambda: finish_pipeline_run(
                    connection,
                    run_id,
                    status=status,
                    result=result,
                )
            )
            return {
                "pipeline_run_id": run_id,
                "status": status,
                **result,
            }
        finally:
            connection.close()
    except BaseException as exc:
        result["duration_seconds"] = round(
            time.monotonic() - started_monotonic,
            3,
        )
        failure_connection = connect_database(options.database_path)
        try:
            _retry_sqlite_write(
                lambda: finish_pipeline_run(
                    failure_connection,
                    run_id,
                    status="failed",
                    result=result,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            failure_connection.close()
        raise


def run_pipeline_once(options: PipelineOptions) -> dict[str, Any]:
    with PipelineLock(options.lock_path, label="CCTV pipeline cycle"):
        return _run_pipeline_locked(options)


def _worker_heartbeat_interval(interval_seconds: int) -> int:
    return max(5, min(30, interval_seconds // 2))


def _persist_pipeline_worker_state(
    options: PipelineOptions,
    **state: Any,
) -> None:
    """Persist worker state despite a transient competing SQLite writer."""
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(6):
        connection = None
        try:
            connection = connect_database(options.database_path)
            update_pipeline_worker_state(connection, **state)
            return
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "locked" not in message and "busy" not in message:
                raise
            last_error = exc
            time.sleep(min(1.0, 0.05 * (2**attempt)))
        finally:
            if connection is not None:
                connection.close()
    if last_error is not None:
        raise last_error


def run_pipeline_worker(
    options: PipelineOptions,
    *,
    interval_seconds: int,
    scan_every_cycles: int = 2,
    max_cycles: int | None = None,
    worker_lock_path: str | Path | None = None,
    on_cycle: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be at least 1")
    if scan_every_cycles <= 0:
        raise ValueError("scan_every_cycles must be positive")
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be positive")

    hostname = socket.gethostname()
    process_id = os.getpid()
    worker_lock = Path(
        worker_lock_path
        or options.lock_path.with_name("cctv-pipeline-worker.lock")
    )
    started_at = int(time.time())
    cycles = 0
    last_error = None
    with PipelineLock(worker_lock, label="CCTV pipeline worker"):
        heartbeat_stop = threading.Event()
        heartbeat_interval = _worker_heartbeat_interval(
            interval_seconds
        )

        def write_worker_heartbeat() -> None:
            _persist_pipeline_worker_state(
                options,
                status=("error" if last_error else "running"),
                hostname=hostname,
                process_id=process_id,
                interval_seconds=interval_seconds,
                scope=options.scope(),
                last_error=last_error,
            )

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(heartbeat_interval):
                try:
                    write_worker_heartbeat()
                except Exception:
                    # The normal cycle update will retry and surface errors.
                    continue

        _persist_pipeline_worker_state(
            options,
            status="running",
            hostname=hostname,
            process_id=process_id,
            interval_seconds=interval_seconds,
            scope=options.scope(),
            started_at=started_at,
        )

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name="cctv-pipeline-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            while max_cycles is None or cycles < max_cycles:
                cycle_options = replace(
                    options,
                    scan=(
                        options.scan
                        and cycles % scan_every_cycles == 0
                    ),
                )
                payload: dict[str, Any]
                try:
                    payload = run_pipeline_once(cycle_options)
                    last_error = None
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    payload = {
                        "status": "failed",
                        "error": last_error,
                    }
                cycles += 1
                payload["worker_cycle"] = cycles
                if on_cycle is not None:
                    on_cycle(payload)

                try:
                    _persist_pipeline_worker_state(
                        options,
                        status=(
                            "error" if last_error else "running"
                        ),
                        hostname=hostname,
                        process_id=process_id,
                        interval_seconds=interval_seconds,
                        scope=options.scope(),
                        last_error=last_error,
                    )
                except sqlite3.OperationalError as exc:
                    # The processing cycle is already durable. A heartbeat
                    # collision must not discard it or stop future scans.
                    last_error = f"{type(exc).__name__}: {exc}"

                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(interval_seconds)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)
            try:
                _persist_pipeline_worker_state(
                    options,
                    status=(
                        "error" if last_error else "stopped"
                    ),
                    hostname=hostname,
                    process_id=process_id,
                    interval_seconds=interval_seconds,
                    scope=options.scope(),
                    last_error=last_error,
                )
            except sqlite3.OperationalError:
                # A stale heartbeat is preferable to crashing during shutdown.
                pass
    return cycles


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def query_pipeline_progress(
    connection,
    *,
    now_ts: int | None = None,
) -> dict[str, Any]:
    timestamp = int(now_ts if now_ts is not None else time.time())
    latest = get_latest_pipeline_run(connection)
    worker = get_pipeline_worker_state(connection)

    worker_scope = (
        _json_object(worker["scope_json"]) if worker is not None else {}
    )
    if worker_scope:
        scope = worker_scope
    elif latest is not None:
        scope = {
            "mode": latest["mode"],
            "camera_ids": _json_list(latest["camera_ids_json"]),
            "since_ts": latest["since_ts"],
            "until_ts": latest["until_ts"],
            **_json_object(latest["options_json"]),
        }
    else:
        scope = {}

    camera_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in scope.get("camera_ids", [])
            if value
        )
    )
    conditions: list[str] = []
    parameters: list[Any] = []
    if camera_ids:
        placeholders = ",".join("?" for _ in camera_ids)
        conditions.append(f"m.camera_id IN ({placeholders})")
        parameters.extend(camera_ids)
    since_ts = scope.get("since_ts")
    until_ts = scope.get("until_ts")
    if since_ts is not None:
        conditions.append("m.capture_start_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("m.capture_start_ts < ?")
        parameters.append(until_ts)
    where = " AND ".join(conditions) if conditions else "1 = 0"

    media_counts = {
        str(row["status"]): int(row["files"])
        for row in connection.execute(
            f"""
            SELECT m.status, COUNT(*) AS files
            FROM media_files m
            WHERE {where}
            GROUP BY m.status
            """,
            parameters,
        )
    }
    job_counts = {
        f"{row['job_type']}:{row['status']}": int(row["jobs"])
        for row in connection.execute(
            f"""
            SELECT j.job_type, j.status, COUNT(*) AS jobs
            FROM processing_jobs j
            JOIN media_files m ON m.id = j.media_file_id
            WHERE {where}
            GROUP BY j.job_type, j.status
            """,
            parameters,
        )
    }
    watermark_label = str(
        scope.get("watermark_event_label", "baby_cry")
    )
    watermark_counts = {
        str(row["status"]): int(row["jobs"])
        for row in connection.execute(
            f"""
            SELECT j.status, COUNT(*) AS jobs
            FROM processing_jobs j
            JOIN media_files m ON m.id = j.media_file_id
            WHERE {where}
              AND j.job_type = 'watermark_time_map'
              AND EXISTS (
                  SELECT 1
                  FROM audio_events a
                  WHERE a.media_file_id = m.id
                    AND a.event_label = ?
              )
            GROUP BY j.status
            """,
            (*parameters, watermark_label),
        )
    }
    event_counts = {
        str(row["event_label"]): int(row["events"])
        for row in connection.execute(
            f"""
            SELECT a.event_label, COUNT(*) AS events
            FROM audio_events a
            JOIN media_files m ON m.id = a.media_file_id
            WHERE {where}
            GROUP BY a.event_label
            """,
            parameters,
        )
    }

    total_files = sum(media_counts.values())
    audio_completed = job_counts.get(
        "audio_event_detection:completed",
        0,
    )
    progress_percent = (
        round(audio_completed * 100.0 / total_files, 2)
        if total_files
        else 0.0
    )
    recent_errors: list[dict[str, Any]] = []
    for row in connection.execute(
        f"""
        SELECT 'probe' AS stage, m.relative_path, m.last_error AS error,
               m.last_seen_at AS updated_at
        FROM media_files m
        WHERE {where}
          AND m.status = 'error'
          AND m.last_error IS NOT NULL
        ORDER BY m.last_seen_at DESC
        LIMIT 5
        """,
        parameters,
    ):
        recent_errors.append(dict(row))
    for row in connection.execute(
        f"""
        SELECT j.job_type AS stage, m.relative_path,
               j.payload_json, j.updated_at
        FROM processing_jobs j
        JOIN media_files m ON m.id = j.media_file_id
        WHERE {where}
          AND j.status = 'error'
        ORDER BY j.updated_at DESC
        LIMIT 5
        """,
        parameters,
    ):
        payload = _json_object(row["payload_json"])
        recent_errors.append(
            {
                "stage": row["stage"],
                "relative_path": row["relative_path"],
                "error": payload.get("error", "Unknown job error"),
                "updated_at": row["updated_at"],
            }
        )
    recent_errors.sort(
        key=lambda item: int(item.get("updated_at") or 0),
        reverse=True,
    )
    recent_errors = recent_errors[:8]

    worker_payload = None
    if worker is not None:
        heartbeat_age = max(
            0,
            timestamp - int(worker["heartbeat_at"]),
        )
        alive_window = max(
            int(worker["interval_seconds"]) * 2 + 60,
            180,
        )
        worker_payload = {
            "status": worker["status"],
            "alive": (
                worker["status"] == "running"
                and heartbeat_age <= alive_window
            ),
            "started_at": worker["started_at"],
            "heartbeat_at": worker["heartbeat_at"],
            "heartbeat_age_seconds": heartbeat_age,
            "hostname": worker["hostname"],
            "process_id": worker["process_id"],
            "interval_seconds": worker["interval_seconds"],
            "last_error": worker["last_error"],
        }

    latest_payload = None
    if latest is not None:
        latest_payload = {
            "id": latest["id"],
            "status": latest["status"],
            "mode": latest["mode"],
            "started_at": latest["started_at"],
            "finished_at": latest["finished_at"],
            "error": latest["error"],
            "result": _json_object(latest["result_json"]),
        }

    return {
        "configured": bool(camera_ids),
        "scope": {
            **scope,
            "camera_ids": list(camera_ids),
        },
        "worker": worker_payload,
        "latest_run": latest_payload,
        "progress": {
            "total_files": total_files,
            "waiting_files": media_counts.get("waiting", 0),
            "ready_files": media_counts.get("ready", 0),
            "probed_files": media_counts.get("probed", 0),
            "error_files": media_counts.get("error", 0),
            "audio_queued": job_counts.get(
                "audio_event_detection:queued",
                0,
            ),
            "audio_running": job_counts.get(
                "audio_event_detection:running",
                0,
            ),
            "audio_error": job_counts.get(
                "audio_event_detection:error",
                0,
            ),
            "audio_completed": audio_completed,
            "watermark_queued": watermark_counts.get("queued", 0),
            "watermark_running": watermark_counts.get("running", 0),
            "watermark_error": watermark_counts.get("error", 0),
            "watermark_completed": watermark_counts.get(
                "completed",
                0,
            ),
            "progress_percent": progress_percent,
        },
        "event_counts": event_counts,
        "recent_errors": recent_errors,
    }
