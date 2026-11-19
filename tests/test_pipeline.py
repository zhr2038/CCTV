from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from cctv_ai.database import (
    connect_database,
    update_pipeline_worker_state,
)
from cctv_ai.pipeline import (
    PipelineAlreadyRunning,
    PipelineLock,
    PipelineOptions,
    _retry_sqlite_write,
    _worker_heartbeat_interval,
    query_pipeline_progress,
    run_pipeline_once,
    run_pipeline_worker,
)


def write_config(path: Path, source_root: Path) -> None:
    path.write_text(
        f"""
[project]
timezone = "UTC"
source_root_windows = "{source_root.as_posix()}"
source_root_linux = "{source_root.as_posix()}"
stability_age_minutes = 90
stable_observations = 2

[privacy]
source_read_only = true
face_recognition = false
cloud_upload = false

[audio]
enabled = true
transcription_mode = "on_demand"
store_continuous_transcripts = false
model_path = "missing-model.tflite"
label_map_path = "missing-labels.csv"
minimum_confidence = 0.60
merge_gap_seconds = 1.0
event_labels = ["baby_cry"]

[[cameras]]
id = "cam"
name = "camera"
location = "test"
kind = "aqara_event"
relative_root = "camera"
audio_enabled = true
""".strip(),
        encoding="utf-8",
    )


class PipelineTests(unittest.TestCase):
    def test_sqlite_write_retries_transient_lock(self) -> None:
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return "saved"

        with patch("cctv_ai.pipeline.time.sleep"):
            self.assertEqual(_retry_sqlite_write(operation), "saved")
        self.assertEqual(attempts, 3)

    def test_worker_heartbeat_interval_is_bounded(self) -> None:
        self.assertEqual(_worker_heartbeat_interval(1), 5)
        self.assertEqual(_worker_heartbeat_interval(10), 5)
        self.assertEqual(_worker_heartbeat_interval(60), 30)
        self.assertEqual(_worker_heartbeat_interval(3_600), 30)

    def test_lock_rejects_a_second_pipeline_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "pipeline.lock"
            with PipelineLock(lock_path, label="first"):
                with self.assertRaises(PipelineAlreadyRunning):
                    with PipelineLock(lock_path, label="second"):
                        self.fail("second lock should not be acquired")

    def test_bounded_empty_cycle_and_worker_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = root / "config.toml"
            write_config(config, source)
            database = root / "index.sqlite3"
            options = PipelineOptions(
                database_path=database,
                config_path=config,
                source_root=source,
                camera_ids=("cam",),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                scan=False,
                probe_limit_per_camera=0,
                audio_limit_per_camera=0,
                watermark_limit_per_camera=0,
                lock_path=root / "cycle.lock",
            )

            result = run_pipeline_once(options)
            cycles = run_pipeline_worker(
                options,
                interval_seconds=1,
                max_cycles=1,
                worker_lock_path=root / "worker.lock",
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["probe"]["selected"], 0)
            self.assertEqual(cycles, 1)
            connection = connect_database(database)
            try:
                runs = connection.execute(
                    "SELECT status FROM pipeline_runs ORDER BY id"
                ).fetchall()
                worker = connection.execute(
                    "SELECT status FROM pipeline_worker_state WHERE id = 1"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                [row["status"] for row in runs],
                ["completed", "completed"],
            )
            self.assertEqual(worker["status"], "stopped")

    def test_pipeline_worker_counts_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = root / "config.toml"
            write_config(config, source)
            options = PipelineOptions(
                database_path=root / "index.sqlite3",
                config_path=config,
                source_root=source,
                camera_ids=("cam",),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                probe_workers=0,
            )
            with self.assertRaisesRegex(
                ValueError,
                "probe_workers must be positive",
            ):
                options.validate()

    def test_progress_uses_latest_scope_and_job_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = root / "config.toml"
            write_config(config, source)
            database = root / "index.sqlite3"
            options = PipelineOptions(
                database_path=database,
                config_path=config,
                source_root=source,
                camera_ids=("cam",),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                scan=False,
                probe_limit_per_camera=0,
                audio_limit_per_camera=0,
                watermark_limit_per_camera=0,
                lock_path=root / "cycle.lock",
            )
            run_pipeline_once(options)

            connection = connect_database(database)
            try:
                ready = connection.execute(
                    """
                    INSERT INTO media_files(
                        camera_id, relative_path, size_bytes, mtime_ns,
                        first_seen_at, last_seen_at, status,
                        capture_start_ts, time_source,
                        requires_watermark_ocr
                    )
                    VALUES ('cam', 'camera/ready.mp4', 10, 1, 1, 1,
                            'ready', 1767225600, 'test', 0)
                    """
                )
                probed = connection.execute(
                    """
                    INSERT INTO media_files(
                        camera_id, relative_path, size_bytes, mtime_ns,
                        first_seen_at, last_seen_at, status,
                        capture_start_ts, duration_seconds, time_source,
                        requires_watermark_ocr
                    )
                    VALUES ('cam', 'camera/probed.mp4', 10, 1, 1, 1,
                            'probed', 1767225601, 10, 'test', 0)
                    """
                )
                del ready
                media_id = int(probed.lastrowid)
                connection.execute(
                    """
                    INSERT INTO processing_jobs(
                        media_file_id, job_type, status, priority,
                        payload_json, created_at, updated_at
                    )
                    VALUES (?, 'audio_event_detection', 'completed',
                            30, '{}', 10, 10)
                    """,
                    (media_id,),
                )
                connection.execute(
                    """
                    INSERT INTO audio_events(
                        media_file_id, event_label, model_label,
                        model_class_index, media_start_seconds,
                        media_end_seconds, wall_start_ts, wall_end_ts,
                        confidence, windows, model_name, created_at
                    )
                    VALUES (?, 'baby_cry', 'Baby cry, infant cry',
                            20, 1, 2, 1767225602, 1767225603,
                            0.9, 1, 'test', 10)
                    """,
                    (media_id,),
                )
                update_pipeline_worker_state(
                    connection,
                    status="running",
                    hostname="test-host",
                    process_id=123,
                    interval_seconds=30,
                    scope=options.scope(),
                    started_at=100,
                    now_ts=120,
                )
                progress = query_pipeline_progress(
                    connection,
                    now_ts=130,
                )
            finally:
                connection.close()

            self.assertTrue(progress["configured"])
            self.assertTrue(progress["worker"]["alive"])
            self.assertEqual(progress["progress"]["total_files"], 2)
            self.assertEqual(progress["progress"]["ready_files"], 1)
            self.assertEqual(progress["progress"]["probed_files"], 1)
            self.assertEqual(progress["progress"]["audio_completed"], 1)
            self.assertEqual(
                progress["progress"]["progress_percent"],
                50.0,
            )
            self.assertEqual(progress["event_counts"]["baby_cry"], 1)


if __name__ == "__main__":
    unittest.main()
