from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cctv_ai.audio_events import AudioEvent
from cctv_ai.database import (
    backfill_audio_event_times,
    backup_database,
    connect_database,
    get_queued_jobs,
    observe_file,
    query_audio_event_fingerprints,
    query_episode_events,
    query_latest_shadow_scores,
    recover_interrupted_pipeline_runs,
    recover_stale_running_jobs,
    save_audio_events,
    save_audio_event_fingerprint,
    save_event_reviews,
    start_scan,
    sync_cameras,
)
from cctv_ai.models import CameraConfig, ParsedCapture


class DatabaseTests(unittest.TestCase):
    def test_stale_running_job_is_requeued_without_retry_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="camera",
                location="test",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    duration_seconds, time_source, requires_watermark_ocr
                )
                VALUES ('cam', 'camera/stale.mp4', 100, 1,
                        1, 1, 'probed', 2000, 60, 'test', 0)
                """
            )
            connection.execute(
                """
                INSERT INTO processing_jobs(
                    media_file_id, job_type, status, attempts,
                    created_at, updated_at
                )
                VALUES (?, 'audio_event_detection', 'running', 1, 100, 100)
                """,
                (int(media.lastrowid),),
            )

            recovered = recover_stale_running_jobs(
                connection,
                stale_before_ts=500,
                now_ts=1_000,
            )
            retryable = get_queued_jobs(
                connection,
                "audio_event_detection",
                1,
                error_retry_delay_seconds=6 * 60 * 60,
                now_ts=1_000,
            )
            job = connection.execute(
                """
                SELECT status, attempts, payload_json
                FROM processing_jobs
                WHERE media_file_id = ?
                """,
                (int(media.lastrowid),),
            ).fetchone()

            self.assertEqual(recovered, 1)
            self.assertEqual(len(retryable), 1)
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["attempts"], 1)
            self.assertIn('"retry": "immediate"', job["payload_json"])
            connection.close()

    def test_audio_fingerprint_and_shadow_scores_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="宝宝房间顶部",
                location="baby_room",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            media_id = int(
                connection.execute(
                    """
                    INSERT INTO media_files(
                        camera_id, relative_path, size_bytes, mtime_ns,
                        first_seen_at, last_seen_at, status,
                        capture_start_ts, duration_seconds, time_source,
                        requires_watermark_ocr
                    )
                    VALUES (
                        'cam', 'camera/video.mp4', 100, 1, 1, 1,
                        'probed', 2000, 60, 'test', 0
                    )
                    """
                ).lastrowid
            )
            event_id = int(
                connection.execute(
                    """
                    INSERT INTO audio_events(
                        media_file_id, event_label, model_label,
                        model_class_index, media_start_seconds,
                        media_end_seconds, wall_start_ts, wall_end_ts,
                        confidence, windows, model_name, created_at
                    )
                    VALUES (
                        ?, 'baby_cry', 'cry', 1, 5, 6,
                        2005, 2006, 0.9, 1, 'test', 1
                    )
                    """,
                    (media_id,),
                ).lastrowid
            )
            mean_scores = b"\x00\x00\x00\x3f" * 2
            max_scores = b"\x00\x00\x40\x3f" * 2
            save_audio_event_fingerprint(
                connection,
                audio_event_id=event_id,
                model_sha256="sha",
                vector_length=2,
                mean_scores=mean_scores,
                max_scores=max_scores,
                windows_seen=2,
                media_start_seconds=4.5,
                media_end_seconds=6.5,
                created_at=2,
            )
            fingerprint = query_audio_event_fingerprints(
                connection,
                event_ids=[event_id],
                model_sha256="sha",
            )[event_id]
            self.assertEqual(fingerprint["mean_scores"], mean_scores)
            self.assertEqual(fingerprint["windows_seen"], 2)

            run_id = int(
                connection.execute(
                    """
                    INSERT INTO shadow_calibration_runs(
                        event_label, created_at, finished_at, status,
                        model_name, model_sha256,
                        minimum_confidence, gap_seconds, metrics_json
                    )
                    VALUES (
                        'baby_cry', 2, 3, 'completed', 'shadow',
                        'sha', 0.6, 30, '{"decision_mode":"shadow_only"}'
                    )
                    """
                ).lastrowid
            )
            connection.execute(
                """
                INSERT INTO shadow_episode_scores(
                    run_id, episode_id, event_label, start_ts,
                    review_status, score, margin, positive_similarity,
                    negative_similarity, review_priority, bucket, reason,
                    event_ids_json, feature_json
                )
                VALUES (
                    ?, 'episode', 'baby_cry', 2005, 'pending',
                    0.5, 0, 0.8, 0.8, 1, 'uncertain', 'test',
                    '[1]', '{}'
                )
                """,
                (run_id,),
            )
            connection.commit()

            run, scores = query_latest_shadow_scores(
                connection,
                event_label="baby_cry",
            )
            self.assertEqual(run["id"], run_id)
            self.assertEqual(run["metrics"]["decision_mode"], "shadow_only")
            self.assertEqual(scores["episode"]["bucket"], "uncertain")
            connection.close()

    def test_connect_database_leaves_no_setup_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "index.sqlite3"
            connection = connect_database(database)
            try:
                self.assertFalse(connection.in_transaction)
            finally:
                connection.close()

    def test_connect_database_remains_readable_during_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "index.sqlite3"
            writer = connect_database(database)
            reader = None
            try:
                writer.execute("BEGIN IMMEDIATE")
                reader = connect_database(database)
                self.assertEqual(
                    reader.execute(
                        "SELECT COUNT(*) FROM cameras"
                    ).fetchone()[0],
                    0,
                )
                self.assertFalse(reader.in_transaction)
            finally:
                if reader is not None:
                    reader.close()
                writer.rollback()
                writer.close()

    def test_error_jobs_respect_retry_delay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="camera",
                location="test",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    duration_seconds, time_source, requires_watermark_ocr
                )
                VALUES ('cam', 'camera/video.mp4', 100, 1,
                        1, 1, 'probed', 2000, 60, 'test', 0)
                """
            )
            connection.execute(
                """
                INSERT INTO processing_jobs(
                    media_file_id, job_type, status, attempts,
                    created_at, updated_at
                )
                VALUES (?, 'audio_event_detection', 'error', 1, 800, 900)
                """,
                (int(media.lastrowid),),
            )

            cooling_down = get_queued_jobs(
                connection,
                "audio_event_detection",
                1,
                error_retry_delay_seconds=200,
                now_ts=1_000,
            )
            retryable = get_queued_jobs(
                connection,
                "audio_event_detection",
                1,
                error_retry_delay_seconds=200,
                now_ts=1_100,
            )

            self.assertEqual(cooling_down, [])
            self.assertEqual(len(retryable), 1)
            connection.close()

    def test_database_backup_is_valid_and_daily_copy_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "index.sqlite3"
            connection = connect_database(database)
            connection.execute(
                """
                INSERT INTO cameras(
                    id, name, location, kind, relative_root, audio_enabled
                )
                VALUES ('cam', 'camera', 'test', 'test', 'camera', 1)
                """
            )
            connection.commit()
            connection.close()

            first = backup_database(
                database,
                backup_directory=root / "backups",
                label="automatic-pre-scan",
                reuse_daily=True,
                now_ts=1_800_000_000,
            )
            second = backup_database(
                database,
                backup_directory=root / "backups",
                label="automatic-pre-scan",
                reuse_daily=True,
                now_ts=1_800_000_100,
            )

            self.assertEqual(first, second)
            self.assertIsNotNone(first)
            backup = sqlite3.connect(first)
            try:
                self.assertEqual(
                    backup.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                )
                self.assertEqual(
                    backup.execute(
                        "SELECT COUNT(*) FROM cameras"
                    ).fetchone()[0],
                    1,
                )
            finally:
                backup.close()

    def test_interrupted_pipeline_runs_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            connection.execute(
                """
                INSERT INTO pipeline_runs(
                    started_at, status, mode, source_root,
                    camera_ids_json, hostname, process_id
                )
                VALUES (100, 'running', 'test', '/archive',
                        '[]', 'host', 123)
                """
            )
            connection.execute(
                """
                INSERT INTO pipeline_runs(
                    started_at, finished_at, status, mode, source_root,
                    camera_ids_json, hostname, process_id
                )
                VALUES (90, 95, 'completed', 'test', '/archive',
                        '[]', 'host', 122)
                """
            )

            recovered = recover_interrupted_pipeline_runs(
                connection,
                now_ts=200,
            )
            rows = connection.execute(
                "SELECT status, finished_at, error FROM pipeline_runs ORDER BY id"
            ).fetchall()

            self.assertEqual(recovered, 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["finished_at"], 200)
            self.assertIn("interrupted", rows[0]["error"])
            self.assertEqual(rows[1]["status"], "completed")
            connection.close()

    def test_file_requires_two_stable_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="camera",
                location="test",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            run_id = start_scan(connection, "/archive", None)
            parsed = ParsedCapture(
                capture_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                capture_end_hint=None,
                time_source="test",
                requires_watermark_ocr=False,
            )
            common = dict(
                connection=connection,
                scan_run_id=run_id,
                camera_id="cam",
                relative_path="camera/20260101/000000.mp4",
                size_bytes=100,
                mtime_ns=1_700_000_000_000_000_000,
                parsed=parsed,
                now_ts=1_800_000_000,
                stability_age_seconds=90 * 60,
                required_observations=2,
            )
            self.assertEqual(observe_file(**common), "waiting")
            self.assertEqual(observe_file(**common), "ready")
            connection.close()

    def test_smb_sub_millisecond_mtime_rounding_preserves_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="camera",
                location="test",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            run_id = start_scan(connection, "/archive", None)
            parsed = ParsedCapture(
                capture_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                capture_end_hint=None,
                time_source="test",
                requires_watermark_ocr=False,
            )
            common = dict(
                connection=connection,
                scan_run_id=run_id,
                camera_id="cam",
                relative_path="camera/video.mp4",
                size_bytes=100,
                mtime_ns=1_700_000_000_000_000_088,
                parsed=parsed,
                now_ts=1_800_000_000,
                stability_age_seconds=90 * 60,
                required_observations=2,
            )
            observe_file(**common)
            observe_file(**common)
            media_id = int(
                connection.execute(
                    "SELECT id FROM media_files"
                ).fetchone()["id"]
            )
            connection.execute(
                "UPDATE media_files SET status = 'probed' WHERE id = ?",
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
                VALUES (?, 'baby_cry', 'cry', 1, 5, 6,
                        1005, 1006, 0.9, 1, 'test', 1)
                """,
                (media_id,),
            )

            rounded = {
                **common,
                "mtime_ns": 1_700_000_000_000_000_000,
            }
            self.assertEqual(observe_file(**rounded), "probed")
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM audio_events"
                ).fetchone()[0],
                1,
            )
            connection.close()

    def test_audio_event_uses_piecewise_watermark_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="bundle",
                name="bedside",
                location="baby_room",
                kind="xiaomi_watermark_bundle",
                relative_root="bundle",
                audio_enabled=True,
                watermark_crop=(0, 0, 100, 20),
            )
            sync_cameras(connection, [camera])
            cursor = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    time_source, requires_watermark_ocr
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "bundle",
                    "bundle/video.mp4",
                    100,
                    1,
                    1,
                    1,
                    "probed",
                    1_700_000_000,
                    "filename_hint",
                    1,
                ),
            )
            media_id = int(cursor.lastrowid)
            job = connection.execute(
                """
                INSERT INTO processing_jobs(
                    media_file_id, job_type, status, created_at, updated_at
                )
                VALUES (?, 'audio_event_detection', 'running', 1, 1)
                """,
                (media_id,),
            )
            save_audio_events(
                connection,
                job_id=int(job.lastrowid),
                media_file_id=media_id,
                events=[
                    AudioEvent(
                        label="baby_cry",
                        model_label="Baby cry, infant cry",
                        model_class_index=20,
                        media_start_seconds=5.0,
                        media_end_seconds=6.0,
                        confidence=0.9,
                    )
                ],
                model_name="test-model",
                metadata={"windows_seen": 1},
            )
            row = connection.execute(
                "SELECT * FROM audio_events"
            ).fetchone()
            self.assertIsNone(row["wall_start_ts"])
            self.assertIsNone(row["wall_end_ts"])
            connection.execute(
                """
                INSERT INTO time_segments(
                    media_file_id, media_start_seconds, media_end_seconds,
                    wall_start_ts, wall_end_ts, source, confidence
                )
                VALUES (?, 2.5, 12.5, 2000, 2010, 'ocr', 0.99)
                """,
                (media_id,),
            )
            backfill_audio_event_times(connection, media_id)
            row = connection.execute(
                "SELECT * FROM audio_events"
            ).fetchone()
            self.assertAlmostEqual(row["wall_start_ts"], 2002.5)
            self.assertAlmostEqual(row["wall_end_ts"], 2003.5)
            status = connection.execute(
                "SELECT status FROM processing_jobs WHERE id = ?",
                (job.lastrowid,),
            ).fetchone()["status"]
            self.assertEqual(status, "completed")
            connection.close()

    def test_event_reviews_round_trip_through_episode_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="宝宝房间顶部",
                location="baby_room",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    duration_seconds, time_source, requires_watermark_ocr
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "cam",
                    "camera/20260101/000000.mp4",
                    100,
                    1,
                    1,
                    1,
                    "probed",
                    2_000,
                    60.0,
                    "test",
                    0,
                ),
            )
            audio_event = connection.execute(
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
                    int(media.lastrowid),
                    "baby_cry",
                    "Baby cry, infant cry",
                    20,
                    5.0,
                    6.0,
                    2_005.0,
                    2_006.0,
                    0.91,
                    1,
                    "test-model",
                    1,
                ),
            )
            event_id = int(audio_event.lastrowid)

            updated = save_event_reviews(
                connection,
                event_ids=[event_id, event_id],
                verdict="uncertain",
                note="需要再听一次",
                reviewer="parent",
            )
            rows = query_episode_events(
                connection,
                event_label="baby_cry",
                since_ts=2_000,
                until_ts=2_100,
            )

            self.assertEqual(updated, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["review_verdict"], "uncertain")
            self.assertEqual(rows[0]["review_note"], "需要再听一次")
            self.assertEqual(rows[0]["reviewer"], "parent")
            with self.assertRaises(ValueError):
                save_event_reviews(
                    connection,
                    event_ids=[event_id],
                    verdict="maybe",
                )
            with self.assertRaises(ValueError):
                save_event_reviews(
                    connection,
                    event_ids=[event_id + 999],
                    verdict="confirmed",
                )
            connection.close()

    def test_audio_reanalysis_restores_review_from_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory) / "index.sqlite3")
            camera = CameraConfig(
                id="cam",
                name="camera",
                location="test",
                kind="aqara_event",
                relative_root="camera",
                audio_enabled=True,
                watermark_crop=None,
            )
            sync_cameras(connection, [camera])
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    duration_seconds, time_source, requires_watermark_ocr
                )
                VALUES ('cam', 'camera/video.mp4', 100, 1700000000000000088,
                        1, 1, 'probed', 2000, 60, 'test', 0)
                """
            )
            media_id = int(media.lastrowid)

            def save() -> int:
                job = connection.execute(
                    """
                    INSERT INTO processing_jobs(
                        media_file_id, job_type, status,
                        created_at, updated_at
                    )
                    VALUES (?, 'audio_event_detection', 'running', 1, 1)
                    ON CONFLICT(media_file_id, job_type) DO UPDATE SET
                        status = 'running'
                    """,
                    (media_id,),
                )
                job_id = int(
                    job.lastrowid
                    or connection.execute(
                        """
                        SELECT id FROM processing_jobs
                        WHERE media_file_id = ?
                          AND job_type = 'audio_event_detection'
                        """,
                        (media_id,),
                    ).fetchone()["id"]
                )
                save_audio_events(
                    connection,
                    job_id=job_id,
                    media_file_id=media_id,
                    events=[
                        AudioEvent(
                            label="baby_cry",
                            model_label="cry",
                            model_class_index=1,
                            media_start_seconds=5.0,
                            media_end_seconds=6.0,
                            confidence=0.9,
                        )
                    ],
                    model_name="test",
                    metadata={},
                )
                return int(
                    connection.execute(
                        "SELECT id FROM audio_events"
                    ).fetchone()["id"]
                )

            first_event_id = save()
            save_event_reviews(
                connection,
                event_ids=[first_event_id],
                verdict="confirmed",
                note="真实哭声",
            )
            second_event_id = save()
            restored = connection.execute(
                """
                SELECT verdict, note FROM event_reviews
                WHERE audio_event_id = ?
                """,
                (second_event_id,),
            ).fetchone()

            self.assertNotEqual(first_event_id, second_event_id)
            self.assertEqual(restored["verdict"], "confirmed")
            self.assertEqual(restored["note"], "真实哭声")
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM event_review_ledger"
                ).fetchone()[0],
                1,
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()
