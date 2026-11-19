from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cctv_ai.database import (
    connect_database,
    enqueue_missing_audio_jobs_for_scope,
)


class AudioJobRepairTests(unittest.TestCase):
    def test_probed_media_gets_missing_audio_job_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(Path(temporary) / "index.sqlite3")
            connection.execute(
                """
                INSERT INTO cameras(
                    id, name, location, kind, relative_root, audio_enabled
                ) VALUES ('cam', 'camera', 'room', 'event', 'cam', 1)
                """
            )
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    time_source, requires_watermark_ocr, duration_seconds,
                    audio_codec
                ) VALUES (
                    'cam', 'cam/a.mp4', 100, 1, 1, 1, 'probed', 1000,
                    'filename', 0, 60, 'aac'
                )
                """
            )
            first = enqueue_missing_audio_jobs_for_scope(
                connection,
                camera_ids=("cam",),
                since_ts=900,
                until_ts=1100,
                audio_event_labels=("baby_cry", "speech"),
            )
            second = enqueue_missing_audio_jobs_for_scope(
                connection,
                camera_ids=("cam",),
                since_ts=900,
                until_ts=1100,
                audio_event_labels=("baby_cry", "speech"),
            )
            row = connection.execute(
                """
                SELECT status, payload_json
                FROM processing_jobs
                WHERE media_file_id = ?
                  AND job_type = 'audio_event_detection'
                """,
                (int(media.lastrowid),),
            ).fetchone()

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["status"], "queued")
            self.assertEqual(
                json.loads(row["payload_json"])["labels"],
                ["baby_cry", "speech"],
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()
