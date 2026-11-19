from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cctv_ai.database import connect_database
from cctv_ai.visual_search import (
    activity_type_for_query,
    search_activity_event_evidence,
)


class StructuredVisualSearchTests(unittest.TestCase):
    def test_feeding_query_returns_existing_event_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(Path(temporary) / "index.sqlite3")
            connection.execute(
                """
                INSERT INTO cameras(
                    id, name, location, kind, relative_root, audio_enabled
                ) VALUES ('cam', '宝宝房', 'baby_room', 'event', 'cam', 1)
                """
            )
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    time_source, requires_watermark_ocr, duration_seconds
                ) VALUES (
                    'cam', 'cam/feed.mp4', 100, 1, 1, 1, 'probed',
                    1000, 'filename', 0, 60
                )
                """
            )
            keyframe = connection.execute(
                """
                INSERT INTO visual_keyframes(
                    media_file_id, media_offset_ms, wall_ts,
                    wall_time_source, relative_image_path, frame_sha256,
                    width, height, sample_reason, extractor_version,
                    created_at
                ) VALUES (?, 5000, 1005, 'capture_clock', 'cam/feed.jpg',
                          'abc', 640, 360, 'event', 'test', 1)
                """,
                (int(media.lastrowid),),
            )
            event = connection.execute(
                """
                INSERT INTO activity_events(
                    event_key, activity_type, activity_subtype,
                    start_ts, end_ts, duration_seconds, camera_ids_json,
                    location, zone, confidence, time_precision, method,
                    model_version, review_status, created_at, updated_at
                ) VALUES (
                    'feeding-event', 'feeding', 'feeding_bottle',
                    1000, 1010, 10, '["cam"]', '宝宝房', 'baby_bed',
                    0.95, 'exact', 'qwen3_vl_temporal', 'test',
                    'machine', 1, 1
                )
                """
            )
            connection.execute(
                """
                INSERT INTO activity_event_evidence(
                    activity_event_id, keyframe_id, evidence_rank, score
                ) VALUES (?, ?, 1, 0.95)
                """,
                (int(event.lastrowid), int(keyframe.lastrowid)),
            )
            connection.commit()

            self.assertEqual(activity_type_for_query("宝宝喝奶"), "feeding")
            results = search_activity_event_evidence(
                connection,
                query="宝宝喝奶",
                since_ts=900,
                until_ts=1100,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["activity_type"], "feeding")
            self.assertEqual(results[0]["match_source"], "structured_event")
            self.assertEqual(
                results[0]["verification_status"],
                "machine_event",
            )
            connection.close()

    def test_unstructured_query_does_not_invent_an_event_type(self) -> None:
        self.assertIsNone(activity_type_for_query("穿红衣服的场景"))


if __name__ == "__main__":
    unittest.main()
