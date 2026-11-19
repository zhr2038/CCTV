from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cctv_ai.activity_inference import Candidate
from cctv_ai.database import connect_database
from cctv_ai.visual_verifier import (
    Verification,
    _extract_label,
    _retry_sqlite_write,
    model_name_for_dir,
    save_verifications,
    select_temporal_event_keyframes,
    select_temporal_feeding_keyframes,
)


class VisualVerifierTests(unittest.TestCase):
    def test_sqlite_write_retries_transient_lock(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return "saved"

        with patch("cctv_ai.visual_verifier.time.sleep"):
            self.assertEqual(_retry_sqlite_write(operation), "saved")
        self.assertEqual(attempts, 3)

    def test_extract_label_rejects_ambiguous_output(self):
        self.assertEqual(
            _extract_label(
                "FEED_BOTTLE",
                {"FEED_BOTTLE", "NOT_FEEDING"},
                "UNCLEAR",
            ),
            "FEED_BOTTLE",
        )
        self.assertEqual(
            _extract_label(
                "FEED_BOTTLE or NOT_FEEDING",
                {"FEED_BOTTLE", "NOT_FEEDING"},
                "UNCLEAR",
            ),
            "UNCLEAR",
        )

    def test_model_name_tracks_checkpoint_size(self):
        self.assertEqual(
            model_name_for_dir("models/qwen2.5-vl-7b-instruct"),
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )
        self.assertEqual(
            model_name_for_dir("models/qwen2.5-vl-3b-instruct"),
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )

    def test_save_verification_clamps_confidence_and_upserts(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(
                Path(temporary) / "index.sqlite3"
            )
            try:
                connection.execute(
                    """
                    INSERT INTO cameras(
                        id, name, location, kind, relative_root, audio_enabled
                    )
                    VALUES ('camera', 'camera', 'room', 'event', 'camera', 1)
                    """
                )
                connection.execute(
                    """
                    INSERT INTO media_files(
                        id, camera_id, relative_path, size_bytes, mtime_ns,
                        first_seen_at, last_seen_at, stable_observations,
                        status, capture_start_ts, time_source,
                        requires_watermark_ocr
                    )
                    VALUES (
                        1, 'camera', 'camera/a.mp4', 1, 1, 1, 1, 2,
                        'ready', 1000, 'filename', 0
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO visual_keyframes(
                        id, media_file_id, media_offset_ms, wall_ts,
                        wall_time_source, relative_image_path, frame_sha256,
                        width, height, sample_reason, extractor_version,
                        created_at
                    )
                    VALUES (
                        1, 1, 0, 1000, 'capture_clock', 'camera/frame.jpg',
                        'hash', 640, 360, 'single', 'test', 1000
                    )
                    """
                )
                save_verifications(
                    connection,
                    task="feeding",
                    prompt_version="prompt-v1",
                    model_name="model",
                    keyframe_ids=(1,),
                    results=(Verification("UNCLEAR", 1.4, "first"),),
                )
                save_verifications(
                    connection,
                    task="feeding",
                    prompt_version="prompt-v1",
                    model_name="model",
                    keyframe_ids=(1,),
                    results=(Verification("NOT_FEEDING", -0.2, "second"),),
                )
                row = connection.execute(
                    """
                    SELECT label, confidence, details_json
                    FROM visual_frame_verifications
                    """
                ).fetchone()
                self.assertEqual(row["label"], "NOT_FEEDING")
                self.assertEqual(row["confidence"], 0.0)
                self.assertEqual(
                    json.loads(row["details_json"])["raw_output"],
                    "second",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM visual_frame_verifications"
                    ).fetchone()[0],
                    1,
                )
            finally:
                connection.close()

    def test_temporal_recall_keeps_broad_peak_without_yolo_gate(self):
        def frame(
            keyframe_id: int,
            wall_ts: float,
            raw: float,
            score: float,
        ):
            return SimpleNamespace(
                keyframe_id=keyframe_id,
                camera_id="bed",
                wall_ts=wall_ts,
                person_confidence=0.0,
                scores={
                    "feeding": {
                        "feeding_bottle": score,
                        "not_feeding": 1.0 - score,
                    }
                },
                raw_sigmoids={
                    "feeding": {
                        "feeding_bottle": raw,
                        "not_feeding": 0.0,
                    }
                },
                verified_feeding_label=None,
            )

        selected = select_temporal_feeding_keyframes(
            (
                frame(1, 1000, 0.10, 0.70),
                frame(2, 1010, 0.20, 0.80),
                frame(3, 1400, 0.01, 0.20),
            ),
            minimum_rank=0.45,
        )
        self.assertEqual(selected, [2, 1])

        single_peak = select_temporal_feeding_keyframes(
            (
                frame(1, 1000, 0.10, 0.70),
                frame(2, 1010, 0.20, 0.80),
            ),
            minimum_rank=0.45,
            max_per_bucket=1,
        )
        self.assertEqual(single_peak, [2])

    def test_temporal_recall_uses_five_minute_buckets(self):
        def frame(keyframe_id: int, wall_ts: float):
            return SimpleNamespace(
                keyframe_id=keyframe_id,
                camera_id="bed",
                wall_ts=wall_ts,
                person_confidence=0.5,
                scores={
                    "feeding": {
                        "feeding_bottle": 0.8,
                        "not_feeding": 0.2,
                    }
                },
                raw_sigmoids={
                    "feeding": {
                        "feeding_bottle": 0.2,
                        "not_feeding": 0.01,
                    }
                },
                verified_feeding_label=None,
            )

        selected = select_temporal_feeding_keyframes(
            (frame(1, 850), frame(2, 950))
        )
        self.assertEqual(set(selected), {1, 2})

    def test_sleep_verification_samples_each_ten_minute_bucket(self):
        frames = [
            SimpleNamespace(
                candidate=Candidate(
                    "inferred_sleep",
                    "low_motion_in_bed",
                    float(index * 30),
                    float(index * 30 + 30),
                    "bed",
                    "宝宝床",
                    "baby_bed",
                    0.8,
                    index + 1,
                    "exact",
                )
            )
            for index in range(25)
        ]
        with patch(
            "cctv_ai.activity_inference.frame_candidates",
            side_effect=lambda frame, **_kwargs: [frame.candidate],
        ):
            selected = select_temporal_event_keyframes(
                frames,
                activity_type="inferred_sleep",
            )
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
