from __future__ import annotations

import sqlite3
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from cctv_ai.activity_inference import (
    Candidate,
    FrameObservation,
    frame_candidates,
    infer_sleep_state_candidates,
    merge_candidates,
    save_activity_events,
)
from cctv_ai.database import connect_database
from cctv_ai.visual_database import (
    query_visual_progress,
    save_extracted_keyframes,
)
from cctv_ai.visual_extract import (
    EXTRACTOR_VERSION,
    ExtractedKeyframe,
    sampling_plan,
)
from cctv_ai.visual_pipeline import (
    _profile_probe,
    _retry_sqlite_write,
    _visual_work_remaining,
)
from cctv_ai.visual_models import (
    SLEEP_PRESENCE_PROMPT_VERSION,
    TEMPORAL_FEEDING_PROMPT_VERSION,
    TEMPORAL_PLAY_PROMPT_VERSION,
    TEMPORAL_POSTURE_PROMPT_VERSION,
)
from cctv_ai.visual_watermark import plausible_watermark_time


class VisualPipelineTests(unittest.TestCase):
    def test_activity_event_id_survives_machine_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(
                Path(temporary) / "index.sqlite3"
            )
            event = {
                "activity_type": "feeding",
                "subtype": "feeding_bottle",
                "start_ts": 1000.0,
                "end_ts": 1060.0,
                "duration_seconds": 60.0,
                "session_span_seconds": 60.0,
                "camera_ids": ["bed"],
                "location": "宝宝床",
                "zone": "baby_bed",
                "confidence": 0.9,
                "time_precision": "exact",
                "evidence": [],
                "support_frames": 2,
            }
            try:
                save_activity_events(
                    connection,
                    events=[event],
                    since_ts=0,
                    until_ts=2000,
                    generated_by_run_id=None,
                    model_version="test-v1",
                )
                first_id = int(
                    connection.execute(
                        "SELECT id FROM activity_events"
                    ).fetchone()["id"]
                )
                connection.execute(
                    """
                    UPDATE activity_events
                    SET details_json = json_set(
                        details_json,
                        '$.growth_milestone',
                        json('{"kind":"first_feed"}')
                    )
                    WHERE id = ?
                    """,
                    (first_id,),
                )
                save_activity_events(
                    connection,
                    events=[event],
                    since_ts=0,
                    until_ts=2000,
                    generated_by_run_id=None,
                    model_version="test-v1",
                )
                second_id = int(
                    connection.execute(
                        "SELECT id FROM activity_events"
                    ).fetchone()["id"]
                )
                self.assertEqual(second_id, first_id)
                details = json.loads(
                    connection.execute(
                        """
                        SELECT details_json
                        FROM activity_events
                        WHERE id = ?
                        """,
                        (first_id,),
                    ).fetchone()["details_json"]
                )
                self.assertEqual(
                    details["growth_milestone"]["kind"],
                    "first_feed",
                )
            finally:
                connection.close()

    def test_reviewed_event_keeps_id_when_machine_bounds_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(
                Path(temporary) / "index.sqlite3"
            )
            event = {
                "activity_type": "feeding",
                "subtype": "feeding_bottle",
                "start_ts": 1000.0,
                "end_ts": 1240.0,
                "duration_seconds": 240.0,
                "session_span_seconds": 240.0,
                "camera_ids": ["bed"],
                "location": "宝宝床",
                "zone": "baby_bed",
                "confidence": 0.95,
                "time_precision": "exact",
                "evidence": [],
                "support_frames": 1,
            }
            try:
                save_activity_events(
                    connection,
                    events=[event],
                    since_ts=0,
                    until_ts=2000,
                    generated_by_run_id=None,
                    model_version="test-v1",
                )
                row = connection.execute(
                    "SELECT id FROM activity_events"
                ).fetchone()
                event_id = int(row["id"])
                connection.execute(
                    "UPDATE activity_events SET review_status='confirmed' WHERE id=?",
                    (event_id,),
                )
                corrected_measurement = {
                    **event,
                    "start_ts": 1120.0,
                    "end_ts": 1121.0,
                    "duration_seconds": 0.0,
                    "session_span_seconds": 1.0,
                    "measurement_mode": "sighting_only",
                    "certainty": "confirmed",
                    "duration_publishable": False,
                }
                save_activity_events(
                    connection,
                    events=[corrected_measurement],
                    since_ts=0,
                    until_ts=2000,
                    generated_by_run_id=None,
                    model_version="test-v1",
                )
                rows = list(
                    connection.execute(
                        "SELECT id, start_ts, end_ts, review_status FROM activity_events"
                    )
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(int(rows[0]["id"]), event_id)
                self.assertEqual(float(rows[0]["start_ts"]), 1120.0)
                self.assertEqual(float(rows[0]["end_ts"]), 1121.0)
                self.assertEqual(rows[0]["review_status"], "confirmed")
            finally:
                connection.close()

    def test_visual_watermark_rejects_impossible_ocr_year(self):
        timezone = ZoneInfo("Asia/Shanghai")
        expected = datetime(2026, 7, 11, 12, tzinfo=timezone)
        self.assertTrue(
            plausible_watermark_time(
                expected,
                expected_start_ts=expected.timestamp() - 60,
                expected_end_ts=expected.timestamp() + 60,
            )
        )
        self.assertFalse(
            plausible_watermark_time(
                expected.replace(year=2926),
                expected_start_ts=expected.timestamp() - 60,
                expected_end_ts=expected.timestamp() + 60,
            )
        )

    def test_sqlite_write_retries_transient_lock(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise sqlite3.OperationalError("database is locked")
            return "saved"

        with patch("cctv_ai.visual_pipeline.time.sleep"):
            self.assertEqual(_retry_sqlite_write(operation), "saved")
        self.assertEqual(attempts, 3)

    def test_visual_worker_stops_when_only_terminal_errors_remain(self):
        self.assertFalse(
            _visual_work_remaining(
                {
                    "status": "partial",
                    "scope_progress": {
                        "files_pending_schedule": 0,
                        "jobs": {"watermark_time_map:error": 4},
                    },
                }
            )
        )
        self.assertTrue(
            _visual_work_remaining(
                {
                    "status": "partial",
                    "scope_progress": {
                        "files_pending_schedule": 0,
                        "jobs": {"visual_embedding:queued": 1},
                    },
                }
            )
        )
        self.assertTrue(
            _visual_work_remaining(
                {"status": "failed", "error": "temporary"}
            )
        )

    def test_sampling_plan_respects_camera_recording_style(self):
        self.assertEqual(
            sampling_plan(
                {
                    "camera_id": "B88880620430",
                    "duration_seconds": 1800,
                }
            ),
            ("interval", 30.0),
        )
        self.assertEqual(
            sampling_plan(
                {
                    "camera_id": "50ec503b63bf",
                    "duration_seconds": 60,
                }
            ),
            ("interval", 20.0),
        )
        mode, offset = sampling_plan(
            {
                "camera_id": "lumi3.8e2ce0d05d9c251e",
                "duration_seconds": 12,
            }
        )
        self.assertEqual(mode, "single")
        self.assertEqual(offset, 3.0)

    def test_keyframe_keeps_source_offset_and_estimated_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "index.sqlite3"
            output = root / "keyframes"
            image = output / "camera" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg")
            connection = connect_database(database)
            try:
                connection.execute(
                    """
                    INSERT INTO cameras(
                        id, name, location, kind, relative_root, audio_enabled
                    )
                    VALUES ('bed', 'bed', 'baby_bedside', 'bundle', 'bed', 1)
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
                        1, 'bed', 'bed/a.mp4', 1, 1, 1, 1, 2,
                        'probed', 1000, 'filename_hint', 1
                    )
                    """
                )
                cursor = connection.execute(
                    """
                    INSERT INTO processing_jobs(
                        media_file_id, job_type, status, priority,
                        created_at, updated_at
                    )
                    VALUES (1, 'keyframe_extract', 'running', 20, 1, 1)
                    """
                )
                media = connection.execute(
                    "SELECT * FROM media_files WHERE id = 1"
                ).fetchone()
                ids = save_extracted_keyframes(
                    connection,
                    job_id=int(cursor.lastrowid),
                    media=media,
                    keyframes=[
                        ExtractedKeyframe(
                            media_offset_seconds=30,
                            image_path=image,
                            frame_sha256="abc",
                            width=640,
                            height=360,
                            brightness=0.4,
                            sharpness=50,
                            motion_score=None,
                            night_score=0.8,
                            sample_reason="continuous_30s",
                        )
                    ],
                    output_root=output,
                    extractor_version=EXTRACTOR_VERSION,
                    metadata={"frames": 1},
                )
                row = connection.execute(
                    "SELECT * FROM visual_keyframes WHERE id = ?",
                    (ids[0],),
                ).fetchone()
                self.assertEqual(row["media_offset_ms"], 30_000)
                self.assertEqual(row["wall_ts"], 1030)
                self.assertEqual(row["wall_time_source"], "estimated")
                self.assertEqual(
                    row["relative_image_path"],
                    "camera/frame.jpg",
                )
            finally:
                connection.close()

    def test_sleep_requires_sustained_evidence(self):
        short = [
            Candidate(
                "inferred_sleep",
                "low_motion_in_bed",
                1000 + index * 30,
                1030 + index * 30,
                "bed",
                "宝宝床",
                "baby_bed",
                0.8,
                index + 1,
                "mapped",
            )
            for index in range(3)
        ]
        self.assertEqual(merge_candidates(short), [])
        sustained = [
            Candidate(
                "inferred_sleep",
                "low_motion_in_bed",
                1000 + index * 30,
                1030 + index * 30,
                "bed",
                "宝宝床",
                "baby_bed",
                0.8,
                index + 1,
                "mapped",
            )
            for index in range(12)
        ]
        events = merge_candidates(sustained)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["activity_type"], "inferred_sleep")
        self.assertGreaterEqual(events[0]["duration_seconds"], 300)

    def test_activity_duration_excludes_unobserved_gap(self):
        candidates = [
            Candidate(
                "play",
                "playing",
                1000,
                1030,
                "camera",
                "客厅",
                "living_room",
                0.8,
                1,
                "exact",
            ),
            Candidate(
                "play",
                "playing",
                1100,
                1130,
                "camera",
                "客厅",
                "living_room",
                0.8,
                2,
                "exact",
            ),
        ]
        events = merge_candidates(candidates)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["end_ts"] - events[0]["start_ts"], 130)
        self.assertEqual(events[0]["duration_seconds"], 60)

    def test_merged_toy_confidence_is_bounded_average(self):
        candidates = [
            Candidate(
                "play",
                "playing",
                1000 + index * 30,
                1030 + index * 30,
                "camera",
                "宝宝床",
                "baby_bed",
                0.8,
                index + 1,
                "exact",
                toy_label="book",
                toy_score=score,
            )
            for index, score in enumerate((0.6, 0.8, 0.7))
        ]
        event = merge_candidates(candidates)[0]
        self.assertAlmostEqual(event["toy_scores"]["book"], 2.1)
        self.assertAlmostEqual(event["toy_confidence"], 0.7)
        self.assertLessEqual(event["toy_confidence"], 1.0)

    def test_single_frame_toy_is_not_published_as_favorite(self):
        candidate = Candidate(
            "play",
            "playing",
            1000,
            1030,
            "camera",
            "宝宝床",
            "baby_bed",
            0.8,
            1,
            "exact",
            toy_label="ball",
            toy_score=0.82,
        )
        event = merge_candidates([candidate])[0]
        self.assertIsNone(event["toy_label"])
        self.assertIsNone(event["toy_confidence"])

    def test_feeding_subtype_flips_merge_into_one_session(self):
        candidates = [
            Candidate(
                "feeding",
                "feeding_bottle",
                1000,
                1030,
                "camera",
                "宝宝床",
                "baby_bed",
                0.9,
                1,
                "exact",
            ),
            Candidate(
                "feeding",
                "breastfeeding",
                1040,
                1070,
                "camera",
                "宝宝床",
                "baby_bed",
                0.6,
                2,
                "exact",
            ),
        ]
        events = merge_candidates(candidates)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["subtype"], "feeding_bottle")
        self.assertEqual(events[0]["duration_seconds"], 60)

    def test_weak_single_frame_feeding_needs_direct_support(self):
        weak = Candidate(
            "feeding",
            "feeding_bottle",
            1000,
            1030,
            "camera",
            "宝宝床",
            "baby_bed",
            0.6,
            1,
            "exact",
        )
        self.assertEqual(merge_candidates([weak]), [])
        supported = Candidate(
            "feeding",
            "feeding_bottle",
            1000,
            1030,
            "camera",
            "宝宝床",
            "baby_bed",
            0.6,
            1,
            "exact",
            direct_support=True,
        )
        self.assertEqual(len(merge_candidates([supported])), 1)

    def test_activity_events_do_not_merge_across_local_midnight(self):
        timezone = ZoneInfo("Asia/Shanghai")
        before = datetime(
            2026, 7, 1, 23, 59, 50, tzinfo=timezone
        ).timestamp()
        after = datetime(
            2026, 7, 2, 0, 0, 20, tzinfo=timezone
        ).timestamp()
        candidates = [
            Candidate(
                "sitting",
                None,
                before - 10,
                before + 10,
                "camera",
                "客厅",
                "living_room",
                0.8,
                1,
                "exact",
            ),
            Candidate(
                "sitting",
                None,
                after - 10,
                after + 10,
                "camera",
                "客厅",
                "living_room",
                0.8,
                2,
                "exact",
            ),
        ]
        self.assertEqual(
            len(merge_candidates(candidates, timezone=timezone)),
            2,
        )

    def test_sleeping_scene_is_not_counted_as_feeding(self):
        frame = FrameObservation(
            keyframe_id=1,
            media_file_id=1,
            camera_id="B88880620430",
            wall_ts=1000,
            wall_time_source="watermark",
            location="宝宝床",
            zone="baby_bed",
            sample_reason="continuous_30s",
            media_duration=600,
            motion_score=0.01,
            night_score=0.8,
            person_confidence=0.8,
            scores={
                "feeding": {
                    "breastfeeding": 0.95,
                    "not_feeding": 0.02,
                },
                "posture": {
                    "sleeping": 0.95,
                    "posture_unknown": 0.03,
                },
                "care": {"care_other": 0.9},
                "play": {"not_playing": 0.9},
                "toy": {"no_toy": 0.9},
            },
            raw_sigmoids={
                "feeding": {
                    "breastfeeding": 0.9,
                    "not_feeding": 0.01,
                },
                "posture": {"sleeping": 0.9},
                "care": {"care_other": 0.2},
                "play": {"not_playing": 0.2},
                "toy": {"no_toy": 0.2},
            },
        )
        self.assertFalse(
            any(
                item.activity_type == "feeding"
                for item in frame_candidates(frame)
            )
        )
    def test_legacy_single_frame_feeding_is_recall_only(self):
        frame = FrameObservation(
            keyframe_id=1,
            media_file_id=1,
            camera_id="B88880620430",
            wall_ts=1000,
            wall_time_source="capture_clock",
            location="宝宝床",
            zone="baby_bed",
            sample_reason="continuous_30s",
            media_duration=600,
            motion_score=0.08,
            night_score=0.0,
            person_confidence=0.8,
            scores={
                "feeding": {
                    "feeding_bottle": 0.8,
                    "not_feeding": 0.1,
                },
                "posture": {"posture_unknown": 0.9},
                "care": {"care_other": 0.9},
                "play": {"not_playing": 0.9},
                "toy": {"no_toy": 0.9},
            },
            raw_sigmoids={
                "feeding": {
                    "feeding_bottle": 0.17,
                    "not_feeding": 0.02,
                },
                "posture": {"posture_unknown": 0.2},
                "care": {"care_other": 0.2},
                "play": {"not_playing": 0.2},
                "toy": {"no_toy": 0.2},
            },
            verified_feeding_label="FEED_BOTTLE",
            verified_feeding_confidence=0.82,
        )
        self.assertTrue(
            any(
                item.activity_type == "feeding"
                for item in frame_candidates(
                    frame,
                    use_verification=False,
                )
            )
        )
        self.assertFalse(
            any(
                item.activity_type == "feeding"
                for item in frame_candidates(frame)
            )
        )
        detector_supported = replace(
            frame,
            feeding_container_confidence=0.4,
        )
        self.assertFalse(
            any(
                item.activity_type == "feeding"
                for item in frame_candidates(detector_supported)
            )
        )

    def test_temporal_qwen_feeding_is_direct_event_evidence(self):
        frame = FrameObservation(
            keyframe_id=1,
            media_file_id=1,
            camera_id="B88880620430",
            wall_ts=1000,
            wall_time_source="capture_clock",
            location="宝宝床",
            zone="baby_bed",
            sample_reason="continuous_30s",
            media_duration=600,
            motion_score=0.01,
            night_score=0.8,
            person_confidence=0.0,
            scores={
                "feeding": {
                    "feeding_bottle": 0.1,
                    "not_feeding": 0.9,
                },
                "posture": {"sleeping": 0.9},
            },
            raw_sigmoids={
                "feeding": {
                    "feeding_bottle": 0.01,
                    "not_feeding": 0.8,
                },
                "posture": {"sleeping": 0.8},
            },
            verified_feeding_label="FEED_BOTTLE",
            verified_feeding_confidence=0.95,
            verified_feeding_model_name=(
                "Qwen/Qwen3-VL-8B-Instruct"
            ),
            verified_feeding_prompt_version=(
                TEMPORAL_FEEDING_PROMPT_VERSION
            ),
        )
        feeding = [
            item
            for item in frame_candidates(frame)
            if item.activity_type == "feeding"
        ]
        self.assertEqual(len(feeding), 1)
        self.assertTrue(feeding[0].direct_support)
        self.assertTrue(feeding[0].temporal_verified)
        events = merge_candidates(feeding)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["method"], "qwen3_vl_temporal")
        self.assertEqual(events[0]["duration_seconds"], 0.0)
        self.assertEqual(events[0]["measurement_mode"], "sighting_only")

        rejected = replace(
            frame,
            verified_feeding_label="UNCLEAR",
            verified_feeding_confidence=0.45,
        )
        self.assertFalse(
            any(
                item.activity_type == "feeding"
                for item in frame_candidates(rejected)
            )
        )

        probable = replace(
            rejected,
            verified_feeding_details={
                "native_source": True,
                "structured_event": {
                    "activity": "feeding",
                    "objects": ["bottle"],
                    "evidence": [
                        "18:16:19 奶嘴在宝宝嘴里",
                        "18:16:20 奶嘴在宝宝嘴里",
                    ],
                },
            },
        )
        probable_candidates = [
            item
            for item in frame_candidates(probable)
            if item.activity_type == "feeding"
        ]
        self.assertEqual(len(probable_candidates), 1)
        self.assertEqual(probable_candidates[0].certainty, "probable")
        probable_events = merge_candidates(probable_candidates)
        self.assertEqual(probable_events[0]["certainty"], "probable")

    def test_toy_can_be_ranked_below_no_toy_during_clear_play(self):
        frame = FrameObservation(
            keyframe_id=1,
            media_file_id=1,
            camera_id="bed",
            wall_ts=1000,
            wall_time_source="capture_clock",
            location="宝宝房间",
            zone="baby_room",
            sample_reason="event_representative",
            media_duration=20,
            motion_score=None,
            night_score=0,
            person_confidence=0.8,
            scores={
                "feeding": {"not_feeding": 0.9},
                "posture": {"posture_unknown": 0.9},
                "care": {"care_other": 0.9},
                "play": {"playing": 0.8, "not_playing": 0.1},
                "toy": {"no_toy": 0.8, "book": 0.1},
            },
            raw_sigmoids={
                "feeding": {"not_feeding": 0.2},
                "posture": {"posture_unknown": 0.2},
                "care": {"care_other": 0.2},
                "play": {"playing": 0.3, "not_playing": 0.01},
                "toy": {"no_toy": 0.5, "book": 0.1},
            },
        )
        play = next(
            item
            for item in frame_candidates(frame)
            if item.activity_type == "play"
        )
        self.assertEqual(play.toy_label, "book")

        rejected = replace(
            frame,
            verified_toy_label="NONE",
            verified_toy_confidence=0.95,
            verified_toy_model_name="Qwen/Qwen3-VL-8B-Instruct",
            verified_toy_prompt_version=TEMPORAL_PLAY_PROMPT_VERSION,
        )
        rejected_candidates = [
            item
            for item in frame_candidates(rejected)
            if item.activity_type == "play"
        ]
        self.assertEqual(len(rejected_candidates), 1)
        self.assertEqual(merge_candidates(rejected_candidates), [])

        verified = replace(
            rejected,
            verified_toy_label="BOOK",
        )
        verified_candidates = [
            item
            for item in frame_candidates(verified)
            if item.activity_type == "play"
        ]
        events = merge_candidates(verified_candidates)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["toy_label"], "book")
        self.assertEqual(events[0]["method"], "qwen3_vl_temporal")

    def test_sleep_event_level_verdict_rejects_empty_bed(self):
        candidates = [
            Candidate(
                "inferred_sleep",
                "low_motion_in_bed",
                float(index * 30),
                float(index * 30 + 30),
                "B88880620430",
                "宝宝床",
                "baby_bed",
                0.9,
                index + 1,
                "exact",
            )
            for index in range(11)
        ]
        candidates[5] = replace(
            candidates[5],
            verification_available=True,
        )
        self.assertEqual(merge_candidates(candidates), [])

        candidates[5] = replace(
            candidates[5],
            verified_support=True,
            temporal_verified=True,
        )
        events = merge_candidates(candidates)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["activity_type"], "inferred_sleep")
        self.assertEqual(events[0]["method"], "qwen3_vl_temporal")

    def test_sleep_anchor_only_claims_native_two_minute_window(self):
        frame = FrameObservation(
            keyframe_id=7,
            media_file_id=1,
            camera_id="B88880620430",
            wall_ts=1000,
            wall_time_source="capture_clock",
            location="宝宝床",
            zone="baby_bed",
            sample_reason="continuous_30s",
            media_duration=600,
            motion_score=0.01,
            night_score=0.5,
            person_confidence=0.0,
            scores={
                "feeding": {"not_feeding": 0.9},
                "posture": {
                    "sleeping": 0.9,
                    "posture_unknown": 0.1,
                },
            },
            raw_sigmoids={
                "feeding": {"not_feeding": 0.2},
                "posture": {"sleeping": 0.8},
            },
        )
        self.assertTrue(
            any(
                item.activity_type == "inferred_sleep"
                for item in frame_candidates(
                    frame,
                    use_verification=False,
                )
            )
        )
        self.assertFalse(
            any(
                item.activity_type == "inferred_sleep"
                for item in frame_candidates(frame)
            )
        )

        verified = replace(
            frame,
            verified_sleep_label="BABY_SLEEPING",
            verified_sleep_confidence=0.95,
            verified_sleep_model_name="Qwen/Qwen3-VL-8B-Instruct",
            verified_sleep_prompt_version=SLEEP_PRESENCE_PROMPT_VERSION,
        )
        candidates = [
            item
            for item in frame_candidates(verified)
            if item.activity_type == "inferred_sleep"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].start_ts, 940)
        self.assertEqual(candidates[0].end_ts, 1060)
        events = merge_candidates(candidates)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["duration_seconds"], 120)
        self.assertEqual(events[0]["measurement_mode"], "native_state_anchor")

    def test_sleep_state_timeline_is_cut_by_awake_and_unclear_points(self):
        base = FrameObservation(
            keyframe_id=1,
            media_file_id=1,
            camera_id="B88880620430",
            wall_ts=0,
            wall_time_source="capture_clock",
            location="宝宝床",
            zone="baby_bed",
            sample_reason="continuous_30s",
            media_duration=3600,
            motion_score=0.01,
            night_score=0.5,
            person_confidence=0.0,
            scores={},
            raw_sigmoids={},
            verified_sleep_label="NOT_SLEEPING",
            verified_sleep_confidence=0.0,
            verified_sleep_model_name="Qwen/Qwen3-VL-8B-Instruct",
            verified_sleep_prompt_version=SLEEP_PRESENCE_PROMPT_VERSION,
        )
        frames = [
            base,
            replace(
                base,
                keyframe_id=2,
                wall_ts=600,
                verified_sleep_label="BABY_SLEEPING",
                verified_sleep_confidence=0.95,
            ),
            replace(
                base,
                keyframe_id=3,
                wall_ts=1200,
                verified_sleep_label="BABY_SLEEPING",
                verified_sleep_confidence=0.95,
            ),
            replace(
                base,
                keyframe_id=4,
                wall_ts=1800,
                verified_sleep_label="UNCLEAR",
            ),
            replace(
                base,
                keyframe_id=5,
                wall_ts=2400,
                verified_sleep_label="BABY_SLEEPING",
                verified_sleep_confidence=0.95,
            ),
        ]
        events = merge_candidates(infer_sleep_state_candidates(frames))
        self.assertEqual(len(events), 2)
        self.assertEqual((events[0]["start_ts"], events[0]["end_ts"]), (300, 1500))
        self.assertEqual((events[1]["start_ts"], events[1]["end_ts"]), (2100, 2460))
        self.assertEqual(events[0]["start_boundary_error_seconds"], 300)
        self.assertEqual(events[0]["end_boundary_error_seconds"], 300)

    def test_posture_event_level_verdict_controls_sitting(self):
        frame = FrameObservation(
            keyframe_id=7,
            media_file_id=1,
            camera_id="B88880620430",
            wall_ts=1000,
            wall_time_source="capture_clock",
            location="宝宝床",
            zone="baby_bed",
            sample_reason="continuous_30s",
            media_duration=600,
            motion_score=0.08,
            night_score=0.0,
            person_confidence=0.9,
            scores={
                "feeding": {"not_feeding": 0.9},
                "posture": {
                    "sitting": 0.85,
                    "posture_unknown": 0.1,
                },
            },
            raw_sigmoids={
                "feeding": {"not_feeding": 0.2},
                "posture": {"sitting": 0.4},
            },
            verified_posture_label="OTHER",
            verified_posture_confidence=0.95,
            verified_posture_model_name=(
                "Qwen/Qwen3-VL-8B-Instruct"
            ),
            verified_posture_prompt_version=(
                TEMPORAL_POSTURE_PROMPT_VERSION
            ),
        )
        rejected = [
            item
            for item in frame_candidates(frame)
            if item.activity_type == "sitting"
        ]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(merge_candidates(rejected), [])

        accepted = replace(
            frame,
            verified_posture_label="SITTING",
        )
        events = merge_candidates(
            [
                item
                for item in frame_candidates(accepted)
                if item.activity_type == "sitting"
            ]
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["method"], "qwen3_vl_temporal")

    def test_visual_progress_is_empty_on_fresh_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(
                Path(temporary) / "index.sqlite3"
            )
            try:
                progress = query_visual_progress(connection)
                self.assertEqual(progress["keyframes"], 0)
                self.assertEqual(progress["activity_events"], 0)
                self.assertEqual(progress["progress_percent"], 0.0)
            finally:
                connection.close()

    def test_visual_progress_uses_all_source_files_as_denominator(self):
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
                progress = query_visual_progress(
                    connection,
                    since_ts=900,
                    until_ts=1100,
                    camera_ids=("camera",),
                )
                self.assertEqual(progress["files_total"], 1)
                self.assertEqual(progress["files_stable"], 1)
                self.assertEqual(progress["files_unstable"], 0)
                self.assertEqual(progress["files_source_error"], 0)
                self.assertEqual(progress["files_scheduled"], 0)
                self.assertEqual(progress["files_pending_schedule"], 1)
                self.assertEqual(progress["progress_percent"], 0.0)
            finally:
                connection.close()

    def test_visual_progress_reports_unstable_and_broken_sources_separately(
        self,
    ):
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
                for media_id, status in (
                    (1, "ready"),
                    (2, "waiting"),
                    (3, "error"),
                ):
                    connection.execute(
                        """
                        INSERT INTO media_files(
                            id, camera_id, relative_path, size_bytes, mtime_ns,
                            first_seen_at, last_seen_at, stable_observations,
                            status, capture_start_ts, time_source,
                            requires_watermark_ocr
                        )
                        VALUES (
                            ?, 'camera', ?, 1, 1, 1, 1, 1,
                            ?, ?, 'filename', 0
                        )
                        """,
                        (
                            media_id,
                            f"camera/{media_id}.mp4",
                            status,
                            1000 + media_id,
                        ),
                    )
                progress = query_visual_progress(
                    connection,
                    since_ts=900,
                    until_ts=1100,
                    camera_ids=("camera",),
                )
                self.assertEqual(progress["files_total"], 3)
                self.assertEqual(progress["files_stable"], 1)
                self.assertEqual(progress["files_unstable"], 1)
                self.assertEqual(progress["files_source_error"], 1)
                self.assertEqual(progress["files_pending_schedule"], 1)
            finally:
                connection.close()

    def test_bedside_profile_probe_uses_filename_envelope(self):
        probe = _profile_probe(
            {
                "capture_start_ts": 1000,
                "capture_end_hint_ts": 1600,
                "size_bytes": 120_000_000,
            },
            {
                "video_codec": "hevc",
                "width": 2880,
                "height": 1620,
                "fps": 15.0,
                "audio_codec": "opus",
                "audio_sample_rate": 48_000,
                "audio_channels": 1,
            },
        )
        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(probe.duration_seconds, 600)
        self.assertEqual(probe.video_codec, "hevc")
        self.assertEqual(probe.width, 2880)
        self.assertEqual(probe.audio_codec, "opus")


if __name__ == "__main__":
    unittest.main()
