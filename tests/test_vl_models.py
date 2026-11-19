from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cctv_ai.database import connect_database
from cctv_ai.model_config import (
    VisionLanguageConfig,
    load_models_config,
)
from cctv_ai.visual_search import VisualSearchService
from cctv_ai.visual_verifier import (
    _feeding_label,
    _toy_label,
    model_name_for_dir,
)
from cctv_ai.vl_models.base import StructuredVisualEvent
from cctv_ai.vl_models.qwen3_vl import _json_object
from cctv_ai.vl_models.tiered import (
    TieredAnalysis,
    TieredVisionAnalyzer,
    save_confirmed_milestone,
)


def event(
    activity: str,
    confidence: float = 0.9,
    *,
    uncertain: bool = False,
    objects: tuple[str, ...] = (),
) -> StructuredVisualEvent:
    description = (
        "奶瓶奶嘴接触宝宝嘴部"
        if activity == "feeding"
        else "直接可见的动作"
    )
    return StructuredVisualEvent(
        activity=activity,
        description=description,
        objects=objects,
        confidence=confidence,
        evidence=("画面证据",),
        uncertain=uncertain,
    )


class FakeVisionModel:
    def __init__(self, result: StructuredVisualEvent, name: str) -> None:
        self.result = result
        self.model_name = name
        self.closed = False

    def analyze_image(self, image, prompt, context=None):
        return self.result

    def analyze_video_frames(self, frames, prompt, context=None):
        return self.result

    def generate_summary(self, events):
        return {}

    def close(self):
        self.closed = True


class VisionModelArchitectureTests(unittest.TestCase):
    def test_default_models_config_is_local_and_tiered(self):
        config = load_models_config("config/models.toml")
        self.assertEqual(config.vision_language.provider, "qwen3-vl")
        self.assertEqual(
            config.vision_language.model_id,
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        self.assertEqual(config.vision_language.quantization, "4bit")
        self.assertEqual(
            config.high_value.model_id,
            "Qwen/Qwen3-VL-32B-Instruct",
        )
        self.assertEqual(
            config.visual_embedding.model_id,
            "Qwen/Qwen3-VL-Embedding-8B",
        )
        self.assertTrue(config.vision_language.local_files_only)

    def test_config_refuses_runtime_network_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "config"
            config_dir.mkdir()
            path = config_dir / "models.toml"
            path.write_text(
                """
[vision_language]
provider = "qwen3-vl"
model = "Qwen3-VL-8B-Instruct"
model_path = ".data/model"
local_files_only = false

[visual_embedding]
provider = "qwen3-vl-embedding"
model = "Qwen3-VL-Embedding-8B"
model_path = ".data/embed"
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "runtime model downloads are prohibited",
            ):
                load_models_config(path)

    def test_fixed_json_parser_and_safety_filter(self):
        parsed = _json_object(
            'before {"activity":"standing","description":"宝宝扶着床栏",'
            '"objects":["baby"],"confidence":0.88,'
            '"evidence":["手扶床栏"],"uncertain":false} after'
        )
        result = StructuredVisualEvent.from_mapping(parsed)
        self.assertEqual(result.activity, "standing")
        self.assertFalse(result.uncertain)
        unsafe = StructuredVisualEvent.from_mapping(
            {
                "activity": "standing",
                "description": "宝宝运动能力优秀，性格外向",
                "objects": ["baby"],
                "confidence": 0.95,
                "evidence": ["扶着床栏"],
                "uncertain": False,
            }
        )
        self.assertTrue(unsafe.uncertain)
        self.assertLessEqual(unsafe.confidence, 0.45)
        self.assertNotIn("优秀", unsafe.description)

    def test_structured_event_normalizes_lying_posture(self):
        result = StructuredVisualEvent.from_mapping(
            {
                "activity": "lying",
                "description": "宝宝清醒地躺在床垫上。",
                "objects": ["baby"],
                "confidence": 0.9,
                "evidence": ["连续画面中宝宝躯干贴着床垫。"],
                "uncertain": False,
            }
        )
        self.assertEqual(result.activity, "lying_awake")
        self.assertFalse(result.uncertain)

    def test_structured_event_maps_to_legacy_business_labels(self):
        self.assertEqual(
            _feeding_label(event("feeding", objects=("baby", "bottle"))),
            "FEED_BOTTLE",
        )
        self.assertEqual(_feeding_label(event("play")), "NOT_FEEDING")
        self.assertEqual(
            _toy_label(event("play", objects=("blocks",))),
            "BLOCKS",
        )
        self.assertEqual(_toy_label(event("sitting")), "NONE")
        self.assertEqual(
            model_name_for_dir("models/qwen3-vl-8b-instruct"),
            "Qwen/Qwen3-VL-8B-Instruct",
        )

        sucking = StructuredVisualEvent.from_mapping(
            {
                "activity": "feeding",
                "description": "成人手持奶瓶，宝宝正在吸吮",
                "objects": ["bottle"],
                "confidence": 0.95,
                "evidence": ["连续画面可见宝宝吸吮奶瓶"],
                "uncertain": False,
            }
        )
        self.assertFalse(sucking.uncertain)
        self.assertEqual(sucking.confidence, 0.95)
        self.assertEqual(_feeding_label(sucking), "FEED_BOTTLE")

    def test_8b_candidate_is_confirmed_by_32b(self):
        primary = FakeVisionModel(
            event("standing", 0.88),
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        high = FakeVisionModel(
            event("standing", 0.91),
            "Qwen/Qwen3-VL-32B-Instruct",
        )
        configs = load_models_config("config/models.toml")
        models = iter((primary, high))
        analyzer = TieredVisionAnalyzer(
            configs.vision_language,
            configs.high_value,
            factory=lambda _config: next(models),
        )
        with patch(
            "cctv_ai.vl_models.tiered._checkpoint_complete",
            return_value=True,
        ):
            result = analyzer.analyze_image("image", "task")
        self.assertTrue(result.confirmed)
        self.assertEqual(result.milestone, "first_stand")
        self.assertEqual(result.final_event.activity, "standing")
        analyzer.close()
        self.assertTrue(primary.closed)
        self.assertTrue(high.closed)

    def test_first_words_requires_confirmed_audio(self):
        configs = load_models_config("config/models.toml")
        primary = FakeVisionModel(
            event("speaking", 0.9),
            "Qwen/Qwen3-VL-8B-Instruct",
        )
        analyzer = TieredVisionAnalyzer(
            configs.vision_language,
            configs.high_value,
            factory=lambda _config: primary,
        )
        result = analyzer.analyze_image("image", "task", {})
        self.assertEqual(result.status, "pending_audio_evidence")

    def test_confirmed_milestone_uses_existing_details_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_database(
                Path(temporary) / "index.sqlite3"
            )
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO activity_events(
                        event_key, activity_type, start_ts, end_ts,
                        duration_seconds, camera_ids_json, location,
                        confidence, time_precision, method, model_version,
                        details_json, created_at, updated_at
                    )
                    VALUES (
                        'event', 'standing', 1, 2, 1, '["cam"]', 'room',
                        0.9, 'exact', 'test', 'test', '{"kept":true}', 1, 1
                    )
                    """
                )
                analysis = TieredAnalysis(
                    event("standing"),
                    "first_stand",
                    "confirmed",
                    event("standing", 0.93),
                )
                self.assertTrue(
                    save_confirmed_milestone(
                        connection,
                        activity_event_id=int(cursor.lastrowid),
                        analysis=analysis,
                    )
                )
                details = json.loads(
                    connection.execute(
                        "SELECT details_json FROM activity_events"
                    ).fetchone()[0]
                )
                self.assertTrue(details["kept"])
                self.assertEqual(
                    details["growth_milestone"]["kind"],
                    "first_stand",
                )
            finally:
                connection.close()

    def test_search_keeps_legacy_index_until_qwen_backfill_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "index.sqlite3"
            connection = connect_database(database)
            try:
                connection.execute(
                    """
                    INSERT INTO cameras(
                        id, name, location, kind, relative_root, audio_enabled
                    ) VALUES ('cam', 'cam', 'room', 'event', 'cam', 1)
                    """
                )
                connection.execute(
                    """
                    INSERT INTO media_files(
                        id, camera_id, relative_path, size_bytes, mtime_ns,
                        first_seen_at, last_seen_at, stable_observations,
                        status, capture_start_ts, time_source,
                        requires_watermark_ocr
                    ) VALUES (
                        1, 'cam', 'a.mp4', 1, 1, 1, 1, 2, 'ready',
                        1, 'test', 0
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
                    ) VALUES (
                        1, 1, 0, 1, 'unknown', 'a.jpg', 'hash',
                        1, 1, 'test', 'test', 1
                    )
                    """
                )
                connection.commit()
                service = VisualSearchService(
                    database_path=database,
                    models_config_path="config/models.toml",
                )
                self.assertEqual(
                    service.select_model(connection),
                    "google/siglip2-base-patch16-224",
                )
                connection.execute(
                    """
                    INSERT INTO visual_embeddings(
                        keyframe_id, model_name, dimensions, dtype,
                        vector, vector_norm, created_at
                    ) VALUES (
                        1, 'Qwen/Qwen3-VL-Embedding-8B', 1,
                        'float16-le', x'0000', 1, 1
                    )
                    """
                )
                self.assertEqual(
                    service.select_model(connection),
                    "Qwen/Qwen3-VL-Embedding-8B",
                )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
