from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from cctv_ai.adapters import parse_capture
from cctv_ai.models import CameraConfig


TZ = "Asia/Shanghai"


def camera(camera_id: str, kind: str) -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        name=camera_id,
        location="test",
        kind=kind,
        relative_root="test",
        audio_enabled=True,
        watermark_crop=(
            (55, 10, 1050, 120)
            if kind == "xiaomi_watermark_bundle"
            else None
        ),
    )


class AdapterTests(unittest.TestCase):
    def test_aqara_timestamp(self) -> None:
        parsed = parse_capture(
            camera("lumi", "aqara_event"),
            Path("/archive/20260728/001545.mp4"),
            TZ,
        )
        self.assertEqual(
            parsed.capture_start,
            datetime(2026, 7, 28, 0, 15, 45, tzinfo=ZoneInfo(TZ)),
        )
        self.assertFalse(parsed.requires_watermark_ocr)

    def test_xiaomi_epoch_timestamp(self) -> None:
        parsed = parse_capture(
            camera("50ec", "xiaomi_minute_event"),
            Path("/archive/2026072818/06M23S_1785233183.mp4"),
            TZ,
        )
        self.assertEqual(parsed.capture_start.strftime("%Y%m%d%H%M%S"), "20260728180623")

    def test_xiaomi_bundle_requires_watermark(self) -> None:
        parsed = parse_capture(
            camera("B888", "xiaomi_watermark_bundle"),
            Path(
                "/archive/"
                "00_20260114220026_20260115145253.mp4"
            ),
            TZ,
        )
        self.assertTrue(parsed.requires_watermark_ocr)
        self.assertEqual(parsed.capture_start.strftime("%Y%m%d%H%M%S"), "20260114220026")
        self.assertEqual(
            parsed.capture_end_hint.strftime("%Y%m%d%H%M%S"),
            "20260115145253",
        )

    def test_xiaomi_bundle_invalid_end_is_not_discarded(self) -> None:
        parsed = parse_capture(
            camera("B888", "xiaomi_watermark_bundle"),
            Path(
                "/archive/"
                "00_20260612183256_20260612182914.mp4"
            ),
            TZ,
        )
        self.assertTrue(parsed.requires_watermark_ocr)
        self.assertIsNone(parsed.capture_end_hint)
        self.assertIn("invalid_end", parsed.time_source)


if __name__ == "__main__":
    unittest.main()
