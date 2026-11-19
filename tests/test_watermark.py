from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from cctv_ai.watermark import (
    WatermarkObservation,
    build_time_segments,
    parse_watermark_text,
    sample_offsets,
)


TZ = ZoneInfo("Asia/Shanghai")


class WatermarkTests(unittest.TestCase):
    def test_parse_watermark(self) -> None:
        value = parse_watermark_text("mi 2026/01/15 14:19:54", "Asia/Shanghai")
        self.assertEqual(value, datetime(2026, 1, 15, 14, 19, 54, tzinfo=TZ))

    def test_parse_watermark_without_ocr_space(self) -> None:
        value = parse_watermark_text(
            "2026/01/1507:38:41",
            "Asia/Shanghai",
        )
        self.assertEqual(value, datetime(2026, 1, 15, 7, 38, 41, tzinfo=TZ))

    def test_parse_watermark_with_colon_between_date_and_time(self) -> None:
        value = parse_watermark_text(
            "2026/07/05:13:16:48",
            "Asia/Shanghai",
        )
        self.assertEqual(value, datetime(2026, 7, 5, 13, 16, 48, tzinfo=TZ))

    def test_parse_watermark_with_first_slash_read_as_one(self) -> None:
        value = parse_watermark_text(
            "mj2026107/06 10:26:55",
            "Asia/Shanghai",
        )
        self.assertEqual(value, datetime(2026, 7, 6, 10, 26, 55, tzinfo=TZ))

    def test_parse_watermark_with_final_colon_read_as_space(self) -> None:
        value = parse_watermark_text(
            "mj2026/07/08 14:54 44",
            "Asia/Shanghai",
        )
        self.assertEqual(value, datetime(2026, 7, 8, 14, 54, 44, tzinfo=TZ))

    def test_parse_watermark_with_date_time_space_read_as_one(self) -> None:
        value = parse_watermark_text(
            "m)2026/07/12107:47:06",
            "Asia/Shanghai",
        )
        self.assertEqual(value, datetime(2026, 7, 12, 7, 47, 6, tzinfo=TZ))

    def test_detect_large_wall_clock_jump(self) -> None:
        observations = [
            WatermarkObservation(
                1.0, datetime(2026, 1, 14, 22, 0, 22, tzinfo=TZ)
            ),
            WatermarkObservation(
                11.0, datetime(2026, 1, 14, 22, 0, 32, tzinfo=TZ)
            ),
            WatermarkObservation(
                400.0, datetime(2026, 1, 15, 14, 19, 54, tzinfo=TZ)
            ),
            WatermarkObservation(
                410.0, datetime(2026, 1, 15, 14, 20, 4, tzinfo=TZ)
            ),
        ]
        segments = build_time_segments(observations)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].media_end_seconds, 11.0)
        self.assertEqual(segments[1].media_start_seconds, 400.0)

    def test_sampling_includes_end(self) -> None:
        offsets = sample_offsets(25.0, 10.0)
        self.assertEqual(offsets, [0.5, 10.5, 20.5, 24.5])


if __name__ == "__main__":
    unittest.main()
