from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from cctv_ai.shadow_calibration import (
    _bucket,
    _classification_metrics,
    build_daily_episodes,
)


def event(
    event_id: int,
    wall_start_ts: float,
    *,
    verdict: str | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "event_label": "baby_cry",
        "model_label": "Baby cry, infant cry",
        "wall_start_ts": wall_start_ts,
        "wall_end_ts": wall_start_ts + 1,
        "confidence": 0.8,
        "media_start_seconds": 10.0,
        "media_end_seconds": 11.0,
        "media_file_id": event_id,
        "camera_id": "baby-top",
        "camera_name": "宝宝房间顶部",
        "location": "baby_room",
        "relative_path": f"baby-top/{event_id}.mp4",
        "review_verdict": verdict,
        "review_note": "",
        "reviewer": "family" if verdict else None,
        "reviewed_at": 1 if verdict else None,
    }


class ShadowCalibrationTests(unittest.TestCase):
    def test_daily_episode_builder_never_merges_across_midnight(self) -> None:
        timezone = ZoneInfo("Asia/Shanghai")
        before_midnight = datetime(
            2026,
            7,
            29,
            23,
            59,
            55,
            tzinfo=timezone,
        ).timestamp()
        after_midnight = datetime(
            2026,
            7,
            30,
            0,
            0,
            5,
            tzinfo=timezone,
        ).timestamp()

        episodes = build_daily_episodes(
            [
                event(1, before_midnight),
                event(2, after_midnight),
            ],
            timezone=timezone,
            gap_seconds=30,
        )

        self.assertEqual(len(episodes), 2)
        self.assertEqual(
            {episode["date"] for episode in episodes},
            {"2026-07-29", "2026-07-30"},
        )

    def test_metrics_and_buckets_are_explicit(self) -> None:
        metrics = _classification_metrics(
            [
                ("confirmed", 0.9),
                ("confirmed", 0.2),
                ("false_positive", 0.1),
                ("false_positive", 0.8),
            ]
        )

        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(_bucket(0.8), "leans_confirmed")
        self.assertEqual(_bucket(0.2), "leans_false_positive")
        self.assertEqual(_bucket(0.5), "uncertain")


if __name__ == "__main__":
    unittest.main()
