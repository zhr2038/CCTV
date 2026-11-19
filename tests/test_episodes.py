from __future__ import annotations

import unittest
from collections import Counter

from cctv_ai.episodes import (
    build_episodes,
    episode_identifier,
    select_informative_review_inbox,
    select_review_inbox,
)


def event(
    event_id: int,
    start: float,
    end: float,
    confidence: float,
    *,
    verdict: str | None = None,
    note: str = "",
    reviewed_at: int | None = None,
    camera_id: str = "baby-top",
    camera_name: str = "宝宝房间顶部",
) -> dict:
    return {
        "event_id": event_id,
        "event_label": "baby_cry",
        "model_label": "Baby cry, infant cry",
        "wall_start_ts": start,
        "wall_end_ts": end,
        "confidence": confidence,
        "media_start_seconds": start,
        "media_end_seconds": end,
        "media_file_id": event_id,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "location": "baby_room",
        "relative_path": f"baby-top/{event_id}.mp4",
        "review_verdict": verdict,
        "review_note": note,
        "reviewer": "family" if verdict else None,
        "reviewed_at": reviewed_at,
    }


class EpisodeTests(unittest.TestCase):
    def test_events_cluster_by_gap_and_choose_strongest_preview(self) -> None:
        rows = [
            event(3, 157.0, 158.0, 0.70),
            event(1, 100.0, 102.0, 0.80),
            event(2, 125.0, 126.0, 0.95),
        ]

        episodes = build_episodes(rows, gap_seconds=30)

        self.assertEqual(len(episodes), 2)
        newest, oldest = episodes
        self.assertEqual(newest["event_ids"], [3])
        self.assertEqual(oldest["event_ids"], [1, 2])
        self.assertEqual(oldest["representative_event"]["event_id"], 2)
        self.assertEqual(oldest["event_count"], 2)
        self.assertAlmostEqual(oldest["duration_seconds"], 26.0)
        self.assertEqual(oldest["review_status"], "pending")

    def test_review_status_is_only_final_when_entire_episode_agrees(self) -> None:
        confirmed = build_episodes(
            [
                event(
                    1,
                    100.0,
                    101.0,
                    0.8,
                    verdict="confirmed",
                    note="第一次",
                    reviewed_at=10,
                ),
                event(
                    2,
                    102.0,
                    103.0,
                    0.9,
                    verdict="confirmed",
                    note="最终备注",
                    reviewed_at=20,
                ),
            ]
        )[0]
        mixed = build_episodes(
            [
                event(
                    1,
                    100.0,
                    101.0,
                    0.8,
                    verdict="confirmed",
                    reviewed_at=10,
                ),
                event(2, 102.0, 103.0, 0.9),
            ]
        )[0]

        self.assertEqual(confirmed["review_status"], "confirmed")
        self.assertEqual(confirmed["review_note"], "最终备注")
        self.assertEqual(mixed["review_status"], "mixed")

    def test_episode_identifier_is_stable_and_gap_must_be_nonnegative(self) -> None:
        self.assertEqual(
            episode_identifier("baby_cry", [3, 1, 2]),
            episode_identifier("baby_cry", [2, 3, 1]),
        )
        with self.assertRaises(ValueError):
            build_episodes([], gap_seconds=-0.1)

    def test_smart_inbox_is_fixed_and_balances_review_lanes(self) -> None:
        rows: list[dict] = []
        next_id = 1
        for index in range(14):
            start = 100.0 + index * 100.0
            rows.append(
                event(
                    next_id,
                    start,
                    start + 1.0,
                    0.62 + (index % 6) * 0.06,
                )
            )
            next_id += 1
            if index < 4:
                rows.append(
                    event(
                        next_id,
                        start + 2.0,
                        start + 3.0,
                        0.74,
                        camera_id="baby-bed",
                        camera_name="宝宝床头",
                    )
                )
                next_id += 1

        episodes = build_episodes(rows, gap_seconds=30)
        first = select_review_inbox(episodes, quota=10)
        status_changed = [
            {**episode, "review_status": "confirmed"}
            for episode in reversed(episodes)
        ]
        second = select_review_inbox(status_changed, quota=10)

        self.assertEqual(len(first), 10)
        self.assertEqual(
            Counter(item["review_lane"] for item in first),
            Counter({"priority": 5, "learning": 3, "audit": 2}),
        )
        self.assertEqual(
            [item["episode_id"] for item in first],
            [item["episode_id"] for item in second],
        )
        multi_camera_ids = {
            episode["episode_id"]
            for episode in episodes
            if len(episode["cameras"]) > 1
        }
        priority_ids = {
            episode["episode_id"]
            for episode in first
            if episode["review_lane"] == "priority"
        }
        self.assertTrue(multi_camera_ids.issubset(priority_ids))
        self.assertEqual(
            [item["review_rank"] for item in first],
            list(range(1, 11)),
        )
        with self.assertRaises(ValueError):
            select_review_inbox(episodes, quota=0)

    def test_five_item_inbox_keeps_one_audit_lane(self) -> None:
        episodes = build_episodes(
            [
                event(
                    index,
                    100.0 + index * 100.0,
                    101.0 + index * 100.0,
                    0.65 + (index % 5) * 0.05,
                )
                for index in range(1, 9)
            ]
        )

        selected = select_review_inbox(episodes, quota=5)

        self.assertEqual(
            Counter(item["review_lane"] for item in selected),
            Counter({"priority": 2, "learning": 2, "audit": 1}),
        )

    def test_shadow_inbox_selects_boundary_and_directional_audits(self) -> None:
        episodes = build_episodes(
            [
                event(
                    index,
                    100.0 + index * 100.0,
                    101.0 + index * 100.0,
                    0.65 + (index % 5) * 0.05,
                )
                for index in range(1, 13)
            ]
        )
        base_pool = select_review_inbox(episodes, quota=10)
        shadow_scores = {
            item["episode_id"]: {
                "score": index / 10,
                "review_priority": (
                    2.0
                    if index in {1, 10}
                    else 1.0 - abs(index / 10 - 0.5)
                ),
            }
            for index, item in enumerate(base_pool, start=1)
        }

        selected = select_informative_review_inbox(
            episodes,
            shadow_scores=shadow_scores,
            quota=5,
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(
            Counter(item["review_lane"] for item in selected),
            Counter({"learning": 3, "audit": 2}),
        )
        self.assertTrue(
            {item["episode_id"] for item in selected}.issubset(
                {item["episode_id"] for item in base_pool}
            )
        )
        self.assertTrue(
            any(
                item["review_reason"] == "偏误报方向抽检"
                for item in selected
            )
        )
        self.assertTrue(
            any(
                item["review_reason"] == "偏哭声方向抽检"
                for item in selected
            )
        )


if __name__ == "__main__":
    unittest.main()
