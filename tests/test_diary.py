from __future__ import annotations

import unittest

from cctv_ai.diary import (
    _clip_event_to_period,
    build_feeding_routine_reference,
    build_life_brief,
    build_life_sessions,
    select_review_queue,
)


class DiaryPeriodTests(unittest.TestCase):
    @staticmethod
    def event(
        event_id: int,
        activity_type: str,
        start_ts: float,
        end_ts: float,
        *,
        subtype: str | None = None,
        confidence: float = 0.7,
        location: str = "宝宝房间",
    ) -> dict:
        return {
            "id": event_id,
            "activity_type": activity_type,
            "activity_name": activity_type,
            "activity_subtype": subtype,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_seconds": end_ts - start_ts,
            "camera_ids": [f"cam-{event_id % 2}"],
            "location": location,
            "confidence": confidence,
            "time_precision": "exact",
            "review_status": "machine",
            "evidence_keyframe_id": event_id,
            "image_url": f"/api/keyframes/{event_id}/image",
        }

    def test_life_sessions_merge_fragments_but_keep_meal_types_apart(self):
        sessions = build_life_sessions(
            [
                self.event(
                    1,
                    "feeding",
                    100,
                    160,
                    subtype="feeding_bottle",
                ),
                self.event(
                    2,
                    "feeding",
                    600,
                    660,
                    subtype="feeding_bottle",
                    location="客厅",
                ),
                self.event(
                    3,
                    "feeding",
                    700,
                    760,
                    subtype="solid_food",
                ),
            ]
        )

        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["source_event_count"], 2)
        self.assertEqual(sessions[0]["locations"], ["宝宝房间", "客厅"])
        self.assertEqual(sessions[0]["visible_seconds"], 120.0)
        self.assertEqual(sessions[1]["activity_subtype"], "solid_food")

    def test_overlapping_camera_fragments_do_not_double_count_duration(self):
        sessions = build_life_sessions(
            [
                self.event(1, "play", 1_000, 1_100),
                self.event(2, "play", 1_050, 1_150, location="客厅"),
            ]
        )

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["span_seconds"], 150.0)
        self.assertEqual(sessions[0]["visible_seconds"], 150.0)

    def test_review_queue_is_capped_and_only_keeps_stat_changing_items(self):
        events = [
            self.event(index, activity, index * 2_000, index * 2_000 + 60)
            for index, activity in enumerate(
                [
                    "feeding",
                    "inferred_sleep",
                    "crawling",
                    "sitting",
                    "play",
                ],
                start=1,
            )
        ]
        sessions = build_life_sessions(events)
        queue = select_review_queue(sessions, limit=3)

        self.assertEqual(queue["shown"], 3)
        self.assertEqual(len(queue["items"]), 3)
        self.assertNotIn(
            "play",
            {item["activity_type"] for item in queue["items"]},
        )
        self.assertGreater(queue["automatic_sessions"], 0)

    def test_life_brief_reports_sessions_and_intervals(self):
        sessions = build_life_sessions(
            [
                self.event(1, "feeding", 100, 160),
                self.event(2, "feeding", 2_000, 2_060),
                self.event(3, "play", 3_000, 3_120, location="客厅"),
            ]
        )
        brief = build_life_brief(sessions, toys=[], source_event_count=3)

        self.assertEqual(brief["feeding"]["sessions"], 2)
        self.assertEqual(
            brief["feeding"]["average_interval_seconds"],
            1_900.0,
        )
        self.assertEqual(brief["play"]["top_location"], "客厅")

    def test_three_hour_routine_merges_top_up_and_estimates_missing_feeds(
        self,
    ) -> None:
        def at(hour: int, minute: int) -> float:
            return hour * 3600 + minute * 60

        sightings = [
            {
                "start_ts": at(hour, minute),
                "certainty": certainty,
                "location": "宝宝床",
                "evidence_points": 1,
            }
            for hour, minute, certainty in (
                (6, 23, "confirmed"),
                (7, 5, "confirmed"),
                (9, 30, "confirmed"),
                (12, 23, "confirmed"),
                (15, 31, "confirmed"),
                (18, 16, "probable"),
            )
        ]

        routine, episodes = build_feeding_routine_reference(
            sightings,
            since_ts=0,
            until_ts=24 * 3600,
            target_interval_seconds=3 * 3600,
            episode_merge_seconds=75 * 60,
            reference_now_ts=25 * 3600,
        )

        self.assertEqual(len(episodes), 5)
        self.assertEqual(episodes[0]["sighting_count"], 2)
        self.assertEqual(episodes[0]["start_ts"], at(6, 23))
        self.assertEqual(episodes[0]["last_sighting_ts"], at(7, 5))
        self.assertEqual(
            [item["certainty"] for item in episodes],
            ["confirmed", "confirmed", "confirmed", "confirmed", "probable"],
        )
        self.assertEqual(routine["estimated_sessions"], 8)
        self.assertEqual(routine["observed_sessions"], 5)
        self.assertEqual(routine["likely_unobserved_sessions"], 3)
        self.assertEqual(len(routine["likely_unobserved_slots"]), 3)
        self.assertEqual(routine["cadence_fit"], "strong")

    def test_three_hour_routine_stops_after_evening_sleep(self) -> None:
        def at(hour: int, minute: int) -> float:
            return hour * 3600 + minute * 60

        sightings = [
            {
                "start_ts": at(hour, minute),
                "certainty": certainty,
                "location": "宝宝床",
                "evidence_points": 1,
            }
            for hour, minute, certainty in (
                (6, 23, "confirmed"),
                (7, 5, "confirmed"),
                (9, 30, "confirmed"),
                (12, 23, "confirmed"),
                (15, 31, "confirmed"),
                (18, 16, "probable"),
            )
        ]

        routine, episodes = build_feeding_routine_reference(
            sightings,
            since_ts=0,
            until_ts=24 * 3600,
            target_interval_seconds=3 * 3600,
            episode_merge_seconds=75 * 60,
            routine_start_ts=at(6, 3),
            routine_end_ts=at(20, 9),
            reference_now_ts=25 * 3600,
        )

        self.assertEqual(len(episodes), 5)
        self.assertEqual(routine["estimated_sessions"], 5)
        self.assertEqual(routine["observed_routine_sessions"], 5)
        self.assertEqual(routine["likely_unobserved_sessions"], 0)
        self.assertEqual(routine["likely_unobserved_slots"], [])
        self.assertEqual(routine["routine_window_start_ts"], at(6, 3))
        self.assertEqual(routine["routine_window_end_ts"], at(20, 9))
        self.assertFalse(routine["nighttime_inference_enabled"])

    def test_cross_midnight_event_only_counts_period_overlap(self) -> None:
        event = {
            "id": 1,
            "start_ts": 900.0,
            "end_ts": 1008.0,
            "duration_seconds": 107.0,
            "session_span_seconds": 108.0,
        }

        clipped = _clip_event_to_period(
            event,
            since_ts=1000.0,
            until_ts=2000.0,
        )

        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertEqual(clipped["start_ts"], 1000.0)
        self.assertEqual(clipped["end_ts"], 1008.0)
        self.assertEqual(clipped["duration_seconds"], 8.0)
        self.assertEqual(clipped["session_span_seconds"], 8.0)
        self.assertTrue(clipped["period_clipped"])
        self.assertEqual(clipped["source_start_ts"], 900.0)

    def test_event_outside_period_is_omitted(self) -> None:
        event = {
            "start_ts": 900.0,
            "end_ts": 1000.0,
            "duration_seconds": 100.0,
            "session_span_seconds": 100.0,
        }

        self.assertIsNone(
            _clip_event_to_period(
                event,
                since_ts=1000.0,
                until_ts=2000.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
