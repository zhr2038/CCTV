from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .visual_models import (
    SLEEP_PRESENCE_PROMPT_VERSION,
    TEMPORAL_FEEDING_PROMPT_VERSION,
)


ACTIVITY_NAMES = {
    "feeding": "喝奶/进食",
    "play": "玩耍",
    "crawling": "爬行姿态",
    "sitting": "坐姿",
    "standing": "站立",
    "lying_awake": "清醒静卧",
    "inferred_sleep": "推定睡眠",
}

SUBTYPE_NAMES = {
    "feeding_bottle": "奶瓶喂奶",
    "breastfeeding": "母乳喂养",
    "solid_food": "辅食",
    "playing": "玩玩具",
    "reading": "亲子阅读",
    "low_motion_in_bed": "床内低运动",
}

SESSION_GAP_SECONDS = {
    "feeding": 20 * 60,
    # Sleep state events are already joined across consecutive verified
    # anchors.  A remaining gap represents an awake/unclear/unknown interval
    # and must not be hidden inside one long sleep session.
    "inferred_sleep": 60,
    "play": 8 * 60,
    "crawling": 3 * 60,
    "sitting": 3 * 60,
    "standing": 3 * 60,
    "lying_awake": 5 * 60,
}

HIGHLIGHT_PRIORITY = {
    "feeding": 7,
    "standing": 6,
    "crawling": 5,
    "sitting": 4,
    "inferred_sleep": 3,
    "play": 2,
    "lying_awake": 1,
}

REVIEW_PRIORITY = {
    "feeding": 7,
    "inferred_sleep": 6,
    "crawling": 5,
    "standing": 5,
    "sitting": 4,
}


def period_bounds(
    value: str,
    timezone: ZoneInfo,
    period: str,
) -> tuple[datetime, datetime]:
    start = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone)
    if period == "day":
        return start, start + timedelta(days=1)
    if period == "week":
        week_start = start - timedelta(days=start.weekday())
        return week_start, week_start + timedelta(days=7)
    if period == "month":
        month_start = start.replace(day=1)
        if month_start.month == 12:
            month_end = month_start.replace(
                year=month_start.year + 1,
                month=1,
            )
        else:
            month_end = month_start.replace(month=month_start.month + 1)
        return month_start, month_end
    raise ValueError(f"Unsupported diary period: {period}")


def available_diary_dates(
    connection: sqlite3.Connection,
    *,
    timezone: ZoneInfo,
) -> list[str]:
    rows = connection.execute(
        """
        SELECT DISTINCT CAST(wall_ts AS INTEGER) AS wall_ts
        FROM visual_keyframes
        WHERE wall_ts BETWEEN ? AND ?
        ORDER BY wall_ts DESC
        """,
        (
            int(datetime(2000, 1, 1, tzinfo=timezone).timestamp()),
            int(time.time()) + 24 * 3600,
        ),
    )
    dates = {
        datetime.fromtimestamp(int(row["wall_ts"]), timezone).date().isoformat()
        for row in rows
    }
    return sorted(dates, reverse=True)


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _effective_event(row: sqlite3.Row) -> dict[str, Any]:
    activity_type = (
        row["corrected_activity_type"]
        or row["activity_type"]
    )
    subtype = (
        row["corrected_activity_subtype"]
        if row["corrected_activity_subtype"] is not None
        else row["activity_subtype"]
    )
    start_ts = float(
        row["corrected_start_ts"]
        if row["corrected_start_ts"] is not None
        else row["start_ts"]
    )
    end_ts = float(
        row["corrected_end_ts"]
        if row["corrected_end_ts"] is not None
        else row["end_ts"]
    )
    has_time_correction = (
        row["corrected_start_ts"] is not None
        or row["corrected_end_ts"] is not None
    )
    duration_seconds = (
        max(0.0, end_ts - start_ts)
        if has_time_correction
        else max(0.0, float(row["duration_seconds"]))
    )
    review_status = row["review_verdict"] or row["review_status"]
    return {
        "id": int(row["id"]),
        "event_key": str(row["event_key"]),
        "activity_type": str(activity_type),
        "activity_name": ACTIVITY_NAMES.get(
            str(activity_type),
            str(activity_type),
        ),
        "activity_subtype": subtype,
        "subtype_name": (
            SUBTYPE_NAMES.get(str(subtype), str(subtype))
            if subtype
            else None
        ),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_seconds": duration_seconds,
        "session_span_seconds": max(0.0, end_ts - start_ts),
        "camera_ids": _json_list(row["camera_ids_json"]),
        "location": str(row["location"]),
        "zone": row["zone"],
        "confidence": float(row["confidence"]),
        "time_precision": str(row["time_precision"]),
        "review_status": str(review_status),
        "review_note": str(row["review_note"] or ""),
        "details": _json_object(row["details_json"]),
        "evidence_keyframe_id": (
            int(row["evidence_keyframe_id"])
            if row["evidence_keyframe_id"] is not None
            else None
        ),
        "evidence_score": (
            float(row["evidence_score"])
            if row["evidence_score"] is not None
            else None
        ),
        "image_url": (
            f"/api/keyframes/{int(row['evidence_keyframe_id'])}/image"
            if row["evidence_keyframe_id"] is not None
            else None
        ),
    }


def query_activity_events(
    connection: sqlite3.Connection,
    *,
    since_ts: float,
    until_ts: float,
    activity_type: str | None = None,
    include_rejected: bool = False,
    event_id: int | None = None,
) -> list[dict[str, Any]]:
    conditions = [
        "e.start_ts < ?",
        "e.end_ts >= ?",
    ]
    parameters: list[Any] = [until_ts, since_ts]
    if activity_type:
        conditions.append(
            "COALESCE(r.corrected_activity_type, e.activity_type) = ?"
        )
        parameters.append(activity_type)
    if event_id is not None:
        conditions.append("e.id = ?")
        parameters.append(event_id)
    if not include_rejected:
        conditions.append(
            "COALESCE(r.verdict, e.review_status) != 'rejected'"
        )
    rows = connection.execute(
        f"""
        SELECT
            e.*,
            r.verdict AS review_verdict,
            r.corrected_activity_type,
            r.corrected_activity_subtype,
            r.corrected_start_ts,
            r.corrected_end_ts,
            r.note AS review_note,
            (
                SELECT ev.keyframe_id
                FROM activity_event_evidence ev
                WHERE ev.activity_event_id = e.id
                ORDER BY ev.evidence_rank
                LIMIT 1
            ) AS evidence_keyframe_id,
            (
                SELECT ev.score
                FROM activity_event_evidence ev
                WHERE ev.activity_event_id = e.id
                ORDER BY ev.evidence_rank
                LIMIT 1
            ) AS evidence_score
        FROM activity_events e
        LEFT JOIN activity_event_reviews r
          ON r.activity_event_id = e.id
        WHERE {' AND '.join(conditions)}
        ORDER BY e.start_ts, e.id
        """,
        parameters,
    )
    return [_effective_event(row) for row in rows]


def _clip_event_to_period(
    event: Mapping[str, Any],
    *,
    since_ts: float,
    until_ts: float,
) -> dict[str, Any] | None:
    """Return the portion of an event that is visible in one diary period.

    Events can cross midnight.  The database keeps the complete source event,
    while a day/week/month view must only count and display the overlapping
    portion; otherwise a few seconds after midnight can import the entire
    previous session into the next day's totals.
    """
    original_start = float(event["start_ts"])
    original_end = float(event["end_ts"])
    clipped_start = max(since_ts, original_start)
    clipped_end = min(until_ts, original_end)
    if clipped_end <= clipped_start:
        return None
    clipped = dict(event)
    clipped_span = clipped_end - clipped_start
    clipped["start_ts"] = clipped_start
    clipped["end_ts"] = clipped_end
    clipped["session_span_seconds"] = clipped_span
    was_clipped = (
        clipped_start != original_start or clipped_end != original_end
    )
    clipped["period_clipped"] = was_clipped
    if was_clipped:
        clipped["source_start_ts"] = original_start
        clipped["source_end_ts"] = original_end
        clipped["source_duration_seconds"] = float(
            event["duration_seconds"]
        )
        clipped["duration_seconds"] = min(
            max(0.0, float(event["duration_seconds"])),
            clipped_span,
        )
    return clipped


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    since_ts: float,
    until_ts: float,
) -> float:
    clipped = sorted(
        (
            max(since_ts, float(start)),
            min(until_ts, float(end)),
        )
        for start, end in intervals
        if end > since_ts and start < until_ts
    )
    if not clipped:
        return 0.0
    total = 0.0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end + 1.0:
            current_end = max(current_end, end)
        else:
            total += max(0.0, current_end - current_start)
            current_start, current_end = start, end
    return total + max(0.0, current_end - current_start)


def _can_join_session(
    current: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
) -> bool:
    if not current:
        return True
    activity_type = str(event["activity_type"])
    if activity_type != str(current[0]["activity_type"]):
        return False
    gap = SESSION_GAP_SECONDS.get(activity_type, 3 * 60)
    if float(event["start_ts"]) > max(
        float(item["end_ts"]) for item in current
    ) + gap:
        return False
    if activity_type == "feeding":
        subtypes = {
            str(item["activity_subtype"])
            for item in current
            if item.get("activity_subtype")
        }
        incoming = event.get("activity_subtype")
        if subtypes and incoming and str(incoming) not in subtypes:
            return False
    return True


def _finalize_life_session(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        events,
        key=lambda item: (float(item["start_ts"]), int(item["id"])),
    )
    activity_type = str(ordered[0]["activity_type"])
    start_ts = min(float(item["start_ts"]) for item in ordered)
    end_ts = max(float(item["end_ts"]) for item in ordered)
    span_seconds = max(0.0, end_ts - start_ts)
    evidence_seconds = sum(
        max(0.0, float(item.get("duration_seconds") or 0.0))
        for item in ordered
    )
    visible_seconds = min(span_seconds, evidence_seconds)
    weights = [
        max(1.0, float(item.get("duration_seconds") or 0.0))
        for item in ordered
    ]
    confidence = sum(
        float(item.get("confidence") or 0.0) * weight
        for item, weight in zip(ordered, weights, strict=True)
    ) / sum(weights)
    representative = max(
        ordered,
        key=lambda item: (
            item.get("evidence_keyframe_id") is not None,
            float(item.get("confidence") or 0.0),
            float(item.get("duration_seconds") or 0.0),
            -int(item["id"]),
        ),
    )
    locations = list(
        dict.fromkeys(
            str(item["location"])
            for item in ordered
            if item.get("location")
        )
    )
    camera_ids = list(
        dict.fromkeys(
            str(camera_id)
            for item in ordered
            for camera_id in item.get("camera_ids", ())
        )
    )
    subtypes = list(
        dict.fromkeys(
            str(item["activity_subtype"])
            for item in ordered
            if item.get("activity_subtype")
        )
    )
    precisions = {
        str(item.get("time_precision") or "estimated") for item in ordered
    }
    precision = (
        "estimated"
        if "estimated" in precisions
        else "mapped"
        if "mapped" in precisions
        else "exact"
    )
    review_statuses = {
        str(item.get("review_status") or "machine") for item in ordered
    }
    review_status = (
        "reviewed"
        if review_statuses - {"machine", "uncertain"}
        else "uncertain"
        if "uncertain" in review_statuses
        else "machine"
    )
    details = [
        dict(item.get("details") or {}) for item in ordered
    ]
    measurement_modes = {
        str(value.get("measurement_mode") or "visible_lower_bound")
        for value in details
    }
    measurement_mode = (
        next(iter(measurement_modes))
        if len(measurement_modes) == 1
        else "mixed_observation"
    )
    certainty = (
        "confirmed"
        if any(value.get("certainty") == "confirmed" for value in details)
        else "probable"
        if any(value.get("certainty") == "probable" for value in details)
        else "uncertain"
    )
    first_details = dict(ordered[0].get("details") or {})
    last_details = dict(ordered[-1].get("details") or {})
    return {
        "id": f"{activity_type}:{int(start_ts)}:{int(representative['id'])}",
        "activity_type": activity_type,
        "activity_name": str(ordered[0]["activity_name"]),
        "activity_subtype": subtypes[0] if len(subtypes) == 1 else None,
        "subtype_name": (
            SUBTYPE_NAMES.get(subtypes[0], subtypes[0])
            if len(subtypes) == 1
            else None
        ),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "span_seconds": round(span_seconds, 3),
        "visible_seconds": round(visible_seconds, 3),
        "source_event_count": len(ordered),
        "source_event_ids": [int(item["id"]) for item in ordered],
        "representative_event_id": int(representative["id"]),
        "evidence_keyframe_id": representative.get("evidence_keyframe_id"),
        "image_url": representative.get("image_url"),
        "evidence_events": sum(
            item.get("evidence_keyframe_id") is not None for item in ordered
        ),
        "locations": locations,
        "location": locations[0] if len(locations) == 1 else "多视角",
        "camera_ids": camera_ids,
        "confidence": round(confidence, 4),
        "time_precision": precision,
        "review_status": review_status,
        "measurement_mode": measurement_mode,
        "certainty": certainty,
        "duration_publishable": bool(
            details and all(
                value.get("duration_publishable") is True
                for value in details
            )
        ),
        "start_boundary_error_seconds": first_details.get(
            "start_boundary_error_seconds"
        ),
        "end_boundary_error_seconds": last_details.get(
            "end_boundary_error_seconds"
        ),
        "evidence_points": sum(
            int(value.get("support_frames") or 0) for value in details
        ),
    }


def build_life_sessions(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge adjacent cross-camera fragments into human-scale life sessions."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    completed: list[dict[str, Any]] = []
    for event in sorted(
        events,
        key=lambda item: (float(item["start_ts"]), int(item["id"])),
    ):
        activity_type = str(event["activity_type"])
        current = grouped[activity_type]
        if current and not _can_join_session(current, event):
            completed.append(_finalize_life_session(current))
            grouped[activity_type] = []
            current = grouped[activity_type]
        current.append(event)
    completed.extend(
        _finalize_life_session(current)
        for current in grouped.values()
        if current
    )
    return sorted(
        completed,
        key=lambda item: (item["start_ts"], item["activity_type"]),
    )


def _sessions_of_type(
    sessions: Sequence[Mapping[str, Any]],
    activity_type: str,
) -> list[Mapping[str, Any]]:
    return [
        item
        for item in sessions
        if item["activity_type"] == activity_type
    ]


def build_life_brief(
    sessions: Sequence[Mapping[str, Any]],
    *,
    toys: Sequence[Mapping[str, Any]],
    source_event_count: int,
) -> dict[str, Any]:
    feeding = _sessions_of_type(sessions, "feeding")
    sleep = _sessions_of_type(sessions, "inferred_sleep")
    play = _sessions_of_type(sessions, "play")
    crawling = _sessions_of_type(sessions, "crawling")
    sitting = _sessions_of_type(sessions, "sitting")
    intervals = [
        max(0.0, float(right["start_ts"]) - float(left["start_ts"]))
        for left, right in zip(feeding, feeding[1:])
    ]
    play_locations: dict[str, float] = defaultdict(float)
    for session in play:
        locations = session.get("locations") or [session.get("location")]
        for location in locations:
            if location:
                play_locations[str(location)] += float(
                    session["visible_seconds"]
                ) / max(1, len(locations))
    top_location = (
        max(play_locations.items(), key=lambda item: item[1])[0]
        if play_locations
        else None
    )
    favorite_toy = next(
        (dict(toy) for toy in toys if toy.get("is_specific")),
        None,
    )
    return {
        "headline": (
            f"已把 {source_event_count} 段可见证据合并为 "
            f"{len(sessions)} 个生活会话。"
        ),
        "feeding": {
            "sessions": len(feeding),
            "confirmed_sessions": sum(
                item.get("certainty") == "confirmed" for item in feeding
            ),
            "probable_sessions": sum(
                item.get("certainty") == "probable" for item in feeding
            ),
            "visible_seconds": round(
                sum(float(item["visible_seconds"]) for item in feeding),
                3,
            ),
            "times": [float(item["start_ts"]) for item in feeding],
            "sightings": [
                {
                    "start_ts": float(item["start_ts"]),
                    "certainty": str(
                        item.get("certainty") or "uncertain"
                    ),
                    "subtype": item.get("activity_subtype"),
                    "location": item.get("location"),
                }
                for item in feeding
            ],
            "intervals_seconds": [round(value, 3) for value in intervals],
            "average_interval_seconds": (
                round(sum(intervals) / len(intervals), 3)
                if intervals
                else None
            ),
        },
        "sleep": {
            "sessions": len(sleep),
            "visible_seconds": round(
                sum(float(item["visible_seconds"]) for item in sleep),
                3,
            ),
            "longest_seconds": round(
                max(
                    (float(item["visible_seconds"]) for item in sleep),
                    default=0.0,
                ),
                3,
            ),
            "max_boundary_error_seconds": max(
                (
                    float(value)
                    for item in sleep
                    for value in (
                        item.get("start_boundary_error_seconds"),
                        item.get("end_boundary_error_seconds"),
                    )
                    if value is not None
                ),
                default=None,
            ),
        },
        "play": {
            "sessions": len(play),
            "visible_seconds": round(
                sum(float(item["visible_seconds"]) for item in play),
                3,
            ),
            "top_location": top_location,
            "top_toy": favorite_toy,
        },
        "mobility": {
            "crawling_sessions": len(crawling),
            "crawling_seconds": round(
                sum(float(item["visible_seconds"]) for item in crawling),
                3,
            ),
            "sitting_sessions": len(sitting),
            "sitting_seconds": round(
                sum(float(item["visible_seconds"]) for item in sitting),
                3,
            ),
        },
    }


def select_life_highlights(
    sessions: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    reasons = {
        "feeding": "监控内可见的喂养会话",
        "standing": "可见站立姿态",
        "crawling": "可见爬行姿态",
        "sitting": "可见坐姿",
        "inferred_sleep": "连续睡眠证据",
        "play": "可见互动较完整",
        "lying_awake": "可见清醒静卧",
    }
    eligible = [
        item
        for item in sessions
        if item.get("image_url") and item.get("representative_event_id")
    ]
    ordered = sorted(
        eligible,
        key=lambda item: (
            HIGHLIGHT_PRIORITY.get(str(item["activity_type"]), 0),
            float(item.get("confidence") or 0.0),
            float(item.get("visible_seconds") or 0.0),
        ),
        reverse=True,
    )
    selected: list[Mapping[str, Any]] = []
    seen_types: set[str] = set()
    for item in ordered:
        activity_type = str(item["activity_type"])
        if activity_type in seen_types:
            continue
        selected.append(item)
        seen_types.add(activity_type)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_ids = {str(item["id"]) for item in selected}
        selected.extend(
            item
            for item in ordered
            if str(item["id"]) not in selected_ids
        )
    return [
        {
            **dict(item),
            "reason": reasons.get(
                str(item["activity_type"]),
                "可见活动证据",
            ),
        }
        for item in selected[:limit]
    ]


def select_review_queue(
    sessions: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> dict[str, Any]:
    machine_sessions = [
        item for item in sessions if item.get("review_status") == "machine"
    ]
    eligible = [
        item
        for item in machine_sessions
        if item.get("activity_type") in REVIEW_PRIORITY
        and item.get("image_url")
        and float(item.get("confidence") or 0.0) < 0.82
    ]
    ordered = sorted(
        eligible,
        key=lambda item: (
            REVIEW_PRIORITY.get(str(item["activity_type"]), 0),
            1.0 - float(item.get("confidence") or 0.0),
            item.get("time_precision") == "estimated",
        ),
        reverse=True,
    )
    selected: list[Mapping[str, Any]] = []
    seen_types: set[str] = set()
    for item in ordered:
        activity_type = str(item["activity_type"])
        if activity_type in seen_types:
            continue
        selected.append(item)
        seen_types.add(activity_type)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_ids = {str(item["id"]) for item in selected}
        selected.extend(
            item
            for item in ordered
            if str(item["id"]) not in selected_ids
        )
    reasons = {
        "feeding": "确认后会改变喝奶次数",
        "inferred_sleep": "确认后会改变睡眠时长",
        "crawling": "确认后会改变爬行统计",
        "standing": "确认后会改变站立统计",
        "sitting": "确认后会改变坐姿统计",
    }
    items = [
        {
            **dict(item),
            "reason": reasons[str(item["activity_type"])],
        }
        for item in selected[:limit]
    ]
    return {
        "items": items,
        "shown": len(items),
        "candidate_sessions": len(eligible),
        "automatic_sessions": max(len(machine_sessions) - len(items), 0),
        "limit": limit,
        "policy": "每天最多只请家长复核三条、且只选会改变统计的候选。",
    }


def build_daily_series(
    sessions: Sequence[Mapping[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    day_start = start
    while day_start < end:
        day_end = min(day_start + timedelta(days=1), end)
        day_sessions = [
            item
            for item in sessions
            if float(item["start_ts"]) < day_end.timestamp()
            and float(item["end_ts"]) > day_start.timestamp()
        ]
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for item in day_sessions:
            activity_type = str(item["activity_type"])
            overlap = max(
                0.0,
                min(float(item["end_ts"]), day_end.timestamp())
                - max(float(item["start_ts"]), day_start.timestamp()),
            )
            totals[activity_type] += min(
                overlap,
                float(item["visible_seconds"]),
            )
            counts[activity_type] += 1
        series.append(
            {
                "date": day_start.date().isoformat(),
                "feeding_sessions": counts["feeding"],
                "play_seconds": round(totals["play"], 3),
                "sleep_seconds": round(totals["inferred_sleep"], 3),
                "crawling_seconds": round(totals["crawling"], 3),
                "sitting_seconds": round(totals["sitting"], 3),
            }
        )
        day_start = day_end
    return series


def query_coverage(
    connection: sqlite3.Connection,
    *,
    since_ts: float,
    until_ts: float,
) -> list[dict[str, Any]]:
    camera_rows = list(
        connection.execute(
            """
            SELECT id, name, location, kind
            FROM cameras
            ORDER BY id
            """
        )
    )
    coverage: list[dict[str, Any]] = []
    period_seconds = max(1.0, until_ts - since_ts)
    for camera in camera_rows:
        camera_id = str(camera["id"])
        media = connection.execute(
            """
            SELECT
                COUNT(*) AS files,
                MIN(capture_start_ts) AS first_ts,
                MAX(capture_start_ts) AS last_ts,
                SUM(
                    CASE WHEN status = 'probed' THEN 1 ELSE 0 END
                ) AS probed_files,
                SUM(
                    CASE WHEN status IN ('ready', 'probed')
                    THEN 1 ELSE 0 END
                ) AS stable_files,
                SUM(
                    CASE WHEN status = 'waiting' THEN 1 ELSE 0 END
                ) AS unstable_files,
                SUM(
                    CASE WHEN status = 'error' THEN 1 ELSE 0 END
                ) AS source_error_files,
                SUM(
                    CASE WHEN status = 'probed'
                    THEN COALESCE(duration_seconds, 0) ELSE 0 END
                ) AS probed_seconds
            FROM media_files
            WHERE camera_id = ?
              AND capture_start_ts < ?
              AND COALESCE(
                    capture_end_ts,
                    capture_start_ts + COALESCE(duration_seconds, 1)
                  ) >= ?
            """,
            (camera_id, until_ts, since_ts),
        ).fetchone()
        keyframes = connection.execute(
            """
            SELECT
                COUNT(*) AS frames,
                SUM(
                    CASE WHEN k.wall_time_source IN (
                        'capture_clock', 'watermark'
                    ) THEN 1 ELSE 0 END
                ) AS precise_frames
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            WHERE m.camera_id = ?
              AND k.wall_ts >= ?
              AND k.wall_ts < ?
            """,
            (camera_id, since_ts, until_ts),
        ).fetchone()
        analyzed_files = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM processing_jobs j
                JOIN media_files m ON m.id = j.media_file_id
                WHERE j.job_type = 'visual_embedding'
                  AND j.status = 'completed'
                  AND m.camera_id = ?
                  AND m.capture_start_ts < ?
                  AND COALESCE(
                        m.capture_end_ts,
                        m.capture_start_ts
                        + COALESCE(m.duration_seconds, 1)
                      ) >= ?
                """,
                (camera_id, until_ts, since_ts),
            ).fetchone()[0]
        )
        is_continuous = camera_id == "B88880620430"
        intervals: list[tuple[float, float]] = []
        if is_continuous:
            for row in connection.execute(
                """
                SELECT t.wall_start_ts, t.wall_end_ts
                FROM time_segments t
                JOIN media_files m ON m.id = t.media_file_id
                WHERE m.camera_id = ?
                  AND t.wall_start_ts < ?
                  AND COALESCE(t.wall_end_ts, t.wall_start_ts) >= ?
                """,
                (camera_id, until_ts, since_ts),
            ):
                intervals.append(
                    (
                        float(row["wall_start_ts"]),
                        float(
                            row["wall_end_ts"]
                            if row["wall_end_ts"] is not None
                            else row["wall_start_ts"]
                        ),
                    )
                )
        else:
            for row in connection.execute(
                """
                SELECT
                    capture_start_ts,
                    capture_start_ts + COALESCE(duration_seconds, 1)
                        AS end_ts
                FROM media_files
                WHERE camera_id = ?
                  AND capture_start_ts < ?
                  AND capture_start_ts
                      + COALESCE(duration_seconds, 1) >= ?
                """,
                (camera_id, until_ts, since_ts),
            ):
                intervals.append(
                    (float(row["capture_start_ts"]), float(row["end_ts"]))
                )
        observed_seconds = _merge_intervals(
            intervals,
            since_ts=since_ts,
            until_ts=until_ts,
        )
        files = int(media["files"] or 0)
        stable_files = int(media["stable_files"] or 0)
        frames = int(keyframes["frames"] or 0)
        precise_frames = int(keyframes["precise_frames"] or 0)
        coverage.append(
            {
                "camera_id": camera_id,
                "name": str(camera["name"]),
                "location": str(camera["location"]),
                "kind": str(camera["kind"]),
                "coverage_mode": (
                    "continuous_recording"
                    if is_continuous
                    else "motion_triggered_clips"
                ),
                "files": files,
                "stable_files": stable_files,
                "unstable_files": int(media["unstable_files"] or 0),
                "source_error_files": int(
                    media["source_error_files"] or 0
                ),
                "probed_files": int(media["probed_files"] or 0),
                "analyzed_files": analyzed_files,
                "analysis_file_percent": (
                    round(analyzed_files * 100.0 / stable_files, 2)
                    if stable_files
                    else 0.0
                ),
                "keyframes": frames,
                "precise_time_percent": (
                    round(precise_frames * 100.0 / frames, 2)
                    if frames
                    else 0.0
                ),
                "observed_seconds": observed_seconds,
                "recording_coverage_percent": (
                    round(observed_seconds * 100.0 / period_seconds, 2)
                    if is_continuous
                    else None
                ),
                "first_ts": media["first_ts"],
                "last_ts": media["last_ts"],
                "negative_evidence_valid": is_continuous,
                "notice": (
                    "该视角近似连续录像；缺失时段仍标为未知。"
                    if is_continuous
                    else "该视角由动作触发；没有片段不代表没有发生。"
                ),
            }
        )
    return coverage


def query_toy_ranking(
    connection: sqlite3.Connection,
    *,
    since_ts: float,
    until_ts: float,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            t.id,
            t.display_name,
            t.toy_type,
            COUNT(DISTINCT l.activity_event_id) AS interaction_count,
            SUM(
                CASE
                    WHEN r.corrected_start_ts IS NOT NULL
                      OR r.corrected_end_ts IS NOT NULL
                    THEN MAX(
                        0.0,
                        COALESCE(r.corrected_end_ts, e.end_ts)
                        - COALESCE(r.corrected_start_ts, e.start_ts)
                    )
                    ELSE e.duration_seconds
                END
            ) AS interaction_seconds,
            MAX(l.confidence) AS confidence,
            (
                SELECT ev.keyframe_id
                FROM toy_event_links l2
                JOIN activity_event_evidence ev
                  ON ev.activity_event_id = l2.activity_event_id
                JOIN activity_events e2
                  ON e2.id = l2.activity_event_id
                LEFT JOIN activity_event_reviews r2
                  ON r2.activity_event_id = e2.id
                WHERE l2.toy_entity_id = t.id
                  AND COALESCE(
                        r2.corrected_start_ts,
                        e2.start_ts
                      ) < ?
                  AND COALESCE(
                        r2.corrected_end_ts,
                        e2.end_ts
                      ) >= ?
                  AND COALESCE(
                        r2.verdict,
                        e2.review_status
                      ) != 'rejected'
                  AND COALESCE(
                        r2.corrected_activity_type,
                        e2.activity_type
                      ) = 'play'
                ORDER BY l2.confidence DESC, ev.evidence_rank
                LIMIT 1
            ) AS keyframe_id
        FROM toy_entities t
        JOIN toy_event_links l ON l.toy_entity_id = t.id
        JOIN activity_events e ON e.id = l.activity_event_id
        LEFT JOIN activity_event_reviews r
          ON r.activity_event_id = e.id
        WHERE COALESCE(r.corrected_start_ts, e.start_ts) < ?
          AND COALESCE(r.corrected_end_ts, e.end_ts) >= ?
          AND COALESCE(r.verdict, e.review_status) != 'rejected'
          AND COALESCE(
                r.corrected_activity_type,
                e.activity_type
              ) = 'play'
        GROUP BY t.id, t.display_name, t.toy_type
        ORDER BY interaction_seconds DESC, interaction_count DESC
        """,
        (until_ts, since_ts, until_ts, since_ts),
    )
    return [
        {
            "id": int(row["id"]),
            "name": str(row["display_name"]),
            "toy_type": str(row["toy_type"]),
            "is_specific": str(row["toy_type"]) != "other_toy",
            "interaction_count": int(row["interaction_count"]),
            "interaction_seconds": float(row["interaction_seconds"] or 0),
            "confidence": float(row["confidence"] or 0),
            "keyframe_id": (
                int(row["keyframe_id"])
                if row["keyframe_id"] is not None
                else None
            ),
            "image_url": (
                f"/api/keyframes/{int(row['keyframe_id'])}/image"
                if row["keyframe_id"] is not None
                else None
            ),
            "meaning": (
                "可见互动时长排序，不代表心理偏好；"
                "未细分类别不参与最爱玩具判断"
            ),
        }
        for row in rows
    ]


def build_feeding_routine_reference(
    feeding: Sequence[Mapping[str, Any]],
    *,
    since_ts: float,
    until_ts: float,
    target_interval_seconds: float = 3 * 60 * 60,
    episode_merge_seconds: float = 75 * 60,
    routine_start_ts: float | None = None,
    routine_end_ts: float | None = None,
    reference_now_ts: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn visible feeding sightings into bounded episodes and a routine estimate.

    A routine estimate is deliberately separate from camera evidence. Multiple
    bottle contacts close together can be a paused or topped-up feed, while
    missing routine slots remain *likely unobserved* rather than fabricated
    activity events.
    """

    interval = max(60.0, float(target_interval_seconds))
    merge_window = min(
        max(0.0, float(episode_merge_seconds)),
        interval * 0.49,
    )
    ordered = sorted(
        feeding,
        key=lambda item: float(item["start_ts"]),
    )
    clusters: list[list[Mapping[str, Any]]] = []
    for sighting in ordered:
        timestamp = float(sighting["start_ts"])
        if (
            not clusters
            or timestamp - float(clusters[-1][0]["start_ts"])
            > merge_window
        ):
            clusters.append([sighting])
        else:
            clusters[-1].append(sighting)

    episodes: list[dict[str, Any]] = []
    for cluster in clusters:
        timestamps = [float(item["start_ts"]) for item in cluster]
        confirmed = sum(
            str(item.get("certainty") or "uncertain") == "confirmed"
            for item in cluster
        )
        probable = sum(
            str(item.get("certainty") or "uncertain") == "probable"
            for item in cluster
        )
        locations = list(
            dict.fromkeys(
                str(item["location"])
                for item in cluster
                if item.get("location")
            )
        )
        episodes.append(
            {
                "start_ts": min(timestamps),
                "last_sighting_ts": max(timestamps),
                "reference_ts": sum(timestamps) / len(timestamps),
                "sighting_count": len(cluster),
                "confirmed_sightings": confirmed,
                "probable_sightings": probable,
                "certainty": (
                    "confirmed"
                    if confirmed
                    else "probable"
                    if probable
                    else "uncertain"
                ),
                "locations": locations,
                "evidence_points": sum(
                    int(item.get("evidence_points") or 0)
                    for item in cluster
                ),
            }
        )

    now_ts = time.time() if reference_now_ts is None else reference_now_ts
    if now_ts < since_ts:
        effective_until_ts = since_ts
    elif now_ts < until_ts:
        effective_until_ts = now_ts
    else:
        effective_until_ts = until_ts
    routine_window_start = max(
        since_ts,
        float(routine_start_ts)
        if routine_start_ts is not None
        else since_ts,
    )
    routine_window_end = min(
        effective_until_ts,
        float(routine_end_ts)
        if routine_end_ts is not None
        else effective_until_ts,
    )
    routine_window_end = max(routine_window_start, routine_window_end)
    effective_seconds = routine_window_end - routine_window_start
    routine_episode_indexes = {
        index
        for index, item in enumerate(episodes)
        if routine_window_start
        <= float(item["reference_ts"])
        < routine_window_end
    }
    routine_episodes = [
        episodes[index] for index in sorted(routine_episode_indexes)
    ]
    episode_intervals = [
        float(right["reference_ts"]) - float(left["reference_ts"])
        for left, right in zip(routine_episodes, routine_episodes[1:])
    ]

    expected_slots: list[float] = []
    phase_offset_seconds: float | None = None
    if routine_episodes and effective_seconds > 0:
        phases = [
            (
                float(item["reference_ts"])
                - routine_window_start
            )
            % interval
            for item in routine_episodes
        ]
        angles = [2.0 * math.pi * phase / interval for phase in phases]
        x = sum(math.cos(angle) for angle in angles)
        y = sum(math.sin(angle) for angle in angles)
        if math.hypot(x, y) < 1e-6:
            phase_offset_seconds = sorted(phases)[len(phases) // 2]
        else:
            angle = math.atan2(y, x) % (2.0 * math.pi)
            phase_offset_seconds = angle * interval / (2.0 * math.pi)
        slot = routine_window_start + phase_offset_seconds
        while slot < routine_window_end:
            expected_slots.append(slot)
            slot += interval

    routine_count = (
        len(expected_slots)
        if expected_slots
        else int(math.floor(effective_seconds / interval + 0.5))
    )
    unused_episode_indexes = set(routine_episode_indexes)
    schedule: list[dict[str, Any]] = []
    for slot in expected_slots:
        nearest_index = min(
            unused_episode_indexes,
            key=lambda index: abs(
                float(episodes[index]["reference_ts"]) - slot
            ),
            default=None,
        )
        nearest_distance = (
            abs(float(episodes[nearest_index]["reference_ts"]) - slot)
            if nearest_index is not None
            else None
        )
        if nearest_index is not None and nearest_distance <= merge_window:
            unused_episode_indexes.remove(nearest_index)
            schedule.append(
                {
                    "expected_ts": round(slot, 3),
                    "status": "observed",
                    "episode_reference_ts": round(
                        float(episodes[nearest_index]["reference_ts"]),
                        3,
                    ),
                }
            )
        else:
            schedule.append(
                {
                    "expected_ts": round(slot, 3),
                    "status": "likely_unobserved",
                    "episode_reference_ts": None,
                }
            )

    likely_unobserved_count = (
        sum(item["status"] == "likely_unobserved" for item in schedule)
        if schedule
        else routine_count
    )
    estimated_count = len(episodes) + likely_unobserved_count

    average_interval = (
        sum(episode_intervals) / len(episode_intervals)
        if episode_intervals
        else None
    )
    sorted_intervals = sorted(episode_intervals)
    median_interval = None
    if sorted_intervals:
        middle = len(sorted_intervals) // 2
        median_interval = (
            sorted_intervals[middle]
            if len(sorted_intervals) % 2
            else (
                sorted_intervals[middle - 1]
                + sorted_intervals[middle]
            )
            / 2.0
        )
    average_deviation = (
        sum(abs(value - interval) for value in episode_intervals)
        / len(episode_intervals)
        if episode_intervals
        else None
    )
    cadence_fit = (
        "strong"
        if len(episode_intervals) >= 2
        and average_deviation is not None
        and average_deviation <= 30 * 60
        else "moderate"
        if episode_intervals
        and average_deviation is not None
        and average_deviation <= 60 * 60
        else "insufficient"
    )

    return {
        "target_interval_seconds": round(interval, 3),
        "episode_merge_seconds": round(merge_window, 3),
        "effective_period_seconds": round(effective_seconds, 3),
        "routine_window_start_ts": round(routine_window_start, 3),
        "routine_window_end_ts": round(routine_window_end, 3),
        "routine_window_source": (
            "sleep_boundaries"
            if routine_start_ts is not None and routine_end_ts is not None
            else "period_bounds"
        ),
        "nighttime_inference_enabled": False,
        "estimated_sessions": estimated_count,
        "observed_sessions": len(episodes),
        "observed_routine_sessions": len(routine_episodes),
        "observed_outside_routine_sessions": (
            len(episodes) - len(routine_episodes)
        ),
        "likely_unobserved_sessions": likely_unobserved_count,
        "phase_offset_seconds": (
            round(phase_offset_seconds, 3)
            if phase_offset_seconds is not None
            else None
        ),
        "average_observed_interval_seconds": (
            round(average_interval, 3)
            if average_interval is not None
            else None
        ),
        "median_observed_interval_seconds": (
            round(median_interval, 3)
            if median_interval is not None
            else None
        ),
        "longest_observed_interval_seconds": (
            round(max(episode_intervals), 3)
            if episode_intervals
            else None
        ),
        "average_target_deviation_seconds": (
            round(average_deviation, 3)
            if average_deviation is not None
            else None
        ),
        "cadence_fit": cadence_fit,
        "schedule": schedule,
        "likely_unobserved_slots": [
            item["expected_ts"]
            for item in schedule
            if item["status"] == "likely_unobserved"
        ],
    }, episodes


def build_family_reference(
    connection: sqlite3.Connection,
    *,
    sessions: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    since_ts: float,
    until_ts: float,
    timezone: ZoneInfo,
    feeding_interval_seconds: float = 3 * 60 * 60,
    feeding_episode_merge_seconds: float = 75 * 60,
) -> dict[str, Any]:
    """Build decision-useful, explicitly bounded feeding and sleep facts."""

    feeding = sorted(
        _sessions_of_type(sessions, "feeding"),
        key=lambda item: float(item["start_ts"]),
    )
    sleep = sorted(
        _sessions_of_type(sessions, "inferred_sleep"),
        key=lambda item: float(item["start_ts"]),
    )

    def verification_counts(task: str, prompt_version: str) -> dict[str, int]:
        return {
            str(row["label"]): int(row["items"])
            for row in connection.execute(
                """
                SELECT v.label, COUNT(*) AS items
                FROM visual_frame_verifications v
                JOIN visual_keyframes k ON k.id = v.keyframe_id
                WHERE v.task = ?
                  AND v.prompt_version = ?
                  AND k.wall_ts >= ?
                  AND k.wall_ts < ?
                GROUP BY v.label
                """,
                (task, prompt_version, since_ts, until_ts),
            )
        }

    feeding_checks = verification_counts(
        "feeding", TEMPORAL_FEEDING_PROMPT_VERSION
    )
    sleep_checks = verification_counts(
        "sleep_presence", SLEEP_PRESENCE_PROMPT_VERSION
    )
    feeding_intervals = [
        max(
            0.0,
            float(right["start_ts"]) - float(left["start_ts"]),
        )
        for left, right in zip(feeding, feeding[1:])
    ]
    bedside = next(
        (
            dict(item)
            for item in coverage
            if item.get("coverage_mode") == "continuous_recording"
        ),
        {},
    )
    period_seconds = max(1.0, until_ts - since_ts)
    recording_coverage = float(
        bedside.get("recording_coverage_percent") or 0.0
    )
    recording_gap_seconds = max(
        0.0,
        period_seconds - float(bedside.get("observed_seconds") or 0.0),
    )
    sleep_seconds = sum(float(item["visible_seconds"]) for item in sleep)
    boundary_errors = [
        float(value)
        for item in sleep
        for value in (
            item.get("start_boundary_error_seconds"),
            item.get("end_boundary_error_seconds"),
        )
        if value is not None
    ]
    publish_sleep = bool(
        sleep
        and recording_coverage >= 80.0
        and sleep_checks.get("BABY_SLEEPING", 0) >= 2
    )
    morning_windows = [
        item
        for item in sleep
        if datetime.fromtimestamp(
            float(item["start_ts"]), timezone
        ).hour < 12
    ]
    evening_windows = [
        item
        for item in sleep
        if datetime.fromtimestamp(
            float(item["start_ts"]), timezone
        ).hour >= 18
    ]
    morning_wake_ts = (
        max(float(item["end_ts"]) for item in morning_windows)
        if morning_windows
        else None
    )
    evening_sleep_start_ts = (
        min(float(item["start_ts"]) for item in evening_windows)
        if evening_windows
        else None
    )
    feeding_routine, feeding_episodes = build_feeding_routine_reference(
        feeding,
        since_ts=since_ts,
        until_ts=until_ts,
        target_interval_seconds=feeding_interval_seconds,
        episode_merge_seconds=feeding_episode_merge_seconds,
        routine_start_ts=morning_wake_ts,
        routine_end_ts=evening_sleep_start_ts,
    )
    return {
        "feeding": {
            "measurement_mode": "camera_sightings_plus_routine_estimate",
            "confirmed_sessions": sum(
                item.get("certainty") == "confirmed"
                for item in feeding_episodes
            ),
            "probable_sessions": sum(
                item.get("certainty") == "probable"
                for item in feeding_episodes
            ),
            "visible_sessions": len(feeding_episodes),
            "confirmed_sightings": sum(
                item.get("certainty") == "confirmed" for item in feeding
            ),
            "probable_sightings": sum(
                item.get("certainty") == "probable" for item in feeding
            ),
            "visible_sightings": len(feeding),
            "duration_available": False,
            "sightings": [
                {
                    "start_ts": float(item["start_ts"]),
                    "certainty": str(
                        item.get("certainty") or "uncertain"
                    ),
                    "subtype": item.get("activity_subtype"),
                    "subtype_name": item.get("subtype_name"),
                    "location": item.get("location"),
                    "evidence_points": int(
                        item.get("evidence_points") or 0
                    ),
                }
                for item in feeding
            ],
            "episodes": feeding_episodes,
            "routine_estimate": feeding_routine,
            "average_sighting_interval_seconds": (
                round(sum(feeding_intervals) / len(feeding_intervals), 3)
                if feeding_intervals
                else None
            ),
            "longest_sighting_gap_seconds": (
                round(max(feeding_intervals), 3)
                if feeding_intervals
                else None
            ),
            "native_sequences_checked": sum(feeding_checks.values()),
            "verification_labels": feeding_checks,
            "count_is_lower_bound": True,
            "reference": (
                "监控目击先按家庭约三小时一次的作息合并为顿；"
                "规律估算与监控确认严格分开，不能据此判断奶量或摄入量。"
            ),
        },
        "sleep": {
            "measurement_mode": "bounded_state_timeline",
            "sessions": len(sleep),
            "estimated_seconds": round(sleep_seconds, 3),
            "duration_publishable": publish_sleep,
            "recording_coverage_percent": recording_coverage,
            "recording_gap_seconds": round(recording_gap_seconds, 3),
            "verified_sleep_points": sleep_checks.get(
                "BABY_SLEEPING", 0
            ),
            "verified_awake_points": sleep_checks.get("NOT_SLEEPING", 0),
            "unclear_points": sleep_checks.get("UNCLEAR", 0),
            "max_boundary_error_seconds": (
                round(max(boundary_errors), 3)
                if boundary_errors
                else None
            ),
            "morning_wake_ts": (
                morning_wake_ts
            ),
            "evening_sleep_start_ts": (
                evening_sleep_start_ts
            ),
            "windows": [
                {
                    "start_ts": float(item["start_ts"]),
                    "end_ts": float(item["end_ts"]),
                    "estimated_seconds": float(item["visible_seconds"]),
                    "start_boundary_error_seconds": item.get(
                        "start_boundary_error_seconds"
                    ),
                    "end_boundary_error_seconds": item.get(
                        "end_boundary_error_seconds"
                    ),
                    "evidence_points": int(
                        item.get("evidence_points") or 0
                    ),
                }
                for item in sleep
            ],
            "reference": (
                "可用于回看入睡、夜间清醒和晨醒的大致时间；"
                "不做睡眠分期或健康诊断。"
            ),
        },
    }


def build_diary_payload(
    connection: sqlite3.Connection,
    *,
    date: str,
    period: str,
    timezone: ZoneInfo,
    feeding_interval_seconds: float = 3 * 60 * 60,
    feeding_episode_merge_seconds: float = 75 * 60,
) -> dict[str, Any]:
    start, end = period_bounds(date, timezone, period)
    since_ts = start.timestamp()
    until_ts = end.timestamp()
    events = [
        clipped
        for event in query_activity_events(
            connection,
            since_ts=since_ts,
            until_ts=until_ts,
        )
        if (
            clipped := _clip_event_to_period(
                event,
                since_ts=since_ts,
                until_ts=until_ts,
            )
        )
        is not None
    ]
    life_sessions = build_life_sessions(events)
    summary: dict[str, dict[str, Any]] = {
        activity_type: {
            "activity_type": activity_type,
            "name": name,
            "count": 0,
            "duration_seconds": 0.0,
            "average_confidence": 0.0,
            "estimated_events": 0,
            "evidence_events": 0,
        }
        for activity_type, name in ACTIVITY_NAMES.items()
    }
    confidence_sums: dict[str, float] = defaultdict(float)
    for event in events:
        item = summary.setdefault(
            event["activity_type"],
            {
                "activity_type": event["activity_type"],
                "name": event["activity_name"],
                "count": 0,
                "duration_seconds": 0.0,
                "average_confidence": 0.0,
                "estimated_events": 0,
                "evidence_events": 0,
            },
        )
        item["count"] += 1
        item["duration_seconds"] += event["duration_seconds"]
        item["estimated_events"] += int(
            event["time_precision"] == "estimated"
        )
        item["evidence_events"] += int(
            event["evidence_keyframe_id"] is not None
        )
        confidence_sums[event["activity_type"]] += event["confidence"]
    for activity_type, item in summary.items():
        if item["count"]:
            item["average_confidence"] = round(
                confidence_sums[activity_type] / item["count"],
                4,
            )
        item["duration_seconds"] = round(item["duration_seconds"], 3)
        sessions = _sessions_of_type(life_sessions, activity_type)
        item["event_count"] = item["count"]
        item["session_count"] = len(sessions)
        item["session_duration_seconds"] = round(
            sum(float(session["visible_seconds"]) for session in sessions),
            3,
        )
    toys = query_toy_ranking(
        connection,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    life_brief = build_life_brief(
        life_sessions,
        toys=toys,
        source_event_count=len(events),
    )
    coverage = query_coverage(
        connection,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    family_reference = build_family_reference(
        connection,
        sessions=life_sessions,
        coverage=coverage,
        since_ts=since_ts,
        until_ts=until_ts,
        timezone=timezone,
        feeding_interval_seconds=feeding_interval_seconds,
        feeding_episode_merge_seconds=feeding_episode_merge_seconds,
    )
    analyzed_files = sum(item["analyzed_files"] for item in coverage)
    source_files = sum(item["files"] for item in coverage)
    stable_source_files = sum(item["stable_files"] for item in coverage)
    unstable_source_files = sum(
        item["unstable_files"] for item in coverage
    )
    source_error_files = sum(
        item["source_error_files"] for item in coverage
    )
    return {
        "timezone": timezone.key,
        "period": period,
        "selected_date": date,
        "start_iso": start.isoformat(),
        "end_iso": end.isoformat(),
        "available_dates": available_diary_dates(
            connection,
            timezone=timezone,
        ),
        "summary": list(summary.values()),
        "events": events,
        "life_sessions": life_sessions,
        "life_brief": life_brief,
        "family_reference": family_reference,
        "daily_series": build_daily_series(
            life_sessions,
            start=start,
            end=end,
        ),
        "highlights": select_life_highlights(life_sessions),
        "review_queue": select_review_queue(life_sessions),
        "toys": toys,
        "coverage": coverage,
        "analysis": {
            "source_files": source_files,
            "stable_source_files": stable_source_files,
            "unstable_source_files": unstable_source_files,
            "source_error_files": source_error_files,
            "analyzed_files": analyzed_files,
            "analysis_file_percent": (
                round(analyzed_files * 100.0 / stable_source_files, 2)
                if stable_source_files
                else 0.0
            ),
            "generated_at": int(time.time()),
            "medical_claim": False,
            "face_recognition": False,
            "cloud_upload": False,
        },
        "notices": [
            "喝奶只保留本地 Qwen3-VL-8B 回读 NAS 原始录像后，在连续"
            "多帧中看到喂养器具与嘴部接触的行为；明确证据与可能证据"
            "分开显示。75 分钟内的多次目击按家庭作息合并为一顿；约三"
            "小时规律仅在晨醒至晚间入睡之间估算，睡着后不补次数；推测"
            "不冒充监控确认，也无法推断奶量。",
            "睡眠为床内低运动候选经原始录像确认闭眼或明确放松睡姿后的"
            "推定；仅躺床不算睡眠。睡着、清醒、不确定锚点共同切分"
            "时间轴，页面会显示本日实际边界误差与录像缺口，不是睡眠分期。",
            "玩具类别来自原始录像多帧中的可见互动；“最常互动”只比较"
            "已识别类别的可见次数与时长，未细分类别不参与判断，"
            "也不代表心理偏好。",
            "爬行与坐姿时长是原片连续多帧确认后的可见姿态下限，"
            "不代表宝宝在整段时间持续移动。",
            "动作触发摄像头没有片段时属于未知，不等于没有活动。",
        ]
        + (
            [
                f"{unstable_source_files} 个文件仍在写入，"
                "不计入本次稳定文件完成率。"
            ]
            if unstable_source_files
            else []
        )
        + (
            [
                f"{source_error_files} 个源视频损坏或不可解码，"
                "已作为数据缺口单独记录。"
            ]
            if source_error_files
            else []
        ),
    }


def query_activity_detail(
    connection: sqlite3.Connection,
    event_id: int,
) -> dict[str, Any] | None:
    rows = query_activity_events(
        connection,
        since_ts=0,
        until_ts=32_503_680_000,
        include_rejected=True,
        event_id=event_id,
    )
    event = rows[0] if rows else None
    if event is None:
        return None
    evidence = []
    for row in connection.execute(
        """
        SELECT
            ev.evidence_rank,
            ev.score,
            ev.role,
            k.id AS keyframe_id,
            k.wall_ts,
            k.wall_time_source,
            k.media_offset_ms,
            k.relative_image_path,
            m.id AS media_file_id,
            m.relative_path,
            m.mtime_ns,
            m.duration_seconds AS media_duration_seconds,
            m.camera_id,
            c.name AS camera_name
        FROM activity_event_evidence ev
        JOIN visual_keyframes k ON k.id = ev.keyframe_id
        JOIN media_files m ON m.id = k.media_file_id
        JOIN cameras c ON c.id = m.camera_id
        WHERE ev.activity_event_id = ?
        ORDER BY ev.evidence_rank
        """,
        (event_id,),
    ):
        evidence.append(
            {
                **dict(row),
                "image_url": (
                    f"/api/keyframes/{int(row['keyframe_id'])}/image"
                ),
                "media_offset_seconds": (
                    float(row["media_offset_ms"]) / 1000.0
                ),
            }
        )
    event["evidence"] = evidence
    return event


def save_activity_review(
    connection: sqlite3.Connection,
    *,
    event_id: int,
    verdict: str,
    corrected_activity_type: str | None,
    corrected_activity_subtype: str | None,
    corrected_start_ts: float | None,
    corrected_end_ts: float | None,
    note: str,
    reviewer: str = "family",
) -> None:
    if verdict not in {"confirmed", "corrected", "rejected", "uncertain"}:
        raise ValueError("Unsupported activity verdict")
    if (
        corrected_start_ts is not None
        and corrected_end_ts is not None
        and corrected_end_ts <= corrected_start_ts
    ):
        raise ValueError("Corrected end time must be after start time")
    exists = connection.execute(
        "SELECT 1 FROM activity_events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if exists is None:
        raise LookupError("Activity event does not exist")
    now_ts = int(time.time())
    connection.execute(
        """
        INSERT INTO activity_event_reviews(
            activity_event_id, verdict,
            corrected_activity_type, corrected_activity_subtype,
            corrected_start_ts, corrected_end_ts,
            note, reviewer, reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(activity_event_id) DO UPDATE SET
            verdict = excluded.verdict,
            corrected_activity_type = excluded.corrected_activity_type,
            corrected_activity_subtype = excluded.corrected_activity_subtype,
            corrected_start_ts = excluded.corrected_start_ts,
            corrected_end_ts = excluded.corrected_end_ts,
            note = excluded.note,
            reviewer = excluded.reviewer,
            reviewed_at = excluded.reviewed_at
        """,
        (
            event_id,
            verdict,
            corrected_activity_type,
            corrected_activity_subtype,
            corrected_start_ts,
            corrected_end_ts,
            note,
            reviewer,
            now_ts,
        ),
    )
    connection.execute(
        """
        UPDATE activity_events
        SET review_status = ?, updated_at = ?
        WHERE id = ?
        """,
        (verdict, now_ts, event_id),
    )
    connection.commit()
