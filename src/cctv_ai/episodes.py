from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping


def _event_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    start = float(row["wall_start_ts"])
    end = float(row["wall_end_ts"] or start)
    return {
        "event_id": int(row["event_id"]),
        "event_label": str(row["event_label"]),
        "model_label": str(row["model_label"]),
        "start_ts": start,
        "end_ts": end,
        "confidence": float(row["confidence"]),
        "media_start_seconds": float(row["media_start_seconds"]),
        "media_end_seconds": float(row["media_end_seconds"]),
        "media_file_id": int(row["media_file_id"]),
        "camera_id": str(row["camera_id"]),
        "camera_name": str(row["camera_name"]),
        "location": str(row["location"]),
        "relative_path": str(row["relative_path"]),
        "review_verdict": row["review_verdict"],
        "review_note": row["review_note"] or "",
        "reviewer": row["reviewer"],
        "reviewed_at": row["reviewed_at"],
    }


def episode_identifier(label: str, event_ids: Iterable[int]) -> str:
    material = f"{label}:" + ",".join(
        str(event_id) for event_id in sorted(event_ids)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _finalize_episode(events: list[dict[str, Any]]) -> dict[str, Any]:
    label = events[0]["event_label"]
    event_ids = [event["event_id"] for event in events]
    representative = max(
        events,
        key=lambda event: (
            event["confidence"],
            event["end_ts"] - event["start_ts"],
        ),
    )
    camera_map: dict[str, str] = {}
    for event in events:
        camera_map[event["camera_id"]] = event["camera_name"]

    reviewed = [
        event for event in events if event["review_verdict"] is not None
    ]
    verdicts = {event["review_verdict"] for event in reviewed}
    if not reviewed:
        review_status = "pending"
    elif len(reviewed) == len(events) and len(verdicts) == 1:
        review_status = str(reviewed[0]["review_verdict"])
    else:
        review_status = "mixed"

    latest_review = max(
        reviewed,
        key=lambda event: int(event["reviewed_at"] or 0),
        default=None,
    )
    start = min(event["start_ts"] for event in events)
    end = max(event["end_ts"] for event in events)
    return {
        "episode_id": episode_identifier(label, event_ids),
        "event_label": label,
        "start_ts": start,
        "end_ts": end,
        "duration_seconds": max(0.0, end - start),
        "event_count": len(events),
        "max_confidence": max(event["confidence"] for event in events),
        "event_ids": event_ids,
        "cameras": [
            {"id": camera_id, "name": name}
            for camera_id, name in camera_map.items()
        ],
        "review_status": review_status,
        "review_note": (
            latest_review["review_note"] if latest_review is not None else ""
        ),
        "reviewed_at": (
            latest_review["reviewed_at"] if latest_review is not None else None
        ),
        "representative_event": representative,
        "events": events,
    }


def build_episodes(
    rows: Iterable[Mapping[str, Any]],
    *,
    gap_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    if gap_seconds < 0:
        raise ValueError("gap_seconds cannot be negative")
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event = _event_dict(row)
        by_label[event["event_label"]].append(event)

    episodes: list[dict[str, Any]] = []
    for events in by_label.values():
        ordered = sorted(
            events,
            key=lambda event: (event["start_ts"], event["event_id"]),
        )
        current: list[dict[str, Any]] = []
        current_end = 0.0
        for event in ordered:
            if (
                current
                and event["start_ts"] > current_end + gap_seconds
            ):
                episodes.append(_finalize_episode(current))
                current = []
            current.append(event)
            current_end = max(current_end, event["end_ts"])
        if current:
            episodes.append(_finalize_episode(current))

    return sorted(
        episodes,
        key=lambda episode: episode["start_ts"],
        reverse=True,
    )


def _stable_episode_rank(episode: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        str(episode["episode_id"]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _priority_key(episode: Mapping[str, Any]) -> tuple[float, int]:
    camera_count = len(episode["cameras"])
    event_count = int(episode["event_count"])
    duration = float(episode["duration_seconds"])
    confidence = float(episode["max_confidence"])
    score = (
        (8.0 if camera_count >= 2 else 0.0)
        + min(event_count, 8) * 0.65
        + min(duration, 60.0) / 20.0
        + confidence * 4.0
    )
    return (-score, _stable_episode_rank(episode))


def _learning_key(episode: Mapping[str, Any]) -> tuple[float, int, int]:
    confidence = float(episode["max_confidence"])
    camera_penalty = max(0, len(episode["cameras"]) - 1)
    event_penalty = max(0, int(episode["event_count"]) - 1)
    return (
        abs(confidence - 0.72) + camera_penalty * 0.08,
        event_penalty,
        _stable_episode_rank(episode),
    )


def _priority_reason(episode: Mapping[str, Any]) -> str:
    if len(episode["cameras"]) >= 2:
        return "多视角同时命中"
    if (
        int(episode["event_count"]) >= 3
        or float(episode["duration_seconds"]) >= 10
    ):
        return "连续命中，优先确认"
    if float(episode["max_confidence"]) >= 0.90:
        return "高置信度候选"
    return "高价值候选"


def select_review_inbox(
    episodes: Iterable[Mapping[str, Any]],
    *,
    quota: int = 10,
) -> list[dict[str, Any]]:
    """Select a small, deterministic review roster without using verdicts.

    The roster intentionally ignores ``review_status``. Reviewing an item
    therefore updates it in place instead of causing another pending item to
    refill the inbox.
    """
    if quota < 1:
        raise ValueError("quota must be at least 1")

    candidates = [dict(episode) for episode in episodes]
    if not candidates:
        return []
    limit = min(quota, len(candidates))
    audit_target = min(
        max(0, limit - 1),
        max(1, int(limit * 0.2 + 0.5)) if limit >= 3 else 0,
    )
    learning_target = min(
        limit - audit_target,
        int(limit * 0.3 + 0.5),
    )
    priority_target = limit - learning_target - audit_target

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(
        episode: Mapping[str, Any],
        *,
        lane: str,
        reason: str,
    ) -> None:
        episode_id = str(episode["episode_id"])
        if episode_id in selected_ids:
            return
        item = dict(episode)
        item["review_lane"] = lane
        item["review_reason"] = reason
        selected_ids.add(episode_id)
        selected.append(item)

    priority_order = sorted(candidates, key=_priority_key)
    for episode in priority_order[:priority_target]:
        add(
            episode,
            lane="priority",
            reason=_priority_reason(episode),
        )

    learning_order = sorted(
        (
            episode
            for episode in candidates
            if str(episode["episode_id"]) not in selected_ids
        ),
        key=_learning_key,
    )
    for episode in learning_order[:learning_target]:
        add(
            episode,
            lane="learning",
            reason="模型边界样本",
        )

    audit_order = sorted(
        (
            episode
            for episode in candidates
            if str(episode["episode_id"]) not in selected_ids
        ),
        key=_stable_episode_rank,
    )
    for episode in audit_order[:audit_target]:
        add(
            episode,
            lane="audit",
            reason="质量抽检",
        )

    if len(selected) < limit:
        for episode in priority_order:
            if len(selected) >= limit:
                break
            add(
                episode,
                lane="priority",
                reason=_priority_reason(episode),
            )

    for rank, episode in enumerate(selected, start=1):
        episode["review_rank"] = rank
    return selected


def select_informative_review_inbox(
    episodes: Iterable[Mapping[str, Any]],
    *,
    shadow_scores: Mapping[str, Mapping[str, Any]],
    quota: int = 5,
    candidate_pool_size: int = 10,
) -> list[dict[str, Any]]:
    """Select boundary examples plus two directional audit examples.

    Scores remain advisory: this function only chooses what a person sees and
    never turns a shadow score into a verdict. The candidate pool stays based
    on the pre-existing deterministic roster so reducing 10 items to 5 does
    not expose an unrelated backlog.
    """
    if quota < 1:
        raise ValueError("quota must be at least 1")
    candidates = list(episodes)
    if not candidates:
        return []
    pool = select_review_inbox(
        candidates,
        quota=max(quota, candidate_pool_size),
    )
    scored = [
        (episode, shadow_scores[str(episode["episode_id"])])
        for episode in pool
        if str(episode["episode_id"]) in shadow_scores
    ]
    if not scored:
        return select_review_inbox(candidates, quota=quota)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(
        episode: Mapping[str, Any],
        *,
        lane: str,
        reason: str,
    ) -> bool:
        episode_id = str(episode["episode_id"])
        if episode_id in selected_ids or len(selected) >= quota:
            return False
        selected_ids.add(episode_id)
        selected.append(
            {
                **episode,
                "review_lane": lane,
                "review_reason": reason,
            }
        )
        return True

    boundary_target = quota if quota < 3 else quota - 2
    for episode, _ in sorted(
        scored,
        key=lambda item: (
            -float(item[1]["review_priority"]),
            _stable_episode_rank(item[0]),
        ),
    )[:boundary_target]:
        add(
            episode,
            lane="learning",
            reason="影子边界样本（只供学习）",
        )

    by_score = sorted(
        scored,
        key=lambda item: (
            float(item[1]["score"]),
            _stable_episode_rank(item[0]),
        ),
    )
    if len(selected) < quota:
        for episode, _ in by_score:
            if add(
                episode,
                lane="audit",
                reason="偏误报方向抽检",
            ):
                break
    if len(selected) < quota:
        for episode, _ in reversed(by_score):
            if add(
                episode,
                lane="audit",
                reason="偏哭声方向抽检",
            ):
                break

    for episode in pool:
        if len(selected) >= quota:
            break
        add(
            episode,
            lane=str(episode["review_lane"]),
            reason=str(episode["review_reason"]),
        )

    for rank, episode in enumerate(selected, start=1):
        episode["review_rank"] = rank
    return selected
