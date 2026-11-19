from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .visual_models import (
    SLEEP_PRESENCE_PROMPT_VERSION,
    TEMPORAL_FEEDING_PROMPT_VERSION,
    TEMPORAL_PLAY_PROMPT_VERSION,
    TEMPORAL_POSTURE_PROMPT_VERSION,
)


ACTIVITY_MODEL_VERSION = "temporal-diary-v1"

CAMERA_LOCATIONS = {
    "lumi1.54ef4482e358": ("客厅", "living_room"),
    "lumi3.8e2ce0d05d9c251e": ("宝宝房间", "baby_room"),
    "50ec503b63bf": ("厨房", "kitchen"),
    "B88880620430": ("宝宝床", "baby_bed"),
}

TOY_NAMES = {
    "ball": "球",
    "plush_toy": "毛绒玩具",
    "blocks": "积木",
    "book": "绘本",
    "toy_vehicle": "玩具车",
    "rattle": "摇铃/手持玩具",
    "activity_gym": "健身架/游戏垫",
    "doll": "玩偶",
    "activity_cube": "机关活动方盒",
    "fruit_sorter": "水果配对托盘",
    "stacking_tower": "彩色叠叠塔/滚珠塔",
    "other_toy": "未细分互动玩具",
}

VERIFIED_FEEDING_LABELS = {
    "FEED_BOTTLE": "feeding_bottle",
    "FEED_BREAST": "breastfeeding",
    "FEED_SOLID": "solid_food",
}

VERIFIED_FEEDING_SEMANTIC_FLOORS = {
    "feeding_bottle": 0.18,
    "breastfeeding": 0.20,
    "solid_food": 0.18,
}

VERIFIED_TOY_LABELS = {
    "ACTIVITY_CUBE": "activity_cube",
    "FRUIT_SORTER": "fruit_sorter",
    "STACKING_TOWER": "stacking_tower",
    "BOOK": "book",
    "PLUSH": "plush_toy",
    "BALL": "ball",
    "RATTLE_TEETHER": "rattle",
    "DOLL": "doll",
    "TOY_VEHICLE": "toy_vehicle",
    "BLOCKS": "blocks",
    "ACTIVITY_GYM": "activity_gym",
    "OTHER": "other_toy",
}


@dataclass(frozen=True)
class FrameObservation:
    keyframe_id: int
    media_file_id: int
    camera_id: str
    wall_ts: float
    wall_time_source: str
    location: str
    zone: str
    sample_reason: str
    media_duration: float
    motion_score: float | None
    night_score: float
    person_confidence: float
    scores: Mapping[str, Mapping[str, float]]
    raw_sigmoids: Mapping[str, Mapping[str, float]]
    feeding_container_confidence: float = 0.0
    verified_feeding_label: str | None = None
    verified_feeding_confidence: float = 0.0
    verified_feeding_model_name: str | None = None
    verified_feeding_prompt_version: str | None = None
    verified_feeding_details: Mapping[str, Any] | None = None
    verified_toy_label: str | None = None
    verified_toy_confidence: float = 0.0
    verified_toy_model_name: str | None = None
    verified_toy_prompt_version: str | None = None
    verified_sleep_label: str | None = None
    verified_sleep_confidence: float = 0.0
    verified_sleep_model_name: str | None = None
    verified_sleep_prompt_version: str | None = None
    verified_sleep_details: Mapping[str, Any] | None = None
    verified_posture_label: str | None = None
    verified_posture_confidence: float = 0.0
    verified_posture_model_name: str | None = None
    verified_posture_prompt_version: str | None = None


@dataclass(frozen=True)
class Candidate:
    activity_type: str
    subtype: str | None
    start_ts: float
    end_ts: float
    camera_id: str
    location: str
    zone: str
    confidence: float
    keyframe_id: int
    time_precision: str
    toy_label: str | None = None
    toy_score: float = 0.0
    direct_support: bool = False
    verified_support: bool = False
    verification_available: bool = False
    semantic_strength: float = 0.0
    temporal_verified: bool = False
    measurement_mode: str = "visible_lower_bound"
    certainty: str = "confirmed"
    start_boundary_error_seconds: float | None = None
    end_boundary_error_seconds: float | None = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _time_precision(source: str) -> str:
    if source == "watermark":
        return "mapped"
    if source == "capture_clock":
        return "exact"
    return "estimated"


def _support_seconds(frame: FrameObservation) -> float:
    if frame.sample_reason.startswith("continuous_"):
        try:
            return max(
                5.0,
                float(
                    frame.sample_reason.removeprefix("continuous_").removesuffix(
                        "s"
                    )
                ),
            )
        except ValueError:
            return 30.0
    return min(max(frame.media_duration, 5.0), 30.0)


def _probable_temporal_feeding(
    frame: FrameObservation,
) -> str | None:
    """Recover strong native-video contact evidence conservatively.

    Qwen occasionally marks an otherwise explicit five-frame bottle sequence
    as ``uncertain`` and emits a low numeric confidence.  Dropping those rows
    entirely caused a systematic under-count.  Keep them separate from
    confirmed feeds: the event must still say ``feeding``, name a feeding
    object, and provide at least two explicit, non-negated mouth-contact
    evidence lines from a native-resolution sequence.
    """

    details = dict(frame.verified_feeding_details or {})
    if details.get("native_source") is not True:
        return None
    structured = details.get("structured_event")
    if not isinstance(structured, Mapping):
        return None
    if str(structured.get("activity") or "").lower() != "feeding":
        return None
    objects = {
        str(value).strip().lower()
        for value in structured.get("objects") or ()
        if value
    }
    subtype = (
        "feeding_bottle"
        if objects & {"bottle", "奶瓶"}
        else "breastfeeding"
        if objects & {"breast", "母乳", "乳房"}
        else "solid_food"
        if objects & {"spoon", "food", "勺子", "食物"}
        else None
    )
    if subtype is None:
        return None
    evidence = [
        str(value).strip()
        for value in structured.get("evidence") or ()
        if str(value).strip()
    ]
    if len(evidence) < 2:
        return None
    positive_cues = (
        "奶嘴在宝宝嘴里",
        "奶嘴在婴儿嘴里",
        "奶嘴在嘴里",
        "奶嘴接触",
        "奶瓶与宝宝嘴部接触",
        "奶瓶与婴儿嘴部接触",
        "勺子与宝宝嘴部接触",
        "勺子与婴儿嘴部接触",
        "食物进入嘴",
        "latched",
        "in the baby's mouth",
        "in the infant's mouth",
        "touching the baby's mouth",
        "touching the infant's mouth",
    )
    negative_cues = (
        "未见",
        "没有",
        "未接触",
        "无法确认",
        "not visible",
        "no contact",
        "cannot confirm",
    )
    explicit_lines = [
        value.lower()
        for value in evidence
        if any(cue in value.lower() for cue in positive_cues)
        and not any(cue in value.lower() for cue in negative_cues)
    ]
    return subtype if len(explicit_lines) >= 2 else None


def _top(
    scores: Mapping[str, Mapping[str, float]],
    group: str,
) -> tuple[str | None, float]:
    values = scores.get(group, {})
    if not values:
        return None, 0.0
    return max(values.items(), key=lambda item: item[1])


def load_frame_observations(
    connection: sqlite3.Connection,
    *,
    since_ts: int,
    until_ts: int,
    camera_ids: Sequence[str],
    embedding_model: str,
    prompt_version: str,
) -> list[FrameObservation]:
    if not camera_ids:
        return []
    placeholders = ",".join("?" for _ in camera_ids)
    rows = list(
        connection.execute(
            f"""
            SELECT
                k.id AS keyframe_id,
                k.media_file_id,
                k.wall_ts,
                k.wall_time_source,
                k.sample_reason,
                k.motion_score,
                COALESCE(k.night_score, 0) AS night_score,
                m.camera_id,
                COALESCE(m.duration_seconds, 0) AS media_duration,
                c.location,
                o.observation_group,
                o.label,
                o.score,
                o.details_json,
                COALESCE(
                    (
                        SELECT MAX(d.confidence)
                        FROM visual_detections d
                        WHERE d.keyframe_id = k.id
                          AND d.class_name = 'person'
                          -- Edge-only detections are commonly a caregiver's
                          -- arm or leg outside the scene. Treat them as
                          -- context, not proof that the baby's action is
                          -- visible enough to classify.
                          AND d.x1 >= 0.01
                          AND d.y1 >= 0.01
                          AND d.x2 <= 0.99
                          AND d.y2 <= 0.99
                    ),
                    0
                ) AS person_confidence,
                COALESCE(
                    (
                        SELECT MAX(d.confidence)
                        FROM visual_detections d
                        WHERE d.keyframe_id = k.id
                          AND d.class_name = 'bottle'
                          -- A bottle elsewhere in the room (or held by a
                          -- caregiver in the foreground) is not direct
                          -- feeding evidence. Its center must be close to a
                          -- small person detection that can plausibly be the
                          -- infant.
                          AND EXISTS (
                              SELECT 1
                              FROM visual_detections p
                              WHERE p.keyframe_id = k.id
                                AND p.class_name = 'person'
                                AND p.confidence >= 0.18
                                AND (p.x2 - p.x1) >= 0.05
                                AND (p.y2 - p.y1) >= 0.12
                                AND (
                                    (p.x2 - p.x1) * (p.y2 - p.y1)
                                ) BETWEEN 0.006 AND 0.24
                                AND ((d.x1 + d.x2) / 2.0)
                                    BETWEEN (p.x1 - 0.08)
                                    AND (p.x2 + 0.08)
                                AND ((d.y1 + d.y2) / 2.0)
                                    BETWEEN (p.y1 - 0.08)
                                    AND (p.y2 + 0.08)
                          )
                    ),
                    0
                ) AS feeding_container_confidence
                ,
                (
                    SELECT v.label
                    FROM visual_frame_verifications v
                    WHERE v.keyframe_id = k.id
                      AND v.task = 'feeding'
                    ORDER BY
                        CASE
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                              AND v.prompt_version =
                                'feeding-temporal-native-v2'
                            THEN 0
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                            THEN 1
                            WHEN v.model_name =
                                'Qwen/Qwen2.5-VL-7B-Instruct'
                            THEN 2
                            ELSE 3
                        END,
                        v.created_at DESC,
                        v.prompt_version DESC
                    LIMIT 1
                ) AS verified_feeding_label,
                COALESCE(
                    (
                        SELECT v.confidence
                        FROM visual_frame_verifications v
                        WHERE v.keyframe_id = k.id
                          AND v.task = 'feeding'
                        ORDER BY
                            CASE
                                WHEN v.model_name =
                                    'Qwen/Qwen3-VL-8B-Instruct'
                                  AND v.prompt_version =
                                    'feeding-temporal-native-v2'
                                THEN 0
                                WHEN v.model_name =
                                    'Qwen/Qwen3-VL-8B-Instruct'
                                THEN 1
                                WHEN v.model_name =
                                    'Qwen/Qwen2.5-VL-7B-Instruct'
                                THEN 2
                                ELSE 3
                            END,
                            v.created_at DESC,
                            v.prompt_version DESC
                        LIMIT 1
                    ),
                    0
                ) AS verified_feeding_confidence,
                (
                    SELECT v.model_name
                    FROM visual_frame_verifications v
                    WHERE v.keyframe_id = k.id
                      AND v.task = 'feeding'
                    ORDER BY
                        CASE
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                              AND v.prompt_version =
                                'feeding-temporal-native-v2'
                            THEN 0
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                            THEN 1
                            WHEN v.model_name =
                                'Qwen/Qwen2.5-VL-7B-Instruct'
                            THEN 2
                            ELSE 3
                        END,
                        v.created_at DESC,
                        v.prompt_version DESC
                    LIMIT 1
                ) AS verified_feeding_model_name,
                (
                    SELECT v.prompt_version
                    FROM visual_frame_verifications v
                    WHERE v.keyframe_id = k.id
                      AND v.task = 'feeding'
                    ORDER BY
                        CASE
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                              AND v.prompt_version =
                                'feeding-temporal-native-v2'
                            THEN 0
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                            THEN 1
                            WHEN v.model_name =
                                'Qwen/Qwen2.5-VL-7B-Instruct'
                            THEN 2
                            ELSE 3
                        END,
                        v.created_at DESC,
                        v.prompt_version DESC
                    LIMIT 1
                ) AS verified_feeding_prompt_version,
                (
                    SELECT v.label
                    FROM visual_frame_verifications v
                    WHERE v.keyframe_id = k.id
                      AND v.task = 'toy'
                    ORDER BY
                        CASE
                            WHEN v.model_name =
                                'Qwen/Qwen3-VL-8B-Instruct'
                            THEN 0
                            WHEN v.model_name =
                                'Qwen/Qwen2.5-VL-7B-Instruct'
                            THEN 1
                            ELSE 2
                        END,
                        v.created_at DESC,
                        v.prompt_version DESC
                    LIMIT 1
                ) AS verified_toy_label,
                COALESCE(
                    (
                        SELECT v.confidence
                        FROM visual_frame_verifications v
                        WHERE v.keyframe_id = k.id
                          AND v.task = 'toy'
                        ORDER BY
                            CASE
                                WHEN v.model_name =
                                    'Qwen/Qwen3-VL-8B-Instruct'
                                THEN 0
                                WHEN v.model_name =
                                    'Qwen/Qwen2.5-VL-7B-Instruct'
                                THEN 1
                                ELSE 2
                            END,
                            v.created_at DESC,
                            v.prompt_version DESC
                        LIMIT 1
                    ),
                    0
                ) AS verified_toy_confidence
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            JOIN visual_observations o ON o.keyframe_id = k.id
            WHERE k.wall_ts >= ?
              AND k.wall_ts < ?
              AND m.camera_id IN ({placeholders})
              AND o.model_name = ?
              AND o.prompt_version = ?
            ORDER BY k.wall_ts, k.id
            """,
            (
                since_ts,
                until_ts,
                *camera_ids,
                embedding_model,
                prompt_version,
            ),
        )
    )
    verification_rows = list(
        connection.execute(
            f"""
            SELECT
                v.keyframe_id,
                v.task,
                v.label,
                v.confidence,
                v.model_name,
                v.prompt_version,
                v.details_json,
                v.created_at
            FROM visual_frame_verifications v
            JOIN visual_keyframes k ON k.id = v.keyframe_id
            JOIN media_files m ON m.id = k.media_file_id
            WHERE k.wall_ts >= ?
              AND k.wall_ts < ?
              AND m.camera_id IN ({placeholders})
            """,
            (since_ts, until_ts, *camera_ids),
        )
    )
    verification_rows.extend(
        connection.execute(
            f"""
            SELECT
                o.keyframe_id,
                'posture' AS task,
                o.label,
                o.score AS confidence,
                o.model_name,
                o.prompt_version,
                o.details_json,
                o.created_at
            FROM visual_observations o
            JOIN visual_keyframes k ON k.id = o.keyframe_id
            JOIN media_files m ON m.id = k.media_file_id
            WHERE o.observation_group = 'posture_verification'
              AND k.wall_ts >= ?
              AND k.wall_ts < ?
              AND m.camera_id IN ({placeholders})
            """,
            (since_ts, until_ts, *camera_ids),
        )
    )

    def verification_priority(row: sqlite3.Row) -> tuple[int, int, int]:
        task = str(row["task"])
        prompt = str(row["prompt_version"])
        prompt_rank = 9
        if (
            task == "feeding"
            and prompt == TEMPORAL_FEEDING_PROMPT_VERSION
        ):
            prompt_rank = 0
        elif task == "feeding" and prompt.startswith(
            "feeding-temporal-"
        ):
            prompt_rank = 1
        elif task == "feeding":
            prompt_rank = 2
        elif (
            task == "toy"
            and prompt == TEMPORAL_PLAY_PROMPT_VERSION
        ):
            prompt_rank = 0
        elif task == "toy":
            prompt_rank = 2
        elif (
            task == "sleep_presence"
            and prompt == SLEEP_PRESENCE_PROMPT_VERSION
        ):
            prompt_rank = 0
        elif (
            task == "posture"
            and prompt == TEMPORAL_POSTURE_PROMPT_VERSION
        ):
            prompt_rank = 0
        model_name = str(row["model_name"])
        model_rank = (
            0
            if model_name == "Qwen/Qwen3-VL-8B-Instruct"
            else 1
            if model_name == "Qwen/Qwen3-VL-32B-Instruct"
            else 2
            if model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
            else 3
        )
        return (
            prompt_rank,
            model_rank,
            -int(row["created_at"]),
        )

    latest_verifications: dict[
        tuple[int, str],
        sqlite3.Row,
    ] = {}
    for verification in verification_rows:
        key = (
            int(verification["keyframe_id"]),
            str(verification["task"]),
        )
        current = latest_verifications.get(key)
        if (
            current is None
            or verification_priority(verification)
            < verification_priority(current)
        ):
            latest_verifications[key] = verification
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        keyframe_id = int(row["keyframe_id"])
        frame = grouped.setdefault(
            keyframe_id,
            {
                "row": row,
                "scores": defaultdict(dict),
                "raw_sigmoids": defaultdict(dict),
            },
        )
        frame["scores"][str(row["observation_group"])][
            str(row["label"])
        ] = float(row["score"])
        try:
            details = json.loads(row["details_json"] or "{}")
        except (TypeError, ValueError):
            details = {}
        frame["raw_sigmoids"][str(row["observation_group"])][
            str(row["label"])
        ] = float(details.get("sigmoid", 0.0))
    result: list[FrameObservation] = []
    for value in grouped.values():
        row = value["row"]
        keyframe_id = int(row["keyframe_id"])
        feeding_verification = latest_verifications.get(
            (keyframe_id, "feeding")
        )
        toy_verification = latest_verifications.get(
            (keyframe_id, "toy")
        )
        sleep_verification = latest_verifications.get(
            (keyframe_id, "sleep_presence")
        )
        posture_verification = latest_verifications.get(
            (keyframe_id, "posture")
        )
        camera_id = str(row["camera_id"])
        localized, zone = CAMERA_LOCATIONS.get(
            camera_id,
            (str(row["location"]), str(row["location"])),
        )
        result.append(
            FrameObservation(
                keyframe_id=keyframe_id,
                media_file_id=int(row["media_file_id"]),
                camera_id=camera_id,
                wall_ts=float(row["wall_ts"]),
                wall_time_source=str(row["wall_time_source"]),
                location=localized,
                zone=zone,
                sample_reason=str(row["sample_reason"]),
                media_duration=float(row["media_duration"]),
                motion_score=(
                    float(row["motion_score"])
                    if row["motion_score"] is not None
                    else None
                ),
                night_score=float(row["night_score"]),
                person_confidence=float(row["person_confidence"]),
                scores={
                    group: dict(labels)
                    for group, labels in value["scores"].items()
                },
                raw_sigmoids={
                    group: dict(labels)
                    for group, labels in value["raw_sigmoids"].items()
                },
                feeding_container_confidence=float(
                    row["feeding_container_confidence"]
                ),
                verified_feeding_label=(
                    str(feeding_verification["label"])
                    if feeding_verification is not None
                    else None
                ),
                verified_feeding_confidence=float(
                    feeding_verification["confidence"]
                    if feeding_verification is not None
                    else 0.0
                ),
                verified_feeding_model_name=(
                    str(feeding_verification["model_name"])
                    if feeding_verification is not None
                    else None
                ),
                verified_feeding_prompt_version=(
                    str(feeding_verification["prompt_version"])
                    if feeding_verification is not None
                    else None
                ),
                verified_feeding_details=(
                    _json_object(feeding_verification["details_json"])
                    if feeding_verification is not None
                    else None
                ),
                verified_toy_label=(
                    str(toy_verification["label"])
                    if toy_verification is not None
                    else None
                ),
                verified_toy_confidence=float(
                    toy_verification["confidence"]
                    if toy_verification is not None
                    else 0.0
                ),
                verified_toy_model_name=(
                    str(toy_verification["model_name"])
                    if toy_verification is not None
                    else None
                ),
                verified_toy_prompt_version=(
                    str(toy_verification["prompt_version"])
                    if toy_verification is not None
                    else None
                ),
                verified_sleep_label=(
                    str(sleep_verification["label"])
                    if sleep_verification is not None
                    else None
                ),
                verified_sleep_confidence=float(
                    sleep_verification["confidence"]
                    if sleep_verification is not None
                    else 0.0
                ),
                verified_sleep_model_name=(
                    str(sleep_verification["model_name"])
                    if sleep_verification is not None
                    else None
                ),
                verified_sleep_prompt_version=(
                    str(sleep_verification["prompt_version"])
                    if sleep_verification is not None
                    else None
                ),
                verified_sleep_details=(
                    _json_object(sleep_verification["details_json"])
                    if sleep_verification is not None
                    else None
                ),
                verified_posture_label=(
                    str(posture_verification["label"])
                    if posture_verification is not None
                    else None
                ),
                verified_posture_confidence=float(
                    posture_verification["confidence"]
                    if posture_verification is not None
                    else 0.0
                ),
                verified_posture_model_name=(
                    str(posture_verification["model_name"])
                    if posture_verification is not None
                    else None
                ),
                verified_posture_prompt_version=(
                    str(posture_verification["prompt_version"])
                    if posture_verification is not None
                    else None
                ),
            )
        )
    return result


def frame_candidates(
    frame: FrameObservation,
    *,
    use_verification: bool = True,
) -> list[Candidate]:
    support = _support_seconds(frame)
    half = support / 2.0
    start = frame.wall_ts - half
    end = frame.wall_ts + half
    precision = _time_precision(frame.wall_time_source)
    toy_values = {
        label: score
        for label, score in frame.scores.get("toy", {}).items()
        if label != "no_toy"
    }
    toy_label = (
        max(toy_values, key=toy_values.__getitem__)
        if toy_values
        else None
    )
    toy_score = toy_values.get(toy_label or "", 0.0)
    toy_raw = frame.raw_sigmoids.get("toy", {}).get(toy_label or "", 0.0)
    if toy_score < 0.08 or toy_raw < 0.04:
        toy_label = None
        toy_score = 0.0
    temporal_play_verification = bool(
        use_verification
        and frame.verified_toy_label
        and frame.verified_toy_model_name
        == "Qwen/Qwen3-VL-8B-Instruct"
        and frame.verified_toy_prompt_version
        == TEMPORAL_PLAY_PROMPT_VERSION
    )
    if temporal_play_verification:
        if frame.verified_toy_label in {"NONE", "UNCLEAR"}:
            toy_label = None
            toy_score = 0.0
        else:
            toy_label = VERIFIED_TOY_LABELS.get(
                frame.verified_toy_label,
                "other_toy",
            )
            toy_score = max(
                toy_score,
                frame.verified_toy_confidence,
            )
    candidates: list[Candidate] = []

    feeding, feeding_score = _top(frame.scores, "feeding")
    feeding_raw = frame.raw_sigmoids.get("feeding", {}).get(
        feeding or "",
        0.0,
    )
    not_feeding_raw = frame.raw_sigmoids.get("feeding", {}).get(
        "not_feeding",
        0.0,
    )
    sleeping_raw = frame.raw_sigmoids.get("posture", {}).get(
        "sleeping",
        0.0,
    )
    care_label, care_score = _top(frame.scores, "care")
    active_despite_sleep_prompt = (
        frame.motion_score is not None
        and frame.motion_score > 0.04
    )
    bottle_score = frame.scores.get("feeding", {}).get(
        "feeding_bottle",
        0.0,
    )
    bottle_raw = frame.raw_sigmoids.get("feeding", {}).get(
        "feeding_bottle",
        0.0,
    )
    visible_feeding_container = (
        frame.feeding_container_confidence >= 0.25
        and bottle_score >= 0.25
        and bottle_raw >= 0.04
    )
    if visible_feeding_container:
        feeding = "feeding_bottle"
        feeding_score = bottle_score
        feeding_raw = bottle_raw
    verification_available = bool(
        use_verification and frame.verified_feeding_label
    )
    verified_feeding = (
        VERIFIED_FEEDING_LABELS.get(
            frame.verified_feeding_label or ""
        )
        if verification_available
        else None
    )
    temporal_verification = bool(
        verification_available
        and frame.verified_feeding_model_name
        == "Qwen/Qwen3-VL-8B-Instruct"
        and frame.verified_feeding_prompt_version
        == TEMPORAL_FEEDING_PROMPT_VERSION
    )
    probable_feeding = (
        _probable_temporal_feeding(frame)
        if temporal_verification
        else None
    )
    if (
        temporal_verification
        and verified_feeding is not None
        and frame.verified_feeding_confidence >= 0.85
    ):
        candidates.append(
            Candidate(
                "feeding",
                verified_feeding,
                frame.wall_ts,
                frame.wall_ts + 1.0,
                frame.camera_id,
                frame.location,
                frame.zone,
                frame.verified_feeding_confidence,
                frame.keyframe_id,
                precision,
                direct_support=True,
                verified_support=True,
                verification_available=True,
                semantic_strength=frame.verified_feeding_confidence,
                temporal_verified=True,
                measurement_mode="sighting_only",
                certainty="confirmed",
            )
        )
        feeding = None
        feeding_score = 0.0
        feeding_raw = 0.0
    elif probable_feeding is not None:
        candidates.append(
            Candidate(
                "feeding",
                probable_feeding,
                frame.wall_ts,
                frame.wall_ts + 1.0,
                frame.camera_id,
                frame.location,
                frame.zone,
                max(0.70, frame.verified_feeding_confidence),
                frame.keyframe_id,
                precision,
                direct_support=True,
                verified_support=True,
                verification_available=True,
                semantic_strength=frame.verified_feeding_confidence,
                temporal_verified=True,
                measurement_mode="sighting_only",
                certainty="probable",
            )
        )
        feeding = None
        feeding_score = 0.0
        feeding_raw = 0.0
    elif use_verification:
        # Low-resolution SigLIP, YOLO and legacy single-frame Qwen checks are
        # recall signals only. A feeding event now requires the native-video
        # temporal verifier; otherwise ordinary holding repeatedly becomes a
        # false bottle or breastfeeding session.
        feeding = None
        feeding_score = 0.0
        feeding_raw = 0.0
    strong_verified_play = (
        temporal_play_verification
        and frame.verified_toy_label not in {
            None,
            "NONE",
            "UNCLEAR",
        }
        and max(
            frame.scores.get("play", {}).get("playing", 0.0),
            frame.scores.get("play", {}).get("reading", 0.0),
        )
        >= 0.70
    )
    if (
        not temporal_verification
        and feeding in {"feeding_bottle", "breastfeeding", "solid_food"}
    ):
        clear_of_negative = feeding_raw >= not_feeding_raw * 1.35
        semantic_floor = (
            VERIFIED_FEEDING_SEMANTIC_FLOORS.get(feeding, 0.18)
            if verification_available
            else 0.08
        )
        if verification_available and visible_feeding_container:
            # The independent bottle detector is the strongest second signal.
            semantic_floor = 0.08
        elif (
            verification_available
            and frame.camera_id == "lumi3.8e2ce0d05d9c251e"
        ):
            # The ceiling camera supplies repeated event views but smaller
            # subjects, so its SigLIP raw score is systematically lower.
            semantic_floor = min(semantic_floor, 0.14)
        diaper_lookalike = (
            feeding == "breastfeeding"
            and care_label == "diaper_change"
            and care_score >= 0.45
        )
        if (
            feeding_score >= (
                0.42 if visible_feeding_container else 0.55
            )
            and feeding_raw >= semantic_floor
            and clear_of_negative
            and frame.person_confidence >= 0.15
            and (sleeping_raw < 0.50 or active_despite_sleep_prompt)
            and not diaper_lookalike
            and (not strong_verified_play or visible_feeding_container)
        ):
            candidates.append(
                Candidate(
                    "feeding",
                    feeding,
                    start,
                    end,
                    frame.camera_id,
                    frame.location,
                    frame.zone,
                    feeding_score,
                    frame.keyframe_id,
                    precision,
                    direct_support=visible_feeding_container,
                    verified_support=verified_feeding is not None,
                    verification_available=verification_available,
                    semantic_strength=feeding_raw,
                )
            )

    play, play_score = _top(frame.scores, "play")
    play_raw = frame.raw_sigmoids.get("play", {}).get(play or "", 0.0)
    verified_play = bool(
        temporal_play_verification
        and frame.verified_toy_label not in {
            None,
            "NONE",
            "UNCLEAR",
        }
        and frame.verified_toy_confidence >= 0.78
    )
    if (
        play in {"playing", "reading"}
        and play_score >= 0.44
        and play_raw >= 0.05
        and frame.person_confidence >= 0.15
        and (sleeping_raw < 0.50 or active_despite_sleep_prompt)
    ):
        candidates.append(
            Candidate(
                "play",
                play,
                start,
                end,
                frame.camera_id,
                frame.location,
                frame.zone,
                play_score,
                frame.keyframe_id,
                precision,
                toy_label=toy_label,
                toy_score=toy_score,
                verified_support=verified_play,
                verification_available=temporal_play_verification,
                temporal_verified=verified_play,
            )
        )

    posture, posture_score = _top(frame.scores, "posture")
    posture_raw = frame.raw_sigmoids.get("posture", {}).get(
        posture or "",
        0.0,
    )
    posture_map = {
        "crawling": "crawling",
        "sitting": "sitting",
        "lying_awake": "lying_awake",
    }
    unknown_posture_score = frame.scores.get("posture", {}).get(
        "posture_unknown",
        0.0,
    )
    temporal_posture_verification = bool(
        use_verification
        and frame.verified_posture_label
        and frame.verified_posture_model_name
        == "Qwen/Qwen3-VL-8B-Instruct"
        and frame.verified_posture_prompt_version
        == TEMPORAL_POSTURE_PROMPT_VERSION
    )
    verified_posture = bool(
        temporal_posture_verification
        and posture in posture_map
        and frame.verified_posture_label
        == posture_map[posture].upper()
        and frame.verified_posture_confidence >= 0.78
    )
    if (
        posture in posture_map
        and posture_score >= 0.48
        and posture_raw >= 0.10
        and posture_score - unknown_posture_score >= 0.18
        and frame.person_confidence >= 0.20
    ):
        candidates.append(
            Candidate(
                posture_map[posture],
                None,
                start,
                end,
                frame.camera_id,
                frame.location,
                frame.zone,
                posture_score,
                frame.keyframe_id,
                precision,
                verified_support=verified_posture,
                verification_available=temporal_posture_verification,
                temporal_verified=verified_posture,
            )
        )

    temporal_sleep_verification = bool(
        use_verification
        and frame.verified_sleep_label
        and frame.verified_sleep_model_name
        == "Qwen/Qwen3-VL-8B-Instruct"
        and frame.verified_sleep_prompt_version
        == SLEEP_PRESENCE_PROMPT_VERSION
    )
    verified_sleep = bool(
        temporal_sleep_verification
        and frame.verified_sleep_label == "BABY_SLEEPING"
        and frame.verified_sleep_confidence >= 0.78
    )
    if (
        frame.camera_id == "B88880620430"
        and posture == "sleeping"
        and posture_score >= 0.34
        and posture_raw >= 0.50
        and (
            frame.motion_score is None
            or frame.motion_score <= 0.045
        )
        and (not use_verification or verified_sleep)
    ):
        movement_factor = (
            1.0
            if frame.motion_score is None
            else max(0.0, 1.0 - frame.motion_score / 0.045)
        )
        confidence = min(
            0.99,
            posture_score * 0.8 + movement_factor * 0.2,
        )
        candidates.append(
            Candidate(
                "inferred_sleep",
                "low_motion_in_bed",
                (
                    frame.wall_ts - 60.0
                    if use_verification
                    else start
                ),
                (
                    frame.wall_ts + 60.0
                    if use_verification
                    else end
                ),
                frame.camera_id,
                frame.location,
                frame.zone,
                confidence,
                frame.keyframe_id,
                precision,
                verified_support=verified_sleep,
                verification_available=temporal_sleep_verification,
                temporal_verified=verified_sleep,
                measurement_mode=(
                    "native_state_anchor"
                    if use_verification
                    else "visible_lower_bound"
                ),
                certainty="confirmed",
                start_boundary_error_seconds=(60.0 if use_verification else None),
                end_boundary_error_seconds=(60.0 if use_verification else None),
            )
        )
    return candidates


def infer_sleep_state_candidates(
    frames: Sequence[FrameObservation],
    *,
    max_anchor_gap_seconds: float = 20.0 * 60.0,
    native_window_seconds: float = 120.0,
) -> list[Candidate]:
    """Turn verified bedside states into bounded, gap-aware sleep cells.

    Every Qwen sleep check describes a two-minute native-video window.  A
    run of positive anchors is joined at the midpoint between adjacent checks.
    A NOT_SLEEPING or UNCLEAR anchor therefore cuts the run instead of being
    silently covered by a fixed +/- five-minute expansion.  If verification
    points are too far apart, only the native window is retained and the
    unsampled time remains unknown.
    """

    if max_anchor_gap_seconds <= 0:
        raise ValueError("max_anchor_gap_seconds must be positive")
    native_half = max(1.0, float(native_window_seconds) / 2.0)
    verified = sorted(
        (
            frame
            for frame in frames
            if frame.camera_id == "B88880620430"
            and frame.verified_sleep_model_name
            == "Qwen/Qwen3-VL-8B-Instruct"
            and frame.verified_sleep_prompt_version
            == SLEEP_PRESENCE_PROMPT_VERSION
            and frame.verified_sleep_label
        ),
        key=lambda frame: (frame.wall_ts, frame.keyframe_id),
    )
    result: list[Candidate] = []
    for index, frame in enumerate(verified):
        if (
            frame.verified_sleep_label != "BABY_SLEEPING"
            or frame.verified_sleep_confidence < 0.78
        ):
            continue
        previous = verified[index - 1] if index else None
        following = verified[index + 1] if index + 1 < len(verified) else None

        previous_gap = (
            frame.wall_ts - previous.wall_ts
            if previous is not None
            else None
        )
        if (
            previous is not None
            and previous_gap is not None
            and previous_gap <= max_anchor_gap_seconds
        ):
            start_ts = (previous.wall_ts + frame.wall_ts) / 2.0
            start_error = (
                previous_gap / 2.0
                if previous.verified_sleep_label != "BABY_SLEEPING"
                else 0.0
            )
        else:
            start_ts = frame.wall_ts - native_half
            start_error = native_half

        following_gap = (
            following.wall_ts - frame.wall_ts
            if following is not None
            else None
        )
        if (
            following is not None
            and following_gap is not None
            and following_gap <= max_anchor_gap_seconds
        ):
            end_ts = (frame.wall_ts + following.wall_ts) / 2.0
            end_error = (
                following_gap / 2.0
                if following.verified_sleep_label != "BABY_SLEEPING"
                else 0.0
            )
        else:
            end_ts = frame.wall_ts + native_half
            end_error = native_half

        result.append(
            Candidate(
                "inferred_sleep",
                "low_motion_in_bed",
                start_ts,
                end_ts,
                frame.camera_id,
                frame.location,
                frame.zone,
                frame.verified_sleep_confidence,
                frame.keyframe_id,
                _time_precision(frame.wall_time_source),
                direct_support=True,
                verified_support=True,
                verification_available=True,
                semantic_strength=frame.verified_sleep_confidence,
                temporal_verified=True,
                measurement_mode="bounded_state_interval",
                certainty="confirmed",
                start_boundary_error_seconds=start_error,
                end_boundary_error_seconds=end_error,
            )
        )
    return result


def _merge_gap(activity_type: str) -> float:
    return {
        # Feeding evidence is deliberately sampled sparsely. Confirmed
        # observations within one typical session should not become several
        # separate "feeds" merely because no candidate frame landed between.
        "feeding": 20.0 * 60.0,
        "play": 120.0,
        "crawling": 60.0,
        "sitting": 60.0,
        "standing": 60.0,
        "lying_awake": 90.0,
        # Verified state cells already touch when sleep is continuous.  Any
        # visible gap now represents an awake/unclear/unsampled interval and
        # must remain visible instead of being merged away.
        "inferred_sleep": 1.0,
    }.get(activity_type, 60.0)


def _event_key(
    activity_type: str,
    subtype: str | None,
    start_ts: float,
    camera_ids: Iterable[str],
    model_version: str,
) -> str:
    identity = "|".join(
        (
            activity_type,
            subtype or "",
            str(round(start_ts / 10)),
            ",".join(sorted(set(camera_ids))),
            model_version,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _observed_duration(candidates: Sequence[Candidate]) -> float:
    intervals = sorted(
        (float(item.start_ts), float(item.end_ts))
        for item in candidates
        if item.end_ts > item.start_ts
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def merge_candidates(
    candidates: Sequence[Candidate],
    *,
    timezone: ZoneInfo = ZoneInfo("Asia/Shanghai"),
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str | None, str],
        list[Candidate],
    ] = defaultdict(list)
    for candidate in candidates:
        midpoint = (candidate.start_ts + candidate.end_ts) / 2.0
        local_day = datetime.fromtimestamp(midpoint, timezone).date().isoformat()
        # Bottle/breast labels can alternate between adjacent frames even
        # though they describe one uninterrupted feeding session. Merge the
        # session first, then choose its strongest supported subtype.
        grouping_subtype = (
            None
            if candidate.activity_type == "feeding"
            else candidate.subtype
        )
        grouped[
            (candidate.activity_type, grouping_subtype, local_day)
        ].append(candidate)
    events: list[dict[str, Any]] = []
    for (activity_type, subtype, _local_day), group in grouped.items():
        ordered = sorted(group, key=lambda item: (item.start_ts, item.end_ts))
        current: list[Candidate] = []

        def flush() -> None:
            if not current:
                return
            start = min(item.start_ts for item in current)
            end = max(item.end_ts for item in current)
            observed_duration = _observed_duration(current)
            evidence = sorted(
                current,
                key=lambda item: item.confidence,
                reverse=True,
            )
            unique_evidence: list[Candidate] = []
            seen_ids: set[int] = set()
            for item in evidence:
                if item.keyframe_id in seen_ids:
                    continue
                seen_ids.add(item.keyframe_id)
                unique_evidence.append(item)
            if activity_type == "inferred_sleep":
                native_sleep_support = any(
                    item.temporal_verified and item.verified_support
                    for item in current
                )
                if (
                    observed_duration < 2 * 60
                    or (
                        not native_sleep_support
                        and len(unique_evidence) < 6
                    )
                ):
                    current.clear()
                    return
            mean_score = sum(
                item.confidence for item in current
            ) / len(current)
            support_bonus = (
                1.0 - math.exp(-len(unique_evidence) / 4.0)
            )
            confidence = min(0.99, mean_score * 0.82 + support_bonus * 0.18)
            verification_count = sum(
                item.verification_available for item in current
            )
            verified_count = sum(
                item.verified_support for item in current
            )
            if activity_type == "feeding":
                direct_count = sum(
                    item.direct_support for item in current
                )
                # A single low-resolution zero-shot frame is too easily
                # confused with holding, soothing or play. A native temporal
                # verdict already represents at least three original-video
                # frames with visible mouth contact, so one such anchor is
                # sufficient evidence for a session.
                if len(unique_evidence) == 1 and direct_count == 0:
                    current.clear()
                    return
            if (
                activity_type
                in {
                    "play",
                    "inferred_sleep",
                    "crawling",
                    "sitting",
                }
                and verification_count
                and verified_count == 0
            ):
                # The event-level verifier anchor is deliberately retained as
                # a candidate even for a negative verdict so the complete
                # zero-shot segment can be rejected here.
                current.clear()
                return
            camera_ids = sorted({item.camera_id for item in current})
            location_scores: dict[tuple[str, str], float] = defaultdict(float)
            for item in current:
                location_scores[(item.location, item.zone)] += item.confidence
            location, zone = max(
                location_scores,
                key=location_scores.__getitem__,
            )
            precisions = {item.time_precision for item in current}
            precision = (
                "estimated"
                if "estimated" in precisions
                else "mapped"
                if precisions == {"mapped"}
                else "exact"
            )
            toy_scores: dict[str, float] = defaultdict(float)
            toy_supports: dict[str, int] = defaultdict(int)
            for item in current:
                if item.toy_label:
                    toy_scores[item.toy_label] += item.toy_score
                    toy_supports[item.toy_label] += 1
            toy = (
                max(toy_scores, key=toy_scores.__getitem__)
                if toy_scores
                else None
            )
            # A single low-resolution view can confuse a nearby toy with the
            # toy the baby is actually using. Publish a named toy only after
            # the same label is independently seen in at least two frames.
            temporal_toy_support = any(
                item.temporal_verified
                and item.verified_support
                and item.toy_label == toy
                for item in current
            )
            if (
                toy
                and toy_supports[toy] < 2
                and not temporal_toy_support
            ):
                toy = None
            toy_confidence = (
                min(
                    1.0,
                    max(
                        0.0,
                        toy_scores[toy] / max(1, toy_supports[toy]),
                    ),
                )
                if toy
                else None
            )
            subtype_scores: dict[str, float] = defaultdict(float)
            for item in current:
                if item.subtype:
                    subtype_scores[item.subtype] += item.confidence
            resolved_subtype = (
                max(subtype_scores, key=subtype_scores.__getitem__)
                if subtype_scores
                else subtype
            )
            measurement_modes = {
                item.measurement_mode for item in current
            }
            measurement_mode = (
                next(iter(measurement_modes))
                if len(measurement_modes) == 1
                else "mixed_observation"
            )
            certainty = (
                "confirmed"
                if any(item.certainty == "confirmed" for item in current)
                else "probable"
                if any(item.certainty == "probable" for item in current)
                else "uncertain"
            )
            first_boundary = min(
                current,
                key=lambda item: (item.start_ts, item.keyframe_id),
            )
            last_boundary = max(
                current,
                key=lambda item: (item.end_ts, -item.keyframe_id),
            )
            duration_publishable = activity_type == "inferred_sleep"
            if (
                activity_type == "feeding"
                and measurement_modes == {"sighting_only"}
            ):
                # The 1-second interval is only a searchable timeline marker.
                # Native five-frame verification confirms the occurrence, not
                # the full feeding duration.
                observed_duration = 0.0
            events.append(
                {
                    "activity_type": activity_type,
                    "subtype": resolved_subtype,
                    "start_ts": start,
                    "end_ts": end,
                    "duration_seconds": observed_duration,
                    "session_span_seconds": max(0.0, end - start),
                    "camera_ids": camera_ids,
                    "location": location,
                    "zone": zone,
                    "confidence": confidence,
                    "time_precision": precision,
                    "evidence": unique_evidence[:5],
                    # Internal callers such as native sleep verification need
                    # every sampled point to cover a multi-hour event. Only
                    # the top five are persisted as user-facing evidence.
                    "all_evidence": unique_evidence,
                    "support_frames": len(unique_evidence),
                    "toy_label": toy,
                    "toy_scores": dict(toy_scores),
                    "toy_confidence": toy_confidence,
                    "subtype_scores": dict(subtype_scores),
                    "measurement_mode": measurement_mode,
                    "certainty": certainty,
                    "duration_publishable": duration_publishable,
                    "start_boundary_error_seconds": (
                        first_boundary.start_boundary_error_seconds
                    ),
                    "end_boundary_error_seconds": (
                        last_boundary.end_boundary_error_seconds
                    ),
                    "method": (
                        "qwen3_vl_temporal"
                        if any(item.temporal_verified for item in current)
                        else "siglip2_temporal_evidence"
                    ),
                }
            )
            current.clear()

        for candidate in ordered:
            if not current:
                current.append(candidate)
                continue
            if candidate.start_ts <= max(
                item.end_ts for item in current
            ) + _merge_gap(activity_type):
                current.append(candidate)
            else:
                flush()
                current.append(candidate)
        flush()
    return sorted(events, key=lambda item: item["start_ts"])


def save_activity_events(
    connection: sqlite3.Connection,
    *,
    events: Sequence[Mapping[str, Any]],
    since_ts: int,
    until_ts: int,
    generated_by_run_id: int | None,
    model_version: str,
) -> dict[str, int]:
    now_ts = int(time.time())
    inserted = 0
    evidence_count = 0
    current_event_keys: list[str] = []
    reviewed_rows = list(
        connection.execute(
            """
            SELECT id, event_key, activity_type, activity_subtype,
                   start_ts, end_ts
            FROM activity_events
            WHERE model_version = ?
              AND review_status != 'machine'
              AND start_ts < ?
              AND end_ts >= ?
            ORDER BY start_ts, id
            """,
            (model_version, until_ts + 20 * 60, since_ts - 20 * 60),
        )
    )
    used_reviewed_ids: set[int] = set()
    for event in events:
        event_key = _event_key(
            str(event["activity_type"]),
            (
                str(event["subtype"])
                if event.get("subtype") is not None
                else None
            ),
            float(event["start_ts"]),
            event["camera_ids"],
            model_version,
        )
        event_midpoint = (
            float(event["start_ts"]) + float(event["end_ts"])
        ) / 2.0
        match_limit = (
            10 * 60
            if event["activity_type"] == "feeding"
            else 15 * 60
            if event["activity_type"] == "inferred_sleep"
            else _merge_gap(str(event["activity_type"])) + 60
        )
        reviewed_match = min(
            (
                row
                for row in reviewed_rows
                if int(row["id"]) not in used_reviewed_ids
                and str(row["activity_type"]) == event["activity_type"]
                and abs(
                    (
                        float(row["start_ts"])
                        + float(row["end_ts"])
                    )
                    / 2.0
                    - event_midpoint
                )
                <= match_limit
            ),
            key=lambda row: abs(
                (float(row["start_ts"]) + float(row["end_ts"])) / 2.0
                - event_midpoint
            ),
            default=None,
        )
        if reviewed_match is not None:
            # Keep the reviewed row ID and its attached family verdict.  A
            # model upgrade may still correct the machine-derived raw bounds;
            # explicit corrected_start/end values live in the review table
            # and remain authoritative in the diary query.
            event_key = str(reviewed_match["event_key"])
            used_reviewed_ids.add(int(reviewed_match["id"]))
        current_event_keys.append(event_key)
        event_row = connection.execute(
            """
            INSERT INTO activity_events(
                event_key, activity_type, activity_subtype,
                start_ts, end_ts, duration_seconds, camera_ids_json,
                location, zone, confidence, time_precision,
                method, model_version, details_json,
                generated_by_run_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_key) DO UPDATE SET
                start_ts = excluded.start_ts,
                end_ts = excluded.end_ts,
                duration_seconds = excluded.duration_seconds,
                confidence = excluded.confidence,
                details_json = json_patch(
                    json_patch(
                        excluded.details_json,
                        CASE
                            WHEN json_type(
                                activity_events.details_json,
                                '$.growth_milestone'
                            ) IS NOT NULL
                            THEN json_object(
                                'growth_milestone',
                                json_extract(
                                    activity_events.details_json,
                                    '$.growth_milestone'
                                )
                            )
                            ELSE '{}'
                        END
                    ),
                    CASE
                        WHEN json_type(
                            activity_events.details_json,
                            '$.growth_milestone_review'
                        ) IS NOT NULL
                        THEN json_object(
                            'growth_milestone_review',
                            json_extract(
                                activity_events.details_json,
                                '$.growth_milestone_review'
                            )
                        )
                        ELSE '{}'
                    END
                ),
                generated_by_run_id = excluded.generated_by_run_id,
                updated_at = excluded.updated_at
            RETURNING id
            """,
            (
                event_key,
                event["activity_type"],
                event.get("subtype"),
                event["start_ts"],
                event["end_ts"],
                event["duration_seconds"],
                _json(event["camera_ids"]),
                event["location"],
                event.get("zone"),
                event["confidence"],
                event["time_precision"],
                event.get("method", "siglip2_temporal_evidence"),
                model_version,
                _json(
                    {
                        "support_frames": event["support_frames"],
                        "session_span_seconds": event.get(
                            "session_span_seconds",
                            event["duration_seconds"],
                        ),
                        "toy_label": event.get("toy_label"),
                        "toy_scores": event.get("toy_scores", {}),
                        "toy_confidence": event.get("toy_confidence"),
                        "subtype_scores": event.get("subtype_scores", {}),
                        "measurement_mode": event.get(
                            "measurement_mode",
                            "visible_lower_bound",
                        ),
                        "certainty": event.get("certainty", "confirmed"),
                        "duration_publishable": bool(
                            event.get("duration_publishable", False)
                        ),
                        "start_boundary_error_seconds": event.get(
                            "start_boundary_error_seconds"
                        ),
                        "end_boundary_error_seconds": event.get(
                            "end_boundary_error_seconds"
                        ),
                        "limitations": (
                            "camera_observation_not_ground_truth"
                        ),
                        "evidence_method": event.get(
                            "method",
                            "siglip2_temporal_evidence",
                        ),
                    }
                ),
                generated_by_run_id,
                now_ts,
                now_ts,
            ),
        ).fetchone()
        event_id = int(event_row["id"])
        connection.execute(
            "DELETE FROM activity_event_evidence WHERE activity_event_id = ?",
            (event_id,),
        )
        for rank, item in enumerate(event["evidence"], start=1):
            connection.execute(
                """
                INSERT INTO activity_event_evidence(
                    activity_event_id, keyframe_id,
                    evidence_rank, score, role
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    item.keyframe_id,
                    rank,
                    item.confidence,
                    "representative" if rank == 1 else "support",
                ),
            )
            evidence_count += 1
        inserted += 1

    # Keep IDs stable for events that survive a rebuild. The web UI, preview
    # cache, and saved human reviews all address an event by ID; deleting every
    # machine event before the upsert made those references race each visual
    # worker cycle even though event_key itself was unchanged.
    connection.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS current_activity_event_keys(
            event_key TEXT PRIMARY KEY
        )
        """
    )
    connection.execute("DELETE FROM current_activity_event_keys")
    connection.executemany(
        """
        INSERT OR IGNORE INTO current_activity_event_keys(event_key)
        VALUES (?)
        """,
        ((event_key,) for event_key in current_event_keys),
    )
    connection.execute(
        """
        DELETE FROM activity_events
        WHERE start_ts >= ? AND start_ts < ?
          AND model_version = ?
          AND review_status = 'machine'
          AND NOT EXISTS (
              SELECT 1
              FROM current_activity_event_keys current
              WHERE current.event_key = activity_events.event_key
          )
        """,
        (since_ts, until_ts, model_version),
    )
    connection.execute("DELETE FROM current_activity_event_keys")

    # The activity builder can be run for a day after a month has already been
    # indexed. Rebuild the small toy lookup from every retained play event so a
    # scoped refresh cannot erase toy history outside its own time window.
    toy_links: list[tuple[str, int, float]] = []
    for row in connection.execute(
        """
        SELECT e.id, e.details_json, e.confidence
        FROM activity_events e
        LEFT JOIN activity_event_reviews r
          ON r.activity_event_id = e.id
        WHERE COALESCE(r.verdict, e.review_status) != 'rejected'
          AND COALESCE(
                r.corrected_activity_type,
                e.activity_type
              ) = 'play'
        """
    ):
        try:
            details = json.loads(str(row["details_json"] or "{}"))
        except (TypeError, ValueError):
            details = {}
        toy_label = details.get("toy_label")
        if not toy_label:
            continue
        score = min(
            1.0,
            max(
                0.0,
                float(
                    details.get(
                        "toy_confidence",
                        row["confidence"],
                    )
                    or 0.0
                ),
            ),
        )
        toy_links.append(
            (str(toy_label), int(row["id"]), float(score))
        )

    connection.execute("DELETE FROM toy_event_links")
    connection.execute("DELETE FROM toy_entities")
    for toy_label, event_id, score in toy_links:
        event = connection.execute(
            """
            SELECT start_ts, end_ts, duration_seconds
            FROM activity_events
            WHERE id = ?
            """,
            (event_id,),
        ).fetchone()
        if event is None:
            continue
        entity_key = f"toy-type:{toy_label}"
        representative = connection.execute(
            """
            SELECT keyframe_id
            FROM activity_event_evidence
            WHERE activity_event_id = ?
            ORDER BY evidence_rank
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO toy_entities(
                entity_key, display_name, toy_type,
                representative_keyframe_id, first_seen_ts, last_seen_ts,
                interaction_count, interaction_seconds, confidence,
                model_version, details_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_key) DO UPDATE SET
                representative_keyframe_id = CASE
                    WHEN excluded.confidence > toy_entities.confidence
                    THEN excluded.representative_keyframe_id
                    ELSE toy_entities.representative_keyframe_id END,
                first_seen_ts = MIN(
                    toy_entities.first_seen_ts,
                    excluded.first_seen_ts
                ),
                last_seen_ts = MAX(
                    toy_entities.last_seen_ts,
                    excluded.last_seen_ts
                ),
                interaction_count = toy_entities.interaction_count + 1,
                interaction_seconds = (
                    toy_entities.interaction_seconds
                    + excluded.interaction_seconds
                ),
                confidence = MAX(
                    toy_entities.confidence,
                    excluded.confidence
                ),
                updated_at = excluded.updated_at
            """,
            (
                entity_key,
                TOY_NAMES.get(toy_label, toy_label),
                toy_label,
                (
                    int(representative["keyframe_id"])
                    if representative
                    else None
                ),
                float(event["start_ts"]),
                float(event["end_ts"]),
                float(event["duration_seconds"]),
                score,
                model_version,
                _json({"meaning": "observable_interaction_type"}),
                now_ts,
            ),
        )
        toy_id = int(
            connection.execute(
                "SELECT id FROM toy_entities WHERE entity_key = ?",
                (entity_key,),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO toy_event_links(
                toy_entity_id, activity_event_id, confidence
            )
            VALUES (?, ?, ?)
            """,
            (toy_id, event_id, score),
        )
    return {
        "events": inserted,
        "evidence": evidence_count,
        "toy_links": len(toy_links),
    }


def rebuild_activity_events(
    connection: sqlite3.Connection,
    *,
    since_ts: int,
    until_ts: int,
    camera_ids: Sequence[str],
    embedding_model: str,
    prompt_version: str,
    generated_by_run_id: int | None = None,
) -> dict[str, int]:
    frames = load_frame_observations(
        connection,
        since_ts=since_ts,
        until_ts=until_ts,
        camera_ids=camera_ids,
        embedding_model=embedding_model,
        prompt_version=prompt_version,
    )
    candidates = [
        candidate
        for frame in frames
        for candidate in frame_candidates(frame)
        if candidate.activity_type != "inferred_sleep"
    ]
    candidates.extend(infer_sleep_state_candidates(frames))
    events = merge_candidates(candidates)
    saved = save_activity_events(
        connection,
        events=events,
        since_ts=since_ts,
        until_ts=until_ts,
        generated_by_run_id=generated_by_run_id,
        model_version=ACTIVITY_MODEL_VERSION,
    )
    return {
        "frames": len(frames),
        "candidates": len(candidates),
        **saved,
    }
