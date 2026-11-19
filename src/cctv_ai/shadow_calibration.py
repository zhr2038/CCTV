from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

from .audio_events import MODEL_NAME, YamnetDetector, load_class_names
from .database import (
    connect_database,
    query_audio_event_fingerprints,
    query_episode_events,
    save_audio_event_fingerprint,
)
from .episodes import build_episodes


SHADOW_MODEL_NAME = "family-yamnet-nearest-neighbor-v1"
FINAL_REVIEW_STATUSES = {"confirmed", "false_positive"}


def build_daily_episodes(
    rows: Iterable[Mapping[str, Any]],
    *,
    timezone: ZoneInfo,
    gap_seconds: float,
) -> list[dict[str, Any]]:
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        date = datetime.fromtimestamp(
            float(row["wall_start_ts"]),
            timezone,
        ).date().isoformat()
        by_date[date].append(row)
    episodes: list[dict[str, Any]] = []
    for date, date_rows in by_date.items():
        for episode in build_episodes(date_rows, gap_seconds=gap_seconds):
            episodes.append({**episode, "date": date})
    return sorted(
        episodes,
        key=lambda episode: float(episode["start_ts"]),
        reverse=True,
    )


def _safe_media_path(source_root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe media path: {relative_path}")
    return source_root.joinpath(*parts)


def _fingerprint_vector(row: Mapping[str, Any]):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Shadow calibration requires the audio-events dependencies"
        ) from exc
    length = int(row["vector_length"])
    mean_scores = np.frombuffer(
        row["mean_scores"],
        dtype="<f4",
        count=length,
    ).astype(np.float32, copy=True)
    max_scores = np.frombuffer(
        row["max_scores"],
        dtype="<f4",
        count=length,
    ).astype(np.float32, copy=True)
    if len(mean_scores) != length or len(max_scores) != length:
        raise ValueError("Stored fingerprint vector is truncated")
    acoustic = np.concatenate(
        [
            np.sqrt(np.clip(mean_scores, 0.0, 1.0)),
            np.sqrt(np.clip(max_scores, 0.0, 1.0)),
        ]
    )
    acoustic_norm = float(np.linalg.norm(acoustic))
    if acoustic_norm <= 0:
        raise ValueError("Stored fingerprint vector is empty")
    return acoustic / acoustic_norm, mean_scores, max_scores


def _episode_feature(
    episode: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
):
    import numpy as np

    acoustic, mean_scores, max_scores = _fingerprint_vector(fingerprint)
    confidences = [
        float(event["confidence"])
        for event in episode["events"]
    ]
    metadata = np.asarray(
        [
            min(
                1.5,
                math.log1p(int(episode["event_count"]))
                / math.log1p(20),
            )
            * 0.30,
            min(
                1.5,
                math.log1p(float(episode["duration_seconds"]))
                / math.log1p(90),
            )
            * 0.30,
            min(2, len(episode["cameras"]) - 1) * 0.25,
            float(episode["max_confidence"]) * 0.15,
            (sum(confidences) / len(confidences)) * 0.15,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([acoustic, metadata])
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Episode feature vector is empty")
    return vector / norm, mean_scores, max_scores


def _similarity_score(
    vector,
    positive_vectors,
    negative_vectors,
) -> tuple[float, float, float, float]:
    import numpy as np

    positive_similarity = float(
        np.max(positive_vectors @ vector)
    )
    negative_similarity = float(
        np.max(negative_vectors @ vector)
    )
    margin = positive_similarity - negative_similarity
    score = 1.0 / (1.0 + math.exp(-12.0 * margin))
    return score, margin, positive_similarity, negative_similarity


def _bucket(score: float) -> str:
    if score >= 0.70:
        return "leans_confirmed"
    if score <= 0.30:
        return "leans_false_positive"
    return "uncertain"


def _reason(
    *,
    score: float,
    positive_similarity: float,
    negative_similarity: float,
    top_class: str,
) -> str:
    if score >= 0.70:
        prefix = "声学指纹更接近已确认样本"
    elif score <= 0.30:
        prefix = "声学指纹更接近已标记误报样本"
    else:
        prefix = "正负样本距离接近，值得人工确认"
    return (
        f"{prefix}；主要声音类别：{top_class}；"
        f"相似度 {positive_similarity:.3f}/{negative_similarity:.3f}"
    )


def _classification_metrics(
    truth_and_scores: Iterable[tuple[str, float]],
) -> dict[str, Any]:
    counts = Counter()
    for truth, score in truth_and_scores:
        predicted = "confirmed" if score >= 0.5 else "false_positive"
        counts[(truth, predicted)] += 1
    true_positive = counts[("confirmed", "confirmed")]
    false_negative = counts[("confirmed", "false_positive")]
    true_negative = counts[("false_positive", "false_positive")]
    false_positive = counts[("false_positive", "confirmed")]
    positive_total = true_positive + false_negative
    negative_total = true_negative + false_positive
    recall = true_positive / positive_total if positive_total else None
    specificity = true_negative / negative_total if negative_total else None
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "samples": positive_total + negative_total,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def _score_labeled_holdout(
    labeled: list[dict[str, Any]],
    *,
    holdout: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> list[tuple[str, float]]:
    import numpy as np

    results: list[tuple[str, float]] = []
    for target in labeled:
        references = [
            item
            for item in labeled
            if not holdout(target, item)
        ]
        positive = [
            item["vector"]
            for item in references
            if item["status"] == "confirmed"
        ]
        negative = [
            item["vector"]
            for item in references
            if item["status"] == "false_positive"
        ]
        if not positive or not negative:
            continue
        score, _, _, _ = _similarity_score(
            target["vector"],
            np.stack(positive),
            np.stack(negative),
        )
        results.append((target["status"], score))
    return results


def run_shadow_calibration(
    *,
    database_path: str | Path,
    source_root: str | Path,
    timezone: str,
    model_path: str | Path,
    class_map_path: str | Path,
    event_label: str = "baby_cry",
    minimum_confidence: float = 0.60,
    gap_seconds: float = 30.0,
    workers: int = 16,
    padding_seconds: float = 0.5,
    timeout_seconds: float = 30.0,
    ffmpeg: str | None = None,
    output_path: str | Path | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0 < minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in (0, 1]")
    if gap_seconds < 0:
        raise ValueError("gap_seconds cannot be negative")

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Shadow calibration requires the audio-events dependencies"
        ) from exc

    database = Path(database_path)
    media_root = Path(source_root)
    model = Path(model_path)
    class_map = Path(class_map_path)
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    class_names = load_class_names(class_map)
    now = int(time.time())

    connection = connect_database(database)
    run_id = int(
        connection.execute(
            """
            INSERT INTO shadow_calibration_runs(
                event_label, created_at, status, model_name, model_sha256,
                minimum_confidence, gap_seconds
            )
            VALUES (?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                event_label,
                now,
                SHADOW_MODEL_NAME,
                model_sha256,
                minimum_confidence,
                gap_seconds,
            ),
        ).lastrowid
    )
    connection.commit()

    try:
        rows = query_episode_events(
            connection,
            event_label=event_label,
            minimum_confidence=minimum_confidence,
        )
        episodes = build_daily_episodes(
            rows,
            timezone=ZoneInfo(timezone),
            gap_seconds=gap_seconds,
        )
        representative_ids = [
            int(episode["representative_event"]["event_id"])
            for episode in episodes
        ]
        fingerprints = query_audio_event_fingerprints(
            connection,
            event_ids=representative_ids,
            model_sha256=model_sha256,
        )
        missing = [
            episode
            for episode in episodes
            if int(episode["representative_event"]["event_id"])
            not in fingerprints
        ]

        local = threading.local()

        def detector() -> YamnetDetector:
            instance = getattr(local, "detector", None)
            if instance is None:
                instance = YamnetDetector(
                    model_path=model,
                    class_map_path=class_map,
                    event_labels=(event_label,),
                    minimum_confidence=minimum_confidence,
                    merge_gap_seconds=1.0,
                )
                local.detector = instance
            return instance

        def extract(episode: Mapping[str, Any]):
            event = episode["representative_event"]
            path = _safe_media_path(
                media_root,
                str(event["relative_path"]),
            )
            fingerprint = detector().fingerprint(
                path,
                media_start_seconds=float(event["media_start_seconds"]),
                media_end_seconds=float(event["media_end_seconds"]),
                padding_seconds=padding_seconds,
                ffmpeg=ffmpeg,
                timeout_seconds=timeout_seconds,
            )
            return int(event["event_id"]), fingerprint

        completed = 0
        errors: list[dict[str, Any]] = []
        if missing:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(extract, episode): episode
                    for episode in missing
                }
                for future in as_completed(futures):
                    episode = futures[future]
                    event = episode["representative_event"]
                    try:
                        event_id, fingerprint = future.result()
                        mean_blob = np.asarray(
                            fingerprint.mean_scores,
                            dtype="<f4",
                        ).tobytes()
                        max_blob = np.asarray(
                            fingerprint.max_scores,
                            dtype="<f4",
                        ).tobytes()
                        save_audio_event_fingerprint(
                            connection,
                            audio_event_id=event_id,
                            model_sha256=fingerprint.model_sha256,
                            vector_length=len(fingerprint.mean_scores),
                            mean_scores=mean_blob,
                            max_scores=max_blob,
                            windows_seen=fingerprint.windows_seen,
                            media_start_seconds=(
                                fingerprint.media_start_seconds
                            ),
                            media_end_seconds=(
                                fingerprint.media_end_seconds
                            ),
                        )
                        completed += 1
                        if completed % 20 == 0:
                            connection.commit()
                    except Exception as exc:
                        errors.append(
                            {
                                "episode_id": episode["episode_id"],
                                "event_id": int(event["event_id"]),
                                "relative_path": event["relative_path"],
                                "error": str(exc),
                            }
                        )
                    done = completed + len(errors)
                    if on_progress and (
                        done == len(missing) or done % 25 == 0
                    ):
                        on_progress(
                            {
                                "stage": "fingerprints",
                                "completed": completed,
                                "failed": len(errors),
                                "total": len(missing),
                                "cached": len(fingerprints),
                            }
                        )
            connection.commit()

        fingerprints = query_audio_event_fingerprints(
            connection,
            event_ids=representative_ids,
            model_sha256=model_sha256,
        )
        featured: list[dict[str, Any]] = []
        for episode in episodes:
            event_id = int(
                episode["representative_event"]["event_id"]
            )
            fingerprint = fingerprints.get(event_id)
            if fingerprint is None:
                continue
            vector, mean_scores, max_scores = _episode_feature(
                episode,
                fingerprint,
            )
            top_indexes = np.argsort(max_scores)[-5:][::-1]
            featured.append(
                {
                    "episode": episode,
                    "vector": vector,
                    "mean_scores": mean_scores,
                    "max_scores": max_scores,
                    "top_classes": [
                        {
                            "name": class_names[int(index)],
                            "score": float(max_scores[int(index)]),
                        }
                        for index in top_indexes
                    ],
                }
            )

        labeled = [
            {
                **item,
                "status": item["episode"]["review_status"],
                "date": item["episode"]["date"],
            }
            for item in featured
            if item["episode"]["review_status"]
            in FINAL_REVIEW_STATUSES
        ]
        positive = [
            item["vector"]
            for item in labeled
            if item["status"] == "confirmed"
        ]
        negative = [
            item["vector"]
            for item in labeled
            if item["status"] == "false_positive"
        ]
        if len(positive) < 2 or len(negative) < 2:
            metrics = {
                "decision_mode": "shadow_only",
                "reason": "至少需要两个确认和两个误报事件段",
                "fingerprint_errors": errors,
            }
            connection.execute(
                """
                UPDATE shadow_calibration_runs
                SET finished_at = ?, status = 'insufficient',
                    positive_episodes = ?, negative_episodes = ?,
                    fingerprinted_episodes = ?, metrics_json = ?
                WHERE id = ?
                """,
                (
                    int(time.time()),
                    len(positive),
                    len(negative),
                    len(featured),
                    json.dumps(metrics, ensure_ascii=False),
                    run_id,
                ),
            )
            connection.commit()
            return {
                "run_id": run_id,
                "status": "insufficient",
                "metrics": metrics,
            }

        positive_matrix = np.stack(positive)
        negative_matrix = np.stack(negative)
        score_rows: list[dict[str, Any]] = []
        for item in featured:
            episode = item["episode"]
            if episode["review_status"] in FINAL_REVIEW_STATUSES:
                references = [
                    other
                    for other in labeled
                    if other["episode"]["episode_id"]
                    != episode["episode_id"]
                ]
                positive_refs = [
                    other["vector"]
                    for other in references
                    if other["status"] == "confirmed"
                ]
                negative_refs = [
                    other["vector"]
                    for other in references
                    if other["status"] == "false_positive"
                ]
                positive_for_score = np.stack(positive_refs)
                negative_for_score = np.stack(negative_refs)
            else:
                positive_for_score = positive_matrix
                negative_for_score = negative_matrix
            score, margin, positive_similarity, negative_similarity = (
                _similarity_score(
                    item["vector"],
                    positive_for_score,
                    negative_for_score,
                )
            )
            novelty = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - max(positive_similarity, negative_similarity),
                ),
            )
            uncertainty = 1.0 - min(1.0, abs(score - 0.5) * 2.0)
            review_priority = uncertainty * 0.75 + novelty * 0.25
            top_class = item["top_classes"][0]["name"]
            feature = {
                "date": episode["date"],
                "event_count": episode["event_count"],
                "duration_seconds": episode["duration_seconds"],
                "camera_count": len(episode["cameras"]),
                "max_confidence": episode["max_confidence"],
                "representative_event_id": int(
                    episode["representative_event"]["event_id"]
                ),
                "top_classes": item["top_classes"],
            }
            score_rows.append(
                {
                    "run_id": run_id,
                    "episode_id": episode["episode_id"],
                    "event_label": event_label,
                    "start_ts": float(episode["start_ts"]),
                    "review_status": episode["review_status"],
                    "score": score,
                    "margin": margin,
                    "positive_similarity": positive_similarity,
                    "negative_similarity": negative_similarity,
                    "review_priority": review_priority,
                    "bucket": _bucket(score),
                    "reason": _reason(
                        score=score,
                        positive_similarity=positive_similarity,
                        negative_similarity=negative_similarity,
                        top_class=top_class,
                    ),
                    "event_ids_json": json.dumps(
                        episode["event_ids"],
                        separators=(",", ":"),
                    ),
                    "feature_json": json.dumps(
                        feature,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "feature": feature,
                }
            )

        loo_results = _score_labeled_holdout(
            labeled,
            holdout=lambda target, reference: (
                target["episode"]["episode_id"]
                == reference["episode"]["episode_id"]
            ),
        )
        date_results = _score_labeled_holdout(
            labeled,
            holdout=lambda target, reference: (
                target["date"] == reference["date"]
            ),
        )
        unreviewed = [
            row
            for row in score_rows
            if row["review_status"] not in FINAL_REVIEW_STATUSES
        ]
        bucket_counts = Counter(row["bucket"] for row in unreviewed)
        metrics = {
            "decision_mode": "shadow_only",
            "training_unit": "episode",
            "model": SHADOW_MODEL_NAME,
            "source_model": MODEL_NAME,
            "positive_episodes": len(positive),
            "negative_episodes": len(negative),
            "fingerprinted_episodes": len(featured),
            "fingerprint_errors": errors,
            "leave_one_episode_out": _classification_metrics(loo_results),
            "leave_one_date_out": _classification_metrics(date_results),
            "unreviewed_bucket_counts": dict(bucket_counts),
            "auto_verdicts_written": 0,
        }

        connection.executemany(
            """
            INSERT INTO shadow_episode_scores(
                run_id, episode_id, event_label, start_ts, review_status,
                score, margin, positive_similarity, negative_similarity,
                review_priority, bucket, reason, event_ids_json, feature_json
            )
            VALUES (
                :run_id, :episode_id, :event_label, :start_ts,
                :review_status, :score, :margin, :positive_similarity,
                :negative_similarity, :review_priority, :bucket, :reason,
                :event_ids_json, :feature_json
            )
            """,
            score_rows,
        )
        connection.execute(
            """
            UPDATE shadow_calibration_runs
            SET finished_at = ?, status = 'completed',
                positive_episodes = ?, negative_episodes = ?,
                fingerprinted_episodes = ?, scored_episodes = ?,
                metrics_json = ?
            WHERE id = ?
            """,
            (
                int(time.time()),
                len(positive),
                len(negative),
                len(featured),
                len(score_rows),
                json.dumps(metrics, ensure_ascii=False),
                run_id,
            ),
        )
        connection.commit()

        uncertain = sorted(
            unreviewed,
            key=lambda row: (
                -row["review_priority"],
                row["start_ts"],
            ),
        )[:20]
        result = {
            "run_id": run_id,
            "status": "completed",
            "metrics": metrics,
            "top_review_candidates": [
                {
                    key: row[key]
                    for key in (
                        "episode_id",
                        "start_ts",
                        "score",
                        "review_priority",
                        "bucket",
                        "reason",
                        "feature",
                    )
                }
                for row in uncertain
            ],
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            temporary.replace(target)
            result["output"] = str(target.resolve())
        return result
    except Exception as exc:
        connection.execute(
            """
            UPDATE shadow_calibration_runs
            SET finished_at = ?, status = 'failed', error = ?
            WHERE id = ?
            """,
            (int(time.time()), str(exc), run_id),
        )
        connection.commit()
        raise
    finally:
        connection.close()
