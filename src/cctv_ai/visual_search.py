from __future__ import annotations

import heapq
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

from .model_config import ModelsConfig, load_models_config
from .vl_models import (
    VisionAnalysisContext,
    VisionLanguageModel,
    create_vision_language_model,
)
from .vl_models.embedding import (
    VisionEmbeddingModel,
    create_embedding_model,
)
from .visual_models import DEFAULT_EMBEDDING_MODEL, SiglipVisualEncoder


ACTIVITY_SEARCH_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "feeding",
        ("喝奶", "喂奶", "奶瓶", "进食", "辅食", "母乳", "feeding"),
    ),
    ("inferred_sleep", ("睡觉", "睡眠", "睡着", "入睡", "sleep")),
    ("crawling", ("爬行", "在爬", "爬了", "crawling")),
    ("sitting", ("坐起来", "坐着", "坐姿", "坐稳", "sitting")),
    ("standing", ("站起来", "站着", "站立", "standing")),
    ("play", ("玩耍", "玩具", "积木", "绘本", "玩", "play")),
)

ACTIVITY_SEARCH_NAMES = {
    "feeding": "喝奶 / 进食",
    "inferred_sleep": "推定睡眠",
    "crawling": "爬行姿态",
    "sitting": "坐姿",
    "standing": "站立",
    "play": "玩耍",
}


def activity_type_for_query(query: str) -> str | None:
    normalized = "".join(str(query).lower().split())
    for activity_type, terms in ACTIVITY_SEARCH_TERMS:
        if any(term in normalized for term in terms):
            return activity_type
    return None


def search_activity_event_evidence(
    connection: sqlite3.Connection,
    *,
    query: str,
    since_ts: float | None = None,
    until_ts: float | None = None,
    camera_ids: Sequence[str] = (),
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return existing structured events before open-ended vector recall.

    A query such as ``宝宝喝奶`` must not return zero while the diary already
    contains feeding events.  These rows are still labelled as machine or
    family-confirmed observations; they are not promoted to ground truth.
    """
    activity_type = activity_type_for_query(query)
    if activity_type is None or limit <= 0:
        return []
    conditions = [
        "e.activity_type = ?",
        "COALESCE(r.verdict, e.review_status) != 'rejected'",
    ]
    parameters: list[Any] = [activity_type]
    if since_ts is not None:
        conditions.append("e.end_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("e.start_ts < ?")
        parameters.append(until_ts)
    if camera_ids:
        placeholders = ",".join("?" for _ in camera_ids)
        conditions.append(f"m.camera_id IN ({placeholders})")
        parameters.extend(camera_ids)
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT
            e.id AS activity_event_id,
            e.activity_type,
            e.activity_subtype,
            e.confidence,
            e.review_status,
            r.verdict AS review_verdict,
            k.id AS keyframe_id,
            k.wall_ts,
            k.wall_time_source,
            k.media_offset_ms,
            k.relative_image_path,
            k.brightness,
            k.motion_score,
            m.id AS media_file_id,
            m.camera_id,
            m.relative_path,
            c.name AS camera_name,
            c.location
        FROM activity_events e
        LEFT JOIN activity_event_reviews r
          ON r.activity_event_id = e.id
        JOIN activity_event_evidence ev
          ON ev.activity_event_id = e.id
         AND ev.evidence_rank = (
             SELECT MIN(first_ev.evidence_rank)
             FROM activity_event_evidence first_ev
             WHERE first_ev.activity_event_id = e.id
         )
        JOIN visual_keyframes k ON k.id = ev.keyframe_id
        JOIN media_files m ON m.id = k.media_file_id
        JOIN cameras c ON c.id = m.camera_id
        WHERE {' AND '.join(conditions)}
        ORDER BY
            CASE COALESCE(r.verdict, e.review_status)
                WHEN 'confirmed' THEN 0
                WHEN 'corrected' THEN 1
                ELSE 2
            END,
            e.start_ts DESC,
            e.confidence DESC
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        review_status = str(row["review_verdict"] or row["review_status"])
        results.append(
            {
                "keyframe_id": int(row["keyframe_id"]),
                "score": float(row["confidence"]),
                "wall_ts": float(row["wall_ts"]),
                "wall_time_source": str(row["wall_time_source"]),
                "media_offset_seconds": (
                    float(row["media_offset_ms"]) / 1000.0
                ),
                "media_file_id": int(row["media_file_id"]),
                "camera_id": str(row["camera_id"]),
                "camera_name": str(row["camera_name"]),
                "location": str(row["location"]),
                "relative_path": str(row["relative_path"]),
                "relative_image_path": str(row["relative_image_path"]),
                "brightness": (
                    float(row["brightness"])
                    if row["brightness"] is not None
                    else None
                ),
                "motion_score": (
                    float(row["motion_score"])
                    if row["motion_score"] is not None
                    else None
                ),
                "image_url": f"/api/keyframes/{int(row['keyframe_id'])}/image",
                "activity_event_id": int(row["activity_event_id"]),
                "activity_type": str(row["activity_type"]),
                "activity_subtype": row["activity_subtype"],
                "activity_name": ACTIVITY_SEARCH_NAMES.get(
                    str(row["activity_type"]),
                    str(row["activity_type"]),
                ),
                "match_source": "structured_event",
                "verification_status": (
                    "family_confirmed"
                    if review_status in {"confirmed", "corrected"}
                    else "machine_event"
                ),
            }
        )
    return results


class VisualSearchService:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model_cache: str | Path = ".data/models/huggingface",
        database_path: str | Path | None = None,
        models_config_path: str | Path | None = "config/models.toml",
    ) -> None:
        self.model_name = model_name
        self.model_cache = Path(model_cache)
        self._encoder: SiglipVisualEncoder | None = None
        self.database_path = (
            Path(database_path) if database_path is not None else None
        )
        self.models_config: ModelsConfig | None = None
        if (
            models_config_path is not None
            and Path(models_config_path).is_file()
        ):
            self.models_config = load_models_config(models_config_path)
        self._qwen_encoder: VisionEmbeddingModel | None = None
        self._search_verifier: VisionLanguageModel | None = None
        self._lock = threading.Lock()

    def _get_encoder(self) -> SiglipVisualEncoder:
        if self._encoder is None:
            self._encoder = SiglipVisualEncoder(
                self.model_name,
                # Interactive search must remain available while the visual
                # worker owns nearly all RTX memory.  The lightweight legacy
                # text encoder is fast enough on CPU and avoids loading a
                # second 8B model into an already full GPU.
                device="cpu",
                local_cache=self.model_cache,
            )
        return self._encoder

    def _get_qwen_encoder(self) -> VisionEmbeddingModel:
        if self.models_config is None:
            raise RuntimeError("Qwen3 embedding configuration is unavailable")
        if self._qwen_encoder is None:
            self._qwen_encoder = create_embedding_model(
                self.models_config.visual_embedding
            )
        return self._qwen_encoder

    def select_model(
        self,
        connection: sqlite3.Connection | None = None,
    ) -> str:
        if self.models_config is None:
            return self.model_name
        embedding = self.models_config.visual_embedding
        owns_connection = False
        if connection is None and self.database_path is not None:
            from .database import connect_database

            connection = connect_database(self.database_path)
            owns_connection = True
        try:
            indexed = 0
            total = 0
            if connection is not None:
                indexed = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM visual_embeddings
                        WHERE model_name = ?
                        """,
                        (embedding.model_id,),
                    ).fetchone()[0]
                )
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM visual_keyframes"
                    ).fetchone()[0]
                )
            coverage = indexed / total if total else 0.0
            if (
                indexed > 0
                and coverage >= embedding.activation_coverage
                and embedding.model_path.is_dir()
            ):
                return embedding.model_id
            if embedding.fallback_until_indexed:
                return embedding.legacy_model
            return embedding.model_id
        finally:
            if owns_connection and connection is not None:
                connection.close()

    def index_progress(
        self,
        connection: sqlite3.Connection,
    ) -> dict[str, Any]:
        """Describe Qwen3 retrieval readiness without loading either model."""
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM visual_keyframes"
            ).fetchone()[0]
        )
        if self.models_config is None:
            indexed = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM visual_embeddings
                    WHERE model_name = ?
                    """,
                    (self.model_name,),
                ).fetchone()[0]
            )
            return {
                "status": "legacy",
                "model": self.model_name,
                "indexed_keyframes": indexed,
                "total_keyframes": total,
                "coverage_percent": (
                    round(indexed * 100.0 / total, 2) if total else 0.0
                ),
                "activation_percent": 100.0,
                "required_keyframes": total,
                "remaining_keyframes": max(total - indexed, 0),
                "model_available": True,
                "active": True,
                "active_model": self.model_name,
                "fallback_model": None,
            }

        embedding = self.models_config.visual_embedding
        indexed = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM visual_embeddings
                WHERE model_name = ?
                """,
                (embedding.model_id,),
            ).fetchone()[0]
        )
        required = math.ceil(total * embedding.activation_coverage)
        coverage = indexed / total if total else 0.0
        available = embedding.model_path.is_dir()
        active_model = self.select_model(connection)
        active = (
            indexed > 0
            and coverage >= embedding.activation_coverage
            and available
            and active_model == embedding.model_id
        )
        status = (
            "active"
            if active
            else "building"
            if available
            else "model_missing"
        )
        return {
            "status": status,
            "model": embedding.model_id,
            "indexed_keyframes": indexed,
            "total_keyframes": total,
            "coverage_percent": round(coverage * 100.0, 2),
            "activation_percent": round(
                embedding.activation_coverage * 100.0,
                2,
            ),
            "required_keyframes": required,
            "remaining_keyframes": max(required - indexed, 0),
            "model_available": available,
            "active": active,
            "active_model": active_model,
            "fallback_model": embedding.legacy_model,
        }

    def encode_query(
        self,
        text: str,
        model_name: str | None = None,
    ):
        with self._lock:
            selected = model_name or self.select_model()
            if (
                self.models_config is not None
                and selected
                == self.models_config.visual_embedding.model_id
            ):
                return self._get_qwen_encoder().encode_text(text)
            return self._get_encoder().encode_text(text)

    def _get_search_verifier(self) -> VisionLanguageModel:
        if self.models_config is None:
            raise RuntimeError("Vision-language configuration is unavailable")
        if self._search_verifier is None:
            self._search_verifier = create_vision_language_model(
                self.models_config.vision_language
            )
        return self._search_verifier

    def verify_results(
        self,
        query: str,
        results: Sequence[dict[str, Any]],
        *,
        image_root: str | Path,
        max_candidates: int = 12,
    ) -> list[dict[str, Any]]:
        """Use 8B only after embedding recall, never as the search index."""
        if not results:
            return []
        root = Path(image_root)
        candidates = list(results[:max(1, max_candidates)])
        valid: list[dict[str, Any]] = []
        images: list[Path] = []
        contexts: list[VisionAnalysisContext] = []
        for result in candidates:
            relative = result.get("relative_image_path")
            if not relative:
                continue
            path = root.joinpath(str(relative))
            if not path.is_file():
                continue
            valid.append(result)
            images.append(path)
            motion_score = result.get("motion_score")
            motion = (
                "unknown"
                if motion_score is None
                else "active" if float(motion_score) >= 0.12
                else "still" if float(motion_score) <= 0.03
                else "low"
            )
            detections = result.get("detections", ())
            contexts.append(
                VisionAnalysisContext(
                    camera=str(result.get("camera_name", "unknown")),
                    time=str(result.get("wall_ts", "unknown")),
                    objects=tuple(
                        str(item.get("class_name", ""))
                        for item in detections
                        if item.get("class_name")
                    ),
                    motion=motion,
                    detections=tuple(detections),
                )
            )
        if not valid:
            return []
        prompt = (
            f'验证这张召回图是否直接满足搜索条件：“{query}”。'
            "只有条件本身在画面中清楚可见时，才输出实际 activity、"
            "uncertain=false 且 confidence>=0.55；不匹配或无法判断时，"
            "输出 activity=unclear、uncertain=true。对于“第一次”，"
            "只验证动作本身，最终由数据库按时间确定最早记录。"
        )
        events = self._get_search_verifier().analyze_images(
            images,
            [prompt] * len(images),
            contexts,
        )
        verified: list[dict[str, Any]] = []
        for result, event in zip(valid, events, strict=True):
            enriched = dict(result)
            enriched["vl_verification"] = event.to_dict()
            if not event.uncertain and event.confidence >= 0.55:
                verified.append(enriched)
        return verified

    def close(self) -> None:
        if self._qwen_encoder is not None:
            self._qwen_encoder.close()
        if self._search_verifier is not None:
            self._search_verifier.close()


def search_keyframes(
    connection: sqlite3.Connection,
    *,
    query_vector,
    model_name: str,
    since_ts: float | None = None,
    until_ts: float | None = None,
    camera_ids: Sequence[str] = (),
    limit: int = 40,
) -> list[dict[str, Any]]:
    import numpy as np

    if limit <= 0:
        return []
    conditions = [
        "e.model_name = ?",
        "k.wall_ts IS NOT NULL",
    ]
    parameters: list[Any] = [model_name]
    if since_ts is not None:
        conditions.append("k.wall_ts >= ?")
        parameters.append(since_ts)
    if until_ts is not None:
        conditions.append("k.wall_ts < ?")
        parameters.append(until_ts)
    if camera_ids:
        placeholders = ",".join("?" for _ in camera_ids)
        conditions.append(f"m.camera_id IN ({placeholders})")
        parameters.extend(camera_ids)
    cursor = connection.execute(
        f"""
        SELECT
            k.id AS keyframe_id,
            k.wall_ts,
            k.wall_time_source,
            k.media_offset_ms,
            k.relative_image_path,
            k.brightness,
            k.motion_score,
            m.id AS media_file_id,
            m.camera_id,
            m.relative_path,
            c.name AS camera_name,
            c.location,
            e.dimensions,
            e.dtype,
            e.vector
        FROM visual_embeddings e
        JOIN visual_keyframes k ON k.id = e.keyframe_id
        JOIN media_files m ON m.id = k.media_file_id
        JOIN cameras c ON c.id = m.camera_id
        WHERE {' AND '.join(conditions)}
        ORDER BY k.id
        """,
        parameters,
    )
    normalized_query = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(normalized_query))
    if not query_norm:
        raise ValueError("Search query produced an empty vector")
    normalized_query /= query_norm
    heap: list[tuple[float, int, dict[str, Any]]] = []
    keep = max(limit * 12, 120)
    while True:
        rows = cursor.fetchmany(2048)
        if not rows:
            break
        vectors = []
        valid_rows = []
        for row in rows:
            if str(row["dtype"]) != "float16-le":
                continue
            vector = np.frombuffer(row["vector"], dtype="<f2").astype(
                np.float32,
                copy=False,
            )
            if vector.shape[0] != normalized_query.shape[0]:
                continue
            vectors.append(vector)
            valid_rows.append(row)
        if not vectors:
            continue
        matrix = np.stack(vectors)
        scores = matrix @ normalized_query
        for row, score_value in zip(valid_rows, scores, strict=True):
            score = float(score_value)
            item = (
                score,
                int(row["keyframe_id"]),
                dict(row),
            )
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)
    ranked = sorted(heap, key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    for score, _keyframe_id, row in ranked:
        if any(
            existing["camera_id"] == row["camera_id"]
            and abs(existing["wall_ts"] - float(row["wall_ts"])) < 20
            for existing in selected
        ):
            continue
        selected.append(
            {
                "keyframe_id": int(row["keyframe_id"]),
                "score": score,
                "wall_ts": float(row["wall_ts"]),
                "wall_time_source": str(row["wall_time_source"]),
                "media_offset_seconds": (
                    float(row["media_offset_ms"]) / 1000.0
                ),
                "media_file_id": int(row["media_file_id"]),
                "camera_id": str(row["camera_id"]),
                "camera_name": str(row["camera_name"]),
                "location": str(row["location"]),
                "relative_path": str(row["relative_path"]),
                "relative_image_path": str(row["relative_image_path"]),
                "brightness": (
                    float(row["brightness"])
                    if row["brightness"] is not None
                    else None
                ),
                "motion_score": (
                    float(row["motion_score"])
                    if row["motion_score"] is not None
                    else None
                ),
                "image_url": (
                    f"/api/keyframes/{int(row['keyframe_id'])}/image"
                ),
            }
        )
        if len(selected) >= limit:
            break
    if not selected:
        return []
    ids = [item["keyframe_id"] for item in selected]
    detections: dict[int, list[dict[str, Any]]] = {value: [] for value in ids}
    for start in range(0, len(ids), 800):
        batch = ids[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        for row in connection.execute(
            f"""
            SELECT keyframe_id, class_name, confidence
            FROM visual_detections
            WHERE keyframe_id IN ({placeholders})
              AND confidence >= 0.25
            ORDER BY keyframe_id, confidence DESC
            """,
            batch,
        ):
            target = detections[int(row["keyframe_id"])]
            if len(target) < 5:
                target.append(
                    {
                        "class_name": str(row["class_name"]),
                        "confidence": float(row["confidence"]),
                    }
                )
    for item in selected:
        item["detections"] = detections[item["keyframe_id"]]
    return selected
