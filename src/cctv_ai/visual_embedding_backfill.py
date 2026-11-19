from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Sequence

from .model_config import VisualEmbeddingConfig
from .vl_models.embedding import create_embedding_model


def _pending_keyframes(
    connection: sqlite3.Connection,
    *,
    model_name: str,
    image_root: Path,
    keyframe_ids: Sequence[int],
    limit: int | None,
) -> list[tuple[int, Path]]:
    conditions = [
        """
        NOT EXISTS (
            SELECT 1
            FROM visual_embeddings e
            WHERE e.keyframe_id = k.id
              AND e.model_name = ?
        )
        """
    ]
    parameters: list[Any] = [model_name]
    if keyframe_ids:
        placeholders = ",".join("?" for _ in keyframe_ids)
        conditions.append(f"k.id IN ({placeholders})")
        parameters.extend(int(value) for value in keyframe_ids)
    sql = f"""
        SELECT k.id, k.relative_image_path
        FROM visual_keyframes k
        WHERE {' AND '.join(conditions)}
        ORDER BY k.id
    """
    if limit is not None:
        sql += "\nLIMIT ?"
        parameters.append(max(0, int(limit)))
    result: list[tuple[int, Path]] = []
    for row in connection.execute(sql, parameters):
        path = image_root.joinpath(str(row["relative_image_path"]))
        if path.is_file():
            result.append((int(row["id"]), path))
    return result


def backfill_qwen3_visual_embeddings(
    connection: sqlite3.Connection,
    *,
    config: VisualEmbeddingConfig,
    image_root: str | Path,
    keyframe_ids: Sequence[int] = (),
    limit: int | None = None,
    batch_size: int | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import numpy as np

    root = Path(image_root).resolve()
    rows = _pending_keyframes(
        connection,
        model_name=config.model_id,
        image_root=root,
        keyframe_ids=keyframe_ids,
        limit=limit,
    )
    payload = {
        "status": "completed",
        "model": config.model_id,
        "pending": len(rows),
        "saved": 0,
        "missing_images": 0,
        "dimensions": config.dimensions,
    }
    if not rows:
        return payload
    encoder = create_embedding_model(config)
    selected_batch_size = max(1, batch_size or config.batch_size)
    try:
        for start in range(0, len(rows), selected_batch_size):
            batch = rows[start : start + selected_batch_size]
            vectors = encoder.encode_images([item[1] for item in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    "Embedding model returned an unexpected vector count"
                )
            now_ts = int(time.time())
            values = []
            for (keyframe_id, _path), vector in zip(
                batch,
                vectors,
                strict=True,
            ):
                normalized = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(normalized))
                if not norm:
                    raise ValueError(
                        f"Empty embedding for keyframe {keyframe_id}"
                    )
                normalized /= norm
                compact = normalized.astype("<f2", copy=False)
                values.append(
                    (
                        keyframe_id,
                        config.model_id,
                        int(compact.shape[0]),
                        "float16-le",
                        compact.tobytes(),
                        float(np.linalg.norm(normalized)),
                        now_ts,
                    )
                )
            connection.executemany(
                """
                INSERT INTO visual_embeddings(
                    keyframe_id, model_name, dimensions, dtype,
                    vector, vector_norm, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(keyframe_id, model_name) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    dtype = excluded.dtype,
                    vector = excluded.vector,
                    vector_norm = excluded.vector_norm,
                    created_at = excluded.created_at
                """,
                values,
            )
            connection.commit()
            payload["saved"] += len(values)
            if on_progress is not None:
                on_progress(dict(payload))
    finally:
        encoder.close()
    return payload
