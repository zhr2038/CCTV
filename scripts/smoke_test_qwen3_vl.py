from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
import sys

from cctv_ai.model_config import load_models_config
from cctv_ai.vl_models import (
    VisionAnalysisContext,
    create_vision_language_model,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run one read-only Qwen3-VL test on a real keyframe."
    )
    parser.add_argument("--db", default=".data/cctv_index.sqlite3")
    parser.add_argument(
        "--models-config",
        default="config/models.toml",
    )
    parser.add_argument(
        "--role",
        choices=("vision-language", "high-value"),
        default="vision-language",
    )
    parser.add_argument(
        "--image-root",
        default=".data/visual/keyframes",
    )
    parser.add_argument("--keyframe-id", type=int, required=True)
    parser.add_argument(
        "--video-frames",
        type=int,
        default=3,
        help="Odd number of neighboring keyframes from the same real video",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT
                k.id, k.media_file_id, k.media_offset_ms, k.wall_ts,
                k.relative_image_path, k.motion_score,
                c.name AS camera_name, m.relative_path
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            WHERE k.id = ?
            """,
            (args.keyframe_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown keyframe: {args.keyframe_id}")
        neighbor_limit = max(1, int(args.video_frames))
        neighbor_rows = list(
            connection.execute(
                """
                SELECT id, media_offset_ms, relative_image_path
                FROM visual_keyframes
                WHERE media_file_id = ?
                ORDER BY ABS(media_offset_ms - ?), media_offset_ms
                LIMIT ?
                """,
                (
                    int(row["media_file_id"]),
                    int(row["media_offset_ms"]),
                    neighbor_limit,
                ),
            )
        )
        neighbor_rows.sort(key=lambda item: int(item["media_offset_ms"]))
        detections = [
            dict(item)
            for item in connection.execute(
                """
                SELECT
                    class_name, confidence, x1, y1, x2, y2
                FROM visual_detections
                WHERE keyframe_id = ?
                  AND confidence >= 0.18
                ORDER BY confidence DESC
                """,
                (args.keyframe_id,),
            )
        ]
    finally:
        connection.close()

    image_path = Path(args.image_root).joinpath(
        str(row["relative_image_path"])
    ).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    frame_paths = [
        Path(args.image_root).joinpath(
            str(item["relative_image_path"])
        ).resolve()
        for item in neighbor_rows
    ]
    frame_paths = [path for path in frame_paths if path.is_file()]
    motion_score = row["motion_score"]
    motion = (
        "unknown"
        if motion_score is None
        else "active" if float(motion_score) >= 0.12
        else "still" if float(motion_score) <= 0.03
        else "low"
    )
    context = VisionAnalysisContext(
        camera=str(row["camera_name"]),
        time=datetime.fromtimestamp(float(row["wall_ts"])).astimezone()
        .isoformat(sep=" ", timespec="seconds"),
        objects=tuple(
            dict.fromkeys(
                str(item["class_name"]) for item in detections
            )
        ),
        motion=motion,
        detections=tuple(detections),
    )
    models = load_models_config(args.models_config)
    selected_config = (
        models.high_value
        if args.role == "high-value"
        else models.vision_language
    )
    load_started = time.perf_counter()
    model = create_vision_language_model(selected_config)
    load_seconds = time.perf_counter() - load_started
    try:
        inference_started = time.perf_counter()
        task_prompt = (
            "根据按时间顺序排列的画面和基础检测上下文判断宝宝此刻的"
            "可见活动。若奶瓶奶嘴直接接触宝宝嘴部，activity 写 "
            "feeding；奶瓶仅在附近则不能判定 feeding。请特别区分"
            "彩色扁平绘本、包装袋、玩具与真正有瓶身和奶嘴的奶瓶。"
        )
        event = (
            model.analyze_video_frames(
                frame_paths,
                task_prompt,
                context,
            )
            if len(frame_paths) > 1
            else model.analyze_image(
                image_path,
                task_prompt,
                context,
            )
        )
        inference_seconds = time.perf_counter() - inference_started
        footprint = getattr(model, "memory_footprint_bytes", lambda: 0)()
        print(
            json.dumps(
                {
                    "status": "completed",
                    "model": model.model_name,
                    "provider": selected_config.provider,
                    "device": selected_config.device,
                    "quantization": selected_config.quantization,
                    "keyframe_id": int(row["id"]),
                    "image": str(image_path),
                    "analysis_mode": (
                        "video_frames"
                        if len(frame_paths) > 1
                        else "single_frame"
                    ),
                    "frame_images": [str(path) for path in frame_paths],
                    "source_video": str(row["relative_path"]),
                    "context": context.to_dict(),
                    "event": event.to_dict(),
                    "load_seconds": round(load_seconds, 3),
                    "inference_seconds": round(inference_seconds, 3),
                    "model_memory_bytes": footprint,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
