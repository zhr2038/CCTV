from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly download a configured local visual model. Runtime "
            "analysis never calls this downloader."
        )
    )
    parser.add_argument(
        "--model",
        help="Optional repository id override",
    )
    parser.add_argument(
        "--models-config",
        default="config/models.toml",
    )
    parser.add_argument(
        "--role",
        choices=("vision-language", "high-value", "embedding"),
        default="vision-language",
    )
    parser.add_argument(
        "--cache-dir",
        default=".data/models/huggingface",
    )
    parser.add_argument(
        "--local-dir",
        help="Optional local destination override",
    )
    parser.add_argument(
        "--source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
    )
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    from cctv_ai.model_config import load_models_config

    configured = load_models_config(args.models_config)
    if args.role == "vision-language":
        role_config = configured.vision_language
    elif args.role == "high-value":
        role_config = configured.high_value
    else:
        role_config = configured.visual_embedding
    model = args.model or role_config.model_id
    local_dir = Path(
        args.local_dir or role_config.model_path
    ).resolve()
    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max(1, args.retries) + 1):
        try:
            if args.source == "modelscope":
                from modelscope import snapshot_download

                downloaded = snapshot_download(
                    model,
                    local_dir=str(local_dir),
                    max_workers=8,
                )
            else:
                from huggingface_hub import snapshot_download

                downloaded = snapshot_download(
                    model,
                    cache_dir=cache,
                    local_dir=str(local_dir),
                )
            print(f"{model}\t{downloaded}", flush=True)
            return 0
        except Exception as exc:
            if attempt >= args.retries:
                raise
            delay = min(60, attempt * 10)
            print(
                f"download attempt {attempt} failed: {exc}; "
                f"retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
