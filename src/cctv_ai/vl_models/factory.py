from __future__ import annotations

from ..model_config import VisionLanguageConfig
from .base import VisionLanguageModel


def create_vision_language_model(
    config: VisionLanguageConfig,
) -> VisionLanguageModel:
    if config.provider == "qwen3-vl":
        from .qwen3_vl import Qwen3VisionLanguageModel

        return Qwen3VisionLanguageModel(config)
    if config.provider == "qwen2.5-vl":
        from .qwen25_vl import Qwen25VisionLanguageModel

        return Qwen25VisionLanguageModel(config)
    raise ValueError(
        f"Unsupported vision-language provider: {config.provider}"
    )
