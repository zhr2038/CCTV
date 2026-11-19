from __future__ import annotations

from ..model_config import VisionLanguageConfig
from .qwen3_vl import Qwen3VisionLanguageModel


class Qwen25VisionLanguageModel(Qwen3VisionLanguageModel):
    """Compatibility adapter for existing local Qwen2.5-VL checkpoints.

    Transformers dispatches the correct architecture from the local
    checkpoint config. Keeping this named adapter lets configuration roll
    back without reconnecting business code to a concrete model class.
    """

    def __init__(self, config: VisionLanguageConfig) -> None:
        super().__init__(config)
