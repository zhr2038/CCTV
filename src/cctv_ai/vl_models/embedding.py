from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..model_config import VisualEmbeddingConfig
from .qwen3_vl import _checkpoint_complete


class VisionEmbeddingModel(ABC):
    model_name: str
    dimensions: int

    @abstractmethod
    def encode_text(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def encode_images(self, image_paths: Sequence[str | Path]):
        raise NotImplementedError

    def close(self) -> None:
        return None


class Qwen3VLEmbeddingModel(VisionEmbeddingModel):
    """Local-only Qwen3-VL shared-space image/text encoder."""

    def __init__(self, config: VisualEmbeddingConfig) -> None:
        if not config.local_files_only:
            raise ValueError("Embedding runtime must remain local_files_only")
        model_path = config.model_path.resolve()
        if not _checkpoint_complete(model_path):
            raise FileNotFoundError(
                "Local Qwen3-VL embedding model is incomplete or missing: "
                f"{model_path}. Download it explicitly before backfilling."
            )
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
            )
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLModel,
                Qwen3VLPreTrainedModel,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL embedding dependencies are missing"
            ) from exc
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        quantization_config = None
        if config.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.quantization not in {"bf16", "none"}:
            raise ValueError(
                f"Unsupported quantization: {config.quantization}"
            )
        class LocalQwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            # This is the minimal inference wrapper published by Qwen's
            # official Qwen3-VL-Embedding repository. The checkpoint contains
            # base `model.*` weights and intentionally has no LM head.
            _checkpoint_conversion_mapping = {}
            accepts_loss_kwargs = False

            def __init__(self, model_config):
                super().__init__(model_config)
                self.model = Qwen3VLModel(model_config)
                self.post_init()

            def forward(self, **kwargs):
                return self.model(**kwargs)

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": (
                torch.bfloat16
                if config.device == "cuda"
                else torch.float32
            ),
            "device_map": "auto" if config.device == "cuda" else "cpu",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        self._torch = torch
        self.config = config
        self.model_name = config.model_id
        self.dimensions = config.dimensions
        self._functional = functional
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            padding_side="right",
            local_files_only=True,
        )
        self._model = LocalQwen3VLForEmbedding.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self._model.eval()
        self._input_device = next(self._model.parameters()).device

    @staticmethod
    def _messages(
        item: Mapping[str, Any],
        instruction: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if item.get("image") is not None:
            content.append(
                {"type": "image", "image": item["image"]}
            )
        if item.get("text") is not None:
            content.append(
                {"type": "text", "text": str(item["text"])}
            )
        if not content:
            content.append({"type": "text", "text": "NULL"})
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": instruction.strip()}
                ],
            },
            {"role": "user", "content": content},
        ]

    def _prepare_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        instruction: str,
    ):
        from PIL import Image

        opened: list[Any] = []
        messages: list[list[dict[str, Any]]] = []
        try:
            for item in items:
                normalized = dict(item)
                image_value = normalized.get("image")
                if isinstance(image_value, (str, Path)):
                    source = Image.open(image_value)
                    try:
                        image = source.convert("RGB")
                    finally:
                        source.close()
                    normalized["image"] = image
                    opened.append(image)
                messages.append(
                    self._messages(normalized, instruction)
                )
            texts = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            images = [
                content["image"]
                for message in messages
                for content in message[1]["content"]
                if content["type"] == "image"
            ]
            return self._processor(
                text=texts,
                images=images or None,
                truncation=True,
                max_length=8192,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in opened:
                image.close()

    def _encode_prepared(self, inputs):
        inputs = inputs.to(self._input_device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        flipped = attention_mask.flip(dims=[1])
        last_one_positions = flipped.argmax(dim=1)
        columns = attention_mask.shape[1] - last_one_positions - 1
        rows = self._torch.arange(
            hidden.shape[0],
            device=hidden.device,
        )
        embeddings = hidden[rows, columns]
        embeddings = embeddings[:, : self.dimensions]
        embeddings = self._functional.normalize(
            embeddings.float(),
            p=2,
            dim=-1,
        )
        return embeddings.detach().cpu().numpy()

    def _encode_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        instruction: str,
    ):
        inputs = self._prepare_items(items, instruction=instruction)
        return self._encode_prepared(inputs)

    def encode_text(self, text: str):
        if not text.strip():
            raise ValueError("Search text cannot be empty")
        return self._encode_items(
            [{"text": text.strip()}],
            instruction=self.config.query_instruction,
        )[0]

    def encode_images(self, image_paths: Sequence[str | Path]):
        if not image_paths:
            return []
        import numpy as np

        path_batches = [
            image_paths[start : start + self.config.batch_size]
            for start in range(
                0,
                len(image_paths),
                self.config.batch_size,
            )
        ]
        instruction = (
            "Represent this family CCTV frame for visual retrieval."
        )

        def prepare(paths: Sequence[str | Path]):
            return self._prepare_items(
                [
                    {"image": str(Path(path).resolve())}
                    for path in paths
                ],
                instruction=instruction,
            )

        # Preparing images and token tensors on the CPU previously left the
        # GPU idle between every model call. Keep one batch prepared ahead so
        # CPU work overlaps the current CUDA forward pass without loading a
        # second copy of the model into VRAM.
        batches = []
        prepared = prepare(path_batches[0])
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen3-vl-prefetch",
        ) as executor:
            for index in range(len(path_batches)):
                next_prepared = (
                    executor.submit(prepare, path_batches[index + 1])
                    if index + 1 < len(path_batches)
                    else None
                )
                batches.append(self._encode_prepared(prepared))
                if next_prepared is not None:
                    prepared = next_prepared.result()
        return np.concatenate(batches, axis=0)

    def close(self) -> None:
        del self._model
        del self._processor
        if self.config.device == "cuda":
            self._torch.cuda.empty_cache()


def create_embedding_model(
    config: VisualEmbeddingConfig,
) -> VisionEmbeddingModel:
    if config.provider == "qwen3-vl-embedding":
        return Qwen3VLEmbeddingModel(config)
    raise ValueError(
        f"Unsupported visual embedding provider: {config.provider}"
    )
