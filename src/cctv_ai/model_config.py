from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib
from typing import Any, Mapping


VALID_QUANTIZATIONS = {"4bit", "8bit", "bf16", "none"}
VALID_VL_PROVIDERS = {"qwen3-vl", "qwen2.5-vl"}
VALID_EMBEDDING_PROVIDERS = {"qwen3-vl-embedding"}


def _model_id(model: str) -> str:
    return model if "/" in model else f"Qwen/{model}"


def _local_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root.joinpath(path)
    return path.resolve()


def _require_local_only(section: Mapping[str, Any], name: str) -> None:
    if not bool(section.get("local_files_only", True)):
        raise ValueError(
            f"{name}.local_files_only must remain true; runtime model "
            "downloads are prohibited"
        )


def _quantization(section: Mapping[str, Any], name: str) -> str:
    value = str(section.get("quantization", "4bit")).lower()
    if value not in VALID_QUANTIZATIONS:
        raise ValueError(
            f"{name}.quantization must be one of "
            f"{', '.join(sorted(VALID_QUANTIZATIONS))}"
        )
    return value


@dataclass(frozen=True)
class VisionLanguageConfig:
    provider: str
    model: str
    model_path: Path
    device: str = "cuda"
    quantization: str = "4bit"
    max_tokens: int = 2048
    temperature: float = 0.1
    local_files_only: bool = True
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 768 * 28 * 28
    candidate_confidence: float = 0.72

    @property
    def model_id(self) -> str:
        return _model_id(self.model)

    def with_model_path(self, value: str | Path) -> VisionLanguageConfig:
        return replace(self, model_path=Path(value).resolve())


@dataclass(frozen=True)
class VisualEmbeddingConfig:
    provider: str
    model: str
    model_path: Path
    device: str = "cuda"
    quantization: str = "4bit"
    dimensions: int = 1024
    batch_size: int = 8
    local_files_only: bool = True
    query_instruction: str = (
        "Retrieve family CCTV frames that visually match the user's query."
    )
    legacy_model: str = "google/siglip2-base-patch16-224"
    legacy_cache: Path = Path(".data/models/huggingface")
    fallback_until_indexed: bool = True
    activation_coverage: float = 0.95

    @property
    def model_id(self) -> str:
        return _model_id(self.model)


@dataclass(frozen=True)
class ModelsConfig:
    vision_language: VisionLanguageConfig
    high_value: VisionLanguageConfig
    visual_embedding: VisualEmbeddingConfig
    path: Path


def _vision_section(
    section: Mapping[str, Any],
    *,
    project_root: Path,
    name: str,
    defaults: VisionLanguageConfig | None = None,
) -> VisionLanguageConfig:
    _require_local_only(section, name)
    provider = str(
        section.get(
            "provider",
            defaults.provider if defaults is not None else "qwen3-vl",
        )
    ).lower()
    if provider not in VALID_VL_PROVIDERS:
        raise ValueError(f"Unsupported {name}.provider: {provider}")
    model = str(
        section.get(
            "model",
            defaults.model if defaults is not None
            else "Qwen3-VL-8B-Instruct",
        )
    )
    default_path = (
        defaults.model_path
        if defaults is not None
        else Path(".data/models/qwen3-vl-8b-instruct")
    )
    max_tokens = int(
        section.get(
            "max_tokens",
            defaults.max_tokens if defaults is not None else 2048,
        )
    )
    if max_tokens <= 0:
        raise ValueError(f"{name}.max_tokens must be positive")
    temperature = float(
        section.get(
            "temperature",
            defaults.temperature if defaults is not None else 0.1,
        )
    )
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"{name}.temperature must be between 0 and 2")
    candidate_confidence = float(
        section.get(
            "candidate_confidence",
            (
                defaults.candidate_confidence
                if defaults is not None
                else 0.72
            ),
        )
    )
    if not 0.0 <= candidate_confidence <= 1.0:
        raise ValueError(
            f"{name}.candidate_confidence must be between 0 and 1"
        )
    return VisionLanguageConfig(
        provider=provider,
        model=model,
        model_path=_local_path(
            section.get("model_path", default_path),
            project_root,
        ),
        device=str(
            section.get(
                "device",
                defaults.device if defaults is not None else "cuda",
            )
        ).lower(),
        quantization=_quantization(
            section
            if "quantization" in section or defaults is None
            else {"quantization": defaults.quantization},
            name,
        ),
        max_tokens=max_tokens,
        temperature=temperature,
        local_files_only=True,
        min_pixels=int(
            section.get(
                "min_pixels",
                (
                    defaults.min_pixels
                    if defaults is not None
                    else 256 * 28 * 28
                ),
            )
        ),
        max_pixels=int(
            section.get(
                "max_pixels",
                (
                    defaults.max_pixels
                    if defaults is not None
                    else 768 * 28 * 28
                ),
            )
        ),
        candidate_confidence=candidate_confidence,
    )


def load_models_config(
    path: str | Path = "config/models.toml",
) -> ModelsConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    project_root = config_path.parent.parent
    vision_raw = raw.get("vision_language")
    if not isinstance(vision_raw, Mapping):
        raise ValueError("models.toml requires [vision_language]")
    primary = _vision_section(
        vision_raw,
        project_root=project_root,
        name="vision_language",
    )
    high_value_raw = vision_raw.get("high_value", {})
    if not isinstance(high_value_raw, Mapping):
        raise ValueError("[vision_language.high_value] must be a table")
    high_value = _vision_section(
        high_value_raw,
        project_root=project_root,
        name="vision_language.high_value",
        defaults=replace(
            primary,
            model="Qwen3-VL-32B-Instruct",
            model_path=project_root.joinpath(
                ".data/models/qwen3-vl-32b-instruct"
            ),
        ),
    )

    embedding_raw = raw.get("visual_embedding")
    if not isinstance(embedding_raw, Mapping):
        raise ValueError("models.toml requires [visual_embedding]")
    _require_local_only(embedding_raw, "visual_embedding")
    provider = str(
        embedding_raw.get("provider", "qwen3-vl-embedding")
    ).lower()
    if provider not in VALID_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported visual_embedding.provider: {provider}"
        )
    dimensions = int(embedding_raw.get("dimensions", 1024))
    if not 64 <= dimensions <= 4096:
        raise ValueError(
            "visual_embedding.dimensions must be between 64 and 4096"
        )
    batch_size = int(embedding_raw.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("visual_embedding.batch_size must be positive")
    activation_coverage = float(
        embedding_raw.get("activation_coverage", 0.95)
    )
    if not 0.0 < activation_coverage <= 1.0:
        raise ValueError(
            "visual_embedding.activation_coverage must be in (0, 1]"
        )
    embedding = VisualEmbeddingConfig(
        provider=provider,
        model=str(
            embedding_raw.get("model", "Qwen3-VL-Embedding-8B")
        ),
        model_path=_local_path(
            embedding_raw.get(
                "model_path",
                ".data/models/qwen3-vl-embedding-8b",
            ),
            project_root,
        ),
        device=str(embedding_raw.get("device", "cuda")).lower(),
        quantization=_quantization(
            embedding_raw,
            "visual_embedding",
        ),
        dimensions=dimensions,
        batch_size=batch_size,
        local_files_only=True,
        query_instruction=str(
            embedding_raw.get(
                "query_instruction",
                (
                    "Retrieve family CCTV frames that visually match the "
                    "user's query."
                ),
            )
        ),
        legacy_model=str(
            embedding_raw.get(
                "legacy_model",
                "google/siglip2-base-patch16-224",
            )
        ),
        legacy_cache=_local_path(
            embedding_raw.get(
                "legacy_cache",
                ".data/models/huggingface",
            ),
            project_root,
        ),
        fallback_until_indexed=bool(
            embedding_raw.get("fallback_until_indexed", True)
        ),
        activation_coverage=activation_coverage,
    )
    return ModelsConfig(
        vision_language=primary,
        high_value=high_value,
        visual_embedding=embedding,
        path=config_path,
    )
