from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_DETECTOR_MODEL = ".data/models/yolo26s.pt"
DEFAULT_EMBEDDING_MODEL = "google/siglip2-base-patch16-224"
PROMPT_VERSION = "baby-diary-prompts-v2"
TEMPORAL_FEEDING_PROMPT_VERSION = "feeding-temporal-native-v2"
TEMPORAL_PLAY_PROMPT_VERSION = "play-temporal-native-v2"
SLEEP_PRESENCE_PROMPT_VERSION = "sleep-state-native-v3"
TEMPORAL_POSTURE_PROMPT_VERSION = "posture-temporal-native-v3"


PROMPT_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "feeding": {
        "feeding_bottle": (
            "a baby drinking milk from a baby bottle",
            "an adult bottle feeding an infant",
        ),
        "breastfeeding": (
            "a mother breastfeeding a baby",
            "an infant nursing at the breast",
        ),
        "solid_food": (
            "a baby eating solid food",
            "an adult spoon feeding a baby",
        ),
        "not_feeding": (
            "a baby who is not eating or drinking",
            "a room scene with no feeding",
        ),
    },
    "play": {
        "playing": (
            "a baby actively playing with toys",
            "a child reaching for and handling a toy",
        ),
        "reading": (
            "an adult reading a picture book with a baby",
            "a baby looking at a children's book",
        ),
        "not_playing": (
            "a baby resting without playing",
            "a room scene with no play activity",
        ),
    },
    "posture": {
        "crawling": (
            "a baby crawling on hands and knees",
            "an infant moving across the floor on all fours",
        ),
        "sitting": (
            "a baby sitting upright",
            "an infant seated on the floor or bed",
        ),
        "standing": (
            "a baby standing upright",
            "an infant standing while holding furniture",
        ),
        "lying_awake": (
            "an awake baby lying down",
            "an infant lying in bed with eyes open",
        ),
        "sleeping": (
            "a sleeping baby lying in bed",
            "an infant asleep under an infrared night camera",
        ),
        "posture_unknown": (
            "no baby is clearly visible",
            "the baby's body posture cannot be seen",
        ),
    },
    "care": {
        "held_by_adult": (
            "an adult holding a baby in their arms",
            "a caregiver carrying an infant",
        ),
        "diaper_change": (
            "an adult changing a baby's diaper",
            "a baby lying down while a caregiver changes clothes or diaper",
        ),
        "care_other": (
            "an adult near a baby without holding or changing the baby",
            "no caregiving action is visible",
        ),
    },
    "toy": {
        "ball": (
            "a baby playing with a ball",
            "a toy ball near a child",
        ),
        "plush_toy": (
            "a baby playing with a stuffed animal",
            "a plush teddy bear toy near a child",
        ),
        "blocks": (
            "a baby playing with building blocks",
            "colorful toy blocks near a child",
        ),
        "book": (
            "a children's picture book near a baby",
            "a baby playing with or reading a book",
        ),
        "toy_vehicle": (
            "a baby playing with a toy car",
            "a toy vehicle near a child",
        ),
        "rattle": (
            "a baby holding a rattle or small handheld toy",
            "an infant playing with a rattle",
        ),
        "activity_gym": (
            "a baby playing under an activity gym",
            "an infant play mat with hanging toys",
        ),
        "doll": (
            "a baby playing with a doll",
            "a toy doll near a child",
        ),
        "other_toy": (
            "a baby interacting with another kind of toy",
            "a toy near a baby",
        ),
        "no_toy": (
            "no toy is visible near the baby",
            "an empty room without toys",
        ),
    },
}


@dataclass(frozen=True)
class PromptReference:
    group: str
    label: str
    prompt: str


def flattened_prompts() -> tuple[PromptReference, ...]:
    return tuple(
        PromptReference(group=group, label=label, prompt=prompt)
        for group, labels in PROMPT_GROUPS.items()
        for label, prompts in labels.items()
        for prompt in prompts
    )


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_DETECTOR_MODEL,
        *,
        device: str = "0",
        image_size: int = 640,
        confidence: float = 0.15,
    ) -> None:
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Visual detection dependencies are missing; install the "
                "project visual extras"
            ) from exc
        self._model = YOLO(str(model_path))
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.model_name = (
            f"{Path(model_path).stem}-coco@{ultralytics.__version__}"
        )

    def predict(
        self,
        image_paths: Sequence[str | Path],
        *,
        batch_size: int = 128,
    ) -> list[list[dict[str, Any]]]:
        if not image_paths:
            return []
        results = self._model.predict(
            source=[str(path) for path in image_paths],
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            half=self.device != "cpu",
            batch=batch_size,
            verbose=False,
            stream=False,
        )
        payload: list[list[dict[str, Any]]] = []
        for result in results:
            height, width = result.orig_shape
            frame: list[dict[str, Any]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                for box, score, class_value in zip(
                    xyxy,
                    confidences,
                    classes,
                    strict=True,
                ):
                    class_id = int(class_value)
                    frame.append(
                        {
                            "class_id": class_id,
                            "class_name": str(result.names[class_id]),
                            "confidence": float(score),
                            "x1": max(0.0, min(1.0, float(box[0]) / width)),
                            "y1": max(0.0, min(1.0, float(box[1]) / height)),
                            "x2": max(0.0, min(1.0, float(box[2]) / width)),
                            "y2": max(0.0, min(1.0, float(box[3]) / height)),
                            "attributes": {},
                        }
                    )
            payload.append(frame)
        return payload


class SiglipVisualEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cuda",
        local_cache: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Visual embedding dependencies are missing; install the "
                "project visual extras"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        cache_dir = str(local_cache) if local_cache is not None else None
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=self.dtype,
            local_files_only=True,
        )
        self._model.to(self.device)
        self._model.eval()
        self.model_name = model_name
        self.prompt_version = PROMPT_VERSION
        self._references = flattened_prompts()
        texts = [reference.prompt for reference in self._references]
        tokenized = self._processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self._text_inputs = {
            key: value.to(self.device)
            for key, value in tokenized.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        # Prompt text never changes during a worker run. Computing the text
        # tower once keeps month-scale backfills focused on the image tower
        # instead of re-encoding the same prompt set for every image batch.
        with self._torch.inference_mode():
            text_features = self._model.get_text_features(
                **self._text_inputs
            ).float()
            self._text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            self._logit_scale = (
                self._model.logit_scale.detach().float().exp()
            )
            self._logit_bias = self._model.logit_bias.detach().float()

    def encode(
        self,
        image_paths: Sequence[str | Path],
        *,
        focus_boxes: Sequence[Sequence[Mapping[str, Any]]] | None = None,
        top_per_group: int = 3,
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        if not image_paths:
            return [], []
        if focus_boxes is not None and len(focus_boxes) != len(image_paths):
            raise ValueError("focus_boxes must align with image_paths")
        from PIL import Image

        images = []
        variant_indexes: list[list[int]] = []
        try:
            for image_index, path in enumerate(image_paths):
                image = Image.open(path)
                rgb = image.convert("RGB")
                image.close()
                indexes = [len(images)]
                images.append(rgb)
                boxes = (
                    list(focus_boxes[image_index])
                    if focus_boxes is not None
                    else []
                )
                selected = [
                    box
                    for box in boxes
                    if str(box.get("class_name", "")) == "person"
                    and float(box.get("confidence", 0)) >= 0.20
                ]
                selected.sort(
                    key=lambda box: float(box.get("confidence", 0)),
                    reverse=True,
                )
                for box in selected[:3]:
                    width, height = rgb.size
                    x1 = float(box["x1"])
                    y1 = float(box["y1"])
                    x2 = float(box["x2"])
                    y2 = float(box["y2"])
                    pad_x = max(0.05, (x2 - x1) * 0.35)
                    pad_y = max(0.05, (y2 - y1) * 0.35)
                    crop = rgb.crop(
                        (
                            round(max(0.0, x1 - pad_x) * width),
                            round(max(0.0, y1 - pad_y) * height),
                            round(min(1.0, x2 + pad_x) * width),
                            round(min(1.0, y2 + pad_y) * height),
                        )
                    )
                    if crop.width >= 48 and crop.height >= 48:
                        indexes.append(len(images))
                        images.append(crop)
                if len(selected) >= 2:
                    width, height = rgb.size
                    x1 = min(float(box["x1"]) for box in selected)
                    y1 = min(float(box["y1"]) for box in selected)
                    x2 = max(float(box["x2"]) for box in selected)
                    y2 = max(float(box["y2"]) for box in selected)
                    pad_x = max(0.05, (x2 - x1) * 0.20)
                    pad_y = max(0.05, (y2 - y1) * 0.20)
                    crop = rgb.crop(
                        (
                            round(max(0.0, x1 - pad_x) * width),
                            round(max(0.0, y1 - pad_y) * height),
                            round(min(1.0, x2 + pad_x) * width),
                            round(min(1.0, y2 + pad_y) * height),
                        )
                    )
                    if crop.width >= 48 and crop.height >= 48:
                        indexes.append(len(images))
                        images.append(crop)
                variant_indexes.append(indexes)
            image_inputs = self._processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = image_inputs["pixel_values"].to(
                device=self.device,
                dtype=self.dtype,
            )
            with self._torch.inference_mode():
                image_features = self._model.get_image_features(
                    pixel_values=pixel_values,
                ).float()
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                logits = (
                    image_features @ self._text_features.t()
                ) * self._logit_scale + self._logit_bias
        finally:
            for image in images:
                image.close()

        embeddings: list[dict[str, Any]] = []
        all_vectors = image_features.detach().cpu().numpy()
        vectors = all_vectors[
            [indexes[0] for indexes in variant_indexes]
        ]
        for vector in vectors:
            compact = vector.astype("<f2", copy=False)
            embeddings.append(
                {
                    "dimensions": int(compact.shape[0]),
                    "dtype": "float16-le",
                    "vector": compact.tobytes(),
                    "vector_norm": float(
                        math.sqrt(float((vector * vector).sum()))
                    ),
                }
            )

        observations: list[list[dict[str, Any]]] = []
        logits_cpu = logits.detach().cpu()
        for image_index, variants in enumerate(variant_indexes):
            frame: list[dict[str, Any]] = []
            for group, labels in PROMPT_GROUPS.items():
                label_logits: list[tuple[str, float, float]] = []
                for label in labels:
                    prompt_indexes = [
                        index
                        for index, reference in enumerate(self._references)
                        if reference.group == group
                        and reference.label == label
                    ]
                    prompt_logits = logits_cpu[
                        variants,
                    ][:, prompt_indexes]
                    best_logit = float(prompt_logits.max().item())
                    best_sigmoid = float(
                        prompt_logits.sigmoid().max().item()
                    )
                    label_logits.append((label, best_logit, best_sigmoid))
                relative = self._torch.softmax(
                    self._torch.tensor(
                        [item[1] for item in label_logits],
                        dtype=self._torch.float32,
                    ),
                    dim=0,
                ).tolist()
                ranked = sorted(
                    zip(label_logits, relative, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                )
                for rank, ((label, raw_logit, sigmoid), score) in enumerate(
                    ranked[:top_per_group],
                    start=1,
                ):
                    frame.append(
                        {
                            "group": group,
                            "label": label,
                            "score": float(score),
                            "details": {
                                "rank": rank,
                                "raw_logit": raw_logit,
                                "sigmoid": sigmoid,
                                "focus_variants": len(variants),
                            },
                        }
                    )
            observations.append(frame)
        return embeddings, observations

    def encode_text(self, text: str):
        if not text.strip():
            raise ValueError("Search text cannot be empty")
        tokenized = self._processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in tokenized.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with self._torch.inference_mode():
            vector = self._model.get_text_features(**inputs).float()
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
        return vector[0].detach().cpu().numpy()
