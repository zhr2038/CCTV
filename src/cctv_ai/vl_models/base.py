from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any, Mapping, Sequence


ALLOWED_ACTIVITIES = {
    "feeding",
    "play",
    "sitting",
    "crawling",
    "standing",
    "lying_awake",
    "sleeping",
    "rolling",
    "speaking",
    "other",
    "unclear",
}

ACTIVITY_ALIASES = {
    "drinking": "feeding",
    "bottle_feeding": "feeding",
    "breastfeeding": "feeding",
    "solid_feeding": "feeding",
    "playing": "play",
    "sit": "sitting",
    "crawl": "crawling",
    "stand": "standing",
    "lying": "lying_awake",
    "sleep": "sleeping",
    "roll": "rolling",
    "talking": "speaking",
    "unknown": "unclear",
}

# These are judgments about the child rather than observations in the frame.
# Observable actions such as crying, smiling or vocalising remain permitted.
PROHIBITED_JUDGMENT_PATTERNS = (
    r"聪明|智力|愚笨|笨拙|天赋",
    r"性格|内向|外向|乖巧|顽皮",
    r"情绪(?:稳定|不稳|异常)|焦虑|抑郁|害怕|伤心|开心",
    r"发育(?:迟缓|异常|落后|超前|正常)",
    r"(?:运动|语言|认知)能力(?:优秀|较差|异常|超前|落后)",
    r"\b(?:intelligen(?:t|ce)|personality|developmental "
    r"(?:delay|disorder)|emotionally? (?:stable|unstable))\b",
)

FEEDING_CONTACT_PATTERNS = (
    r"接触(?:宝宝|婴儿)?(?:的)?嘴",
    r"(?:宝宝|婴儿)?嘴(?:部)?(?:接触|含住|含着|咬住|衔住)",
    r"奶嘴(?:进入|放入|含在|接触)",
    r"(?:勺子|食物)(?:进入|放入|接触)(?:宝宝|婴儿)?(?:的)?嘴",
    r"(?:宝宝|婴儿)?嘴(?:部)?(?:持续)?(?:贴|压|含)(?:在|住)?乳房",
    r"乳房(?:位置)?(?:与)?(?:宝宝|婴儿)?嘴(?:部)?(?:贴合|接触)",
    r"(?:宝宝|婴儿)?(?:正在)?吸吮",
    r"\b(?:touch(?:es|ing)?|in contact with) (?:the )?"
    r"(?:baby'?s? )?mouth\b",
    r"\b(?:nipple|spoon|food) (?:is )?in (?:the )?"
    r"(?:baby'?s? )?mouth\b",
    r"\b(?:baby|infant) (?:is )?(?:sucking|nursing)\b",
)


def _safe_strings(values: Any, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text[:240])
        if len(result) >= limit:
            break
    return tuple(result)


def _contains_prohibited_judgment(value: str) -> bool:
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in PROHIBITED_JUDGMENT_PATTERNS
    )


def _sanitize_observation(value: Any) -> tuple[str, bool]:
    text = str(value or "").strip()[:800]
    if not text:
        return "画面信息不足，无法确定具体活动。", True
    if _contains_prohibited_judgment(text):
        return "画面仅保留可直接观察的动作，其他判断已被过滤。", True
    return text, False


@dataclass(frozen=True)
class VisionAnalysisContext:
    camera: str
    time: str
    objects: tuple[str, ...] = ()
    motion: str = "unknown"
    detections: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> VisionAnalysisContext:
        source = value or {}
        raw_time = source.get("time", "unknown")
        if isinstance(raw_time, datetime):
            raw_time = raw_time.isoformat(sep=" ", timespec="seconds")
        detections = source.get("detections", ())
        if not isinstance(detections, (list, tuple)):
            detections = ()
        return cls(
            camera=str(source.get("camera", "unknown")),
            time=str(raw_time),
            objects=_safe_strings(source.get("objects", ()), limit=20),
            motion=str(source.get("motion", "unknown")),
            detections=tuple(
                item for item in detections if isinstance(item, Mapping)
            )[:20],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "time": self.time,
            "objects": list(self.objects),
            "motion": self.motion,
            "detections": [dict(item) for item in self.detections],
            "detection_warning": (
                "objects and detections are untrusted detector candidates; "
                "confidence below 0.40 is weak and is never event proof"
            ),
        }


@dataclass(frozen=True)
class StructuredVisualEvent:
    activity: str
    description: str
    objects: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]
    uncertain: bool

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> StructuredVisualEvent:
        activity = str(value.get("activity", "unclear")).strip().lower()
        activity = ACTIVITY_ALIASES.get(activity, activity)
        if activity not in ALLOWED_ACTIVITIES:
            activity = "unclear"
        description, filtered = _sanitize_observation(
            value.get("description")
        )
        evidence: list[str] = []
        evidence_filtered = False
        for item in _safe_strings(value.get("evidence", ()), limit=8):
            sanitized, was_filtered = _sanitize_observation(item)
            evidence_filtered = evidence_filtered or was_filtered
            if not was_filtered and sanitized not in evidence:
                evidence.append(sanitized)
        objects = tuple(
            item.lower()
            for item in _safe_strings(value.get("objects", ()), limit=20)
        )
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        uncertain = bool(value.get("uncertain", False))
        if activity == "feeding":
            contact_text = " ".join(
                (description, *evidence)
            )
            if not any(
                re.search(
                    pattern,
                    contact_text,
                    flags=re.IGNORECASE,
                )
                for pattern in FEEDING_CONTACT_PATTERNS
            ):
                # A detected bottle or an object merely near/aimed at the
                # mouth is a common false positive in these fixed cameras.
                uncertain = True
                confidence = min(confidence, 0.45)
        if filtered or evidence_filtered:
            uncertain = True
            confidence = min(confidence, 0.45)
        if activity == "unclear":
            uncertain = True
        return cls(
            activity=activity,
            description=description,
            objects=objects,
            confidence=confidence,
            evidence=tuple(evidence),
            uncertain=uncertain,
        )

    @classmethod
    def uncertain_result(
        cls,
        description: str = "画面信息不足，无法确定具体活动。",
    ) -> StructuredVisualEvent:
        return cls(
            activity="unclear",
            description=description,
            objects=(),
            confidence=0.0,
            evidence=(),
            uncertain=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity,
            "description": self.description,
            "objects": list(self.objects),
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "uncertain": self.uncertain,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )


class VisionLanguageModel(ABC):
    model_name: str

    @abstractmethod
    def analyze_image(
        self,
        image: Any,
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None = None,
    ) -> StructuredVisualEvent:
        raise NotImplementedError

    @abstractmethod
    def analyze_video_frames(
        self,
        frames: Sequence[Any],
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None = None,
    ) -> StructuredVisualEvent:
        raise NotImplementedError

    @abstractmethod
    def generate_summary(
        self,
        events: Sequence[StructuredVisualEvent | Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def analyze_images(
        self,
        images: Sequence[Any],
        prompts: Sequence[str],
        contexts: Sequence[
            VisionAnalysisContext | Mapping[str, Any] | None
        ],
    ) -> list[StructuredVisualEvent]:
        if not (len(images) == len(prompts) == len(contexts)):
            raise ValueError("images, prompts and contexts must align")
        return [
            self.analyze_image(image, prompt, context)
            for image, prompt, context in zip(
                images,
                prompts,
                contexts,
                strict=True,
            )
        ]

    def analyze_video_batches(
        self,
        frame_groups: Sequence[Sequence[Any]],
        prompts: Sequence[str],
        contexts: Sequence[
            VisionAnalysisContext | Mapping[str, Any] | None
        ],
    ) -> list[StructuredVisualEvent]:
        """Analyze several independent frame sequences.

        Backends can override this to batch the vision encoder and decoder.
        The default keeps third-party providers compatible without requiring
        business code to know whether a provider supports native batching.
        """
        if not (
            len(frame_groups) == len(prompts) == len(contexts)
        ):
            raise ValueError(
                "frame_groups, prompts and contexts must align"
            )
        return [
            self.analyze_video_frames(frames, prompt, context)
            for frames, prompt, context in zip(
                frame_groups,
                prompts,
                contexts,
                strict=True,
            )
        ]

    def close(self) -> None:
        return None
