from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .model_config import VisionLanguageConfig, load_models_config
from .visual_models import (
    SLEEP_PRESENCE_PROMPT_VERSION,
    TEMPORAL_FEEDING_PROMPT_VERSION,
    TEMPORAL_PLAY_PROMPT_VERSION,
    TEMPORAL_POSTURE_PROMPT_VERSION,
)
from .vl_models import (
    StructuredVisualEvent,
    VisionAnalysisContext,
    create_vision_language_model,
)

DEFAULT_VLM_MODEL_DIR = Path(".data/models/qwen2.5-vl-7b-instruct")
DEFAULT_VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN3_VLM_MODEL_DIR = Path(
    ".data/models/qwen3-vl-8b-instruct"
)
DEFAULT_QWEN3_VLM_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
FEEDING_PROMPT_VERSION = "feeding-contact-v2"
TOY_PROMPT_VERSION = "toy-detail-v1"
STRUCTURED_FEEDING_PROMPT_VERSION = "feeding-structured-v3"
STRUCTURED_TOY_PROMPT_VERSION = "toy-structured-v2"


def _retry_sqlite_write(
    operation: Callable[[], Any],
    *,
    attempts: int = 8,
) -> Any:
    """Retry short verifier writes while another local worker owns SQLite."""
    delay_seconds = 0.10
    for attempt in range(attempts):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if (
                "locked" not in message
                and "busy" not in message
            ) or attempt + 1 >= attempts:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 2.0)
    raise AssertionError("unreachable")

FEEDING_LABELS = {
    "FEED_BOTTLE",
    "FEED_BREAST",
    "FEED_SOLID",
    "NOT_FEEDING",
    "UNCLEAR",
}
TOY_LABELS = {
    "ACTIVITY_CUBE",
    "FRUIT_SORTER",
    "STACKING_TOWER",
    "BOOK",
    "PLUSH",
    "BALL",
    "RATTLE_TEETHER",
    "DOLL",
    "TOY_VEHICLE",
    "BLOCKS",
    "ACTIVITY_GYM",
    "OTHER",
    "NONE",
    "UNCLEAR",
}

FEEDING_PROMPT = """
两张图拼在一起：左边是家庭监控全景，右边是同一时刻宝宝附近的
局部。严格判断宝宝此刻是否正在喝奶或进食。只有直接看见奶嘴、乳头、
勺子或食物与宝宝嘴部接触，才可回答 FEED_*。

以下都不是进食：成人只是抱着、安抚、换尿布、擦脸；宝宝吃手、咬衣物、
牙胶或玩具；奶瓶只是出现在画面里；物体只是靠近嘴但看不出接触。
YOLO 的 bottle 只是候选，彩色扁平绘本、包装袋和玩具常被误检成奶瓶；
必须独立看清奶瓶形状与奶嘴接触，不能照抄检测结果。
宝宝只是靠在穿着衣服的成人胸前也不是母乳喂养。嘴部被手、头部、被子
或画质遮挡时回答 UNCLEAR，不要猜测。

只回答一个标签：
FEED_BOTTLE 表示奶瓶奶嘴正接触宝宝嘴部；
FEED_BREAST 表示宝宝嘴部正含乳头；
FEED_SOLID 表示勺子或食物正接触宝宝嘴部；
NOT_FEEDING 表示明确没有以上接触；
UNCLEAR 表示嘴部被遮挡或画质不足。

不要解释，不要重复标签列表。
""".strip()

TOY_PROMPT = """
两张图拼在一起：左边是家庭监控全景，右边是同一时刻宝宝附近的
局部。识别宝宝正在直接触摸、拿着或持续注视的主要玩具。

只回答一个标签：
ACTIVITY_CUBE=带孔洞或机关的立方体活动盒；
FRUIT_SORTER=装彩色水果形状件的红白托盘；
STACKING_TOWER=竖直彩色叠叠塔或滚珠塔；
BOOK=绘本；PLUSH=毛绒玩具；BALL=球；
RATTLE_TEETHER=摇铃、牙胶或小手持玩具；
DOLL=玩偶；TOY_VEHICLE=玩具车；BLOCKS=积木；
ACTIVITY_GYM=带拱架悬挂玩具的游戏垫；
OTHER=确实在互动但类别不同；NONE=没有直接互动。

不要解释，不要重复标签列表。
""".strip()

STRUCTURED_FEEDING_PROMPT = """
判断这一时刻是否直接可见宝宝正在喝奶或进食。只有奶嘴、乳头、勺子或
食物与宝宝嘴部发生可见接触，activity 才能写 feeding。奶瓶只是在画面
里、成人只是抱着或宝宝咬手/玩具，都不是 feeding。嘴部遮挡或画质不足
时写 unclear。

若确认进食，请在 objects 中使用可检索的英文物体词：奶瓶用 bottle，
母乳用 breast，勺子用 spoon，固体食物用 food。evidence 必须写画面里
直接可见的接触证据。
""".strip()

STRUCTURED_TOY_PROMPT = """
判断宝宝是否正在直接触摸、拿着或持续注视某个玩具。activity 只能在
确有可见互动时写 play；没有互动写 other；遮挡或画质不足写 unclear。

objects 使用可检索的英文物体词，优先从 activity_cube、fruit_sorter、
stacking_tower、book、plush、ball、rattle_teether、doll、toy_vehicle、
blocks、activity_gym 中选择。evidence 只写直接可见的互动动作。
""".strip()

TEMPORAL_FEEDING_PROMPT = """
按时间顺序逐帧核对这组同一摄像头的原始分辨率画面，只判断一次连续喂养
行为。不要从成人抱姿或手在嘴边推测。

判定 feeding 必须满足以下一种可复核模式，并且至少连续两帧成立：
1. 奶瓶：能辨认出完整奶瓶或瓶身与奶嘴，奶嘴与宝宝嘴部接触；透明杯、
手机、手、安抚奶嘴、牙胶和玩具不是奶瓶。
2. 母乳：成人胸部明确裸露，宝宝嘴部持续贴在乳房位置；宝宝头部可以遮住
乳头，但“裸露乳房 + 嘴部贴合”的组合必须在至少两帧持续。成人穿着上衣、
宝宝仅靠在胸前不算。
3. 固体食物：勺子或明确食物与宝宝嘴部接触。

如果画面明确不满足以上条件，activity 写 other，uncertain=false。只有
确实存在疑似喂养器具但关键接触被遮挡时才写 unclear；不要把普通抱抱都
写 unclear。确认 feeding 时 objects 必须含 bottle、breast、spoon 或 food
之一，evidence 必须写出在哪些帧看见什么物体以及持续接触。
""".strip()

TEMPORAL_PLAY_PROMPT = """
这些原始分辨率画面来自同一次候选活动。逐帧判断宝宝是否在持续、主动地
和玩具互动。只有至少两帧直接看见宝宝触摸、拿取、操作玩具，或持续把
视线和动作指向同一玩具，activity 才写 play。

成人独自拿着玩具、玩具只是散落在附近、宝宝睡觉或躺着、换尿布、安抚、
吃手、咬安抚奶嘴，都不是玩耍，activity 写 other。关键动作被遮挡且无法
判断时才写 unclear。

确认 play 时，objects 优先使用 activity_cube、fruit_sorter、
stacking_tower、book、plush、ball、rattle_teether、doll、toy_vehicle、
blocks、activity_gym；无法归类但确实互动时使用 toy。evidence 必须注明
至少两个帧号和宝宝直接互动的动作。description 只写一个短句，evidence
最多两项；一项可以合并注明多个帧。
""".strip()

SLEEP_PRESENCE_PROMPT = """
这些原始分辨率画面从同一段候选睡眠区间抽取。这不是“宝宝是否在床”
检测，而是严格判断宝宝是否确实处于睡眠状态。

activity 只有满足以下要求才写 sleeping：
- 至少两帧直接看见宝宝的头部或身体轮廓；
- 直接看见闭眼和放松面部，或在夜视/侧卧看不清眼睛时，三帧均呈现
  明确放松睡姿且跨约两分钟姿态基本不变；
- 没有睁眼注视、抬头、伸手抓取、翻玩物品、与成人互动等清醒行为。

宝宝只是躺着、含安抚奶嘴、被成人抱着或短时间没有大动作，都不足以证明
睡眠。睁眼、视线变化、主动手部动作、抬头、坐起或爬行时写 lying_awake
或 other。不能把夜间、低运动、被子隆起或“在床”本身当作睡眠证据。
脸部被遮挡且只有“躺着不动”而没有闭眼或明确放松睡姿时写 unclear。
空床、只有枕头被子或仅有成人写 other。

evidence 必须明确写出闭眼，或夜视下可复核的放松睡姿及跨三帧不变；
若写不出这类证据，不能输出 sleeping。description 只写一个短句，
evidence 最多两项；一项可以合并注明多个帧。
""".strip()

TEMPORAL_POSTURE_PROMPT = """
这些原始分辨率画面来自同一次宝宝姿态候选。只描述宝宝本人至少连续两帧
都能直接看到的主要姿态：

- sitting：宝宝臀部着床或着地，躯干自主保持明显直立；
- crawling：宝宝俯卧，以手膝或手臂支撑，并在连续画面中出现爬行动作；
- standing：宝宝双脚承重、躯干直立，可以扶着固定物；
- lying_awake：宝宝躺着但明显清醒。

成人抱着、扶起或遮住宝宝时，不能把成人姿势算成宝宝姿势。只有单帧、
身体关键部位看不清或姿态在画面间不一致时写 unclear；明确不属于上述
姿态时写 other。evidence 必须注明至少两个帧号及可见的身体支撑关系。
description 只写一个短句，evidence 最多两项；一项可以合并注明多个帧。
""".strip()


@dataclass(frozen=True)
class Verification:
    label: str
    confidence: float
    raw_output: str
    structured_event: StructuredVisualEvent | None = None
    metadata: Mapping[str, Any] | None = None


def model_name_for_dir(model_dir: str | Path) -> str:
    """Return a stable provenance name for a local Qwen checkpoint."""
    name = Path(model_dir).name.lower()
    if "qwen3-vl-32b" in name:
        return "Qwen/Qwen3-VL-32B-Instruct"
    if "qwen3-vl-8b" in name:
        return DEFAULT_QWEN3_VLM_MODEL_NAME
    if "qwen2.5-vl-7b" in name:
        return DEFAULT_VLM_MODEL_NAME
    if "qwen2.5-vl-3b" in name:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    return f"local/{Path(model_dir).name}"


def _config_for_model_dir(
    config: VisionLanguageConfig,
    model_dir: str | Path,
) -> VisionLanguageConfig:
    model_name = model_name_for_dir(model_dir)
    provider = (
        "qwen2.5-vl"
        if "Qwen2.5-VL" in model_name
        else "qwen3-vl"
    )
    model = model_name.split("/", 1)[-1]
    return replace(
        config,
        provider=provider,
        model=model,
        model_path=Path(model_dir).resolve(),
    )


def _extract_label(
    value: str,
    allowed: set[str],
    fallback: str,
) -> str:
    upper = value.upper()
    matches = [
        label
        for label in allowed
        if re.search(rf"(?<![A-Z_]){re.escape(label)}(?![A-Z_])", upper)
    ]
    return matches[0] if len(matches) == 1 else fallback


def _box_area(box: Mapping[str, Any]) -> float:
    return max(0.0, float(box["x2"]) - float(box["x1"])) * max(
        0.0,
        float(box["y2"]) - float(box["y1"]),
    )


def _select_baby_box(
    boxes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    candidates = [
        box
        for box in boxes
        if str(box.get("class_name", "")) == "person"
        and float(box.get("confidence", 0.0)) >= 0.18
        and float(box.get("x2", 0.0)) - float(box.get("x1", 0.0)) >= 0.05
        and float(box.get("y2", 0.0)) - float(box.get("y1", 0.0)) >= 0.12
        and 0.006 <= _box_area(box) <= 0.24
    ]
    if not candidates:
        return None
    # In these fixed cameras the infant is normally the smallest complete
    # person detection. Confidence breaks ties between overlapping boxes.
    return min(
        candidates,
        key=lambda box: (
            _box_area(box),
            -float(box.get("confidence", 0.0)),
        ),
    )


def _composite_image(
    path: Path,
    boxes: Sequence[Mapping[str, Any]],
):
    from PIL import Image, ImageOps

    source = Image.open(path)
    rgb = source.convert("RGB")
    source.close()
    try:
        overview = Image.new("RGB", (640, 360), (238, 236, 230))
        contained = ImageOps.contain(rgb, (640, 360))
        overview.paste(
            contained,
            (
                (overview.width - contained.width) // 2,
                (overview.height - contained.height) // 2,
            ),
        )
        contained.close()
        selected = _select_baby_box(boxes)
        if selected is None:
            return overview
        width, height = rgb.size
        x1 = float(selected["x1"])
        y1 = float(selected["y1"])
        x2 = float(selected["x2"])
        y2 = float(selected["y2"])
        pad_x = max(0.15, (x2 - x1) * 1.8)
        pad_y = max(0.16, (y2 - y1) * 1.3)
        crop = rgb.crop(
            (
                round(max(0.0, x1 - pad_x) * width),
                round(max(0.0, y1 - pad_y) * height),
                round(min(1.0, x2 + pad_x) * width),
                round(min(1.0, y2 + pad_y) * height),
            )
        )
        detail = ImageOps.fit(crop, (360, 360))
        crop.close()
        composite = Image.new("RGB", (1000, 360), (238, 236, 230))
        composite.paste(overview, (0, 0))
        composite.paste(detail, (640, 0))
        overview.close()
        detail.close()
        return composite
    finally:
        rgb.close()


class LocalVisualVerifier:
    def __init__(
        self,
        model_dir: str | Path = DEFAULT_VLM_MODEL_DIR,
        *,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Visual verifier dependencies are missing; install the "
                "project visual extras"
            ) from exc
        model_path = Path(model_dir).resolve()
        if not model_path.joinpath("model.safetensors.index.json").is_file():
            raise FileNotFoundError(
                f"Local visual verifier model is incomplete: {model_path}"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._torch = torch
        self.device = device
        self.model_dir = model_path
        self.model_name = model_name_for_dir(model_path)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            use_fast=True,
        )
        # Qwen-VL is decoder-only. Batched generation must left-pad prompts;
        # right padding can shift the generation boundary between examples.
        self._processor.tokenizer.padding_side = "left"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            attn_implementation="sdpa",
        )
        self._model.eval()
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def classify(
        self,
        *,
        task: str,
        image_paths: Sequence[Path],
        boxes: Sequence[Sequence[Mapping[str, Any]]],
        batch_size: int = 8,
    ) -> list[Verification]:
        if len(image_paths) != len(boxes):
            raise ValueError("image_paths and boxes must align")
        if task == "feeding":
            prompt = FEEDING_PROMPT
            allowed = FEEDING_LABELS
            fallback = "UNCLEAR"
        elif task == "toy":
            prompt = TOY_PROMPT
            allowed = TOY_LABELS
            fallback = "OTHER"
        else:
            raise ValueError(f"Unsupported verification task: {task}")
        results: list[Verification] = []
        for start in range(0, len(image_paths), max(1, batch_size)):
            stop = min(len(image_paths), start + max(1, batch_size))
            images = [
                _composite_image(path, frame_boxes)
                for path, frame_boxes in zip(
                    image_paths[start:stop],
                    boxes[start:stop],
                    strict=True,
                )
            ]
            try:
                messages = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    for image in images
                ]
                texts = [
                    self._processor.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for message in messages
                ]
                inputs = self._processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with self._torch.inference_mode():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=12,
                        do_sample=False,
                        use_cache=True,
                    )
                trimmed = [
                    output[len(source) :]
                    for source, output in zip(
                        inputs.input_ids,
                        generated,
                        strict=True,
                    )
                ]
                decoded = self._processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for output in decoded:
                    label = _extract_label(output, allowed, fallback)
                    confidence = (
                        0.60
                        if label in {"UNCLEAR", "OTHER"}
                        else 0.82
                    )
                    results.append(
                        Verification(
                            label=label,
                            confidence=confidence,
                            raw_output=output.strip(),
                        )
                    )
            finally:
                for image in images:
                    image.close()
        return results

    def close(self) -> None:
        del self._model
        del self._processor
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


def _feeding_label(event: StructuredVisualEvent) -> str:
    if event.uncertain or event.activity == "unclear":
        return "UNCLEAR"
    if event.activity != "feeding":
        return "NOT_FEEDING"
    evidence_text = " ".join(
        (event.description, *event.objects, *event.evidence)
    ).lower()
    direct_contact = any(
        token in evidence_text
        for token in (
            "接触宝宝嘴",
            "接触婴儿嘴",
            "嘴部接触",
            "嘴含住",
            "嘴里",
            "嘴中",
            "嘴部贴在乳房",
            "嘴持续贴在乳房",
            "嘴部与乳房贴合",
            "嘴部接触乳房",
            "吸吮",
            "in the baby mouth",
            "in the baby's mouth",
            "touching the baby mouth",
            "touching the baby's mouth",
            "baby is sucking",
            "infant is sucking",
            "baby is nursing",
        )
    )
    if not direct_contact:
        return "UNCLEAR"
    if any(
        token in evidence_text
        for token in ("bottle", "奶瓶", "奶嘴")
    ):
        return "FEED_BOTTLE"
    if any(
        token in evidence_text
        for token in ("breast", "乳头", "母乳")
    ):
        return "FEED_BREAST"
    if any(
        token in evidence_text
        for token in ("spoon", "food", "勺", "食物", "辅食")
    ):
        return "FEED_SOLID"
    return "UNCLEAR"


TOY_OBJECT_LABELS = {
    "activity_cube": "ACTIVITY_CUBE",
    "fruit_sorter": "FRUIT_SORTER",
    "stacking_tower": "STACKING_TOWER",
    "book": "BOOK",
    "plush": "PLUSH",
    "ball": "BALL",
    "rattle_teether": "RATTLE_TEETHER",
    "doll": "DOLL",
    "toy_vehicle": "TOY_VEHICLE",
    "blocks": "BLOCKS",
    "activity_gym": "ACTIVITY_GYM",
}


def _toy_label(event: StructuredVisualEvent) -> str:
    if event.uncertain or event.activity == "unclear":
        return "UNCLEAR"
    if event.activity != "play":
        return "NONE"
    joined = " ".join(event.objects).lower().replace("-", "_")
    for token, label in TOY_OBJECT_LABELS.items():
        if token in joined:
            return label
    return "OTHER"


def _sleep_presence_label(event: StructuredVisualEvent) -> str:
    if event.uncertain or event.activity == "unclear":
        return "UNCLEAR"
    if event.activity != "sleeping":
        return "NOT_SLEEPING"
    return "BABY_SLEEPING"


def _posture_label(event: StructuredVisualEvent) -> str:
    if event.uncertain or event.activity == "unclear":
        return "UNCLEAR"
    return {
        "sitting": "SITTING",
        "crawling": "CRAWLING",
        "standing": "STANDING",
        "lying_awake": "LYING_AWAKE",
        "other": "OTHER",
    }.get(event.activity, "OTHER")


class ConfiguredVisualVerifier:
    """Business-compatible verifier backed by the abstract VL interface."""

    def __init__(self, config: VisionLanguageConfig) -> None:
        self._model = create_vision_language_model(config)
        self.model_name = self._model.model_name

    def classify(
        self,
        *,
        task: str,
        image_paths: Sequence[Path],
        boxes: Sequence[Sequence[Mapping[str, Any]]],
        contexts: Sequence[VisionAnalysisContext],
        batch_size: int = 8,
    ) -> list[Verification]:
        if not (
            len(image_paths) == len(boxes) == len(contexts)
        ):
            raise ValueError(
                "image_paths, boxes and contexts must align"
            )
        if task == "feeding":
            prompt = STRUCTURED_FEEDING_PROMPT
            label_for_event = _feeding_label
        elif task == "toy":
            prompt = STRUCTURED_TOY_PROMPT
            label_for_event = _toy_label
        else:
            raise ValueError(f"Unsupported verification task: {task}")
        results: list[Verification] = []
        for start in range(0, len(image_paths), max(1, batch_size)):
            stop = min(len(image_paths), start + max(1, batch_size))
            images = [
                _composite_image(path, frame_boxes)
                for path, frame_boxes in zip(
                    image_paths[start:stop],
                    boxes[start:stop],
                    strict=True,
                )
            ]
            try:
                events = self._model.analyze_images(
                    images,
                    [prompt] * len(images),
                    list(contexts[start:stop]),
                )
                for event in events:
                    results.append(
                        Verification(
                            label=label_for_event(event),
                            confidence=event.confidence,
                            raw_output=event.to_json(),
                            structured_event=event,
                        )
                    )
            finally:
                for image in images:
                    image.close()
        return results

    def classify_temporal_feeding(
        self,
        *,
        frame_paths: Sequence[Any],
        context: VisionAnalysisContext,
        keyframe_ids: Sequence[int],
    ) -> Verification:
        if len(frame_paths) != len(keyframe_ids):
            raise ValueError("frame_paths and keyframe_ids must align")
        if len(frame_paths) < 3:
            event = StructuredVisualEvent(
                activity="unclear",
                description="连续可用画面不足，无法确认进食动作。",
                objects=(),
                confidence=0.45,
                evidence=("少于三张连续画面",),
                uncertain=True,
            )
        else:
            event = self._model.analyze_video_frames(
                frame_paths,
                TEMPORAL_FEEDING_PROMPT,
                context,
            )
        label = _feeding_label(event)
        if (
            label in {"FEED_BOTTLE", "FEED_BREAST", "FEED_SOLID"}
            and event.confidence < 0.80
        ):
            label = "UNCLEAR"
        return Verification(
            label=label,
            confidence=event.confidence,
            raw_output=event.to_json(),
            structured_event=event,
            metadata={
                "meaning": "qwen3_temporal_feeding_verification",
                "temporal_keyframe_ids": [
                    int(keyframe_id) for keyframe_id in keyframe_ids
                ],
                "temporal_frame_count": len(frame_paths),
            },
        )

    def classify_temporal_batches(
        self,
        *,
        task: str,
        frame_groups: Sequence[Sequence[Any]],
        contexts: Sequence[VisionAnalysisContext],
        keyframe_groups: Sequence[Sequence[int]],
    ) -> list[Verification]:
        if not (
            len(frame_groups) == len(contexts) == len(keyframe_groups)
        ):
            raise ValueError(
                "frame_groups, contexts and keyframe_groups must align"
            )
        if task == "feeding":
            prompt = TEMPORAL_FEEDING_PROMPT
            label_for_event = _feeding_label
            meaning = "qwen3_native_temporal_feeding_verification"
        elif task == "toy":
            prompt = TEMPORAL_PLAY_PROMPT
            label_for_event = _toy_label
            meaning = "qwen3_native_temporal_play_verification"
        elif task == "sleep_presence":
            prompt = SLEEP_PRESENCE_PROMPT
            label_for_event = _sleep_presence_label
            meaning = "qwen3_native_sleep_presence_verification"
        elif task == "posture":
            prompt = TEMPORAL_POSTURE_PROMPT
            label_for_event = _posture_label
            meaning = "qwen3_native_temporal_posture_verification"
        else:
            raise ValueError(
                f"Unsupported temporal verification task: {task}"
            )
        results: list[Verification | None] = [None] * len(frame_groups)
        valid_indexes = [
            index
            for index, frames in enumerate(frame_groups)
            if len(frames) >= 3
        ]
        if valid_indexes:
            events = self._model.analyze_video_batches(
                [frame_groups[index] for index in valid_indexes],
                [prompt] * len(valid_indexes),
                [contexts[index] for index in valid_indexes],
            )
            for index, event in zip(
                valid_indexes,
                events,
                strict=True,
            ):
                label = label_for_event(event)
                if (
                    task == "feeding"
                    and label
                    in {"FEED_BOTTLE", "FEED_BREAST", "FEED_SOLID"}
                    and (
                        event.confidence < 0.85
                        or event.uncertain
                    )
                ):
                    label = "UNCLEAR"
                if (
                    task in {"toy", "sleep_presence", "posture"}
                    and label not in {
                        "NONE",
                        "NOT_SLEEPING",
                        "OTHER",
                        "UNCLEAR",
                    }
                    and (
                        event.confidence < 0.78
                        or event.uncertain
                    )
                ):
                    label = "UNCLEAR"
                results[index] = Verification(
                    label=label,
                    confidence=event.confidence,
                    raw_output=event.to_json(),
                    structured_event=event,
                    metadata={
                        "meaning": meaning,
                        "temporal_keyframe_ids": [
                            int(keyframe_id)
                            for keyframe_id in keyframe_groups[index]
                        ],
                        "temporal_frame_count": len(frame_groups[index]),
                    },
                )
        for index, result in enumerate(results):
            if result is not None:
                continue
            event = StructuredVisualEvent(
                activity="unclear",
                description="连续可用原始画面不足，无法确认活动。",
                objects=(),
                confidence=0.0,
                evidence=("少于三张连续原始画面",),
                uncertain=True,
            )
            results[index] = Verification(
                label="UNCLEAR",
                confidence=0.0,
                raw_output=event.to_json(),
                structured_event=event,
                metadata={
                    "meaning": meaning,
                    "temporal_keyframe_ids": [
                        int(keyframe_id)
                        for keyframe_id in keyframe_groups[index]
                    ],
                    "temporal_frame_count": len(frame_groups[index]),
                },
            )
        return [
            result
            for result in results
            if result is not None
        ]

    def close(self) -> None:
        self._model.close()


def _chunks(values: Sequence[int], size: int = 800):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _frame_metadata(
    connection: sqlite3.Connection,
    keyframe_ids: Sequence[int],
    *,
    image_root: Path,
) -> tuple[
    dict[int, Path],
    dict[int, list[dict[str, Any]]],
    dict[int, VisionAnalysisContext],
]:
    paths: dict[int, Path] = {}
    boxes: dict[int, list[dict[str, Any]]] = {
        keyframe_id: [] for keyframe_id in keyframe_ids
    }
    context_rows: dict[int, dict[str, Any]] = {}
    for chunk in _chunks(list(keyframe_ids)):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"""
            SELECT
                k.id, k.relative_image_path, k.wall_ts, k.motion_score,
                c.name AS camera_name
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            JOIN cameras c ON c.id = m.camera_id
            WHERE k.id IN ({placeholders})
            """,
            chunk,
        ):
            keyframe_id = int(row["id"])
            paths[keyframe_id] = image_root.joinpath(
                str(row["relative_image_path"])
            )
            context_rows[keyframe_id] = dict(row)
        for row in connection.execute(
            f"""
            SELECT
                keyframe_id, class_name, confidence,
                x1, y1, x2, y2
            FROM visual_detections
            WHERE keyframe_id IN ({placeholders})
              AND confidence >= 0.18
            ORDER BY keyframe_id, confidence DESC
            """,
            chunk,
        ):
            boxes[int(row["keyframe_id"])].append(dict(row))
    contexts: dict[int, VisionAnalysisContext] = {}
    for keyframe_id, row in context_rows.items():
        detections = boxes.get(keyframe_id, [])
        objects = tuple(
            dict.fromkeys(
                str(item["class_name"])
                for item in detections
            )
        )
        motion_score = row.get("motion_score")
        if motion_score is None:
            motion = "unknown"
        elif float(motion_score) >= 0.12:
            motion = "active"
        elif float(motion_score) <= 0.03:
            motion = "still"
        else:
            motion = "low"
        wall_ts = row.get("wall_ts")
        time_value = (
            datetime.fromtimestamp(float(wall_ts)).astimezone().isoformat(
                sep=" ",
                timespec="seconds",
            )
            if wall_ts is not None
            else "unknown"
        )
        contexts[keyframe_id] = VisionAnalysisContext(
            camera=str(row["camera_name"]),
            time=time_value,
            objects=objects,
            motion=motion,
            detections=tuple(
                {
                    "class_name": str(item["class_name"]),
                    "confidence": round(
                        float(item["confidence"]),
                        4,
                    ),
                    "box": [
                        round(float(item["x1"]), 4),
                        round(float(item["y1"]), 4),
                        round(float(item["x2"]), 4),
                        round(float(item["y2"]), 4),
                    ],
                }
                for item in detections[:20]
            ),
        )
    return paths, boxes, contexts


def _existing_verifications(
    connection: sqlite3.Connection,
    *,
    task: str,
    model_name: str,
    prompt_version: str,
) -> set[int]:
    return {
        int(row[0])
        for row in connection.execute(
            """
            SELECT keyframe_id
            FROM visual_frame_verifications
            WHERE task = ?
              AND model_name = ?
              AND prompt_version = ?
            """,
            (task, model_name, prompt_version),
        )
    }


def _existing_native_verifications(
    connection: sqlite3.Connection,
    *,
    task: str,
    model_name: str,
    prompt_version: str,
) -> set[int]:
    result: set[int] = set()
    for row in connection.execute(
        """
        SELECT keyframe_id, details_json
        FROM visual_frame_verifications
        WHERE task = ?
          AND model_name = ?
          AND prompt_version = ?
        """,
        (task, model_name, prompt_version),
    ):
        try:
            details = json.loads(str(row["details_json"] or "{}"))
        except (TypeError, ValueError):
            details = {}
        if details.get("native_source") is True:
            result.add(int(row["keyframe_id"]))
    return result


def _existing_native_posture_observations(
    connection: sqlite3.Connection,
    *,
    model_name: str,
    prompt_version: str,
) -> set[int]:
    result: set[int] = set()
    for row in connection.execute(
        """
        SELECT keyframe_id, details_json
        FROM visual_observations
        WHERE observation_group = 'posture_verification'
          AND model_name = ?
          AND prompt_version = ?
        """,
        (model_name, prompt_version),
    ):
        try:
            details = json.loads(str(row["details_json"] or "{}"))
        except (TypeError, ValueError):
            details = {}
        if details.get("native_source") is True:
            result.add(int(row["keyframe_id"]))
    return result


def _verification_details(result: Verification) -> dict[str, Any]:
    details = {
        "raw_output": result.raw_output,
        "structured_event": (
            result.structured_event.to_dict()
            if result.structured_event is not None
            else None
        ),
        "meaning": "local_vlm_secondary_verification",
    }
    if result.metadata:
        details.update(dict(result.metadata))
    return details


def save_verifications(
    connection: sqlite3.Connection,
    *,
    task: str,
    prompt_version: str,
    model_name: str,
    keyframe_ids: Sequence[int],
    results: Sequence[Verification],
) -> int:
    if len(keyframe_ids) != len(results):
        raise ValueError("keyframe_ids and results must align")
    now_ts = int(time.time())
    before = connection.total_changes
    connection.executemany(
        """
        INSERT INTO visual_frame_verifications(
            keyframe_id, task, label, confidence,
            model_name, prompt_version, details_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            keyframe_id, task, model_name, prompt_version
        ) DO UPDATE SET
            label = excluded.label,
            confidence = excluded.confidence,
            details_json = excluded.details_json,
            created_at = excluded.created_at
        """,
        [
            (
                keyframe_id,
                task,
                result.label,
                min(1.0, max(0.0, result.confidence)),
                model_name,
                prompt_version,
                json.dumps(
                    _verification_details(result),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                now_ts,
            )
            for keyframe_id, result in zip(
                keyframe_ids,
                results,
                strict=True,
            )
        ],
    )
    return connection.total_changes - before


def save_posture_observations(
    connection: sqlite3.Connection,
    *,
    prompt_version: str,
    model_name: str,
    keyframe_ids: Sequence[int],
    results: Sequence[Verification],
) -> int:
    if len(keyframe_ids) != len(results):
        raise ValueError("keyframe_ids and results must align")
    now_ts = int(time.time())
    for keyframe_id, result in zip(
        keyframe_ids,
        results,
        strict=True,
    ):
        connection.execute(
            """
            DELETE FROM visual_observations
            WHERE keyframe_id = ?
              AND observation_group = 'posture_verification'
              AND model_name = ?
              AND prompt_version = ?
            """,
            (keyframe_id, model_name, prompt_version),
        )
        connection.execute(
            """
            INSERT INTO visual_observations(
                keyframe_id,
                observation_group,
                label,
                score,
                model_name,
                prompt_version,
                details_json,
                created_at
            )
            VALUES (?, 'posture_verification', ?, ?, ?, ?, ?, ?)
            """,
            (
                keyframe_id,
                result.label,
                min(1.0, max(0.0, result.confidence)),
                model_name,
                prompt_version,
                json.dumps(
                    _verification_details(result),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                now_ts,
            ),
        )
    return len(results)


def _feeding_recall_rank(frame: Any) -> float:
    raw = frame.raw_sigmoids.get("feeding", {})
    scores = frame.scores.get("feeding", {})
    labels = ("feeding_bottle", "breastfeeding", "solid_food")
    label = max(labels, key=lambda value: float(raw.get(value, 0.0)))
    semantic = float(raw.get(label, 0.0))
    normalized = float(scores.get(label, 0.0))
    negative = float(raw.get("not_feeding", 0.0))
    person = min(1.0, max(0.0, float(frame.person_confidence) / 0.5))
    return (
        semantic
        * (0.65 + 0.35 * normalized)
        * (0.75 + 0.25 * person)
        / (0.08 + max(0.0, negative))
    )


def select_temporal_feeding_keyframes(
    frames: Sequence[Any],
    *,
    base_candidate_ids: set[int] | None = None,
    bucket_seconds: int = 5 * 60,
    minimum_rank: float = 0.25,
    max_per_bucket: int = 2,
) -> list[int]:
    """Select broad, time-deduplicated feeding recall candidates.

    SigLIP is used only as a cheap recall signal. A candidate is not a
    feeding event until the local VLM checks neighboring frames.
    """
    if bucket_seconds < 1:
        raise ValueError("bucket_seconds must be positive")
    if max_per_bucket < 1:
        raise ValueError("max_per_bucket must be positive")
    base_ids = base_candidate_ids or set()
    selected: dict[
        tuple[str, int], list[tuple[float, float, int]]
    ] = defaultdict(list)
    for frame in frames:
        raw = frame.raw_sigmoids.get("feeding", {})
        scores = frame.scores.get("feeding", {})
        labels = ("feeding_bottle", "breastfeeding", "solid_food")
        label = max(labels, key=lambda value: float(raw.get(value, 0.0)))
        semantic = float(raw.get(label, 0.0))
        normalized = float(scores.get(label, 0.0))
        rank = _feeding_recall_rank(frame)
        base_candidate = int(frame.keyframe_id) in base_ids
        broad_candidate = (
            semantic >= 0.04
            and normalized >= 0.40
            and rank >= minimum_rank
        )
        legacy_positive = frame.verified_feeding_label in {
            "FEED_BOTTLE",
            "FEED_BREAST",
            "FEED_SOLID",
        }
        if not (base_candidate or broad_candidate or legacy_positive):
            continue
        # Legacy 640x360 VLM positives and YOLO bottle hits are useful recall
        # hints but proved too noisy to rank above the visual semantics. Native
        # results are stored under another prompt and never need re-selection.
        quality = rank
        key = (
            str(frame.camera_id),
            int(float(frame.wall_ts) // bucket_seconds),
        )
        candidate = (
            quality,
            float(frame.wall_ts),
            int(frame.keyframe_id),
        )
        selected[key].append(candidate)
    bucket_peaks = [
        candidate
        for values in selected.values()
        for candidate in sorted(values, reverse=True)[:max_per_bucket]
    ]
    return [
        item[2]
        for item in sorted(
            bucket_peaks,
            # Process the strongest visual candidates across the whole month
            # first. Reverse-chronological ordering previously spent every
            # bounded worker run on the newest days and left the beginning of
            # the month permanently under-verified.
            key=lambda value: (value[0], -value[1], -value[2]),
            reverse=True,
        )
    ]


def select_temporal_event_keyframes(
    frames: Sequence[Any],
    *,
    activity_type: str,
    sleep_bucket_seconds: int = 10 * 60,
) -> list[int]:
    """Choose high-signal native-video anchors for preliminary events.

    Play and posture events are short enough for one representative anchor.
    Sleep can span hours, so one anchor every ten minutes is needed to trim
    empty-bed time and estimate its boundaries.
    """
    if activity_type not in {
        "play",
        "inferred_sleep",
        "crawling",
        "sitting",
    }:
        raise ValueError(
            f"Unsupported temporal event type: {activity_type}"
        )
    from .activity_inference import frame_candidates, merge_candidates

    candidates = [
        candidate
        for frame in frames
        for candidate in frame_candidates(
            frame,
            use_verification=False,
        )
        if candidate.activity_type == activity_type
    ]
    events = merge_candidates(candidates)
    anchors: list[tuple[float, float, int]] = []
    for event in events:
        evidence = list(
            event.get("all_evidence")
            or event.get("evidence")
            or ()
        )
        if not evidence:
            continue
        if activity_type == "inferred_sleep":
            if sleep_bucket_seconds < 1:
                raise ValueError(
                    "sleep_bucket_seconds must be positive"
                )
            buckets: dict[int, Any] = {}
            for item in evidence:
                item_midpoint = (
                    float(item.start_ts) + float(item.end_ts)
                ) / 2.0
                bucket = int(item_midpoint // sleep_bucket_seconds)
                current = buckets.get(bucket)
                if current is None or (
                    -abs(
                        item_midpoint
                        - (
                            bucket + 0.5
                        )
                        * sleep_bucket_seconds
                    ),
                    float(item.confidence),
                    -int(item.keyframe_id),
                ) > (
                    -abs(
                        (
                            float(current.start_ts)
                            + float(current.end_ts)
                        )
                        / 2.0
                        - (
                            bucket + 0.5
                        )
                        * sleep_bucket_seconds
                    ),
                    float(current.confidence),
                    -int(current.keyframe_id),
                ):
                    buckets[bucket] = item
            for item in buckets.values():
                anchors.append(
                    (
                        float(item.confidence),
                        float(item.start_ts),
                        int(item.keyframe_id),
                    )
                )
            continue
        midpoint = (
            float(event["start_ts"]) + float(event["end_ts"])
        ) / 2.0
        anchor = max(
            evidence,
            key=lambda item: (
                float(item.confidence),
                -abs(
                    (
                        float(item.start_ts) + float(item.end_ts)
                    )
                    / 2.0
                    - midpoint
                ),
                -int(item.keyframe_id),
            ),
        )
        anchors.append(
            (
                float(anchor.confidence),
                float(event["start_ts"]),
                int(anchor.keyframe_id),
            )
        )
    return [
        item[2]
        for item in sorted(
            anchors,
            key=lambda value: (value[0], -value[1], -value[2]),
            reverse=True,
        )
    ]


def _temporal_neighbor_rows(
    connection: sqlite3.Connection,
    keyframe_id: int,
    *,
    frame_count: int = 5,
    radius_seconds: float = 120.0,
) -> list[sqlite3.Row]:
    target = connection.execute(
        """
        SELECT k.wall_ts, m.camera_id
        FROM visual_keyframes k
        JOIN media_files m ON m.id = k.media_file_id
        WHERE k.id = ?
        """,
        (keyframe_id,),
    ).fetchone()
    if target is None or target["wall_ts"] is None:
        return []
    wall_ts = float(target["wall_ts"])
    rows = list(
        connection.execute(
            """
            SELECT
                k.id,
                k.wall_ts,
                k.relative_image_path,
                k.media_offset_ms,
                m.relative_path AS media_relative_path,
                m.width AS media_width,
                m.height AS media_height
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            WHERE m.camera_id = ?
              AND k.wall_ts BETWEEN ? AND ?
            ORDER BY ABS(k.wall_ts - ?), k.wall_ts, k.id
            LIMIT ?
            """,
            (
                str(target["camera_id"]),
                wall_ts - radius_seconds,
                wall_ts + radius_seconds,
                wall_ts,
                max(3, int(frame_count)),
            ),
        )
    )
    rows.sort(key=lambda row: (float(row["wall_ts"]), int(row["id"])))
    return rows


def _native_frames_for_rows(
    rows: Sequence[sqlite3.Row],
    *,
    media_root: Path,
) -> tuple[list[Any], list[int], list[str]]:
    """Read original-resolution frames with one decoder handle per video."""
    import cv2
    from PIL import Image

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        relative = str(row["media_relative_path"] or "")
        if relative:
            grouped.setdefault(relative, []).append(row)
    decoded: dict[int, Any] = {}
    errors: list[str] = []
    for relative, group in grouped.items():
        source = media_root.joinpath(
            *relative.replace("\\", "/").split("/")
        )
        if not source.is_file():
            errors.append(f"missing:{relative}")
            continue
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                errors.append(f"open_failed:{relative}")
                continue
            for row in sorted(
                group,
                key=lambda item: float(item["media_offset_ms"]),
            ):
                keyframe_id = int(row["id"])
                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    float(row["media_offset_ms"]),
                )
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    errors.append(
                        f"decode_failed:{relative}:{keyframe_id}"
                    )
                    continue
                height, width = bgr.shape[:2]
                # A 640x360 cached keyframe is precisely what caused small
                # bottles and empty beds to be confused. Do not silently call
                # a low-resolution decode "native evidence".
                expected_width = int(row["media_width"] or 0)
                expected_height = int(row["media_height"] or 0)
                if (
                    width < max(960, int(expected_width * 0.70))
                    or height < max(540, int(expected_height * 0.70))
                ):
                    errors.append(
                        f"low_resolution:{relative}:{keyframe_id}:"
                        f"{width}x{height}"
                    )
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded[keyframe_id] = Image.fromarray(rgb)
        finally:
            capture.release()
    ordered_ids = [
        int(row["id"])
        for row in rows
        if int(row["id"]) in decoded
    ]
    return (
        [decoded[keyframe_id] for keyframe_id in ordered_ids],
        ordered_ids,
        errors,
    )


def _native_anchor_metadata(
    connection: sqlite3.Connection,
    keyframe_ids: Sequence[int],
) -> dict[int, sqlite3.Row]:
    result: dict[int, sqlite3.Row] = {}
    for chunk in _chunks(list(keyframe_ids)):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"""
            SELECT
                k.id,
                k.media_offset_ms,
                m.relative_path AS media_relative_path,
                m.width AS media_width,
                m.height AS media_height,
                m.duration_seconds
            FROM visual_keyframes k
            JOIN media_files m ON m.id = k.media_file_id
            WHERE k.id IN ({placeholders})
            """,
            tuple(chunk),
        ):
            result[int(row["id"])] = row
    return result


def _native_frames_around_anchor(
    row: sqlite3.Row | None,
    *,
    media_root: Path,
    deltas_seconds: Sequence[float],
) -> tuple[list[Any], list[int], list[str], list[int]]:
    """Decode a short, dense native-resolution sequence around one anchor."""
    import cv2
    from PIL import Image

    if row is None:
        return [], [], ["missing_anchor_metadata"], []
    relative = str(row["media_relative_path"] or "")
    source = media_root.joinpath(
        *relative.replace("\\", "/").split("/")
    )
    if not source.is_file():
        return [], [], [f"missing:{relative}"], []
    anchor_ms = int(row["media_offset_ms"])
    duration_ms = max(
        0,
        int(float(row["duration_seconds"] or 0.0) * 1000.0),
    )
    desired = [
        anchor_ms + int(delta * 1000.0)
        for delta in deltas_seconds
    ]
    window_start = max(0, min(desired))
    window_end = min(
        max(0, duration_ms - 1),
        max(desired),
    )
    sample_count = max(1, len(deltas_seconds))
    if window_end > window_start and sample_count > 1:
        offsets = [
            round(
                window_start
                + (window_end - window_start)
                * index
                / (sample_count - 1)
            )
            for index in range(sample_count)
        ]
    else:
        offsets = [min(max(0, anchor_ms), max(0, duration_ms - 1))]
    offsets = sorted(set(offsets))
    capture = cv2.VideoCapture(str(source))
    images: list[Any] = []
    decoded_offsets: list[int] = []
    errors: list[str] = []
    keyframe_id = int(row["id"])
    try:
        if not capture.isOpened():
            return [], [], [f"open_failed:{relative}"], []
        for offset_ms in offsets:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(offset_ms))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                errors.append(
                    f"decode_failed:{relative}:{offset_ms}"
                )
                continue
            height, width = bgr.shape[:2]
            expected_width = int(row["media_width"] or 0)
            expected_height = int(row["media_height"] or 0)
            if (
                width < max(960, int(expected_width * 0.70))
                or height < max(540, int(expected_height * 0.70))
            ):
                errors.append(
                    f"low_resolution:{relative}:{offset_ms}:"
                    f"{width}x{height}"
                )
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
            decoded_offsets.append(offset_ms)
    finally:
        capture.release()
    return (
        images,
        [keyframe_id] * len(images),
        errors,
        decoded_offsets,
    )


def _close_images(images: Sequence[Any]) -> None:
    for image in images:
        close = getattr(image, "close", None)
        if callable(close):
            close()


def _temporal_prompt_version(task: str) -> str:
    return {
        "feeding": TEMPORAL_FEEDING_PROMPT_VERSION,
        "toy": TEMPORAL_PLAY_PROMPT_VERSION,
        "sleep_presence": SLEEP_PRESENCE_PROMPT_VERSION,
        "posture": TEMPORAL_POSTURE_PROMPT_VERSION,
    }[task]


def _verify_temporal_task(
    connection: sqlite3.Connection,
    *,
    verifier: ConfiguredVisualVerifier,
    task: str,
    candidate_ids: Sequence[int],
    image_root: Path,
    media_root: Path | None,
    batch_size: int,
    max_frames: int | None,
    force: bool,
    on_progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    prompt_version = _temporal_prompt_version(task)
    candidates = list(dict.fromkeys(int(value) for value in candidate_ids))
    if force:
        existing: set[int] = set()
    elif task == "posture":
        existing = _existing_native_posture_observations(
            connection,
            model_name=verifier.model_name,
            prompt_version=prompt_version,
        )
    else:
        existing = _existing_native_verifications(
            connection,
            task=task,
            model_name=verifier.model_name,
            prompt_version=prompt_version,
        )
    pending = [
        keyframe_id
        for keyframe_id in candidates
        if keyframe_id not in existing
    ]
    if max_frames is not None:
        pending = pending[: max(0, int(max_frames))]
    payload: dict[str, Any] = {
        "candidates": len(candidates),
        "pending": len(pending),
        "saved": 0,
        "labels": {},
        "missing_native_sequences": 0,
        "native_frames": 0,
        "decode_errors": 0,
    }
    if not pending:
        return payload
    _, _, contexts = _frame_metadata(
        connection,
        pending,
        image_root=image_root,
    )
    anchor_metadata = _native_anchor_metadata(connection, pending)
    deltas_seconds = (
        (-60.0, 0.0, 60.0)
        if task == "sleep_presence"
        else (-20.0, -10.0, 0.0, 10.0, 20.0)
    )
    # Feeding/play/posture retain five source frames, where four sequences keep
    # enough VRAM headroom for verbose JSON. Sleep presence only needs the
    # start/center/end of a two-minute window; eight three-frame sequences fit
    # comfortably on the 32 GB worker GPU while using its idle VRAM headroom.
    batch_cap = 8 if task == "sleep_presence" else 4
    effective_batch_size = max(1, min(int(batch_size), batch_cap))

    batch_groups = [
        pending[start : start + effective_batch_size]
        for start in range(0, len(pending), effective_batch_size)
    ]

    def decode_batch(
        batch_ids: Sequence[int],
    ) -> list[tuple[list[Any], list[int], list[str], list[int]]]:
        if media_root is None:
            return [
                ([], [], ["media_root_not_configured"], [])
                for _ in batch_ids
            ]
        with ThreadPoolExecutor(
            max_workers=min(4, len(batch_ids))
        ) as executor:
            return list(
                executor.map(
                    lambda keyframe_id: _native_frames_around_anchor(
                        anchor_metadata.get(keyframe_id),
                        media_root=media_root,
                        deltas_seconds=deltas_seconds,
                    ),
                    batch_ids,
                )
            )

    # NAS seeks and native-frame decoding previously left the GPU idle between
    # batches. Keep exactly one batch prefetched on the CPU while Qwen analyzes
    # the current batch; this improves utilization without increasing VRAM.
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    decode_future = prefetch_executor.submit(
        decode_batch,
        batch_groups[0],
    )
    try:
        for batch_index, batch_ids in enumerate(batch_groups):
            decoded = decode_future.result()
            decode_future = (
                prefetch_executor.submit(
                    decode_batch,
                    batch_groups[batch_index + 1],
                )
                if batch_index + 1 < len(batch_groups)
                else None
            )
            frame_groups = [item[0] for item in decoded]
            keyframe_groups = [item[1] for item in decoded]
            errors_by_group = [item[2] for item in decoded]
            offsets_by_group = [item[3] for item in decoded]
            try:
                for frames, errors in zip(
                    frame_groups,
                    errors_by_group,
                    strict=True,
                ):
                    payload["native_frames"] += len(frames)
                    payload["decode_errors"] += len(errors)
                    if len(frames) < 3:
                        payload["missing_native_sequences"] += 1
                batch_contexts = [
                    replace(
                        contexts[keyframe_id],
                        # Detector labels are useful for broad recall but
                        # biased the VLM toward hallucinating bottles and
                        # people in empty beds. Native verification must be
                        # grounded in pixels alone.
                        objects=(),
                        detections=(),
                    )
                    for keyframe_id in batch_ids
                ]
                results = verifier.classify_temporal_batches(
                    task=task,
                    frame_groups=frame_groups,
                    contexts=batch_contexts,
                    keyframe_groups=keyframe_groups,
                )
                enriched: list[Verification] = []
                for (
                    result,
                    images,
                    native_ids,
                    errors,
                    sample_offsets,
                ) in zip(
                    results,
                    frame_groups,
                    keyframe_groups,
                    errors_by_group,
                    offsets_by_group,
                    strict=True,
                ):
                    metadata = dict(result.metadata or {})
                    metadata.update(
                        {
                            "native_source": len(images) >= 3,
                            "native_keyframe_ids": list(native_ids),
                            "native_sample_offsets_ms": list(sample_offsets),
                            "native_frame_sizes": [
                                list(getattr(image, "size", (0, 0)))
                                for image in images
                            ],
                            "native_decode_errors": list(errors),
                        }
                    )
                    enriched.append(replace(result, metadata=metadata))
            finally:
                _close_images(
                    [
                        image
                        for group in frame_groups
                        for image in group
                    ]
                )
            if task == "posture":
                saved = _retry_sqlite_write(
                    lambda: save_posture_observations(
                        connection,
                        prompt_version=prompt_version,
                        model_name=verifier.model_name,
                        keyframe_ids=batch_ids,
                        results=enriched,
                    )
                )
            else:
                saved = _retry_sqlite_write(
                    lambda: save_verifications(
                        connection,
                        task=task,
                        prompt_version=prompt_version,
                        model_name=verifier.model_name,
                        keyframe_ids=batch_ids,
                        results=enriched,
                    )
                )
            _retry_sqlite_write(connection.commit)
            payload["saved"] += saved
            labels = payload["labels"]
            for result in enriched:
                labels[result.label] = labels.get(result.label, 0) + 1
            processed = min(
                (batch_index + 1) * effective_batch_size,
                len(pending),
            )
            if on_progress is not None:
                on_progress(
                    {
                        "task": f"{task}_temporal",
                        "saved": payload["saved"],
                        "pending": len(pending),
                        "processed": processed,
                        "labels": dict(labels),
                        "native_frames": payload["native_frames"],
                        "decode_errors": payload["decode_errors"],
                    }
                )
    finally:
        # If analysis fails, finish the one in-flight CPU decode and release
        # its PIL images before propagating the error.
        if decode_future is not None:
            try:
                abandoned = decode_future.result()
            except Exception:
                abandoned = []
            _close_images(
                [
                    image
                    for decoded_group in abandoned
                    for image in decoded_group[0]
                ]
            )
        prefetch_executor.shutdown(wait=True)
    return payload


def verify_candidate_frames(
    connection: sqlite3.Connection,
    *,
    since_ts: int,
    until_ts: int,
    camera_ids: Sequence[str],
    embedding_model: str,
    prompt_version: str,
    image_root: str | Path,
    media_root: str | Path | None = None,
    model_dir: str | Path | None = None,
    models_config_path: str | Path = "config/models.toml",
    tasks: Sequence[str] = (
        "feeding",
        "toy",
        "sleep_presence",
        "posture",
    ),
    batch_size: int = 8,
    max_frames: int | None = 0,
    temporal_max_frames: int | None = 256,
    feeding_bucket_seconds: int = 5 * 60,
    feeding_minimum_rank: float = 0.25,
    feeding_dates: Sequence[str] = (),
    timezone_name: str = "Asia/Shanghai",
    force: bool = False,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from .activity_inference import (
        frame_candidates,
        load_frame_observations,
    )

    frames = load_frame_observations(
        connection,
        since_ts=since_ts,
        until_ts=until_ts,
        camera_ids=camera_ids,
        embedding_model=embedding_model,
        prompt_version=prompt_version,
    )
    selected_tasks = tuple(dict.fromkeys(tasks))
    unsupported = set(selected_tasks) - {
        "feeding",
        "toy",
        "sleep_presence",
        "posture",
    }
    if unsupported:
        raise ValueError(
            "Unsupported verification tasks: "
            + ", ".join(sorted(unsupported))
        )
    candidate_ids: dict[str, set[int]] = {
        task: set() for task in selected_tasks
    }
    for frame in frames:
        types = {
            item.activity_type
            for item in frame_candidates(
                frame,
                use_verification=False,
            )
        }
        if "feeding" in candidate_ids and "feeding" in types:
            candidate_ids["feeding"].add(frame.keyframe_id)
        if "toy" in candidate_ids and "play" in types:
            candidate_ids["toy"].add(frame.keyframe_id)

    temporal_candidates: dict[str, list[int]] = {}
    if "feeding" in candidate_ids:
        selected_feeding_dates = {
            datetime.fromisoformat(value).date().isoformat()
            for value in feeding_dates
        }
        feeding_timezone = ZoneInfo(timezone_name)
        feeding_frames = (
            [
                frame
                for frame in frames
                if datetime.fromtimestamp(
                    float(frame.wall_ts),
                    feeding_timezone,
                ).date().isoformat()
                in selected_feeding_dates
            ]
            if selected_feeding_dates
            else frames
        )
        temporal_candidates["feeding"] = (
            select_temporal_feeding_keyframes(
                feeding_frames,
                base_candidate_ids=candidate_ids["feeding"],
                bucket_seconds=feeding_bucket_seconds,
                minimum_rank=feeding_minimum_rank,
            )
        )
    if "toy" in candidate_ids:
        temporal_candidates["toy"] = (
            select_temporal_event_keyframes(
                frames,
                activity_type="play",
            )
        )
    if "sleep_presence" in candidate_ids:
        temporal_candidates["sleep_presence"] = (
            select_temporal_event_keyframes(
                frames,
                activity_type="inferred_sleep",
            )
        )
    if "posture" in candidate_ids:
        temporal_candidates["posture"] = list(
            dict.fromkeys(
                [
                    *select_temporal_event_keyframes(
                        frames,
                        activity_type="crawling",
                    ),
                    *select_temporal_event_keyframes(
                        frames,
                        activity_type="sitting",
                    ),
                ]
            )
        )

    models_config = load_models_config(models_config_path)
    model_config = models_config.vision_language
    if model_dir is not None:
        model_config = _config_for_model_dir(model_config, model_dir)
    versions = {
        "feeding": STRUCTURED_FEEDING_PROMPT_VERSION,
        "toy": STRUCTURED_TOY_PROMPT_VERSION,
    }
    verifier_model_name = model_config.model_id
    pending: dict[str, list[int]] = {}
    for task, values in candidate_ids.items():
        if task not in versions:
            pending[task] = []
            continue
        existing = (
            set()
            if force
        else _existing_verifications(
            connection,
            task=task,
            model_name=verifier_model_name,
            prompt_version=versions[task],
            )
        )
        selected = sorted(values - existing)
        if max_frames is not None:
            selected = selected[: max(0, max_frames)]
        pending[task] = selected

    temporal_pending: dict[str, list[int]] = {}
    for task, values in temporal_candidates.items():
        if force:
            existing = set()
        elif task == "posture":
            existing = _existing_native_posture_observations(
                connection,
                model_name=verifier_model_name,
                prompt_version=_temporal_prompt_version(task),
            )
        else:
            existing = _existing_native_verifications(
                connection,
                task=task,
                model_name=verifier_model_name,
                prompt_version=_temporal_prompt_version(task),
            )
        selected = [
            keyframe_id
            for keyframe_id in values
            if keyframe_id not in existing
        ]
        if temporal_max_frames is not None:
            selected = selected[
                : max(0, int(temporal_max_frames))
            ]
        temporal_pending[task] = selected
    total_pending = sum(
        len(values) for values in pending.values()
    ) + sum(len(values) for values in temporal_pending.values())
    payload: dict[str, Any] = {
        "model": verifier_model_name,
        "frames_loaded": len(frames),
        "tasks": {},
        "pending": total_pending,
        "temporal": {
            task: {
                "candidates": len(temporal_candidates[task]),
                "pending": len(temporal_pending[task]),
            }
            for task in temporal_candidates
        },
    }
    if not total_pending:
        payload["status"] = "completed"
        return payload

    verifier = ConfiguredVisualVerifier(model_config)
    try:
        root = Path(image_root).resolve()
        native_root = (
            Path(media_root)
            if media_root is not None
            else None
        )
        for task in selected_tasks:
            ids = pending[task]
            task_payload = {
                "candidates": len(candidate_ids[task]),
                "pending": len(ids),
                "saved": 0,
                "labels": {},
            }
            payload["tasks"][task] = task_payload
            if ids and task in versions:
                paths, boxes, contexts = _frame_metadata(
                    connection,
                    ids,
                    image_root=root,
                )
                valid_ids = [
                    keyframe_id
                    for keyframe_id in ids
                    if keyframe_id in paths and paths[keyframe_id].is_file()
                ]
                for start in range(0, len(valid_ids), max(1, batch_size)):
                    batch_ids = valid_ids[
                        start : start + max(1, batch_size)
                    ]
                    results = verifier.classify(
                        task=task,
                        image_paths=[
                            paths[keyframe_id]
                            for keyframe_id in batch_ids
                        ],
                        boxes=[
                            boxes[keyframe_id]
                            for keyframe_id in batch_ids
                        ],
                        contexts=[
                            contexts[keyframe_id]
                            for keyframe_id in batch_ids
                        ],
                        batch_size=batch_size,
                    )
                    saved = _retry_sqlite_write(
                        lambda: save_verifications(
                            connection,
                            task=task,
                            prompt_version=versions[task],
                            model_name=verifier.model_name,
                            keyframe_ids=batch_ids,
                            results=results,
                        )
                    )
                    _retry_sqlite_write(connection.commit)
                    task_payload["saved"] += saved
                    for result in results:
                        labels = task_payload["labels"]
                        labels[result.label] = (
                            labels.get(result.label, 0) + 1
                        )
                    if on_progress is not None:
                        on_progress(
                            {
                                "task": task,
                                "saved": task_payload["saved"],
                                "pending": len(ids),
                                "labels": dict(task_payload["labels"]),
                            }
                        )
                task_payload["missing_images"] = len(ids) - len(valid_ids)
        temporal_payloads: dict[str, Any] = {}
        for task in selected_tasks:
            temporal_payloads[task] = _verify_temporal_task(
                connection,
                verifier=verifier,
                task=task,
                candidate_ids=temporal_candidates[task],
                image_root=root,
                media_root=native_root,
                batch_size=batch_size,
                max_frames=temporal_max_frames,
                force=force,
                on_progress=on_progress,
            )
        payload["temporal"] = temporal_payloads
    finally:
        verifier.close()
    payload["status"] = "completed"
    return payload
