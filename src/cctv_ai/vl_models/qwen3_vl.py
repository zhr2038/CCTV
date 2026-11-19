from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..model_config import VisionLanguageConfig
from .base import (
    StructuredVisualEvent,
    VisionAnalysisContext,
    VisionLanguageModel,
)


SYSTEM_PROMPT = """
你是家庭监控录像的视觉证据分析器。你只能描述图像中直接可见的动作、
姿态、人物和物体，不得评价宝宝的智力、性格、情绪或发育情况，也不得
做医学或发育异常诊断。已有 YOLO 检测是辅助上下文，可能有误；你要结合
画面做二次理解，但不能凭空补全被遮挡的信息。检测置信度低于 0.40 的
物体只是弱候选，绝不能当成事件证据；即使上下文写了 bottle，也必须在
画面中独立看清奶瓶形状和奶嘴。彩色扁平绘本、包装袋、玩具和成人的手
常被误检成 bottle，不能据此判断喂奶。

最终输出必须且只能是一个 JSON 对象，不要 Markdown，不要解释。字段固定：
{
  "activity": "feeding|play|sitting|crawling|standing|lying_awake|sleeping|rolling|speaking|other|unclear",
  "description": "中文、只写看到的事实",
  "objects": ["lowercase object token"],
  "confidence": 0.0,
  "evidence": ["可见证据"],
  "uncertain": false
}
无法直接确认时使用 activity="unclear"、uncertain=true，并降低 confidence。
""".strip()

SUMMARY_SYSTEM_PROMPT = """
你是家庭成长日记的事实整理器。只汇总输入事件中已经记录的可见事实，
不得评价智力、性格、情绪或发育水平，不得诊断。只输出 JSON：
{"summary":"中文事实摘要","event_count":0,"uncertain_count":0}
""".strip()


def _checkpoint_complete(path: Path) -> bool:
    if not path.joinpath("config.json").is_file():
        return False
    return (
        path.joinpath("model.safetensors.index.json").is_file()
        or path.joinpath("model.safetensors").is_file()
    )


def _json_object(value: str) -> Mapping[str, Any]:
    text = value.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for index, character in enumerate(text[start:], start=start):
            if escaped:
                escaped = False
                continue
            if character == "\\" and in_string:
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
            elif not in_string and character == "{":
                depth += 1
            elif not in_string and character == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end < 0:
            raise
        payload = json.loads(text[start:end])
    if not isinstance(payload, Mapping):
        raise ValueError("Model output JSON must be an object")
    return payload


def _open_image(value: Any):
    from PIL import Image

    if isinstance(value, (str, Path)):
        source = Image.open(value)
        try:
            return source.convert("RGB"), True
        finally:
            source.close()
    if isinstance(value, Image.Image):
        if value.mode == "RGB":
            return value, False
        return value.convert("RGB"), True
    raise TypeError(f"Unsupported image input: {type(value).__name__}")


class Qwen3VisionLanguageModel(VisionLanguageModel):
    def __init__(self, config: VisionLanguageConfig) -> None:
        if not config.local_files_only:
            raise ValueError("Qwen runtime must remain local_files_only")
        model_path = config.model_path.resolve()
        if not _checkpoint_complete(model_path):
            raise FileNotFoundError(
                "Local visual model is incomplete or missing: "
                f"{model_path}. Run scripts/download_vlm_model.py explicitly; "
                "runtime download is disabled."
            )
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL dependencies are missing; install the project "
                "visual extras and bitsandbytes"
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

        self._torch = torch
        self.config = config
        self.device = config.device
        self.model_dir = model_path
        self.model_name = config.model_id
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            use_fast=True,
            local_files_only=True,
        )
        self._processor.tokenizer.padding_side = "left"
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": (
                torch.bfloat16
                if config.device == "cuda"
                else torch.float32
            ),
            "attn_implementation": "sdpa",
            "device_map": "auto" if config.device == "cuda" else "cpu",
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **load_kwargs,
        )
        self._model.eval()
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None
        self._input_device = next(self._model.parameters()).device
        if config.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def _messages(
        self,
        images: Sequence[Any],
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        visual_context = (
            context
            if isinstance(context, VisionAnalysisContext)
            else VisionAnalysisContext.from_mapping(context)
        )
        content = [
            {"type": "image", "image": image}
            for image in images
        ]
        content.append(
            {
                "type": "text",
                "text": (
                    "基础检测上下文：\n"
                    f"{json.dumps(visual_context.to_dict(), ensure_ascii=False)}"
                    "\n分析任务：\n"
                    f"{prompt.strip()}"
                ),
            }
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": content},
        ]

    def _generation_kwargs(self) -> dict[str, Any]:
        # The configured value is a safety ceiling. A fixed event object should
        # stay compact; 128 tokens avoids multi-minute long-tail generations
        # when one item in a video batch fails to emit EOS.
        kwargs: dict[str, Any] = {
            "max_new_tokens": min(self.config.max_tokens, 128),
            "use_cache": True,
        }
        if self.config.temperature > 0:
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": 0.9,
                }
            )
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _generate_batch(
        self,
        messages: Sequence[Sequence[Mapping[str, Any]]],
        image_groups: Sequence[Sequence[Any]],
    ) -> list[str]:
        texts = [
            self._processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        processor_images = [
            image
            for group in image_groups
            for image in group
        ]
        inputs = self._processor(
            text=texts,
            images=processor_images,
            padding=True,
            return_tensors="pt",
        ).to(self._input_device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                **self._generation_kwargs(),
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(
                inputs.input_ids,
                generated,
                strict=True,
            )
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def _event_from_output(output: str) -> StructuredVisualEvent:
        try:
            return StructuredVisualEvent.from_mapping(
                _json_object(output)
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return StructuredVisualEvent.uncertain_result(
                "模型输出不是有效的结构化视觉证据。"
            )

    def analyze_image(
        self,
        image: Any,
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None = None,
    ) -> StructuredVisualEvent:
        return self.analyze_images(
            (image,),
            (prompt,),
            (context,),
        )[0]

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
        opened: list[Any] = []
        owned: list[Any] = []
        try:
            for image in images:
                normalized, should_close = _open_image(image)
                opened.append(normalized)
                if should_close:
                    owned.append(normalized)
            messages = [
                self._messages((image,), prompt, context)
                for image, prompt, context in zip(
                    opened,
                    prompts,
                    contexts,
                    strict=True,
                )
            ]
            outputs = self._generate_batch(
                messages,
                [(image,) for image in opened],
            )
            return [
                self._event_from_output(output)
                for output in outputs
            ]
        finally:
            for image in owned:
                image.close()

    def analyze_video_frames(
        self,
        frames: Sequence[Any],
        prompt: str,
        context: VisionAnalysisContext | Mapping[str, Any] | None = None,
    ) -> StructuredVisualEvent:
        if not frames:
            raise ValueError("frames cannot be empty")
        opened: list[Any] = []
        owned: list[Any] = []
        try:
            for frame in frames:
                normalized, should_close = _open_image(frame)
                opened.append(normalized)
                if should_close:
                    owned.append(normalized)
            messages = [self._messages(opened, prompt, context)]
            output = self._generate_batch(messages, [opened])[0]
            return self._event_from_output(output)
        finally:
            for image in owned:
                image.close()

    def analyze_video_batches(
        self,
        frame_groups: Sequence[Sequence[Any]],
        prompts: Sequence[str],
        contexts: Sequence[
            VisionAnalysisContext | Mapping[str, Any] | None
        ],
    ) -> list[StructuredVisualEvent]:
        if not (
            len(frame_groups) == len(prompts) == len(contexts)
        ):
            raise ValueError(
                "frame_groups, prompts and contexts must align"
            )
        if any(not frames for frames in frame_groups):
            raise ValueError("frame groups cannot be empty")
        opened_groups: list[list[Any]] = []
        owned: list[Any] = []
        try:
            for frames in frame_groups:
                opened: list[Any] = []
                for frame in frames:
                    normalized, should_close = _open_image(frame)
                    opened.append(normalized)
                    if should_close:
                        owned.append(normalized)
                opened_groups.append(opened)
            messages = [
                self._messages(frames, prompt, context)
                for frames, prompt, context in zip(
                    opened_groups,
                    prompts,
                    contexts,
                    strict=True,
                )
            ]
            outputs = self._generate_batch(messages, opened_groups)
            return [
                self._event_from_output(output)
                for output in outputs
            ]
        finally:
            for image in owned:
                image.close()

    def generate_summary(
        self,
        events: Sequence[StructuredVisualEvent | Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        payload = [
            (
                event.to_dict()
                if isinstance(event, StructuredVisualEvent)
                else StructuredVisualEvent.from_mapping(event).to_dict()
            )
            for event in events
        ]
        messages = [
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SUMMARY_SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                payload,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                },
            ]
        ]
        text = self._processor.apply_chat_template(
            messages[0],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self._input_device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                **self._generation_kwargs(),
            )
        output = generated[0][len(inputs.input_ids[0]) :]
        decoded = self._processor.decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        try:
            result = dict(_json_object(decoded))
        except (ValueError, TypeError, json.JSONDecodeError):
            return {
                "summary": "事件摘要生成失败，未写入自然语言推断。",
                "event_count": len(payload),
                "uncertain_count": sum(
                    bool(item["uncertain"]) for item in payload
                ),
            }
        summary, filtered = (
            StructuredVisualEvent.from_mapping(
                {
                    "activity": "other",
                    "description": result.get("summary", ""),
                    "objects": [],
                    "confidence": 1.0,
                    "evidence": [],
                    "uncertain": False,
                }
            ).description,
            False,
        )
        return {
            "summary": summary,
            "event_count": len(payload),
            "uncertain_count": sum(
                bool(item["uncertain"]) for item in payload
            ),
            "filtered": filtered,
        }

    def memory_footprint_bytes(self) -> int:
        return int(self._model.get_memory_footprint())

    def close(self) -> None:
        del self._model
        del self._processor
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
