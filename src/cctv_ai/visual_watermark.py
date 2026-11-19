from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .watermark import (
    WatermarkObservation,
    build_time_segments,
    parse_watermark_text,
)


@dataclass(frozen=True)
class VisualWatermarkRun:
    observations: tuple[WatermarkObservation, ...]
    segments: tuple
    frames_selected: int
    recognized: int
    failures: int
    sample_interval_seconds: float
    model: str


class VisualWatermarkRecognizer:
    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "Watermark OCR dependencies are missing; install the "
                "watermark-ocr extra"
            ) from exc
        self._engine = RapidOCR()
        self.model_name = "rapidocr-local-keyframe-v1"

    def recognize(
        self,
        image_path: str | Path,
        *,
        timezone: str,
        source_width: int,
        source_height: int,
        crop: tuple[int, int, int, int],
        minimum_confidence: float = 0.90,
    ) -> WatermarkObservation | None:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        x, y, crop_width, crop_height = crop
        scaled_x = round(x * width / source_width)
        scaled_y = round(y * height / source_height)
        scaled_width = round(crop_width * width / source_width)
        scaled_height = round(crop_height * height / source_height)
        region = image[
            max(0, scaled_y) : min(height, scaled_y + scaled_height),
            max(0, scaled_x) : min(width, scaled_x + scaled_width),
        ]
        if region.size == 0:
            return None
        region = cv2.resize(
            region,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )
        result = self._engine(
            region,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
        if not result.txts or not result.scores:
            return None
        confidence = float(result.scores[0])
        if confidence < minimum_confidence:
            return None
        try:
            wall_time = parse_watermark_text(result.txts[0], timezone)
        except ValueError:
            return None
        return WatermarkObservation(
            media_seconds=0,
            wall_time=wall_time,
            confidence=confidence,
        )


def plausible_watermark_time(
    wall_time: datetime,
    *,
    expected_start_ts: float | None,
    expected_end_ts: float | None,
    maximum_clock_skew_seconds: float = 6 * 3600,
) -> bool:
    """Reject confident OCR text that is impossible for the source file."""
    if expected_start_ts is None:
        return True
    end_ts = (
        float(expected_end_ts)
        if expected_end_ts is not None
        else float(expected_start_ts)
    )
    timestamp = wall_time.timestamp()
    return (
        float(expected_start_ts) - maximum_clock_skew_seconds
        <= timestamp
        <= end_ts + maximum_clock_skew_seconds
    )


def _select_frames(
    frames: Sequence[Mapping[str, Any]],
    interval_seconds: float,
) -> list[Mapping[str, Any]]:
    if not frames:
        return []
    ordered = sorted(frames, key=lambda row: int(row["media_offset_ms"]))
    selected = [ordered[0]]
    next_offset = (
        float(ordered[0]["media_offset_ms"]) / 1000.0 + interval_seconds
    )
    for frame in ordered[1:-1]:
        offset = float(frame["media_offset_ms"]) / 1000.0
        if offset + 0.001 >= next_offset:
            selected.append(frame)
            while next_offset <= offset + 0.001:
                next_offset += interval_seconds
    if ordered[-1]["id"] != selected[-1]["id"]:
        selected.append(ordered[-1])
    return selected


def map_keyframe_watermarks(
    recognizer: VisualWatermarkRecognizer,
    frames: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    timezone: str,
    source_width: int,
    source_height: int,
    crop: tuple[int, int, int, int],
    interval_seconds: float = 120.0,
    minimum_confidence: float = 0.90,
    expected_start_ts: float | None = None,
    expected_end_ts: float | None = None,
) -> VisualWatermarkRun:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    selected = _select_frames(frames, interval_seconds)
    observations: list[WatermarkObservation] = []
    root = Path(output_root).resolve()
    failures = 0
    for frame in selected:
        relative = Path(str(frame["relative_image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe keyframe path")
        path = root.joinpath(relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Keyframe escaped visual output root") from exc
        observation = recognizer.recognize(
            path,
            timezone=timezone,
            source_width=source_width,
            source_height=source_height,
            crop=crop,
            minimum_confidence=minimum_confidence,
        )
        if observation is None:
            failures += 1
            continue
        if not plausible_watermark_time(
            observation.wall_time,
            expected_start_ts=expected_start_ts,
            expected_end_ts=expected_end_ts,
        ):
            failures += 1
            continue
        observations.append(
            WatermarkObservation(
                media_seconds=float(frame["media_offset_ms"]) / 1000.0,
                wall_time=observation.wall_time,
                confidence=observation.confidence,
            )
        )
    if not observations:
        raise RuntimeError("No timestamp could be read from local keyframes")
    segments = build_time_segments(
        observations,
        # Media bundles can contain discontinuities shorter than the OCR
        # interval. Offset rounding is only a few seconds, so a 30-second
        # residual is a conservative boundary for a new wall-clock segment.
        jump_threshold_seconds=30.0,
    )
    return VisualWatermarkRun(
        observations=tuple(observations),
        segments=tuple(segments),
        frames_selected=len(selected),
        recognized=len(observations),
        failures=failures,
        sample_interval_seconds=interval_seconds,
        model=recognizer.model_name,
    )
