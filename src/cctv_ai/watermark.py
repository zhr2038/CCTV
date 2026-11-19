from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


WATERMARK_RE = re.compile(
    r"(?P<year>\d{4})[/.-](?P<month>\d{2})[/.-](?P<day>\d{2})"
    r"[\s:.-]*(?P<hour>\d{2}):(?P<minute>\d{2})[\s:.-](?P<second>\d{2})"
)


@dataclass(frozen=True)
class WatermarkObservation:
    media_seconds: float
    wall_time: datetime
    confidence: float = 1.0


@dataclass(frozen=True)
class TimeSegment:
    media_start_seconds: float
    media_end_seconds: float | None
    wall_start: datetime
    wall_end: datetime | None
    confidence: float


def parse_watermark_text(text: str, timezone: str) -> datetime:
    normalized = text
    # At low preview resolutions RapidOCR occasionally renders the first
    # slash as a narrow "1", e.g. 2026107/06 for 2026/07/06.  Only repair
    # this tightly scoped seven-digit date prefix; the remaining calendar
    # validation below still rejects impossible values.
    normalized = re.sub(
        r"(?<!\d)(?P<year>20\d{2})1(?P<month>\d{2})(?=[/.-]\d{2}\D)",
        r"\g<year>/\g<month>",
        normalized,
    )
    normalized = re.sub(
        r"(?P<date>20\d{2}[/.-]\d{2}[/.-]\d{2})1"
        r"(?P<clock>\d{2}:\d{2}[\s:.-]\d{2})",
        r"\g<date> \g<clock>",
        normalized,
    )
    match = WATERMARK_RE.search(normalized)
    if not match:
        raise ValueError(f"No timestamp found in OCR text: {text!r}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    return datetime(
        values["year"],
        values["month"],
        values["day"],
        values["hour"],
        values["minute"],
        values["second"],
        tzinfo=ZoneInfo(timezone),
    )


def sample_offsets(
    duration_seconds: float, coarse_interval_seconds: float = 10.0
) -> list[float]:
    if duration_seconds <= 0:
        return []
    offsets: list[float] = []
    value = 0.5
    while value < duration_seconds:
        offsets.append(round(value, 3))
        value += coarse_interval_seconds
    final_offset = max(0.0, duration_seconds - 0.5)
    if not offsets or abs(offsets[-1] - final_offset) > 0.25:
        offsets.append(round(final_offset, 3))
    return offsets


def build_time_segments(
    observations: list[WatermarkObservation],
    *,
    jump_threshold_seconds: float = 30.0,
) -> list[TimeSegment]:
    if not observations:
        return []
    ordered = sorted(observations, key=lambda item: item.media_seconds)
    segments: list[TimeSegment] = []
    segment_start = ordered[0]
    previous = ordered[0]
    confidence = ordered[0].confidence

    for current in ordered[1:]:
        media_delta = current.media_seconds - previous.media_seconds
        wall_delta = (current.wall_time - previous.wall_time).total_seconds()
        residual = abs(wall_delta - media_delta)
        if residual > jump_threshold_seconds:
            segments.append(
                TimeSegment(
                    media_start_seconds=segment_start.media_seconds,
                    media_end_seconds=previous.media_seconds,
                    wall_start=segment_start.wall_time,
                    wall_end=previous.wall_time,
                    confidence=confidence,
                )
            )
            segment_start = current
            confidence = current.confidence
        else:
            confidence = min(confidence, current.confidence)
        previous = current

    segments.append(
        TimeSegment(
            media_start_seconds=segment_start.media_seconds,
            media_end_seconds=previous.media_seconds,
            wall_start=segment_start.wall_time,
            wall_end=previous.wall_time,
            confidence=confidence,
        )
    )
    return segments
