from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

from .models import CameraConfig, ParsedCapture


AQARA_DATE_RE = re.compile(r"^\d{8}$")
AQARA_TIME_RE = re.compile(r"^(?P<time>\d{6})\.mp4$", re.IGNORECASE)
XIAOMI_MINUTE_RE = re.compile(
    r"^\d{2}M\d{2}S_(?P<epoch>\d{10})\.mp4$", re.IGNORECASE
)
XIAOMI_BUNDLE_RE = re.compile(
    r"^00_(?P<start>\d{14})_(?P<end>\d{14})\.mp4$", re.IGNORECASE
)


def _local_datetime(value: str, fmt: str, timezone: str) -> datetime:
    return datetime.strptime(value, fmt).replace(tzinfo=ZoneInfo(timezone))


def parse_capture(
    camera: CameraConfig, path: Path, timezone: str
) -> ParsedCapture:
    if camera.kind == "aqara_event":
        date_value = path.parent.name
        match = AQARA_TIME_RE.match(path.name)
        if not AQARA_DATE_RE.match(date_value) or not match:
            raise ValueError(f"Invalid Aqara path: {path}")
        start = _local_datetime(
            date_value + match.group("time"), "%Y%m%d%H%M%S", timezone
        )
        return ParsedCapture(
            capture_start=start,
            capture_end_hint=None,
            time_source="directory_date_plus_filename_time",
            requires_watermark_ocr=False,
        )

    if camera.kind == "xiaomi_minute_event":
        match = XIAOMI_MINUTE_RE.match(path.name)
        if not match:
            raise ValueError(f"Invalid Xiaomi minute-event path: {path}")
        start = datetime.fromtimestamp(
            int(match.group("epoch")), tz=ZoneInfo(timezone)
        )
        expected_hour = start.strftime("%Y%m%d%H")
        if path.parent.name != expected_hour:
            raise ValueError(
                f"Xiaomi epoch/hour mismatch: {path.parent.name} != {expected_hour}"
            )
        return ParsedCapture(
            capture_start=start,
            capture_end_hint=None,
            time_source="filename_unix_epoch",
            requires_watermark_ocr=False,
        )

    if camera.kind == "xiaomi_watermark_bundle":
        match = XIAOMI_BUNDLE_RE.match(path.name)
        if not match:
            raise ValueError(f"Invalid Xiaomi bundle path: {path}")
        start = _local_datetime(match.group("start"), "%Y%m%d%H%M%S", timezone)
        end = _local_datetime(match.group("end"), "%Y%m%d%H%M%S", timezone)
        if end < start:
            end = None
            time_source = (
                "filename_start_only_invalid_end_watermark_required"
            )
        else:
            time_source = "filename_envelope_watermark_required"
        return ParsedCapture(
            capture_start=start,
            capture_end_hint=end,
            time_source=time_source,
            requires_watermark_ocr=True,
        )

    raise ValueError(f"Unsupported camera kind: {camera.kind}")


def _iter_mp4(directory: Path) -> Iterator[Path]:
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if (
                    entry.is_file(follow_symlinks=False)
                    and entry.name.lower().endswith(".mp4")
                ):
                    yield Path(entry.path)
    except FileNotFoundError:
        return


def iter_camera_files(
    source_root: Path,
    camera: CameraConfig,
    timezone: str,
    since: datetime | None = None,
) -> Iterator[Path]:
    camera_root = source_root.joinpath(*camera.relative_root.split("/"))
    if not camera_root.is_dir():
        return

    if camera.kind == "aqara_event":
        minimum_date = since.astimezone(ZoneInfo(timezone)).strftime("%Y%m%d") if since else None
        with os.scandir(camera_root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not AQARA_DATE_RE.match(entry.name):
                    continue
                if minimum_date and entry.name < minimum_date:
                    continue
                yield from _iter_mp4(Path(entry.path))
        return

    if camera.kind == "xiaomi_minute_event":
        minimum_hour = since.astimezone(ZoneInfo(timezone)).strftime("%Y%m%d%H") if since else None
        with os.scandir(camera_root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if len(entry.name) != 10 or not entry.name.isdigit():
                    continue
                if minimum_hour and entry.name < minimum_hour:
                    continue
                yield from _iter_mp4(Path(entry.path))
        return

    if camera.kind == "xiaomi_watermark_bundle":
        for path in _iter_mp4(camera_root):
            if since is None:
                yield path
                continue
            try:
                parsed = parse_capture(camera, path, timezone)
            except ValueError:
                yield path
                continue
            if (parsed.capture_end_hint or parsed.capture_start) >= since:
                yield path
        return

    raise ValueError(f"Unsupported camera kind: {camera.kind}")
