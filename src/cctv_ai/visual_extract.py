from __future__ import annotations

import hashlib
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .media import resolve_ffmpeg


EXTRACTOR_VERSION = "adaptive-keyframes-v1"


@dataclass(frozen=True)
class ExtractedKeyframe:
    media_offset_seconds: float
    image_path: Path
    frame_sha256: str
    width: int
    height: int
    brightness: float
    sharpness: float
    motion_score: float | None
    night_score: float
    sample_reason: str


def _safe_source_path(source_root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("Unsafe media path")
    candidate = source_root.joinpath(*parts)
    return candidate


def _output_dimensions(
    source_width: int | None,
    source_height: int | None,
    target_width: int,
) -> tuple[int, int]:
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    if not source_width or not source_height:
        return target_width, max(2, round(target_width * 9 / 16 / 2) * 2)
    height = max(
        2,
        round(target_width * source_height / source_width / 2) * 2,
    )
    return target_width, height


def sampling_plan(
    media: Mapping[str, Any],
    *,
    bedside_interval_seconds: float = 30.0,
    kitchen_interval_seconds: float = 20.0,
) -> tuple[str, float]:
    """Return a sampling mode and interval/offset for one media file.

    Motion-triggered Aqara clips are already a temporal pre-filter, so one
    representative frame per clip is enough for the first pass. Kitchen clips
    are roughly one minute and receive three samples. Continuous bedside
    bundles are sampled every 30 seconds by default.
    """
    camera_id = str(media["camera_id"])
    duration = float(media["duration_seconds"] or 0)
    if camera_id == "B88880620430":
        return "interval", bedside_interval_seconds
    if camera_id == "50ec503b63bf":
        return "interval", kitchen_interval_seconds
    if duration > 0:
        return "single", min(max(duration * 0.5, 0.5), 3.0)
    return "single", 1.0


def _ffmpeg_filter(
    *,
    width: int,
    height: int,
    interval_seconds: float | None,
    hardware: bool,
) -> str:
    filters: list[str] = []
    if interval_seconds is not None:
        filters.append(f"fps=1/{interval_seconds:g}")
    if hardware:
        filters.extend(
            [
                f"scale_cuda={width}:{height}",
                "hwdownload",
                "format=nv12",
                "format=bgr24",
            ]
        )
    else:
        filters.extend([f"scale={width}:{height}", "format=bgr24"])
    return ",".join(filters)


def _decode_raw_frames(
    path: Path,
    *,
    source_width: int | None,
    source_height: int | None,
    target_width: int,
    mode: str,
    interval_or_offset: float,
    ffmpeg: str | None,
    hwaccel: str | None,
    timeout_seconds: float,
) -> tuple[list[tuple[float, bytes]], int, int, str | None]:
    executable = resolve_ffmpeg(ffmpeg)
    width, height = _output_dimensions(
        source_width,
        source_height,
        target_width,
    )

    def command(hardware: bool) -> list[str]:
        args = [executable, "-hide_banner", "-loglevel", "error"]
        if mode == "single":
            args.extend(["-ss", f"{interval_or_offset:.3f}"])
        if hardware:
            args.extend(
                ["-hwaccel", hwaccel or "cuda", "-hwaccel_output_format", "cuda"]
            )
        if mode == "interval":
            # CCTV encoders place IDR frames frequently. Decoding keyframes only
            # is orders of magnitude faster and remains sufficient for a diary
            # index sampled in tens of seconds.
            args.extend(["-skip_frame", "nokey"])
        args.extend(["-i", str(path), "-an"])
        interval = interval_or_offset if mode == "interval" else None
        args.extend(
            [
                "-vf",
                _ffmpeg_filter(
                    width=width,
                    height=height,
                    interval_seconds=interval,
                    hardware=hardware,
                ),
                "-fps_mode",
                "passthrough",
            ]
        )
        if mode == "single":
            args.extend(["-frames:v", "1"])
        args.extend(["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"])
        return args

    fallback_error: str | None = None
    raw: bytes
    for hardware in ([True, False] if hwaccel else [False]):
        try:
            completed = subprocess.run(
                command(hardware),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Frame extraction timed out after {timeout_seconds:g}s"
            ) from exc
        if completed.returncode == 0 and completed.stdout:
            raw = completed.stdout
            break
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        if hardware:
            fallback_error = error[-1000:] or (
                f"Hardware ffmpeg exited {completed.returncode}"
            )
            continue
        raise RuntimeError(
            error[-2000:] or f"ffmpeg exited {completed.returncode}"
        )
    else:  # pragma: no cover - defensive, loop always returns or raises
        raise RuntimeError("Unable to extract frames")

    frame_size = width * height * 3
    if len(raw) < frame_size:
        raise RuntimeError(
            f"Decoded output is too short: {len(raw)} of {frame_size} bytes"
        )
    count = len(raw) // frame_size
    frames: list[tuple[float, bytes]] = []
    for index in range(count):
        start = index * frame_size
        if mode == "single":
            offset = interval_or_offset
        else:
            offset = index * interval_or_offset
        frames.append((offset, raw[start : start + frame_size]))
    return frames, width, height, fallback_error


def _frame_metrics(image) -> tuple[float, float, float]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    night_score = float(np.mean(gray < 52))
    return brightness, sharpness, night_score


def _write_jpeg(path: Path, image, *, quality: int) -> tuple[str, int]:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode a keyframe")
    data = encoded.tobytes()
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return digest, len(data)


def extract_media_keyframes(
    media: Mapping[str, Any],
    *,
    source_root: str | Path,
    output_root: str | Path,
    ffmpeg: str | None = None,
    hwaccel: str | None = "cuda",
    target_width: int = 640,
    jpeg_quality: int = 82,
    bedside_interval_seconds: float = 30.0,
    kitchen_interval_seconds: float = 20.0,
    timeout_seconds: float = 360.0,
    extractor_version: str = EXTRACTOR_VERSION,
) -> tuple[list[ExtractedKeyframe], dict[str, Any]]:
    import cv2
    import numpy as np

    started = time.monotonic()
    media_id = int(media["id"])
    source = _safe_source_path(
        Path(source_root),
        str(media["relative_path"]),
    )
    if not source.is_file():
        raise FileNotFoundError(str(source))
    mode, interval_or_offset = sampling_plan(
        media,
        bedside_interval_seconds=bedside_interval_seconds,
        kitchen_interval_seconds=kitchen_interval_seconds,
    )
    requested_interval_or_offset = interval_or_offset
    sparse_video_fallback = False
    try:
        raw_frames, width, height, fallback_error = _decode_raw_frames(
            source,
            source_width=media["width"],
            source_height=media["height"],
            target_width=target_width,
            mode=mode,
            interval_or_offset=interval_or_offset,
            ffmpeg=ffmpeg,
            hwaccel=hwaccel,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError:
        # Some motion-triggered clips contain only one or two frames over tens
        # of seconds. Seeking to the nominal representative offset can then
        # succeed with no decoded frame. Preserve the file in the index by
        # retrying from its first frame; genuinely unreadable files still fail.
        if mode != "single" or interval_or_offset <= 0:
            raise
        interval_or_offset = 0.0
        sparse_video_fallback = True
        raw_frames, width, height, fallback_error = _decode_raw_frames(
            source,
            source_width=media["width"],
            source_height=media["height"],
            target_width=target_width,
            mode=mode,
            interval_or_offset=interval_or_offset,
            ffmpeg=ffmpeg,
            hwaccel=hwaccel,
            timeout_seconds=timeout_seconds,
        )
    relative_dir = Path(
        str(media["camera_id"]),
        f"{int(media['capture_start_ts']):010d}",
    )
    output_dir = Path(output_root).joinpath(relative_dir)
    keyframes: list[ExtractedKeyframe] = []
    previous_gray = None
    written_bytes = 0
    reason = (
        f"continuous_{interval_or_offset:g}s"
        if mode == "interval"
        else "event_representative"
    )
    for offset, frame_bytes in raw_frames:
        image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            height,
            width,
            3,
        )
        brightness, sharpness, night_score = _frame_metrics(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        motion_score = (
            float(cv2.absdiff(gray, previous_gray).mean() / 255.0)
            if previous_gray is not None
            else None
        )
        previous_gray = gray
        offset_ms = max(0, round(offset * 1000))
        filename = f"{media_id}-{offset_ms:010d}-{extractor_version}.jpg"
        destination = output_dir.joinpath(filename)
        digest, byte_count = _write_jpeg(
            destination,
            image,
            quality=jpeg_quality,
        )
        written_bytes += byte_count
        keyframes.append(
            ExtractedKeyframe(
                media_offset_seconds=offset,
                image_path=destination,
                frame_sha256=digest,
                width=width,
                height=height,
                brightness=brightness,
                sharpness=sharpness,
                motion_score=motion_score,
                night_score=night_score,
                sample_reason=reason,
            )
        )
    return keyframes, {
        "frames": len(keyframes),
        "mode": mode,
        "interval_or_offset_seconds": interval_or_offset,
        "requested_interval_or_offset_seconds": (
            requested_interval_or_offset
        ),
        "sparse_video_fallback": sparse_video_fallback,
        "target_width": width,
        "target_height": height,
        "written_bytes": written_bytes,
        "hwaccel": hwaccel,
        "hwaccel_fallback_error": fallback_error,
        "extractor_version": extractor_version,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def remove_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
