from __future__ import annotations

import subprocess
import queue
import re
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .watermark import (
    WatermarkObservation,
    build_time_segments,
    parse_watermark_text,
)
from .media import resolve_ffmpeg


@dataclass(frozen=True)
class WatermarkRun:
    observations: tuple[WatermarkObservation, ...]
    segments: tuple
    frames_seen: int
    ocr_failures: int
    low_confidence: int
    ffmpeg: str
    model: str


SHOWINFO_PTS_RE = re.compile(
    r"\bpts_time:(?P<seconds>[-+0-9.eE]+)"
)


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_watermark_observations(
    path: str | Path,
    *,
    timezone: str,
    crop: tuple[int, int, int, int],
    interval_seconds: float,
    ffmpeg: str | None = None,
    hwaccel: str | None = None,
    minimum_confidence: float = 0.90,
) -> Iterator[WatermarkObservation | None]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    try:
        import numpy as np
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Watermark OCR dependencies are missing; install with "
            "`pip install -e .[watermark-ocr]`"
        ) from exc

    x, y, width, height = crop
    ffmpeg_executable = resolve_ffmpeg(ffmpeg)
    command = [
        ffmpeg_executable,
        "-hide_banner",
    ]
    if hwaccel:
        command.extend(["-hwaccel", hwaccel])
    command.extend(
        [
            "-loglevel",
            "info",
            "-skip_frame",
            "nokey",
            "-i",
            str(path),
            "-vf",
            f"crop={width}:{height}:{x}:{y},format=gray,showinfo",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    engine = RapidOCR()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Unable to open ffmpeg pipes")

    frame_size = width * height
    pts_queue: queue.Queue[float | None] = queue.Queue()
    stderr_tail: deque[str] = deque(maxlen=80)

    def read_stderr() -> None:
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                stderr_tail.append(line)
            match = SHOWINFO_PTS_RE.search(line)
            if match:
                pts_queue.put(float(match.group("seconds")))
        pts_queue.put(None)

    stderr_thread = threading.Thread(
        target=read_stderr,
        name="watermark-ffmpeg-stderr",
        daemon=True,
    )
    stderr_thread.start()

    next_sample_seconds = 0.0
    last_selected_seconds: float | None = None
    last_decoded: tuple[float, bytes] | None = None

    def recognize(
        frame_bytes: bytes,
        media_seconds: float,
    ) -> WatermarkObservation | None:
        image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            height,
            width,
        )
        result = engine(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
        if not result.txts or not result.scores:
            return None
        text = result.txts[0]
        confidence = float(result.scores[0])
        if confidence < minimum_confidence:
            return None
        try:
            wall_time = parse_watermark_text(text, timezone)
        except ValueError:
            return None
        return WatermarkObservation(
            media_seconds=media_seconds,
            wall_time=wall_time,
            confidence=confidence,
        )

    try:
        while True:
            frame_bytes = _read_exact(process.stdout, frame_size)
            if not frame_bytes:
                break
            if len(frame_bytes) != frame_size:
                raise RuntimeError(
                    f"Partial raw frame: {len(frame_bytes)} of {frame_size}"
                )
            try:
                media_seconds = pts_queue.get(timeout=30)
            except queue.Empty as exc:
                raise RuntimeError(
                    "Timed out waiting for an ffmpeg keyframe timestamp"
                ) from exc
            if media_seconds is None:
                raise RuntimeError(
                    "ffmpeg ended before emitting a timestamp for every frame"
                )
            last_decoded = (media_seconds, frame_bytes)
            if media_seconds + 0.001 < next_sample_seconds:
                continue
            yield recognize(frame_bytes, media_seconds)
            last_selected_seconds = media_seconds
            while next_sample_seconds <= media_seconds + 0.001:
                next_sample_seconds += interval_seconds

        if (
            last_decoded is not None
            and (
                last_selected_seconds is None
                or last_decoded[0] - last_selected_seconds
                > interval_seconds / 2
            )
        ):
            yield recognize(last_decoded[1], last_decoded[0])

        return_code = process.wait()
        stderr_thread.join(timeout=5)
        if return_code != 0:
            raise RuntimeError(
                "\n".join(stderr_tail)
                or f"ffmpeg watermark decode exited with {return_code}"
            )
    except Exception:
        process.kill()
        process.wait()
        stderr_thread.join(timeout=5)
        raise


def run_watermark_ocr(
    path: str | Path,
    *,
    timezone: str,
    crop: tuple[int, int, int, int],
    interval_seconds: float,
    ffmpeg: str | None = None,
    hwaccel: str | None = None,
    minimum_confidence: float = 0.90,
) -> WatermarkRun:
    observations: list[WatermarkObservation] = []
    frames_seen = 0
    failures = 0
    for observation in iter_watermark_observations(
        path,
        timezone=timezone,
        crop=crop,
        interval_seconds=interval_seconds,
        ffmpeg=ffmpeg,
        hwaccel=hwaccel,
        minimum_confidence=minimum_confidence,
    ):
        frames_seen += 1
        if observation is None:
            failures += 1
        else:
            observations.append(observation)
    if not observations:
        raise RuntimeError("No valid timestamp watermarks were recognized")
    segments = build_time_segments(
        observations,
        jump_threshold_seconds=max(30.0, interval_seconds * 3),
    )
    return WatermarkRun(
        observations=tuple(observations),
        segments=tuple(segments),
        frames_seen=frames_seen,
        ocr_failures=failures,
        low_confidence=0,
        ffmpeg=resolve_ffmpeg(ffmpeg),
        model="RapidOCR recognition model",
    )
