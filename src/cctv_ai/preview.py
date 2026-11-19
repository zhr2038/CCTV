from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .media import resolve_ffmpeg


@dataclass(frozen=True)
class PreviewResult:
    path: Path
    cached: bool
    source_start_seconds: float
    duration_seconds: float
    encoder: str


def _source_path(source_root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise ValueError(f"Unsafe media path: {relative_path}")
    parsed = PurePosixPath(relative_path)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or ".." in parsed.parts
        or any(":" in part for part in parsed.parts)
    ):
        raise ValueError(f"Unsafe media path: {relative_path}")
    return source_root.joinpath(*parsed.parts)


def _preview_name(event: Mapping[str, Any]) -> str:
    material = "|".join(
        (
            str(event["event_id"]),
            str(event["relative_path"]),
            str(event["mtime_ns"]),
            str(event["media_start_seconds"]),
            str(event["media_end_seconds"]),
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"event-{int(event['event_id'])}-{digest}.mp4"


def _run_transcode(
    *,
    ffmpeg: str,
    source: Path,
    output: Path,
    start_seconds: float,
    duration_seconds: float,
    encoder: str,
) -> tuple[bool, str]:
    if encoder == "h264_nvenc":
        video_options = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-cq",
            "25",
            "-b:v",
            "0",
        ]
    elif encoder == "libx264":
        video_options = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
        ]
    else:
        raise ValueError(f"Unsupported preview encoder: {encoder}")

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale=-2:720,setsar=1",
        *video_options,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    message = result.stderr.strip()
    valid = (
        result.returncode == 0
        and output.is_file()
        and output.stat().st_size > 1024
    )
    return valid, message


def generate_event_preview(
    event: Mapping[str, Any],
    *,
    source_root: str | Path,
    output_dir: str | Path,
    ffmpeg: str | None = None,
    before_seconds: float = 6.0,
    after_seconds: float = 10.0,
    encoder: str = "auto",
) -> PreviewResult:
    if before_seconds < 0 or after_seconds < 0:
        raise ValueError("Preview padding cannot be negative")
    source = _source_path(
        Path(source_root),
        str(event["relative_path"]),
    )
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / _preview_name(event)
    event_start = float(event["media_start_seconds"])
    event_end = float(event["media_end_seconds"])
    start = max(0.0, event_start - before_seconds)
    requested_end = event_end + after_seconds
    media_duration = event["duration_seconds"]
    if media_duration is not None:
        requested_end = min(requested_end, float(media_duration))
    duration = max(1.0, requested_end - start)

    if output.is_file() and output.stat().st_size > 1024:
        return PreviewResult(
            path=output,
            cached=True,
            source_start_seconds=start,
            duration_seconds=duration,
            encoder="cached",
        )

    executable = resolve_ffmpeg(ffmpeg)
    encoders = (
        ("h264_nvenc", "libx264")
        if encoder == "auto"
        else (encoder,)
    )
    errors: list[str] = []
    for selected_encoder in encoders:
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{selected_encoder}.tmp.mp4"
        )
        if temporary.exists():
            temporary.unlink()
        try:
            valid, message = _run_transcode(
                ffmpeg=executable,
                source=source,
                output=temporary,
                start_seconds=start,
                duration_seconds=duration,
                encoder=selected_encoder,
            )
            if valid:
                os.replace(temporary, output)
                return PreviewResult(
                    path=output,
                    cached=False,
                    source_start_seconds=start,
                    duration_seconds=duration,
                    encoder=selected_encoder,
                )
            errors.append(
                f"{selected_encoder}: "
                + (message[-3000:] or "ffmpeg produced no preview")
            )
        finally:
            if temporary.exists():
                temporary.unlink()

    raise RuntimeError("Preview generation failed\n" + "\n".join(errors))
