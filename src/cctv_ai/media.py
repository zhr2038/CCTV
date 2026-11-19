from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import MediaProbe


def _user_path_executable(command: str) -> str | None:
    if os.name != "nt" or Path(command).name != command:
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
        ) as key:
            user_path, _ = winreg.QueryValueEx(key, "Path")
    except (OSError, ImportError):
        return None
    suffixes = ("", ".exe") if not command.lower().endswith(".exe") else ("",)
    for raw_directory in str(user_path).split(os.pathsep):
        directory = Path(os.path.expandvars(raw_directory.strip().strip('"')))
        for suffix in suffixes:
            candidate = directory / f"{command}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


@lru_cache(maxsize=8)
def resolve_ffprobe(explicit: str | None = None) -> str:
    command = explicit or "ffprobe"
    path = Path(command)
    if path.is_file():
        return str(path)
    discovered = shutil.which(command) or _user_path_executable(command)
    if discovered:
        return discovered
    raise RuntimeError(
        "ffprobe was not found; install FFmpeg or pass --ffprobe"
    )


def resolve_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
        discovered = shutil.which(explicit) or _user_path_executable(explicit)
        if discovered:
            return discovered
        raise RuntimeError(
            "ffmpeg was not found at the requested path"
        )
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    user_ffmpeg = _user_path_executable("ffmpeg")
    if user_ffmpeg:
        return user_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg was not found; install an OCR/audio extra or pass --ffmpeg"
        ) from exc
    return get_ffmpeg_exe()


def _integer(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(value)


def _float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    return float(value)


def _frame_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def parse_ffprobe(payload: dict[str, Any]) -> MediaProbe:
    streams = payload.get("streams", [])
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        {},
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        {},
    )
    format_data = payload.get("format", {})
    return MediaProbe(
        duration_seconds=_float(format_data.get("duration")),
        format_name=format_data.get("format_name"),
        size_bytes=_integer(format_data.get("size")),
        bit_rate=_integer(format_data.get("bit_rate")),
        video_codec=video.get("codec_name"),
        width=_integer(video.get("width")),
        height=_integer(video.get("height")),
        fps=_frame_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        audio_codec=audio.get("codec_name"),
        audio_sample_rate=_integer(audio.get("sample_rate")),
        audio_channels=_integer(audio.get("channels")),
    )


def probe_media(
    path: str | Path,
    ffprobe: str | None = None,
) -> MediaProbe:
    executable = resolve_ffprobe(ffprobe)
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,size,bit_rate,format_name:"
            "stream=codec_type,codec_name,width,height,avg_frame_rate,"
            "r_frame_rate,sample_rate,channels"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or f"ffprobe exited with {result.returncode}"
        raise RuntimeError(message)
    return parse_ffprobe(json.loads(result.stdout))
