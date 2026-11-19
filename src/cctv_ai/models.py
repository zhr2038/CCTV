from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    location: str
    kind: str
    relative_root: str
    audio_enabled: bool
    watermark_crop: tuple[int, int, int, int] | None = None
    watermark_sample_seconds: float = 5.0


@dataclass(frozen=True)
class ProjectConfig:
    timezone: str
    source_root_windows: str
    source_root_linux: str
    stability_age_minutes: int
    stable_observations: int
    source_read_only: bool
    face_recognition: bool
    cloud_upload: bool
    audio_enabled: bool
    transcription_mode: str
    store_continuous_transcripts: bool
    audio_event_labels: tuple[str, ...]
    audio_model_path: str
    audio_label_map_path: str
    audio_minimum_confidence: float
    audio_merge_gap_seconds: float
    cameras: tuple[CameraConfig, ...]
    feeding_interval_minutes: int = 180
    feeding_episode_merge_minutes: int = 75

    def source_root(self, override: str | None = None) -> Path:
        if override:
            return Path(override)
        import os

        return Path(
            self.source_root_windows if os.name == "nt" else self.source_root_linux
        )


@dataclass(frozen=True)
class ParsedCapture:
    capture_start: datetime
    capture_end_hint: datetime | None
    time_source: str
    requires_watermark_ocr: bool


@dataclass(frozen=True)
class MediaProbe:
    duration_seconds: float | None
    format_name: str | None
    size_bytes: int | None
    bit_rate: int | None
    video_codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
