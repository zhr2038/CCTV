from __future__ import annotations

import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import CameraConfig, ProjectConfig


VALID_CAMERA_KINDS = {
    "aqara_event",
    "xiaomi_minute_event",
    "xiaomi_watermark_bundle",
}


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project = raw["project"]
    privacy = raw["privacy"]
    audio = raw["audio"]
    family_routine = raw.get("family_routine", {})
    cameras = tuple(
        CameraConfig(
            id=item["id"],
            name=item["name"],
            location=item["location"],
            kind=item["kind"],
            relative_root=item["relative_root"].strip("/"),
            audio_enabled=bool(item.get("audio_enabled", True)),
            watermark_crop=(
                tuple(int(value) for value in item["watermark_crop"])
                if "watermark_crop" in item
                else None
            ),
            watermark_sample_seconds=float(
                item.get("watermark_sample_seconds", 5.0)
            ),
        )
        for item in raw["cameras"]
    )

    ids = [camera.id for camera in cameras]
    roots = [camera.relative_root for camera in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("Camera ids must be unique")
    if len(roots) != len(set(roots)):
        raise ValueError("Camera relative roots must be unique")
    invalid = sorted({camera.kind for camera in cameras} - VALID_CAMERA_KINDS)
    if invalid:
        raise ValueError(f"Unsupported camera kinds: {', '.join(invalid)}")
    for camera in cameras:
        if camera.watermark_crop is not None:
            if len(camera.watermark_crop) != 4:
                raise ValueError(
                    f"Camera {camera.id} watermark_crop must have four values"
                )
            if any(value < 0 for value in camera.watermark_crop):
                raise ValueError(
                    f"Camera {camera.id} watermark_crop cannot be negative"
                )
        if (
            camera.kind == "xiaomi_watermark_bundle"
            and camera.watermark_crop is None
        ):
            raise ValueError(
                f"Camera {camera.id} requires a watermark_crop"
            )
        if camera.watermark_sample_seconds <= 0:
            raise ValueError(
                f"Camera {camera.id} watermark_sample_seconds must be positive"
            )

    timezone = project["timezone"]
    ZoneInfo(timezone)

    transcription_mode = audio["transcription_mode"]
    if transcription_mode not in {"disabled", "on_demand", "continuous"}:
        raise ValueError(f"Unsupported transcription mode: {transcription_mode}")
    if (
        transcription_mode == "continuous"
        and not audio.get("store_continuous_transcripts", False)
    ):
        raise ValueError(
            "Continuous transcription requires an explicit transcript storage decision"
        )
    audio_minimum_confidence = float(
        audio.get("minimum_confidence", 0.60)
    )
    if not 0 < audio_minimum_confidence <= 1:
        raise ValueError("Audio minimum_confidence must be in (0, 1]")
    audio_merge_gap_seconds = float(
        audio.get("merge_gap_seconds", 1.0)
    )
    if audio_merge_gap_seconds < 0:
        raise ValueError("Audio merge_gap_seconds cannot be negative")

    feeding_interval_minutes = int(
        family_routine.get("feeding_interval_minutes", 180)
    )
    feeding_episode_merge_minutes = int(
        family_routine.get("feeding_episode_merge_minutes", 75)
    )
    if feeding_interval_minutes <= 0:
        raise ValueError("feeding_interval_minutes must be positive")
    if not 0 < feeding_episode_merge_minutes < feeding_interval_minutes / 2:
        raise ValueError(
            "feeding_episode_merge_minutes must be positive and less than "
            "half of feeding_interval_minutes"
        )

    return ProjectConfig(
        timezone=timezone,
        source_root_windows=project["source_root_windows"],
        source_root_linux=project["source_root_linux"],
        stability_age_minutes=int(project["stability_age_minutes"]),
        stable_observations=int(project["stable_observations"]),
        source_read_only=bool(privacy["source_read_only"]),
        face_recognition=bool(privacy["face_recognition"]),
        cloud_upload=bool(privacy["cloud_upload"]),
        audio_enabled=bool(audio["enabled"]),
        transcription_mode=transcription_mode,
        store_continuous_transcripts=bool(
            audio.get("store_continuous_transcripts", False)
        ),
        audio_event_labels=tuple(audio.get("event_labels", ())),
        audio_model_path=str(
            audio.get("model_path", ".data/models/yamnet.tflite")
        ),
        audio_label_map_path=str(
            audio.get(
                "label_map_path",
                ".data/models/yamnet_class_map.csv",
            )
        ),
        audio_minimum_confidence=audio_minimum_confidence,
        audio_merge_gap_seconds=audio_merge_gap_seconds,
        cameras=cameras,
        feeding_interval_minutes=feeding_interval_minutes,
        feeding_episode_merge_minutes=feeding_episode_merge_minutes,
    )
