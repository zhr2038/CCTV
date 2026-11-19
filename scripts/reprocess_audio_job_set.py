from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from cctv_ai.audio_events import YamnetDetector
from cctv_ai.config import load_config
from cctv_ai.database import (
    connect_database,
    mark_job_error,
    mark_job_running,
    save_audio_events,
    save_probe,
    save_time_segments,
)
from cctv_ai.media import probe_media
from cctv_ai.watermark_ocr import run_watermark_ocr


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def source_path(root: Path, relative_path: str) -> Path:
    parsed = PurePosixPath(relative_path)
    if (
        "\\" in relative_path
        or parsed.is_absolute()
        or not parsed.parts
        or ".." in parsed.parts
        or any(":" in part for part in parsed.parts)
    ):
        raise ValueError(f"Unsafe media path: {relative_path}")
    return root.joinpath(*parsed.parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a strictly bounded set of prior audio jobs after an "
            "index recovery."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--job-created-before", type=int, required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--watermark-event-label", default="baby_cry")
    parser.add_argument("--watermark-hwaccel")
    parser.add_argument("--watermark-minimum-confidence", type=float, default=0.90)
    parser.add_argument("--audio-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--recovery-label", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if not config.source_read_only:
        raise ValueError("Recovery requires source_read_only=true")
    if config.cloud_upload:
        raise ValueError("Recovery requires cloud_upload=false")

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {root}")
    camera_ids = tuple(dict.fromkeys(args.camera))
    camera_map = {camera.id: camera for camera in config.cameras}
    unknown = sorted(set(camera_ids) - set(camera_map))
    if unknown:
        raise ValueError("Unknown cameras: " + ", ".join(unknown))

    connection = connect_database(args.db)
    placeholders = ",".join("?" for _ in camera_ids)
    rows = list(
        connection.execute(
            f"""
            SELECT m.*
            FROM processing_jobs j
            JOIN media_files m ON m.id = j.media_file_id
            WHERE j.job_type = 'audio_event_detection'
              AND j.created_at <= ?
              AND m.camera_id IN ({placeholders})
            ORDER BY m.camera_id, m.capture_start_ts
            """,
            (args.job_created_before, *camera_ids),
        )
    )
    if len(rows) != args.expected_files:
        connection.close()
        raise RuntimeError(
            f"Expected exactly {args.expected_files} recovery files, "
            f"found {len(rows)}"
        )
    emit({"phase": "selected", "files": len(rows)})

    failures: list[dict] = []
    for index, row in enumerate(rows, 1):
        relative_path = str(row["relative_path"])
        try:
            probe = probe_media(source_path(root, relative_path))
            camera = camera_map[str(row["camera_id"])]
            save_probe(
                connection,
                row,
                probe,
                audio_enabled=(
                    config.audio_enabled and camera.audio_enabled
                ),
                audio_event_labels=config.audio_event_labels,
            )
            connection.execute(
                """
                UPDATE processing_jobs
                SET status = 'queued', attempts = 0, updated_at = ?
                WHERE media_file_id = ?
                  AND job_type IN (
                      'audio_event_detection', 'watermark_time_map'
                  )
                """,
                (int(time.time()), int(row["id"])),
            )
            connection.commit()
            emit(
                {
                    "phase": "probe",
                    "index": index,
                    "files": len(rows),
                    "media_id": int(row["id"]),
                    "duration": probe.duration_seconds,
                    "path": relative_path,
                }
            )
        except Exception as exc:
            connection.rollback()
            failures.append(
                {
                    "phase": "probe",
                    "media_id": int(row["id"]),
                    "error": str(exc),
                }
            )
            emit({"phase": "probe_error", **failures[-1]})

    if failures:
        connection.close()
        emit({"phase": "failed", "failures": failures})
        return 2

    emit({"phase": "audio_model_loading"})
    detector = YamnetDetector(
        model_path=config.audio_model_path,
        class_map_path=config.audio_label_map_path,
        event_labels=config.audio_event_labels,
        minimum_confidence=config.audio_minimum_confidence,
        merge_gap_seconds=config.audio_merge_gap_seconds,
    )
    for index, row in enumerate(rows, 1):
        job = connection.execute(
            """
            SELECT id
            FROM processing_jobs
            WHERE media_file_id = ?
              AND job_type = 'audio_event_detection'
            """,
            (int(row["id"]),),
        ).fetchone()
        job_id = int(job["id"])
        if not mark_job_running(connection, job_id):
            failures.append(
                {
                    "phase": "audio",
                    "media_id": int(row["id"]),
                    "error": "Could not claim audio job",
                }
            )
            continue
        connection.commit()
        try:
            analysis = detector.analyze(
                source_path(root, str(row["relative_path"])),
                timeout_seconds=args.audio_timeout_seconds,
            )
            counts = Counter(event.label for event in analysis.events)
            save_audio_events(
                connection,
                job_id=job_id,
                media_file_id=int(row["id"]),
                events=analysis.events,
                model_name=analysis.model,
                metadata={
                    "windows_seen": analysis.windows_seen,
                    "raw_detections": analysis.raw_detections,
                    "model": analysis.model,
                    "model_sha256": analysis.model_sha256,
                    "minimum_confidence": analysis.minimum_confidence,
                    "merge_gap_seconds": analysis.merge_gap_seconds,
                    "labels": list(config.audio_event_labels),
                    "recovery": args.recovery_label,
                },
            )
            connection.commit()
            emit(
                {
                    "phase": "audio",
                    "index": index,
                    "files": len(rows),
                    "media_id": int(row["id"]),
                    "events": dict(sorted(counts.items())),
                    "windows": analysis.windows_seen,
                }
            )
        except Exception as exc:
            mark_job_error(connection, job_id, str(exc))
            connection.commit()
            failures.append(
                {
                    "phase": "audio",
                    "media_id": int(row["id"]),
                    "error": str(exc),
                }
            )
            emit({"phase": "audio_error", **failures[-1]})

    original_ids = tuple(int(row["id"]) for row in rows)
    media_placeholders = ",".join("?" for _ in original_ids)
    watermark_rows = list(
        connection.execute(
            f"""
            SELECT m.*, j.id AS job_id
            FROM media_files m
            JOIN processing_jobs j ON j.media_file_id = m.id
            WHERE m.id IN ({media_placeholders})
              AND m.requires_watermark_ocr = 1
              AND j.job_type = 'watermark_time_map'
              AND EXISTS (
                  SELECT 1
                  FROM audio_events a
                  WHERE a.media_file_id = m.id
                    AND a.event_label = ?
              )
            ORDER BY m.capture_start_ts
            """,
            (*original_ids, args.watermark_event_label),
        )
    )
    emit({"phase": "watermark_selected", "files": len(watermark_rows)})
    for index, row in enumerate(watermark_rows, 1):
        camera = camera_map[str(row["camera_id"])]
        if camera.watermark_crop is None:
            failures.append(
                {
                    "phase": "watermark",
                    "media_id": int(row["id"]),
                    "error": "Camera has no configured watermark crop",
                }
            )
            continue
        job_id = int(row["job_id"])
        connection.execute(
            """
            UPDATE processing_jobs
            SET status = 'queued', attempts = 0
            WHERE id = ?
            """,
            (job_id,),
        )
        if not mark_job_running(connection, job_id):
            failures.append(
                {
                    "phase": "watermark",
                    "media_id": int(row["id"]),
                    "error": "Could not claim watermark job",
                }
            )
            continue
        connection.commit()
        selected_hwaccel = args.watermark_hwaccel
        fallback_error = None
        interval = camera.watermark_sample_seconds
        try:
            try:
                run = run_watermark_ocr(
                    source_path(root, str(row["relative_path"])),
                    timezone=config.timezone,
                    crop=camera.watermark_crop,
                    interval_seconds=interval,
                    hwaccel=selected_hwaccel,
                    minimum_confidence=(
                        args.watermark_minimum_confidence
                    ),
                )
            except Exception as exc:
                if not selected_hwaccel:
                    raise
                fallback_error = str(exc)
                selected_hwaccel = None
                run = run_watermark_ocr(
                    source_path(root, str(row["relative_path"])),
                    timezone=config.timezone,
                    crop=camera.watermark_crop,
                    interval_seconds=interval,
                    hwaccel=None,
                    minimum_confidence=(
                        args.watermark_minimum_confidence
                    ),
                )
            save_time_segments(
                connection,
                job_id=job_id,
                media_file_id=int(row["id"]),
                segments=run.segments,
                source="watermark_ocr:rapidocr",
                metadata={
                    "frames_seen": run.frames_seen,
                    "valid_observations": len(run.observations),
                    "ocr_failures": run.ocr_failures,
                    "segments": len(run.segments),
                    "sample_interval_seconds": interval,
                    "minimum_confidence": (
                        args.watermark_minimum_confidence
                    ),
                    "hwaccel": selected_hwaccel,
                    "hwaccel_fallback_error": fallback_error,
                    "model": run.model,
                    "recovery": args.recovery_label,
                },
            )
            connection.commit()
            emit(
                {
                    "phase": "watermark",
                    "index": index,
                    "files": len(watermark_rows),
                    "media_id": int(row["id"]),
                    "frames": run.frames_seen,
                    "valid": len(run.observations),
                    "ocr_failures": run.ocr_failures,
                    "segments": len(run.segments),
                    "hwaccel": selected_hwaccel,
                }
            )
        except Exception as exc:
            mark_job_error(connection, job_id, str(exc))
            connection.commit()
            failures.append(
                {
                    "phase": "watermark",
                    "media_id": int(row["id"]),
                    "error": str(exc),
                }
            )
            emit({"phase": "watermark_error", **failures[-1]})

    event_counts = {
        str(row["event_label"]): int(row["events"])
        for row in connection.execute(
            f"""
            SELECT a.event_label, COUNT(*) AS events
            FROM audio_events a
            WHERE a.media_file_id IN ({media_placeholders})
            GROUP BY a.event_label
            """,
            original_ids,
        )
    }
    unmapped = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM audio_events a
            WHERE a.media_file_id IN ({media_placeholders})
              AND a.event_label = ?
              AND a.wall_start_ts IS NULL
            """,
            (*original_ids, args.watermark_event_label),
        ).fetchone()[0]
    )
    connection.close()
    emit(
        {
            "phase": "complete",
            "files": len(rows),
            "watermark_files": len(watermark_rows),
            "events": event_counts,
            "unmapped_target_events": unmapped,
            "failures": failures,
        }
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
