from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load_config
from .database import (
    backup_database,
    build_report,
    connect_database,
    get_queued_jobs,
    get_ready_files,
    import_analysis_database,
    mark_job_error,
    mark_job_running,
    query_audio_events,
    save_audio_events,
    save_probe,
    save_probe_error,
    save_time_segments,
)
from .media import probe_media
from .scanner import scan_archive
from .watermark import sample_offsets


DEFAULT_CONFIG = "config/cameras.toml"
DEFAULT_DATABASE = ".data/cctv_index.sqlite3"


def _parse_since(value: str | None, timezone: str) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
    return parsed


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--db", default=DEFAULT_DATABASE)
    parser.add_argument("--root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cctv-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan source files read-only")
    _common_config(scan)
    scan.add_argument("--since", help="ISO date/time, default timezone is project TZ")
    scan.add_argument(
        "--camera",
        action="append",
        help="Limit scanning to one or more camera ids",
    )

    probe = subparsers.add_parser("probe", help="Probe stable media with ffprobe")
    _common_config(probe)
    probe.add_argument("--limit", type=int, default=100)
    probe.add_argument("--camera")
    probe.add_argument("--ffprobe", default="ffprobe")

    report = subparsers.add_parser("report", help="Show manifest and job counts")
    report.add_argument("--db", default=DEFAULT_DATABASE)
    report.add_argument("--json", action="store_true")

    database_backup = subparsers.add_parser(
        "db-backup",
        help="Create an integrity-checked local SQLite backup",
    )
    database_backup.add_argument("--db", default=DEFAULT_DATABASE)
    database_backup.add_argument("--backup-dir")
    database_backup.add_argument("--label", default="manual")

    watermark = subparsers.add_parser(
        "watermark-plan", help="Print coarse OCR sample offsets"
    )
    watermark.add_argument("duration", type=float)
    watermark.add_argument("--interval", type=float, default=10.0)

    watermark_run = subparsers.add_parser(
        "watermark-run",
        help="Build B888 media-offset to wall-time mappings",
    )
    _common_config(watermark_run)
    watermark_run.add_argument("--limit", type=int, default=1)
    watermark_run.add_argument("--camera", default="B88880620430")
    watermark_run.add_argument("--media-id", type=int)
    watermark_run.add_argument(
        "--event-label",
        help="Only map files containing this detected audio event",
    )
    watermark_run.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rebuild an existing mapping, for refinement or decoder changes",
    )
    watermark_run.add_argument("--interval", type=float)
    watermark_run.add_argument("--ffmpeg")
    watermark_run.add_argument(
        "--hwaccel",
        help="Optional ffmpeg hardware decoder, for example cuda or qsv",
    )
    watermark_run.add_argument("--minimum-confidence", type=float, default=0.90)

    audio_install = subparsers.add_parser(
        "audio-model-install",
        help="Install the pinned local YAMNet model and class map",
    )
    audio_install.add_argument(
        "--model-dir",
        default=".data/models",
    )

    audio_run = subparsers.add_parser(
        "audio-run",
        help="Detect configured audio events locally",
    )
    _common_config(audio_run)
    audio_run.add_argument("--limit", type=int, default=4)
    audio_run.add_argument("--camera")
    audio_run.add_argument("--media-id", type=int)
    audio_run.add_argument("--ffmpeg")
    audio_run.add_argument("--model")
    audio_run.add_argument("--labels")
    audio_run.add_argument("--minimum-confidence", type=float)
    audio_run.add_argument("--merge-gap", type=float)
    audio_run.add_argument("--timeout-seconds", type=float, default=180.0)

    shadow_calibrate = subparsers.add_parser(
        "shadow-calibrate",
        help=(
            "Build cached event fingerprints and score review candidates "
            "without writing verdicts"
        ),
    )
    _common_config(shadow_calibrate)
    shadow_calibrate.add_argument("--event-label", default="baby_cry")
    shadow_calibrate.add_argument("--minimum-confidence", type=float)
    shadow_calibrate.add_argument("--gap-seconds", type=float, default=30.0)
    shadow_calibrate.add_argument("--workers", type=int, default=16)
    shadow_calibrate.add_argument(
        "--padding-seconds",
        type=float,
        default=0.5,
    )
    shadow_calibrate.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
    )
    shadow_calibrate.add_argument("--ffmpeg")
    shadow_calibrate.add_argument("--model")
    shadow_calibrate.add_argument("--labels")
    shadow_calibrate.add_argument(
        "--output",
        default=".data/shadow/baby_cry-latest.json",
    )

    events = subparsers.add_parser(
        "events",
        help="Query detected audio events and their source offsets",
    )
    events.add_argument("--config", default=DEFAULT_CONFIG)
    events.add_argument("--db", default=DEFAULT_DATABASE)
    events.add_argument("--camera")
    events.add_argument("--label")
    events.add_argument("--since")
    events.add_argument("--until")
    events.add_argument("--minimum-confidence", type=float, default=0.0)
    events.add_argument("--limit", type=int, default=100)
    events.add_argument("--json", action="store_true")

    merge_analysis = subparsers.add_parser(
        "merge-analysis",
        help="Merge probed media and derived results into a full index",
    )
    merge_analysis.add_argument("--db", required=True)
    merge_analysis.add_argument("--from-db", required=True)

    serve = subparsers.add_parser(
        "serve",
        help="Run the local family event review web app",
    )
    _common_config(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--allowed-network",
        action="append",
        default=[],
        help="CIDR allowed to access the review UI; repeat as needed",
    )
    serve.add_argument("--preview-dir", default=".data/review_clips")
    serve.add_argument("--ffmpeg")
    serve.add_argument(
        "--models-config",
        default="config/models.toml",
    )

    pipeline_run = subparsers.add_parser(
        "pipeline-run",
        help="Run one bounded scan/probe/audio/watermark pipeline cycle",
    )
    _pipeline_arguments(pipeline_run)

    pipeline_worker = subparsers.add_parser(
        "pipeline-worker",
        help="Continuously run bounded pipeline cycles",
    )
    _pipeline_arguments(pipeline_worker)
    pipeline_worker.add_argument(
        "--interval-seconds",
        type=int,
        default=30 * 60,
    )
    pipeline_worker.add_argument(
        "--scan-every-cycles",
        type=int,
        default=2,
    )
    pipeline_worker.add_argument("--max-cycles", type=int)
    pipeline_worker.add_argument(
        "--worker-lock",
        default=".data/cctv-pipeline-worker.lock",
    )

    visual_run = subparsers.add_parser(
        "visual-run",
        help="Run one bounded local GPU visual-analysis cycle",
    )
    _visual_arguments(visual_run)

    visual_worker = subparsers.add_parser(
        "visual-worker",
        help="Resume visual-analysis cycles until the selected scope is complete",
    )
    _visual_arguments(visual_worker)
    visual_worker.add_argument("--interval-seconds", type=int, default=1)
    visual_worker.add_argument("--max-cycles", type=int)

    visual_progress = subparsers.add_parser(
        "visual-progress",
        help="Show keyframe, detection, embedding and activity progress",
    )
    _common_config(visual_progress)
    visual_progress.add_argument("--since")
    visual_progress.add_argument("--until")
    visual_progress.add_argument("--camera", action="append", default=[])

    visual_verify = subparsers.add_parser(
        "visual-verify",
        help="Run the local VLM over feeding and toy candidate frames",
    )
    _common_config(visual_verify)
    visual_verify.add_argument("--since", required=True)
    visual_verify.add_argument("--until", required=True)
    visual_verify.add_argument("--camera", action="append", default=[])
    visual_verify.add_argument(
        "--embedding-model",
        default="google/siglip2-base-patch16-224",
    )
    visual_verify.add_argument(
        "--image-root",
        default=".data/visual/keyframes",
    )
    visual_verify.add_argument(
        "--media-root",
        help=(
            "Original recording root used for native-resolution temporal "
            "verification. Positive events are not accepted from the "
            "640x360 keyframe cache alone."
        ),
    )
    visual_verify.add_argument(
        "--model-dir",
        help=(
            "Optional local checkpoint override. Provider and model family "
            "are otherwise read from --models-config."
        ),
    )
    visual_verify.add_argument(
        "--models-config",
        default="config/models.toml",
    )
    visual_verify.add_argument(
        "--task",
        action="append",
        choices=("feeding", "toy", "sleep_presence", "posture"),
        default=[],
    )
    visual_verify.add_argument("--batch-size", type=int, default=8)
    visual_verify.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Optional legacy single-frame checks per task. The default is "
            "zero because event-level native verification is authoritative."
        ),
    )
    visual_verify.add_argument(
        "--temporal-max-frames",
        type=int,
        default=256,
        help=(
            "Maximum broad feeding candidates to verify from neighboring "
            "frames in this resumable run"
        ),
    )
    visual_verify.add_argument(
        "--feeding-bucket-seconds",
        type=int,
        default=5 * 60,
        help=(
            "Time-deduplication bucket for broad feeding recall. Use 120 "
            "for a targeted dense audit after the normal five-minute pass."
        ),
    )
    visual_verify.add_argument(
        "--feeding-minimum-rank",
        type=float,
        default=0.25,
        help="Minimum cheap-model feeding recall rank before native review.",
    )
    visual_verify.add_argument(
        "--feeding-date",
        action="append",
        default=[],
        help=(
            "Optional local YYYY-MM-DD date to include in feeding recall; "
            "repeat to run a dense audit only on deficient days."
        ),
    )
    visual_verify.add_argument("--force", action="store_true")

    visual_embed_qwen3 = subparsers.add_parser(
        "visual-embed-qwen3",
        help=(
            "Backfill the independent Qwen3-VL-Embedding search index "
            "without changing activity observations"
        ),
    )
    visual_embed_qwen3.add_argument(
        "--db",
        default=DEFAULT_DATABASE,
    )
    visual_embed_qwen3.add_argument(
        "--models-config",
        default="config/models.toml",
    )
    visual_embed_qwen3.add_argument(
        "--image-root",
        default=".data/visual/keyframes",
    )
    visual_embed_qwen3.add_argument(
        "--keyframe-id",
        type=int,
        action="append",
        default=[],
    )
    visual_embed_qwen3.add_argument("--limit", type=int)
    visual_embed_qwen3.add_argument("--batch-size", type=int)

    visual_milestones = subparsers.add_parser(
        "visual-milestones",
        help=(
            "Escalate first sitting/crawling/standing candidates from "
            "Qwen3-VL-8B to the configured 32B model"
        ),
    )
    visual_milestones.add_argument(
        "--db",
        default=DEFAULT_DATABASE,
    )
    visual_milestones.add_argument(
        "--models-config",
        default="config/models.toml",
    )
    visual_milestones.add_argument(
        "--image-root",
        default=".data/visual/keyframes",
    )
    visual_milestones.add_argument("--since", required=True)
    visual_milestones.add_argument("--until", required=True)
    visual_milestones.add_argument(
        "--timezone",
        default="Asia/Shanghai",
    )
    visual_milestones.add_argument("--limit", type=int, default=12)

    visual_rebuild = subparsers.add_parser(
        "visual-rebuild",
        help="Rebuild activity events from completed visual observations",
    )
    _common_config(visual_rebuild)
    visual_rebuild.add_argument("--since", required=True)
    visual_rebuild.add_argument("--until", required=True)
    visual_rebuild.add_argument("--camera", action="append", default=[])
    visual_rebuild.add_argument(
        "--embedding-model",
        default="google/siglip2-base-patch16-224",
    )

    return parser


def _pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    _common_config(parser)
    parser.add_argument(
        "--since",
        required=True,
        help="Inclusive ISO start time for the backfill scope",
    )
    parser.add_argument("--until", help="Exclusive ISO end time")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        help="Camera id; repeat for multiple cameras",
    )
    parser.add_argument(
        "--no-scan",
        dest="scan",
        action="store_false",
        help="Skip the archive scan for this cycle",
    )
    parser.set_defaults(scan=True)
    parser.add_argument(
        "--probe-limit-per-camera",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=1,
        help="Number of media probes to run concurrently",
    )
    parser.add_argument(
        "--audio-limit-per-camera",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=1,
        help="Number of local audio analyses to run concurrently",
    )
    parser.add_argument(
        "--watermark-limit-per-camera",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--watermark-workers",
        type=int,
        default=1,
        help="Number of watermark OCR analyses to run concurrently",
    )
    parser.add_argument(
        "--watermark-event-label",
        default="baby_cry",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--audio-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--watermark-hwaccel")
    parser.add_argument("--watermark-interval", type=float)
    parser.add_argument(
        "--watermark-minimum-confidence",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--stale-job-seconds",
        type=int,
        default=2 * 60 * 60,
    )
    parser.add_argument(
        "--error-retry-delay-seconds",
        type=int,
        default=6 * 60 * 60,
    )
    parser.add_argument(
        "--lock",
        default=".data/cctv-pipeline.lock",
    )
    parser.add_argument("--mode", default="backfill")


def _visual_arguments(parser: argparse.ArgumentParser) -> None:
    _common_config(parser)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument(
        "--output-root",
        default=".data/visual/keyframes",
    )
    parser.add_argument("--ffprobe")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--hwaccel", default="cuda")
    parser.add_argument(
        "--detector-model",
        default=".data/models/yolo26s.pt",
    )
    parser.add_argument(
        "--embedding-model",
        default="google/siglip2-base-patch16-224",
    )
    parser.add_argument(
        "--model-cache",
        default=".data/models/huggingface",
    )
    parser.add_argument("--probe-limit", type=int, default=2048)
    parser.add_argument("--probe-workers", type=int, default=24)
    parser.add_argument("--extract-limit", type=int, default=2048)
    parser.add_argument("--extract-workers", type=int, default=16)
    parser.add_argument("--watermark-limit", type=int, default=256)
    parser.add_argument("--watermark-workers", type=int, default=4)
    parser.add_argument(
        "--watermark-interval-seconds",
        type=float,
        default=120.0,
    )
    parser.add_argument("--inference-file-limit", type=int, default=2048)
    parser.add_argument("--inference-frame-limit", type=int, default=8192)
    parser.add_argument("--detector-batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=96)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument(
        "--bedside-interval-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--kitchen-interval-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--error-retry-delay-seconds",
        type=int,
        default=60 * 60,
    )
    parser.add_argument(
        "--lock",
        default=".data/cctv-visual-pipeline.lock",
    )


def _scan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source_root = config.source_root(args.root)
    summary = scan_archive(
        config,
        args.db,
        source_root,
        since=_parse_since(args.since, config.timezone),
        camera_ids=args.camera,
    )
    print(
        json.dumps(
            {
                "scan_run_id": summary.scan_run_id,
                "files_seen": summary.files_seen,
                "parse_errors": summary.parse_errors,
                "statuses": summary.statuses,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _probe(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source_root = config.source_root(args.root)
    connection = connect_database(args.db)
    camera_map = {camera.id: camera for camera in config.cameras}
    rows = get_ready_files(connection, args.limit, args.camera)
    completed = 0
    failed = 0
    try:
        for row in rows:
            path = source_root.joinpath(*row["relative_path"].split("/"))
            camera = camera_map[row["camera_id"]]
            try:
                probe = probe_media(path, args.ffprobe)
                save_probe(
                    connection,
                    row,
                    probe,
                    audio_enabled=(
                        config.audio_enabled and camera.audio_enabled
                    ),
                    audio_event_labels=config.audio_event_labels,
                )
                completed += 1
            except Exception as exc:
                save_probe_error(connection, int(row["id"]), str(exc))
                failed += 1
            connection.commit()
    finally:
        connection.close()
    print(
        json.dumps(
            {"selected": len(rows), "completed": completed, "failed": failed},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 2


def _report(args: argparse.Namespace) -> int:
    connection = connect_database(args.db)
    try:
        report = build_report(connection)
    finally:
        connection.close()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    totals = report["totals"]
    print(f"files={totals['files']} bytes={totals['bytes']}")
    for camera in report["cameras"]:
        print(
            "{id} {name}: files={files} bytes={bytes} "
            "waiting={waiting} ready={ready} probed={probed} errors={errors}".format(
                **camera
            )
        )
    for job in report["jobs"]:
        print(f"job {job['job_type']} {job['status']}={job['jobs']}")
    for event in report["audio_events"]:
        print(
            "audio {event_label}: events={events} "
            "max_confidence={maximum_confidence:.3f}".format(**event)
        )
    return 0


def _database_backup(args: argparse.Namespace) -> int:
    backup = backup_database(
        args.db,
        backup_directory=args.backup_dir,
        label=args.label,
    )
    if backup is None:
        raise FileNotFoundError(f"Database does not exist: {args.db}")
    print(
        json.dumps(
            {"database": str(Path(args.db).resolve()), "backup": str(backup)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _watermark_run(args: argparse.Namespace) -> int:
    from .watermark_ocr import run_watermark_ocr

    config = load_config(args.config)
    source_root = config.source_root(args.root)
    camera_map = {camera.id: camera for camera in config.cameras}
    if args.camera not in camera_map:
        raise ValueError(f"Unknown camera: {args.camera}")
    camera = camera_map[args.camera]
    if camera.watermark_crop is None:
        raise ValueError(f"Camera {camera.id} has no watermark crop")
    interval = args.interval or camera.watermark_sample_seconds

    connection = connect_database(args.db)
    rows = get_queued_jobs(
        connection,
        "watermark_time_map",
        args.limit,
        args.camera,
        include_completed=args.rerun_completed,
        media_file_id=args.media_id,
        audio_event_label=args.event_label,
    )
    completed = 0
    failed = 0
    results: list[dict] = []
    try:
        for row in rows:
            path = source_root.joinpath(*row["relative_path"].split("/"))
            if not mark_job_running(
                connection,
                int(row["job_id"]),
                include_completed=args.rerun_completed,
            ):
                continue
            connection.commit()
            try:
                run = run_watermark_ocr(
                    path,
                    timezone=config.timezone,
                    crop=camera.watermark_crop,
                    interval_seconds=interval,
                    ffmpeg=args.ffmpeg,
                    hwaccel=args.hwaccel,
                    minimum_confidence=args.minimum_confidence,
                )
                metadata = {
                    "frames_seen": run.frames_seen,
                    "valid_observations": len(run.observations),
                    "ocr_failures": run.ocr_failures,
                    "segments": len(run.segments),
                    "sample_interval_seconds": interval,
                    "minimum_confidence": args.minimum_confidence,
                    "hwaccel": args.hwaccel,
                    "model": run.model,
                }
                save_time_segments(
                    connection,
                    job_id=int(row["job_id"]),
                    media_file_id=int(row["id"]),
                    segments=run.segments,
                    source="watermark_ocr:rapidocr",
                    metadata=metadata,
                )
                completed += 1
                results.append(
                    {
                        "relative_path": row["relative_path"],
                        **metadata,
                    }
                )
            except Exception as exc:
                mark_job_error(connection, int(row["job_id"]), str(exc))
                failed += 1
                results.append(
                    {
                        "relative_path": row["relative_path"],
                        "error": str(exc),
                    }
                )
            connection.commit()
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "selected": len(rows),
                "completed": completed,
                "failed": failed,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 2


def _audio_model_install(args: argparse.Namespace) -> int:
    from .audio_model import install_yamnet

    result = install_yamnet(args.model_dir)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _audio_run(args: argparse.Namespace) -> int:
    from .audio_events import YamnetDetector

    config = load_config(args.config)
    if not config.audio_enabled:
        raise ValueError("Audio analysis is disabled in the project config")
    source_root = config.source_root(args.root)
    detector = YamnetDetector(
        model_path=args.model or config.audio_model_path,
        class_map_path=args.labels or config.audio_label_map_path,
        event_labels=config.audio_event_labels,
        minimum_confidence=(
            args.minimum_confidence
            if args.minimum_confidence is not None
            else config.audio_minimum_confidence
        ),
        merge_gap_seconds=(
            args.merge_gap
            if args.merge_gap is not None
            else config.audio_merge_gap_seconds
        ),
    )

    connection = connect_database(args.db)
    rows = get_queued_jobs(
        connection,
        "audio_event_detection",
        args.limit,
        args.camera,
        media_file_id=args.media_id,
    )
    completed = 0
    failed = 0
    results: list[dict] = []
    try:
        for row in rows:
            path = source_root.joinpath(*row["relative_path"].split("/"))
            if not mark_job_running(connection, int(row["job_id"])):
                continue
            connection.commit()
            try:
                analysis = detector.analyze(
                    path,
                    ffmpeg=args.ffmpeg,
                    timeout_seconds=args.timeout_seconds,
                )
                counts = Counter(event.label for event in analysis.events)
                metadata = {
                    "windows_seen": analysis.windows_seen,
                    "raw_detections": analysis.raw_detections,
                    "model": analysis.model,
                    "model_sha256": analysis.model_sha256,
                    "minimum_confidence": analysis.minimum_confidence,
                    "merge_gap_seconds": analysis.merge_gap_seconds,
                    "labels": list(config.audio_event_labels),
                }
                save_audio_events(
                    connection,
                    job_id=int(row["job_id"]),
                    media_file_id=int(row["id"]),
                    events=analysis.events,
                    model_name=analysis.model,
                    metadata=metadata,
                )
                completed += 1
                results.append(
                    {
                        "relative_path": row["relative_path"],
                        "windows_seen": analysis.windows_seen,
                        "events": dict(sorted(counts.items())),
                    }
                )
            except Exception as exc:
                mark_job_error(connection, int(row["job_id"]), str(exc))
                failed += 1
                results.append(
                    {
                        "relative_path": row["relative_path"],
                        "error": str(exc),
                    }
                )
            connection.commit()
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "selected": len(rows),
                "completed": completed,
                "failed": failed,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failed == 0 else 2


def _shadow_calibrate(args: argparse.Namespace) -> int:
    from .shadow_calibration import run_shadow_calibration

    config = load_config(args.config)
    if not config.audio_enabled:
        raise ValueError("Audio analysis is disabled in the project config")

    def emit(payload: dict) -> None:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    result = run_shadow_calibration(
        database_path=args.db,
        source_root=config.source_root(args.root),
        timezone=config.timezone,
        model_path=args.model or config.audio_model_path,
        class_map_path=args.labels or config.audio_label_map_path,
        event_label=args.event_label,
        minimum_confidence=(
            args.minimum_confidence
            if args.minimum_confidence is not None
            else config.audio_minimum_confidence
        ),
        gap_seconds=args.gap_seconds,
        workers=args.workers,
        padding_seconds=args.padding_seconds,
        timeout_seconds=args.timeout_seconds,
        ffmpeg=args.ffmpeg,
        output_path=args.output,
        on_progress=emit,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


def _events(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    since = _parse_since(args.since, config.timezone)
    until = _parse_since(args.until, config.timezone)
    connection = connect_database(args.db)
    try:
        rows = query_audio_events(
            connection,
            camera_id=args.camera,
            event_label=args.label,
            since_ts=since.timestamp() if since else None,
            until_ts=until.timestamp() if until else None,
            minimum_confidence=args.minimum_confidence,
            limit=args.limit,
        )
    finally:
        connection.close()

    timezone = ZoneInfo(config.timezone)
    results: list[dict] = []
    for row in rows:
        wall_start = (
            datetime.fromtimestamp(row["wall_start_ts"], timezone).isoformat()
            if row["wall_start_ts"] is not None
            else None
        )
        wall_end = (
            datetime.fromtimestamp(row["wall_end_ts"], timezone).isoformat()
            if row["wall_end_ts"] is not None
            else None
        )
        results.append(
            {
                "event_id": row["event_id"],
                "event_label": row["event_label"],
                "model_label": row["model_label"],
                "confidence": row["confidence"],
                "camera_id": row["camera_id"],
                "camera_name": row["camera_name"],
                "location": row["location"],
                "wall_start": wall_start,
                "wall_end": wall_end,
                "time_status": (
                    "mapped" if wall_start is not None else "watermark_pending"
                ),
                "media_file_id": row["media_file_id"],
                "relative_path": row["relative_path"],
                "media_start_seconds": row["media_start_seconds"],
                "media_end_seconds": row["media_end_seconds"],
            }
        )

    if args.json:
        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for event in results:
        when = event["wall_start"] or "待水印映射"
        print(
            f"{when} {event['camera_name']} {event['event_label']} "
            f"{event['confidence']:.3f} "
            f"{event['relative_path']}@{event['media_start_seconds']:.3f}s"
        )
    return 0


def _merge_analysis(args: argparse.Namespace) -> int:
    target = Path(args.db).resolve()
    source = Path(args.from_db).resolve()
    if target == source:
        raise ValueError("Target and source databases must be different")
    connection = connect_database(target)
    try:
        result = import_analysis_database(connection, source)
    finally:
        connection.close()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Web dependencies are missing; install with "
            "`pip install -e .[web]`"
        ) from exc
    from .web import create_app

    app = create_app(
        database_path=args.db,
        config_path=args.config,
        source_root=args.root,
        preview_dir=args.preview_dir,
        ffmpeg=args.ffmpeg,
        allowed_networks=args.allowed_network,
        models_config_path=args.models_config,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def _visual_options(args: argparse.Namespace):
    from .visual_pipeline import (
        DEFAULT_CAMERAS,
        VisualPipelineOptions,
    )

    config = load_config(args.config)
    since = _parse_since(args.since, config.timezone)
    until = _parse_since(args.until, config.timezone)
    if since is None or until is None:
        raise ValueError("visual analysis requires --since and --until")
    return VisualPipelineOptions(
        database_path=Path(args.db),
        config_path=Path(args.config),
        source_root=config.source_root(args.root),
        output_root=Path(args.output_root),
        since_ts=int(since.timestamp()),
        until_ts=int(until.timestamp()),
        camera_ids=tuple(dict.fromkeys(args.camera or DEFAULT_CAMERAS)),
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        hwaccel=args.hwaccel or None,
        detector_model=args.detector_model,
        embedding_model=args.embedding_model,
        model_cache=Path(args.model_cache),
        probe_limit=args.probe_limit,
        probe_workers=args.probe_workers,
        extract_limit=args.extract_limit,
        extract_workers=args.extract_workers,
        watermark_limit=args.watermark_limit,
        watermark_workers=args.watermark_workers,
        watermark_interval_seconds=args.watermark_interval_seconds,
        inference_file_limit=args.inference_file_limit,
        inference_frame_limit=args.inference_frame_limit,
        detector_batch_size=args.detector_batch_size,
        embedding_batch_size=args.embedding_batch_size,
        target_width=args.target_width,
        bedside_interval_seconds=args.bedside_interval_seconds,
        kitchen_interval_seconds=args.kitchen_interval_seconds,
        error_retry_delay_seconds=args.error_retry_delay_seconds,
        lock_path=Path(args.lock),
    )


def _visual_run(args: argparse.Namespace) -> int:
    from .pipeline import PipelineAlreadyRunning
    from .visual_pipeline import run_visual_cycle

    try:
        result = run_visual_cycle(_visual_options(args))
    except PipelineAlreadyRunning as exc:
        print(
            json.dumps(
                {"status": "skipped", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 3
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


def _visual_worker(args: argparse.Namespace) -> int:
    from .pipeline import PipelineAlreadyRunning
    from .visual_pipeline import run_visual_worker

    def emit(payload: dict) -> None:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    try:
        run_visual_worker(
            _visual_options(args),
            interval_seconds=args.interval_seconds,
            max_cycles=args.max_cycles,
            on_cycle=emit,
        )
    except PipelineAlreadyRunning as exc:
        print(
            json.dumps(
                {"status": "skipped", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 3
    return 0


def _visual_progress(args: argparse.Namespace) -> int:
    from .visual_database import query_visual_progress

    config = load_config(args.config)
    since = _parse_since(args.since, config.timezone)
    until = _parse_since(args.until, config.timezone)
    connection = connect_database(args.db)
    try:
        payload = query_visual_progress(
            connection,
            since_ts=(int(since.timestamp()) if since else None),
            until_ts=(int(until.timestamp()) if until else None),
            camera_ids=tuple(dict.fromkeys(args.camera)),
        )
    finally:
        connection.close()
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _visual_rebuild(args: argparse.Namespace) -> int:
    from .activity_inference import rebuild_activity_events
    from .visual_models import PROMPT_VERSION
    from .visual_pipeline import DEFAULT_CAMERAS

    config = load_config(args.config)
    since = _parse_since(args.since, config.timezone)
    until = _parse_since(args.until, config.timezone)
    if since is None or until is None:
        raise ValueError("visual activity rebuild requires --since and --until")
    connection = connect_database(args.db)
    try:
        payload = rebuild_activity_events(
            connection,
            since_ts=int(since.timestamp()),
            until_ts=int(until.timestamp()),
            camera_ids=tuple(
                dict.fromkeys(args.camera or DEFAULT_CAMERAS)
            ),
            embedding_model=args.embedding_model,
            prompt_version=PROMPT_VERSION,
        )
        connection.commit()
    finally:
        connection.close()
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _visual_verify(args: argparse.Namespace) -> int:
    from .visual_models import PROMPT_VERSION
    from .visual_pipeline import DEFAULT_CAMERAS
    from .visual_verifier import verify_candidate_frames

    config = load_config(args.config)
    since = _parse_since(args.since, config.timezone)
    until = _parse_since(args.until, config.timezone)
    if since is None or until is None:
        raise ValueError("visual verification requires --since and --until")
    connection = connect_database(args.db)
    try:
        def emit_progress(progress: dict) -> None:
            print(
                json.dumps(
                    {"status": "running", **progress},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        payload = verify_candidate_frames(
            connection,
            since_ts=int(since.timestamp()),
            until_ts=int(until.timestamp()),
            camera_ids=tuple(
                dict.fromkeys(args.camera or DEFAULT_CAMERAS)
            ),
            embedding_model=args.embedding_model,
            prompt_version=PROMPT_VERSION,
            image_root=args.image_root,
            media_root=args.media_root,
            model_dir=args.model_dir,
            models_config_path=args.models_config,
            tasks=tuple(
                args.task
                or ("feeding", "toy", "sleep_presence", "posture")
            ),
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            temporal_max_frames=args.temporal_max_frames,
            feeding_bucket_seconds=args.feeding_bucket_seconds,
            feeding_minimum_rank=args.feeding_minimum_rank,
            feeding_dates=tuple(dict.fromkeys(args.feeding_date)),
            timezone_name=config.timezone,
            force=args.force,
            on_progress=emit_progress,
        )
    finally:
        connection.close()
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _visual_embed_qwen3(args: argparse.Namespace) -> int:
    from .model_config import load_models_config
    from .visual_embedding_backfill import (
        backfill_qwen3_visual_embeddings,
    )

    models = load_models_config(args.models_config)
    connection = connect_database(args.db)
    try:
        def emit_progress(progress: dict) -> None:
            print(
                json.dumps(
                    {"status": "running", **progress},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        payload = backfill_qwen3_visual_embeddings(
            connection,
            config=models.visual_embedding,
            image_root=args.image_root,
            keyframe_ids=tuple(dict.fromkeys(args.keyframe_id)),
            limit=args.limit,
            batch_size=args.batch_size,
            on_progress=emit_progress,
        )
    finally:
        connection.close()
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _visual_milestones(args: argparse.Namespace) -> int:
    from .milestone_pipeline import verify_growth_milestones
    from .model_config import load_models_config

    since = _parse_since(args.since, args.timezone)
    until = _parse_since(args.until, args.timezone)
    if since is None or until is None:
        raise ValueError("milestone verification requires a time range")
    models = load_models_config(args.models_config)
    connection = connect_database(args.db)
    try:
        def emit_progress(progress: dict) -> None:
            print(
                json.dumps(
                    {"status": "running", **progress},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        payload = verify_growth_milestones(
            connection,
            models=models,
            image_root=args.image_root,
            since_ts=int(since.timestamp()),
            until_ts=int(until.timestamp()),
            limit=args.limit,
            on_progress=emit_progress,
        )
    finally:
        connection.close()
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _pipeline_options(args: argparse.Namespace):
    from .pipeline import PipelineOptions

    config = load_config(args.config)
    return PipelineOptions(
        database_path=Path(args.db),
        config_path=Path(args.config),
        source_root=args.root,
        camera_ids=tuple(dict.fromkeys(args.camera)),
        since=_parse_since(args.since, config.timezone),
        until=_parse_since(args.until, config.timezone),
        scan=args.scan,
        probe_limit_per_camera=args.probe_limit_per_camera,
        probe_workers=args.probe_workers,
        audio_limit_per_camera=args.audio_limit_per_camera,
        audio_workers=args.audio_workers,
        watermark_limit_per_camera=(
            args.watermark_limit_per_camera
        ),
        watermark_workers=args.watermark_workers,
        watermark_event_label=args.watermark_event_label,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        audio_timeout_seconds=args.audio_timeout_seconds,
        watermark_hwaccel=args.watermark_hwaccel,
        watermark_interval_seconds=args.watermark_interval,
        watermark_minimum_confidence=(
            args.watermark_minimum_confidence
        ),
        stale_job_seconds=args.stale_job_seconds,
        error_retry_delay_seconds=args.error_retry_delay_seconds,
        lock_path=Path(args.lock),
        mode=args.mode,
    )


def _pipeline_run(args: argparse.Namespace) -> int:
    from .pipeline import PipelineAlreadyRunning, run_pipeline_once

    try:
        result = run_pipeline_once(_pipeline_options(args))
    except PipelineAlreadyRunning as exc:
        print(
            json.dumps(
                {"status": "skipped", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 3
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 2


def _pipeline_worker(args: argparse.Namespace) -> int:
    from .pipeline import PipelineAlreadyRunning, run_pipeline_worker

    def emit(payload: dict) -> None:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    try:
        run_pipeline_worker(
            _pipeline_options(args),
            interval_seconds=args.interval_seconds,
            scan_every_cycles=args.scan_every_cycles,
            max_cycles=args.max_cycles,
            worker_lock_path=args.worker_lock,
            on_cycle=emit,
        )
    except PipelineAlreadyRunning as exc:
        print(
            json.dumps(
                {"status": "skipped", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return _scan(args)
    if args.command == "probe":
        return _probe(args)
    if args.command == "report":
        return _report(args)
    if args.command == "db-backup":
        return _database_backup(args)
    if args.command == "watermark-plan":
        print(
            json.dumps(
                sample_offsets(args.duration, args.interval),
                indent=2,
            )
        )
        return 0
    if args.command == "watermark-run":
        return _watermark_run(args)
    if args.command == "audio-model-install":
        return _audio_model_install(args)
    if args.command == "audio-run":
        return _audio_run(args)
    if args.command == "shadow-calibrate":
        return _shadow_calibrate(args)
    if args.command == "events":
        return _events(args)
    if args.command == "merge-analysis":
        return _merge_analysis(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "pipeline-run":
        return _pipeline_run(args)
    if args.command == "pipeline-worker":
        return _pipeline_worker(args)
    if args.command == "visual-run":
        return _visual_run(args)
    if args.command == "visual-worker":
        return _visual_worker(args)
    if args.command == "visual-progress":
        return _visual_progress(args)
    if args.command == "visual-verify":
        return _visual_verify(args)
    if args.command == "visual-embed-qwen3":
        return _visual_embed_qwen3(args)
    if args.command == "visual-milestones":
        return _visual_milestones(args)
    if args.command == "visual-rebuild":
        return _visual_rebuild(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
