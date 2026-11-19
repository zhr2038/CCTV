"""Render reviewed keeps, retaining whole source copies and a source edit map."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil

from .growth_workflow import command, connect, meta, safe_output, save_json, sha_file


def merge_keeps(records):
    """Merge overlapping decisions about the same source without double-playing."""
    grouped = {}
    for record in records:
        key = record["source_sha256"]
        grouped.setdefault(key, []).append(record)
    merged = []
    for sources in grouped.values():
        current = None
        for row in sorted(sources, key=lambda r: r["media_start_seconds"]):
            if current is not None and row["media_start_seconds"] <= current["media_end_seconds"]:
                current["media_end_seconds"] = max(current["media_end_seconds"], row["media_end_seconds"])
                current["reviewed_window_ids"].extend(row["reviewed_window_ids"])
            else:
                current = dict(row)
                current["reviewed_window_ids"] = list(row["reviewed_window_ids"])
                merged.append(current)
    return sorted(merged, key=lambda r: (r["capture_hint"], r["source_relative_path"], r["media_start_seconds"]))


def subtract_exclusions(records, exclusions):
    result = []
    for record in records:
        pieces = [(record['media_start_seconds'], record['media_end_seconds'])]
        for excluded in exclusions:
            if excluded['source_sha256'] != record['source_sha256']:
                continue
            start, end = excluded['media_start_seconds'], excluded['media_end_seconds']
            updated = []
            for low, high in pieces:
                if end <= low or start >= high:
                    updated.append((low, high))
                else:
                    if low < start:
                        updated.append((low, start))
                    if end < high:
                        updated.append((end, high))
            pieces = updated
        for low, high in pieces:
            result.append({**record, 'media_start_seconds': low, 'media_end_seconds': high})
    return result


def render_reviewed(batches, output):
    records = []
    exclusions = []
    manifests = []
    for name in batches:
        batch = Path(name).resolve()
        if not (batch / 'queue.sqlite3').is_file():
            raise FileNotFoundError('Unknown batch queue')
        db = connect(batch)
        try:
            manifest = meta(db, "manifest")
            safe_output(Path(output), Path(manifest["source"]))
            manifests.append(manifest)
            for row in db.execute("""SELECT w.*,f.cache,f.sha256,f.relative,f.capture_hint,f.time_source,f.watermark_required
                FROM windows w JOIN files f ON f.id=w.file_id JOIN reviews r ON r.window_id=w.id
                WHERE r.verdict='keep'"""):
                records.append({"source_relative_path": row["relative"], "source_sha256": row["sha256"],
                                "source_copy": str(batch / row["cache"]), "capture_hint": row["capture_hint"],
                                "time_source": row["time_source"], "watermark_required": bool(row["watermark_required"]),
                                "media_start_seconds": row["start"], "media_end_seconds": row["end"],
                                "reviewed_window_ids": [str(batch) + "::" + row["id"]]})
            for row in db.execute("""SELECT w.start,w.end,f.sha256 FROM windows w JOIN files f ON f.id=w.file_id
                JOIN reviews r ON r.window_id=w.id WHERE r.verdict='not_for_film'"""):
                exclusions.append({'source_sha256': row['sha256'], 'media_start_seconds': row['start'], 'media_end_seconds': row['end']})
        finally:
            db.close()
    if not records:
        raise ValueError("No explicit keep reviews. Review/export/import first; no AI label is promoted to a parent decision.")
    edit = subtract_exclusions(merge_keeps(records), exclusions)
    if not edit:
        raise ValueError('All selected ranges overlap explicit not-for-film decisions; review the decisions before export')
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Use a new export folder to preserve previous films")
    destination.mkdir(parents=True)
    (destination / "originals").mkdir()
    (destination / "segments").mkdir()
    timeline = 0.0
    segments = []
    for index, item in enumerate(edit):
        source = Path(item["source_copy"])
        if sha_file(source) != item["source_sha256"]:
            raise RuntimeError("Selected source copy changed; export halted")
        original = destination / "originals" / (item["source_sha256"] + ".mp4")
        if not original.exists():
            if shutil.disk_usage(destination).free < source.stat().st_size + 10_000_000_000:
                raise RuntimeError("Keep at least10GB free while protecting selected originals")
            shutil.copyfile(source, original)
            if sha_file(original) != item["source_sha256"]:
                raise RuntimeError("Protected whole original checksum mismatch")
        item["protected_full_original"] = original.relative_to(destination).as_posix()
        duration = item["media_end_seconds"] - item["media_start_seconds"]
        if duration <= 0:
            raise ValueError("Invalid selected range")
        info = json.loads(command(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", original]))
        has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
        segment = destination / "segments" / f"{index:04d}.mp4"
        args = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(item["media_start_seconds"]), "-i", original]
        if not has_audio:
            args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        args += ["-t", str(duration), "-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
                 "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-threads", "4", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k", "-movflags", "+faststart", segment]
        command(args)
        measured = float(json.loads(command(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", segment]))["format"]["duration"])
        item["film_start_seconds"] = timeline
        item["film_end_seconds"] = timeline + measured
        timeline += measured
        segments.append(segment)
    listing = destination / "segments.txt"
    # Paths are controlled relative names, so concat does not need unsafe quoting.
    listing.write_text("\n".join("file 'segments/" + path.name + "'" for path in segments), encoding="utf-8")
    movie = destination / "growth_film.mp4"
    command(["ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "1", "-i", listing,
             "-c", "copy", "-movflags", "+faststart", movie])
    command(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", movie, "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    result = {"created_at": datetime.now().astimezone().isoformat(), "movie": str(movie),
              "duration_seconds": float(json.loads(command(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", movie]))["format"]["duration"]),
              "movie_sha256": sha_file(movie), "selection_basis": "explicit imported keep reviews",
              "source_deletions": 0, "full_movie_decode_passed": True,
              "source_archive_replaces_NAS_authorized": False,
              "manual_not_for_film_intervals_take_precedence": True, "manual_excluded_ranges": exclusions,
              "edits": edit, "batch_scopes": manifests}
    save_json(destination / "edit.json", result)
    print(json.dumps({"movie": str(movie), "segments": len(edit), "duration_seconds": result["duration_seconds"], "source_deletions": 0}))
