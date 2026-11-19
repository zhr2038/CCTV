from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))
from nas_growth import daemon, lifecycle
from nas_growth import db as db_module
from nas_growth.db import ArchiveDB


DAY = "2026-07-20"


def source(name, size, hour=12):
    return {"day": DAY, "camera": "cam", "source_rel": name, "wall_ts": f"{DAY}T{hour:02d}:30:00",
            "size": size, "mtime": 1.0}


def test_heavy_cap_fills_night_light_work_after_full_daytime_page(tmp_path):
    db = ArchiveDB(tmp_path / "state.db")
    heavy = [source(f"heavy_{i}.mp4", 61_000_000) for i in range(40)]
    light = [source(f"light_{i}.mp4", 5_000, 23) for i in range(6)]
    db.upsert_discovered_files(heavy + light)
    dm = SimpleNamespace(db=db)
    batch = daemon.GrowthDaemon.pick_next_batch_files(dm, [DAY], DAY, 6, set(), 1, 2)
    assert len(batch) == 6 and sum(r["size"] > 60_000_000 for r in batch) == 1
    batch = daemon.GrowthDaemon.pick_next_batch_files(dm, [DAY], DAY, 6, {"light_0.mp4"}, 2, 2)
    assert {r["source_rel"] for r in batch} == {r["source_rel"] for r in light[1:]}


def test_requeue_changed_and_returned_source(tmp_path):
    db = ArchiveDB(tmp_path / "state.db")
    row = source("file.mp4", 100)
    assert db.upsert_discovered_files([row]) == 1
    db.mark_file_done(row["source_rel"], {"status": "missing"})
    assert db.upsert_discovered_files([row]) == 1
    assert db.upsert_discovered_files([row]) == 0
    db.mark_file_done(row["source_rel"], {})
    assert db.upsert_discovered_files([{**row, "size": 120}]) == 1
    assert db.get_queued_files_for_day(DAY)[0]["size"] == 120


def test_legacy_t2_failure_has_a_bounded_persistent_retry_budget(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as con:
        legacy_schema = db_module.SCHEMA_SQL.replace("    retry_count INTEGER NOT NULL DEFAULT 0,\n    retry_at REAL,\n", "")
        con.executescript(legacy_schema)
        con.executemany(
            "INSERT INTO files(source_rel,day,camera,status,error,updated_at) VALUES(?,?,?,'error',?,?)",
            [("retry.mp4", DAY, "cam", "t2_decode_failed", 1000), ("other.mp4", DAY, "cam", "unreadable_media", 1000)],
        )
    current = [1000]
    monkeypatch.setattr(db_module.time, "time", lambda: current[0])
    db = ArchiveDB(db_path)
    assert db.requeue_due_t2_failures(DAY, DAY) == 0
    current[0] += 60
    assert db.requeue_due_t2_failures(DAY, DAY) == 1
    assert ArchiveDB(db_path).requeue_due_t2_failures(DAY, DAY) == 0
    assert [r["source_rel"] for r in db.get_queued_files_for_day(DAY)] == ["retry.mp4"]
    error = {"status": "error", "error": "t2_decode_failed"}
    db.mark_file_done("retry.mp4", error)
    current[0] += 299
    assert db.requeue_due_t2_failures(DAY, DAY) == 0
    current[0] += 1
    assert ArchiveDB(db_path).requeue_due_t2_failures(DAY, DAY) == 1
    db.mark_file_done("retry.mp4", error)
    current[0] += 100_000
    assert ArchiveDB(db_path).requeue_due_t2_failures(DAY, DAY) == 0
    with db.session() as con:
        row = con.execute("SELECT status,retry_count,retry_at FROM files WHERE source_rel='retry.mp4'").fetchone()
    assert tuple(row) == ("error", 2, None)
    assert db.upsert_discovered_files([source("retry.mp4", 200)]) == 1
    assert db.get_queued_files_for_day(DAY)[0]["retry_count"] == 0


@pytest.mark.parametrize("offline_days", [2, 30])
def test_continuous_live_days_resume_after_downtime_without_expanding_historical_window(tmp_path, offline_days):
    db = ArchiveDB(tmp_path / "state.db")
    first = dt.date(2026, 8, 10)
    attempted = []
    available = set()

    def discover(day):
        attempted.append(day)
        if day in available:
            return db.upsert_discovered_files([{**source(f"{day}.mp4", 5000), "day": day}])
        return 0

    def start():
        dm = daemon.GrowthDaemon.__new__(daemon.GrowthDaemon)
        dm.cfg = {"scheduler": {"p1_start_day": "2026-07-05", "p1_end_day": "2026-07-03"}}
        dm.db = db
        dm._first_live_day = None
        dm._next_t2_retry_check = 0
        dm._exhausted_days = set()
        dm.discover_day_files = discover
        return dm

    start().ensure_rolling_days(first)
    assert set(attempted) == {"2026-08-09", "2026-07-05", "2026-07-04", "2026-07-03"}
    resumed = first + dt.timedelta(days=offline_days)
    missed = (resumed - dt.timedelta(days=2)).isoformat()
    available.update([missed, "2026-07-04"])
    attempted.clear()
    active, _, _ = start().ensure_rolling_days(resumed)
    assert active == [missed, "2026-07-04"]
    assert "2026-08-08" not in attempted


def test_upgrade_infers_prior_live_days_without_a_saved_scan_checkpoint(tmp_path):
    db = ArchiveDB(tmp_path / "state.db")
    db.upsert_discovered_files([
        {**source("historical.mp4", 10), "day": "2026-07-01"},
        {**source("prior_live.mp4", 10), "day": "2026-08-09"},
    ])
    assert db.first_live_day("2026-07-05", "2026-09-01") == "2026-08-09"


def test_inflight_vlm_responses_preserve_provider_cooldown(tmp_path, monkeypatch):
    current = [1000]
    monkeypatch.setattr(db_module.time, "time", lambda: current[0])
    db = ArchiveDB(tmp_path / "state.db")
    db.record_vlm_result("provider", "model", False, 0, "quota", 21600)
    deadline = db.get_vlm_cooldown("provider")
    current[0] += 5
    db.record_vlm_result("provider", "model", True, 2, usage={"total_tokens": 30})
    db.record_vlm_result("provider", "model", False, 0, "transient", 60)
    assert db.get_vlm_cooldown("provider") == deadline
    row = db.get_dashboard_snapshot()["vlm_stats"][0]
    assert (row["calls_ok"], row["calls_err"], row["total_tokens"], row["last_error"]) == (1, 2, 30, "quota")
    current[0] = deadline + 1
    db.record_vlm_result("provider", "model", True, 2)
    assert db.get_vlm_cooldown("provider") == 0


@pytest.mark.parametrize("stage", ["t1", "t2"])
def test_failed_required_output_is_not_archived(tmp_path, monkeypatch, stage):
    db = ArchiveDB(tmp_path / "state.db")
    row = source("src.mp4", 99)
    db.upsert_discovered_files([row])
    dm = SimpleNamespace(db=db, ff="ffmpeg", fp="ffprobe", cfg={"media": {}}, data_root=tmp_path,
                         yolo=None, use_vaapi=False, vaapi_dev="", _resolve_source_path=lambda r: Path(r["source_rel"]))
    monkeypatch.setattr(daemon.med, "probe_media", lambda *a: dict(duration=10, width=10, height=10, has_audio=False))

    def analyze(_ff, _src, t2, *_args, **_kwargs):
        if stage == "t1":
            t2.parent.mkdir(parents=True)
            t2.write_bytes(b"T2")
        return dict(n_frames=10, boxes=[[]] * 10, gmot=np.ones(10), bmot=np.zeros(10), pconf=np.zeros(10), cands=[])

    monkeypatch.setattr(daemon.med, "stream_1fps_t2_and_analyze", analyze)
    monkeypatch.setattr(daemon.med, "encode_t1_segments", lambda *a, **kw: (False, "encode failed"))
    daemon.GrowthDaemon._process_one_file(dm, row, {})
    result = db.get_day_rows(DAY)[0]
    assert result["status"] == "error"
    assert result["error"] == {"t1": "t1_encode_failed", "t2": "t2_output_missing"}[stage]


def test_retention_missing_backup_or_failed_history_blocks_enforcement(tmp_path):
    db = ArchiveDB(tmp_path / "state.db")
    orig = tmp_path / "cctv" / "src.mp4"
    orig.parent.mkdir()
    orig.write_bytes(b"original")
    os.utime(orig, (1, 1))
    row = source(orig.name, orig.stat().st_size)
    db.upsert_discovered_files([row])
    t2 = tmp_path / "t2.mp4"
    t2.write_bytes(b"media")
    db.mark_file_done(row["source_rel"], {"t2_rel": t2.name, "t2_bytes": t2.stat().st_size})
    summary = tmp_path / "days" / DAY / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"status": "done"}), encoding="utf8")
    when = dt.datetime(2026, 11, 18, tzinfo=dt.timezone.utc)
    backup = tmp_path / "backup.json"
    cfg = {"lifecycle": {"retention_mode": "enforce", "baidu_status_path": str(backup)}}
    db.set_state("p1_backfill_complete", True)
    backup.write_text(json.dumps({"phase": "completed_with_missing", "initial_complete": "2026-11-01"}))
    result = lifecycle.run_retention_pass(db, tmp_path, orig.parent, cfg, now_dt=when)
    assert not result["can_prune"] and orig.exists()
    backup.write_text(json.dumps({"phase": "completed", "initial_complete": "2026-11-01"}))
    other = {**source("other.mp4", 100), "day": "2026-07-21"}
    db.upsert_discovered_files([other])
    db.mark_file_done(other["source_rel"], {"status": "error"})
    result = lifecycle.run_retention_pass(db, tmp_path, orig.parent, cfg, now_dt=when)
    assert not result["can_prune"] and orig.exists()
    db.mark_file_done(other["source_rel"], {})
    orig.write_bytes(b"replacement source")
    os.utime(orig, (1, 1))
    result = lifecycle.run_retention_pass(db, tmp_path, orig.parent, cfg, now_dt=when)
    assert result["can_prune"] and result["eligible_files"] == 0 and orig.exists()


def test_days_error_files_syncs_after_retry_recovery_and_startup(tmp_path):
    db_path = tmp_path / "state.db"
    db = ArchiveDB(db_path)
    row = source("bedside_bundle.mp4", 120)
    db.upsert_discovered_files([row])
    db.mark_file_done(row["source_rel"], {"status": "error", "error": "t2_decode_failed"})
    db.upsert_day_summary(DAY, "P0_LIVE", "preview", {"photos": [], "milestones_count": 0, "speech_count": 0})
    snap1 = db.get_dashboard_snapshot()
    assert snap1["days"][0]["error_files"] == 1
    assert snap1["days"][0]["done_files"] == 0

    # Force immediate retry and confirm error_files drops without touching done_files
    with db.session() as con:
        con.execute("UPDATE files SET retry_at=? WHERE source_rel=?", (0.0, row["source_rel"]))
    assert db.requeue_due_t2_failures(DAY, DAY) == 1
    snap_requeued = db.get_dashboard_snapshot()
    assert snap_requeued["days"][0]["error_files"] == 0
    assert snap_requeued["days"][0]["done_files"] == 0

    # Simulate stale cached error_files column in existing DB -> reconciled on re-open
    db.mark_file_done(row["source_rel"], {"status": "error", "error": "t2_decode_failed"})
    with db.session() as con:
        con.execute("UPDATE files SET status='done', error=NULL WHERE source_rel=?", (row["source_rel"],))
    assert db.get_dashboard_snapshot()["days"][0]["error_files"] == 1
    db_reopened = ArchiveDB(db_path)
    assert db_reopened.get_dashboard_snapshot()["days"][0]["error_files"] == 0

    # Recovered done clears error_files immediately as well
    db_reopened.mark_file_done(row["source_rel"], {"status": "error", "error": "t2_decode_failed"})
    assert db_reopened.get_dashboard_snapshot()["days"][0]["error_files"] == 1
    db_reopened.mark_file_done(row["source_rel"], {"status": "done", "duration": 900.0, "width": 2880, "height": 1620})
    snap_done = db_reopened.get_dashboard_snapshot()
    assert snap_done["days"][0]["error_files"] == 0
    assert snap_done["days"][0]["done_files"] == 0  # preserved for curator delta detection


def test_rolling_days_spills_over_when_active_days_have_only_heavy_bundles(tmp_path):
    db = ArchiveDB(tmp_path / "state.db")
    files = [
        {"day": "2026-11-17", "camera": "bed", "source_rel": f"bed_17_{i}.mp4", "wall_ts": f"2026-11-17T0{i}:00:00", "size": 128_000_000, "mtime": 1.0}
        for i in range(2)
    ] + [
        {"day": "2026-11-16", "camera": "bed", "source_rel": f"bed_16_{i}.mp4", "wall_ts": f"2026-11-16T0{i}:00:00", "size": 128_000_000, "mtime": 1.0}
        for i in range(2)
    ] + [
        {"day": "2026-11-15", "camera": "room", "source_rel": f"room_15_{i}.mp4", "wall_ts": f"2026-11-15T1{i}:00:00", "size": 5_000_000, "mtime": 1.0}
        for i in range(4)
    ]
    db.upsert_discovered_files(files)

    dm = daemon.GrowthDaemon.__new__(daemon.GrowthDaemon)
    dm.cfg = {
        "scheduler": {
            "p1_start_day": "2026-11-16",
            "p1_end_day": "2025-12-17",
            "rolling_active_days": 2,
            "light_spillover_days": 2,
            "workers": 5,
            "max_heavy_workers": 2,
        }
    }
    dm.db = db
    dm._first_live_day = None
    dm._next_t2_retry_check = 0.0
    dm._exhausted_days = set()
    dm.discover_day_files = lambda _d: 0

    active_days, _, _ = dm.ensure_rolling_days(dt.date(2026, 11, 19))
    assert active_days == ["2026-11-17", "2026-11-16", "2026-11-15"]

    batch = dm.pick_next_batch_files(active_days, "2026-11-19", 5, set(), 0, max_heavy=2)
    assert len(batch) == 5
    assert sum(int(r["size"]) > 60_000_000 for r in batch) == 2
    assert sum(int(r["size"]) <= 60_000_000 for r in batch) == 3
    assert all(r["day"] == "2026-11-15" for r in batch if int(r["size"]) <= 60_000_000)
