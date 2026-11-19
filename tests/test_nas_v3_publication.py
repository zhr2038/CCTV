import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))
from nas_growth import app, curation_queue
from nas_growth.curation_queue import CurationQueue
from nas_growth.daemon import read_container_headroom_kib
from nas_growth.db import ArchiveDB
from nas_growth.lifecycle import ArchiveCloudSync


def test_final_request_survives_error_restart_and_new_footage(tmp_path, monkeypatch):
    db = ArchiveDB(tmp_path / "archive.db")
    queue = CurationQueue(db)
    now = [1000]
    monkeypatch.setattr(curation_queue.time, "time", lambda: now[0])
    queue.request("2026-11-18", "P0_LIVE", False, 40)
    day, first = queue.next_ready()
    queue.request(day, "P0_LIVE", True, 80)
    queue.finish(day, first)
    queue = CurationQueue(db)
    day, final = queue.next_ready()
    assert final["final"] and final["done_files"] == 80
    queue.finish(day, final, "encoder failure")
    queue.request("2026-11-17", "P1_BACKFILL", True, 1)
    assert queue.next_ready()[0] == "2026-11-17"
    now[0] += 60
    day, retry = CurationQueue(db).next_ready()
    assert day == "2026-11-18" and retry["attempts"] == 1
    queue.finish(day, retry)
    assert day not in CurationQueue(db).jobs


def test_curation_db_failure_does_not_acknowledge_job(tmp_path, monkeypatch):
    queue = CurationQueue(ArchiveDB(tmp_path / "archive.db"))
    queue.request("2026-11-18", "P0", True, 1)
    day, job = queue.next_ready()
    with monkeypatch.context() as mp:
        def fail(*args):
            raise OSError("disk unavailable")
        mp.setattr(queue.db, "set_state", fail)
        with pytest.raises(OSError):
            queue.finish(day, job)
    assert queue.next_ready() == (day, job)
    queue.finish(day, job)
    assert queue.next_ready() is None


def test_sync_only_collects_published_products(tmp_path):
    db = ArchiveDB(tmp_path / "state" / "archive.db")
    sync = ArchiveCloudSync(tmp_path, db, {"cameras": {"bed": {"role": "bedside_crib"}}})
    gallery = {"v2/highlights/play.mp4", "v2/photos/photo.jpg", "v2/monthly_cards/month.jpg",
               "v2/smooth_kitchen_1x.mp4", "v2/deliverables.json", "days/2026-11-18/summary.json",
               "days/2026-11-18/timelapse_kitchen.mp4", "day-20261108/highlight_20261108.mp4"}
    excluded = {"v2/t0_keepers/source.mp4", "v2/work/test.mp4", "v2/.staging/photo.jpg",
                "day-20261108/t1/cam/test.mp4", "days/2026-11-18/timelapse_bed.mp4",
                "v2/highlights/play.tmp_v.mp4", "day-20261108/timelapse_bed.mp4",
                "state/growth_v3.sqlite3", "days/2026-11-18/timelapse_kitchen.tmp_concat.mp4"}
    for rel in gallery | excluded:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"sample")
    assert {rel for rel, _, _ in sync._iter_curated_files()} == gallery


def test_cgroup_admission_budget_accounts_for_reclaimable_cache(tmp_path):
    for name, value in {"memory.max": "3221225472", "memory.current": "3019898880",
                        "memory.stat": "anon 2097152000\ninactive_file 134217728\n"}.items():
        (tmp_path / name).write_text(value)
    assert read_container_headroom_kib(tmp_path) == 327680
    (tmp_path / "memory.max").write_text("max")
    assert read_container_headroom_kib(tmp_path) is None


def test_process_day_cannot_run_alongside_a_processing_owner(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[project]\ndata_root="{tmp_path.as_posix()}"\n')
    state = tmp_path / "state"
    state.mkdir()
    with (state / "processor.lock").open("a+b") as owner:
        app._lock_processor(owner)
        assert app.main(["--config", str(cfg), "process-day", "--day", "2026-11-18"]) == 2
        assert app.main(["--config", str(cfg), "ctl", "pause"]) == 0
    monkeypatch.setattr(app, "_run_processor", lambda *args: 0)
    assert app.main(["--config", str(cfg), "process-day", "--day", "2026-11-18"]) == 0
