"""Regression tests for Work Item ③: Monthly & 1st-year film scheduling, compilation, provenance, retry, and review."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "nas-growth-v3"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from nas_growth import daemon as daemon_mod, film_compiler, lifecycle as lc, media_nas as med, web_server  # noqa: E402
from nas_growth.daemon import GrowthDaemon  # noqa: E402
from nas_growth.db import ArchiveDB  # noqa: E402


def _resolve_ff_pair() -> tuple[str, str]:
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    if not ff or not fp:
        winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        for ff_cand in sorted(winget_root.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe")):
            fp_cand = ff_cand.with_name("ffprobe.exe")
            if ff_cand.is_file() and fp_cand.is_file():
                ff = str(ff_cand)
                fp = str(fp_cand)
                break
    assert ff and (shutil.which(ff) or Path(ff).is_file()), f"ffmpeg missing on PATH: {ff}"
    assert fp and (shutil.which(fp) or Path(fp).is_file()), f"ffprobe missing on PATH: {fp}"
    return ff, fp


def _make_sample_mp4(ff: str, out_mp4: Path, color: str = "blue", dur_s: float = 5.0) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=25:d={dur_s}",
        "-f", "lavfi", "-i", f"sine=frequency=520:sample_rate=16000:duration={dur_s}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-shortest", str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"ffmpeg testsrc failed: {proc.stderr}"


def _make_cfg(tmp_path: Path) -> dict:
    return {
        "project": {
            "name": "Film Test V3",
            "birthday": "2025-12-17",
            "cctv_root": str(tmp_path / "cctv"),
            "vault_root": str(tmp_path / "vault"),
            "data_root": str(tmp_path / "data"),
            "timezone": "Asia/Shanghai",
        },
        "scheduler": {
            "p0_interval_seconds": 900,
            "active_file_cooldown_seconds": 0,
            "closed_file_cooldown_seconds": 0,
            "min_host_mem_available_kib": 1024,
            "workers": 2,
            "max_heavy_workers": 2,
            "p1_start_day": "2026-11-16",
            "p1_end_day": "2025-12-17",
            "p1_chunk_files": 48,
        },
        "lifecycle": {
            "source_retention_days": 90,
            "retention_mode": "observe_only",
            "protect_vault_forever": True,
            "monthly_compile_day": 18,
            "archive_cloud_sync_enabled": False,
        },
        "asr": {
            "enabled": False,
            "yolo_onnx": str(tmp_path / "models" / "yolo.onnx"),
            "yamnet_model": str(tmp_path / "models" / "yamnet.tflite"),
            "yamnet_class_map": str(tmp_path / "models" / "yamnet_class_map.csv"),
        },
        "media": {
            "use_vaapi": False,
            "smooth_fps": 25,
            "smooth_fade_s": 0.4,
            "smooth_qp": 24,
            "smooth_width": 640,
            "smooth_height": 360,
        },
        "vlm": {"tiers": {"fast": [], "pro": []}, "providers": {}},
        "cameras": {
            "cam_living": {
                "name": "客厅看娃",
                "subdir": "aqara_video/living",
                "enabled": True,
                "zoom_factor": 1.3,
                "milestone_watch": True,
                "daytime_photos": True,
                "t1_mode": "always",
            },
            "cam_bed": {
                "name": "床头监控",
                "subdir": "xiaomi_camera_videos/bed",
                "enabled": True,
                "zoom_factor": 1.0,
                "milestone_watch": True,
                "daytime_photos": False,
                "t1_mode": "always",
            },
        },
    }


def _seed_clip_milestone(
    ff: str,
    db: ArchiveDB,
    cctv_root: Path,
    data_root: Path,
    *,
    ms_id: str,
    day: str,
    camera: str,
    rel: str,
    wall_ts: str,
    color: str,
    category: str,
    caption: str,
    score: float = 5.0,
    review_status: str = "approved",
) -> Path:
    src = cctv_root / rel
    _make_sample_mp4(ff, src, color=color, dur_s=5.0)
    db.upsert_discovered_files([{
        "source_rel": rel,
        "source_root": "cctv",
        "day": day,
        "camera": camera,
        "wall_ts": wall_ts,
        "size": src.stat().st_size,
        "mtime": time.time() - 3600,
    }])
    db.mark_file_done(rel, {
        "status": "done",
        "duration": 5.0,
        "width": 640,
        "height": 360,
        "has_audio": True,
        "active_s": 5,
        "person_s": 5,
        "t1_mode": "always",
    })
    t0_rel = f"v2/t0_keepers/{camera}_{day}_{src.name}"
    t0_dst = data_root / t0_rel
    t0_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, t0_dst)
    db.upsert_milestone({
        "id": ms_id,
        "day": day,
        "camera": camera,
        "category": category,
        "source_rel": rel,
        "start_s": 0.2,
        "end_s": 4.2,
        "caption_zh": caption,
        "speaker_note": "",
        "independence": "independent",
        "score": score,
        "video_rel": f"v2/highlights/{ms_id}.mp4",
        "poster_rel": "",
        "t0_rel": t0_rel,
        "t0_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "review_status": review_status,
        "meta": {"cam_zh": "客厅看娃" if camera == "cam_living" else "床头监控", "zoom_x": 1.3},
    })
    return src


def test_monthly_and_annual_schedule_and_catchup_gating(tmp_path: Path):
    """Schedule exposes due monthly (1-11m) and 1st-year annual specs gated on cutoff-day queue completion."""
    cfg = _make_cfg(tmp_path)
    data_root = Path(cfg["project"]["data_root"])
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    spec11 = lc.build_film_spec(dt.date(2025, 12, 17), 11, "monthly", compile_day=18)
    assert spec11["film_id"] == "month_11m"
    assert spec11["period_start"] == "2026-10-17"
    assert spec11["period_end"] == "2026-11-17"

    spec12 = lc.build_film_spec(dt.date(2025, 12, 17), 12, "annual", compile_day=18)
    assert spec12["film_id"] == "annual_01y"
    assert spec12["period_start"] == "2025-12-17"
    assert spec12["period_end"] == "2026-12-17"

    # Seed queued file on cutoff day 2026-11-17 -> due production must wait (ready=False)
    db.upsert_discovered_files([{
        "source_rel": "aqara_video/living/20261117/q1.mp4",
        "source_root": "cctv",
        "day": "2026-11-17",
        "camera": "cam_living",
        "wall_ts": "2026-11-17T23:55:00",
        "size": 100,
        "mtime": 1700000000.0,
    }])
    sch_blocked = lc.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 20), data_root, compile_day=18, db=db)
    due11 = [d for d in sch_blocked["due_productions"] if d["film_id"] == "month_11m"]
    assert len(due11) == 1
    assert due11[0]["ready"] is False

    # Mark cutoff day file done -> ready flips to True (catch-up past 18th)
    db.mark_file_done("aqara_video/living/20261117/q1.mp4", {"status": "done", "duration": 10.0, "width": 640, "height": 360})
    sch_ready = lc.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 20), data_root, compile_day=18, db=db)
    due11_ready = [d for d in sch_ready["due_productions"] if d["film_id"] == "month_11m"]
    assert due11_ready[0]["ready"] is True


def test_real_ffmpeg_monthly_annual_compilation_provenance_retry_and_review(tmp_path: Path, monkeypatch):
    """End-to-end FFmpeg film compilation, per-shot provenance, refusal exclusion, atomic retry, review, and sync."""
    ff, fp = _resolve_ff_pair()
    cfg = _make_cfg(tmp_path)
    data_root = Path(cfg["project"]["data_root"])
    cctv_root = Path(cfg["project"]["cctv_root"])
    vault_root = Path(cfg["project"]["vault_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    cctv_root.mkdir(parents=True, exist_ok=True)
    vault_root.mkdir(parents=True, exist_ok=True)

    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon_mod.med, "resolve_font", lambda *_a, **_kw: None)
    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *_a, **_kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *_a, **_kw: object())

    # Seed confirmed milestones across 3 months (08, 10, 11) plus one pending_review refusal & one rejected vlm_review
    rel_aug = "aqara_video/living/20260815/aug_100000.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_aug_play", day="2026-08-15", camera="cam_living", rel=rel_aug,
        wall_ts="2026-08-15T10:00:00", color="green", category="toy_play", caption="8月开心摇铃铛", score=4.8,
    )
    rel_oct = "aqara_video/living/20261020/oct_090000.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_oct_stand", day="2026-10-20", camera="cam_living", rel=rel_oct,
        wall_ts="2026-10-20T09:00:00", color="blue", category="standing", caption="10月扶沙发站起", score=5.0,
    )
    rel_nov = "xiaomi_camera_videos/bed/20261110/nov_160000.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_nov_walk", day="2026-11-10", camera="cam_bed", rel=rel_nov,
        wall_ts="2026-11-10T16:00:00", color="navy", category="walking", caption="11月迈出两小步", score=5.2,
    )
    rel_refused = "aqara_video/living/20261112/refused_180000.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_nov_refused", day="2026-11-12", camera="cam_living", rel=rel_refused,
        wall_ts="2026-11-12T18:00:00", color="red", category="daily_care", caption="云端安全策略拒答，已保留原视频区间待人工复核",
        score=4.9, review_status="pending_review",
    )
    db.upsert_vlm_review({
        "review_id": f"ms|{rel_refused}|0.2",
        "day": "2026-11-12",
        "camera": "cam_living",
        "kind": "milestone",
        "source_rel": rel_refused,
        "start_s": 0.2,
        "end_s": 4.2,
        "suggested_category": "daily_care",
        "status": "pending_review",
        "reason": "content_policy_blocked",
        "meta": {"milestone_id": "ms_nov_refused"},
    })

    dm = GrowthDaemon(cfg, db, ff_bin=ff, fp_bin=fp)

    # 1. Compile Monthly Film (month 11: 2026-10-18 .. 2026-11-17)
    spec_11 = lc.build_film_spec(dt.date(2025, 12, 17), 11, "monthly", compile_day=18)
    pub_11 = dm.compile_film(spec_11, force=True, out_w=640, out_h=360)
    assert pub_11["status"] == "published"
    assert pub_11["review_status"] == "pending_review"
    mp4_11 = data_root / pub_11["video_rel"]
    jpg_11 = data_root / pub_11["poster_rel"]
    man_11_p = data_root / pub_11["manifest_rel"]
    assert mp4_11.is_file() and mp4_11.stat().st_size > 1024
    assert jpg_11.is_file() and jpg_11.stat().st_size > 512
    assert man_11_p.is_file()

    vinfo = med.probe_media(fp, str(mp4_11))
    assert vinfo["duration"] > 4.0 and vinfo["width"] == 640 and vinfo["height"] == 360

    man_11 = json.loads(man_11_p.read_text(encoding="utf-8"))
    shot_rels = [s["source_rel"] for s in man_11["shots"]]
    assert shot_rels == [rel_oct, rel_nov]
    assert rel_refused not in shot_rels
    for s in man_11["shots"]:
        assert s["shot_index"] >= 1
        assert s["clip_wall_start_ts"]
        assert s["clip_wall_end_ts"]
        assert s["source_end_s"] > s["source_start_s"]
        assert s["film_offset_end_s"] > s["film_offset_start_s"]

    # Month card generated atomically too
    deliv = json.loads((data_root / "v2" / "deliverables.json").read_text(encoding="utf-8"))
    assert len(deliv["monthly_cards"]) == 1
    assert (data_root / "v2" / "monthly_cards" / deliv["monthly_cards"][0]["card"]).is_file()

    # 2. Compile 1st-Year Annual Film -> includes 3 calendar months (08, 10, 11), excludes refusal
    spec_y1 = lc.build_film_spec(dt.date(2025, 12, 17), 12, "annual", compile_day=18)
    pub_y1 = dm.compile_film(spec_y1, force=True, out_w=640, out_h=360)
    assert pub_y1["status"] == "published"
    man_y1 = json.loads((data_root / pub_y1["manifest_rel"]).read_text(encoding="utf-8"))
    assert [s["source_rel"] for s in man_y1["shots"]] == [rel_aug, rel_oct, rel_nov]
    assert med.probe_media(fp, str(data_root / pub_y1["video_rel"]))["duration"] > 6.0

    # 3. Failure preserves existing published artifacts and schedules retry_wait backoff
    orig_sha = hashlib.sha256(mp4_11.read_bytes()).hexdigest()
    orig_render = film_compiler.med.render_smooth_film_nas
    monkeypatch.setattr(film_compiler.med, "render_smooth_film_nas", lambda *_a, **_kw: None)
    failed_11 = dm.compile_film(spec_11, force=True, out_w=640, out_h=360)
    assert failed_11["status"] == "retry_wait"
    assert failed_11["attempts"] == 1
    assert failed_11["retry_at"] is not None and failed_11["retry_at"] > time.time()
    assert "film_render_incomplete" in (failed_11["last_error"] or "")
    # Existing published MP4 preserved intact!
    assert hashlib.sha256(mp4_11.read_bytes()).hexdigest() == orig_sha
    assert db.next_ready_film_production(now=time.time()) is None
    ready_later = db.next_ready_film_production(now=float(failed_11["retry_at"]) + 1.0)
    assert ready_later is not None and ready_later["film_id"] == "month_11m"

    monkeypatch.setattr(film_compiler.med, "render_smooth_film_nas", orig_render)
    recovered_11 = dm.compile_film(ready_later, force=False, out_w=640, out_h=360)
    assert recovered_11["status"] == "published"

    # 4. Sync allowlist + HTTP human review (/api/ctl/film/review approve, reject, retry w/ exclude_shots)
    sync = lc.ArchiveCloudSync(data_root, db, cfg)
    pre_approve_rels = {r for r, _p, _st in sync._iter_curated_files()}
    assert pub_11["video_rel"] not in pre_approve_rels
    assert not any(r.startswith("v2/t0_keepers") for r in pre_approve_rels)

    srv = web_server.create_server("127.0.0.1", 0, data_root, db, dm, cfg)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        films_get = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/films", timeout=5).read().decode("utf-8"))
        assert len(films_get["films"]) == 2
        assert any(f["film_id"] == "month_11m" and len(f["shots"]) == 2 for f in films_get["films"])

        req_app = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ctl/film/review",
            data=json.dumps({"film_id": "month_11m", "action": "approve", "note": "画质清晰通过"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_app = json.loads(urllib.request.urlopen(req_app, timeout=5).read().decode("utf-8"))
        assert res_app["film"]["review_status"] == "approved"
        assert json.loads(man_11_p.read_text(encoding="utf-8"))["review_status"] == "approved"

        post_approve_rels = {r for r, _p, _st in sync._iter_curated_files()}
        assert pub_11["video_rel"] in post_approve_rels
        assert pub_11["manifest_rel"] in post_approve_rels
        assert pub_y1["video_rel"] not in post_approve_rels  # annual still pending_review

        # Request retry excluding the first shot -> re-compile omits rel_oct
        req_retry = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ctl/film/review",
            data=json.dumps({
                "film_id": "month_11m",
                "action": "retry",
                "note": "移除十月旧镜头重编",
                "exclude_shots": [rel_oct],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res_retry = json.loads(urllib.request.urlopen(req_retry, timeout=5).read().decode("utf-8"))
        assert res_retry["film"]["status"] == "queued"

        job_retry = db.next_ready_film_production()
        assert job_retry is not None and job_retry["film_id"] == "month_11m"
        recompiled = dm.compile_film(job_retry, out_w=640, out_h=360)
        assert recompiled["status"] == "published"
        man_after_retry = json.loads((data_root / recompiled["manifest_rel"]).read_text(encoding="utf-8"))
        assert [s["source_rel"] for s in man_after_retry["shots"]] == [rel_nov]
    finally:
        srv.shutdown()
        srv.server_close()


def test_legacy_v2_seed_cards_do_not_block_due_month_11m_and_p1_backlog_does_not_stall_cutoff_ready(tmp_path: Path):
    """Legacy V2 seed cards (ending 2026-11-11) must not suppress V3 month_11m on 2026-11-18."""
    cfg = _make_cfg(tmp_path)
    data_root = Path(cfg["project"]["data_root"])
    (data_root / "v2").mkdir(parents=True, exist_ok=True)
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    legacy_cards = [{"month": "2025-12", "day": "2025-12-23", "card": "month_2025-12_2025-12-23.jpg"}]
    for m_idx, d_iso in enumerate(
        [
            "2026-01-06", "2026-02-19", "2026-03-27", "2026-04-01", "2026-05-29",
            "2026-06-18", "2026-07-12", "2026-08-23", "2026-09-16", "2026-10-25",
            "2026-11-11",
        ],
        start=1,
    ):
        legacy_cards.append({"month": d_iso[:7], "day": d_iso, "card": f"month_{d_iso[:7]}_{d_iso}.jpg"})
    (data_root / "v2" / "deliverables.json").write_text(json.dumps({"monthly_cards": legacy_cards}), encoding="utf-8")

    # Historical P1 backlog still queued + 2026-11-17 still has 1 queued file
    db.upsert_discovered_files([
        {
            "source_rel": "aqara_video/living/20261020/p1_queued.mp4",
            "source_root": "cctv",
            "day": "2026-10-20",
            "camera": "cam_living",
            "wall_ts": "2026-10-20T10:00:00",
            "size": 1000,
            "mtime": 1700000000.0,
        },
        {
            "source_rel": "aqara_video/living/20261117/p0_last.mp4",
            "source_root": "cctv",
            "day": "2026-11-17",
            "camera": "cam_living",
            "wall_ts": "2026-11-17T23:55:00",
            "size": 1000,
            "mtime": 1700000000.0,
        },
    ])
    db.upsert_milestone({
        "id": "ms_11_demo",
        "day": "2026-11-10",
        "camera": "cam_living",
        "category": "toy_play",
        "source_rel": "aqara_video/living/20261110/demo.mp4",
        "start_s": 0.0,
        "end_s": 4.0,
        "caption_zh": "11月宝宝抓握玩具",
        "score": 4.9,
        "review_status": "approved",
    })

    sch_wait = lc.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 18), data_root, compile_day=18, db=db)
    assert sch_wait["due_today"] is True
    assert sch_wait["next_months"] == 11
    assert sch_wait["next_compile_date"] == "2026-11-18"
    assert sch_wait["cutoff_day_queued"] == 1
    assert sch_wait["cutoff_day_ready"] is False
    assert [d["film_id"] for d in sch_wait["due_productions"]] == ["month_11m"]
    assert sch_wait["due_productions"][0]["ready"] is False

    # Finishing 2026-11-17 unblocks month_11m even while October P1 backfill remains queued
    db.mark_file_done("aqara_video/living/20261117/p0_last.mp4", {"status": "done", "duration": 10.0, "width": 640, "height": 360})
    sch_go = lc.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 18), data_root, compile_day=18, db=db)
    assert sch_go["due_today"] is True
    assert sch_go["cutoff_day_queued"] == 0
    assert sch_go["cutoff_day_ready"] is True
    assert [d["film_id"] for d in sch_go["due_productions"]] == ["month_11m"]
    assert sch_go["due_productions"][0]["ready"] is True


def test_seasonal_and_camera_chronicle_specs_and_director_pyramid(tmp_path: Path, monkeypatch):
    """Verify seasonal/chronicle catalogs, Tier-A daily_script pyramid, daily_care filter, HIGH director screenplay, and UI compile endpoint."""
    ff, fp = _resolve_ff_pair()
    cfg = _make_cfg(tmp_path)
    cfg["cameras"]["cam_living"]["role"] = "living_room"
    cfg["cameras"]["cam_bed"]["role"] = "bedside_crib"
    cfg["vlm"] = {
        "tiers": {"fast": ["p_dir"], "pro": ["p_dir"], "director": ["p_dir"]},
        "providers": {"p_dir": {"model": "gemini-4-carbon", "secret_file": "s"}},
    }
    data_root = Path(cfg["project"]["data_root"])
    cctv_root = Path(cfg["project"]["cctv_root"])
    vault_root = Path(cfg["project"]["vault_root"])
    (data_root / "v2").mkdir(parents=True, exist_ok=True)
    cctv_root.mkdir(parents=True, exist_ok=True)
    vault_root.mkdir(parents=True, exist_ok=True)
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon_mod.med, "resolve_font", lambda *_a, **_kw: None)
    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *_a, **_kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *_a, **_kw: object())

    # 1. Verify seasonal + camera_chronicle spec builders & resolver
    sp_aut = lc.build_seasonal_film_spec(dt.date(2025, 12, 17), 4, compile_day=18)
    assert sp_aut["film_id"] == "season_04_autumn"
    assert sp_aut["period_start"] == "2026-09-17" and sp_aut["period_end"] == "2026-12-17"
    res_s = lc.resolve_film_spec_by_id("2025-12-17", "season_04_autumn", dt.date(2026, 11, 19))
    assert res_s["film_id"] == "season_04_autumn"
    res_ang = lc.resolve_film_spec_by_id("2025-12-17", "angle_bedside_crib", dt.date(2026, 11, 19))
    assert res_ang["kind"] == "camera_chronicle" and res_ang["angle"] == "bedside_crib"

    # 2. Seed Tier-A daily_script.json shots + speech milestone (with raw technical prefix) + routine daily_care (to filter out)
    rel_d1 = "aqara_video/living/20261114/091000.mp4"
    _make_sample_mp4(ff, cctv_root / rel_d1, color="teal", dur_s=6.0)
    db.upsert_discovered_files([{
        "source_rel": rel_d1, "source_root": "cctv", "day": "2026-11-14", "camera": "cam_living",
        "wall_ts": "2026-11-14T09:10:00", "size": 2000, "mtime": time.time() - 500,
    }])
    db.mark_file_done(rel_d1, {"status": "done", "duration": 6.0, "width": 640, "height": 360, "has_audio": True})
    day14_dir = data_root / "days" / "2026-11-14"
    day14_dir.mkdir(parents=True, exist_ok=True)
    (day14_dir / "daily_script.json").write_text(json.dumps({
        "day": "2026-11-14",
        "film_title_zh": "第333天",
        "shots": [{
            "shot_id": "S01",
            "source_rel": rel_d1,
            "start_s": 0.4,
            "dur_s": 4.8,
            "camera": "客厅看娃",
            "camera_id": "cam_living",
            "shot_scale": "closeup",
            "pan_style": "gentle_push",
            "emotion_tag": "joy",
            "source_kind": "laughter_peak",
            "boost_vocal": True,
            "cinematic_score": 4.8,
            "lyric_zh": "阳光斜照地垫，伸手抓起小响铃",
            "categories": ["toy_play"],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    rel_sp = "xiaomi_camera_videos/bed/20261116/180658.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_sp_baba", day="2026-11-16", camera="cam_bed", rel=rel_sp,
        wall_ts="2026-11-16T18:06:58", color="orange", category="speech",
        caption="纯童声 · 清晰「爸 (单音节)」（第 1.4 秒处）（基频 415.7 Hz · 信噪比 100.5× · 置信度 0.78） · 小手扶栏望向门边",
        score=5.0,
    )
    rel_care = "aqara_video/living/20261115/120000.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_care_drop", day="2026-11-15", camera="cam_living", rel=rel_care,
        wall_ts="2026-11-15T12:00:00", color="gray", category="daily_care",
        caption="卧床换尿布照护互动", score=4.4,
    )

    dm = GrowthDaemon(cfg, db, ff_bin=ff, fp_bin=fp)
    director_calls: list[tuple[str, str]] = []

    def fake_ask(tag: str, images: list[bytes], prompt: str, tier: str = "director"):
        director_calls.append((tier, tag))
        return {
            "_model": "gemini-4-carbon",
            "film_title_zh": "第11个月 · 清晨响铃与一声爸",
            "logline_zh": "这个月既会伸手摇响铃，也在傍晚清晰喊出一声爸。",
            "joy_summary_zh": "握着玩具摇出声时满脸亮晶晶",
            "growth_changes_zh": "小手抓握与主动开口都更稳当",
            "closing_epilogue_zh": "满十一月，每一步都算数。",
            "edit_list": [
                {"shot_id": "S01", "chapter_zh": "第一幕 · 初见本月模样", "trim_dur_s": 9.0, "shot_scale": "closeup", "transition": "chapter_dissolve", "lyric_zh": "晨光里伸手摇起心爱小响铃"},
                {"shot_id": "S02", "chapter_zh": "第三幕 · 欢笑与初语", "trim_dur_s": 4.0, "shot_scale": "medium", "transition": "warm_dissolve", "lyric_zh": "这一声清脆的「爸」被稳稳记住"},
            ],
        }

    monkeypatch.setattr(dm.vlm, "ask", fake_ask)

    pub_11 = dm.compile_film(lc.build_film_spec(dt.date(2025, 12, 17), 11, "monthly"), force=True, out_w=640, out_h=360)
    assert pub_11["status"] == "published"
    assert any(t == "director" and tag.startswith("film_script|month_11m|") for t, tag in director_calls)
    man_11 = json.loads((data_root / pub_11["manifest_rel"]).read_text(encoding="utf-8"))
    # Strict dur clamp against verified_dur_s (4.8s) even when director requested 9.0s, and daily_care excluded!
    shot_rels = [s["source_rel"] for s in man_11["shots"]]
    assert shot_rels == [rel_d1, rel_sp]
    assert rel_care not in shot_rels
    assert man_11["shots"][0]["category"] != "speech"
    assert man_11["shots"][1]["category"] == "speech"
    assert man_11["shots"][0]["duration_s"] <= 4.8
    assert man_11["story"]["film_title_zh"] == "第11个月 · 清晨响铃与一声爸"
    assert all("基频" not in s["caption_zh"] and "信噪比" not in s["caption_zh"] for s in man_11["shots"])

    # 3. Camera chronicle compile updates smooth_films[bedside_crib] with director_v3=True
    pub_bed = dm.compile_film(res_ang, force=True, out_w=640, out_h=360)
    assert pub_bed["status"] == "published"
    deliv = json.loads((data_root / "v2" / "deliverables.json").read_text(encoding="utf-8"))
    assert deliv["smooth_films"]["bedside_crib"]["director_v3"] is True
    assert deliv["smooth_films"]["bedside_crib"]["film_id"] == "angle_bedside_crib"

    # 4. Schedule exposes seasonal_catalog & camera_chronicles; HTTP /api/ctl/film/compile queues season_04_autumn
    sch = lc.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 19), data_root, compile_day=18, db=db)
    assert len(sch["seasonal_catalog"]) == 4 and any(s["film_id"] == "season_04_autumn" and s["has_footage"] for s in sch["seasonal_catalog"])
    assert len(sch["camera_chronicles"]) == 4

    srv = web_server.create_server("127.0.0.1", 0, data_root, db, dm, cfg)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ctl/film/compile",
            data=json.dumps({"film_id": "season_04_autumn", "force": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert res["queued"] is True and res["film"]["film_id"] == "season_04_autumn"
        films_api = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/films", timeout=5).read().decode("utf-8"))
        m11_api = next(f for f in films_api["films"] if f["film_id"] == "month_11m")
        assert m11_api["story"]["film_title_zh"] == "第11个月 · 清晨响铃与一声爸"
    finally:
        srv.shutdown()
        srv.server_close()


def test_cinema_polling_incremental_guard_and_range_206(tmp_path: Path) -> None:
    """Verify 5-second status polling diffs cinema signatures without tearing down playing videos and Range 206 works."""
    cfg = _make_cfg(tmp_path)
    data_root = Path(cfg["project"]["data_root"])
    (data_root / "v2" / "films").mkdir(parents=True, exist_ok=True)
    dummy_bytes = b"ftypisom" + bytes(range(64))
    (data_root / "v2" / "films" / "dummy.mp4").write_bytes(dummy_bytes)
    (data_root / "v2" / "deliverables.json").write_text(
        json.dumps({
            "smooth_films": {
                "living": {
                    "angle": "living",
                    "name": "客厅",
                    "file": "films/dummy.mp4",
                    "poster": "films/dummy.jpg",
                    "shots": 1,
                    "duration_s": 5.0,
                    "zoom_x": 1.3,
                }
            }
        }),
        encoding="utf-8",
    )
    db = ArchiveDB(data_root / "state" / "archive.db")
    dm = type("_StubDaemon", (), {"is_paused": staticmethod(lambda: False)})()
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, dm, cfg)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
        for token in (
            "_cinemaSigs",
            "_containerHasActiveVideo",
            "cinemaMainVideoSlot",
            "cinemaScheduleBanner",
            "renderCinema(false)",
            'preload="${posUrl ? \'none\' : \'metadata\'}"',
        ):
            assert token in html, f"missing expected guard token: {token}"

        status_block = html.split("function updateSystemStatus(", 1)[1].split("function renderCameraHealth(", 1)[0]
        assert "renderCinema(false);" in status_block
        assert "renderCinema();" not in status_block

        m_js = re.search(r"<script>\n(.*?)\n</script>", html, flags=re.S)
        assert m_js is not None
        node_bin = shutil.which("node")
        if node_bin:
            js_file = tmp_path / "inline_check.js"
            js_file.write_text(m_js.group(1), encoding="utf-8")
            chk = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True, timeout=15)
            assert chk.returncode == 0, f"JS syntax check failed: {chk.stderr}"

        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5).read().decode("utf-8"))
        assert isinstance(st.get("smooth_films"), list) and len(st["smooth_films"]) == 1

        req_rng = urllib.request.Request(
            f"http://127.0.0.1:{port}/v2/films/dummy.mp4",
            headers={"Range": "bytes=0-15"},
        )
        with urllib.request.urlopen(req_rng, timeout=5) as resp:
            assert resp.status == 206
            assert resp.headers.get("Content-Range") == f"bytes 0-15/{len(dummy_bytes)}"
            assert resp.read() == dummy_bytes[:16]
    finally:
        srv.shutdown()
        srv.server_close()


def test_camera_chronicle_queue_status_ui_and_rendering_protection(tmp_path: Path, monkeypatch) -> None:
    """Verify rendering-protection on re-enqueue, stale-rendering recovery, and live queue UI/API exposure."""
    ff, fp = _resolve_ff_pair()
    cfg = _make_cfg(tmp_path)
    data_root = Path(cfg["project"]["data_root"])
    (data_root / "v2").mkdir(parents=True, exist_ok=True)
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon_mod.med, "resolve_font", lambda *_a, **_kw: None)
    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *_a, **_kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *_a, **_kw: object())

    dm = GrowthDaemon(cfg, db, ff_bin=ff, fp_bin=fp)
    spec_lr = lc.build_camera_chronicle_spec(dt.date(2025, 12, 17), "living_room", dt.date(2026, 11, 19))
    db.enqueue_film_production(spec_lr, force=True)
    db.mark_film_rendering("angle_living_room")

    # 1. preserve_rendering=True must not clobber an active rendering job back to queued
    kept = db.enqueue_film_production(spec_lr, force=True, preserve_rendering=True)
    assert kept["status"] == "rendering"

    with dm._job_lock:
        dm._curator_status = "angle_living_room · 高思考导演剪辑"
    res_dup = dm.request_film_compile("angle_living_room", force=True)
    assert res_dup["queued"] is True
    assert res_dup["already_rendering"] is True
    assert res_dup["film"]["status"] == "rendering"
    assert res_dup["curator_status"] == "angle_living_room · 高思考导演剪辑"
    assert db.get_film_production("angle_living_room")["status"] == "rendering"

    # 2. Stale rendering rows recover cleanly back to queued on startup
    assert db.reset_stale_rendering_films() == 1
    assert db.reset_stale_rendering_films() == 0
    assert db.get_film_production("angle_living_room")["status"] == "queued"

    # 3. HTTP /api/films, /api/status, and / expose live curator_status, paused, and queue UI tokens
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, dm, cfg)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        films_api = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/films", timeout=5).read().decode("utf-8"))
        assert films_api["curator_status"] == "angle_living_room · 高思考导演剪辑"
        assert films_api["paused"] is False

        status_api = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5).read().decode("utf-8"))
        assert status_api["curator_status"] == "angle_living_room · 高思考导演剪辑"
        assert status_api["paused"] is False

        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
        for token in (
            "_filmQueueStatusUi",
            "showToast",
            "already_rendering",
            "已加入导演队列（排队中）",
            'id="toastBar"',
        ):
            assert token in html, f"missing expected queue UI token: {token}"

        m_js = re.search(r"<script>\n(.*?)\n</script>", html, flags=re.S)
        assert m_js is not None
        node_bin = shutil.which("node")
        if node_bin:
            js_file = tmp_path / "queue_ui_check.js"
            js_file.write_text(m_js.group(1), encoding="utf-8")
            chk = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True, timeout=15)
            assert chk.returncode == 0, f"JS syntax check failed: {chk.stderr}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_growth_chronicle_tier_d_scout_and_monthly_coverage(tmp_path: Path, monkeypatch) -> None:
    """Verify Tier-D cross-month anchor scouting, HIGH audit, monthly coverage guarantee, and hero playback selection."""
    ff, fp = _resolve_ff_pair()
    cfg = _make_cfg(tmp_path)
    cfg["vlm"] = {
        "tiers": {"fast": ["mock_fast"], "pro": ["mock_pro"], "director": ["mock_dir"]},
        "providers": {"mock_fast": {"kind": "openai", "model": "mock-fast"}},
    }
    data_root = Path(cfg["project"]["data_root"])
    cctv_root = Path(cfg["project"]["cctv_root"])
    (data_root / "v2").mkdir(parents=True, exist_ok=True)
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon_mod.med, "resolve_font", lambda *_a, **_kw: None)
    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *_a, **_kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *_a, **_kw: object())

    def _make_textured_mp4(out_mp4: Path, dur_s: float = 8.0) -> None:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                ff, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=25:d={dur_s}",
                "-f", "lavfi", "-i", f"sine=frequency=520:sample_rate=16000:duration={dur_s}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "64k", "-shortest", str(out_mp4),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"ffmpeg testsrc2 failed: {proc.stderr}"

    # Create unarchived raw clips in CCTV across 4 earlier months (NOT inserted into SQLite files!)
    unarchived_days = ["2025-12-23", "2026-03-27", "2026-07-12", "2026-10-25"]
    for d_s in unarchived_days:
        ymd = d_s.replace("-", "")
        _make_textured_mp4(cctv_root / "aqara_video" / "living" / ymd / "103000.mp4")

    # Seed deliverables monthly cards spanning winter/spring/summer/autumn + Nov file & speech milestone in DB
    (data_root / "v2" / "deliverables.json").write_text(
        json.dumps({
            "monthly_cards": [
                {"month": d[:7], "day": d, "card": f"card_{d}.jpg"}
                for d in unarchived_days
            ],
            "monthly_film": {"file": "monthly_growth_1to11m.mp4", "poster": "monthly_growth_1to11m.jpg", "shots": 12, "duration_s": 43.0},
        }),
        encoding="utf-8",
    )

    rel_nov = "aqara_video/living/20261117/110000.mp4"
    _seed_clip_milestone(
        ff, db, cctv_root, data_root,
        ms_id="ms_gc_nov", day="2026-11-17", camera="cam_living", rel=rel_nov,
        wall_ts="2026-11-17T11:00:00", color="gold", category="speech",
        caption="清脆叫出「爸」", score=4.9,
    )

    # Unlock all seasons in schedule using historical monthly_cards
    sch = lc.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 19), data_root, compile_day=18, db=db)
    assert len(sch["seasonal_catalog"]) == 4
    assert all(s["has_footage"] is True for s in sch["seasonal_catalog"])
    assert sch["growth_chronicle"]["film_id"] == "growth_chronicle"

    dm = GrowthDaemon(cfg, db, ff_bin=ff, fp_bin=fp)
    seen_batch_calls: list[tuple[str, str]] = []

    def fake_ask_batch(tasks, tier: str = "fast", progress_cb=None):
        out = {}
        for tid, ptag, *_rest in tasks:
            seen_batch_calls.append((tier, ptag))
            if str(ptag).startswith("film_scout|"):
                out[tid] = {
                    "child_clear": "yes",
                    "face_visibility": "clear_front",
                    "daily_care": "no",
                    "best_time_s": 4.0,
                    "excitement_score": 4.4,
                    "child_box": [0.2, 0.2, 0.8, 0.8],
                    "caption_zh": "在客厅抬头笑看玩具",
                }
            else:
                out[tid] = {
                    "usable_for_highlight": "yes",
                    "child_clear": "yes",
                    "face_visibility": "clear_front",
                    "daily_care": "no",
                    "best_in_rel_s": 0.4,
                    "best_out_rel_s": 5.8,
                    "cinematic_score": 4.6,
                    "shot_scale": "medium",
                    "pan_style": "gentle_push",
                    "emotion_tag": "joy",
                    "child_box": [0.2, 0.2, 0.8, 0.8],
                    "director_note_zh": "镜头清晰记录宝宝新成长",
                }
        return out

    def fake_ask(tag: str, images: list[bytes], prompt: str, tier: str = "director", **_kw):
        # Deliberately omit S02 (2026-03) in edit_list to test monthly coverage restoration!
        lines_m = re.findall(r'"shot_id":\s*"(S\d+)"', prompt)
        chosen_sids = [sid for sid in lines_m if sid != "S02"]
        return {
            "_model": "gemini-4-carbon",
            "film_title_zh": "一路长成小太阳",
            "logline_zh": "从初生仰卧到满十一月龄叫出爸，每个月都留下光影见证。",
            "joy_summary_zh": "抬头看见熟悉家人就眉眼弯弯",
            "growth_changes_zh": "从小小一团慢慢长到扶站张望",
            "closing_epilogue_zh": "岁岁平安，步步可爱。",
            "edit_list": [
                {
                    "shot_id": sid,
                    "chapter_zh": "第一幕 · 初生温柔",
                    "trim_dur_s": 5.0,
                    "shot_scale": "medium",
                    "pan_style": "gentle_push",
                    "transition": "warm_dissolve",
                    "lyric_zh": f"珍贵的月龄成长足迹 {sid}",
                }
                for sid in chosen_sids
            ],
        }

    monkeypatch.setattr(dm.vlm, "ask_batch", fake_ask_batch)
    monkeypatch.setattr(dm.vlm, "ask", fake_ask)

    spec_gc = lc.resolve_film_spec_by_id("2025-12-17", "growth_chronicle", dt.date(2026, 11, 19))
    pub_gc = dm.compile_film(spec_gc, force=True, out_w=640, out_h=360)
    assert pub_gc["status"] == "published", f"failed compile: {pub_gc}"
    assert any(t == "fast" and ptag.startswith("film_scout|") for t, ptag in seen_batch_calls)
    assert any(t == "director" and ptag.startswith("film_edit|") for t, ptag in seen_batch_calls)

    man_gc = json.loads((data_root / pub_gc["manifest_rel"]).read_text(encoding="utf-8"))
    shot_months = [s["day"][:7] for s in man_gc["shots"]]
    assert set(shot_months) == {"2025-12", "2026-03", "2026-07", "2026-10", "2026-11"}
    assert shot_months == sorted(shot_months)

    deliv = json.loads((data_root / "v2" / "deliverables.json").read_text(encoding="utf-8"))
    assert deliv["growth_chronicle"]["film_id"] == "growth_chronicle"
    assert deliv["growth_chronicle"]["shots"] == len(man_gc["shots"])

    # Verify web UI favors growth_chronicle over single-month fallback and passes JS check
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, dm, cfg)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
        assert "f.kind === 'growth_chronicle'" in html
        assert "cinemaMainMeta" in html
        m_js = re.search(r"<script>\n(.*?)\n</script>", html, flags=re.S)
        assert m_js is not None
        node_bin = shutil.which("node")
        if node_bin:
            js_file = tmp_path / "gc_ui_check.js"
            js_file.write_text(m_js.group(1), encoding="utf-8")
            chk = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True, timeout=15)
            assert chk.returncode == 0, f"JS syntax check failed: {chk.stderr}"
    finally:
        srv.shutdown()
        srv.server_close()
