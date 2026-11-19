"""Unit tests for NAS Unattended Growth Archive V3 modules."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

NAS_PKG_ROOT = Path(__file__).resolve().parents[1] / "nas-growth-v3"
if str(NAS_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(NAS_PKG_ROOT))

from nas_growth import asr_worker, cloud_vlm, daemon as daemon_mod, detectors, lifecycle, media_nas, web_server  # noqa: E402
from nas_growth.daemon import GrowthDaemon  # noqa: E402
from nas_growth.db import ArchiveDB  # noqa: E402


def _v3_config_path() -> Path:
    local_cfg = NAS_PKG_ROOT / "config.toml"
    return local_cfg if local_cfg.is_file() else (NAS_PKG_ROOT / "config.example.toml")


def test_gemini_box_2d_normalization_and_secret_redaction():
    raw = '```json\n{"child_clear":"yes","box_2d":[500, 125, 875, 375],"caption_zh":"站立玩耍"}\n```'
    parsed = cloud_vlm.parse_json_obj(raw)
    assert parsed["child_box"] == [0.125, 0.5, 0.375, 0.875]

    fallback_raw = '{"child_clear":"yes","child_box":[0.2,0.1,0.6,0.8]}'
    parsed2 = cloud_vlm.parse_json_obj(fallback_raw)
    assert parsed2["child_box"] == [0.2, 0.1, 0.6, 0.8]

    leaked = "Error with AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz12345 and Bearer sk-abcdef12345"
    red = cloud_vlm.redact_secrets(leaked)
    assert "AIza" not in red and "sk-abcdef" not in red


def test_bedside_privacy_gate():
    bed_cfg = {"role": "bedside_crib", "day_hours": [8, 20]}
    assert cloud_vlm.is_cloud_allowed(bed_cfg, "2026-11-17T14:30:00") is True
    assert cloud_vlm.is_cloud_allowed(bed_cfg, "2026-11-17T22:15:00") is False
    assert cloud_vlm.is_cloud_allowed(bed_cfg, "2026-11-17T03:00:00") is False
    assert cloud_vlm.is_cloud_allowed(bed_cfg, None) is False
    assert cloud_vlm.is_cloud_allowed({"role": "kitchen_4k"}, "2026-11-17T22:15:00") is True


def test_db_p0_vs_p1_ordering_and_vlm_cache(tmp_path: Path):
    db = ArchiveDB(tmp_path / "state.sqlite3")
    db.upsert_discovered_files([
        {"source_rel": "a/20261116/100000.mp4", "day": "2026-11-16", "camera": "cam1", "wall_ts": "2026-11-16T10:00:00", "size": 1000, "mtime": 1.0},
        {"source_rel": "a/20261115/100000.mp4", "day": "2026-11-15", "camera": "cam1", "wall_ts": "2026-11-15T10:00:00", "size": 1000, "mtime": 1.0},
        {"source_rel": "a/20261117/100000.mp4", "day": "2026-11-17", "camera": "cam1", "wall_ts": "2026-11-17T10:00:00", "size": 1000, "mtime": 1.0},
    ])
    nxt = db.get_next_incomplete_historical_day({"2026-11-18", "2026-11-17"}, "2025-12-17", "2026-11-16")
    assert nxt == "2026-11-16"

    db.put_vlm_cache("k1", "fast", "gemini_hydrogen", "gemini-4-hydrogen", "tag", {"child_clear": "yes", "_internal": 1}, 1.2)
    got = db.get_vlm_cache("k1")
    assert got["child_clear"] == "yes"
    assert "_internal" not in got
    assert got["_model"] == "gemini-4-hydrogen"


def test_kinship_regexes_mama_and_papa():
    for s in ("妈妈", "妈妈呀", "麻麻", "妈咪"):
        assert asr_worker.REDUPLICATED_MA_RE.match(s)
    for s in ("妈", "媽", "阿嬷", "妈呀"):
        assert asr_worker.SINGLE_MA_RE.match(s)
        assert not asr_worker.REDUPLICATED_MA_RE.match(s)
    for s in ("爸爸", "粑粑"):
        assert asr_worker.REDUPLICATED_BA_RE.match(s)
    for s in ("爸", "吧", "爸呀"):
        assert asr_worker.SINGLE_BA_RE.match(s)


def test_bedside_static_mask_and_crop():
    boxes = [[(0.85, 0.2, 0.2, 0.4, 0.4)] for _ in range(15)]
    bmot = np.full(15, 0.002, dtype=np.float32)
    mask = detectors.compute_bedside_static_mask(boxes, bmot, min_run_s=12)
    assert bool(mask.all())

    cw, ch, cx, cy = media_nas.crop_16_9(2688, 1520, [0.4, 0.4, 0.5, 0.55], zoom_strength=2.0)
    assert cw == 1344 and ch == 756 and cw % 2 == 0 and ch % 2 == 0

    # With min_native_w=1920 floor enabled, never crops narrower than 1920px native (1:1 crisp 1080p)
    cw2, ch2, _x2, _y2 = media_nas.crop_16_9(2688, 1520, [0.4, 0.4, 0.5, 0.55], zoom_strength=2.5, min_native_w=1920)
    assert cw2 == 1920 and ch2 == 1080
    cw3, ch3, _x3, _y3 = media_nas.crop_16_9(2304, 1296, [0.4, 0.4, 0.5, 0.55], zoom_strength=2.0, min_native_w=1920)
    assert cw3 == 1920 and ch3 == 1080


def test_web_server_range_and_security(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "state").mkdir(parents=True)
    (data_root / "state" / "secret_note.txt").write_text("no", encoding="utf-8")
    (data_root / "v2").mkdir(parents=True)
    vid = data_root / "v2" / "clip.mp4"
    vid.write_bytes(b"0123456789ABCDEF")
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    class StubDaemon:
        def is_paused(self): return False
        def set_paused(self, _v): pass

    srv = web_server.create_server("127.0.0.1", 0, data_root, db, StubDaemon(), {})
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health.json", timeout=3) as r:
            assert r.status == 200 and json.loads(r.read())["ok"] is True

        req = urllib.request.Request(f"http://127.0.0.1:{port}/v2/clip.mp4", headers={"Range": "bytes=4-9"})
        with urllib.request.urlopen(req, timeout=3) as r:
            assert r.status == 206
            assert r.headers.get("Content-Range") == "bytes 4-9/16"
            assert r.read() == b"456789"

        for forbidden in ("/state/secret_note.txt", "/state/growth_v3.sqlite3"):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{forbidden}", timeout=3)
                assert False, "Expected 403"
            except urllib.error.HTTPError as e:
                assert e.code == 403
    finally:
        srv.shutdown()
        srv.server_close()


def test_monitor_page_and_api(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "state").mkdir(parents=True)
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    db.upsert_discovered_files([{
        "source_rel": "aqara_video/cam1/20261117/090000.mp4",
        "source_root": "cctv",
        "day": "2026-11-17",
        "camera": "cam1",
        "wall_ts": "2026-11-17T09:00:00",
        "size": 5000000,
        "mtime": 1700000000.0,
    }])
    db.mark_file_done("aqara_video/cam1/20261117/090000.mp4", {
        "duration": 10.0, "width": 1920, "height": 1080, "has_audio": True,
        "active_s": 5, "person_s": 5, "t1_bytes": 500000, "t2_bytes": 200000,
    })
    db.record_vlm_result("gemini_hydrogen", "gemini-4-hydrogen", True, 3.2, usage={"input_tokens": 520, "output_tokens": 180, "total_tokens": 700})
    db.put_vlm_cache("k_ph", "fast", "gemini_hydrogen", "gemini-4-hydrogen", "ph|aqara_video/cam1/20261117/090000.mp4|1.0", {"child_clear": "yes", "photo_score": 4, "caption_zh": "宝宝玩积木"}, 3.2, usage={"input_tokens": 520, "output_tokens": 180, "total_tokens": 700})
    db.set_state("active_job", {"phase": "P0_LIVE", "day": "2026-11-17", "step": "并发视频分析", "detail": "测试进度", "host_mem_kib": 7000000})

    mon_data = db.get_live_monitor_data()
    assert mon_data["totals"]["done_files"] == 1
    assert mon_data["token_totals"] == {"input_tokens": 520, "output_tokens": 180, "total_tokens": 700}
    assert len(mon_data["day_queues"]) == 1
    assert len(mon_data["camera_queues"]) == 1
    assert len(mon_data["recent_files"]) == 1
    assert mon_data["recent_vlm"][0]["caption_zh"] == "宝宝玩积木"
    assert mon_data["recent_vlm"][0]["total_tokens"] == 700

    class StubDaemon:
        def is_paused(self): return False
        def set_paused(self, _v): pass

    cfg = {"cameras": {"cam1": {"name": "客厅测试"}}}
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, StubDaemon(), cfg)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as r:
            assert r.status == 200
            st_payload = json.loads(r.read().decode("utf-8"))
            assert st_payload["token_totals"]["total_tokens"] == 700
            assert len(st_payload["vlm_fleet"]) == 1

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/monitor", timeout=3) as r:
            assert r.status == 200
            payload = json.loads(r.read().decode("utf-8"))
            assert payload["camera_names"]["cam1"] == "客厅测试"
            assert payload["state"]["active_job"]["step"] == "并发视频分析"
            assert payload["token_totals"]["total_tokens"] == 700
            assert payload["vlm_fleet"][0]["model"] == "gemini-4-hydrogen"

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/monitor", timeout=3) as r:
            assert r.status == 200
            html_mon = r.read().decode("utf-8")
            assert "实时进度与工作监控大屏" in html_mon and "客厅测试" in html_mon and "kTok" in html_mon
            assert "vlmFleetGrid" in html_mon

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as r:
            assert r.status == 200
            idx_html = r.read().decode("utf-8")
            assert "/monitor" in idx_html and "opsTokTot" in idx_html
    finally:
        srv.shutdown()
        srv.server_close()


def _make_test_daemon(tmp_path: Path, monkeypatch) -> tuple[GrowthDaemon, Path, Path]:
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    src_rel = "aqara_video/cam1/20261117/090000.mp4"
    src = cctv_root / src_rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"DUMMY_MP4_DATA")

    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *a, **kw: False)
    monkeypatch.setattr(daemon_mod.med, "concat_mp4_list", lambda *a, **kw: False)
    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", lambda *a, **kw: [b"FAKE_JPEG"])

    cfg = {
        "project": {"cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {"asr_min_host_mem_available_kib": 999_999_999},
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False},
        "cameras": {
            "cam1": {
                "name": "宝宝房顶部", "family": "aqara", "relative_root": "aqara_video/cam1",
                "enabled": True, "role": "bedroom_ceiling", "zoom_factor": 2.0,
                "daytime_photos": True, "milestone_watch": False,
            },
        },
        "vlm": {"tiers": {"fast": ["g"], "pro": ["g"]}, "providers": {}},
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    db.upsert_discovered_files([{
        "source_rel": src_rel, "source_root": "cctv", "day": "2026-11-17",
        "camera": "cam1", "wall_ts": "2026-11-17T09:00:00", "size": 5000000, "mtime": 1.0,
    }])
    db.mark_file_done(src_rel, {
        "duration": 10.0, "width": 1920, "height": 1080, "has_audio": False,
        "active_s": 5, "person_s": 5, "t1_bytes": 500, "t2_bytes": 200,
        "candidates": [{"t": 2.0, "score": 0.9, "box": [0.2, 0.2, 0.6, 0.8], "sharp": 95.0, "sat": 40.0, "luma": 130.0, "motion": 0.05}],
        "bursts": [],
    })

    day_dir = data_root / "days" / "2026-11-17"
    pdir = day_dir / "photos"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "full_old.jpg").write_bytes(b"OLD_FULL")
    (pdir / "framed_old.jpg").write_bytes(b"OLD_FRAMED")
    old_entry = {
        "full": "days/2026-11-17/photos/full_old.jpg",
        "framed": "days/2026-11-17/photos/framed_old.jpg",
        "camera": "宝宝房顶部", "zoom_x": 2.0, "time": "09:00:02", "score": 5, "caption_zh": "上一轮保留照片",
    }
    (day_dir / "summary.json").write_text(json.dumps({"day": "2026-11-17", "photos": [old_entry]}, ensure_ascii=False), encoding="utf-8")

    dm = GrowthDaemon(cfg, db)
    return dm, data_root, day_dir


def test_finalize_day_preserves_photos_when_all_vlm_fail(tmp_path: Path, monkeypatch):
    dm, _data_root, day_dir = _make_test_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(dm.vlm, "ask", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("All VLM providers unavailable")))

    summary = dm.finalize_day("2026-11-17", "P0_LIVE", final=False)
    pdir = day_dir / "photos"
    assert (pdir / "full_old.jpg").read_bytes() == b"OLD_FULL"
    assert (pdir / "framed_old.jpg").read_bytes() == b"OLD_FRAMED"
    assert len(summary["photos"]) == 1
    assert summary["photos"][0]["full"] == "days/2026-11-17/photos/full_old.jpg"
    disk_sum = json.loads((day_dir / "summary.json").read_text(encoding="utf-8"))
    assert disk_sum["photos"] == summary["photos"]


def test_finalize_day_preserves_photos_on_interrupted_generation_and_swaps_on_success(tmp_path: Path, monkeypatch):
    dm, _data_root, day_dir = _make_test_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(dm.vlm, "ask", lambda *a, **kw: {
        "child_clear": "yes", "face_or_expression_clear": "yes", "composition_good": "yes",
        "osd_box_on_child": "no", "adult_back_or_awkward": "no", "private_care": "no",
        "photo_score": 5, "child_box": [0.2, 0.2, 0.6, 0.8], "caption_zh": "宝宝正面开心笑",
    })

    def broken_save_pair(_ff, _src, _t, _w, _h, _box, _z, full_out: Path, _framed_out: Path, **_kw) -> bool:
        full_out.write_bytes(b"PARTIAL_FULL")
        raise OSError("disk/ffmpeg interrupted")

    monkeypatch.setattr(daemon_mod.med, "save_photo_pair", broken_save_pair)
    sum_err = dm.finalize_day("2026-11-17", "P0_LIVE", final=False)
    pdir = day_dir / "photos"
    assert (pdir / "full_old.jpg").read_bytes() == b"OLD_FULL"
    assert (pdir / "framed_old.jpg").read_bytes() == b"OLD_FRAMED"
    assert list(day_dir.glob(".photos_staging_*")) == []
    assert sum_err["photos"][0]["full"] == "days/2026-11-17/photos/full_old.jpg"

    def good_save_pair(_ff, _src, _t, _w, _h, _box, _z, full_out: Path, framed_out: Path, **_kw) -> bool:
        full_out.write_bytes(b"NEW_FULL")
        framed_out.write_bytes(b"NEW_FRAMED")
        return True

    monkeypatch.setattr(daemon_mod.med, "save_photo_pair", good_save_pair)
    sum_ok = dm.finalize_day("2026-11-17", "P0_LIVE", final=False)
    assert not (pdir / "full_old.jpg").exists()
    assert not (pdir / "framed_old.jpg").exists()
    assert list(day_dir.glob(".photos_staging_*")) == []
    assert len(sum_ok["photos"]) == 1
    assert sum_ok["photos"][0]["caption_zh"] == "宝宝正面开心笑"
    assert (pdir / Path(sum_ok["photos"][0]["full"]).name).read_bytes() == b"NEW_FULL"
    assert (pdir / Path(sum_ok["photos"][0]["framed"]).name).read_bytes() == b"NEW_FRAMED"


def test_probe_media_pts_jump_clamp(monkeypatch):
    class StubProc:
        returncode = 0
        stdout = json.dumps({
            "streams": [
                {"codec_type": "audio", "codec_name": "aac", "nb_frames": "1"},
                {"codec_type": "video", "codec_name": "h264", "width": 2688, "height": 1520, "r_frame_rate": "20/1", "nb_frames": "79"},
            ],
            "format": {"duration": "2722.317000"},
        })

    monkeypatch.setattr(media_nas.subprocess, "run", lambda *a, **kw: StubProc())
    info = media_nas.probe_media("ffprobe", "074003.mp4")
    assert info["pts_jump"] is True
    assert info["duration"] == 3.95
    assert info["has_audio"] is False


def test_daily_care_preserved_and_safety_block_local_fallback(tmp_path: Path, monkeypatch):
    # Part A: If an upstream VLM blocks a bath/diaper frame for SAFETY, do not retry/cool down and fall back locally
    db = ArchiveDB(tmp_path / "vlm.sqlite3")
    cfg_vlm = {
        "vlm": {
            "request_timeout_seconds": 10,
            "cooldown_seconds": 900,
            "tiers": {"pro": ["gem_primary", "oai_backup"]},
            "providers": {
                "gem_primary": {"kind": "gemini", "model": "gemini-4-carbon", "base_url": "https://example.invalid/v1beta", "secret_file": "/sec/g"},
                "oai_backup": {"kind": "openai", "model": "gpt-6.1-sol", "base_url": "https://example.invalid/v1", "secret_file": "/sec/o"},
            },
        }
    }
    router = cloud_vlm.CloudVLMRouter(cfg_vlm, db, secret_overrides={"/sec/g": "k1", "/sec/o": "k2"})

    class StubResp:
        def raise_for_status(self): pass
        def json(self): return {"promptFeedback": {"blockReason": "SAFETY"}}

    def fake_post(url, **kw):
        if "gpt" in url or "chat/completions" in url:
            raise AssertionError("Should not forward blocked care frame to secondary cloud")
        return StubResp()

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", lambda self, url, **kw: fake_post(url, **kw))
    ans = router.ask("ms|clip|1.0", [b"JPG"], cloud_vlm.PROMPT_MILESTONE, tier="pro")
    assert ans["review_status"] == "pending_review"
    assert ans["vlm_refused"] is True
    assert ans["child_clear"] == "pending_review"
    assert ans["categories"] == []
    assert ans["suggested_category"] == "daily_care"
    assert ans["_model"] == "vlm_refusal_pending"
    assert db.get_dashboard_snapshot()["vlm_stats"] == []

    # Part B: daily_care milestone & daily highlight curation works seamlessly
    dm, data_root, _day_dir = _make_test_daemon(tmp_path / "dm", monkeypatch)
    dm.cfg["cameras"]["cam1"]["milestone_watch"] = True
    dm.cfg["cameras"]["cam1"]["daytime_photos"] = False
    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [b"FAKE_JPEG"] * len(times))
    monkeypatch.setattr(daemon_mod.med, "promote_t0_keeper", lambda _src, _dst: (True, "deadbeef"))
    monkeypatch.setattr(daemon_mod.med, "render_single_clip_with_badge", lambda *a, **kw: True)

    def fake_smooth(_ff, _shots, out_mp4: Path, *a, **kw):
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"MP4HIGHLIGHT")
        return {"file": out_mp4.name, "shots": 1, "duration_s": 5.0, "bytes": 12}

    monkeypatch.setattr(daemon_mod.med, "render_smooth_film_nas", fake_smooth)
    monkeypatch.setattr(dm.vlm, "ask", lambda *a, **kw: {
        "child_clear": "yes", "categories": ["daily_care"], "independence": "assisted",
        "excitement_score": 4, "child_box": [0.0, 0.0, 0.0, 0.0], "caption_zh": "给洗完澡的宝宝擦干身体穿衣服",
    })
    sum_out = dm.finalize_day("2026-11-17", "P0_LIVE", final=False)
    assert sum_out["milestones_count"] == 1
    assert sum_out["highlight_rel"] == "days/2026-11-17/highlight_20261117.mp4"
    ms_rows = dm.db.get_dashboard_snapshot()["milestones"]
    assert len(ms_rows) == 1 and ms_rows[0]["category"] == "daily_care"


def test_localized_privacy_mosaic_preserves_face_and_blurs_target():
    parsed = cloud_vlm.parse_json_obj('{"child_clear":"yes","privacy_exposed":"yes","privacy_box_2d":[600,400,850,600]}')
    assert parsed["privacy_box"] == [0.4, 0.6, 0.6, 0.85]

    grid = np.zeros((200, 300, 3), dtype=np.uint8)
    grid[0::2, 0::2] = 255
    grid[1::2, 1::2] = 255
    ro_raw = grid.tobytes()
    ro_frm = np.frombuffer(ro_raw, dtype=np.uint8).reshape(200, 300, 3)
    out = media_nas.apply_privacy_mosaic(ro_frm, [0.4, 0.6, 0.6, 0.85])

    assert np.array_equal(out[10:50, 10:50], grid[10:50, 10:50])
    assert float(out[135:155, 135:165].std()) < float(grid[135:155, 135:165].std()) * 0.25

    pb = daemon_mod._resolve_privacy_box({"local_care_fallback": "yes"}, [0.2, 0.2, 0.6, 0.8])
    assert pb is not None and 0.2 <= pb[0] < pb[2] <= 0.6 and 0.2 <= pb[1] < pb[3] <= 0.8


def test_lifecycle_source_retention_four_safety_gates(tmp_path: Path):
    cctv_root = tmp_path / "cctv"
    vault_root = tmp_path / "vault"
    data_root = tmp_path / "data"
    baidu_p = tmp_path / "baidu_status.json"
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    now_dt = dt.datetime(2026, 11, 18, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    old_mtime = (now_dt - dt.timedelta(days=120)).timestamp()
    recent_mtime = (now_dt - dt.timedelta(days=10)).timestamp()

    old_cctv = cctv_root / "aqara_video/cam1/20260720/090000.mp4"
    recent_cctv = cctv_root / "aqara_video/cam1/20261108/090000.mp4"
    old_vault = vault_root / "aqara_video/cam1/20260720/080000.mp4"
    for p, mt in ((old_cctv, old_mtime), (old_vault, old_mtime), (recent_cctv, recent_mtime)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RAW_VIDEO_DATA")
        os.utime(p, (mt, mt))

    # Valid T1/T2 outputs and DB done rows
    for day, rel, root_k in (
        ("2026-07-20", "aqara_video/cam1/20260720/090000.mp4", "cctv"),
        ("2026-07-20", "aqara_video/cam1/20260720/080000.mp4", "vault"),
        ("2026-11-08", "aqara_video/cam1/20261108/090000.mp4", "cctv"),
    ):
        db.upsert_discovered_files([{
            "source_rel": rel, "source_root": root_k, "day": day, "camera": "cam1",
            "wall_ts": f"{day}T09:00:00", "size": 14, "mtime": old_mtime,
        }])
        stem = Path(rel).stem
        t1_rel = f"days/{day}/t1/cam1/{stem}.mp4"
        t2_rel = f"days/{day}/t2/cam1/{stem}.mp4"
        (data_root / t1_rel).parent.mkdir(parents=True, exist_ok=True)
        (data_root / t2_rel).parent.mkdir(parents=True, exist_ok=True)
        (data_root / t1_rel).write_bytes(b"T1")
        (data_root / t2_rel).write_bytes(b"T2")
        db.mark_file_done(rel, {"duration": 10.0, "width": 1920, "height": 1080, "t1_rel": t1_rel, "t1_bytes": 2, "t2_rel": t2_rel, "t2_bytes": 2})

    sum_path = data_root / "days/2026-07-20/summary.json"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps({"day": "2026-07-20", "status": "preview"}), encoding="utf-8")

    cfg = {
        "project": {"timezone": "Asia/Shanghai", "birthday": "2025-12-17", "cctv_root": str(cctv_root), "data_root": str(data_root)},
        "scheduler": {"p1_end_day": "2025-12-17", "p1_start_day": "2026-11-16"},
        "lifecycle": {
            "source_retention_days": 90,
            "retention_mode": "enforce",
            "protect_vault_forever": True,
            "baidu_status_path": str(baidu_p),
        },
    }

    # Gate 1 blocked: Baidu initial backup not finished
    baidu_p.write_text(json.dumps({"phase": "uploading", "initial_complete": None, "uploaded_files_total": 10, "scanned_files": 100}), encoding="utf-8")
    db.set_state("p1_backfill_complete", True)
    r1 = lifecycle.run_retention_pass(db, data_root, cctv_root, cfg, now_dt=now_dt)
    assert r1["can_prune"] is False and r1["audit"]["last_deleted_files"] == 0
    assert old_cctv.exists() and old_vault.exists() and recent_cctv.exists()

    # Gate 2 blocked: Baidu finished, but P1 backfill not finished
    baidu_p.write_text(json.dumps({"phase": "completed", "initial_complete": "2026-11-18T01:00:00+08:00", "uploaded_files_total": 100, "scanned_files": 100}), encoding="utf-8")
    db.set_state("p1_backfill_complete", False)
    r2 = lifecycle.run_retention_pass(db, data_root, cctv_root, cfg, now_dt=now_dt)
    assert r2["can_prune"] is False and r2["audit"]["last_deleted_files"] == 0
    assert old_cctv.exists()

    # Gate 3 blocked: both complete, but summary status is only preview (not done)
    db.set_state("p1_backfill_complete", True)
    r3 = lifecycle.run_retention_pass(db, data_root, cctv_root, cfg, now_dt=now_dt)
    assert r3["can_prune"] is True and r3["eligible_files"] == 0 and r3["audit"]["last_deleted_files"] == 0
    assert old_cctv.exists()

    # All gates pass + observe_only mode -> counts eligible without deleting anything
    sum_path.write_text(json.dumps({"day": "2026-07-20", "status": "done"}), encoding="utf-8")
    cfg_obs = {**cfg, "lifecycle": {**cfg["lifecycle"], "retention_mode": "observe_only"}}
    r4 = lifecycle.run_retention_pass(db, data_root, cctv_root, cfg_obs, now_dt=now_dt)
    assert r4["can_prune"] is True and r4["eligible_files"] == 1 and r4["audit"]["last_deleted_files"] == 0
    assert old_cctv.exists() and old_vault.exists() and recent_cctv.exists()

    # Enforce mode -> deletes ONLY expired verified /cctv file; preserves <90d recent AND /vault forever
    r5 = lifecycle.run_retention_pass(db, data_root, cctv_root, cfg, now_dt=now_dt)
    assert r5["audit"]["last_deleted_files"] == 1
    assert not old_cctv.exists()
    assert old_vault.exists()
    assert recent_cctv.exists()


def test_redesigned_gallery_tabs_lightbox_and_milestones_api(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "state").mkdir(parents=True)
    (data_root / "day-20261108").mkdir(parents=True)
    (data_root / "day-20261108" / "highlight_20261108.mp4").write_bytes(b"MP4")
    (data_root / "v2" / "photos").mkdir(parents=True)
    (data_root / "v2" / "photos" / "full_1.jpg").write_bytes(b"JPG1")
    (data_root / "v2" / "photos" / "framed_1.jpg").write_bytes(b"JPG2")
    (data_root / "v2" / "deliverables.json").write_text(json.dumps({
        "photos": [{"full": "full_1.jpg", "framed": "framed_1.jpg", "camera": "客厅", "zoom_x": 2.0, "time": "10:00:00", "score": 5, "caption_zh": "笑脸"}],
        "highlights": [],
        "monthly_cards": [],
        "smooth_films": {},
    }, ensure_ascii=False), encoding="utf-8")
    baidu_p = tmp_path / "baidu.json"
    baidu_p.write_text(json.dumps({"phase": "uploading", "uploaded_files_total": 300, "scanned_files": 1000, "uploaded_bytes_total": 3000, "scanned_bytes": 10000}), encoding="utf-8")

    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    db.upsert_milestone({
        "id": "sp_1", "day": "2026-11-17", "camera": "cam1", "category": "speech",
        "source_rel": "s1.mp4", "start_s": 1.0, "end_s": 5.0,
        "caption_zh": "清晰叫「妈妈」", "speaker_note": "基频 360 Hz",
        "video_rel": "v2/highlights/sp_1.mp4", "poster_rel": "v2/highlights/sp_1.jpg",
    })
    db.upsert_milestone({
        "id": "dc_1", "day": "2026-11-17", "camera": "cam2", "category": "daily_care",
        "source_rel": "s2.mp4", "start_s": 2.0, "end_s": 6.0,
        "caption_zh": "洗完澡给宝宝擦干身体", "independence": "assisted",
        "video_rel": "v2/highlights/dc_1.mp4", "poster_rel": "v2/highlights/dc_1.jpg",
    })

    class StubDaemon:
        def is_paused(self): return False
        def set_paused(self, _v): pass

    cfg = {
        "project": {"birthday": "2025-12-17", "timezone": "Asia/Shanghai", "cctv_root": str(tmp_path / "cctv"), "data_root": str(data_root)},
        "cameras": {"cam1": {"name": "客厅"}, "cam2": {"name": "宝宝床头"}},
        "lifecycle": {"baidu_status_path": str(baidu_p), "source_retention_days": 90},
    }
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, StubDaemon(), cfg)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as r:
            body = r.read().decode("utf-8")
            assert r.status == 200
            for marker in ("时光成长馆", "view-home", "view-daily", "view-milestones", "view-cinema", "view-system", "lightbox", "opsPhase", "opsTokTot", "/monitor"):
                assert marker in body

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/milestones?category=daily_care", timeout=3) as r:
            ms = json.loads(r.read().decode("utf-8"))
            assert len(ms["items"]) == 1 and ms["items"][0]["category"] == "daily_care"
            assert ms["category_counts"]["speech"] == 1 and ms["category_counts"]["daily_care"] == 1

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/milestones?q=" + urllib.parse.quote("妈妈"), timeout=3) as r:
            ms_q = json.loads(r.read().decode("utf-8"))
            assert len(ms_q["items"]) == 1 and ms_q["items"][0]["id"] == "sp_1"

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/day/2026-11-08", timeout=3) as r:
            d1108 = json.loads(r.read().decode("utf-8"))
            assert d1108["highlight_rel"] == "day-20261108/highlight_20261108.mp4"
            assert len(d1108["photos"]) == 1
            assert d1108["age_info"]["months"] == 10 and d1108["age_info"]["days"] == 22

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as r:
            st = json.loads(r.read().decode("utf-8"))
            assert st["lifecycle"]["can_prune"] is False
            assert st["lifecycle"]["baidu_backup"]["pct"] == 30.0
            assert st["lifecycle"]["monthly_schedule"]["compile_day"] == 18
    finally:
        srv.shutdown()
        srv.server_close()


def test_monthly_schedule_compiles_on_18th_after_17th_recordings(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "v2").mkdir(parents=True)
    deliv_p = data_root / "v2" / "deliverables.json"
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    # 11 cards already built -> on 2026-11-18 next milestone shifts to 2026-12-18 (1st birthday, collecting through 2026-12-17)
    deliv_p.write_text(json.dumps({"monthly_cards": [{"month": f"m{i}"} for i in range(11)]}), encoding="utf-8")
    s1 = lifecycle.compute_monthly_schedule("2025-12-17", dt.date(2026, 11, 18), data_root, compile_day=18, db=db)
    assert s1["compile_day"] == 18
    assert s1["next_compile_date"] == "2026-12-18"
    assert s1["milestone_day_cutoff"] == "2026-12-17"
    assert s1["next_months"] == 12
    assert "周岁" in s1["next_title_zh"]

    # On 2026-12-18 before the 12th card is built, waits for 2026-12-17 queued files to finish first
    db.upsert_discovered_files([{
        "source_rel": "aqara_video/cam1/20261217/235000.mp4",
        "day": "2026-12-17",
        "camera": "cam1",
        "wall_ts": "2026-12-17T23:50:00",
        "size": 1000,
        "mtime": 1.0,
    }])
    s2 = lifecycle.compute_monthly_schedule("2025-12-17", dt.date(2026, 12, 18), data_root, compile_day=18, db=db)
    assert s2["next_compile_date"] == "2026-12-18"
    assert s2["due_today"] is True
    assert s2["cutoff_day_queued"] == 1
    assert s2["cutoff_day_ready"] is False
    assert "剩余 1 档" in s2["note_zh"]

    db.mark_file_done("aqara_video/cam1/20261217/235000.mp4", {"duration": 10.0, "width": 1920, "height": 1080})
    s3 = lifecycle.compute_monthly_schedule("2025-12-17", dt.date(2026, 12, 18), data_root, compile_day=18, db=db)
    assert s3["cutoff_day_queued"] == 0
    assert s3["cutoff_day_ready"] is True


def test_old_kitchen_xiaomi_minute_discovery_health_gating_and_kitchen_all_filter(tmp_path: Path, monkeypatch):
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    # 2026-08-18 10:15:30 Asia/Shanghai
    ts_utc = dt.datetime(2026, 8, 18, 10, 15, 30, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
    epoch_s = int(ts_utc)
    src = cctv_root / "xiaomi_camera_videos/50ec503b63bf/2026081810" / f"15M30S_{epoch_s}.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"X" * 8192)
    old_mt = time.time() - 7200
    os.utime(src, (old_mt, old_mt))

    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *a, **kw: False)
    monkeypatch.setattr(daemon_mod.med, "concat_mp4_list", lambda *a, **kw: False)
    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [b"FAKE_JPEG"] * len(times))

    cfg = {
        "project": {"birthday": "2025-12-17", "cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {"asr_min_host_mem_available_kib": 999_999_999, "closed_file_cooldown_seconds": 60},
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False, "stale_warning_days": 1.5},
        "cameras": {
            "50ec503b63bf": {
                "name": "厨房（旧小米）", "family": "xiaomi_minute", "relative_root": "xiaomi_camera_videos/50ec503b63bf",
                "enabled": True, "historical_archive": True, "role": "kitchen", "zoom_factor": 1.0,
                "t1_mode": "verified_child_only", "daytime_photos": False, "milestone_watch": False,
            },
            "lumi3.932ece08fac40c05": {
                "name": "厨房 G350", "family": "aqara", "relative_root": "aqara_video/lumi3.932ece08fac40c05",
                "enabled": True, "role": "kitchen_4k", "zoom_factor": 1.0,
            },
        },
        "vlm": {"tiers": {"fast": ["g"], "pro": ["g"]}, "providers": {}},
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    dm = GrowthDaemon(cfg, db)

    # 1. Health status must be 'historical' (no false stale_warning alarm)
    h_res = dm.refresh_camera_health()
    old_h = next(h for h in h_res["health"] if h["id"] == "50ec503b63bf")
    assert old_h["status"] == "historical"
    assert old_h["latest_ts"].startswith("2026-08-18T10:15:30")

    # 2. Discover day files inside hourly subdirs
    ins = dm.discover_day_files("2026-08-18")
    assert ins == 1
    qrows = db.get_queued_files_for_day("2026-08-18")
    assert len(qrows) == 1 and qrows[0]["camera"] == "50ec503b63bf"

    # 3. finalize_day triggers verified_child_only kitchen VLM gating on 50ec503b63bf
    rel = qrows[0]["source_rel"]
    db.mark_file_done(rel, {
        "duration": 60.0, "width": 1920, "height": 1088, "has_audio": True,
        "t1_mode": "verified_child_only", "t2_bytes": 100,
        "candidates": [{"t": 10.0, "score": 0.9, "box": [0.2, 0.2, 0.5, 0.6], "sharp": 90.0, "sat": 35.0, "luma": 120.0, "motion": 0.05}],
    })
    monkeypatch.setattr(dm.vlm, "ask", lambda *a, **kw: {"child_clear": "yes", "caption_zh": "旧厨房妈妈抱着宝宝"})
    encoded = []

    def fake_enc(ff, s_path, out_p: Path, *a, **kw):
        encoded.append(out_p)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"T1OK" * 400)
        return True, 6.0

    monkeypatch.setattr(daemon_mod.med, "encode_t1_segments", fake_enc)
    dm.finalize_day("2026-08-18", "P1_BACKFILL", final=True)
    assert len(encoded) == 1
    row_after = db.get_day_rows("2026-08-18")[0]
    assert row_after["t1_mode"] == "verified_child_confirmed" and row_after["t1_rel"]

    # 4. /api/milestones?camera=kitchen_all merges old Xiaomi kitchen + G350 kitchen
    db.upsert_milestone({
        "id": "kit_old", "day": "2026-08-18", "camera": "50ec503b63bf", "category": "kitchen_visit",
        "source_rel": rel, "start_s": 1.0, "end_s": 6.0, "caption_zh": "旧厨房画面",
        "video_rel": "v2/highlights/kit_old.mp4", "poster_rel": "v2/highlights/kit_old.jpg",
    })
    db.upsert_milestone({
        "id": "kit_new", "day": "2026-11-15", "camera": "lumi3.932ece08fac40c05", "category": "kitchen_visit",
        "source_rel": "n.mp4", "start_s": 1.0, "end_s": 6.0, "caption_zh": "新厨房 G350 画面",
        "video_rel": "v2/highlights/kit_new.mp4", "poster_rel": "v2/highlights/kit_new.jpg",
    })
    res_all = web_server._collect_milestones(data_root, db, cfg, camera="kitchen_all")
    assert {it["id"] for it in res_all["items"]} == {"kit_old", "kit_new"}


def _make_test_sa_json(project_id: str, client_email: str) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    return json.dumps({
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": "test-key-id",
        "private_key": pem,
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
    })


def _make_test_overrides(cfg: dict) -> dict[str, str]:
    sa1 = _make_test_sa_json("proj-one", "sa1@proj-one.iam.gserviceaccount.com")
    sa2 = _make_test_sa_json("proj-two", "sa2@proj-two.iam.gserviceaccount.com")
    overrides: dict[str, str] = {}
    for pc in cfg["vlm"]["providers"].values():
        sf = pc.get("secret_file")
        if not sf:
            continue
        if "gcp_vertex_sa_1" in sf:
            overrides[sf] = sa1
        elif "gcp_vertex_sa_2" in sf:
            overrides[sf] = sa2
        else:
            overrides[sf] = "sk-sp-dummy-test-key"
    return overrides


def test_parallel_vlm_load_balancing_hydrogen_and_qwen4_max(tmp_path: Path, monkeypatch):
    import tomllib
    from concurrent.futures import ThreadPoolExecutor

    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    assert cfg["scheduler"]["workers"] == 4
    assert cfg["scheduler"]["min_host_mem_available_kib"] == 4194304
    assert cfg["vlm"]["providers"]["qwen4_max"]["model"] == "qwen4-max"
    assert {"vertex_1_hydrogen", "vertex_2_hydrogen", "qwen4_max", "deepseek_v41_flash"}.issubset(
        set(cfg["vlm"]["parallel_tiers"]["fast"])
    )

    db = ArchiveDB(tmp_path / "vlm_par.sqlite3")
    overrides = _make_test_overrides(cfg)
    cfg_pair = json.loads(json.dumps(cfg))
    cfg_pair["vlm"]["parallel_tiers"]["fast"] = ["vertex_1_hydrogen", "qwen4_max"]
    router = cloud_vlm.CloudVLMRouter(cfg_pair, db, secret_overrides=overrides)

    hit_lock = threading.Lock()
    hits: dict[str, int] = {}
    fail_qwen = {"mode": None}

    class StubResp:
        def __init__(self, status_code: int, body: dict):
            self.status_code = status_code
            self._body = body
            self.text = json.dumps(body, ensure_ascii=False)

        def raise_for_status(self):
            if self.status_code >= 400:
                err = cloud_vlm.requests.HTTPError(f"HTTP {self.status_code}")
                err.response = self
                raise err

        def json(self):
            return self._body

    def fake_post(self_sess, url: str, **kw):
        time.sleep(0.05)
        payload = kw.get("json") or {}
        if "oauth2.googleapis.com/token" in url:
            return StubResp(200, {"access_token": "ya29.test-token", "expires_in": 3600, "token_type": "Bearer"})
        if "aiplatform.googleapis.com" in url or "generativelanguage.googleapis.com" in url:
            model_used = "gemini-4-hydrogen"
            with hit_lock:
                hits[model_used] = hits.get(model_used, 0) + 1
            ans = json.dumps({"child_clear": "yes", "photo_score": 5, "box_2d": [200, 200, 700, 700], "caption_zh": "开心笑"})
            return StubResp(200, {
                "candidates": [{"content": {"parts": [{"text": ans}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 80, "totalTokenCount": 580},
            })
        model_used = str(payload.get("model") or "unknown")
        if model_used == "qwen4-max" and fail_qwen["mode"] == "rate_limit":
            return StubResp(429, {"error": "rate_limited"})
        if model_used == "qwen4-max" and fail_qwen["mode"] == "quota":
            return StubResp(429, {"error": {"code": "insufficient_quota", "message": "Token quota exceeded"}})
        with hit_lock:
            hits[model_used] = hits.get(model_used, 0) + 1
        ans = json.dumps({"child_clear": "yes", "photo_score": 4, "box_2d": [200, 200, 700, 700], "caption_zh": "玩耍"})
        return StubResp(200, {
            "choices": [{"message": {"content": ans}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 480, "completion_tokens": 70, "total_tokens": 550},
        })

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(
            lambda i: router.ask(f"ph|clip_{i}|1.0", [f"JPG_{i}".encode()], cloud_vlm.PROMPT_PHOTO, tier="fast"),
            range(6),
        ))
    assert len(results) == 6
    assert hits.get("gemini-4-hydrogen", 0) >= 2
    assert hits.get("qwen4-max", 0) >= 2

    # Transient rate limit triggers short cooldown (<= 300s) and immediately fails over
    db_rl = ArchiveDB(tmp_path / "vlm_rl.sqlite3")
    cfg_qwen_first = json.loads(json.dumps(cfg))
    cfg_qwen_first["vlm"]["parallel_tiers"]["fast"] = ["qwen4_max", "vertex_1_hydrogen"]
    r_rl = cloud_vlm.CloudVLMRouter(cfg_qwen_first, db_rl, secret_overrides=overrides)
    fail_qwen["mode"] = "rate_limit"
    ans_rl = r_rl.ask("ph|clip_rl|1.0", [b"JPG_RL"], cloud_vlm.PROMPT_PHOTO, tier="fast")
    assert ans_rl["child_clear"] == "yes" and ans_rl["_model"] == "gemini-4-hydrogen"
    cd_rl = db_rl.get_vlm_cooldown("qwen4_max") - time.time()
    assert 30 < cd_rl <= 300

    # Quota exhaustion triggers long cooldown (> 3600s) and immediately fails over
    db_q = ArchiveDB(tmp_path / "vlm_q.sqlite3")
    r_q = cloud_vlm.CloudVLMRouter(cfg_qwen_first, db_q, secret_overrides=overrides)
    fail_qwen["mode"] = "quota"
    ans_q = r_q.ask("ph|clip_quota|1.0", [b"JPG_Q"], cloud_vlm.PROMPT_PHOTO, tier="fast")
    assert ans_q["child_clear"] == "yes" and ans_q["_model"] == "gemini-4-hydrogen"
    cd_q = db_q.get_vlm_cooldown("qwen4_max") - time.time()
    assert cd_q > 3600


def test_rolling_window_pulls_in_nov16_immediately_after_nov18_done(tmp_path: Path, monkeypatch):
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    old_mtime = time.time() - 7200.0
    for d_comp, ts in (("20261116", "090000"), ("20261116", "090100"), ("20261116", "090200"), ("20261115", "090000")):
        p = cctv_root / "aqara_video" / "cam1" / d_comp / f"{ts}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"X" * 8192)
        os.utime(p, (old_mtime, old_mtime))

    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *a, **kw: False)

    cfg = {
        "project": {"cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {
            "workers": 6,
            "max_heavy_workers": 2,
            "rolling_active_days": 2,
            "closed_file_cooldown_seconds": 900,
            "active_file_cooldown_seconds": 5400,
            "p1_start_day": "2026-11-16",
            "p1_end_day": "2025-12-17",
        },
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False},
        "cameras": {
            "cam1": {"name": "宝宝房顶部", "family": "aqara", "relative_root": "aqara_video/cam1", "enabled": True, "role": "bedroom_ceiling"},
        },
        "vlm": {"tiers": {"fast": ["g"], "pro": ["g"]}, "providers": {}},
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    dm = GrowthDaemon(cfg, db)

    # 2026-11-18 is completely done; 2026-11-17 still has queued files (including heavy >60MB files)
    db.upsert_discovered_files([
        {"source_rel": "aqara_video/cam1/20261118/080000.mp4", "day": "2026-11-18", "camera": "cam1", "wall_ts": "2026-11-18T08:00:00", "size": 5000, "mtime": old_mtime},
        {"source_rel": "aqara_video/cam1/20261117/090000.mp4", "day": "2026-11-17", "camera": "cam1", "wall_ts": "2026-11-17T09:00:00", "size": 5000, "mtime": old_mtime},
        {"source_rel": "aqara_video/cam1/20261117/090100.mp4", "day": "2026-11-17", "camera": "cam1", "wall_ts": "2026-11-17T09:01:00", "size": 5000, "mtime": old_mtime},
        {"source_rel": "aqara_video/cam1/20261117/090200.mp4", "day": "2026-11-17", "camera": "cam1", "wall_ts": "2026-11-17T09:02:00", "size": 5000, "mtime": old_mtime},
        {"source_rel": "aqara_video/cam1/20261117/090300.mp4", "day": "2026-11-17", "camera": "cam1", "wall_ts": "2026-11-17T09:03:00", "size": 5000, "mtime": old_mtime},
        {"source_rel": "xiaomi_camera_videos/B888/heavy1.mp4", "day": "2026-11-17", "camera": "B888", "wall_ts": "2026-11-17T10:00:00", "size": 134_000_000, "mtime": old_mtime},
        {"source_rel": "xiaomi_camera_videos/B888/heavy2.mp4", "day": "2026-11-17", "camera": "B888", "wall_ts": "2026-11-17T10:10:00", "size": 134_000_000, "mtime": old_mtime},
        {"source_rel": "xiaomi_camera_videos/B888/heavy3.mp4", "day": "2026-11-17", "camera": "B888", "wall_ts": "2026-11-17T10:20:00", "size": 134_000_000, "mtime": old_mtime},
        # Pre-indexed Aqara_Cloud_G3 days (12th, 11th) must NOT jump ahead of 16th/15th
        {"source_rel": "Aqara_Cloud_G3/camera_12.MP4", "day": "2026-11-12", "camera": "cam1", "wall_ts": "2026-11-12T10:00:00", "size": 5000, "mtime": old_mtime},
        {"source_rel": "Aqara_Cloud_G3/camera_11.MP4", "day": "2026-11-11", "camera": "cam1", "wall_ts": "2026-11-11T10:00:00", "size": 5000, "mtime": old_mtime},
    ])
    db.mark_file_done("aqara_video/cam1/20261118/080000.mp4", {"duration": 10.0, "width": 1920, "height": 1080})

    active_days, all_p1_done, _empty_done = dm.ensure_rolling_days(dt.date(2026, 11, 18))
    assert active_days == ["2026-11-17", "2026-11-16"]
    assert all_p1_done is False
    assert db.get_state("p1_backfill_complete") is False

    batch = dm.pick_next_batch_files(active_days, "2026-11-18", slots=6, inflight_rels=set(), inflight_heavy=0, max_heavy=2)
    assert len(batch) == 6
    batch_days = {r["day"] for r in batch}
    assert batch_days == {"2026-11-17", "2026-11-16"}
    assert sum(1 for r in batch if int(r.get("size") or 0) > 60_000_000) <= 2

    # Once 2026-11-17 drains, 2026-11-15 enters strictly next (not skipping to pre-indexed 2026-11-12)
    for r17 in db.get_day_rows("2026-11-17"):
        db.mark_file_done(r17["source_rel"], {"duration": 10.0, "width": 1920, "height": 1080})
    active_days_next, all_p1_done2, _ = dm.ensure_rolling_days(dt.date(2026, 11, 18))
    assert active_days_next == ["2026-11-16", "2026-11-15"]
    assert all_p1_done2 is False


def test_aqara_cloud_g3_window_late_backfill_and_reopen_done_day(tmp_path: Path, monkeypatch):
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    tz = dt.timezone(dt.timedelta(hours=8))
    old_mtime = time.time() - 7200.0
    fresh_mtime = time.time() - 30.0

    def _mk(p: Path, mt: float = old_mtime) -> Path:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"V" * 8192)
        os.utime(p, (mt, mt))
        return p

    # Existing LAN SMB clip on 2026-11-11 10:00:00
    _mk(cctv_root / "aqara_video/lumi1.54ef4482e358/20261111/100000.mp4")

    cb = cctv_root / "Aqara_Cloud_G3"
    ep11_overlap = int(dt.datetime(2026, 11, 11, 10, 0, 10, tzinfo=tz).timestamp() * 1000)
    ep11_ok = int(dt.datetime(2026, 11, 11, 18, 0, 0, tzinfo=tz).timestamp() * 1000)
    ep12_sub = int(dt.datetime(2026, 11, 12, 16, 20, 0, tzinfo=tz).timestamp() * 1000)
    ep10_oow = int(dt.datetime(2026, 11, 10, 12, 0, 0, tzinfo=tz).timestamp() * 1000)
    ep13_cool = int(dt.datetime(2026, 11, 13, 11, 0, 0, tzinfo=tz).timestamp() * 1000)

    _mk(cb / f"camera_{ep11_overlap}.MP4")        # duplicate within LAN window -> skip
    _mk(cb / f"camera_{ep11_ok}.MP4")             # valid in window -> discover
    _mk(cb / "1112" / f"camera_{ep12_sub} (1).mp4")  # subfolder + suffix -> discover
    _mk(cb / f"20261115_120000.mp4")              # explicit YYYYMMDD_HHMMSS -> discover
    _mk(cb / f"camera_{ep10_oow}.MP4")            # out of 11-11..11-17 window -> skip
    _mk(cb / f"camera_{ep13_cool}.MP4", mt=fresh_mtime)  # still cooling down -> skip
    _mk(cb / f"camera_{ep11_ok}.MP4.part")        # partial upload -> skip

    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *a, **kw: False)

    cfg = {
        "project": {"cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {"closed_file_cooldown_seconds": 900, "active_file_cooldown_seconds": 5400},
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False, "stale_warning_days": 5.0},
        "cameras": {
            "lumi1.54ef4482e358": {
                "name": "客厅", "family": "aqara", "relative_root": "aqara_video/lumi1.54ef4482e358",
                "cloud_backfill_dir": "Aqara_Cloud_G3",
                "cloud_backfill_start_day": "2026-11-11",
                "cloud_backfill_end_day": "2026-11-17",
                "enabled": True, "role": "living_room", "zoom_factor": 1.2,
            },
        },
        "vlm": {"tiers": {"fast": ["g"], "pro": ["g"]}, "providers": {}},
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    dm = GrowthDaemon(cfg, db)

    # Check camera health exposes cloud_backfill window info
    h_res = dm.refresh_camera_health()
    liv = h_res["health"][0]
    assert liv["cloud_backfill"]["ready_files"] == 4
    assert liv["cloud_backfill"]["cooling_files"] == 1
    assert liv["cloud_backfill"]["days"] == ["2026-11-11", "2026-11-12", "2026-11-13", "2026-11-15"]
    assert "Aqara_Cloud_G3" in liv["note"] and "11-11~11-17" in liv["note"]

    # Initial backfill sync discovers 2026-11-15, 2026-11-12, 2026-11-11 (not 2026-11-10 or cooling 2026-11-13)
    reopened_0 = dm.sync_cloud_backfill_window()
    assert reopened_0 == ["2026-11-11", "2026-11-12", "2026-11-15"]
    rows11 = db.get_day_rows("2026-11-11")
    rels11 = {r["source_rel"] for r in rows11}
    assert rels11 == {
        "aqara_video/lumi1.54ef4482e358/20261111/100000.mp4",
        f"Aqara_Cloud_G3/camera_{ep11_ok}.MP4",
    }
    assert {r["source_rel"] for r in db.get_day_rows("2026-11-12")} == {f"Aqara_Cloud_G3/1112/camera_{ep12_sub} (1).mp4"}
    assert {r["source_rel"] for r in db.get_day_rows("2026-11-15")} == {"Aqara_Cloud_G3/20261115_120000.mp4"}
    assert db.get_day_rows("2026-11-10") == []
    assert db.get_day_rows("2026-11-13") == []

    # Simulate 2026-11-12 finished ("done") & exhausted before user uploaded the rest of Nov 12
    for r in db.get_day_rows("2026-11-12"):
        db.mark_file_done(r["source_rel"], {"duration": 30.0, "width": 2304, "height": 1296})
    d12_sum_p = data_root / "days" / "2026-11-12" / "summary.json"
    d12_sum_p.parent.mkdir(parents=True, exist_ok=True)
    d12_sum_obj = {"day": "2026-11-12", "tier": "P1_BACKFILL", "status": "done", "photos": [], "timelapses": {}}
    d12_sum_p.write_text(json.dumps(d12_sum_obj, ensure_ascii=False), encoding="utf-8")
    db.upsert_day_summary("2026-11-12", "P1_BACKFILL", "done", d12_sum_obj)
    dm._exhausted_days.add("2026-11-12")

    # User drops another settled Nov 12 clip later
    ep12_late = int(dt.datetime(2026, 11, 12, 19, 45, 0, tzinfo=tz).timestamp() * 1000)
    _mk(cb / f"camera_{ep12_late}.MP4")

    reopened_1 = dm.sync_cloud_backfill_window()
    assert reopened_1 == ["2026-11-12"]
    assert "2026-11-12" not in dm._exhausted_days
    assert json.loads(d12_sum_p.read_text(encoding="utf-8"))["status"] == "preview"
    assert db.count_day_stats("2026-11-12")["queued"] == 1



def test_work_stealing_pool_deepseek_flash_and_carbon_exclusive_pro(tmp_path: Path, monkeypatch):
    import io
    import tomllib
    from PIL import Image

    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    ds_cfg = cfg["vlm"]["providers"]["deepseek_v41_flash"]
    assert ds_cfg["model"] == "DeepSeek-V4.1-Flash-1017"
    assert ds_cfg["base_url"] == "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
    assert ds_cfg["secret_file"] == "/run/secrets/qwen_api_key"
    assert cfg["vlm"]["parallel_tiers"]["pro"] == ["gpt_62_sol", "qwen4_max", "gemini_carbon", "vertex_1_carbon", "vertex_2_carbon"]
    configured_models = {pc.get("model") for pc in cfg["vlm"]["providers"].values()}
    assert "gemini-3.8-flash" not in configured_models
    assert "gemini-3.1-pro-preview" not in configured_models
    assert "gpt-5.6-sol" not in configured_models
    assert "gpt-6.1-sol" not in configured_models
    assert "kimi-k3.1-pro" not in configured_models
    assert cfg["vlm"]["providers"]["gpt_62_sol"]["model"] == "gpt-6.2-sol"
    assert cfg["vlm"]["providers"]["gpt_62_sol"]["fallback_models"] == ["gpt-6.1-sol", "gpt-6-sol"]
    assert cfg["vlm"]["providers"]["vertex_1_carbon"]["thinking_level"] == "LOW"
    fast_pool = cfg["vlm"]["parallel_tiers"]["fast"]
    assert "gemini_carbon" not in fast_pool and "vertex_1_carbon" not in fast_pool
    assert {
        "qwen4_max", "deepseek_v41_flash", "gpt_62_sol",
        "gemini_hydrogen", "vertex_1_hydrogen", "vertex_2_hydrogen",
    }.issubset(set(fast_pool))

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 180, 210)).save(buf, "JPEG")
    tiny_jpg = buf.getvalue()

    db = ArchiveDB(tmp_path / "ws_pool.sqlite3")
    overrides = _make_test_overrides(cfg)
    router = cloud_vlm.CloudVLMRouter(cfg, db, secret_overrides=overrides)

    lock = threading.Lock()
    counts: dict[str, int] = {}
    ds_payload_checks: list[tuple[int, bool]] = []
    gpt_quota_ok = {"ok": False}

    class StubResp:
        def __init__(self, status_code: int, body: dict):
            self.status_code = status_code
            self._body = body
            self.text = json.dumps(body, ensure_ascii=False)

        def raise_for_status(self):
            if self.status_code >= 400:
                err = cloud_vlm.requests.HTTPError(f"HTTP {self.status_code}")
                err.response = self
                raise err

        def json(self):
            return self._body

    def fake_post(self_sess, url: str, **kw):
        time.sleep(0.02)
        payload = kw.get("json") or {}
        if "oauth2.googleapis.com/token" in url:
            return StubResp(200, {"access_token": "ya29.test-ws-token", "expires_in": 3600, "token_type": "Bearer"})
        if "aiplatform.googleapis.com" in url or "generativelanguage.googleapis.com" in url:
            m = "gemini-4-carbon" if "gemini-4-carbon" in url else "gemini-4-hydrogen"
            with lock:
                counts[m] = counts.get(m, 0) + 1
            ans = json.dumps({"child_clear": "yes", "categories": ["toy_play"], "excitement_score": 4, "box_2d": [200, 200, 700, 700], "caption_zh": "开心玩耍"})
            return StubResp(200, {
                "candidates": [{"content": {"parts": [{"text": ans}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 50, "totalTokenCount": 350},
            })

        m = str(payload.get("model") or "")
        with lock:
            counts[m] = counts.get(m, 0) + 1
        if m == "gpt-6.2-sol" and not gpt_quota_ok["ok"]:
            return StubResp(429, {"error": {"code": "insufficient_quota", "message": "Account quota exhausted"}})
        if m == "DeepSeek-V4.1-Flash-1017":
            return StubResp(404, {"error": {"message": "Model DeepSeek-V4.1-Flash-1017 not found"}})
        if m == "deepseek-v4.1-flash-1017":
            img_parts = [p for p in payload["messages"][0]["content"] if p.get("type") == "image_url"]
            with lock:
                ds_payload_checks.append((len(img_parts), "temperature" not in payload))
        ans = json.dumps({"child_clear": "yes", "photo_score": 5, "box_2d": [150, 150, 750, 750], "caption_zh": "宝宝出镜"})
        return StubResp(200, {
            "choices": [{"message": {"content": ans}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 280, "completion_tokens": 45, "total_tokens": 325},
        })

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)

    # 1. Fast work-stealing batch: gpt-6.2-sol exits once quota fails (at most initial worker_slots in-flight),
    #    its tasks are re-queued & completed; DeepSeek-V4.1-Flash-1017 auto-resolves lowercase ID, stitches 3 frames -> 1 collage,
    #    omits temperature; gemini-4-carbon NEVER steals fast tasks.
    fast_tasks = [
        (i, f"kit|clip_{i}|1.0", [tiny_jpg, tiny_jpg, tiny_jpg], cloud_vlm.PROMPT_KITCHEN_CHILD)
        for i in range(12)
    ]
    res_fast = router.ask_batch(fast_tasks, tier="fast")
    assert len(res_fast) == 12
    gpt_slots = int(cfg["vlm"]["providers"]["gpt_62_sol"].get("worker_slots", 1))
    assert 1 <= counts.get("gpt-6.2-sol", 0) <= gpt_slots
    assert db.get_vlm_cooldown("gpt_62_sol") - time.time() > 3600
    assert counts.get("gemini-4-carbon", 0) == 0
    assert counts.get("deepseek-v4.1-flash-1017", 0) >= 1
    assert ds_payload_checks and all(n_img == 1 and no_temp for n_img, no_temp in ds_payload_checks)

    # 2. Pro batch while non-Carbon pro providers are cooling down: gemini-4-carbon handles all pro highlights seamlessly
    db.record_vlm_result("qwen4_max", "qwen4-max", False, 0.0, "rate_limit", 120)
    counts_before_pro = dict(counts)
    pro_tasks = [
        (i, f"ms|clip_{i}|1.0", [tiny_jpg, tiny_jpg, tiny_jpg], cloud_vlm.PROMPT_MILESTONE)
        for i in range(4)
    ]
    res_pro = router.ask_batch(pro_tasks, tier="pro")
    assert len(res_pro) == 4
    assert all(v["_model"] == "gemini-4-carbon" for v in res_pro.values())
    assert counts.get("gemini-4-carbon", 0) == 4
    for k, old_v in counts_before_pro.items():
        if k != "gemini-4-carbon":
            assert counts.get(k, 0) == old_v

    # 3. Pro batch with all 3 pro families active: gemini-4-carbon, gpt-6.2-sol, and qwen4-max co-process highlights,
    #    while hydrogen / deepseek NEVER touch pro tasks.
    db_pro_co = ArchiveDB(tmp_path / "ws_pro_co.sqlite3")
    router_pro_co = cloud_vlm.CloudVLMRouter(cfg, db_pro_co, secret_overrides=overrides)
    gpt_quota_ok["ok"] = True
    counts_before_co = dict(counts)
    pro_co_tasks = [
        (i, f"ms|co_clip_{i}|1.0", [tiny_jpg, tiny_jpg, tiny_jpg], cloud_vlm.PROMPT_MILESTONE)
        for i in range(6)
    ]
    res_pro_co = router_pro_co.ask_batch(pro_co_tasks, tier="pro")
    assert len(res_pro_co) == 6
    assert {v["_model"] for v in res_pro_co.values()} == {"gemini-4-carbon", "gpt-6.2-sol", "qwen4-max"}
    assert counts.get("gemini-4-carbon", 0) > counts_before_co.get("gemini-4-carbon", 0)
    assert counts.get("gpt-6.2-sol", 0) > counts_before_co.get("gpt-6.2-sol", 0)
    assert counts.get("qwen4-max", 0) > counts_before_co.get("qwen4-max", 0)
    for k in ("gemini-4-hydrogen", "DeepSeek-V4.1-Flash-1017", "deepseek-v4.1-flash-1017"):
        assert counts.get(k, 0) == counts_before_co.get(k, 0)


def test_vertex_multi_account_jwt_inline_data_and_per_minute_429(tmp_path: Path, monkeypatch):
    import tomllib

    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    sa1_path = tmp_path / "gcp_vertex_sa_1.json"
    sa2_path = tmp_path / "gcp_vertex_sa_2.json"
    sa1_json = _make_test_sa_json("proj-one", "sa1@proj-one.iam.gserviceaccount.com")
    sa2_json = _make_test_sa_json("proj-two", "sa2@proj-two.iam.gserviceaccount.com")
    sa1_path.write_text(sa1_json, encoding="utf-8")

    # Point vertex_1_* and vertex_2_* at real temp files so adding sa2 activates Account 2 dynamically
    cfg_dyn = json.loads(json.dumps(cfg))
    cfg_dyn["vlm"]["parallel_tiers"]["pro"] = ["vertex_1_carbon", "vertex_2_carbon"]
    for pname, pc in cfg_dyn["vlm"]["providers"].items():
        if pname.startswith("vertex_1_"):
            pc["secret_file"] = str(sa1_path)
        elif pname.startswith("vertex_2_"):
            pc["secret_file"] = str(sa2_path)
        else:
            pc["secret_file"] = str(tmp_path / "other_secret.txt")
    (tmp_path / "other_secret.txt").write_text("dummy-other-secret", encoding="utf-8")

    # Verify redact_secrets strips OAuth tokens and private keys
    red = cloud_vlm.redact_secrets('Bearer ya29.c.AbCdEf123 {"private_key":"-----BEGIN PRIVATE KEY-----\\nABC\\n-----END PRIVATE KEY-----"}')
    assert "ya29" not in red and "BEGIN PRIVATE KEY" not in red

    db = ArchiveDB(tmp_path / "vertex_multi.sqlite3")
    router = cloud_vlm.CloudVLMRouter(cfg_dyn, db)

    lock = threading.Lock()
    token_calls: list[str] = []
    vertex_reqs: list[dict] = []
    inject_error = {"mode": None}

    class StubResp:
        def __init__(self, status_code: int, body: dict):
            self.status_code = status_code
            self._body = body
            self.text = json.dumps(body, ensure_ascii=False)

        def raise_for_status(self):
            if self.status_code >= 400:
                err = cloud_vlm.requests.HTTPError(f"HTTP {self.status_code}")
                err.response = self
                raise err

        def json(self):
            return self._body

    def fake_post(self_sess, url: str, **kw):
        if "oauth2.googleapis.com/token" in url:
            assertion = (kw.get("data") or {}).get("assertion", "")
            hdr_b64, claims_b64, _sig = assertion.split(".")
            pad = "=" * (-len(claims_b64) % 4)
            claims = json.loads(__import__("base64").urlsafe_b64decode(claims_b64 + pad).decode("utf-8"))
            with lock:
                token_calls.append(claims["iss"])
            tok = "ya29.tok-one" if claims["iss"].startswith("sa1@") else "ya29.tok-two"
            return StubResp(200, {"access_token": tok, "expires_in": 3600, "token_type": "Bearer"})

        assert "aiplatform.googleapis.com" in url, f"Unexpected non-Vertex URL called: {url}"
        payload = kw.get("json") or {}
        if inject_error["mode"] == "per_minute_429":
            return StubResp(429, {
                "error": {
                    "code": 429,
                    "message": "Quota exceeded for quota metric 'Generate content requests' and limit 'GenerateContent request limit per minute'",
                    "status": "RESOURCE_EXHAUSTED",
                }
            })
        if inject_error["mode"] == "billing_disabled":
            return StubResp(403, {
                "error": {
                    "code": 403,
                    "message": "This API method requires billing to be enabled. BILLING_DISABLED",
                    "status": "PERMISSION_DENIED",
                }
            })
        with lock:
            vertex_reqs.append({"url": url, "payload": payload, "headers": kw.get("headers") or {}})
        ans = json.dumps({"child_clear": "yes", "categories": ["walking"], "independence": "independent", "excitement_score": 5, "box_2d": [100, 100, 800, 800], "caption_zh": "宝宝独立迈步"})
        return StubResp(200, {
            "candidates": [{"content": {"parts": [{"text": ans}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 400, "candidatesTokenCount": 60, "totalTokenCount": 460},
        })

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)

    # 1 & 2. With only sa1 present, vertex_2_* is skipped and pro routes 100% to vertex_1_carbon (proj-one)
    r1 = router.ask("ms|c1|1.0", [b"JPG1"], cloud_vlm.PROMPT_MILESTONE, tier="pro")
    r2 = router.ask("ms|c2|1.0", [b"JPG2"], cloud_vlm.PROMPT_MILESTONE, tier="pro")
    assert r1["_provider"] == "vertex_1_carbon" and r2["_provider"] == "vertex_1_carbon"
    assert len(vertex_reqs) == 2
    for vr in vertex_reqs:
        assert "/projects/proj-one/locations/global/publishers/google/models/gemini-4-carbon:generateContent" in vr["url"]
        c0 = vr["payload"]["contents"][0]
        assert c0["role"] == "user"
        assert c0["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"
        assert vr["headers"]["Authorization"] == "Bearer ya29.tok-one"
    # 3. Token cache hit: OAuth token endpoint called only once for sa1 across multiple requests
    assert token_calls == ["sa1@proj-one.iam.gserviceaccount.com"]

    # 4. Drop sa2 onto disk without recreating router or editing config -> pro load-balances across BOTH projects
    sa2_path.write_text(sa2_json, encoding="utf-8")
    provs_seen = set()
    urls_seen = set()
    for idx in range(4):
        ans = router.ask(f"ms|dual_{idx}|1.0", [f"JPGD_{idx}".encode()], cloud_vlm.PROMPT_MILESTONE, tier="pro")
        provs_seen.add(ans["_provider"])
    for vr in vertex_reqs[2:]:
        if "/projects/proj-one/" in vr["url"]:
            urls_seen.add("proj-one")
        if "/projects/proj-two/" in vr["url"]:
            urls_seen.add("proj-two")
    assert provs_seen == {"vertex_1_carbon", "vertex_2_carbon"}
    assert urls_seen == {"proj-one", "proj-two"}
    assert token_calls.count("sa1@proj-one.iam.gserviceaccount.com") == 1
    assert token_calls.count("sa2@proj-two.iam.gserviceaccount.com") == 1

    # 5a. Vertex AI 429 "per minute" triggers short rate_limit_cooldown_seconds (<= 300s), NOT 6h quota cooldown
    db_pm = ArchiveDB(tmp_path / "pm_429.sqlite3")
    cfg_single = json.loads(json.dumps(cfg_dyn))
    cfg_single["vlm"]["parallel_tiers"]["pro"] = ["vertex_1_carbon"]
    cfg_single["vlm"]["tiers"]["pro"] = ["vertex_1_carbon"]
    r_pm = cloud_vlm.CloudVLMRouter(cfg_single, db_pm)
    inject_error["mode"] = "per_minute_429"
    try:
        r_pm.ask("ms|pm|1.0", [b"JPG_PM"], cloud_vlm.PROMPT_MILESTONE, tier="pro")
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass
    cd_pm = db_pm.get_vlm_cooldown("vertex_1_carbon") - time.time()
    assert 30 < cd_pm <= 300

    # 5b. BILLING_DISABLED triggers long quota_cooldown_seconds (> 3600s)
    db_bd = ArchiveDB(tmp_path / "bd_403.sqlite3")
    r_bd = cloud_vlm.CloudVLMRouter(cfg_single, db_bd)
    inject_error["mode"] = "billing_disabled"
    try:
        r_bd.ask("ms|bd|1.0", [b"JPG_BD"], cloud_vlm.PROMPT_MILESTONE, tier="pro")
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass
    cd_bd = db_bd.get_vlm_cooldown("vertex_1_carbon") - time.time()
    assert cd_bd > 3600


def test_prune_retired_vlm_stats_and_fleet_grouping(tmp_path: Path):
    import tomllib

    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    db = ArchiveDB(tmp_path / "fleet.sqlite3")
    db.record_vlm_result("gemini_hydrogen", "gemini-4-hydrogen", True, 5.0, usage={"input_tokens": 300, "output_tokens": 100, "total_tokens": 400})
    db.record_vlm_result("vertex_1_hydrogen", "gemini-4-hydrogen", True, 4.0, usage={"input_tokens": 250, "output_tokens": 50, "total_tokens": 300})
    db.record_vlm_result("vertex_2_carbon", "gemini-4-carbon", True, 18.0, usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700})
    db.record_vlm_result("gpt_62_sol", "gpt-6.2-sol", True, 3.5, usage={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300})
    for ret_p, ret_m in (
        ("gpt_56_sol", "gpt-5.6-sol"),
        ("gpt_61_sol", "gpt-6.1-sol"),
        ("gemini_38_flash", "gemini-3.8-flash"),
        ("kimi_k31_pro", "kimi-k3.1-pro"),
    ):
        db.record_vlm_result(ret_p, ret_m, True, 9.0, usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})

    deleted = db.prune_retired_vlm_stats(set(cfg["vlm"]["providers"].keys()))
    assert deleted == 4
    rem_provs = {v["provider"] for v in db.get_dashboard_snapshot()["vlm_stats"]}
    assert rem_provs == {"gemini_hydrogen", "vertex_1_hydrogen", "vertex_2_carbon", "gpt_62_sol"}

    snap = db.get_dashboard_snapshot()
    fleet = web_server._build_vlm_fleet(cfg, snap)
    assert [c["model"] for c in fleet] == [
        "gpt-6.2-sol",
        "qwen4-max",
        "gemini-4-carbon",
        "DeepSeek-V4.1-Flash-1017",
        "gemini-4-hydrogen",
    ]
    assert fleet[0]["role"] == "pro" and fleet[0]["role_zh"] == "高光 + 筛选 · 全能"
    assert "OpenAI 云端 (CLIProxy)" in fleet[0]["channel_zh"]
    assert fleet[0]["slots"] == 2 and fleet[0]["calls_ok"] == 1 and fleet[0]["avg_latency_s"] == 3.5
    assert fleet[1]["role"] == "pro" and fleet[1]["role_zh"] == "高光 + 筛选 · 全能"
    assert fleet[2]["role"] == "pro" and fleet[2]["role_zh"] == "高光精审 · 主力" and fleet[2]["calls_ok"] == 1 and fleet[2]["total_tokens"] == 700
    assert fleet[4]["role"] == "fast" and fleet[4]["calls_ok"] == 2 and fleet[4]["total_tokens"] == 700
    sub_lbls = [sc["label"] for sc in fleet[4]["sub_channels"]]
    assert "GCP#1" in sub_lbls and "GCP#2" in sub_lbls and "AI Studio" in sub_lbls


def test_finalize_day_keeps_old_photos_intact_if_mid_curation_fails(tmp_path: Path, monkeypatch):
    dm, _data_root, day_dir = _make_test_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(dm.vlm, "ask", lambda *a, **kw: {
        "child_clear": "yes", "face_or_expression_clear": "yes", "composition_good": "yes",
        "osd_box_on_child": "no", "adult_back_or_awkward": "no", "private_care": "no",
        "photo_score": 5, "child_box": [0.2, 0.2, 0.6, 0.8], "caption_zh": "新精选照",
    })
    monkeypatch.setattr(
        daemon_mod.med,
        "save_photo_pair",
        lambda _ff, _src, _t, _w, _h, _box, _z, full_out, framed_out, **_kw: (
            full_out.write_bytes(b"NEW_FULL") and framed_out.write_bytes(b"NEW_FRAMED") or True
        ),
    )

    def broken_director(**_kw):
        raise RuntimeError("director stage crashed")

    monkeypatch.setattr(daemon_mod.dl_dir, "curate_daily_highlight_and_milestones", broken_director)
    try:
        dm.finalize_day("2026-11-17", "P0_LIVE", final=False)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass

    pdir = day_dir / "photos"
    assert (pdir / "full_old.jpg").read_bytes() == b"OLD_FULL"
    assert (pdir / "framed_old.jpg").read_bytes() == b"OLD_FRAMED"
    assert list(day_dir.glob(".photos_staging_*")) == []
    disk_sum = json.loads((day_dir / "summary.json").read_text(encoding="utf-8"))
    assert disk_sum["photos"][0]["full"] == "days/2026-11-17/photos/full_old.jpg"


def test_finalize_day_defers_photo_swap_until_after_director_stage(tmp_path: Path, monkeypatch):
    dm, _data_root, day_dir = _make_test_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(dm.vlm, "ask", lambda *a, **kw: {
        "child_clear": "yes", "face_or_expression_clear": "yes", "composition_good": "yes",
        "osd_box_on_child": "no", "adult_back_or_awkward": "no", "private_care": "no",
        "photo_score": 5, "child_box": [0.2, 0.2, 0.6, 0.8], "caption_zh": "延迟提交新照",
    })

    def good_save_pair(_ff, _src, _t, _w, _h, _box, _z, full_out: Path, framed_out: Path, **_kw) -> bool:
        full_out.write_bytes(b"NEW_FULL")
        framed_out.write_bytes(b"NEW_FRAMED")
        return True

    monkeypatch.setattr(daemon_mod.med, "save_photo_pair", good_save_pair)

    observed_during_director: dict[str, bool] = {}

    def spy_director(**_kw):
        pdir = day_dir / "photos"
        observed_during_director["old_still_present"] = (pdir / "full_old.jpg").is_file()
        observed_during_director["staging_exists"] = len(list(day_dir.glob(".photos_staging_*"))) == 1
        return [], None, []

    monkeypatch.setattr(daemon_mod.dl_dir, "curate_daily_highlight_and_milestones", spy_director)
    summary = dm.finalize_day("2026-11-17", "P0_LIVE", final=False)

    assert observed_during_director == {"old_still_present": True, "staging_exists": True}
    pdir = day_dir / "photos"
    assert not (pdir / "full_old.jpg").exists()
    assert list(day_dir.glob(".photos_staging_*")) == []
    assert len(summary["photos"]) == 1
    assert (pdir / Path(summary["photos"][0]["full"]).name).read_bytes() == b"NEW_FULL"
    assert (pdir / Path(summary["photos"][0]["framed"]).name).read_bytes() == b"NEW_FRAMED"


def test_web_server_filters_missing_photos_and_falls_back_single_variant(tmp_path: Path):
    data_root = tmp_path / "data"
    day = "2026-11-19"
    pdir = data_root / "days" / day / "photos"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "full_both.jpg").write_bytes(b"BOTH_FULL")
    (pdir / "framed_both.jpg").write_bytes(b"BOTH_FRAMED")
    (pdir / "full_only.jpg").write_bytes(b"ONLY_FULL")
    (data_root / "days" / day / "highlight_20261119.mp4").write_bytes(b"MP4")

    summary = {
        "day": day,
        "status": "preview",
        "highlight_rel": f"days/{day}/highlight_20261119.mp4",
        "photos": [
            {
                "full": f"days/{day}/photos/full_both.jpg",
                "framed": f"days/{day}/photos/framed_both.jpg",
                "camera": "厨房 G350", "zoom_x": 1.0, "time": "08:00:00", "score": 4, "caption_zh": "双图正常",
            },
            {
                "full": f"days/{day}/photos/full_only.jpg",
                "framed": f"days/{day}/photos/framed_missing.jpg",
                "camera": "厨房 G350", "zoom_x": 1.0, "time": "09:00:00", "score": 4, "caption_zh": "仅原图存在",
            },
            {
                "full": f"days/{day}/photos/full_gone.jpg",
                "framed": f"days/{day}/photos/framed_gone.jpg",
                "camera": "厨房 G350", "zoom_x": 1.0, "time": "10:00:00", "score": 4, "caption_zh": "双图都已丢失",
            },
        ],
    }
    (data_root / "days" / day / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    db.upsert_day_summary(day, "P0_LIVE", "preview", summary)
    cfg = {"project": {"name": "Test", "birthday": "2025-12-17", "data_root": str(data_root)}, "cameras": {}}

    day_pay = web_server._build_day_payload(day, data_root, db, cfg)
    assert [p["caption_zh"] for p in day_pay["photos"]] == ["双图正常", "仅原图存在"]
    assert day_pay["photos"][1]["framed"] == f"days/{day}/photos/full_only.jpg"
    assert day_pay["photos"][1]["full"] == f"days/{day}/photos/full_only.jpg"

    boot = web_server._build_home_boot(data_root, db, cfg)
    assert [p["caption_zh"] for p in boot["featured_photos"]] == ["双图正常", "仅原图存在"]
    html = web_server.render_live_html(data_root, db, cfg)
    assert "handlePhotoImgError" in html and "mediaVerUrl" in html

