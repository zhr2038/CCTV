"""Regression tests for Work Item ②: VLM refusal pending_review & persistent kitchen pagination."""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "nas-growth-v3"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from nas_growth import cloud_vlm, daemon as daemon_mod, web_server  # noqa: E402
from nas_growth.daemon import GrowthDaemon  # noqa: E402
from nas_growth.db import ArchiveDB  # noqa: E402


def _make_cfg(tmp_path: Path, *, kitchen_batch_size: int = 14) -> dict:
    return {
        "project": {
            "name": "Test Archive V3",
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
            "whisper_dir": str(tmp_path / "models" / "whisper"),
        },
        "media": {
            "use_vaapi": False,
            "t1_height": 480,
            "t1_fps": 15,
            "t1_qp": 28,
            "t2_height": 360,
            "t2_qp": 30,
            "timelapse_fps": 10,
            "smooth_fps": 30,
            "smooth_fade_s": 0.4,
            "smooth_width": 1280,
            "smooth_height": 720,
            "max_highlight_shots": 12,
        },
        "vlm": {
            "request_timeout_seconds": 10,
            "cooldown_seconds": 300,
            "kitchen_batch_size": kitchen_batch_size,
            "tiers": {"fast": ["p_fast"], "pro": ["p_pro"]},
            "providers": {
                "p_fast": {
                    "kind": "gemini",
                    "model": "gemini-4-hydrogen",
                    "base_url": "https://example.invalid/v1beta",
                    "secret_file": "/sec/fast",
                },
                "p_pro": {
                    "kind": "openai",
                    "model": "gemini-4-carbon",
                    "base_url": "https://example.invalid/v1",
                    "secret_file": "/sec/pro",
                },
            },
        },
        "cameras": {
            "cam_living": {
                "name": "客厅看娃",
                "subdir": "aqara_video/living",
                "enabled": True,
                "zoom_factor": 1.4,
                "milestone_watch": True,
                "daytime_photos": True,
                "t1_mode": "always",
            },
            "cam_kitchen": {
                "name": "厨房台面",
                "subdir": "aqara_video/kitchen",
                "enabled": True,
                "zoom_factor": 1.3,
                "milestone_watch": False,
                "daytime_photos": False,
                "t1_mode": "verified_child_only",
            },
        },
    }


def _make_daemon(tmp_path: Path, monkeypatch, *, kitchen_batch_size: int = 14) -> tuple[GrowthDaemon, Path, Path]:
    cfg = _make_cfg(tmp_path, kitchen_batch_size=kitchen_batch_size)
    data_root = Path(cfg["project"]["data_root"])
    cctv_root = Path(cfg["project"]["cctv_root"])
    vault_root = Path(cfg["project"]["vault_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    cctv_root.mkdir(parents=True, exist_ok=True)
    vault_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon_mod.med, "resolve_font", lambda *_a, **_kw: None)
    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *_a, **_kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *_a, **_kw: object())
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    return GrowthDaemon(cfg, db, ff_bin="ffmpeg", fp_bin="ffprobe"), data_root, cctv_root


def _fake_jpeg(seed: int) -> bytes:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:, :, 0] = (seed * 37) % 251
    img[:, :, 1] = (seed * 73 + 20) % 251
    img[:, :, 2] = (seed * 19 + 90) % 251
    cv2.putText(img, f"F{seed:03d}", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return bytes(buf)


def test_refusal_never_auto_confirms_in_ask_batch_or_finalize_day(tmp_path: Path, monkeypatch):
    """Cloud VLM policy refusals stay pending_review across ask_batch, ms, ph, kit, and sp."""
    db = ArchiveDB(tmp_path / "router.sqlite3")
    cfg_vlm = {
        "vlm": {
            "request_timeout_seconds": 10,
            "cooldown_seconds": 600,
            "tiers": {"fast": ["gem_f", "oai_b"], "pro": ["gem_p", "oai_b"]},
            "providers": {
                "gem_f": {"kind": "gemini", "model": "gemini-4-hydrogen", "base_url": "https://example.invalid/v1beta", "secret_file": "/sec/g"},
                "gem_p": {"kind": "gemini", "model": "gemini-4-carbon", "base_url": "https://example.invalid/v1beta", "secret_file": "/sec/g"},
                "oai_b": {"kind": "openai", "model": "gpt-6.2-sol", "base_url": "https://example.invalid/v1", "secret_file": "/sec/o"},
            },
        }
    }
    router = cloud_vlm.CloudVLMRouter(cfg_vlm, db, secret_overrides={"/sec/g": "k1", "/sec/o": "k2"})

    class SafetyResp:
        status_code = 200
        text = '{"promptFeedback":{"blockReason":"SAFETY"}}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"promptFeedback": {"blockReason": "SAFETY"}}

    calls: list[str] = []

    def fake_post(url, **kw):
        calls.append(url)
        if "chat/completions" in url:
            raise AssertionError("Policy refusal must not forward to secondary cloud provider")
        return SafetyResp()

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", lambda self, url, **kw: fake_post(url, **kw))
    batch_res = router.ask_batch([(0, "ms|c1|1.0", [_fake_jpeg(1), _fake_jpeg(2)], cloud_vlm.PROMPT_MILESTONE)], tier="pro")
    ans = batch_res[0]
    assert ans["review_status"] == "pending_review"
    assert ans["vlm_refused"] is True
    assert ans["child_clear"] == "pending_review"
    assert ans["categories"] == []
    assert ans["_model"] == "vlm_refusal_pending"
    assert db.get_dashboard_snapshot()["vlm_stats"] == []
    assert len(calls) == 1

    # End-to-end finalize_day with refusals across ms, ph, kit, and sp
    dm, data_root, cctv_root = _make_daemon(tmp_path / "dm_refusal", monkeypatch)
    day = "2026-11-16"
    day_dir = data_root / "days" / day
    living_dir = cctv_root / "aqara_video" / "living" / "20261116"
    kitchen_dir = cctv_root / "aqara_video" / "kitchen" / "20261116"
    living_dir.mkdir(parents=True, exist_ok=True)
    kitchen_dir.mkdir(parents=True, exist_ok=True)

    living_src = living_dir / "living_100000.mp4"
    living_src.write_bytes(b"LIVING_CLIP")
    kitchen_src = kitchen_dir / "kitchen_101500.mp4"
    kitchen_src.write_bytes(b"KITCHEN_CLIP")

    # Also stage a dummy t2 file for kitchen so kitchen processing is tested completely
    t2_k = day_dir / "t2" / "cam_kitchen" / "kitchen_101500.jpg"
    t2_k.parent.mkdir(parents=True, exist_ok=True)
    t2_k.write_bytes(_fake_jpeg(9))

    dm.db.upsert_discovered_files([
        {
            "source_rel": "aqara_video/living/20261116/living_100000.mp4",
            "source_root": "cctv",
            "day": day,
            "camera": "cam_living",
            "wall_ts": f"{day}T10:00:00",
            "size": 11,
            "mtime": 1700000000.0,
        },
        {
            "source_rel": "aqara_video/kitchen/20261116/kitchen_101500.mp4",
            "source_root": "cctv",
            "day": day,
            "camera": "cam_kitchen",
            "wall_ts": f"{day}T10:15:00",
            "size": 12,
            "mtime": 1700000010.0,
        },
    ])
    dm.db.mark_file_done(
        "aqara_video/living/20261116/living_100000.mp4",
        {
            "status": "done",
            "duration": 60.0,
            "width": 1280,
            "height": 720,
            "has_audio": True,
            "active_s": 25,
            "person_s": 25,
            "t1_mode": "always",
            "t1_rel": f"days/{day}/t1/cam_living/living_100000.mp4",
            "t1_bytes": 1000,
            "max_box_score": 0.92,
            "audio_summary": {
                "speech_prob": 0.88,
                "max_child_voc": 0.72,
                "segments": [[5.0, 20.0]],
                "transcript_hits": [
                    {
                        "burst_start_s": 8.0,
                        "burst_end_s": 10.5,
                        "matched_word": "妈妈",
                        "phonetic_prob": 0.91,
                        "f0_median_hz": 395.0,
                        "snr_ratio": 2.4,
                    }
                ],
            },
            "candidates": [{"t": 12.0, "score": 0.92, "box": [0.2, 0.2, 0.6, 0.8], "luma": 130.0, "sat": 45.0, "sharp": 90.0}],
            "bursts": [
                {
                    "start_s": 8.0,
                    "end_s": 14.0,
                    "snr_ratio": 2.4,
                    "y_child_babble": 0.72,
                    "source_rel": "aqara_video/living/20261116/living_100000.mp4",
                    "camera": "cam_living",
                }
            ],
        },
    )
    dm.cfg["asr"]["enabled"] = True
    dm.cfg["asr"]["model_dir"] = str(tmp_path / "whisper")
    monkeypatch.setattr(daemon_mod.asr_worker, "checkpoint_jobs", lambda _md, _pl: [{"job_id": "sp_job_1"}])
    monkeypatch.setattr(
        daemon_mod.asr_worker,
        "read_checkpoint",
        lambda _cp, _jobs: (
            {"sp_job_1"},
            [
                {
                    "source_rel": "aqara_video/living/20261116/living_100000.mp4",
                    "source_path": str(living_src),
                    "camera": "cam_living",
                    "utterance_t_s": 9.0,
                    "start_s": 6.0,
                    "end_s": 13.0,
                    "burst_start_s": 8.0,
                    "matched_word": "妈妈",
                    "phonetic_prob": 0.91,
                    "f0_median_hz": 395.0,
                    "snr_ratio": 2.4,
                }
            ],
        ),
    )
    dm.db.mark_file_done(
        "aqara_video/kitchen/20261116/kitchen_101500.mp4",
        {
            "status": "done",
            "duration": 45.0,
            "width": 1280,
            "height": 720,
            "has_audio": False,
            "active_s": 10,
            "person_s": 10,
            "t1_mode": "verified_child_only",
            "t1_rel": None,
            "t1_bytes": 0,
            "t2_rel": f"days/{day}/t2/cam_kitchen/kitchen_101500.jpg",
            "t2_bytes": t2_k.stat().st_size,
            "max_box_score": 0.86,
            "audio_summary": {"segments": [[4.0, 12.0]]},
            "candidates": [{"t": 8.0, "score": 0.86, "box": [0.25, 0.25, 0.65, 0.75]}],
            "bursts": [],
        },
    )

    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [_fake_jpeg(i + 10) for i in range(len(times))])
    monkeypatch.setattr(daemon_mod.med, "promote_t0_keeper", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Must not promote refusal to T0")))
    monkeypatch.setattr(daemon_mod.med, "render_single_clip_with_badge", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Must not render refusal milestone")))
    monkeypatch.setattr(daemon_mod.med, "render_smooth_film_nas", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Must not render refusal highlight")))
    monkeypatch.setattr(daemon_mod.med, "encode_t1_segments", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Must not encode kitchen T1 on refusal")))
    monkeypatch.setattr(daemon_mod.med, "save_photo_pair", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Must not save photo on refusal")))

    monkeypatch.setattr(
        dm.vlm,
        "ask",
        lambda *_a, **_kw: {
            "child_clear": "pending_review",
            "categories": [],
            "suggested_category": "daily_care",
            "vlm_refused": True,
            "review_status": "pending_review",
            "refusal_reason": "content_policy_blocked",
            "child_box": [0.2, 0.2, 0.6, 0.8],
            "caption_zh": "云端模型安全策略拒答，已保留原视频区间待人工复核",
            "_provider": "gem_f",
            "_model": "vlm_refusal_pending",
        },
    )

    summary = dm.finalize_day(day, "P0_LIVE", final=True)
    assert summary["milestones_count"] == 0
    assert summary["speech_count"] == 0
    assert summary["photos"] == []
    assert summary["pending_review_count"] >= 4
    assert summary["highlight_rel"] is None

    revs = dm.db.list_vlm_reviews(status="pending_review", day=day)
    kinds = {rv["kind"] for rv in revs}
    assert {"milestone", "photo", "kitchen", "speech"} <= kinds
    assert dm.db.get_day_milestones(day) == []
    snap = dm.db.get_dashboard_snapshot()
    assert snap["archive_counts"]["pending_review_count"] >= 4
    assert snap["archive_counts"]["milestones_count"] == 0


def test_backward_compat_legacy_cache_milestones_deliverables_and_human_review(tmp_path: Path):
    """Legacy local_care_fallback cache, milestones, and deliverables are demoted until human approval."""
    db_path = tmp_path / "data" / "state" / "growth_v3.sqlite3"
    data_root = tmp_path / "data"
    (data_root / "days" / "2026-11-08").mkdir(parents=True, exist_ok=True)
    (data_root / "v2" / "photos").mkdir(parents=True, exist_ok=True)
    for fn in ("f_fallback.jpg", "c_fallback.jpg", "f_real.jpg", "c_real.jpg"):
        (data_root / "v2" / "photos" / fn).write_bytes(b"JPG")
    (data_root / "v2" / "highlights").mkdir(parents=True, exist_ok=True)

    db = ArchiveDB(db_path)
    # 1. Seed legacy local_care_fallback vlm_cache entry
    db.put_vlm_cache(
        "legacy_key_1",
        "pro",
        "gem_primary",
        "local_care_fallback",
        "ms|legacy|1.0",
        {
            "child_clear": "yes",
            "categories": ["daily_care"],
            "excitement_score": 4,
            "local_care_fallback": "yes",
            "caption_zh": "日常温馨照护 · 本地托底留存（云端未返回细节）",
        },
        0.0,
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    normalized_cache = db.get_vlm_cache("legacy_key_1")
    assert normalized_cache is not None
    assert normalized_cache["review_status"] == "pending_review"
    assert normalized_cache["categories"] == []
    assert normalized_cache["child_clear"] == "pending_review"

    # 2. Seed a legacy milestone directly as approved, then re-open ArchiveDB to run migration
    with db.session() as con:
        con.execute(
            """INSERT INTO milestones
               (id, day, camera, category, source_rel, start_s, end_s, caption_zh, speaker_note,
                independence, score, video_rel, poster_rel, t0_rel, t0_sha256, review_status, meta_json, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "legacy_care_1",
                "2026-11-08",
                "cam_living",
                "daily_care",
                "aqara_video/living/20261108/c1.mp4",
                10.0,
                16.0,
                "日常温馨照护 · 本地托底留存",
                "",
                "assisted",
                4.0,
                "v2/highlights/legacy_care_1.mp4",
                "",
                "",
                "",
                "approved",
                json.dumps({"local_care_fallback": "yes"}),
                time.time(),
            ),
        )
    db2 = ArchiveDB(db_path)
    assert db2.get_day_milestones("2026-11-08") == []
    assert len(db2.get_day_milestones("2026-11-08", include_pending=True)) == 1
    assert db2.get_vlm_review("legacy_ms|legacy_care_1")["status"] == "pending_review"

    # 3. Seed legacy deliverables.json and summary.json with fallback markers alongside one real record
    (data_root / "v2" / "deliverables.json").write_text(
        json.dumps(
            {
                "highlights": [
                    {
                        "id": "deliv_fallback_hl",
                        "day": "2026-11-08",
                        "camera": "cam_living",
                        "cam_zh": "客厅",
                        "category": "daily_care",
                        "video": "fallback.mp4",
                        "poster": "",
                        "caption_zh": "日常温馨照护（隐私保护 · 云端自动避让）",
                        "local_care_fallback": "yes",
                    },
                    {
                        "id": "deliv_real_hl",
                        "day": "2026-11-08",
                        "camera": "cam_living",
                        "cam_zh": "客厅",
                        "category": "standing",
                        "video": "real_stand.mp4",
                        "poster": "",
                        "caption_zh": "宝宝扶站稳稳一笑",
                    },
                ],
                "photos": [
                    {
                        "full": "f_fallback.jpg",
                        "framed": "c_fallback.jpg",
                        "camera": "客厅",
                        "zoom_x": 1.4,
                        "time": "10:00:00",
                        "score": 4,
                        "caption_zh": "日常温馨照护 · 本地托底留存",
                        "local_care_fallback": "yes",
                    },
                    {
                        "full": "f_real.jpg",
                        "framed": "c_real.jpg",
                        "camera": "客厅",
                        "zoom_x": 1.4,
                        "time": "11:00:00",
                        "score": 5,
                        "caption_zh": "宝宝开心拍手",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data_root / "days" / "2026-11-08" / "summary.json").write_text(
        json.dumps(
            {
                "day": "2026-11-08",
                "photos": [
                    {
                        "full": "v2/photos/f_fallback.jpg",
                        "framed": "v2/photos/c_fallback.jpg",
                        "camera": "客厅",
                        "caption_zh": "云端安全策略拒答，已保留原视频区间待人工复核",
                        "vlm_refused": True,
                    },
                    {
                        "full": "v2/photos/f_real.jpg",
                        "framed": "v2/photos/c_real.jpg",
                        "camera": "客厅",
                        "caption_zh": "宝宝开心拍手",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = _make_cfg(tmp_path)
    cfg["project"]["data_root"] = str(data_root)
    ms_items = web_server._collect_milestones(data_root, db2, cfg)
    assert [it["id"] for it in ms_items["items"]] == ["real_stand.mp4"]
    day_pay = web_server._build_day_payload("2026-11-08", data_root, db2, cfg)
    assert [p["caption_zh"] for p in day_pay["photos"]] == ["宝宝开心拍手"]

    # 4. Exercise HTTP /api/reviews and /api/ctl/vlm_review human approval
    class DummyDaemon:
        def is_paused(self):
            return False

        def set_paused(self, _v):
            return None

    srv = web_server.create_server("127.0.0.1", 0, data_root, db2, DummyDaemon(), cfg)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        revs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/reviews?status=pending_review", timeout=5).read().decode("utf-8"))
        assert any(r["review_id"] == "legacy_ms|legacy_care_1" for r in revs["items"])

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ctl/vlm_review",
            data=json.dumps({"review_id": "legacy_ms|legacy_care_1", "action": "approve", "note": "人工核实通过"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert res["review"]["status"] == "approved"

        ms_after = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/milestones", timeout=5).read().decode("utf-8"))
        assert "legacy_care_1" in {m["id"] for m in ms_after["items"]}
    finally:
        srv.shutdown()
        srv.server_close()


    # Also verify upgrading a legacy SQLite DB whose milestones table lacks review_status column
    old_schema_db = tmp_path / "legacy_pre_review_col.sqlite3"
    import sqlite3 as _sq
    _con = _sq.connect(str(old_schema_db))
    _con.executescript("""
        CREATE TABLE milestones (
            id TEXT PRIMARY KEY,
            day TEXT NOT NULL,
            camera TEXT NOT NULL,
            category TEXT NOT NULL,
            source_rel TEXT NOT NULL,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            caption_zh TEXT NOT NULL,
            speaker_note TEXT NOT NULL DEFAULT '',
            independence TEXT NOT NULL DEFAULT 'unclear',
            score REAL NOT NULL DEFAULT 0.0,
            video_rel TEXT,
            poster_rel TEXT,
            t0_rel TEXT,
            t0_sha256 TEXT,
            meta_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        INSERT INTO milestones VALUES (
            'old_care_row', '2026-11-08', 'cam_living', 'daily_care', 'aqara/1.mp4',
            1.0, 5.0, '日常温馨照护 · 本地托底留存', '', 'assisted', 4.0,
            'v2/highlights/old_care_row.mp4', '', '', '', '{"local_care_fallback":"yes"}', 1700000000.0
        );
    """)
    _con.close()
    migrated_old = ArchiveDB(old_schema_db)
    assert migrated_old.get_day_milestones("2026-11-08") == []
    assert migrated_old.get_vlm_review("legacy_ms|old_care_row")["status"] == "pending_review"


def test_kitchen_persistent_pagination_restart_refusal_dedup_and_invalidation(tmp_path: Path, monkeypatch):
    """Kitchen gate paginates >14 candidates across restarts, records refusals, deduplicates, and invalidates changed files."""
    dm, data_root, cctv_root = _make_daemon(tmp_path, monkeypatch, kitchen_batch_size=14)
    day = "2026-11-16"
    kit_dir = cctv_root / "aqara_video" / "kitchen" / "20261116"
    kit_dir.mkdir(parents=True, exist_ok=True)
    t2_dir = data_root / "days" / day / "t2" / "cam_kitchen"
    t2_dir.mkdir(parents=True, exist_ok=True)

    discovered = []
    frame_by_src: dict[str, bytes] = {}
    base_dt = 1794800000.0
    for i in range(30):
        mm = i
        hh = 10 + (mm // 60)
        rel = f"aqara_video/kitchen/20261116/clip_{i:02d}.mp4"
        fp = cctv_root / rel
        fp.write_bytes(f"VIDEO_{i:02d}".encode("utf-8"))
        t2p = t2_dir / f"clip_{i:02d}.jpg"
        t2p.write_bytes(_fake_jpeg(i + 1))
        frame_by_src[str(fp)] = _fake_jpeg(i + 100)
        discovered.append({
            "source_rel": rel,
            "source_root": "cctv",
            "day": day,
            "camera": "cam_kitchen",
            "wall_ts": f"{day}T{hh:02d}:{mm % 60:02d}:00",
            "size": fp.stat().st_size,
            "mtime": base_dt + i * 60.0,
        })
    dm.db.upsert_discovered_files(discovered)
    for i, d in enumerate(discovered):
        rel = d["source_rel"]
        dm.db.mark_file_done(
            rel,
            {
                "status": "done",
                "duration": 30.0,
                "width": 1280,
                "height": 720,
                "has_audio": False,
                "active_s": 5,
                "person_s": 5,
                "t1_mode": "verified_child_only",
                "t1_rel": None,
                "t1_bytes": 0,
                "t2_rel": f"days/{day}/t2/cam_kitchen/clip_{i:02d}.jpg",
                "t2_bytes": 512,
                "max_box_score": round(0.99 - i * 0.01, 3),
                "audio_summary": {"segments": [[2.0, 8.0]]},
                "candidates": [{"t": 5.0, "score": round(0.99 - i * 0.01, 3), "box": [0.2, 0.2, 0.6, 0.7]}],
                "bursts": [],
            },
        )

    def stub_extract(_ff, src_path, times, **_kw):
        raw = frame_by_src[str(src_path)]
        return [raw for _ in times]

    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", stub_extract)
    monkeypatch.setattr(daemon_mod.med, "encode_t1_segments", lambda _ff, _s, out_p, *_a, **_kw: (Path(out_p).parent.mkdir(parents=True, exist_ok=True) or Path(out_p).write_bytes(b"K_T1") or True, 0))
    monkeypatch.setattr(daemon_mod.med, "render_smooth_film_nas", lambda *_a, **_kw: None)

    vlm_calls: list[str] = []

    def stub_vlm_ask(prompt_tag: str, _imgs, _prompt, tier: str = "fast"):
        vlm_calls.append(prompt_tag)
        if "clip_02.mp4" in prompt_tag:
            return {
                "child_clear": "pending_review",
                "categories": [],
                "suggested_category": "daily_care",
                "vlm_refused": True,
                "review_status": "pending_review",
                "refusal_reason": "content_policy_blocked",
                "_provider": "p_fast",
                "_model": "vlm_refusal_pending",
            }
        if "clip_00.mp4" in prompt_tag:
            return {"child_clear": "yes", "child_box": [0.2, 0.2, 0.6, 0.7], "_provider": "p_fast", "_model": "gemini-4-hydrogen"}
        return {"child_clear": "no", "_provider": "p_fast", "_model": "gemini-4-hydrogen"}

    monkeypatch.setattr(dm.vlm, "ask", stub_vlm_ask)

    # Pass 1: evaluates #00..#13 (14 candidates), with #02 refused & persisted as pending_review
    dm.finalize_day(day, "P0_LIVE", final=False)
    assert len(vlm_calls) == 14
    assert all(f"clip_{i:02d}" in vlm_calls[i] for i in range(14))
    revs1 = dm.db.get_kitchen_reviews_for_day(day)
    assert len(revs1) == 14
    assert revs1["kit|aqara_video/kitchen/20261116/clip_00.mp4|5.0"]["status"] == "confirmed"
    assert revs1["kit|aqara_video/kitchen/20261116/clip_02.mp4|5.0"]["status"] == "pending_review"
    assert dm.db.get_vlm_review("kit|aqara_video/kitchen/20261116/clip_02.mp4|5.0")["status"] == "pending_review"

    # Simulate process restart: new ArchiveDB + new GrowthDaemon instance
    db_restarted = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    dm_restarted = GrowthDaemon(dm.cfg, db_restarted, ff_bin="ffmpeg", fp_bin="ffprobe")
    monkeypatch.setattr(dm_restarted.vlm, "ask", stub_vlm_ask)

    # Pass 2: skips #00..#13 and advances to #14..#27 (14 candidates)
    vlm_calls.clear()
    dm_restarted.finalize_day(day, "P0_LIVE", final=False)
    assert len(vlm_calls) == 14
    assert any("clip_14" in tag for tag in vlm_calls)
    assert not any("clip_01" in tag for tag in vlm_calls)

    # Pass 3: finishes #28..#29 (2 candidates)
    vlm_calls.clear()
    dm_restarted.finalize_day(day, "P0_LIVE", final=False)
    assert len(vlm_calls) == 2

    # Pass 4: all 30 covered -> 0 new VLM calls
    vlm_calls.clear()
    dm_restarted.finalize_day(day, "P0_LIVE", final=False)
    assert len(vlm_calls) == 0

    # Add 2 dedup test clips:
    # - dup_wall: within <4s of clip_05 (10:05:00 + 5s vs 10:05:02 + 5s -> diff 2s)
    # - dup_sha: different timestamp (>1h later), identical JPEG bytes as clip_01
    for idx_dup, (rel, wts, img_bytes) in enumerate([
        ("aqara_video/kitchen/20261116/clip_30_walldup.mp4", f"{day}T10:05:02", _fake_jpeg(240)),
        ("aqara_video/kitchen/20261116/clip_31_shadup.mp4", f"{day}T12:30:00", frame_by_src[str(cctv_root / "aqara_video/kitchen/20261116/clip_01.mp4")]),
    ]):
        fp = cctv_root / rel
        fp.write_bytes(f"DUP_{idx_dup}".encode("utf-8"))
        t2p = t2_dir / f"{fp.stem}.jpg"
        t2p.write_bytes(_fake_jpeg(210 + idx_dup))
        frame_by_src[str(fp)] = img_bytes
        dm_restarted.db.upsert_discovered_files([{
            "source_rel": rel, "source_root": "cctv", "day": day, "camera": "cam_kitchen",
            "wall_ts": wts, "size": fp.stat().st_size, "mtime": base_dt + 4000.0 + idx_dup,
        }])
        dm_restarted.db.mark_file_done(rel, {
            "status": "done", "duration": 30.0, "width": 1280, "height": 720, "has_audio": False,
            "active_s": 5, "person_s": 5, "t1_mode": "verified_child_only", "t1_rel": None, "t1_bytes": 0,
            "t2_rel": f"days/{day}/t2/cam_kitchen/{fp.stem}.jpg", "t2_bytes": 512, "max_box_score": 0.6,
            "audio_summary": {"segments": [[2.0, 8.0]]},
            "candidates": [{"t": 5.0, "score": 0.6, "box": [0.2, 0.2, 0.5, 0.5]}], "bursts": [],
        })

    vlm_calls.clear()
    dm_restarted.finalize_day(day, "P0_LIVE", final=False)
    assert len(vlm_calls) == 0
    revs_after_dup = dm_restarted.db.get_kitchen_reviews_for_day(day)
    assert revs_after_dup["kit|aqara_video/kitchen/20261116/clip_30_walldup.mp4|5.0"]["model"] == "wall_ts_dedup"
    assert revs_after_dup["kit|aqara_video/kitchen/20261116/clip_31_shadup.mp4|5.0"]["model"] == "frame_sha256_dedup"

    # Modify source file size/mtime -> stale kitchen review must invalidate and re-run
    changed_rel = "aqara_video/kitchen/20261116/clip_05.mp4"
    dm_restarted.db.upsert_discovered_files([{
        "source_rel": changed_rel, "source_root": "cctv", "day": day, "camera": "cam_kitchen",
        "wall_ts": f"{day}T10:05:00", "size": 9999, "mtime": base_dt + 9999.0,
    }])
    assert "kit|aqara_video/kitchen/20261116/clip_05.mp4|5.0" not in dm_restarted.db.get_kitchen_reviews_for_day(day)


def test_kitchen_mid_pagination_skips_duplicate_highlight_rerender(tmp_path: Path, monkeypatch):
    dm, data_root, cctv_root = _make_daemon(tmp_path, monkeypatch, kitchen_batch_size=2)
    day = "2026-11-16"
    day_dir = data_root / "days" / day
    kit_dir = cctv_root / "aqara_video" / "kitchen" / "20261116"
    liv_dir = cctv_root / "aqara_video" / "living" / "20261116"
    kit_dir.mkdir(parents=True, exist_ok=True)
    liv_dir.mkdir(parents=True, exist_ok=True)

    discovered = []
    for i in range(4):
        rel = f"aqara_video/kitchen/20261116/k_{i:02d}.mp4"
        fp = cctv_root / rel
        fp.write_bytes(f"K{i}".encode("utf-8"))
        discovered.append({
            "source_rel": rel, "source_root": "cctv", "day": day, "camera": "cam_kitchen",
            "wall_ts": f"{day}T10:0{i}:00", "size": fp.stat().st_size, "mtime": 1794800000.0 + i,
        })
    liv_rel = "aqara_video/living/20261116/liv_00.mp4"
    liv_fp = cctv_root / liv_rel
    liv_fp.write_bytes(b"LIVING")
    discovered.append({
        "source_rel": liv_rel, "source_root": "cctv", "day": day, "camera": "cam_living",
        "wall_ts": f"{day}T11:00:00", "size": liv_fp.stat().st_size, "mtime": 1794800100.0,
    })
    dm.db.upsert_discovered_files(discovered)

    for i in range(4):
        dm.db.mark_file_done(f"aqara_video/kitchen/20261116/k_{i:02d}.mp4", {
            "status": "done", "duration": 20.0, "width": 1280, "height": 720, "has_audio": False,
            "active_s": 5, "person_s": 5, "t1_mode": "verified_child_only", "t1_rel": None, "t1_bytes": 0,
            "max_box_score": 0.9 - i * 0.05, "audio_summary": {"segments": [[1.0, 7.0]]},
            "candidates": [{"t": 4.0, "score": 0.9 - i * 0.05, "box": [0.2, 0.2, 0.6, 0.7]}], "bursts": [],
        })
    dm.db.mark_file_done(liv_rel, {
        "status": "done", "duration": 20.0, "width": 1280, "height": 720, "has_audio": False,
        "active_s": 10, "person_s": 10, "t1_mode": "always",
        "t1_rel": f"days/{day}/t1/cam_living/liv_00.mp4", "t1_bytes": 100,
        "max_box_score": 0.95, "audio_summary": {"segments": [[2.0, 12.0]]},
        "candidates": [{"t": 6.0, "score": 0.95, "box": [0.2, 0.2, 0.6, 0.7], "luma": 120.0, "sat": 35.0, "sharp": 80.0}],
        "bursts": [],
    })

    day_dir.mkdir(parents=True, exist_ok=True)
    hl_rel = f"days/{day}/highlight_{day.replace('-', '')}.mp4"
    (data_root / hl_rel).write_bytes(b"PREVIEW_HL")
    (day_dir / "summary.json").write_text(json.dumps({
        "day": day, "done_files": 5, "highlight_rel": hl_rel,
        "daily_story": {"film_title_zh": "预览", "shots": [{"shot_id": "S01"}]},
        "photos": [], "timelapses": {},
    }, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [_fake_jpeg(i + 1) for i in range(len(times))])
    monkeypatch.setattr(daemon_mod.med, "concat_mp4_list", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon_mod.med, "promote_t0_keeper", lambda *_a, **_kw: (True, "sha"))
    monkeypatch.setattr(daemon_mod.med, "render_single_clip_with_badge", lambda *_a, **_kw: True)
    monkeypatch.setattr(daemon_mod.med, "save_photo_pair", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        daemon_mod.med,
        "encode_t1_segments",
        lambda _ff, _s, out_p, *_a, **_kw: (Path(out_p).parent.mkdir(parents=True, exist_ok=True) or Path(out_p).write_bytes(b"K_T1") or True, 6.0),
    )

    render_calls: list[int] = []

    def fake_render(_ff, shots, out_mp4: Path, *_a, **_kw):
        render_calls.append(len(shots))
        out_mp4.write_bytes(b"FINAL_HL")
        out_mp4.with_suffix(".jpg").write_bytes(_fake_jpeg(9))
        return {"shots": len(shots), "duration_s": 88.0}

    monkeypatch.setattr(daemon_mod.med, "render_smooth_film_nas", fake_render)

    def _stub_ask(tag: str, *_a, **_kw):
        if str(tag).startswith("dir_script|"):
            return {
                "film_title_zh": "终审短片",
                "closing_epilogue_zh": "温暖收尾",
                "edit_list": [{"shot_id": "S01", "chapter_zh": "第二幕 · 小小探索家", "trim_dur_s": 6.0, "shot_scale": "medium", "transition": "chapter_dissolve", "lyric_zh": "开心探索"}],
                "_provider": "p_pro", "_model": "gemini-4-carbon",
            }
        return {
            "child_clear": "yes", "face_clear": "yes", "edge_clipped": "no", "awake_active": "yes",
            "moment_worth": 5, "usable_for_highlight": "yes", "face_visibility": "clear_front",
            "best_in_rel_s": 1.0, "best_out_rel_s": 7.0, "shot_scale": "medium", "pan_style": "gentle_push",
            "cinematic_score": 5, "director_note_zh": "开心互动",
            "face_or_expression_clear": "yes", "composition_good": "yes", "adult_back_or_awkward": "no",
            "photo_score": 2, "excitement_score": 4, "categories": ["happy_interaction"],
            "child_box": [0.2, 0.2, 0.6, 0.7], "_provider": "p_fast", "_model": "gemini-4-hydrogen",
        }

    monkeypatch.setattr(dm.vlm, "ask", _stub_ask)

    mid = dm.finalize_day(day, "P1_BACKFILL", final=True, force_curate=False)
    assert mid["kitchen_unreviewed_remaining"] == 2
    assert mid["highlight_rel"] == hl_rel
    assert render_calls == []

    done_res = dm.finalize_day(day, "P1_BACKFILL", final=True, force_curate=False)
    assert done_res["kitchen_unreviewed_remaining"] == 0
    assert len(render_calls) == 1
