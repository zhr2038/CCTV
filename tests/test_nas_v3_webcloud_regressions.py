import json
import sys
import threading
import tomllib
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))

from nas_growth import app as app_mod, cloud_vlm, curation_queue, daemon, web_server
from nas_growth.db import ArchiveDB


def _v3_config_path() -> Path:
    base = Path(__file__).resolve().parents[1] / "nas-growth-v3"
    local_cfg = base / "config.toml"
    return local_cfg if local_cfg.is_file() else (base / "config.example.toml")


def test_bedside_checks_each_requested_frame_before_decoding(monkeypatch):
    calls = []
    monkeypatch.setattr(daemon, "extract_jpegs_ram", lambda ff, src, offsets, **kw: calls.append(offsets) or [b"JPEG"])
    cfg = {"crib": {"role": "bedside_crib", "family": "xiaomi_bundle", "day_hours": [7, 22]}}
    row = {"camera": "crib", "wall_ts": "2026-11-17T12:00:00"}
    src = Path("00_20261117215958_20261117221703.mp4")
    dm = SimpleNamespace(ff="ffmpeg")
    assert daemon.GrowthDaemon._extract_cloud_frames(dm, row, src, [0.5, 1.0], cfg) == [b"JPEG"]
    assert daemon.GrowthDaemon._extract_cloud_frames(dm, row, src, [0.5, 2.0], cfg) == []
    assert daemon.GrowthDaemon._extract_cloud_frames(dm, row, Path("unknown.mp4"), [0.0], cfg) == []
    assert calls == [[0.5, 1.0]]


def test_failed_batch_item_can_use_an_already_idle_provider(tmp_path, monkeypatch):
    providers = {name: {"model": name, "secret_file": "stub"} for name in ["first", "second"]}
    cfg = {"vlm": {"tiers": {"fast": list(providers)}, "providers": providers}}
    router = cloud_vlm.CloudVLMRouter(cfg, ArchiveDB(tmp_path / "test.db"), {"stub": "unit-test"})
    second_result = threading.Event()
    calls = []

    def invoke(name, provider, cache_key, tier, tag, images, prompt):
        calls.append((name, tag))
        if name == "first":
            assert second_result.wait(2)
            raise RuntimeError("Per-item provider failure without cooldown")
        second_result.set()
        return {"review": tag}

    monkeypatch.setattr(router, "_invoke_provider", invoke)
    tasks = [(i, str(i), [bytes([i])], "review") for i in range(2)]
    assert router.ask_batch(tasks) == {i: {"review": str(i)} for i in range(2)}
    assert sorted(calls) == [("first", "0"), ("second", "0"), ("second", "1")]


def test_dashboard_reads_last_verified_audit_without_scanning_sources(tmp_path, monkeypatch):
    db = ArchiveDB(tmp_path / "test.db")
    monkeypatch.setattr(web_server.lc, "evaluate_source_retention", lambda *a, **k: pytest.fail("web ran retention"))
    saved = {"cutoff_date": "2026-01-01", "audit": {"checked_at": "2026-01-01T12:00:00+08:00"}, "eligible_files": 5}
    old = web_server._build_lifecycle_snapshot(tmp_path, db, {}, {"state": {"lifecycle_status": saved}})
    assert old["eligible_files"] == 5 and old["audit"] == saved["audit"]
    pending = web_server._build_lifecycle_snapshot(tmp_path, db, {}, {"state": {}})
    assert pending["can_prune"] is False


def test_day_detail_queries_its_milestones_without_recent_list_limit(tmp_path, monkeypatch):
    db = ArchiveDB(tmp_path / "test.db")
    monkeypatch.setattr(db, "query_milestones", lambda *a, **kw: pytest.fail("day used recent list"))
    monkeypatch.setattr(db, "get_dashboard_snapshot", lambda: pytest.fail("day scanned global totals"))
    day = "2026-01-20"
    db.upsert_milestone({
        "id": "older", "day": day, "camera": "cam1", "category": "walking", "source_rel": "original.mp4",
        "start_s": 0, "end_s": 5, "caption_zh": "扶走", "score": 4, "video_rel": "v2/highlights/older.mp4",
    })
    payload = web_server._build_day_payload(day, tmp_path, db, {})
    assert [item["id"] for item in payload["milestones"]] == ["older"]


def test_config_hot_reload_and_study_camera(tmp_path, monkeypatch):
    prod_cfg_path = _v3_config_path()
    with prod_cfg_path.open("rb") as fh:
        prod_cfg = tomllib.load(fh)
    assert prod_cfg["scheduler"]["workers"] == 4
    study_cam = next(c for c in prod_cfg["cameras"].values() if c.get("role") == "study_or_room")
    assert study_cam["name"] == "书房"
    assert study_cam["relative_root"].startswith("aqara_video/")
    assert study_cam["enabled"] is True

    # Test resolve_config_path prefers /app/conf only when default path passed and override exists
    assert app_mod.resolve_config_path("/custom/config.toml") == Path("/custom/config.toml")

    cfg_file = tmp_path / "config.toml"
    data_root = tmp_path / "data"
    cctv_root = tmp_path / "cctv"
    vault_root = tmp_path / "vault"
    data_root.mkdir(parents=True)
    cctv_root.mkdir(parents=True)
    vault_root.mkdir(parents=True)

    cfg_file.write_text(
        f"""[project]
name = "Hot Reload Test"
birthday = "2025-12-17"
cctv_root = "{cctv_root.as_posix()}"
vault_root = "{vault_root.as_posix()}"
data_root = "{data_root.as_posix()}"

[scheduler]
workers = 4

[asr]
enabled = false
yolo_onnx = "{(tmp_path / 'yolo.onnx').as_posix()}"
yamnet_model = "{(tmp_path / 'yamnet.tflite').as_posix()}"
yamnet_class_map = "{(tmp_path / 'map.csv').as_posix()}"

[vlm.tiers]
fast = []
pro = []

[vlm.providers]

[auto_discovery]
enabled = false

[cameras."cam_a"]
name = "旧机位"
family = "aqara"
relative_root = "aqara_video/cam_a"
enabled = true
role = "living_room"
zoom_factor = 1.2
t1_res = 720
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(daemon.med, "check_vaapi", lambda *_a, **_kw: False)
    monkeypatch.setattr(daemon.med, "resolve_font", lambda *_a, **_kw: None)
    monkeypatch.setattr(daemon.det, "YoloOnnxDetector", lambda *_a, **_kw: object())
    monkeypatch.setattr(daemon.det, "YamnetLite", lambda *_a, **_kw: object())

    cfg = app_mod.load_config(cfg_file)
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    dm = daemon.GrowthDaemon(cfg, db, ff_bin="ffmpeg", fp_bin="ffprobe", config_path=cfg_file)
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, dm, cfg)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        st1 = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/monitor", timeout=5).read().decode("utf-8"))
        assert st1["camera_names"]["cam_a"] == "旧机位"
        assert "lumi3.8ea8ae7afa2e6240" not in st1["camera_names"]

        cfg_file.write_text(
            cfg_file.read_text(encoding="utf-8")
            + '\n[cameras."lumi3.8ea8ae7afa2e6240"]\nname = "书房"\nfamily = "aqara"\nrelative_root = "aqara_video/lumi3.8ea8ae7afa2e6240"\nenabled = true\nrole = "study_or_room"\nzoom_factor = 1.25\nt1_res = 720\n',
            encoding="utf-8",
        )
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ctl/reload_config",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        rel_res = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert rel_res["reloaded"] is True
        assert rel_res["camera_names"]["lumi3.8ea8ae7afa2e6240"] == "书房"

        st2 = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/monitor", timeout=5).read().decode("utf-8"))
        assert st2["camera_names"]["lumi3.8ea8ae7afa2e6240"] == "书房"
    finally:
        srv.shutdown()
        srv.server_close()


def test_curation_queue_clear_backoff(tmp_path, monkeypatch):
    db = ArchiveDB(tmp_path / "queue.db")
    now = [1000.0]
    monkeypatch.setattr(curation_queue.time, "time", lambda: now[0])
    q = curation_queue.CurationQueue(db)
    q.request("2026-11-18", "P0_LIVE", True, 1892)
    day, job = q.next_ready()
    q.finish(day, job, "TimeoutExpired")
    assert q.next_ready() is None
    assert "retry_at" in q.jobs["2026-11-18"] and "error" in q.jobs["2026-11-18"]
    q.clear_backoff("2026-11-18")
    q2 = curation_queue.CurationQueue(db)
    assert q2.next_ready()[0] == "2026-11-18"
    assert "retry_at" not in q2.jobs["2026-11-18"]
    assert "error" not in q2.jobs["2026-11-18"]


def test_finalize_day_spreads_full_day_and_orders_shots_chronologically(tmp_path, monkeypatch):
    day = "2026-11-18"
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    monkeypatch.setattr(daemon.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon.med, "check_vaapi", lambda *a, **kw: False)
    monkeypatch.setattr(daemon.med, "resolve_font", lambda *a, **kw: None)
    monkeypatch.setattr(daemon.med, "concat_mp4_list", lambda *a, **kw: False)
    monkeypatch.setattr(daemon.med, "promote_t0_keeper", lambda *a, **kw: (True, "sha"))
    monkeypatch.setattr(daemon.med, "render_single_clip_with_badge", lambda *a, **kw: True)
    monkeypatch.setattr(daemon, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [b"JPG"] * len(times))

    captured_shots: list[dict] = []

    def fake_render(_ff, shots, out_mp4: Path, *a, **kw):
        captured_shots.extend(shots)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"HIGHLIGHT")
        return {"file": out_mp4.name, "shots": len(shots), "duration_s": 30.0, "bytes": 9}

    monkeypatch.setattr(daemon.med, "render_smooth_film_nas", fake_render)

    cfg = {
        "project": {"cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {"asr_min_host_mem_available_kib": 999_999_999},
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"enabled": False, "yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False},
        "cameras": {
            "bed": {"name": "宝宝床头", "family": "xiaomi_bundle", "relative_root": "xiaomi_camera_videos/bed", "enabled": True, "role": "bedside_crib", "day_hours": [7, 22], "zoom_factor": 1.25, "milestone_watch": True, "daytime_photos": False},
            "room": {"name": "宝宝房顶部", "family": "aqara", "relative_root": "aqara_video/room", "enabled": True, "role": "bedroom_ceiling", "zoom_factor": 1.4, "milestone_watch": True, "daytime_photos": False},
        },
        "vlm": {"tiers": {"fast": ["g"], "pro": ["g"]}, "providers": {}},
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    samples = [
        # Off-hours high-scoring candidates that must yield to 8 waking-hour (07:00-20:59) clips
        ("room", "aqara_video/room/20261118/001900.mp4", f"{day}T00:19:00", 30.0, 5.0, 0.999),
        ("room", "aqara_video/room/20261118/213000.mp4", f"{day}T21:30:00", 30.0, 5.0, 0.998),
        # Bedside bundle collision: 07:26:19 + 1345s = 07:48:44 vs 07:49:03 + 164s = 07:51:47 (< 900s apart)
        ("bed", "xiaomi_camera_videos/bed/00_20261118072619_20261118074903.mp4", f"{day}T07:26:19", 1500.0, 1345.0, 0.90),
        ("bed", "xiaomi_camera_videos/bed/00_20261118074903_20261118075933.mp4", f"{day}T07:49:03", 1500.0, 164.0, 0.95),
        # Mid-morning and afternoon candidates with lower scores than evening cluster
        ("room", "aqara_video/room/20261118/103000.mp4", f"{day}T10:30:00", 30.0, 5.0, 0.85),
        ("room", "aqara_video/room/20261118/153000.mp4", f"{day}T15:30:00", 30.0, 5.0, 0.88),
        # High-scoring evening cluster (19:50 included in top 8 and given exc=1 to test sleep filtering)
        ("room", "aqara_video/room/20261118/181000.mp4", f"{day}T18:10:00", 30.0, 5.0, 0.99),
        ("room", "aqara_video/room/20261118/183000.mp4", f"{day}T18:30:00", 30.0, 5.0, 0.98),
        ("room", "aqara_video/room/20261118/185000.mp4", f"{day}T18:50:00", 30.0, 5.0, 0.93),
        ("room", "aqara_video/room/20261118/191000.mp4", f"{day}T19:10:00", 30.0, 5.0, 0.96),
        ("room", "aqara_video/room/20261118/193000.mp4", f"{day}T19:30:00", 30.0, 5.0, 0.94),
        ("room", "aqara_video/room/20261118/195000.mp4", f"{day}T19:50:00", 30.0, 5.0, 0.955),
    ]
    for cam, rel, wts, dur, t_cand, score in samples:
        p = cctv_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"MP4")
        db.upsert_discovered_files([{"source_rel": rel, "source_root": "cctv", "day": day, "camera": cam, "wall_ts": wts, "size": 1000, "mtime": 1.0}])
        db.mark_file_done(rel, {
            "duration": dur, "width": 1920, "height": 1080, "has_audio": False,
            "active_s": 10, "person_s": 10, "t1_bytes": 100, "t2_bytes": 100,
            "candidates": [{"t": t_cand, "score": score, "box": [0.2, 0.2, 0.6, 0.8], "sharp": 80.0, "sat": 35.0, "luma": 120.0, "motion": 0.1}],
            "bursts": [],
        })

    dm = daemon.GrowthDaemon(cfg, db)
    vlm_tags: list[str] = []

    def fake_ask(tag, *_a, **_kw):
        vlm_tags.append(tag)
        return {
            "child_clear": "yes", "categories": ["toy_play"], "independence": "independent",
            "excitement_score": 1 if "195000.mp4" in tag else 4,
            "child_box": [0.2, 0.2, 0.6, 0.8], "caption_zh": "开心玩耍",
        }

    monkeypatch.setattr(dm.vlm, "ask", fake_ask)

    summary = dm.finalize_day(day, "P0_LIVE", final=False)
    assert summary["highlight_rel"] == f"days/{day}/highlight_20261118.mp4"
    assert not any("001900.mp4" in t or "213000.mp4" in t for t in vlm_tags)
    assert any("195000.mp4" in t for t in vlm_tags)
    assert len(captured_shots) == 7
    assert not any("00_20261118072619" in sh["source"] for sh in captured_shots)
    assert not any("195000.mp4" in sh["source"] for sh in captured_shots)
    shot_times = [sh["wall_time_hms"] for sh in captured_shots]
    assert all("×" not in sh["sub_zh"] for sh in captured_shots)
    assert any(t.startswith("07:51:") for t in shot_times)
    assert any(t.startswith("10:30:") for t in shot_times)
    assert any(t.startswith("15:30:") for t in shot_times)
    assert any(t.startswith("19:30:") for t in shot_times)
    assert shot_times == sorted(shot_times)


def test_read_timeout_does_not_double_retry_and_enters_cooldown(tmp_path, monkeypatch):
    import time

    db = ArchiveDB(tmp_path / "timeout.db")
    cfg = {
        "vlm": {
            "request_timeout_seconds": 45,
            "cooldown_seconds": 600,
            "tiers": {"fast": ["v_slow"]},
            "providers": {
                "v_slow": {
                    "kind": "openai",
                    "model": "m-slow",
                    "base_url": "http://127.0.0.1:9/v1",
                    "secret_file": "sec",
                    "timeout_seconds": 12,
                }
            },
        }
    }
    router = cloud_vlm.CloudVLMRouter(cfg, db, secret_overrides={"sec": "k"})
    post_calls: list[tuple[float, float]] = []

    def raise_timeout(self_sess, url, **kw):
        post_calls.append(kw.get("timeout"))
        raise cloud_vlm.requests.ReadTimeout("read timed out")

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", raise_timeout)
    with pytest.raises(RuntimeError):
        router.ask("ph|slow|1.0", [b"JPG"], "prompt", tier="fast")
    assert post_calls == [(10, 12.0)]
    assert db.get_vlm_cooldown("v_slow") - time.time() > 500


def test_openai_fallback_models_and_four_family_fair_batch(tmp_path, monkeypatch):
    import base64
    import io
    import time
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from PIL import Image

    cfg_path = _v3_config_path()
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode("ascii")
    sa1 = json.dumps({"client_email": "sa1@p1.iam.gserviceaccount.com", "private_key": pem, "project_id": "p1"})
    sa2 = json.dumps({"client_email": "sa2@p2.iam.gserviceaccount.com", "private_key": pem, "project_id": "p2"})
    overrides = {}
    for pc in cfg["vlm"]["providers"].values():
        sf = pc.get("secret_file", "")
        overrides[sf] = sa1 if "sa_1" in sf else (sa2 if "sa_2" in sf else "dummy-key")

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 180, 210)).save(buf, "JPEG")
    tiny_jpg = buf.getvalue()

    db = ArchiveDB(tmp_path / "fair.db")
    router = cloud_vlm.CloudVLMRouter(cfg, db, secret_overrides=overrides)
    lock = threading.Lock()
    gpt_req_models: list[str] = []

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
            return StubResp(200, {"access_token": "ya29.tok", "expires_in": 3600})
        if "aiplatform.googleapis.com" in url or "generativelanguage.googleapis.com" in url:
            ans = json.dumps({"child_clear": "yes", "photo_score": 4, "box_2d": [100, 100, 700, 700], "caption_zh": "ok"})
            return StubResp(200, {"candidates": [{"content": {"parts": [{"text": ans}]}, "finishReason": "STOP"}]})
        m = str(payload.get("model") or "")
        if m == "gpt-6.2-sol":
            with lock:
                gpt_req_models.append(m)
            return StubResp(400, {"error": {"code": "model_not_found", "message": "unknown provider for model gpt-6.2-sol"}})
        if m == "gpt-6.1-sol":
            with lock:
                gpt_req_models.append(m)
        if m == "DeepSeek-V4.1-Flash-1017":
            return StubResp(404, {"error": {"message": "not found"}})
        ans = json.dumps({"child_clear": "yes", "photo_score": 4, "box_2d": [100, 100, 700, 700], "caption_zh": "ok"})
        return StubResp(200, {"choices": [{"message": {"content": ans}, "finish_reason": "stop"}]})

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)
    tasks = [(i, f"ph|fair_{i}|1.0", [tiny_jpg, tiny_jpg], cloud_vlm.PROMPT_PHOTO) for i in range(4)]
    res = router.ask_batch(tasks, tier="fast")
    assert len(res) == 4
    assert {v["_model"] for v in res.values()} == {
        "qwen4-max",
        "DeepSeek-V4.1-Flash-1017",
        "gpt-6.2-sol",
        "gemini-4-hydrogen",
    }
    assert "gpt-6.2-sol" in gpt_req_models and "gpt-6.1-sol" in gpt_req_models
    assert router._resolved_models["gpt_62_sol"][0] == "gpt-6.1-sol"
