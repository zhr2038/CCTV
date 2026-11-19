from __future__ import annotations

import base64
import datetime as dt
import io
import json
import sys
import threading
import tomllib
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))

from nas_growth import cloud_vlm, daemon as daemon_mod, family_faces as ff, web_server
from nas_growth.db import ArchiveDB


def _v3_config_path() -> Path:
    base = Path(__file__).resolve().parents[1] / "nas-growth-v3"
    local_cfg = base / "config.toml"
    return local_cfg if local_cfg.is_file() else (base / "config.example.toml")


def _make_jpg(w: int = 240, h: int = 180, color: tuple[int, int, int] = (180, 145, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


def _unit_vec(seed: int, dim: int = 512) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    v = v / float(np.linalg.norm(v))
    return [round(float(x), 6) for x in v.tolist()]


def _noisy_vec(base: list[float], seed: int, noise_scale: float = 0.08) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(base, dtype=np.float32) + rng.normal(0.0, noise_scale, size=len(base)).astype(np.float32)
    v = v / float(np.linalg.norm(v))
    return [round(float(x), 6) for x in v.tolist()]


def test_real_onnx_engine_smoke_and_missing_graceful_fallback(tmp_path):
    models_dir = Path(__file__).resolve().parents[1] / "nas-growth-v3" / "models" / "face"
    det_p = models_dir / "det_500m.onnx"
    rec_p = models_dir / "w600k_mbf.onnx"
    assert det_p.is_file() and rec_p.is_file()

    eng = ff.ScrfdArcFaceEngine(det_p, rec_p)
    assert eng.available is True
    blank = np.zeros((360, 480, 3), dtype=np.uint8)
    assert eng.detect_faces(blank) == []

    kps = [[180, 140], [240, 140], [210, 175], [185, 210], [235, 210]]
    emb = eng.extract_embedding(np.full((360, 480, 3), 120, dtype=np.uint8), kps)
    assert emb is not None and len(emb) == 512
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3

    # Graceful degradation when model paths do not exist
    missing_eng = ff.ScrfdArcFaceEngine(tmp_path / "nope_det.onnx", tmp_path / "nope_rec.onnx")
    assert missing_eng.available is False
    assert missing_eng.detect_faces(blank) == []
    assert missing_eng.extract_embedding(blank, kps) is None


def test_default_members_date_filtering_and_embedding_match(tmp_path):
    db = ArchiveDB(tmp_path / "fam.sqlite3")
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    svc = ff.FamilyFaceService(db, tmp_path / "data", cfg)

    all_m = db.list_family_members()
    labels = [m["label_zh"] for m in all_m]
    assert labels == ["妈妈", "爸爸", "奶奶", "爷爷", "姥姥", "姥爷", "月嫂", "小姨"]

    # Date window check for 月嫂 derived from config active_from .. active_until
    yuesao = next(m for m in all_m if m["member_id"] == "yuesao")
    active_day = str(yuesao["active_from"])
    expired_day = (dt.date.fromisoformat(str(yuesao["active_until"])) + dt.timedelta(days=60)).isoformat()
    assert "月嫂" in svc.active_roles_for_day(active_day)
    assert "月嫂" not in svc.active_roles_for_day(expired_day)

    mama_v = _unit_vec(101)
    nainai_v = _unit_vec(202)
    baba_v = _unit_vec(303)
    db.add_family_face_sample({
        "sample_id": "s_mama", "member_id": "mama", "thumb_rel": "t1.jpg",
        "embedding": mama_v, "det_score": 0.92,
    })
    db.add_family_face_sample({
        "sample_id": "s_nainai", "member_id": "nainai", "thumb_rel": "t2.jpg",
        "embedding": nainai_v, "det_score": 0.91,
    })
    db.add_family_face_sample({
        "sample_id": "s_baba", "member_id": "baba", "thumb_rel": "t3.jpg",
        "embedding": baba_v, "det_score": 0.93,
    })

    gallery = db.get_family_face_gallery(day="2026-11-19")
    res_nainai = ff.match_embedding(_noisy_vec(nainai_v, 7, 0.04), gallery)
    assert res_nainai["status"] == "high"
    assert res_nainai["label_zh"] == "奶奶"

    res_unknown = ff.match_embedding(_unit_vec(999), gallery)
    assert res_unknown["status"] == "unknown"
    assert res_unknown["label_zh"] is None


def test_sanitize_caption_kinship_protects_speech_and_fixes_grandma_mama_confusion():
    # 1. Local ArcFace confirms 奶奶 -> corrects VLM hallucinated 妈妈 to 奶奶
    out1 = ff.sanitize_caption_kinship(
        "午后爬近妈妈怀里，妈妈温柔地拍拍宝宝",
        verified_local_roles={"奶奶"},
        verified_vlm_roles={"妈妈"},  # conflicting VLM guess must be overridden
        has_enrolled_ref=True,
    )
    assert out1 == "午后爬近奶奶怀里，奶奶温柔地拍拍宝宝"

    # 2. Local ArcFace confirms 爷爷 -> corrects VLM hallucinated 爸爸 to 爷爷
    out2 = ff.sanitize_caption_kinship(
        "早上在爸爸怀里玩钥匙",
        verified_local_roles={"爷爷"},
        has_enrolled_ref=True,
    )
    assert out2 == "早上在爷爷怀里玩钥匙"

    # 3. Baby's own speech quotes ("叫了一声「妈妈」", "冒出短短一声「爸」") stay untouched!
    out3 = ff.sanitize_caption_kinship(
        "奶奶抱着时宝宝清脆叫了一声「妈妈」，又对妈妈笑了笑",
        verified_local_roles={"奶奶"},
        has_enrolled_ref=True,
    )
    assert "叫了一声「妈妈」" in out3
    assert "又对奶奶笑了笑" in out3

    out3b = ff.sanitize_caption_kinship(
        "在小床上爬着爬着，冒出短短一声「爸」，旁边的爸爸妈妈笑起来",
        verified_local_roles=set(),
        has_enrolled_ref=False,
    )
    assert "冒出短短一声「爸」" in out3b
    assert "爸爸妈妈" not in out3b and "家人在" in out3b or "旁边的家人笑起来" in out3b

    # 4. Inactive role (月嫂 in November) neutralized to 家人
    out4 = ff.sanitize_caption_kinship(
        "月嫂阿姨陪宝宝在客厅垫上玩",
        verified_local_roles={"月嫂"},
        active_roles={"妈妈", "爸爸", "奶奶", "爷爷", "姥姥", "姥爷"},
        has_enrolled_ref=True,
    )
    assert "月嫂" not in out4 and "家人" in out4

    # 5. Unenrolled / unverified day never hallucinates 妈妈 or 爸爸
    out5 = ff.sanitize_caption_kinship(
        "牵着妈妈跨过床沿，爸爸在厨房看火",
        verified_local_roles=set(),
        verified_vlm_roles={"妈妈"},
        has_enrolled_ref=False,
        strict_unverified=True,
    )
    assert "妈妈" not in out5 and "爸爸" not in out5 and "家人" in out5

    # 6. Local ArcFace confirms 小姨 -> corrects VLM hallucinated 妈妈 to 小姨 and preserves VLM 小姨 role
    out6 = ff.sanitize_caption_kinship(
        "妈妈逗得宝宝咯咯笑",
        verified_local_roles={"小姨"},
        verified_vlm_roles={"妈妈"},
        active_roles={"妈妈", "爸爸", "奶奶", "爷爷", "姥姥", "姥爷", "小姨"},
        has_enrolled_ref=True,
    )
    assert out6 == "小姨逗得宝宝咯咯笑"
    assert ff.extract_high_conf_vlm_roles({"caregivers_seen": [{"role": "小姨", "confidence": "high"}]}) == {"小姨"}


def test_vlm_router_passes_ref_sheet_on_multi_image_and_skips_collage(tmp_path, monkeypatch):
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    db = ArchiveDB(tmp_path / "vlm_ref.sqlite3")
    overrides = {pc["secret_file"]: "test-key" for pc in cfg["vlm"]["providers"].values()}
    router = cloud_vlm.CloudVLMRouter(cfg, db, secret_overrides=overrides)

    seen: list[tuple[str, int]] = []

    class StubResp:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            ans = json.dumps({"child_clear": "yes", "photo_score": 5, "caption_zh": "奶奶抱着宝宝"})
            return {"choices": [{"message": {"content": ans}, "finish_reason": "stop"}]}

    def fake_post(self_sess, url: str, **kw):
        payload = kw["json"]
        content = payload["messages"][0]["content"]
        n_imgs = sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
        seen.append((payload["model"], n_imgs))
        return StubResp()

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)
    frame_jpg = _make_jpg(120, 80, (120, 160, 200))
    ref1 = _make_jpg(160, 120, (200, 160, 120))
    ref2 = _make_jpg(160, 120, (210, 150, 130))

    # Multi-image OpenAI endpoint (qwen4_max) receives 1 ref sheet + 2 frames = 3 images
    pc_qwen = cfg["vlm"]["providers"]["qwen4_max"]
    router._call_openai_compat("qwen4_max", pc_qwen, "k", router._session_for("qwen4_max"), [frame_jpg, frame_jpg], "hi", reference_images=[("家庭成员识别对照卡", ref1)])
    assert seen[-1] == ("qwen4-max", 3)

    # Single-image collage endpoint skips reference sheet so collage stays single image
    pc_ds = cfg["vlm"]["providers"]["deepseek_v41_flash"]
    router._call_openai_compat("deepseek_v41_flash", pc_ds, "k", router._session_for("deepseek_v41_flash"), [frame_jpg, frame_jpg], "hi", reference_images=[("家庭成员识别对照卡", ref1)])
    assert seen[-1] == ("DeepSeek-V4.1-Flash-1017", 1)

    # Cache key changes when reference sheet changes
    k1 = router._cache_key_for("ph|1", [frame_jpg], "prompt", reference_images=[("ref", ref1)])
    k2 = router._cache_key_for("ph|1", [frame_jpg], "prompt", reference_images=[("ref", ref2)])
    k0 = router._cache_key_for("ph|1", [frame_jpg], "prompt")
    assert k1 != k2 and k1 != k0


def test_daemon_and_director_integrate_face_match_ref_sheet_and_guard(tmp_path, monkeypatch):
    day = "2026-11-19"
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    jpg = _make_jpg(160, 90)

    monkeypatch.setattr(daemon_mod.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon_mod.med, "check_vaapi", lambda *a, **kw: False)
    monkeypatch.setattr(daemon_mod.med, "resolve_font", lambda *a, **kw: None)
    monkeypatch.setattr(daemon_mod.med, "concat_mp4_list", lambda *a, **kw: False)
    monkeypatch.setattr(daemon_mod.med, "promote_t0_keeper", lambda *a, **kw: (True, "sha"))
    monkeypatch.setattr(daemon_mod.med, "render_single_clip_with_badge", lambda *a, **kw: True)
    monkeypatch.setattr(daemon_mod, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [jpg] * len(times))

    def fake_photo_pair(_ff, _src, _t, _w, _h, _box, _z, f_out: Path, fr_out: Path, **_kw):
        f_out.parent.mkdir(parents=True, exist_ok=True)
        f_out.write_bytes(jpg)
        fr_out.write_bytes(jpg)
        return True

    monkeypatch.setattr(daemon_mod.med, "save_photo_pair", fake_photo_pair)

    rendered_call: dict = {}

    def fake_render(_ff, shots, out_mp4: Path, *a, **kw):
        rendered_call["shots"] = list(shots)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"MP4")
        out_mp4.with_suffix(".jpg").write_bytes(jpg)
        return {"file": out_mp4.name, "poster": out_mp4.with_suffix(".jpg").name, "shots": len(shots), "duration_s": 88.0, "bytes": 3}

    monkeypatch.setattr(daemon_mod.med, "render_smooth_film_nas", fake_render)

    cfg = {
        "project": {"birthday": "2025-12-17", "cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {"asr_min_host_mem_available_kib": 999_999_999},
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"enabled": False, "yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False},
        "face_id": {"enabled": True, "attach_reference_sheet": True, "strict_unverified_to_family": True},
        "cameras": {
            "room": {"name": "宝宝房顶部", "family": "aqara", "relative_root": "aqara_video/room", "enabled": True, "role": "bedroom_ceiling", "zoom_policy": "smart_child_crop", "zoom_factor": 1.4, "milestone_watch": True, "daytime_photos": True},
        },
        "vlm": {
            "tiers": {"fast": ["p_fast"], "pro": ["p_pro"], "director": ["p_dir"]},
            "providers": {"p_fast": {"model": "fast-m", "secret_file": "s"}, "p_pro": {"model": "pro-m", "secret_file": "s"}, "p_dir": {"model": "dir-m", "secret_file": "s"}},
        },
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    for hr, mm in [(8, 10), (10, 20), (14, 15), (17, 30)]:
        rel = f"aqara_video/room/{day.replace('-', '')}/{hr:02d}{mm:02d}00.mp4"
        p = cctv_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"MP4")
        db.upsert_discovered_files([{"source_rel": rel, "source_root": "cctv", "day": day, "camera": "room", "wall_ts": f"{day}T{hr:02d}:{mm:02d}:00", "size": 2000, "mtime": 1.0}])
        db.mark_file_done(rel, {
            "duration": 20.0, "width": 1920, "height": 1080, "has_audio": True,
            "active_s": 10, "person_s": 10, "t1_bytes": 100, "t2_bytes": 100,
            "audio_summary": {"max_laughter": 0.6, "max_child_voc": 0.4},
            "candidates": [{"t": 6.0, "score": 0.9, "box": [0.2, 0.2, 0.6, 0.7], "sharp": 85.0, "sat": 35.0, "luma": 130.0}],
            "bursts": [],
        })

    dm = daemon_mod.GrowthDaemon(cfg, db)
    # Enroll one visual reference sample for 奶奶 so reference sheet is created
    dm.face_service.enroll_image_bytes("nainai", jpg, source_note="test_nainai.jpg", allow_visual_only=True)

    # Mock identify_images to return 奶奶 for afternoon clips and unknown for morning clips
    def fake_identify(images_jpeg, *, day=None, child_box=None):
        return {
            "verified_roles": {"奶奶"},
            "possible_roles": set(),
            "unknown_faces": 0,
            "details": [{"label_zh": "奶奶", "member_id": "nainai", "confidence": "high", "best_score": 0.61, "avg_score": 0.61, "margin": 0.22, "frames_hit": len(images_jpeg)}],
            "active_roster_labels": ["妈妈", "爸爸", "奶奶", "爷爷", "姥姥", "姥爷"],
            "inactive_roster_labels": ["月嫂"],
            "active_member_notes": ["奶奶(宝宝奶奶)"],
            "has_enrolled_members": True,
            "has_enrolled_embeddings": True,
        }

    monkeypatch.setattr(dm.face_service, "identify_images", fake_identify)

    seen_prompts_with_refs: list[tuple[str, bool, str]] = []

    def fake_ask(tag: str, images: list[bytes], prompt: str, tier: str = "fast", reference_images=None):
        seen_prompts_with_refs.append((tag, bool(reference_images), prompt))
        if tag.startswith("ph|"):
            return {
                "child_clear": "yes", "face_or_expression_clear": "yes", "composition_good": "yes",
                "adult_back_or_awkward": "no", "osd_box_on_child": "no", "photo_score": 5,
                "box_2d": [200, 200, 700, 700],
                # Deliberately return hallucinated 妈妈 to test correction to 奶奶!
                "caption_zh": "妈妈温柔地抱着宝宝笑",
            }
        if tag.startswith("hlc|"):
            return {"child_clear": "yes", "face_clear": "yes", "edge_clipped": "no", "awake_active": "yes", "moment_worth": 5, "brief_zh": "开心互动"}
        if tag.startswith("dl_edit|"):
            return {
                "usable_for_highlight": "yes", "child_clear": "yes", "face_visibility": "clear_front",
                "best_in_rel_s": 0.5, "best_out_rel_s": 6.5, "shot_scale": "wide",
                "pan_style": "gentle_push", "box_2d": [200, 200, 700, 700],
                "categories": ["happy_interaction"], "independence": "assisted",
                "emotion_tag": "joy", "cinematic_score": 5,
                # Deliberately emit wrong 妈妈 despite local face match saying 奶奶
                "director_note_zh": "宝宝爬向妈妈怀里大笑",
                "_model": "gemini-4-carbon",
            }
        if tag.startswith("dir_script|"):
            return {
                "film_title_zh": "温暖的一天",
                "logline_zh": "今天宝宝开心地扑进妈妈怀里。",
                "joy_summary_zh": "在妈妈身边玩得最开心。",
                "growth_changes_zh": "迈步更稳。",
                "closing_epilogue_zh": "慢慢长大。",
                "edit_list": [
                    {"shot_id": "S01", "chapter_zh": "第一幕 · 晨光初醒", "trim_dur_s": 6.0, "shot_scale": "wide", "lyric_zh": "清早扑进妈妈怀里笑"}
                ],
                "_model": "gpt-6.2-sol",
            }
        raise AssertionError(f"Unexpected tag: {tag}")

    monkeypatch.setattr(dm.vlm, "ask", fake_ask)
    summary = dm.finalize_day(day, "P0_LIVE", final=True)

    # Reference sheet was attached to VLM calls and local match prompt block injected
    assert any(has_ref and "家庭成员身份与称呼红线" in prm for _tag, has_ref, prm in seen_prompts_with_refs)
    assert any("奶奶（高置信确认" in prm for _tag, _hr, prm in seen_prompts_with_refs)

    # Photos and daily story must have replaced hallucinated 妈妈 -> 奶奶 and tagged people=["奶奶"]
    assert summary["photos"]
    assert summary["photos"][0]["caption_zh"] == "奶奶温柔地抱着宝宝笑"
    assert summary["photos"][0]["people"] == ["奶奶"]

    story = summary["daily_story"]
    assert "奶奶" in story["logline_zh"] and "妈妈" not in story["logline_zh"]
    assert "奶奶" in story["joy_summary_zh"] and "妈妈" not in story["joy_summary_zh"]
    assert all("妈妈" not in sh["lyric_zh"] and "奶奶" in sh["lyric_zh"] for sh in story["shots"])
    assert story["shots"][0]["people"] == ["奶奶"]


def test_web_family_face_api_endpoints(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    cfg["project"]["data_root"] = str(data_root)
    fsvc = ff.FamilyFaceService(db, data_root, cfg)

    # Create a sample photo in archive so /api/family/discover can inspect
    ph_dir = data_root / "days" / "2026-11-19" / "photos"
    ph_dir.mkdir(parents=True, exist_ok=True)
    ph_dir.joinpath("full_01_客厅_100000.jpg").write_bytes(_make_jpg(320, 240))

    stub_daemon = SimpleNamespace(
        is_paused=lambda: False,
        face_service=fsvc,
        reload_config_if_changed=lambda force=False: False,
    )
    srv = web_server.create_server("127.0.0.1", 0, data_root, db, stub_daemon, cfg)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    try:
        # 1. GET /api/family/members
        m_res = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/family/members").read().decode("utf-8"))
        assert len(m_res["members"]) == 8
        assert "xiaoyi" in {m["member_id"] for m in m_res["members"]}
        assert m_res["engine"]["engine_available"] is True

        # 2. POST /api/family/member (update notes for 奶奶)
        req_upd = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/family/member",
            data=json.dumps({"member_id": "nainai", "label_zh": "奶奶", "notes": "银色短发戴眼镜"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        upd_res = json.loads(urllib.request.urlopen(req_upd).read().decode("utf-8"))
        assert upd_res["member"]["notes"] == "银色短发戴眼镜"

        # 3. POST /api/family/sample (visual-only enrollment)
        b64_jpg = base64.b64encode(_make_jpg(240, 240)).decode("ascii")
        req_samp = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/family/sample",
            data=json.dumps({"member_id": "nainai", "image_b64": b64_jpg, "allow_visual_only": True, "source_note": "life.jpg"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        s_res = json.loads(urllib.request.urlopen(req_samp).read().decode("utf-8"))
        sid = s_res["sample"]["sample_id"]
        assert (data_root / s_res["sample"]["thumb_rel"]).is_file()
        assert (data_root / "family_faces" / "reference_sheet_latest.jpg").is_file()

        # 4. GET /api/family/members now returns reference_sheet_rel
        m_res2 = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/family/members").read().decode("utf-8"))
        assert m_res2["reference_sheet_rel"] == "family_faces/reference_sheet_latest.jpg"

        # 5. POST /api/family/discover + /api/family/candidate/assign (mock candidate detection cleanly)
        fake_emb = _unit_vec(55)
        monkeypatch.setattr(fsvc._get_engine(), "detect_faces", lambda *a, **kw: [{"score": 0.91, "box_px": [40, 30, 140, 150], "box_norm": [0.12, 0.12, 0.44, 0.62], "kps": [[70, 70], [110, 70], [90, 95], [75, 120], [105, 120]]}])
        monkeypatch.setattr(fsvc._get_engine(), "extract_embedding", lambda *a, **kw: fake_emb)

        req_disc = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/family/discover",
            data=json.dumps({"max_days": 5, "max_photos": 10}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        d_res = json.loads(urllib.request.urlopen(req_disc).read().decode("utf-8"))
        assert len(d_res["candidates"]) == 1
        cid = d_res["candidates"][0]["cand_id"]

        req_assign = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/family/candidate/assign",
            data=json.dumps({"cand_id": cid, "member_id": "laolao"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        a_res = json.loads(urllib.request.urlopen(req_assign).read().decode("utf-8"))
        assert a_res["assigned"] is True
        assert a_res["sample"]["has_embedding"] is True

        # 6. POST /api/family/sample/delete
        req_del = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/family/sample/delete",
            data=json.dumps({"sample_id": sid}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        del_res = json.loads(urllib.request.urlopen(req_del).read().decode("utf-8"))
        assert del_res["deleted"] is True

        # 7. Root HTML includes the new 家人认脸 tab
        html_txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
        assert "家人认脸" in html_txt and "/api/family/sample" in html_txt
    finally:
        srv.shutdown()
        srv.server_close()


def test_pose_gate_blocks_back_of_head_false_match_and_ranks_ref_sheet(tmp_path, monkeypatch):
    # 1. Pose gate geometry checks
    bad_back = {"box_px": [100, 100, 300, 350], "kps": [[150, 180], [156, 182], [152, 220], [148, 260], [155, 260]]}
    good_front = {"box_px": [100, 100, 300, 350], "kps": [[150, 180], [230, 180], [190, 220], [160, 265], [220, 265]]}
    assert ff.is_reliable_face_pose(bad_back, min_eye_dx_ratio=0.20) is False
    assert ff.is_reliable_face_pose(good_front, min_eye_dx_ratio=0.20) is True

    # 2. Collapsed back-of-head face must NEVER falsely match 姥姥 even if its embedding equals 姥姥
    db = ArchiveDB(tmp_path / "pose_gate.sqlite3")
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    fsvc = ff.FamilyFaceService(db, tmp_path / "data", cfg)
    laolao_v = _unit_vec(777)
    db.add_family_face_sample({"sample_id": "s_ll", "member_id": "laolao", "thumb_rel": "t_ll.jpg", "embedding": laolao_v, "det_score": 0.78})
    eng = fsvc._get_engine()
    monkeypatch.setattr(eng, "detect_faces", lambda *a, **kw: [{
        "score": 0.62, "box_px": [100, 100, 300, 350], "box_norm": [0.2, 0.2, 0.6, 0.7],
        "kps": [[150, 180], [154, 181], [151, 220], [148, 260], [154, 260]],
    }])
    monkeypatch.setattr(eng, "extract_embedding", lambda *a, **kw: laolao_v)
    res_back = fsvc.identify_images([_make_jpg(480, 360)], day="2026-11-19")
    assert "姥姥" not in res_back["verified_roles"] and "姥姥" not in res_back["possible_roles"]

    # 3. Reference sheet ranks higher-det_score embedded samples ahead of newer low-score samples
    data_root = tmp_path / "ref_data"
    db2 = ArchiveDB(tmp_path / "ref_rank.sqlite3")
    fsvc2 = ff.FamilyFaceService(db2, data_root, cfg)
    (data_root / "family_faces" / "nainai").mkdir(parents=True, exist_ok=True)
    # Create older high-score red, newer low-score blue, older high-score green
    (data_root / "family_faces" / "nainai" / "old_hi1.jpg").write_bytes(_make_jpg(120, 120, (220, 20, 20)))
    (data_root / "family_faces" / "nainai" / "old_hi2.jpg").write_bytes(_make_jpg(120, 120, (20, 220, 20)))
    (data_root / "family_faces" / "nainai" / "new_lo.jpg").write_bytes(_make_jpg(120, 120, (20, 20, 220)))
    db2.add_family_face_sample({"sample_id": "s_hi1", "member_id": "nainai", "thumb_rel": "family_faces/nainai/old_hi1.jpg", "embedding": _unit_vec(1), "det_score": 0.85, "created_at": 100.0})
    db2.add_family_face_sample({"sample_id": "s_hi2", "member_id": "nainai", "thumb_rel": "family_faces/nainai/old_hi2.jpg", "embedding": _unit_vec(2), "det_score": 0.82, "created_at": 110.0})
    db2.add_family_face_sample({"sample_id": "s_lo", "member_id": "nainai", "thumb_rel": "family_faces/nainai/new_lo.jpg", "embedding": _unit_vec(3), "det_score": 0.56, "created_at": 200.0})
    sheet = fsvc2.get_reference_sheet(day=None)
    assert sheet is not None
    with Image.open(io.BytesIO(sheet)) as sim:
        arr = np.asarray(sim.convert("RGB"))
        # Neither red nor green high-det_score tile was displaced by low-det_score blue tile (high blue channel dominance)
        assert int(((arr[:, :, 2] > 180) & (arr[:, :, 0] < 60)).sum()) == 0


def test_cold_adult_nouns_sanitized_and_film_compiler_uses_face_id(tmp_path, monkeypatch):
    # 1. Cold surveillance nouns ("一名男子" / "一位女士" / "家长") resolved to confirmed role or warmed to "家人"
    assert (
        ff.sanitize_caption_kinship(
            "客厅内，一名男子站在扭扭车旁，一位女士正扶着宝宝",
            verified_local_roles={"爷爷"},
            has_enrolled_ref=True,
        )
        == "客厅内，爷爷站在扭扭车旁，家人正扶着宝宝"
    )
    assert (
        ff.sanitize_caption_kinship(
            "家长抱着宝宝站在冰箱旁，另一名女子递来水杯",
            verified_local_roles=set(),
            has_enrolled_ref=True,
            strict_unverified=True,
        )
        == "家人抱着宝宝站在冰箱旁，另一位家人递来水杯"
        or "家人抱着宝宝站在冰箱旁" in ff.sanitize_caption_kinship(
            "家长抱着宝宝站在冰箱旁，一名女子递来水杯",
            verified_local_roles=set(),
            has_enrolled_ref=True,
            strict_unverified=True,
        )
    )
    cleaned_cold = ff.sanitize_caption_kinship(
        "家长抱着宝宝站在冰箱旁，一名女子递来水杯",
        verified_local_roles=set(),
        has_enrolled_ref=True,
        strict_unverified=True,
    )
    assert all(bad not in cleaned_cold for bad in ("家长", "女子", "男子", "女士")) and "家人" in cleaned_cold

    # 1b. Same-group elder guard + expanded cold terms ("大人" / "成年男性" / "两位成年女性" / "两名成人")
    elder_fixed = ff.sanitize_caption_kinship(
        "爷爷抱着宝宝看姥姥忙碌，随后两位成年女性、一名成年男性与两名成人一起在旁照料，宝宝贴着大人的下巴。",
        verified_local_roles={"爷爷"},
        verified_vlm_roles={"姥姥"},
        has_enrolled_ref=True,
        strict_unverified=True,
        day_confirmed_roles={"奶奶", "爷爷"},
    )
    assert "姥姥" not in elder_fixed and "奶奶" in elder_fixed
    assert all(bad not in elder_fixed for bad in ("成年女性", "成年男性", "成人", "大人"))

    adult_single = ff.sanitize_caption_kinship(
        "宝宝贴着大人的下巴笑",
        verified_local_roles={"奶奶"},
        has_enrolled_ref=True,
    )
    assert adult_single == "宝宝贴着奶奶的下巴笑"

    adult_unverified = ff.sanitize_caption_kinship(
        "宝宝贴着大人的下巴笑",
        verified_local_roles=set(),
        has_enrolled_ref=True,
        strict_unverified=True,
    )
    assert adult_unverified == "宝宝贴着家人的下巴笑"

    # 2. film_compiler._audit_unverified_candidates runs Face-ID, passes reference_images, and fixes hallucinated 妈妈 -> 奶奶
    from nas_growth import film_compiler as fc

    db = ArchiveDB(tmp_path / "fc_face.sqlite3")
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    fsvc = ff.FamilyFaceService(db, tmp_path / "data", cfg)
    fsvc.enroll_image_bytes("nainai", _make_jpg(180, 180), source_note="test_nn.jpg", allow_visual_only=True)

    jpg = _make_jpg(320, 240)
    monkeypatch.setattr(fc.cvlm, "extract_jpegs_ram", lambda *a, **kw: [jpg, jpg, jpg, jpg])
    monkeypatch.setattr(
        fsvc,
        "identify_images",
        lambda imgs, *, day=None, child_box=None: {
            "verified_roles": {"奶奶"},
            "possible_roles": set(),
            "unknown_faces": 0,
            "details": [{"label_zh": "奶奶", "member_id": "nainai", "confidence": "high", "best_score": 0.62, "avg_score": 0.62, "margin": 0.25, "frames_hit": 2}],
            "active_roster_labels": ["妈妈", "爸爸", "奶奶", "爷爷", "姥姥", "姥爷", "小姨"],
            "inactive_roster_labels": ["月嫂"],
            "active_member_notes": ["奶奶(宝宝奶奶)"],
            "has_enrolled_members": True,
            "has_enrolled_embeddings": True,
        },
    )

    seen_calls: list[tuple[bool, str]] = []

    class StubVLM:
        vlm_cfg = {"providers": {"p": {"model": "m"}}}

        def ask_batch(self, tasks, tier="director"):
            out = {}
            for item in tasks:
                idx_k, _tag, _imgs, prm = item[:4]
                ref_imgs = item[4] if len(item) >= 5 else None
                seen_calls.append((bool(ref_imgs), prm))
                out[idx_k] = {
                    "usable_for_highlight": "yes",
                    "child_clear": "yes",
                    "face_visibility": "clear_front",
                    "best_in_rel_s": 0.2,
                    "best_out_rel_s": 5.2,
                    "shot_scale": "wide",
                    "pan_style": "gentle_push",
                    "cinematic_score": 5,
                    "director_note_zh": "妈妈抱着宝宝开心大笑",
                    "caregivers_seen": [{"role": "妈妈", "confidence": "high"}],
                }
            return out

    audited = fc._audit_unverified_candidates(
        [
            {
                "id": "cand_1",
                "day": "2026-11-19",
                "camera": "cam1",
                "camera_name": "客厅",
                "source_rel": "aqara_video/cam1/20261119/120000.mp4",
                "source_path": "dummy.mp4",
                "source_start_s": 1.0,
                "source_end_s": 7.0,
                "clip_total_dur_s": 15.0,
                "dur_s": 6.0,
                "caption_zh": "客厅片段",
                "pre_verified": False,
            }
        ],
        "ffmpeg",
        StubVLM(),
        False,
        face_service=fsvc,
    )
    assert len(audited) == 1
    assert seen_calls and seen_calls[0][0] is True and "家庭成员身份与称呼红线" in seen_calls[0][1]
    assert audited[0]["people"] == ["奶奶"]
    assert audited[0]["verified_local_roles"] == ["奶奶"]
    assert audited[0]["caption_zh"] == "奶奶抱着宝宝开心大笑"


def test_vlm_router_prefers_multi_image_over_collage_when_ref_sheet_attached(tmp_path, monkeypatch):
    full_cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "nas-growth-v3" / "config.toml").read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(full_cfg))
    cfg["vlm"]["providers"] = {
        "deepseek_v41_flash": dict(full_cfg["vlm"]["providers"]["deepseek_v41_flash"]),
        "qwen4_max": dict(full_cfg["vlm"]["providers"]["qwen4_max"]),
    }
    # Put single-image collage provider first intentionally to verify ref-aware routing priority
    cfg["vlm"]["parallel_tiers"] = {"fast": ["deepseek_v41_flash", "qwen4_max"]}
    cfg["vlm"]["tiers"] = {"fast": ["deepseek_v41_flash", "qwen4_max"]}
    cfg["vlm"]["max_slots_per_model"] = 1

    db = ArchiveDB(tmp_path / "vlm_ref_route.sqlite3")
    overrides = {pc["secret_file"]: "test-key" for pc in cfg["vlm"]["providers"].values()}
    router = cloud_vlm.CloudVLMRouter(cfg, db, secret_overrides=overrides)

    calls: list[tuple[str, int]] = []

    class StubResp:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            ans = json.dumps({"child_clear": "yes", "photo_score": 4, "caption_zh": "爷爷抱着宝宝"})
            return {"choices": [{"message": {"content": ans}, "finish_reason": "stop"}]}

    def fake_post(self_sess, url: str, **kw):
        payload = kw["json"]
        content = payload["messages"][0]["content"]
        n_imgs = sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
        calls.append((payload["model"], n_imgs))
        return StubResp()

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)
    frame_jpg = _make_jpg(160, 120, (120, 160, 200))
    ref_jpg = _make_jpg(180, 140, (200, 160, 120))
    refs = [("家庭成员识别对照卡", ref_jpg)]

    # 1. ask() with reference_images prefers multi-image qwen4-max (receives 2 images: ref + frame)
    router.ask("ph|ask_ref|1.0", [frame_jpg], "prompt", tier="fast", reference_images=refs)
    assert calls[-1] == ("qwen4-max", 2)

    # 2. ask_batch() with reference_images sends all ref-attached tasks to qwen4-max, zero to DeepSeek
    calls.clear()
    ref_tasks = [
        (i, f"ph|batch_ref_{i}|1.0", [frame_jpg], f"prompt_{i}", refs)
        for i in range(4)
    ]
    res_ref = router.ask_batch(ref_tasks, tier="fast")
    assert len(res_ref) == 4
    assert all(model == "qwen4-max" and n_imgs == 2 for model, n_imgs in calls)

    # 3. ask_batch() without reference_images still lets DeepSeek participate in work-stealing
    calls.clear()
    plain_tasks = [
        (i, f"ph|batch_plain_{i}|1.0", [frame_jpg], f"plain_{i}")
        for i in range(6)
    ]
    res_plain = router.ask_batch(plain_tasks, tier="fast")
    assert len(res_plain) == 6
    assert any(model == "DeepSeek-V4.1-Flash-1017" for model, _n in calls)

    # 4. When qwen4_max is cooling down, ref-attached ask_batch gracefully falls back to DeepSeek with 1 image
    db.record_vlm_result("qwen4_max", "qwen4-max", False, 0.0, "rate_limited", cooldown_s=600.0)
    calls.clear()
    res_fb = router.ask_batch([(0, "ph|batch_fb|1.0", [frame_jpg], "prompt_fb", refs)], tier="fast")
    assert len(res_fb) == 1
    assert calls == [("DeepSeek-V4.1-Flash-1017", 1)]


def test_face_id_config_thresholds_and_hires_forwarding(tmp_path, monkeypatch):
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    fcfg = cfg["face_id"]
    assert float(fcfg["det_score_threshold"]) == 0.42
    assert int(fcfg["min_face_px"]) == 24
    assert int(fcfg["hi_res_edge"]) == 1920

    db = ArchiveDB(tmp_path / "thr.sqlite3")
    fsvc = ff.FamilyFaceService(db, tmp_path / "data", cfg)
    db.add_family_face_sample({
        "sample_id": "s_yeye", "member_id": "yeye", "thumb_rel": "t.jpg",
        "embedding": _unit_vec(42), "det_score": 0.88,
    })

    seen_calls: list[dict[str, float | int]] = []
    eng = fsvc._get_engine()
    monkeypatch.setattr(eng, "detect_faces", lambda bgr, **kw: seen_calls.append(kw) or [])

    fsvc.identify_images([_make_jpg(640, 360)], day="2026-11-19")
    assert len(seen_calls) == 1
    assert seen_calls[0]["score_thr"] == 0.42
    assert seen_calls[0]["min_face_px"] == 24
