from __future__ import annotations

import io
import json
import sys
import threading
import tomllib
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))

from nas_growth import cloud_vlm, curation_queue, daily_director, daemon, media_nas, web_server
from nas_growth.db import ArchiveDB


def _tiny_jpg(color: tuple[int, int, int] = (140, 180, 210)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (96, 54), color).save(buf, "JPEG")
    return buf.getvalue()


def _v3_config_path() -> Path:
    base = Path(__file__).resolve().parents[1] / "nas-growth-v3"
    local_cfg = base / "config.toml"
    return local_cfg if local_cfg.is_file() else (base / "config.example.toml")


def test_director_tier_enforces_high_thinking_without_affecting_pro_or_fast(tmp_path, monkeypatch):
    cfg = tomllib.loads(_v3_config_path().read_text(encoding="utf-8"))
    db = ArchiveDB(tmp_path / "dir_vlm.sqlite3")
    overrides = {pc["secret_file"]: "test-key" for pc in cfg["vlm"]["providers"].values()}
    router = cloud_vlm.CloudVLMRouter(cfg, db, secret_overrides=overrides)

    seen_reqs: list[tuple[str, dict]] = []

    class StubResp:
        def __init__(self, code: int, body: dict):
            self.status_code = code
            self._body = body
            self.text = json.dumps(body, ensure_ascii=False)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._body

    def fake_post(self_sess, url: str, **kw):
        payload = kw.get("json") or {}
        seen_reqs.append((url, {"payload": payload, "timeout": kw.get("timeout")}))
        if "generativelanguage.googleapis.com" in url:
            ans = json.dumps({"usable_for_highlight": "yes", "child_clear": "yes", "cinematic_score": 5})
            return StubResp(200, {"candidates": [{"content": {"parts": [{"text": ans}]}, "finishReason": "STOP"}]})
        ans = json.dumps({"usable_for_highlight": "yes", "child_clear": "yes", "cinematic_score": 5})
        return StubResp(200, {"choices": [{"message": {"content": ans}, "finish_reason": "stop"}]})

    monkeypatch.setattr(cloud_vlm.requests.Session, "post", fake_post)
    jpg = _tiny_jpg()

    r1 = router.ask("dl_edit|c1|1.0", [jpg, jpg], "edit", tier="director")
    r2 = router.ask("dl_edit|c2|1.0", [jpg, jpg], "edit", tier="director")
    r_pro = router.ask("ms|c3|1.0", [jpg], "milestone", tier="pro")

    assert {r1["_model"], r2["_model"]} == {"gemini-4-carbon", "gpt-6.2-sol"}
    gem_dir = [req for url, req in seen_reqs if "gemini-4-carbon" in url and req["timeout"] == (10, 150.0)]
    gpt_dir = [req for url, req in seen_reqs if "chat/completions" in url and req["payload"].get("reasoning_effort") == "high"]
    assert gem_dir and gem_dir[0]["payload"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert gem_dir[0]["payload"]["generationConfig"]["maxOutputTokens"] == 8192
    assert gpt_dir and gpt_dir[0]["payload"]["max_tokens"] == 4096 and "temperature" not in gpt_dir[0]["payload"]
    assert gpt_dir[0]["timeout"] == (10, 150.0)
    assert r_pro["_model"] in {"gpt-6.2-sol", "qwen4-max", "gemini-4-carbon"}

    # Ensure no retired kimi models remain anywhere
    assert not any("kimi" in p for p in cfg["vlm"]["providers"])


def test_director_rejects_partial_edge_clipped_and_curates_full_day_four_acts(tmp_path, monkeypatch):
    day = "2026-11-18"
    cctv_root = tmp_path / "cctv"
    data_root = tmp_path / "data"
    jpg = _tiny_jpg()

    monkeypatch.setattr(daemon.det, "YoloOnnxDetector", lambda *a, **kw: object())
    monkeypatch.setattr(daemon.det, "YamnetLite", lambda *a, **kw: object())
    monkeypatch.setattr(daemon.med, "check_vaapi", lambda *a, **kw: False)
    monkeypatch.setattr(daemon.med, "resolve_font", lambda *a, **kw: None)
    monkeypatch.setattr(daemon.med, "concat_mp4_list", lambda *a, **kw: False)
    monkeypatch.setattr(daemon.med, "promote_t0_keeper", lambda *a, **kw: (True, "sha"))
    monkeypatch.setattr(daemon.med, "render_single_clip_with_badge", lambda *a, **kw: True)
    monkeypatch.setattr(daemon, "extract_jpegs_ram", lambda _ff, _s, times, **kw: [jpg] * len(times))

    rendered_call: dict = {}

    def fake_render(_ff, shots, out_mp4: Path, *a, **kw):
        rendered_call["shots"] = list(shots)
        rendered_call["film_header"] = kw.get("film_header")
        rendered_call["epilogue"] = kw.get("closing_epilogue_zh")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"HIGHLIGHT_MP4")
        out_mp4.with_suffix(".jpg").write_bytes(jpg)
        return {"file": out_mp4.name, "poster": out_mp4.with_suffix(".jpg").name, "shots": len(shots), "duration_s": 96.4, "bytes": 13}

    monkeypatch.setattr(daemon.med, "render_smooth_film_nas", fake_render)

    cfg = {
        "project": {"birthday": "2025-12-17", "cctv_root": str(cctv_root), "vault_root": str(tmp_path / "vault"), "data_root": str(data_root), "timezone": "Asia/Shanghai"},
        "scheduler": {"asr_min_host_mem_available_kib": 999_999_999},
        "media": {"prefer_vaapi": False, "font_candidates": []},
        "asr": {"enabled": False, "yolo_onnx": "x", "yamnet_model": "y", "yamnet_class_map": "z", "model_dir": "m"},
        "auto_discovery": {"enabled": False},
        "cameras": {
            "room": {"name": "宝宝房顶部", "family": "aqara", "relative_root": "aqara_video/room", "enabled": True, "role": "bedroom_ceiling", "zoom_policy": "smart_child_crop", "zoom_factor": 1.4, "milestone_watch": True, "daytime_photos": False},
            "kit": {"name": "厨房 G350", "family": "aqara", "relative_root": "aqara_video/kit", "enabled": True, "role": "kitchen_4k", "zoom_policy": "never", "zoom_factor": 1.0, "milestone_watch": True, "daytime_photos": False},
        },
        "vlm": {
            "tiers": {"fast": ["p_fast"], "pro": ["p_pro"], "director": ["p_dir"]},
            "providers": {"p_fast": {"model": "fast-m", "secret_file": "s"}, "p_pro": {"model": "pro-m", "secret_file": "s"}, "p_dir": {"model": "dir-m", "secret_file": "s"}},
        },
    }
    db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")

    # Populate 22 files spanning 07:30 -> 20:30 plus one bad edge-clipped shot & one partial-face shot
    file_specs = []
    for hr_i, (hh, mm) in enumerate([
        (7, 35), (7, 50), (8, 20), (8, 55), (9, 25), (10, 10), (10, 45), (11, 15),
        (11, 50), (12, 30), (13, 40), (14, 20), (14, 53), (15, 30), (16, 15), (16, 56),
        (17, 25), (18, 6), (18, 45), (19, 20), (19, 58), (20, 18),
    ]):
        cam = "kit" if hh in (11, 15) and mm in (50, 30) else "room"
        rel = f"aqara_video/{cam}/{day.replace('-', '')}/{hh:02d}{mm:02d}00.mp4"
        file_specs.append((cam, rel, f"{day}T{hh:02d}:{mm:02d}:00", 28.0, hh, mm))

    for cam, rel, wts, dur, hh, mm in file_specs:
        p = cctv_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"MP4")
        db.upsert_discovered_files([{"source_rel": rel, "source_root": "cctv", "day": day, "camera": cam, "wall_ts": wts, "size": 2000, "mtime": 1.0}])
        is_laugh = hh in (7, 16, 19)
        db.mark_file_done(rel, {
            "duration": dur, "width": 2560, "height": 1440, "has_audio": True,
            "active_s": 14, "person_s": 14, "t1_bytes": 100, "t2_bytes": 100,
            "audio_summary": {"max_laughter": 0.91 if is_laugh else 0.05, "max_child_voc": 0.72 if hh == 18 else 0.1},
            "candidates": [{"t": 8.0, "score": 0.92, "box": [0.25, 0.25, 0.65, 0.78], "sharp": 90.0, "sat": 35.0, "luma": 130.0, "motion": 0.1}],
            "bursts": [],
        })

    dm = daemon.GrowthDaemon(cfg, db)
    calls_by_tag: list[tuple[str, str]] = []

    def fake_ask(tag: str, images: list[bytes], prompt: str, tier: str = "fast"):
        calls_by_tag.append((tier, tag))
        if tag.startswith("hlc|"):
            if "145300.mp4" in tag:
                return {"child_clear": "partial", "face_clear": "side", "edge_clipped": "yes", "awake_active": "yes", "moment_worth": 4, "brief_zh": "角落露半边脸"}
            return {"child_clear": "yes", "face_clear": "yes", "edge_clipped": "no", "awake_active": "yes", "moment_worth": 5, "box_2d": [200, 220, 780, 720], "brief_zh": "正面清晰互动"}
        if tag.startswith("dl_edit|"):
            if "142000.mp4" in tag:
                return {"usable_for_highlight": "no", "child_clear": "partial", "face_visibility": "edge_clipped", "cinematic_score": 1, "director_note_zh": "仅露边缘衣角"}
            is_kit = "kit/" in tag
            return {
                "usable_for_highlight": "yes",
                "child_clear": "yes",
                "face_visibility": "clear_front",
                "best_in_rel_s": 1.2,
                "best_out_rel_s": 7.8,
                "shot_scale": "wide" if is_kit else "closeup",
                "pan_style": "gentle_push",
                "box_2d": [220, 240, 780, 740],
                "privacy_exposed": "no",
                "privacy_box_2d": [0, 0, 0, 0],
                "categories": ["kitchen_visit"] if is_kit else ["happy_interaction"],
                "independence": "assisted" if is_kit else "independent",
                "emotion_tag": "joy",
                "cinematic_score": 5,
                "director_note_zh": "厨房里对着镜头好奇张望" if is_kit else "开心抬手咯咯大笑",
                "_model": "gemini-4-carbon",
            }
        if tag.startswith("dir_script|"):
            # Return fewer shots on purpose (~40s) to verify auto-fill tops up film duration to >= 84s!
            return {
                "film_title_zh": "第337天 · 晨起甜笑与晚间欢呼",
                "logline_zh": "从清晨醒来到晚上洗漱前的开心互动，宝宝今天笑容格外多。",
                "joy_summary_zh": "清晨互动大笑与晚间和家人玩耍时笑得最开心。",
                "growth_changes_zh": "眼神交流更专注，肢体探索更舒展自信。",
                "closing_epilogue_zh": "第337天，慢慢长大的每秒都算数。",
                "edit_list": [
                    {"shot_id": f"S{i:02d}", "chapter_zh": "第一幕 · 晨光初醒" if i <= 2 else "第三幕 · 今日欢笑与高光", "trim_dur_s": 6.5, "shot_scale": "closeup", "transition": "warm_dissolve", "lyric_zh": f"第{i}个被好好记住的温暖瞬间"}
                    for i in range(1, 7)
                ],
                "_model": "gpt-6.2-sol",
            }
        raise AssertionError(f"Unexpected tag: {tag}")

    monkeypatch.setattr(dm.vlm, "ask", fake_ask)
    summary = dm.finalize_day(day, "P0_LIVE", final=False)

    assert any(tier == "director" and tag.startswith("dl_edit|") for tier, tag in calls_by_tag)
    assert any(tier == "director" and tag.startswith("dir_script|") for tier, tag in calls_by_tag)

    shots = rendered_call["shots"]
    assert not any("145300.mp4" in s["source_rel"] or "142000.mp4" in s["source_rel"] for s in shots)
    assert len(shots) >= 13
    assert daily_director._estimate_net_duration_s(shots) >= 84.0
    assert rendered_call["film_header"]["title_zh"] == "第337天 · 晨起甜笑与晚间欢呼"
    assert rendered_call["epilogue"] == "第337天，慢慢长大的每秒都算数。"
    assert shots[0]["chapter_zh"] == "第一幕 · 晨光初醒"
    assert any(s["chapter_zh"].startswith("第四幕") for s in shots)

    script_file = data_root / "days" / day / "daily_script.json"
    assert script_file.is_file()
    assert summary["daily_story"]["film_title_zh"] == "第337天 · 晨起甜笑与晚间欢呼"
    assert summary["highlight_duration_s"] == 96.4


def test_web_curate_day_endpoint_and_daily_story_card(tmp_path):
    data_root = tmp_path / "data"
    day = "2026-11-18"
    day_dir = data_root / "days" / day
    day_dir.mkdir(parents=True)
    (day_dir / "highlight_20261118.mp4").write_bytes(b"MP4")
    (day_dir / "highlight_20261118.jpg").write_bytes(b"JPG")
    (day_dir / "summary.json").write_text(json.dumps({
        "day": day,
        "highlight_rel": f"days/{day}/highlight_20261118.mp4",
        "highlight_duration_s": 102.5,
        "highlight_shots_count": 16,
        "updated_at": "2026-11-18T22:10:00+08:00",
        "daily_story": {
            "film_title_zh": "第337天 · 清晨甜笑与一声“爸”",
            "logline_zh": "今天宝宝从早上醒来就一路笑意盈盈，傍晚还清脆叫出“爸”。",
            "joy_summary_zh": "抓到小红车和傍晚亲子互动时笑声连连",
            "growth_changes_zh": "主动发音更清晰",
            "shots": [{"index": 1, "shot_id": "S01", "chapter_zh": "第一幕 · 晨光初醒", "time": "07:51:20", "camera": "宝宝房顶部", "dur_s": 6.5, "lyric_zh": "迎着晨光醒来露出甜甜一笑"}],
        },
        "photos": [],
        "timelapses": {},
    }, ensure_ascii=False), encoding="utf-8")

    db = ArchiveDB(data_root / "state" / "web.sqlite3")
    q = curation_queue.CurationQueue(db)
    triggered: list[str] = []

    class StubDaemon:
        def is_paused(self): return False
        def set_paused(self, _v): pass
        def request_day_curation(self, d, tier=None):
            triggered.append(d)
            return q.force_request(d, tier or "P0_LIVE", True, 1899)

    srv = web_server.create_server("127.0.0.1", 0, data_root, db, StubDaemon(), {"project": {"birthday": "2025-12-17"}})
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        day_json = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/day/{day}", timeout=5).read().decode("utf-8"))
        assert day_json["daily_story"]["film_title_zh"] == "第337天 · 清晨甜笑与一声“爸”"
        assert day_json["highlight_poster_rel"] == f"days/{day}/highlight_20261118.jpg"

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ctl/curate_day",
            data=json.dumps({"day": day}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        res = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
        assert triggered == [day] and "force_ts" in res
    finally:
        srv.shutdown()
        srv.server_close()


def test_cinema_overlay_patches_and_transition_durations():
    assert media_nas._transition_fade_frames({"transition": "soft_cut"}, 0.55, 30) == 6
    assert media_nas._transition_fade_frames({"transition": "warm_dissolve"}, 0.55, 30) == 16
    assert media_nas._transition_fade_frames({"transition": "chapter_dissolve"}, 0.55, 30) == 24
    # On Windows test hosts without external fonts, None degrades cleanly
    assert media_nas.render_cinema_subtitle_patch(1920, 1080, "暖光初醒", "07:51", None) is None
    stamped = daily_director.stamp_timecode_badges([_tiny_jpg()], [3.4])
    assert len(stamped) == 1 and len(stamped[0]) > 100
    sheets = daily_director.build_storyboard_sheets([
        {"shot_id": "S01", "hh_mm": "07:51", "cam_short": "房顶", "poster_jpg": _tiny_jpg()},
        {"shot_id": "S02", "hh_mm": "18:06", "cam_short": "床头", "poster_jpg": _tiny_jpg()},
    ])
    assert len(sheets) == 1 and len(sheets[0]) > 200

    # Verify long bedside bundle keeps multi-burst laugh peaks rather than jumping to far-away adult candidate
    bundle_row = {
        "duration": 900.0,
        "candidates_json": json.dumps([
            {"t": 850.0, "score": 0.9, "luma": 120.0, "sharp": 80.0, "box": [0.72, 0.1, 0.99, 0.9]},
        ]),
        "bursts_json": json.dumps([
            {"burst_start_s": 140.0, "burst_end_s": 144.0, "y_laughter": 0.85, "y_child_babble": 0.6, "snr_ratio": 9.0},
            {"burst_start_s": 420.0, "burst_end_s": 425.0, "y_laughter": 0.72, "y_child_babble": 0.5, "snr_ratio": 8.0},
        ]),
        "audio_summary_json": "{}",
    }
    b_peaks = daily_director._locate_burst_peaks_in_row(bundle_row, max_peaks=2)
    assert len(b_peaks) == 2 and round(b_peaks[0][0]) == 142 and round(b_peaks[1][0]) == 422

    # Verify assemble_final_edit_list never extends a shot beyond Stage-3 verified dur_s
    ver_mock = [{
        "shot_id": "S01",
        "row": {"source_rel": "k/191110.mp4", "duration": 10.0, "width": 3840, "height": 2160, "has_audio": True},
        "src": Path("191110.mp4"),
        "cam_id": "k",
        "cam_name": "厨房 G350",
        "cam_short": "厨房",
        "c_cfg": {"zoom_policy": "never"},
        "source_kind": "kitchen_confirmed",
        "abs_in": 0.3,
        "dur_s": 4.8,
        "max_clip_dur": 9.2,
        "shot_scale": "wide",
        "pan_style": "steady",
        "child_box": [0.2, 0.2, 0.5, 0.7],
        "privacy_box": None,
        "categories": [],
        "independence": "unclear",
        "emotion_tag": "focused",
        "cinematic_score": 4.5,
        "director_note_zh": "在厨房好奇张望",
        "hh_mm_ss": "19:11:10",
        "hh_mm": "19:11",
        "act_hint": "第四幕 · 温柔夜色",
        "wall_ep": 1795000270.0,
        "sort_key": (1795000270.0, "k/191110.mp4", 0.3),
        "poster_jpg": _tiny_jpg(),
        "boost_vocal": False,
        "speech_meta": None,
        "ans": {},
    }]
    clamped_shots, _doc = daily_director.assemble_final_edit_list(
        "2026-11-18",
        {"day_n": 337, "label_zh": "11个月"},
        ver_mock,
        {"edit_list": [{"shot_id": "S01", "trim_dur_s": 8.0, "lyric_zh": "傍晚厨房陪伴"}]},
    )
    assert len(clamped_shots) == 1 and clamped_shots[0]["dur_s"] == 4.8

    # Verify top-right chapter pill coordinate placement avoids top-left OSD area when a TTF font is available
    win_font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if win_font.is_file():
        coords, _bgr, _a = media_nas.render_chapter_pill_patch(1920, 1080, "第一幕 · 晨光初醒", str(win_font))
        y0, y1, x0, x1 = coords
        assert x0 > 1200 and x1 <= 1920 and y0 < 80
