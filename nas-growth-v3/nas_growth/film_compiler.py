"""Long-period Director & Screenwriter Film Compiler (Monthly, Seasonal, Camera Chronicle & First-Year).

Reuses the 4-stage HIGH-thinking cinema pipeline:
  1. Three-tier material pyramid (`daily_script.json` -> confirmed milestones -> done archive fallback)
  2. High-thinking multi-frame audit for any unverified historical Tier-C candidate
  3. Master Period Director (`tier="director"`) composing multi-act narrative scripts over storyboard sheets
  4. Variable-transition cinema rendering, accurate per-shot provenance offsets, atomic publish & review
"""
from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import cloud_vlm as cvlm
from . import daily_director as dd
from . import family_faces as ff_mod
from . import lifecycle as lc
from . import media_nas as med
from .cloud_vlm import is_cloud_allowed
from .db import ArchiveDB, is_confirmed_milestone_record, is_unconfirmed_refusal


PERIOD_KIND_PROFILES: dict[str, dict[str, Any]] = {
    "monthly": {
        "acts": (
            "第一幕 · 初见本月模样",
            "第二幕 · 探索与新本领",
            "第三幕 · 欢笑与初语",
            "第四幕 · 满月温柔定格",
        ),
        "target_min_s": 108.0,
        "target_max_s": 168.0,
        "min_shots": 14,
        "max_shots": 24,
        "cand_pool": 32,
        "default_shot_s": 6.4,
        "genre_zh": "四幕月龄成长短剧",
    },
    "seasonal": {
        "acts": (
            "第一幕 · 季初初见",
            "第二幕 · 日日舒展",
            "第三幕 · 欢笑与突破",
            "第四幕 · 换季长大印记",
        ),
        "target_min_s": 120.0,
        "target_max_s": 195.0,
        "min_shots": 16,
        "max_shots": 26,
        "cand_pool": 36,
        "default_shot_s": 6.6,
        "genre_zh": "四季成长珍藏大片",
    },
    "camera_chronicle": {
        "acts": (
            "第一幕 · 最初的模样",
            "第二幕 · 悄悄抽条",
            "第三幕 · 越发自在",
            "第四幕 · 如今的小大人",
        ),
        "target_min_s": 96.0,
        "target_max_s": 162.0,
        "min_shots": 12,
        "max_shots": 24,
        "cand_pool": 32,
        "default_shot_s": 6.2,
        "genre_zh": "同视角流年变迁影集",
    },
    "annual": {
        "acts": (
            "第一幕 · 初生啼声",
            "第二幕 · 翻身看世界",
            "第三幕 · 坐起与探索",
            "第四幕 · 扶站迈步",
            "第五幕 · 初语欢笑",
            "第六幕 · 周岁灯火",
        ),
        "target_min_s": 180.0,
        "target_max_s": 320.0,
        "min_shots": 24,
        "max_shots": 42,
        "cand_pool": 46,
        "default_shot_s": 6.6,
        "genre_zh": "六幕周岁成长纪录片",
    },
    "growth_chronicle": {
        "acts": (
            "第一幕 · 初生温柔",
            "第二幕 · 抬头挥手",
            "第三幕 · 翻身抓握",
            "第四幕 · 独坐看世界",
            "第五幕 · 爬行探索",
            "第六幕 · 扶站初语",
        ),
        "target_min_s": 150.0,
        "target_max_s": 230.0,
        "min_shots": 22,
        "max_shots": 32,
        "cand_pool": 44,
        "default_shot_s": 6.5,
        "genre_zh": "六幕全周期成长流年大片",
    },
}

PROMPT_PERIOD_ANCHOR_SCOUT = """你正在为“宝宝跨月成长流年大片”做历史代表片段快速初筛。这组图来自同一段昼间家庭录像，每张图左上角带有绝对秒数标签（如 +15s）。
请严格判断这段画面中是否出现清晰可见的婴幼儿（约0-14个月宝宝）和适合珍藏的温馨日常或发育表现：
1. 空床/空房间/纯黑白夜视、只有大人走动做家务、或宝宝只有模糊背影/被严重遮挡：必须将 `child_clear` 设为 `"no"` 或将 `face_visibility` 设为 `"back_of_head"`/`"occluded_no_face"`，且 `excitement_score <= 1`。
2. 换尿布/洗澡光身擦拭等仅属于日常护理且可能露隐私部位的场景：请把 `daily_care` 设为 `"yes"`。
3. 称呼铁律：除非确有证据，否则严禁臆猜“妈妈/爸爸/奶奶/爷爷”，描述里的大人统一写“家人”。
请仅返回单个标准 JSON 对象：
{
  "child_clear": "yes|partial|no",
  "face_visibility": "clear_front|clear_three_quarter|clear_action_profile|distant_clear|back_of_head|far_blurry|occluded_no_face",
  "daily_care": "yes|no",
  "best_time_s": 15.0,
  "excitement_score": 4,
  "box_2d": [ymin, xmin, ymax, xmax],
  "caption_zh": "12～22字温暖客观描述，写清宝宝状态与新变化"
}"""


def _frame_color_quality(jpg_bytes: bytes) -> tuple[float, float, float]:
    """Return (luma, color_saturation, sharpness) for a JPEG buffer to reject dark/IR/blurry frames locally."""
    if not jpg_bytes:
        return 0.0, 0.0, 0.0
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return 0.0, 0.0, 0.0
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        luma = float(np.mean(0.114 * b + 0.587 * g + 0.299 * r))
        color_sat = float(np.mean(np.abs(r - g) + np.abs(g - b)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(luma, 2), round(color_sat, 2), round(sharp, 2)
    except Exception:
        return 80.0, 15.0, 40.0


def _iter_month_strings(
    period_start: str,
    period_end: str,
    cap_date: dt.date | None = None,
) -> list[str]:
    """Return sorted YYYY-MM months intersecting [period_start, min(period_end, cap_date)]."""
    try:
        d0 = dt.date.fromisoformat(period_start[:10])
        d1 = dt.date.fromisoformat(period_end[:10])
    except ValueError:
        return []
    if cap_date is not None:
        d1 = min(d1, cap_date)
    if d1 < d0:
        return []
    out: list[str] = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append(f"{y:04d}-{m:02d}")
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _camera_role_priority(cam_cfg: dict[str, Any]) -> int:
    role = str(cam_cfg.get("role") or "")
    order = {
        "bedside_crib": 0,
        "bedroom_ceiling": 1,
        "living_room": 2,
        "study_or_room": 3,
        "kitchen_4k": 4,
        "kitchen": 5,
    }
    return order.get(role, 4)


def _index_camera_month_files(
    cctv_root: Path,
    vault_root: Path,
    cams: dict[str, Any],
    allowed_cams: set[str] | None,
    min_day: str,
    max_day: str,
    *,
    min_bytes: int = 8192,
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Read-only scan indexing daytime candidate source files grouped by YYYY-MM -> day -> camera."""
    out: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    min_ym, max_ym = min_day[:7], max_day[:7]
    min_ymd, max_ymd = min_day.replace("-", ""), max_day.replace("-", "")
    seen_rels: set[str] = set()

    def _add(ym: str, day_s: str, cid: str, rel: str, root_name: str, path: Path, wall_ts: str, hour: int, size: int) -> None:
        if rel in seen_rels or not (min_day <= day_s <= max_day) or not (8 <= hour < 19):
            return
        seen_rels.add(rel)
        out.setdefault(ym, {}).setdefault(day_s, {}).setdefault(cid, []).append({
            "source_rel": rel,
            "source_root": root_name,
            "path": path,
            "day": day_s,
            "camera": cid,
            "wall_ts": wall_ts,
            "hour": hour,
            "size": size,
        })

    for cid, c_cfg in sorted(cams.items(), key=lambda kv: _camera_role_priority(kv[1])):
        if not c_cfg.get("enabled", True):
            continue
        if allowed_cams is not None and cid not in allowed_cams:
            continue
        fam = str(c_cfg.get("family") or "aqara")
        rel_root = str(c_cfg.get("relative_root") or c_cfg.get("subdir") or "").strip("/")
        if not rel_root:
            continue
        for root_name, root_base in (("cctv", cctv_root), ("vault", vault_root)):
            cdir = root_base / rel_root
            if not cdir.is_dir():
                continue
            if fam == "aqara":
                try:
                    subdirs = sorted(cdir.iterdir(), key=lambda p: p.name)
                except OSError:
                    continue
                for ddir in subdirs:
                    ymd = ddir.name
                    if len(ymd) != 8 or not ymd.isdigit() or not ddir.is_dir():
                        continue
                    if not (min_ymd <= ymd <= max_ymd):
                        continue
                    day_s = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
                    ym = day_s[:7]
                    try:
                        entries = list(os.scandir(ddir))
                    except OSError:
                        continue
                    for ent in entries:
                        if not ent.name.lower().endswith(".mp4") or len(ent.name) < 10 or not ent.name[:6].isdigit():
                            continue
                        hh = int(ent.name[:2])
                        if not (8 <= hh < 19):
                            continue
                        try:
                            st = ent.stat()
                        except OSError:
                            continue
                        if st.st_size < min_bytes:
                            continue
                        rel = f"{rel_root}/{ymd}/{ent.name}"
                        wall_ts = f"{day_s}T{ent.name[0:2]}:{ent.name[2:4]}:{ent.name[4:6]}"
                        _add(ym, day_s, cid, rel, root_name, Path(ent.path), wall_ts, hh, int(st.st_size))
            elif fam == "xiaomi_bundle":
                try:
                    entries = list(os.scandir(cdir))
                except OSError:
                    continue
                for ent in entries:
                    if not ent.name.lower().endswith(".mp4"):
                        continue
                    m = re.match(r"^00_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:_|$)", ent.name)
                    if not m:
                        continue
                    y, mo, da, hh_s, mi_s, sc_s = m.groups()
                    hh = int(hh_s)
                    if not (8 <= hh < 19):
                        continue
                    day_s = f"{y}-{mo}-{da}"
                    ym = f"{y}-{mo}"
                    if not (min_ym <= ym <= max_ym) or not (min_day <= day_s <= max_day):
                        continue
                    try:
                        st = ent.stat()
                    except OSError:
                        continue
                    if st.st_size < min_bytes:
                        continue
                    rel = f"{rel_root}/{ent.name}"
                    wall_ts = f"{day_s}T{hh_s}:{mi_s}:{sc_s}"
                    _add(ym, day_s, cid, rel, root_name, Path(ent.path), wall_ts, hh, int(st.st_size))
            elif fam == "xiaomi_minute":
                try:
                    hdirs = sorted(cdir.iterdir(), key=lambda p: p.name)
                except OSError:
                    continue
                for hdir in hdirs:
                    hname = hdir.name
                    if len(hname) != 10 or not hname.isdigit() or not hdir.is_dir():
                        continue
                    ymd, hh_s = hname[:8], hname[8:10]
                    hh = int(hh_s)
                    if not (min_ymd <= ymd <= max_ymd) or not (8 <= hh < 19):
                        continue
                    day_s = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
                    ym = day_s[:7]
                    try:
                        entries = list(os.scandir(hdir))
                    except OSError:
                        continue
                    for ent in entries:
                        if not ent.name.lower().endswith(".mp4"):
                            continue
                        mm = re.match(r"^(\d{2})M(\d{2})S", ent.name, re.I)
                        mi_s, sc_s = (mm.group(1), mm.group(2)) if mm else ("00", "00")
                        try:
                            st = ent.stat()
                        except OSError:
                            continue
                        if st.st_size < min_bytes:
                            continue
                        rel = f"{rel_root}/{hname}/{ent.name}"
                        wall_ts = f"{day_s}T{hh_s}:{mi_s}:{sc_s}"
                        _add(ym, day_s, cid, rel, root_name, Path(ent.path), wall_ts, hh, int(st.st_size))
    return out


def _pick_month_sample_clips(
    month_days: dict[str, dict[str, list[dict[str, Any]]]],
    card_anchor_day: str | None,
    cams: dict[str, Any],
    blocked_windows: dict[str, list[tuple[float, float]]],
    excluded: set[str],
    *,
    clips_budget: int,
) -> list[dict[str, Any]]:
    """Choose representative daytime clips across anchor days, cameras, and morning/midday/afternoon."""
    if not month_days:
        return []

    def _day_score(day_s: str) -> tuple[int, int, int]:
        cam_map = month_days[day_s]
        non_kit_cams = sum(
            1 for cid in cam_map if str(cams.get(cid, {}).get("role") or "") not in ("kitchen", "kitchen_4k")
        )
        is_card = 1 if (card_anchor_day and day_s == card_anchor_day) else 0
        dist17 = -abs(int(day_s[8:10]) - 17)
        return (is_card, non_kit_cams, dist17)

    ordered_days = sorted(month_days, key=_day_score, reverse=True)[:4]
    if not ordered_days:
        return []

    bucket_picks: list[dict[str, Any]] = []
    seen_rels: set[str] = set()
    buckets = ((8, 11), (11, 15), (15, 19))

    # Pass A: cycle anchor days -> time buckets -> non-kitchen cameras -> kitchen cameras
    for role_group in ("main", "kitchen"):
        for day_s in ordered_days:
            cam_map = month_days[day_s]
            sorted_cids = sorted(cam_map, key=lambda cid: _camera_role_priority(cams.get(cid, {})))
            for cid in sorted_cids:
                c_cfg = cams.get(cid, {})
                is_kit = str(c_cfg.get("role") or "") in ("kitchen", "kitchen_4k")
                if (role_group == "main") == is_kit:
                    continue
                items = cam_map[cid]
                for h_lo, h_hi in buckets:
                    cands = [
                        it for it in items
                        if h_lo <= it["hour"] < h_hi
                        and it["source_rel"] not in seen_rels
                        and is_cloud_allowed(c_cfg, it["wall_ts"], [0.0])
                        and not _is_window_blocked(it["source_rel"], 0.0, 12.0, blocked_windows)
                        and not _shot_is_excluded(excluded, {"id": it["source_rel"], "source_rel": it["source_rel"], "start_s": 0.0})
                    ]
                    if not cands:
                        continue
                    best = max(cands, key=lambda x: x["size"])
                    seen_rels.add(best["source_rel"])
                    bucket_picks.append(best)
                    if len(bucket_picks) >= clips_budget:
                        return bucket_picks
    return bucket_picks


def _scout_unarchived_month_anchors(
    cfg: dict[str, Any],
    data_root: Path,
    cctv_root: Path,
    vault_root: Path,
    ff_bin: str,
    fp_bin: str,
    vlm: Any,
    spec: dict[str, Any],
    cams: dict[str, Any],
    allowed_cams: set[str] | None,
    blocked_windows: dict[str, list[tuple[float, float]]],
    excluded: set[str],
    existing_cands: list[dict[str, Any]],
    *,
    progress_fn: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Tier-D fast cross-month anchor scouting for unarchived months in raw /cctv & /vault storage."""
    period_start = str(spec["period_start"])
    period_end = str(spec["period_end"])
    tz_name = str(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
    today_d = dt.datetime.now(ZoneInfo(tz_name)).date()
    all_months = _iter_month_strings(period_start, period_end, cap_date=today_d)
    if not all_months:
        return []

    eff_end = min(period_end, today_d.isoformat())
    existing_by_month: dict[str, int] = {}
    for c in existing_cands:
        ym = str(c.get("day") or "")[:7]
        if ym:
            existing_by_month[ym] = existing_by_month.get(ym, 0) + 1

    min_per_month = 3 if len(all_months) <= 4 else 2
    missing_months = [ym for ym in all_months if existing_by_month.get(ym, 0) < min_per_month]
    if not missing_months:
        return []

    clips_per_month = 8 if len(missing_months) >= 6 else 12
    keep_per_month = 3 if len(missing_months) >= 6 else 5

    card_day_by_month: dict[str, str] = {}
    deliv_p = data_root / "v2" / "deliverables.json"
    if deliv_p.is_file():
        try:
            deliv = json.loads(deliv_p.read_text(encoding="utf-8"))
            for c in (deliv.get("monthly_cards") or []):
                if isinstance(c, dict) and c.get("month") and c.get("day"):
                    card_day_by_month.setdefault(str(c["month"]), str(c["day"]))
        except Exception:
            pass

    if progress_fn is not None:
        try:
            progress_fn(f"跨月侦察准备（扫描 {len(missing_months)} 个未归档月份）")
        except Exception:
            pass

    indexed = _index_camera_month_files(
        cctv_root, vault_root, cams, allowed_cams, period_start, eff_end
    )
    sampled_clips: list[dict[str, Any]] = []
    for ym in missing_months:
        picks = _pick_month_sample_clips(
            indexed.get(ym) or {},
            card_day_by_month.get(ym),
            cams,
            blocked_windows,
            excluded,
            clips_budget=clips_per_month,
        )
        sampled_clips.extend(picks)

    if not sampled_clips:
        return []

    def _probe_and_filter(idx_clip: int) -> dict[str, Any] | None:
        it = sampled_clips[idx_clip]
        src_p = it["path"]
        c_cfg = cams.get(it["camera"], {})
        try:
            info = med.probe_media(fp_bin, str(src_p))
        except Exception:
            return None
        dur = float(info.get("duration") or 0.0)
        if dur < 4.0 or int(info.get("width") or 0) <= 0:
            return None
        fracs = (0.20, 0.50, 0.80) if dur <= 120.0 else (0.12, 0.30, 0.50, 0.70, 0.88)
        raw_offs = [round(max(0.5, min(dur - 0.6, dur * f)), 2) for f in fracs]
        offsets = [o for o in raw_offs if is_cloud_allowed(c_cfg, it["wall_ts"], [o])]
        if not offsets:
            return None
        raw_imgs = cvlm.extract_jpegs_ram(ff_bin, str(src_p), offsets, max_edge=640, quality=80)
        if not raw_imgs:
            return None
        passing_imgs: list[bytes] = []
        passing_offs: list[float] = []
        best_sharp = 0.0
        for img_b, off_s in zip(raw_imgs, offsets):
            luma, sat, sharp = _frame_color_quality(img_b)
            if luma >= 48.0 and sat >= 3.5 and sharp >= 14.0:
                passing_imgs.append(img_b)
                passing_offs.append(off_s)
                best_sharp = max(best_sharp, sharp)
        if not passing_imgs:
            return None
        keep_n = min(4, len(passing_imgs))
        step_idx = [int(k * (len(passing_imgs) - 1) / max(1, keep_n - 1)) for k in range(keep_n)] if keep_n > 1 else [0]
        sel_imgs = [passing_imgs[k] for k in step_idx]
        sel_offs = [passing_offs[k] for k in step_idx]
        return {
            **it,
            "idx": idx_clip,
            "duration": dur,
            "width": int(info["width"]),
            "height": int(info["height"]),
            "has_audio": bool(info.get("has_audio", False)),
            "stamped": dd.stamp_timecode_badges(sel_imgs, sel_offs),
            "sel_offsets": sel_offs,
            "best_sharp": best_sharp,
        }

    prepped: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(3, len(sampled_clips))) as pool:
        for res in pool.map(_probe_and_filter, range(len(sampled_clips))):
            if res is not None:
                prepped.append(res)

    if not prepped:
        return []

    has_vlm = _vlm_has_providers(vlm)
    if has_vlm:
        if progress_fn is not None:
            try:
                progress_fn(f"跨月抽样侦察（{len(missing_months)} 个月 / {len(prepped)} 片段）")
            except Exception:
                pass
        tasks = [
            (
                p["idx"],
                f"film_scout|{p['source_rel']}|{p['day']}|v1",
                p["stamped"],
                f"{PROMPT_PERIOD_ANCHOR_SCOUT}\n\n[片段参考] 日期：{p['day']}；机位：{cams.get(p['camera'], {}).get('name', p['camera'])}；时长：{p['duration']:.1f}s。",
            )
            for p in prepped
        ]
        ans_map = vlm.ask_batch(tasks, tier="fast") if tasks else {}
    else:
        ans_map = {}

    per_month_strict: dict[str, list[dict[str, Any]]] = {}
    per_month_relaxed: dict[str, list[dict[str, Any]]] = {}

    for p in prepped:
        ym = p["day"][:7]
        cid = p["camera"]
        c_cfg = cams.get(cid, {})
        dur = float(p["duration"])
        mid_off = p["sel_offsets"][len(p["sel_offsets"]) // 2]

        if has_vlm:
            ans = ans_map.get(p["idx"])
            if not ans or is_unconfirmed_refusal(ans):
                continue
            if str(ans.get("daily_care") or "").lower() == "yes":
                continue
            c_clr = str(ans.get("child_clear") or "").strip().lower()
            fvis = str(ans.get("face_visibility") or "").strip()
            if c_clr not in ("yes", "partial") or fvis in dd.BLOCKED_FACE_VIS:
                continue
            exc = float(ans.get("excitement_score") or 0.0)
            if exc > 10.0:
                exc = round(exc / 20.0, 2)
            if exc < 2.0:
                continue
            raw_t = ans.get("best_time_s")
            try:
                best_t = float(raw_t) if raw_t is not None else mid_off
            except (TypeError, ValueError):
                best_t = mid_off
            best_t = max(1.5, min(max(1.5, dur - 3.5), best_t))
            cbox = ans.get("child_box")
            if not (isinstance(cbox, list) and len(cbox) == 4 and cbox != [0.0, 0.0, 0.0, 0.0]):
                cbox = None
            raw_cap = str(ans.get("caption_zh") or "").strip()
            cap = ff_mod.sanitize_caption_kinship(
                dd._clean_warm_text(raw_cap, 28) or f"{c_cfg.get('name', cid)} · 温暖成长时刻",
                strict_unverified=True,
            )
            score_v = min(5.0, max(2.0, exc))
        else:
            best_t = mid_off
            cbox = None
            cap = f"{c_cfg.get('name', cid)} · 历史代表成长片段"
            score_v = 3.8

        s0 = round(max(0.0, best_t - 3.2), 2)
        s1 = round(min(dur - 0.05, s0 + 7.2), 2)
        if s1 - s0 < 2.0:
            continue
        if _is_window_blocked(p["source_rel"], s0, s1, blocked_windows):
            continue
        cand_id = f"scout|{p['source_rel']}|{s0:.1f}"
        cand = {
            "id": cand_id,
            "day": p["day"],
            "camera": cid,
            "camera_name": c_cfg.get("name", cid),
            "c_cfg": c_cfg,
            "source_rel": p["source_rel"],
            "source_root_pref": p["source_root"],
            "t0_rel": None,
            "wall_ts": p["wall_ts"],
            "start_s": s0,
            "end_s": s1,
            "category": "daily_moment",
            "caption_zh": cap,
            "lyric_hint": cap,
            "box": cbox,
            "privacy_box": None,
            "zoom_x": float(c_cfg.get("zoom_factor", 1.25)),
            "shot_scale": "medium" if str(c_cfg.get("role") or "") != "kitchen_4k" else "wide",
            "pan_style": "gentle_push",
            "emotion_tag": "joy" if score_v >= 3.5 else "curious",
            "cinematic_score": score_v,
            "source_kind": "historical_anchor_scout",
            "boost_vocal": False,
            "people": [],
            "verified_local_roles": [],
            "verified_vlm_roles": [],
            "pre_verified": False,
            "must_include": False,
            "rank": 88.0 + score_v * 3.5 + (4.0 if str(c_cfg.get("role") or "") == "bedside_crib" else 0.0) + min(2.0, p["best_sharp"] / 200.0),
            "provenance_kind": "historical_anchor_scout",
        }
        if _shot_is_excluded(excluded, cand):
            continue
        if not has_vlm or score_v >= 3.0:
            per_month_strict.setdefault(ym, []).append(cand)
        else:
            per_month_relaxed.setdefault(ym, []).append(cand)

    selected_out: list[dict[str, Any]] = []
    for ym in missing_months:
        pool = per_month_strict.get(ym) or per_month_relaxed.get(ym) or []
        pool.sort(key=lambda c: (-float(c["rank"]), c["day"], c["source_rel"]))
        selected_out.extend(pool[:keep_per_month])
    return selected_out

_SPEECH_TECH_PREFIX_RE = re.compile(
    r"^纯童声\s*[·・]\s*清晰[「“\"']([^「」\"'（(]+)[^」\"']*[」”\"'](?:（[^）]*）|\([^)]*\))*\s*(?:[·・]\s*)?"
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sanitize_milestone_caption(caption: str, category: str) -> str:
    txt = str(caption or "").strip()
    if not txt:
        return "清晰清脆童声瞬间" if category == "speech" else "成长高光瞬间"
    m = _SPEECH_TECH_PREFIX_RE.match(txt)
    if m:
        word = re.sub(r"\s*\([^)]*\)\s*", "", m.group(1)).strip() or "声"
        rest = txt[m.end() :].strip(" ·・")
        txt = f"清脆叫出「{word}」· {rest}" if rest else f"清脆叫出「{word}」的瞬间"
    for cold in ("监控画面", "监控视角", "镜头捕捉到", "在镜头前", "未搀扶"):
        txt = txt.replace(cold, "")
    return txt.strip(" ·，；")


def parse_clip_wall_start(wall_ts: str | None, src: Path | None, day: str) -> dt.datetime | None:
    """Resolve the local capture timestamp of the beginning of a source clip."""
    if wall_ts:
        try:
            d = dt.datetime.fromisoformat(str(wall_ts))
            return d.replace(tzinfo=None)
        except ValueError:
            pass
    if src is not None:
        stem = src.stem
        m_xm = re.match(r"^00_(\d{14})(?:_|$)", stem)
        if m_xm:
            try:
                return dt.datetime.strptime(m_xm.group(1), "%Y%m%d%H%M%S")
            except ValueError:
                pass
        m_t0_aq = re.search(r"(\d{8})_(\d{6})(?:_|$)", stem)
        if m_t0_aq:
            try:
                return dt.datetime.strptime(m_t0_aq.group(1) + m_t0_aq.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                pass
        m_t0_mi = re.search(r"(\d{10})_(\d{2})M(\d{2})S", stem, re.I)
        if m_t0_mi:
            try:
                return dt.datetime.strptime(m_t0_mi.group(1), "%Y%m%d%H").replace(
                    minute=min(59, int(m_t0_mi.group(2))),
                    second=min(59, int(m_t0_mi.group(3))),
                )
            except ValueError:
                pass
        if len(stem) >= 6 and stem[:6].isdigit() and re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            try:
                return dt.datetime.strptime(day.replace("-", "") + stem[:6], "%Y%m%d%H%M%S")
            except ValueError:
                pass
        m_min = re.match(r"^(?P<m>\d{2})M(?P<s>\d{2})S", stem, re.I)
        pdir = src.parent.name
        if m_min and len(pdir) == 10 and pdir.isdigit():
            try:
                return dt.datetime.strptime(pdir, "%Y%m%d%H").replace(
                    minute=min(59, int(m_min.group("m"))),
                    second=min(59, int(m_min.group("s"))),
                )
            except ValueError:
                pass
    if re.match(r"^\d{4}-\d{2}-\d{2}$", day):
        try:
            return dt.datetime.fromisoformat(f"{day}T12:00:00")
        except ValueError:
            return None
    return None


def resolve_shot_source(
    source_rel: str,
    t0_rel: str | None,
    data_root: Path,
    cctv_root: Path,
    vault_root: Path,
    source_root: str = "cctv",
) -> tuple[Path | None, str]:
    """Resolve a read-only playable path for a source clip (t0_keeper -> cctv -> vault)."""
    if t0_rel:
        rel_t0 = Path(str(t0_rel))
        if not rel_t0.is_absolute() and ".." not in rel_t0.parts:
            cand = data_root / rel_t0
            if cand.is_file():
                return cand, "t0_keeper"
    rel_s = str(source_rel or "").replace("\\", "/").lstrip("/")
    rel_p = Path(rel_s)
    if rel_s and not rel_p.is_absolute() and ".." not in rel_p.parts:
        order = (
            (("cctv", cctv_root), ("vault", vault_root))
            if source_root == "cctv"
            else (("vault", vault_root), ("cctv", cctv_root))
        )
        for rname, rbase in order:
            cand = rbase / rel_p
            if cand.is_file():
                return cand, rname
        # Allow fallback lookup by basename inside v2/t0_keepers for isolated offline pilots
        keepers = data_root / "v2" / "t0_keepers"
        if keepers.is_dir():
            fname = rel_p.name
            for cand in sorted(keepers.glob(f"*_{fname}")):
                if cand.is_file():
                    return cand, "t0_keeper"
    return None, "missing"


def _blocked_source_windows(db: ArchiveDB, period_start: str, period_end: str) -> dict[str, list[tuple[float, float]]]:
    """Return unapproved VLM refusal source windows so unreviewed clips never leak into compiled films."""
    blocked: dict[str, list[tuple[float, float]]] = {}
    for rv in db.list_vlm_reviews(status="all", limit=800):
        day = str(rv.get("day") or "")
        if not (period_start <= day <= period_end):
            continue
        if rv.get("status") != "approved":
            blocked.setdefault(str(rv["source_rel"]), []).append(
                (float(rv.get("start_s", 0.0)), float(rv.get("end_s", 0.0)))
            )
    return blocked


def _is_window_blocked(
    source_rel: str,
    start_s: float,
    end_s: float,
    blocked_windows: dict[str, list[tuple[float, float]]],
) -> bool:
    for bs, be in blocked_windows.get(source_rel, []):
        if end_s >= (bs - 2.0) and start_s <= (be + 2.0):
            return True
    return False


def _shot_is_excluded(excluded: set[str], cand: dict[str, Any]) -> bool:
    keys = {
        str(cand.get("id") or ""),
        str(cand.get("source_rel") or ""),
        f"{cand.get('source_rel')}|{float(cand.get('start_s', 0.0)):.1f}",
    }
    return bool({k for k in keys if k} & excluded)


def _compute_shot_film_offsets(
    shots: list[dict[str, Any]],
    fade_s: float,
    fps: int,
    *,
    has_cinema_bookends: bool = False,
) -> float:
    """Populate exact `film_offset_start_s` & `film_offset_end_s` matching `render_smooth_film_nas`."""
    default_fade_frames = max(4, int(round(fade_s * fps)))
    end_fade_frames = (
        max(default_fade_frames, int(round(0.85 * fps)))
        if has_cinema_bookends
        else default_fade_frames
    )
    cum_frames = 0
    tail_frames = 0
    for idx, sh in enumerate(shots):
        is_first = idx == 0
        is_last = idx == len(shots) - 1
        in_fade = tail_frames if idx > 0 else default_fade_frames
        out_fade = (
            end_fade_frames
            if is_last
            else med._transition_fade_frames(shots[idx + 1], fade_s, fps)
        )
        req_frames = int(round(float(sh["dur_s"]) * fps))
        if is_first and has_cinema_bookends:
            req_frames = max(req_frames, out_fade + int(round(3.8 * fps)))
        n_frames = max(in_fade + out_fade + 12, req_frames)
        sh["shot_index"] = idx + 1
        sh["film_offset_start_s"] = round(cum_frames / float(fps), 2)
        sh["film_offset_end_s"] = round((cum_frames + n_frames) / float(fps), 2)
        cum_frames += n_frames - out_fade
        tail_frames = out_fade
    total_frames = cum_frames + tail_frames if shots else 0
    return round(total_frames / float(fps), 2)


def _allowed_cameras_for_spec(spec: dict[str, Any], cams: dict[str, Any]) -> set[str] | None:
    if str(spec.get("kind") or "") != "camera_chronicle":
        return None
    fid = str(spec.get("film_id") or "")
    ang = str(spec.get("angle") or (fid[len("angle_") :] if fid.startswith("angle_") else ""))
    ang_meta = lc.CANONICAL_CHRONICLE_ANGLES.get(ang)
    if not ang_meta:
        return None
    target_roles = set(ang_meta["roles"])
    matched = {
        cid for cid, cfg in cams.items()
        if str(cfg.get("role") or "") in target_roles
    }
    return matched


def _infer_camera_id_from_rel(source_rel: str, cams: dict[str, Any]) -> str:
    norm = str(source_rel or "").replace("\\", "/")
    for cid, cfg in cams.items():
        rel_root = str(cfg.get("relative_root") or cfg.get("subdir") or "").strip("/")
        if rel_root and norm.startswith(rel_root + "/"):
            return str(cid)
        if f"/{cid}/" in norm or norm.startswith(f"{cid}/"):
            return str(cid)
    return ""


def _parse_box_nearest(cands_json: str | None, t_s: float) -> list[float] | None:
    try:
        pcs = json.loads(cands_json or "[]")
    except ValueError:
        return None
    if not pcs:
        return None
    best = min(pcs, key=lambda pc: abs(float(pc.get("t") or 0.0) - t_s))
    box = best.get("box")
    if isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
        return [float(x) for x in box]
    return None


def select_period_shots(
    cfg: dict[str, Any],
    db: ArchiveDB,
    data_root: Path,
    cctv_root: Path,
    vault_root: Path,
    fp_bin: str,
    spec: dict[str, Any],
    cams: dict[str, Any],
    *,
    max_shots: int | None = None,
    ff_bin: str | None = None,
    vlm: Any = None,
    progress_fn: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Select verified, non-refused shots across [period_start, period_end] via a 3-tier pyramid."""
    period_start = str(spec["period_start"])
    period_end = str(spec["period_end"])
    kind = str(spec.get("kind") or "monthly")
    profile = PERIOD_KIND_PROFILES.get(kind, PERIOD_KIND_PROFILES["monthly"])
    bday = str(cfg.get("project", {}).get("birthday", "2025-12-17"))
    excluded = {str(x) for x in (spec.get("excluded_shots") or [])}
    blocked_windows = _blocked_source_windows(db, period_start, period_end)
    allowed_cams = _allowed_cameras_for_spec(spec, cams)

    cand_limit = int(max_shots) if max_shots else int(profile["cand_pool"])
    raw_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    with db.session() as con:
        file_rows = [
            dict(r)
            for r in con.execute(
                "SELECT * FROM files WHERE day BETWEEN ? AND ? ORDER BY day ASC, wall_ts ASC",
                (period_start, period_end),
            ).fetchall()
        ]
    file_by_rel = {str(fr["source_rel"]): fr for fr in file_rows}

    def _cam_allowed(cam_id: str) -> bool:
        if allowed_cams is None:
            return True
        return cam_id in allowed_cams

    # Tier A: Pre-verified daily highlight shots from days/<YYYY-MM-DD>/daily_script.json
    days_dir = data_root / "days"
    if days_dir.is_dir():
        for day_path in sorted(days_dir.iterdir()):
            if not day_path.is_dir():
                continue
            day_s = day_path.name
            if not (period_start <= day_s <= period_end):
                continue
            sc_file = day_path / "daily_script.json"
            if not sc_file.is_file():
                continue
            try:
                story = json.loads(sc_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for sh in (story.get("shots") or []):
                srel = str(sh.get("source_rel") or "").strip()
                if not srel:
                    continue
                fr = file_by_rel.get(srel) or {}
                cam_id = str(sh.get("camera_id") or fr.get("camera") or _infer_camera_id_from_rel(srel, cams))
                if not _cam_allowed(cam_id):
                    continue
                c_cfg = cams.get(cam_id, {})
                s0 = max(0.0, float(sh.get("start_s") or 0.0))
                dur_h = max(3.6, min(8.8, float(sh.get("dur_s") or 5.5)))
                s1 = s0 + dur_h
                if _is_window_blocked(srel, s0, s1, blocked_windows):
                    continue
                cats = [str(x) for x in (sh.get("categories") or [])]
                lyric_h = str(sh.get("lyric_zh") or "").strip()
                src_kind = str(sh.get("source_kind") or "")
                is_sp = (
                    src_kind == "speech"
                    or str(sh.get("emotion_tag") or "").strip().lower() == "speech"
                    or ("纯童声" in lyric_h)
                )
                boost_v = bool(is_sp or sh.get("boost_vocal"))
                if "daily_care" in cats and not is_sp:
                    continue
                sk = (srel, int(round(s0 / 5.0)))
                if sk in seen_keys:
                    continue
                box = sh.get("box") or _parse_box_nearest(fr.get("candidates_json"), s0 + dur_h * 0.5)
                score_v = float(sh.get("cinematic_score") or 4.2)
                role = str(c_cfg.get("role") or "")
                rank = (
                    125.0
                    + score_v * 3.5
                    + (18.0 if is_sp else 0.0)
                    + (6.0 if role == "bedside_crib" else 0.0)
                    + (4.0 if role in ("kitchen_4k", "kitchen") else 0.0)
                )
                cand = {
                    "id": f"ds|{day_s}|{sh.get('shot_id') or srel}|{s0:.1f}",
                    "day": day_s,
                    "camera": cam_id,
                    "camera_name": str(sh.get("camera") or c_cfg.get("name", cam_id)),
                    "c_cfg": c_cfg,
                    "source_rel": srel,
                    "source_root_pref": str(fr.get("source_root") or "cctv"),
                    "t0_rel": None,
                    "wall_ts": fr.get("wall_ts"),
                    "start_s": s0,
                    "end_s": s1,
                    "category": "speech" if is_sp else (cats[0] if cats else "daily_moment"),
                    "caption_zh": lyric_h or f"{c_cfg.get('name', cam_id)} · 精选成长片段",
                    "lyric_hint": lyric_h,
                    "box": box,
                    "privacy_box": sh.get("privacy_box"),
                    "zoom_x": float(c_cfg.get("zoom_factor", 1.25)),
                    "shot_scale": str(sh.get("shot_scale") or "medium"),
                    "pan_style": str(sh.get("pan_style") or "gentle_push"),
                    "emotion_tag": str(sh.get("emotion_tag") or ("speech" if is_sp else "joy")),
                    "cinematic_score": score_v,
                    "source_kind": "speech" if is_sp else (src_kind or "daily_highlight"),
                    "boost_vocal": boost_v,
                    "people": list(sh.get("people") or []),
                    "verified_local_roles": list(sh.get("verified_local_roles") or []),
                    "verified_vlm_roles": list(sh.get("verified_vlm_roles") or []),
                    "pre_verified": True,
                    "must_include": is_sp,
                    "rank": rank,
                    "provenance_kind": "daily_highlight_shot",
                }
                if _shot_is_excluded(excluded, cand):
                    continue
                seen_keys.add(sk)
                raw_candidates.append(cand)

    # Tier B: Confirmed non-refusal milestones (filtering out routine daily_care when growth/joy clips exist)
    ms_rows = db.query_milestones(limit=800, include_pending=False, min_day=period_start, max_day=period_end)
    non_care_ms: list[dict[str, Any]] = []
    care_ms: list[dict[str, Any]] = []
    for m in ms_rows:
        if not is_confirmed_milestone_record(m):
            continue
        cam_id = str(m["camera"])
        if not _cam_allowed(cam_id):
            continue
        s0 = max(0.0, float(m.get("start_s") or 0.0))
        s1 = min(s0 + 8.5, max(s0 + 3.6, float(m.get("end_s") or (s0 + 4.8))))
        srel = str(m["source_rel"])
        if _is_window_blocked(srel, s0, s1, blocked_windows):
            continue
        try:
            meta = json.loads(m.get("meta_json") or "{}")
        except ValueError:
            meta = {}
        c_cfg = cams.get(cam_id, {})
        fr = file_by_rel.get(srel) or {}
        cat = str(m.get("category") or "milestone")
        cap_clean = _sanitize_milestone_caption(str(m.get("caption_zh") or ""), cat)
        is_sp = cat == "speech"
        cand = {
            "id": str(m["id"]),
            "day": str(m["day"]),
            "camera": cam_id,
            "camera_name": meta.get("cam_zh") or c_cfg.get("name", cam_id),
            "c_cfg": c_cfg,
            "source_rel": srel,
            "source_root_pref": str(fr.get("source_root") or "cctv"),
            "t0_rel": m.get("t0_rel"),
            "wall_ts": fr.get("wall_ts"),
            "start_s": s0,
            "end_s": s1,
            "category": cat,
            "caption_zh": cap_clean,
            "lyric_hint": cap_clean,
            "box": meta.get("child_box"),
            "privacy_box": meta.get("privacy_box"),
            "zoom_x": float(meta.get("zoom_x") or c_cfg.get("zoom_factor", 1.25)),
            "shot_scale": "medium" if (is_sp or c_cfg.get("role") == "bedside_crib") else "wide",
            "pan_style": "gentle_push",
            "emotion_tag": "speech" if is_sp else "joy",
            "cinematic_score": min(5.0, float(m.get("score") or 4.2)),
            "source_kind": "speech" if is_sp else "confirmed_milestone",
            "boost_vocal": is_sp,
            "people": list(meta.get("verified_people") or meta.get("people") or []),
            "verified_local_roles": list(meta.get("verified_local_roles") or []),
            "verified_vlm_roles": list(meta.get("verified_vlm_roles") or []),
            "pre_verified": True,
            "must_include": is_sp,
            "rank": 100.0 + float(m.get("score") or 4.0) * 3.0 + (20.0 if is_sp else 0.0),
            "provenance_kind": "confirmed_milestone",
        }
        if _shot_is_excluded(excluded, cand):
            continue
        if cat == "daily_care":
            care_ms.append(cand)
        else:
            non_care_ms.append(cand)

    for cand in non_care_ms:
        sk = (cand["source_rel"], int(round(cand["start_s"] / 5.0)))
        if sk not in seen_keys:
            seen_keys.add(sk)
            raw_candidates.append(cand)

    if len(raw_candidates) < 2:
        for cand in care_ms:
            sk = (cand["source_rel"], int(round(cand["start_s"] / 5.0)))
            if sk not in seen_keys:
                seen_keys.add(sk)
                raw_candidates.append(cand)

    # Tier C: Processed daytime done files (strictly audited by Stage-2 HIGH VLM before inclusion when providers exist)
    need_tier_c = len(raw_candidates) < cand_limit
    if need_tier_c:
        tier_c_pool: list[dict[str, Any]] = []
        for fr in file_rows:
            if fr.get("status") != "done":
                continue
            t1_mode = str(fr.get("t1_mode") or "")
            if t1_mode == "verified_child_only" and not fr.get("t1_rel"):
                continue
            cam_id = str(fr["camera"])
            if not _cam_allowed(cam_id):
                continue
            c_cfg = cams.get(cam_id, {})
            wall_ts = fr.get("wall_ts")
            dur_f = float(fr.get("duration") or 0.0)
            if dur_f < 2.0:
                continue
            cam_role = str(c_cfg.get("role") or "")
            try:
                aud_s = json.loads(fr.get("audio_summary_json") or "{}")
            except ValueError:
                aud_s = {}
            max_laugh = float(aud_s.get("max_laughter") or 0.0)
            picks: list[tuple[float, list[float] | None, float, bool]] = []
            if max_laugh >= 0.25 or cam_role == "bedside_crib":
                for pk_t, pk_box, pk_vis in dd._locate_burst_peaks_in_row(fr, max_peaks=1):
                    picks.append((pk_t, pk_box, pk_vis + (0.6 if max_laugh >= 0.25 else 0.0), max_laugh >= 0.35))
            try:
                pcs = json.loads(fr.get("candidates_json") or "[]")
            except ValueError:
                pcs = []
            if pcs:
                top_p = pcs[0]
                if top_p.get("luma", 0) >= 60 and top_p.get("sharp", 0) >= 25:
                    picks.append((float(top_p.get("t") or 0.5), top_p.get("box"), float(top_p.get("score") or 0.5), False))
            if not picks:
                continue
            t_c, box_c, sc_c, boost_c = picks[0]
            if not is_cloud_allowed(c_cfg, wall_ts, [0.0]) or not is_cloud_allowed(c_cfg, wall_ts, [t_c]):
                continue
            s0 = max(0.0, t_c - 2.0)
            dur_c = min(7.2, max(2.2, dur_f - s0 - 0.1))
            s1 = s0 + dur_c
            srel = str(fr["source_rel"])
            if _is_window_blocked(srel, s0, s1, blocked_windows):
                continue
            sk = (srel, int(round(s0 / 5.0)))
            if sk in seen_keys:
                continue
            cand = {
                "id": f"file|{srel}|{s0:.1f}",
                "day": str(fr["day"]),
                "camera": cam_id,
                "camera_name": c_cfg.get("name", cam_id),
                "c_cfg": c_cfg,
                "source_rel": srel,
                "source_root_pref": str(fr.get("source_root") or "cctv"),
                "t0_rel": None,
                "wall_ts": wall_ts,
                "start_s": s0,
                "end_s": s1,
                "category": "laughter_peak" if boost_c else "daily_moment",
                "caption_zh": f"{c_cfg.get('name', cam_id)} · 温暖成长片段",
                "lyric_hint": "",
                "box": box_c,
                "privacy_box": None,
                "zoom_x": float(c_cfg.get("zoom_factor", 1.25)),
                "shot_scale": "medium" if cam_role != "kitchen_4k" else "wide",
                "pan_style": "gentle_push",
                "emotion_tag": "joy" if boost_c else "curious",
                "cinematic_score": 3.5,
                "source_kind": "verified_day_clip",
                "boost_vocal": boost_c,
                "people": [],
                "verified_local_roles": [],
                "verified_vlm_roles": [],
                "pre_verified": False,
                "must_include": False,
                "rank": 40.0 + sc_c * 4.0 + min(3.0, float(fr.get("active_s") or 0) / 10.0) + (8.0 if boost_c else 0.0),
                "provenance_kind": "verified_day_clip",
            }
            if not _shot_is_excluded(excluded, cand):
                seen_keys.add(sk)
                tier_c_pool.append(cand)
        tier_c_pool.sort(key=lambda c: (-float(c["rank"]), c["day"], c["source_rel"]))
        raw_candidates.extend(tier_c_pool[: max(cand_limit * 2, 48)])

    # Tier D: Fast cross-month anchor scouting for unarchived months in long-span films
    if kind in ("growth_chronicle", "annual", "seasonal", "camera_chronicle") and ff_bin:
        scouted = _scout_unarchived_month_anchors(
            cfg,
            data_root,
            cctv_root,
            vault_root,
            ff_bin,
            fp_bin,
            vlm,
            spec,
            cams,
            allowed_cams,
            blocked_windows,
            excluded,
            raw_candidates,
            progress_fn=progress_fn,
        )
        for cand in scouted:
            sk = (cand["source_rel"], int(round(cand["start_s"] / 5.0)))
            if sk not in seen_keys:
                seen_keys.add(sk)
                raw_candidates.append(cand)

    # Probe and resolve physically readable sources only (cap per month on cross-month spans to bound ffprobe cost)
    is_cross_month = kind in ("growth_chronicle", "annual", "seasonal", "camera_chronicle")
    ranked_raw = sorted(raw_candidates, key=lambda c: (-float(c["rank"]), c["day"], c["source_rel"]))
    if is_cross_month:
        per_m_cnt: dict[str, int] = {}
        trimmed_raw: list[dict[str, Any]] = []
        per_m_probe_cap = max(8, cand_limit // 2)
        for cand in ranked_raw:
            ym = str(cand.get("day") or "")[:7]
            if cand.get("must_include") or per_m_cnt.get(ym, 0) < per_m_probe_cap:
                trimmed_raw.append(cand)
                per_m_cnt[ym] = per_m_cnt.get(ym, 0) + 1
        ranked_raw = trimmed_raw

    readable: list[dict[str, Any]] = []
    for cand in ranked_raw:
        src_p, root_lbl = resolve_shot_source(
            cand["source_rel"],
            cand.get("t0_rel"),
            data_root,
            cctv_root,
            vault_root,
            source_root=cand.get("source_root_pref", "cctv"),
        )
        if src_p is None:
            continue
        try:
            info = med.probe_media(fp_bin, str(src_p))
        except Exception:
            continue
        clip_dur = float(info.get("duration") or 0.0)
        if clip_dur < 1.5 or int(info.get("width") or 0) <= 0:
            continue
        s0 = max(0.0, min(float(cand["start_s"]), clip_dur - 1.5))
        max_avail = clip_dur - s0 - 0.05
        req_dur = float(cand["end_s"]) - float(cand["start_s"])
        dur_s = round(min(max_avail, max(1.4, min(8.8, req_dur))), 2)
        s1 = round(s0 + dur_s, 2)
        wdt0 = parse_clip_wall_start(cand.get("wall_ts"), src_p, cand["day"])
        wall_start_iso = (wdt0 + dt.timedelta(seconds=s0)).isoformat(timespec="seconds") if wdt0 else f"{cand['day']}T12:00:00"
        wall_end_iso = (wdt0 + dt.timedelta(seconds=s1)).isoformat(timespec="seconds") if wdt0 else wall_start_iso
        age_info = lc.compute_age_info(bday, dt.date.fromisoformat(cand["day"])) if re.match(r"^\d{4}-\d{2}-\d{2}$", cand["day"]) else {}
        readable.append({
            **cand,
            "source_path": str(src_p),
            "source_root": root_lbl,
            "source_start_s": round(s0, 2),
            "source_end_s": s1,
            "dur_s": dur_s,
            "verified_dur_s": dur_s,
            "clip_total_dur_s": round(clip_dur, 2),
            "nat_w": int(info["width"]),
            "nat_h": int(info["height"]),
            "has_audio": bool(info.get("has_audio", False)),
            "clip_wall_start_ts": wall_start_iso,
            "clip_wall_end_ts": wall_end_iso,
            "age_label": age_info.get("short_zh") or age_info.get("label_zh") or "",
        })

    if not readable:
        return []

    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    # Pass 0: keep must_include speech shots (capped for multi-month spans so recent speech never starves earlier months)
    speech_cap = max(2, cand_limit // 4) if is_cross_month else cand_limit
    speech_picked = 0
    for sh in readable:
        if sh.get("must_include") and sh["id"] not in chosen_ids:
            chosen.append(sh)
            chosen_ids.add(sh["id"])
            speech_picked += 1
            if len(chosen) >= cand_limit or speech_picked >= speech_cap:
                break

    if is_cross_month and len(chosen) < cand_limit:
        by_month: dict[str, list[dict[str, Any]]] = {}
        for sh in readable:
            by_month.setdefault(sh["day"][:7], []).append(sh)
        for _turn in range(max(4, cand_limit // max(1, len(by_month)))):
            for ym in sorted(by_month):
                for sh in by_month[ym]:
                    if sh["id"] in chosen_ids:
                        continue
                    chosen.append(sh)
                    chosen_ids.add(sh["id"])
                    break
                if len(chosen) >= cand_limit:
                    break
            if len(chosen) >= cand_limit:
                break
    elif len(chosen) < cand_limit:
        seen_days: set[str] = set()
        for sh in readable:
            if sh["id"] in chosen_ids:
                continue
            if sh["day"] not in seen_days:
                chosen.append(sh)
                chosen_ids.add(sh["id"])
                seen_days.add(sh["day"])
                if len(chosen) >= cand_limit:
                    break

    if len(chosen) < cand_limit:
        for sh in readable:
            if sh["id"] in chosen_ids:
                continue
            chosen.append(sh)
            chosen_ids.add(sh["id"])
            if len(chosen) >= cand_limit:
                break

    chosen.sort(key=lambda s: (s["day"], s["clip_wall_start_ts"], s["source_rel"], s["source_start_s"]))
    return chosen


def _vlm_has_providers(vlm: Any) -> bool:
    return bool(vlm is not None and ((getattr(vlm, "vlm_cfg", None) or {}).get("providers")))


def _audit_unverified_candidates(
    shots: list[dict[str, Any]],
    ff_bin: str,
    vlm: Any,
    has_pre_verified: bool,
    *,
    face_service: Any = None,
) -> list[dict[str, Any]]:
    """Verify any unverified Tier-C historical candidates through HIGH-thinking multi-frame edit + Face-ID checks."""
    if not _vlm_has_providers(vlm):
        return shots
    to_audit = [i for i, sh in enumerate(shots) if not sh.get("pre_verified")]
    if not to_audit:
        return shots

    has_face_svc = bool(face_service is not None and getattr(face_service, "enabled", True))
    use_ref = bool(has_face_svc and (face_service.fcfg or {}).get("attach_reference_sheet", True))
    hi_edge = int((getattr(face_service, "fcfg", None) or {}).get("hi_res_edge", 1280))

    def _prep_strip(idx_sh: int) -> tuple[int, list[bytes], float, float, dict[str, Any] | None] | None:
        sh = shots[idx_sh]
        w0 = max(0.0, float(sh["source_start_s"]) - 1.2)
        clip_tot = float(sh.get("clip_total_dur_s") or (sh["source_end_s"] + 2.0))
        w1 = min(clip_tot, max(w0 + 4.5, float(sh["source_end_s"]) + 1.5))
        span = max(3.0, w1 - w0)
        n_f = 6 if span >= 5.5 else 4
        lo = w0 + 0.35
        hi = max(lo + 0.5, w1 - 0.35)
        offsets = [round(lo + (hi - lo) * k / float(max(1, n_f - 1)), 2) for k in range(n_f)]
        face_match: dict[str, Any] | None = None
        use_hires = bool(has_face_svc and face_service.has_enrolled_members(sh["day"]))
        raw_imgs = cvlm.extract_jpegs_ram(
            ff_bin, sh["source_path"], offsets, max_edge=(hi_edge if use_hires else 768), quality=85,
        )
        if len(raw_imgs) < 2:
            return None
        if has_face_svc:
            try:
                face_match = face_service.identify_images(raw_imgs, day=sh["day"], child_box=sh.get("box"))
            except Exception:
                face_match = None
        if use_hires:
            raw_imgs = dd._resize_jpegs_max_edge(raw_imgs, 768)
        rel_offsets = [max(0.0, o - w0) for o in offsets[: len(raw_imgs)]]
        return idx_sh, dd.stamp_timecode_badges(raw_imgs, rel_offsets), w0, w1, face_match

    prepped: list[tuple[int, list[bytes], float, float, dict[str, Any] | None]] = []
    with ThreadPoolExecutor(max_workers=min(4, len(to_audit))) as pool:
        for item in pool.map(_prep_strip, to_audit):
            if item is not None:
                prepped.append(item)

    ref_by_day: dict[str, list[tuple[str, bytes]] | None] = {}

    def _ref_for_day(d_str: str) -> list[tuple[str, bytes]] | None:
        if not use_ref:
            return None
        if d_str not in ref_by_day:
            try:
                sb = face_service.get_reference_sheet(d_str)
            except Exception:
                sb = None
            ref_by_day[d_str] = [("家庭成员识别对照卡", sb)] if sb else None
        return ref_by_day[d_str]

    tasks = []
    for idx_sh, stamped, w0, w1, face_match in prepped:
        sh = shots[idx_sh]
        ref_imgs = _ref_for_day(sh["day"])
        fam_blk = (
            "\n\n" + face_service.build_vlm_family_prompt_block(sh["day"], face_match, has_ref_sheet=bool(ref_imgs))
            if has_face_svc
            else ""
        )
        prm = f"{dd.PROMPT_DIRECTOR_SHOT_EDIT}{fam_blk}\n\n[片段参考] 机位：{sh['camera_name']}；窗口跨度：+0.0s 至 +{max(1.0, w1 - w0):.1f}s。"
        ptag = f"film_edit|{sh['source_rel']}|{w0:.1f}"
        if ref_imgs:
            tasks.append((idx_sh, ptag, stamped, prm, ref_imgs))
        else:
            tasks.append((idx_sh, ptag, stamped, prm))
    ans_map = vlm.ask_batch(tasks, tier="director") if tasks else {}
    prep_by_idx = {idx_sh: (w0, w1, face_match) for idx_sh, _st, w0, w1, face_match in prepped}

    out: list[dict[str, Any]] = []
    for idx, sh in enumerate(shots):
        if sh.get("pre_verified"):
            out.append(sh)
            continue
        ans = ans_map.get(idx)
        info = prep_by_idx.get(idx)
        if not ans or not info:
            if not has_pre_verified:
                out.append(sh)
            continue
        w0, w1, face_match = info
        if is_unconfirmed_refusal(ans):
            continue
        fvis = str(ans.get("face_visibility") or "").strip()
        c_clr = str(ans.get("child_clear") or "").strip()
        if ans.get("usable_for_highlight") != "yes" or fvis in dd.BLOCKED_FACE_VIS:
            continue
        if (str(ans.get("daily_care") or "").lower() == "yes" or "daily_care" in (ans.get("categories") or [])) and str(ans.get("emotion_tag") or "").lower() != "speech":
            continue
        if not (c_clr == "yes" or (c_clr == "partial" and fvis in ("clear_front", "clear_three_quarter", "clear_action_profile"))):
            continue
        sc_v = float(ans.get("cinematic_score") or ans.get("excitement_score") or 0.0)
        if sc_v > 10.0:
            sc_v = round(sc_v / 20.0, 2)
        if sc_v < 2.0:
            continue

        win_span = max(3.6, w1 - w0)
        clip_tot = float(sh.get("clip_total_dur_s") or w1)
        raw_in = float(ans.get("best_in_rel_s") if ans.get("best_in_rel_s") is not None else 0.0)
        raw_out = float(ans.get("best_out_rel_s") if ans.get("best_out_rel_s") is not None else min(win_span, 6.2))
        if raw_in > win_span + 1.5 and (w0 - 1.5) <= raw_in <= (w1 + 1.5):
            raw_in -= w0
            raw_out -= w0
        in_rel = max(0.0, min(max(0.0, win_span - 4.0), raw_in))
        out_rel = min(win_span, max(in_rel + min(4.0, win_span), min(in_rel + 8.6, raw_out)))
        abs_in = round(max(0.0, w0 + in_rel), 2)
        abs_out = round(min(clip_tot, max(abs_in + min(3.5, clip_tot - abs_in), w0 + out_rel)), 2)
        dur_v = round(max(2.0, abs_out - abs_in), 2)

        cbox = ans.get("child_box")
        if not (isinstance(cbox, list) and len(cbox) == 4 and cbox != [0.0, 0.0, 0.0, 0.0]):
            cbox = sh.get("box")
        pbox = ans.get("privacy_box")
        if not (isinstance(pbox, list) and len(pbox) == 4 and pbox[2] > pbox[0] and pbox[3] > pbox[1]):
            pbox = med.estimate_pelvic_privacy_box(cbox) if ans.get("privacy_exposed") == "yes" else None

        scale = str(ans.get("shot_scale") or sh.get("shot_scale") or "medium").lower()
        if scale not in ("closeup", "medium", "wide"):
            scale = "medium"
        pan = str(ans.get("pan_style") or sh.get("pan_style") or "gentle_push").lower()
        if pan not in ("gentle_push", "gentle_pull", "steady"):
            pan = "gentle_push"
        note_raw = str(ans.get("director_note_zh") or ans.get("caption_zh") or sh.get("caption_zh") or "")
        vlm_roles = ff_mod.extract_high_conf_vlm_roles(ans)
        if has_face_svc:
            guarded_note, people_list = face_service.sanitize_text(
                note_raw, day=sh["day"], match_res=face_match, vlm_ans=ans,
            )
            local_roles = sorted((face_match or {}).get("verified_roles") or ())
        else:
            local_roles = list(sh.get("verified_local_roles") or [])
            guarded_note = ff_mod.sanitize_caption_kinship(
                note_raw,
                verified_local_roles=set(local_roles),
                verified_vlm_roles=vlm_roles,
                has_enrolled_ref=bool(local_roles),
                strict_unverified=True,
            )
            people_list = sorted(set(local_roles))
        note = dd._clean_warm_text(guarded_note, 36)

        out.append({
            **sh,
            "source_start_s": abs_in,
            "source_end_s": abs_out,
            "dur_s": dur_v,
            "verified_dur_s": dur_v,
            "box": cbox,
            "privacy_box": pbox,
            "shot_scale": scale,
            "pan_style": pan,
            "emotion_tag": str(ans.get("emotion_tag") or sh.get("emotion_tag") or "joy"),
            "cinematic_score": sc_v,
            "caption_zh": note or sh["caption_zh"],
            "lyric_hint": note or sh.get("lyric_hint") or sh["caption_zh"],
            "people": people_list,
            "verified_local_roles": local_roles,
            "verified_vlm_roles": sorted(vlm_roles),
            "pre_verified": True,
        })
    return out


def _enrich_period_shot_faces(
    shots: list[dict[str, Any]],
    ff_bin: str,
    face_service: Any,
) -> list[dict[str, Any]]:
    """Populate missing local face-ID matches & sanitize captions for pre-verified period shots when enrolled."""
    if face_service is None or not getattr(face_service, "enabled", True):
        return shots
    if not any(face_service.has_enrolled_members(sh.get("day")) for sh in shots):
        return shots
    hi_edge = int((getattr(face_service, "fcfg", None) or {}).get("hi_res_edge", 1280))
    out: list[dict[str, Any]] = []
    for sh in shots:
        if sh.get("people") or sh.get("verified_local_roles") or not sh.get("source_path"):
            out.append(sh)
            continue
        if not face_service.has_enrolled_members(sh.get("day")):
            out.append(sh)
            continue
        s0 = float(sh.get("source_start_s") or 0.0)
        s1 = max(s0 + 0.5, float(sh.get("source_end_s") or (s0 + float(sh.get("dur_s") or 2.0))))
        mid = (s0 + s1) * 0.5
        imgs = cvlm.extract_jpegs_ram(ff_bin, sh["source_path"], [round(s0 + 0.25, 2), round(mid, 2)], max_edge=hi_edge, quality=85)
        if not imgs:
            out.append(sh)
            continue
        try:
            fm = face_service.identify_images(imgs, day=sh.get("day"), child_box=sh.get("box"))
        except Exception:
            out.append(sh)
            continue
        v_roles = sorted(fm.get("verified_roles") or ())
        if not v_roles:
            out.append(sh)
            continue
        cleaned, ppl = face_service.sanitize_text(str(sh.get("caption_zh") or ""), day=sh.get("day"), match_res=fm)
        out.append({
            **sh,
            "caption_zh": cleaned or sh.get("caption_zh"),
            "lyric_hint": cleaned or sh.get("lyric_hint"),
            "people": ppl,
            "verified_local_roles": v_roles,
        })
    return out


def _is_speech_shot(sh: dict[str, Any]) -> bool:
    return (
        str(sh.get("category") or "") == "speech"
        or str(sh.get("source_kind") or "") == "speech"
        or str(sh.get("emotion_tag") or "").strip().lower() == "speech"
    )


def _format_period_sub(sh: dict[str, Any]) -> str:
    day_short = str(sh.get("day") or "")[5:] or str(sh.get("day") or "")
    age_lbl = str(sh.get("age_label") or "").strip()
    cam_nm = str(sh.get("camera_name") or sh.get("camera") or "").strip()
    parts = [p for p in (day_short, age_lbl, cam_nm) if p]
    if _is_speech_shot(sh):
        parts.append("纯童声原声")
    return " · ".join(parts)


def _act_index_for_shot(idx: int, total: int, n_acts: int) -> int:
    if total <= 1:
        return 0
    return min(n_acts - 1, int(idx * n_acts // max(1, total)))


def build_render_shot_specs(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build render_smooth_film_nas input dicts with cinema framing, CJK titles, and subtitles."""
    out: list[dict[str, Any]] = []
    for sh in shots:
        zoom = float(
            sh.get("zoom_eff")
            or (
                dd._compute_shot_zoom(
                    str(sh.get("shot_scale") or "medium"),
                    sh.get("c_cfg") or {"zoom_factor": sh.get("zoom_x", 1.25)},
                    sh.get("box"),
                )
                if sh.get("cinema_style")
                else (sh.get("zoom_x") or 1.0)
            )
        )
        if sh.get("cinema_style"):
            title_zh = str(sh.get("lyric_zh") or sh.get("caption_zh") or "")[:24]
            sub_zh = _format_period_sub(sh)
        else:
            age_part = f" · {sh['age_label']}" if sh.get("age_label") else ""
            t_hm = str(sh.get("clip_wall_start_ts") or "")[11:19]
            title_zh = f"{sh['day']} {t_hm}{age_part} · {sh['camera_name']}"
            sub_zh = str(sh.get("caption_zh") or "")[:36]
        out.append({
            "shot_id": sh.get("shot_id") or sh.get("id"),
            "source": sh["source_path"],
            "start_s": sh["source_start_s"],
            "dur_s": sh["dur_s"],
            "nat_w": sh["nat_w"],
            "nat_h": sh["nat_h"],
            "box": sh.get("box") if zoom > 1.02 or not sh.get("cinema_style") else None,
            "zoom": zoom,
            "shot_scale": str(sh.get("shot_scale") or "medium"),
            "pan_style": str(sh.get("pan_style") or "gentle_push"),
            "transition": str(sh.get("transition") or "warm_dissolve"),
            "chapter_zh": str(sh.get("render_chapter_zh") or ""),
            "boost_vocal": bool(sh.get("boost_vocal", False)),
            "cinema_style": bool(sh.get("cinema_style", True)),
            "has_audio": bool(sh.get("has_audio", False)),
            "title_zh": title_zh,
            "sub_zh": sub_zh,
            "privacy_box": sh.get("privacy_box"),
        })
    return out


def _estimate_period_net_s(shots: list[dict[str, Any]], fade_s: float = 0.55) -> float:
    if not shots:
        return 0.0
    tot = sum(float(s["dur_s"]) for s in shots)
    for i in range(1, len(shots)):
        tr = str(shots[i].get("transition") or "warm_dissolve")
        ov = 0.20 if tr == "soft_cut" else (0.80 if tr == "chapter_dissolve" else fade_s)
        tot -= ov
    return round(max(0.0, tot), 2)


def assemble_period_edit_list(
    spec: dict[str, Any],
    candidates: list[dict[str, Any]],
    script_ans: dict[str, Any] | None,
    cams: dict[str, Any],
    *,
    strict_kinship: bool = True,
    has_enrolled_ref: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assemble period candidates + Master Director screenplay with strict verified duration bounds."""
    if not candidates:
        return [], {}

    kind = str(spec.get("kind") or "monthly")
    profile = PERIOD_KIND_PROFILES.get(kind, PERIOD_KIND_PROFILES["monthly"])
    acts = tuple(profile["acts"])
    n_acts = len(acts)
    target_min_s = float(profile["target_min_s"])
    target_max_s = float(profile["target_max_s"])
    max_shots = int(profile["max_shots"])
    min_shots = int(profile["min_shots"])

    # Assign deterministic chronological shot IDs
    cands = []
    for idx, sh in enumerate(candidates):
        sid = sh.get("shot_id") or f"S{idx + 1:02d}"
        cands.append({**sh, "shot_id": sid, "_orig_idx": idx})
    by_id = {c["shot_id"]: c for c in cands}

    sa = script_ans if isinstance(script_ans, dict) and not is_unconfirmed_refusal(script_ans) else {}

    def _safe_dur(cand: dict[str, Any], req_s: Any) -> float:
        ver_dur = float(cand.get("verified_dur_s") or cand["dur_s"])
        try:
            req = float(req_s) if req_s is not None else ver_dur
        except (TypeError, ValueError):
            req = ver_dur
        floor_s = min(ver_dur, 3.6)
        return round(min(ver_dur, max(floor_s, req)), 2)

    chosen_map: dict[str, dict[str, Any]] = {}
    for entry in (sa.get("edit_list") or []):
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("shot_id") or "").strip()
        if sid not in by_id or sid in chosen_map:
            continue
        chosen_map[sid] = entry

    if not chosen_map:
        for c in cands[:max_shots]:
            chosen_map[c["shot_id"]] = {
                "shot_id": c["shot_id"],
                "chapter_zh": acts[_act_index_for_shot(c["_orig_idx"], len(cands), n_acts)],
                "trim_dur_s": c.get("verified_dur_s") or c["dur_s"],
                "shot_scale": c.get("shot_scale") or "medium",
                "pan_style": c.get("pan_style") or "gentle_push",
                "lyric_zh": c.get("lyric_hint") or c.get("caption_zh") or "",
            }

    # Always preserve confirmed speech shots
    for c in cands:
        if _is_speech_shot(c) and c["shot_id"] not in chosen_map:
            chosen_map[c["shot_id"]] = {
                "shot_id": c["shot_id"],
                "chapter_zh": acts[_act_index_for_shot(c["_orig_idx"], len(cands), n_acts)],
                "trim_dur_s": c.get("verified_dur_s") or c["dur_s"],
                "shot_scale": c.get("shot_scale") or "medium",
                "pan_style": c.get("pan_style") or "gentle_push",
                "lyric_zh": c.get("lyric_hint") or c.get("caption_zh") or "珍贵童声一刻",
            }

    is_cross_month = kind in ("growth_chronicle", "annual", "seasonal", "camera_chronicle")
    if is_cross_month:
        covered_yms = {by_id[sid]["day"][:7] for sid in chosen_map}
        best_by_ym: dict[str, dict[str, Any]] = {}
        for c in cands:
            ym = c["day"][:7]
            prev = best_by_ym.get(ym)
            if prev is None or float(c.get("rank") or 0.0) > float(prev.get("rank") or 0.0):
                best_by_ym[ym] = c
        for ym, best_c in sorted(best_by_ym.items()):
            if ym not in covered_yms and best_c["shot_id"] not in chosen_map:
                chosen_map[best_c["shot_id"]] = {
                    "shot_id": best_c["shot_id"],
                    "chapter_zh": acts[_act_index_for_shot(best_c["_orig_idx"], len(cands), n_acts)],
                    "trim_dur_s": best_c.get("verified_dur_s") or best_c["dur_s"],
                    "shot_scale": best_c.get("shot_scale") or "medium",
                    "pan_style": best_c.get("pan_style") or "gentle_push",
                    "lyric_zh": best_c.get("lyric_hint") or best_c.get("caption_zh") or "",
                }

    selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (by_id[sid], chosen_map[sid])
        for sid in sorted(chosen_map, key=lambda s: by_id[s]["_orig_idx"])
    ]
    while len(selected_pairs) > max_shots:
        ym_counts: dict[str, int] = {}
        for cc, _sp in selected_pairs:
            ym_counts[cc["day"][:7]] = ym_counts.get(cc["day"][:7], 0) + 1
        droppable = [
            i for i, (cc, _sp) in enumerate(selected_pairs)
            if not _is_speech_shot(cc) and (not is_cross_month or ym_counts.get(cc["day"][:7], 0) > 1)
        ]
        if not droppable:
            droppable = [i for i, (cc, _sp) in enumerate(selected_pairs) if not _is_speech_shot(cc)]
        if not droppable:
            selected_pairs = selected_pairs[:max_shots]
            break
        drop_i = min(droppable, key=lambda idx: float(selected_pairs[idx][0].get("rank") or 0.0))
        selected_pairs.pop(drop_i)

    def _pair_tr(i: int, prev_act_s: str | None, prev_c: dict[str, Any] | None, c: dict[str, Any], sp: dict[str, Any]) -> tuple[str, str, str]:
        raw_ch = str(sp.get("chapter_zh") or "").strip()
        fallback_act = acts[_act_index_for_shot(c["_orig_idx"], len(cands), n_acts)]
        eff_act = raw_ch if any(a[:3] in raw_ch for a in acts) else fallback_act
        show_ch = eff_act if (eff_act != prev_act_s) else ""
        tr_s = str(sp.get("transition") or "").strip()
        if i == 0 or show_ch:
            tr_s = "chapter_dissolve"
        elif tr_s not in ("chapter_dissolve", "warm_dissolve", "soft_cut"):
            same_day_cam = bool(
                prev_c and prev_c["day"] == c["day"] and prev_c["camera"] == c["camera"]
            )
            tr_s = "soft_cut" if same_day_cam else "warm_dissolve"
        return show_ch, eff_act, tr_s

    def _est(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
        ordered = sorted(pairs, key=lambda p: p[0]["_orig_idx"])
        p_act: str | None = None
        p_c: dict[str, Any] | None = None
        sim: list[dict[str, Any]] = []
        for i, (cc, sp) in enumerate(ordered):
            show_ch, eff_a, tr_i = _pair_tr(i, p_act, p_c, cc, sp)
            if show_ch:
                p_act = eff_a
            p_c = cc
            sim.append({"dur_s": _safe_dur(cc, sp.get("trim_dur_s")), "transition": tr_i})
        return _estimate_period_net_s(sim)

    unused = [c for c in cands if c["shot_id"] not in chosen_map]
    unused.sort(key=lambda x: (-float(x["rank"]), x["_orig_idx"]))
    while unused and len(selected_pairs) < max_shots and _est(selected_pairs) < target_min_s:
        nxt = unused.pop(0)
        selected_pairs.append((
            nxt,
            {
                "shot_id": nxt["shot_id"],
                "chapter_zh": acts[_act_index_for_shot(nxt["_orig_idx"], len(cands), n_acts)],
                "trim_dur_s": nxt.get("verified_dur_s") or nxt["dur_s"],
                "shot_scale": nxt.get("shot_scale") or "medium",
                "pan_style": nxt.get("pan_style") or "gentle_push",
                "lyric_zh": nxt.get("lyric_hint") or nxt.get("caption_zh") or "",
            },
        ))

    selected_pairs.sort(key=lambda p: p[0]["_orig_idx"])
    while len(selected_pairs) > min_shots and _est(selected_pairs) > target_max_s:
        ym_counts_trim: dict[str, int] = {}
        for cc, _sp in selected_pairs:
            ym_counts_trim[cc["day"][:7]] = ym_counts_trim.get(cc["day"][:7], 0) + 1
        removable = [
            i for i, (cc, _sp) in enumerate(selected_pairs)
            if i not in (0, len(selected_pairs) - 1)
            and not _is_speech_shot(cc)
            and (not is_cross_month or ym_counts_trim.get(cc["day"][:7], 0) > 1)
        ]
        if not removable:
            break
        drop_i = min(removable, key=lambda idx: float(selected_pairs[idx][0].get("rank") or 0.0))
        selected_pairs.pop(drop_i)

    # Kinship protection aggregate across period shots
    all_local_roles: set[str] = set()
    all_vlm_roles: set[str] = set()
    for c, _sp in selected_pairs:
        all_local_roles.update(c.get("verified_local_roles") or ())
        all_vlm_roles.update(c.get("verified_vlm_roles") or ())
    has_any_ref = bool(has_enrolled_ref or all_local_roles)

    final_shots: list[dict[str, Any]] = []
    storyboard_rows: list[dict[str, Any]] = []
    prev_act: str | None = None
    prev_c: dict[str, Any] | None = None

    for idx, (c, sp) in enumerate(selected_pairs):
        show_ch, eff_act, tr = _pair_tr(idx, prev_act, prev_c, c, sp)
        if show_ch:
            prev_act = eff_act
        c_cfg = c.get("c_cfg") or cams.get(c["camera"], {}) or {"zoom_factor": c.get("zoom_x", 1.25)}
        scale = str(sp.get("shot_scale") or c.get("shot_scale") or "medium").lower()
        if scale not in ("closeup", "medium", "wide"):
            scale = "medium"
        pan = str(sp.get("pan_style") or c.get("pan_style") or "gentle_push").lower()
        if pan not in ("gentle_push", "gentle_pull", "steady"):
            pan = "gentle_push"
        dur_s = _safe_dur(c, sp.get("trim_dur_s"))
        zoom_eff = dd._compute_shot_zoom(scale, c_cfg, c.get("box"))

        raw_lyric = dd._clean_warm_text(sp.get("lyric_zh") or c.get("lyric_hint") or c.get("caption_zh") or "", 24)
        c_local = set(c.get("verified_local_roles") or ())
        c_vlm = set(c.get("verified_vlm_roles") or ())
        lyric = ff_mod.sanitize_caption_kinship(
            raw_lyric or f"{c.get('age_label') or c['day']}的温暖时刻",
            verified_local_roles=c_local,
            verified_vlm_roles=c_vlm,
            has_enrolled_ref=bool(has_enrolled_ref or c_local),
            strict_unverified=strict_kinship,
        )

        enriched = {
            **c,
            "shot_index": idx + 1,
            "dur_s": dur_s,
            "source_end_s": round(float(c["source_start_s"]) + dur_s, 2),
            "shot_scale": scale,
            "pan_style": pan,
            "transition": tr,
            "chapter_zh": eff_act,
            "render_chapter_zh": show_ch,
            "lyric_zh": lyric,
            "zoom_eff": zoom_eff,
            "cinema_style": True,
        }
        final_shots.append(enriched)
        storyboard_rows.append({
            "index": idx + 1,
            "shot_id": c["shot_id"],
            "day": c["day"],
            "age_label": c.get("age_label") or "",
            "chapter_zh": eff_act,
            "camera": c["camera_name"],
            "shot_scale": scale,
            "dur_s": dur_s,
            "transition": tr,
            "people": list(c.get("people") or []),
            "lyric_zh": lyric,
            "source_rel": c["source_rel"],
        })
        prev_c = c

    def _guard_global(txt: str) -> str:
        return ff_mod.sanitize_caption_kinship(
            dd._clean_warm_text(txt, 140),
            verified_local_roles=all_local_roles,
            verified_vlm_roles=all_vlm_roles,
            has_enrolled_ref=has_any_ref,
            strict_unverified=strict_kinship,
        )

    period_span = f"{spec['period_start']} ～ {spec['period_end']}"
    cams_summary = "、".join(dict.fromkeys(sh["camera_name"].replace(" G350", "") for sh in final_shots[:4]))
    default_title = str(spec.get("title_zh") or spec["film_id"])
    default_logline = f"从 {final_shots[0]['day']} 到 {final_shots[-1]['day']}，跨越 {len({s['day'] for s in final_shots})} 天、串联 {cams_summary} 的 {len(final_shots)} 个清晰成长瞬间。"
    default_joy = "与家人欢笑互动、探索新鲜角落与清脆发声的高光时刻"
    default_growth = f"跨越 {period_span}，身形、神态与自主动作都悄悄长大了一截"
    default_epilogue = f"{default_title} · 愿每一次抬头微笑都被温柔珍藏。"

    film_title = _guard_global(str(sa.get("film_title_zh") or default_title))[:32] or default_title
    logline = _guard_global(str(sa.get("logline_zh") or default_logline))[:140] or default_logline
    joy_summary = _guard_global(str(sa.get("joy_summary_zh") or default_joy))[:90] or default_joy
    growth_changes = _guard_global(str(sa.get("growth_changes_zh") or default_growth))[:90] or default_growth
    epilogue = _guard_global(str(sa.get("closing_epilogue_zh") or default_epilogue))[:36] or default_epilogue

    film_header = {
        "kicker_zh": f"{period_span} · {len(final_shots)} 镜精选",
        "title_zh": film_title,
        "sub_zh": f"{spec['title_zh']} · {profile['genre_zh']}",
    }

    story_doc = {
        "film_id": str(spec["film_id"]),
        "kind": kind,
        "film_title_zh": film_title,
        "logline_zh": logline,
        "joy_summary_zh": joy_summary,
        "growth_changes_zh": growth_changes,
        "closing_epilogue_zh": epilogue,
        "film_header": film_header,
        "director_model": str(sa.get("_model") or ("offline" if not sa else "director")),
        "est_duration_s": _estimate_period_net_s(final_shots),
        "shots": storyboard_rows,
    }
    return final_shots, story_doc


def _direct_period_screenplay(
    spec: dict[str, Any],
    shots: list[dict[str, Any]],
    ff_bin: str,
    vlm: Any,
    *,
    face_service: Any = None,
) -> dict[str, Any] | None:
    """Ask highest-thinking Master Director (`tier="director"`) to arrange the period screenplay."""
    if not _vlm_has_providers(vlm) or len(shots) < 2:
        return None
    kind = str(spec.get("kind") or "monthly")
    profile = PERIOD_KIND_PROFILES.get(kind, PERIOD_KIND_PROFILES["monthly"])
    max_cand = min(len(shots), int(profile["max_shots"]) + 10)
    work = shots[:max_cand]

    def _poster(idx_sh: tuple[int, dict[str, Any]]) -> tuple[int, bytes]:
        i, sh = idx_sh
        mid_t = float(sh["source_start_s"]) + float(sh["dur_s"]) * 0.45
        imgs = cvlm.extract_jpegs_ram(ff_bin, sh["source_path"], [mid_t], max_edge=640, quality=82)
        return i, (imgs[0] if imgs else b"")

    posters: dict[int, bytes] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, jb in pool.map(_poster, list(enumerate(work))):
            if jb:
                posters[i] = jb

    sheet_inputs = []
    manifest_rows = []
    for i, sh in enumerate(work):
        sid = sh.get("shot_id") or f"S{i + 1:02d}"
        sh["shot_id"] = sid
        cam_short = str(sh["camera_name"]).replace(" G350", "").replace("宝宝", "")
        sheet_inputs.append({
            "shot_id": sid,
            "hh_mm": f"{sh['day'][5:]} {str(sh['clip_wall_start_ts'])[11:16]}",
            "cam_short": cam_short,
            "poster_jpg": posters.get(i, b""),
        })
        manifest_rows.append({
            "shot_id": sid,
            "day": sh["day"],
            "age": sh.get("age_label") or "",
            "time": str(sh["clip_wall_start_ts"])[11:19],
            "camera": sh["camera_name"],
            "avail_dur_s": round(float(sh.get("verified_dur_s") or sh["dur_s"]), 2),
            "scale": sh.get("shot_scale") or "medium",
            "emotion": sh.get("emotion_tag") or "joy",
            "has_speech": _is_speech_shot(sh),
            "people": list(sh.get("people") or []),
            "note": sh.get("lyric_hint") or sh.get("caption_zh") or "",
        })

    sheets = dd.build_storyboard_sheets(sheet_inputs, max_per_sheet=14)
    if not sheets:
        return None

    roster_tip = ""
    ref_imgs: list[tuple[str, bytes]] | None = None
    if face_service is not None and getattr(face_service, "enabled", True):
        active_lbls = sorted(face_service.active_roles_for_day(None))
        if active_lbls:
            roster_tip = f"\n6. **成员白名单**：家庭有效成员包括 {'、'.join(active_lbls)}；请对照每条分镜 `people` 字段与附带对照卡写称呼，拿不准一律写“家人”。"
        if bool((face_service.fcfg or {}).get("attach_reference_sheet", True)) and face_service.has_enrolled_members(None):
            try:
                sb = face_service.get_reference_sheet(day=None)
                if sb:
                    ref_imgs = [("家庭成员识别对照卡", sb)]
            except Exception:
                ref_imgs = None

    acts_desc = " -> ".join(profile["acts"])
    months_present = sorted({str(c.get("day") or "")[:7] for c in work if c.get("day")})
    cross_month_rule = (
        f"\n7. **全跨度按月串联铁律**：候选池跨越 {len(months_present)} 个月（{'、'.join(months_present)}），必须在成片中**每个月都保留至少 1～3 镜**，按时间顺序讲清从出生到现在的月龄外貌、身形与动作蜕变，严禁只集中在最近某几天！"
        if (kind in ("growth_chronicle", "annual", "seasonal", "camera_chronicle") and len(months_present) >= 2)
        else ""
    )
    prompt = f"""你是家庭成长纪录片总导演与编剧，正在用【电影长镜头与流年蒙太奇思维】，将一段跨度为 {spec['period_start']} ～ {spec['period_end']} 的精选高质量素材，编排为一部值得珍藏一辈子的《{spec['title_zh']}》（{profile['genre_zh']}）。

【叙事与节奏硬标准】
1. 必须按时间线讲好宝宝在这段成长周期里的外貌神态变化、新学会的小本领（坐/爬/扶站/迈步/抓握/指认）与开心高光：
   分幕结构：{acts_desc}
2. 镜头挑选：从清单中选取 **{profile['min_shots']}～{profile['max_shots']} 镜**，单镜 `trim_dur_s` 严禁超过该镜头的 `avail_dur_s`（通常 4.2～8.5 秒），全片净时长控制在 **{int(profile['target_min_s'])}～{int(profile['target_max_s'])} 秒**。
3. 凡是 `has_speech=true`（宝宝喊出爸/妈等珍贵原声）务必保留并安排在情感高潮幕。
4. 景别交替（`wide` 与 `medium`/`closeup` 穿插）与真实转场：`chapter_dissolve`（跨入新一幕）、`soft_cut`（同天同房间连贯延续）、`warm_dissolve`（跨日/换场景柔和叠化）。
5. **人物称呼与真实性铁律**：只写分镜与画面里真实发生过的细节；除非 `people` 字段明确给出具体亲属角色，否则统一用“家人”或省略主语，绝对禁止臆猜“爸爸/妈妈/奶奶”等称呼。`lyric_zh` 限 10～22 个汉字，温暖自然如家庭手帐，禁用“监控/男子/女子/捕捉”等词。{roster_tip}{cross_month_rule}

请只返回单个标准 JSON 对象：
{{
  "film_title_zh": "14字以内文艺片名（体现本月/本季独特成长特征）",
  "logline_zh": "45字以内导演导语（概括这段时间的变化与主线）",
  "joy_summary_zh": "35字以内：这段时间让宝宝最开心的事",
  "growth_changes_zh": "35字以内：外貌、神态与动作上最明显的新变化",
  "closing_epilogue_zh": "20字以内温暖片尾寄语",
  "edit_list": [
    {{
      "shot_id": "S01",
      "chapter_zh": "对应上方一幕完整名称",
      "trim_dur_s": 6.4,
      "shot_scale": "closeup|medium|wide",
      "pan_style": "gentle_push|gentle_pull|steady",
      "transition": "chapter_dissolve|warm_dissolve|soft_cut",
      "lyric_zh": "温暖有画面感的字幕台词"
    }}
  ]
}}

【周期档案】作品：{spec['title_zh']}（类型：{kind}，周期：{spec['period_start']} ～ {spec['period_end']}）
【可选高质量分镜清单（共 {len(manifest_rows)} 镜，按时间顺序）】
{json.dumps(manifest_rows, ensure_ascii=False)}"""

    sig_raw = "|".join(f"{r['shot_id']}:{r['day']}:{r['avail_dur_s']}" for r in manifest_rows)
    sig = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()[:12]
    try:
        ask_sig = inspect.signature(vlm.ask)
        if ref_imgs and "reference_images" in ask_sig.parameters:
            return vlm.ask(f"film_script|{spec['film_id']}|{sig}|v1", sheets, prompt, tier="director", reference_images=ref_imgs)
        return vlm.ask(f"film_script|{spec['film_id']}|{sig}|v1", sheets, prompt, tier="director")
    except Exception:
        return None


def _pick_best_period_card_shot(shots: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the clearest non-blurry shot biased toward the latter half of the period to represent latest growth."""
    if len(shots) == 1:
        return shots[0]
    latter = shots[len(shots) // 2 :]
    pool = [s for s in latter if not s.get("privacy_box")] or latter
    return max(pool, key=lambda s: float(s.get("cinematic_score") or s.get("rank") or 0.0))


def _render_best_period_card(
    ff_bin: str,
    shot: dict[str, Any],
    card_path: Path,
    out_w: int = 1920,
    out_h: int = 1080,
) -> bool:
    """Render a crisp period highlight card into card_path."""
    card_path.parent.mkdir(parents=True, exist_ok=True)
    nat_w, nat_h = int(shot["nat_w"]), int(shot["nat_h"])
    zoom = float(shot.get("zoom_eff") or shot.get("zoom_x") or 1.0)
    cw, ch, cx, cy = med.crop_16_9(nat_w, nat_h, shot.get("box"), zoom_strength=zoom, min_native_w=out_w)
    mid_t = float(shot["source_start_s"]) + min(1.2, float(shot["dur_s"]) * 0.45)
    vf = (
        f"crop={cw}:{ch}:{cx}:{cy},scale={out_w}:{out_h}:flags=lanczos"
        if (cw != out_w or ch != out_h)
        else f"crop={cw}:{ch}:{cx}:{cy}"
    )
    tmp = card_path.with_suffix(".tmp.jpg")
    try:
        with med.open_fast_clip_window(ff_bin, shot["source_path"], max(0.0, mid_t), 1.5, include_audio=False) as (clip_src, clip_mid_t):
            r = subprocess.run(
                [
                    ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{max(0.0, clip_mid_t):.2f}", "-i", clip_src,
                    "-vf", vf, "-frames:v", "1", "-q:v", "2", str(tmp),
                ],
                capture_output=True,
                timeout=30,
            )
            if r.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                os.replace(tmp, card_path)
                return True
    except (OSError, subprocess.SubprocessError):
        pass
    tmp.unlink(missing_ok=True)
    return False


def _public_shots_manifest(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip host-local absolute paths while retaining full provenance (source_rel + exact timestamps)."""
    out: list[dict[str, Any]] = []
    for sh in shots:
        out.append({
            "shot_index": int(sh["shot_index"]),
            "id": str(sh["id"]),
            "day": str(sh["day"]),
            "age_label": str(sh.get("age_label") or ""),
            "camera": str(sh["camera"]),
            "camera_name": str(sh["camera_name"]),
            "source_root": str(sh["source_root"]),
            "source_rel": str(sh["source_rel"]),
            "t0_rel": sh.get("t0_rel"),
            "clip_wall_start_ts": str(sh["clip_wall_start_ts"]),
            "clip_wall_end_ts": str(sh["clip_wall_end_ts"]),
            "source_start_s": float(sh["source_start_s"]),
            "source_end_s": float(sh["source_end_s"]),
            "duration_s": float(sh["dur_s"]),
            "film_offset_start_s": float(sh["film_offset_start_s"]),
            "film_offset_end_s": float(sh["film_offset_end_s"]),
            "category": str(sh.get("category") or ""),
            "chapter_zh": str(sh.get("chapter_zh") or ""),
            "people": list(sh.get("people") or []),
            "lyric_zh": str(sh.get("lyric_zh") or ""),
            "shot_scale": str(sh.get("shot_scale") or "medium"),
            "pan_style": str(sh.get("pan_style") or "gentle_push"),
            "transition": str(sh.get("transition") or "warm_dissolve"),
            "caption_zh": str(sh.get("lyric_zh") or sh.get("caption_zh") or ""),
            "zoom_x": float(sh.get("zoom_eff") or sh.get("zoom_x") or 1.0),
            "provenance_kind": str(sh.get("provenance_kind") or ""),
        })
    return out


def _atomic_swap_files(pairs: list[tuple[Path, Path]]) -> None:
    """Publish staged -> target pairs together with rollback preservation on failure."""
    baks: list[tuple[Path, Path, bool]] = []
    suffix = f".bak_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    try:
        for staged, dst in pairs:
            dst.parent.mkdir(parents=True, exist_ok=True)
            bak = dst.with_name(dst.name + suffix)
            had_prev = dst.exists()
            if had_prev:
                os.replace(dst, bak)
            baks.append((dst, bak, had_prev))
            os.replace(staged, dst)
    except OSError:
        for dst, bak, had_prev in reversed(baks):
            if dst.exists() and had_prev and bak.exists():
                dst.unlink(missing_ok=True)
            if had_prev and bak.exists() and not dst.exists():
                try:
                    os.replace(bak, dst)
                except OSError:
                    pass
        raise
    else:
        for _dst, bak, _had in baks:
            bak.unlink(missing_ok=True)


def sync_deliverables_with_films(data_root: Path, db: ArchiveDB) -> dict[str, Any]:
    """Update v2/deliverables.json with compiled films, monthly cards, and upgraded viewpoint chronicles."""
    deliv_p = data_root / "v2" / "deliverables.json"
    deliv: dict[str, Any] = {}
    if deliv_p.is_file():
        try:
            loaded = json.loads(deliv_p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                deliv = loaded
        except Exception:
            deliv = {}
    deliv.setdefault("version", "v3")
    films = db.list_film_productions()
    pub_films = [fp for fp in films if fp.get("status") == "published" and fp.get("video_rel")]
    deliv["films"] = [
        {
            "film_id": fp["film_id"],
            "kind": fp["kind"],
            "age_months": fp["age_months"],
            "title_zh": fp["title_zh"],
            "period_start": fp["period_start"],
            "period_end": fp["period_end"],
            "status": fp["status"],
            "review_status": fp["review_status"],
            "video": fp["video_rel"],
            "poster": fp["poster_rel"],
            "card": fp["card_rel"],
            "manifest": fp["manifest_rel"],
            "duration_s": fp["duration_s"],
            "shots": fp["shots_count"],
            "bytes": fp["bytes"],
        }
        for fp in pub_films
    ]
    existing_cards = [
        c for c in (deliv.get("monthly_cards") or [])
        if isinstance(c, dict) and c.get("card") and not str(c.get("film_id") or "").startswith("month_")
    ]
    compiled_cards: list[dict[str, Any]] = []
    seen_cards = {str(c.get("card")) for c in existing_cards}
    for fp in pub_films:
        if fp.get("kind") != "monthly" or not fp.get("card_rel") or fp.get("review_status") == "rejected":
            continue
        card_name = Path(str(fp["card_rel"])).name
        if card_name in seen_cards:
            continue
        seen_cards.add(card_name)
        compiled_cards.append({
            "film_id": fp["film_id"],
            "card": card_name,
            "age_lbl": f"{int(fp['age_months'])}月龄",
            "camera": "月度精选",
            "zoom_x": 1.0,
            "month": str(fp["period_end"])[:7],
            "day_n": int(fp["age_months"]) * 30,
            "stage_zh": fp["title_zh"],
        })
    deliv["monthly_cards"] = existing_cards + compiled_cards

    # Upgrade smooth_films entries atomically when corresponding camera_chronicle films publish
    smooth_map = dict(deliv.get("smooth_films") or {})
    for fp in pub_films:
        if fp.get("kind") != "camera_chronicle" or fp.get("review_status") == "rejected":
            continue
        fid = str(fp["film_id"])
        ang = fid[len("angle_") :] if fid.startswith("angle_") else ""
        ang_meta = lc.CANONICAL_CHRONICLE_ANGLES.get(ang)
        if not ang_meta:
            continue
        story_logline = ""
        mrel = fp.get("manifest_rel")
        if mrel and (data_root / str(mrel)).is_file():
            try:
                mobj = json.loads((data_root / str(mrel)).read_text(encoding="utf-8"))
                story_logline = str(((mobj.get("story") or {}).get("logline_zh")) or "")
            except Exception:
                pass
        smooth_map[ang] = {
            "file": f"films/{fid}.mp4",
            "poster": f"films/{fid}.jpg",
            "shots": fp["shots_count"],
            "duration_s": fp["duration_s"],
            "bytes": fp["bytes"],
            "angle": ang,
            "name": ang_meta["name"],
            "zoom_x": ang_meta["zoom_x"],
            "film_id": fid,
            "director_v3": True,
            "review_status": fp["review_status"],
            "logline_zh": story_logline,
        }
    if smooth_map:
        deliv["smooth_films"] = smooth_map

    approved_or_pub = [fp for fp in pub_films if fp.get("review_status") == "approved"] or pub_films
    if approved_or_pub:
        top = approved_or_pub[0]
        deliv["latest_compiled_film"] = {
            "film_id": top["film_id"],
            "title_zh": top["title_zh"],
            "review_status": top["review_status"],
            "video_rel": top["video_rel"],
            "poster_rel": top["poster_rel"],
            "manifest_rel": top["manifest_rel"],
            "shots": top["shots_count"],
            "duration_s": top["duration_s"],
            "bytes": top["bytes"],
        }
    gc_pub = next((fp for fp in pub_films if fp.get("film_id") == "growth_chronicle"), None)
    if gc_pub is not None:
        deliv["growth_chronicle"] = {
            "film_id": gc_pub["film_id"],
            "title_zh": gc_pub["title_zh"],
            "review_status": gc_pub["review_status"],
            "video_rel": gc_pub["video_rel"],
            "poster_rel": gc_pub["poster_rel"],
            "manifest_rel": gc_pub["manifest_rel"],
            "shots": gc_pub["shots_count"],
            "duration_s": gc_pub["duration_s"],
            "bytes": gc_pub["bytes"],
            "period_start": gc_pub["period_start"],
            "period_end": gc_pub["period_end"],
        }
    _atomic_write_json(deliv_p, deliv)
    return deliv


def update_published_film_manifest_review(data_root: Path, db: ArchiveDB, film: dict[str, Any]) -> None:
    """Keep manifest.json and deliverables.json synchronized with SQLite review_status."""
    mrel = film.get("manifest_rel")
    if mrel:
        mp = data_root / str(mrel)
        if mp.is_file():
            try:
                mobj = json.loads(mp.read_text(encoding="utf-8"))
                if isinstance(mobj, dict):
                    mobj["review_status"] = film.get("review_status", "pending_review")
                    mobj["reviewer_note"] = film.get("reviewer_note", "")
                    mobj["reviewed_at"] = film.get("reviewed_at")
                    _atomic_write_json(mp, mobj)
            except Exception:
                pass
    sync_deliverables_with_films(data_root, db)


def render_and_publish_film(
    cfg: dict[str, Any],
    db: ArchiveDB,
    data_root: Path,
    cctv_root: Path,
    vault_root: Path,
    ff_bin: str,
    fp_bin: str,
    font_path: str | None,
    use_vaapi: bool,
    vaapi_dev: str,
    spec: dict[str, Any],
    cams: dict[str, Any],
    *,
    fade_s: float = 0.55,
    fps: int = 30,
    out_w: int = 1920,
    out_h: int = 1080,
    qp: int = 19,
    max_shots: int | None = None,
    vlm: Any = None,
    face_service: Any = None,
    progress_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Select verified shots, direct screenplay, render into staging, verify, and atomically publish."""
    film_id = str(spec["film_id"])
    db.enqueue_film_production(spec, force=False)
    cur = db.get_film_production(film_id) or {}
    merged_spec = {
        **spec,
        "excluded_shots": list(cur.get("excluded_shots") or spec.get("excluded_shots") or []),
    }
    db.mark_film_rendering(film_id)

    def _prog(stage: str) -> None:
        if progress_fn is not None:
            try:
                progress_fn(stage)
            except Exception:
                pass

    _prog("聚拢三层素材池")
    candidates = select_period_shots(
        cfg, db, data_root, cctv_root, vault_root, fp_bin, merged_spec, cams,
        max_shots=max_shots, ff_bin=ff_bin, vlm=vlm, progress_fn=_prog,
    )
    if not candidates:
        failed = db.mark_film_failed(film_id, "no_confirmed_shots_in_period")
        return failed or {}

    if face_service is None and bool((cfg.get("face_id") or {}).get("enabled", True)):
        face_service = ff_mod.FamilyFaceService(db, data_root, cfg, font_path=font_path)

    has_pre = any(bool(c.get("pre_verified")) for c in candidates)
    if any(not c.get("pre_verified") for c in candidates):
        _prog("高思考复审历史片段")
        candidates = _audit_unverified_candidates(candidates, ff_bin, vlm, has_pre, face_service=face_service)
    if not candidates:
        failed = db.mark_film_failed(film_id, "no_clear_verified_shots_after_audit")
        return failed or {}
    candidates = _enrich_period_shot_faces(candidates, ff_bin, face_service)

    strict_kinship = bool(cfg.get("vlm", {}).get("strict_kinship_unverified", True))
    _prog("总导演编排剧本")
    script_ans = _direct_period_screenplay(merged_spec, candidates, ff_bin, vlm, face_service=face_service)
    has_any_enrolled = bool(face_service is not None and face_service.has_enrolled_members(None))
    shots, story_doc = assemble_period_edit_list(
        merged_spec,
        candidates,
        script_ans,
        cams,
        strict_kinship=strict_kinship,
        has_enrolled_ref=has_any_enrolled,
    )
    if not shots:
        failed = db.mark_film_failed(film_id, "empty_edit_list")
        return failed or {}

    render_shots = build_render_shot_specs(shots)
    fade_eff = min(fade_s, max(0.15, min(float(s["dur_s"]) for s in render_shots) * 0.25))
    film_header = story_doc.get("film_header")
    epilogue_zh = str(story_doc.get("closing_epilogue_zh") or "")
    has_bookends = bool(film_header or epilogue_zh)
    _compute_shot_film_offsets(shots, fade_eff, fps, has_cinema_bookends=has_bookends)
    public_shots = _public_shots_manifest(shots)

    films_dir = data_root / "v2" / "films"
    stage_dir = films_dir / f".staging_{film_id}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        _prog(f"渲染 1080p 影院成片（{len(render_shots)} 镜）")
        staged_mp4 = stage_dir / f"{film_id}.mp4"
        meta = med.render_smooth_film_nas(
            ff_bin,
            render_shots,
            staged_mp4,
            font_path,
            use_vaapi,
            vaapi_dev,
            fade_s=fade_eff,
            fps=fps,
            out_w=out_w,
            out_h=out_h,
            qp=qp,
            film_header=film_header,
            closing_epilogue_zh=epilogue_zh or None,
        )
        staged_jpg = staged_mp4.with_suffix(".jpg")
        if (
            not meta
            or not staged_mp4.is_file()
            or staged_mp4.stat().st_size <= 1000
            or not staged_jpg.is_file()
            or staged_jpg.stat().st_size <= 0
            or int(meta.get("shots") or 0) != len(shots)
        ):
            raise RuntimeError(f"film_render_incomplete shots={meta.get('shots') if meta else 0}/{len(shots)}")

        probe = med.probe_media(fp_bin, str(staged_mp4))
        if int(probe.get("width") or 0) <= 0 or float(probe.get("duration") or 0.0) <= 0.8:
            raise RuntimeError("film_probe_failed")

        has_card = False
        card_rel: str | None = None
        staged_card: Path | None = None
        if merged_spec["kind"] == "monthly":
            card_name = f"{film_id}_{merged_spec['period_end']}.jpg"
            staged_card = stage_dir / card_name
            best_card_shot = _pick_best_period_card_shot(shots)
            has_card = _render_best_period_card(ff_bin, best_card_shot, staged_card, out_w=out_w, out_h=out_h)
            card_rel = f"v2/monthly_cards/{card_name}" if has_card else None

        video_rel = f"v2/films/{film_id}.mp4"
        poster_rel = f"v2/films/{film_id}.jpg"
        manifest_rel = f"v2/films/{film_id}.manifest.json"

        tz_name = str(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
        dur_actual = float(meta.get("duration_s") or probe["duration"])
        story_doc["actual_duration_s"] = round(dur_actual, 2)
        manifest_obj = {
            "version": "v3",
            "film_id": film_id,
            "kind": merged_spec["kind"],
            "age_months": int(merged_spec["age_months"]),
            "title_zh": story_doc.get("film_title_zh") or merged_spec["title_zh"],
            "period_title_zh": merged_spec["title_zh"],
            "period_start": merged_spec["period_start"],
            "period_end": merged_spec["period_end"],
            "review_status": "pending_review",
            "generated_at": dt.datetime.now(ZoneInfo(tz_name)).isoformat(timespec="seconds"),
            "duration_s": dur_actual,
            "shots_count": len(public_shots),
            "bytes": int(staged_mp4.stat().st_size),
            "video_rel": video_rel,
            "poster_rel": poster_rel,
            "card_rel": card_rel,
            "story": story_doc,
            "shots": public_shots,
        }
        staged_manifest = stage_dir / f"{film_id}.manifest.json"
        _atomic_write_json(staged_manifest, manifest_obj)

        swap_pairs: list[tuple[Path, Path]] = [
            (staged_mp4, data_root / video_rel),
            (staged_jpg, data_root / poster_rel),
            (staged_manifest, data_root / manifest_rel),
        ]
        if has_card and staged_card is not None and card_rel is not None:
            swap_pairs.append((staged_card, data_root / card_rel))
        _atomic_swap_files(swap_pairs)

        pub = db.mark_film_published(film_id, {
            "review_status": "pending_review",
            "video_rel": video_rel,
            "poster_rel": poster_rel,
            "card_rel": card_rel,
            "manifest_rel": manifest_rel,
            "duration_s": manifest_obj["duration_s"],
            "shots_count": len(public_shots),
            "bytes": manifest_obj["bytes"],
            "shots": public_shots,
        })
        sync_deliverables_with_films(data_root, db)
        return pub or {}
    except Exception as exc:
        failed = db.mark_film_failed(film_id, f"{type(exc).__name__}: {exc}")
        return failed or {}
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
