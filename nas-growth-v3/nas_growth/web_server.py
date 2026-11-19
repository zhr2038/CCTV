"""Unified Growth Archive V3 + Live Unattended NAS Web Server (HTTP Range streaming, JSON APIs, LAN control)."""
from __future__ import annotations

import base64
import datetime as dt
import ipaddress
import json
import mimetypes
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from zoneinfo import ZoneInfo

from . import film_compiler as fc
from . import lifecycle as lc
from .cloud_vlm import redact_secrets
from .db import ArchiveDB, is_confirmed_milestone_record, is_unconfirmed_refusal
from .family_faces import FamilyFaceService


RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_PHASE_ZH = {
    "P0_LIVE": "P0 优先处理当日/昨日",
    "P1_BACKFILL": "P1 空闲历史倒序回溯",
    "PAUSED": "已暂停",
    "YIELD_LOW_MEM": "礼让量化内存阈值等待中",
    "IDLE_CAUGHT_UP": "P0/P1 已追平待命中",
}

CATEGORY_ZH = {
    "speech": "叫妈妈 / 叫爸爸（纯童声）",
    "standing": "独立/扶物站立",
    "walking": "迈步学走",
    "sofa_descent": "上下床沿/沙发",
    "toy_play": "专注玩耍",
    "happy_interaction": "开心亲子互动",
    "daily_care": "温馨日常照护",
    "kitchen_visit": "厨房探索",
}


def _fmt_gb(n: int | float) -> str:
    val = float(n or 0)
    if val >= 1e12:
        return f"{val / 1e12:.2f} TB"
    return f"{val / 1e9:.2f} GB" if val >= 1e9 else f"{val / 1e6:.1f} MB"


def _camera_names_map(cfg: dict[str, Any], snap: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for cid, c in (cfg.get("cameras") or {}).items():
        if isinstance(c, dict) and c.get("name"):
            names[cid] = str(c["name"])
    ch_cams = ((snap.get("state") or {}).get("camera_health") or {}).get("cameras") or {}
    for cid, c in ch_cams.items():
        if isinstance(c, dict) and c.get("name"):
            names.setdefault(cid, str(c["name"]))
    return names


def _build_vlm_fleet(cfg: dict[str, Any], snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate configured VLM providers into clean per-model fleet cards for UI rendering."""
    vlm_cfg = cfg.get("vlm") or {}
    providers = vlm_cfg.get("providers") or {}
    raw_stats = snap.get("vlm_stats") or []
    stats_by_prov = {str(v.get("provider")): v for v in raw_stats if v.get("provider")}
    now_ts = time.time()

    if not providers:
        cards: list[dict[str, Any]] = []
        tot_all = sum(int(v.get("total_tokens") or 0) for v in raw_stats)
        for v in raw_stats:
            cd_until = float(v.get("cooldown_until") or 0.0)
            is_cd = cd_until > now_ts
            err_txt = str(v.get("last_error") or "")
            if is_cd:
                status = "quota_out" if "QUOTA_EXHAUSTED" in err_txt else "rate_limited"
                status_zh = "额度耗尽已退" if status == "quota_out" else "限流冷却中"
            else:
                status = "online"
                status_zh = "在线活跃" if int(v.get("calls_ok") or 0) > 0 else "在线就绪"
            ok_n = int(v.get("calls_ok") or 0)
            tok_n = int(v.get("total_tokens") or 0)
            cards.append({
                "model": str(v.get("model") or v.get("provider") or "vlm"),
                "role": "fast",
                "role_zh": "云端审片",
                "channel_zh": str(v.get("provider") or ""),
                "slots": 1,
                "status": status,
                "status_zh": status_zh,
                "calls_ok": ok_n,
                "calls_err": int(v.get("calls_err") or 0),
                "avg_latency_s": round(float(v.get("total_latency_s") or 0.0) / ok_n, 1) if ok_n > 0 else None,
                "input_tokens": int(v.get("input_tokens") or 0),
                "output_tokens": int(v.get("output_tokens") or 0),
                "total_tokens": tok_n,
                "token_share_pct": round(tok_n * 100.0 / tot_all, 1) if tot_all > 0 else 0.0,
                "sub_channels": [],
            })
        return cards

    par_tiers = vlm_cfg.get("parallel_tiers") or {}
    pro_pool = list(par_tiers.get("pro") or [])
    fast_pool = list(par_tiers.get("fast") or [])
    pro_set = set(pro_pool)
    fast_set = set(fast_pool)

    # Group all configured providers (including fallback keys like gemini_hydrogen) by logical model
    grouped: dict[str, list[str]] = {}
    for pname, pcfg in providers.items():
        if not isinstance(pcfg, dict):
            continue
        disp = str(pcfg.get("display_model") or pcfg.get("model") or pname)
        grouped.setdefault(disp, []).append(pname)

    # Order models: pro pool first, then fast pool, omitting standalone legacy fallbacks not in parallel pools
    ordered_models: list[str] = []
    if pro_pool or fast_pool:
        for pname in pro_pool + fast_pool:
            pcfg = providers.get(pname)
            if not isinstance(pcfg, dict):
                continue
            disp = str(pcfg.get("display_model") or pcfg.get("model") or pname)
            if disp not in ordered_models:
                ordered_models.append(disp)
    else:
        ordered_models = list(grouped.keys())

    fleet: list[dict[str, Any]] = []
    for disp_model in ordered_models:
        members = grouped.get(disp_model, [])
        if not members:
            continue
        is_pro = any(p in pro_set for p in members)
        is_fast = any(p in fast_set for p in members)
        if is_pro and is_fast:
            role = "pro"
            role_zh = "高光 + 筛选 · 全能"
            target_pool = pro_set | fast_set
        elif is_pro:
            role = "pro"
            role_zh = "高光精审 · 主力"
            target_pool = pro_set
        else:
            role = "fast"
            role_zh = "日常筛选 · 抢单"
            target_pool = fast_set
        primary_provs = [p for p in members if p in target_pool] or list(members)

        calls_ok = sum(int(stats_by_prov.get(p, {}).get("calls_ok") or 0) for p in members)
        calls_err = sum(int(stats_by_prov.get(p, {}).get("calls_err") or 0) for p in members)
        lat_s = sum(float(stats_by_prov.get(p, {}).get("total_latency_s") or 0.0) for p in members)
        in_tok = sum(int(stats_by_prov.get(p, {}).get("input_tokens") or 0) for p in members)
        out_tok = sum(int(stats_by_prov.get(p, {}).get("output_tokens") or 0) for p in members)
        tot_tok = sum(int(stats_by_prov.get(p, {}).get("total_tokens") or 0) for p in members)
        slots = sum(max(1, int(providers[p].get("worker_slots", 1))) for p in primary_provs)

        sub_channels: list[dict[str, Any]] = []
        has_vertex = any(providers[p].get("kind") == "vertex" for p in members)
        if has_vertex:
            v_members = sorted([p for p in members if providers[p].get("kind") == "vertex"])
            for idx, vp in enumerate(v_members, start=1):
                st_v = stats_by_prov.get(vp, {})
                m_num = re.search(r"vertex_(\d+)_", vp)
                lbl = f"GCP#{m_num.group(1) if m_num else idx}"
                sub_channels.append({
                    "label": lbl,
                    "ok": int(st_v.get("calls_ok") or 0),
                    "cooling": float(st_v.get("cooldown_until") or 0.0) > now_ts,
                })
            for gp in members:
                if providers[gp].get("kind") != "vertex":
                    g_ok = int(stats_by_prov.get(gp, {}).get("calls_ok") or 0)
                    if gp in target_pool:
                        sub_channels.append({
                            "label": "AI Studio",
                            "ok": g_ok,
                            "cooling": float(stats_by_prov.get(gp, {}).get("cooldown_until") or 0.0) > now_ts,
                        })
                    elif g_ok > 0:
                        sub_channels.append({
                            "label": "AI Studio历史",
                            "ok": g_ok,
                            "cooling": float(stats_by_prov.get(gp, {}).get("cooldown_until") or 0.0) > now_ts,
                        })
            if any(providers[p].get("kind") != "vertex" for p in primary_provs):
                channel_zh = "AI Studio + Vertex 多路"
            else:
                channel_zh = "GCP Vertex 双账号" if len(v_members) >= 2 else "GCP Vertex"
        else:
            burl = str(providers[primary_provs[0]].get("base_url") or "")
            if "qianwenaiapi.com" in burl or "dashscope" in burl:
                channel_zh = f"千问 API · {slots}槽"
            elif "192.168." in burl or ":8317" in burl or "localhost" in burl:
                channel_zh = f"OpenAI 云端 (CLIProxy) · {slots}槽"
            else:
                channel_zh = f"云端通道 · {slots}槽"

        any_avail = any(float(stats_by_prov.get(p, {}).get("cooldown_until") or 0.0) <= now_ts for p in primary_provs)
        if any_avail:
            status = "online"
            status_zh = "在线活跃" if calls_ok > 0 else "在线就绪"
        elif any("QUOTA_EXHAUSTED" in str(stats_by_prov.get(p, {}).get("last_error") or "") for p in primary_provs):
            status = "quota_out"
            status_zh = "额度耗尽已退"
        else:
            status = "rate_limited"
            status_zh = "限流冷却中"

        fleet.append({
            "model": disp_model,
            "role": role,
            "role_zh": role_zh,
            "channel_zh": channel_zh,
            "slots": slots,
            "status": status,
            "status_zh": status_zh,
            "calls_ok": calls_ok,
            "calls_err": calls_err,
            "avg_latency_s": round(lat_s / calls_ok, 1) if calls_ok > 0 else None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "token_share_pct": 0.0,
            "sub_channels": sub_channels,
        })

    grand_tot = sum(c["total_tokens"] for c in fleet)
    for c in fleet:
        c["token_share_pct"] = round(c["total_tokens"] * 100.0 / grand_tot, 1) if grand_tot > 0 else 0.0
    return fleet


def _load_deliverables(data_root: Path) -> dict[str, Any]:
    p = data_root / "v2" / "deliverables.json"
    if not p.is_file():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _build_lifecycle_snapshot(data_root: Path, db: ArchiveDB, cfg: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
    tz_name = str(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
    bday = str(cfg.get("project", {}).get("birthday", "2025-12-17"))
    now_tz = dt.datetime.now(ZoneInfo(tz_name))
    lcfg = cfg.get("lifecycle", {})
    compile_day = int(lcfg.get("monthly_compile_day", 18))

    cached = (snap.get("state") or {}).get("lifecycle_status")
    if isinstance(cached, dict) and cached.get("cutoff_date"):
        # The daemon owns the disk-intensive audit. Keep its timestamp and result
        # together even when scans are paused or a new audit is in progress.
        base = dict(cached)
        checked_at = (cached.get("audit") or {}).get("checked_at")
        try:
            audit_age = (now_tz - dt.datetime.fromisoformat(checked_at)).total_seconds()
        except (TypeError, ValueError):
            audit_age = -1
        if not 0 <= audit_age <= 3600:
            base["can_prune"] = False
            base["guard_summary_zh"] = f"等待新的留存巡检，上次检查 {checked_at or '时间未知'}"
    else:
        base = {
            "retention_days": int(lcfg.get("source_retention_days", 90)),
            "retention_mode": lcfg.get("retention_mode", "observe_only"),
            "can_prune": False,
            "gates": {},
            "guard_summary_zh": "等待守护进程完成首次留存巡检",
        }
    if lcfg.get("baidu_status_path") or lcfg.get("baidu_status_url"):
        base["baidu_backup"] = lc._resolve_baidu_status(lcfg)
    else:
        base.setdefault("baidu_backup", {"available": False})
    base["age"] = lc.compute_age_info(bday, now_tz.date())
    base["monthly_schedule"] = lc.compute_monthly_schedule(bday, now_tz.date(), data_root, compile_day=compile_day, db=db)
    base["archive_cloud_sync"] = (snap.get("state") or {}).get("archive_cloud_sync") or {
        "status": "pending",
        "destination": str(lcfg.get("openlist_destination", "/Growth_Archive_V3")),
        "note_zh": "待守护进程首次巡检后同步",
    }
    base["independent_dir"] = str(cfg.get("project", {}).get("data_root", data_root))
    return base


def _collect_milestones(
    data_root: Path,
    db: ArchiveDB,
    cfg: dict[str, Any],
    snap: dict[str, Any] | None = None,
    deliv: dict[str, Any] | None = None,
    *,
    category: str = "",
    camera: str = "",
    q: str = "",
    day: str = "",
) -> dict[str, Any]:
    if snap is None:
        snap = {"state": {"camera_health": db.get_state("camera_health", {})}}
    deliv = deliv if deliv is not None else _load_deliverables(data_root)
    cam_names = _camera_names_map(cfg, snap)
    bday = str(cfg.get("project", {}).get("birthday", "2025-12-17"))

    all_items: list[dict[str, Any]] = []
    seen_vid: set[str] = set()

    rows = db.get_day_milestones(day) if day else db.query_milestones(limit=400)
    for m in rows:
        vrel = m.get("video_rel") or ""
        if not vrel or vrel in seen_vid:
            continue
        seen_vid.add(vrel)
        try:
            meta = json.loads(m.get("meta_json") or "{}")
        except Exception:
            meta = {}
        day_s = str(m.get("day") or "")
        age_lbl = ""
        if re.match(r"^\d{4}-\d{2}-\d{2}$", day_s):
            age_lbl = lc.compute_age_info(bday, dt.date.fromisoformat(day_s)).get("short_zh", "")
        cat = str(m.get("category") or "other")
        cam_id = str(m.get("camera") or "")
        all_items.append({
            "id": m.get("id") or vrel,
            "day": day_s,
            "age_label": age_lbl,
            "category": cat,
            "cat_zh": CATEGORY_ZH.get(cat, cat),
            "camera": cam_id,
            "cam_zh": meta.get("cam_zh") or cam_names.get(cam_id, cam_id),
            "zoom_x": float(meta.get("zoom_x", 1.0) or 1.0),
            "time_str": day_s,
            "caption_zh": str(m.get("caption_zh") or ""),
            "speaker_note": str(m.get("speaker_note") or ""),
            "independence": str(m.get("independence") or "unclear"),
            "score": float(m.get("score") or 0.0),
            "video": vrel,
            "poster": str(m.get("poster_rel") or ""),
            "t0_copy": Path(m["t0_rel"]).name if m.get("t0_rel") else "",
        })

    for hl in deliv.get("highlights", []):
        if not is_confirmed_milestone_record(hl):
            continue
        vrel = f"v2/highlights/{hl['video']}"
        if vrel in seen_vid:
            continue
        t_str = str(hl.get("time_str") or "")
        day_s = str(hl.get("day") or (t_str[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", t_str) else "2026-11-08"))
        if day and day != day_s:
            continue
        seen_vid.add(vrel)
        age_lbl = lc.compute_age_info(bday, dt.date.fromisoformat(day_s)).get("short_zh", "") if day_s else ""
        cat = str(hl.get("category") or "other")
        cam_zh = str(hl.get("cam_zh") or "")
        all_items.append({
            "id": hl.get("video"),
            "day": day_s,
            "age_label": age_lbl,
            "category": cat,
            "cat_zh": hl.get("cat_zh") or CATEGORY_ZH.get(cat, cat),
            "camera": hl.get("camera") or cam_zh,
            "cam_zh": cam_zh,
            "zoom_x": float(hl.get("zoom_x", 1.0) or 1.0),
            "time_str": t_str,
            "caption_zh": str(hl.get("caption_zh") or ""),
            "speaker_note": str(hl.get("speaker_note") or ""),
            "independence": str(hl.get("independence") or ""),
            "score": float(hl.get("score", 4.5) or 4.5),
            "video": vrel,
            "poster": f"v2/highlights/{hl['poster']}" if hl.get("poster") else "",
            "t0_copy": str(hl.get("t0_copy") or ""),
        })

    cat_counts: dict[str, int] = {"all": len(all_items)}
    for it in all_items:
        c = it["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    q_low = q.strip().lower()
    cam_f = camera.strip()
    cat_f = category.strip()
    kitchen_cam_ids = {
        cid
        for cid, cinfo in (cfg.get("cameras") or {}).items()
        if isinstance(cinfo, dict)
        and (str(cinfo.get("role", "")).startswith("kitchen") or "厨房" in str(cinfo.get("name", "")))
    }
    filtered: list[dict[str, Any]] = []
    for it in all_items:
        if cat_f and cat_f != "all" and it["category"] != cat_f:
            continue
        if cam_f and cam_f != "all":
            if cam_f == "kitchen_all":
                if (
                    it["camera"] not in kitchen_cam_ids
                    and "厨房" not in it["cam_zh"]
                    and "kitchen" not in it["camera"].lower()
                ):
                    continue
            elif cam_f not in (it["camera"], it["cam_zh"]):
                continue
        if q_low:
            hay = " ".join([it["caption_zh"], it["day"], it["cam_zh"], it["cat_zh"], it["speaker_note"], it["time_str"]]).lower()
            if q_low not in hay:
                continue
        filtered.append(it)

    return {
        "items": filtered,
        "total": len(all_items),
        "category_counts": cat_counts,
        "categories": CATEGORY_ZH,
    }


def _valid_photo_rel(data_root: Path, rel: Any) -> str:
    """Validate that a relative gallery photo path stays under data_root (excluding state/) and exists."""
    if not isinstance(rel, str):
        return ""
    rel_clean = rel.strip().lstrip("/")
    if not rel_clean or rel_clean.startswith("state") or ".." in Path(rel_clean).parts:
        return ""
    try:
        root_res = data_root.resolve()
        target = (data_root / rel_clean).resolve()
        if (
            not target.is_relative_to(root_res)
            or target.is_relative_to(root_res / "state")
            or not target.is_file()
            or target.stat().st_size <= 0
        ):
            return ""
    except OSError:
        return ""
    return rel_clean


def _filter_existing_photos(data_root: Path, photos: Any) -> list[dict[str, Any]]:
    """Filter out refused or missing photo records and fall back to whichever variant exists."""
    out: list[dict[str, Any]] = []
    for raw in photos or []:
        if not isinstance(raw, dict) or is_unconfirmed_refusal(raw):
            continue
        full_rel = _valid_photo_rel(data_root, raw.get("full"))
        framed_rel = _valid_photo_rel(data_root, raw.get("framed"))
        if not full_rel and not framed_rel:
            continue
        item = dict(raw)
        item["full"] = full_rel or framed_rel
        item["framed"] = framed_rel or full_rel
        out.append(item)
    return out


def _build_day_payload(
    day_str: str,
    data_root: Path,
    db: ArchiveDB,
    cfg: dict[str, Any],
    snap: dict[str, Any] | None = None,
    deliv: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if snap is None:
        snap = {"state": {"camera_health": db.get_state("camera_health", {})}}
    deliv = deliv if deliv is not None else _load_deliverables(data_root)
    cam_names = _camera_names_map(cfg, snap)
    bday = str(cfg.get("project", {}).get("birthday", "2025-12-17"))
    age_info = lc.compute_age_info(bday, dt.date.fromisoformat(day_str))

    sp = data_root / "days" / day_str / "summary.json"
    if sp.is_file():
        payload = json.loads(sp.read_text(encoding="utf-8"))
    else:
        payload = {"day": day_str, **db.count_day_stats(day_str), "photos": [], "timelapses": {}}

    if day_str == "2026-11-08":
        seed_hl = "day-20261108/highlight_20261108.mp4"
        if not payload.get("highlight_rel") and (data_root / seed_hl).is_file():
            payload["highlight_rel"] = seed_hl
        if not payload.get("photos") and deliv.get("photos"):
            payload["photos"] = [
                {
                    "full": f"v2/photos/{p['full']}",
                    "framed": f"v2/photos/{p['framed']}",
                    "camera": p["camera"],
                    "zoom_x": p["zoom_x"],
                    "time": p["time"],
                    "score": p["score"],
                    "caption_zh": p["caption_zh"],
                    "people": list(p.get("people") or []),
                }
                for p in deliv.get("photos", [])
            ]
        if not payload.get("status"):
            payload["status"] = "featured_v3"

    tl_map = payload.get("timelapses") or {}
    payload["timelapse_items"] = [
        {"cid": cid, "name": cam_names.get(cid, cid), "url": url}
        for cid, url in tl_map.items()
    ]
    payload["photos"] = _filter_existing_photos(data_root, payload.get("photos"))
    payload["milestones"] = _collect_milestones(data_root, db, cfg, snap, deliv, day=day_str)["items"]
    payload["age_info"] = age_info
    if not payload.get("daily_story"):
        scr_p = data_root / "days" / day_str / "daily_script.json"
        if scr_p.is_file():
            try:
                loaded_scr = json.loads(scr_p.read_text(encoding="utf-8"))
                if isinstance(loaded_scr, dict):
                    payload["daily_story"] = loaded_scr
            except Exception:
                pass
    hl_rel = payload.get("highlight_rel")
    if isinstance(hl_rel, str) and hl_rel.endswith(".mp4"):
        poster_rel = hl_rel[:-4] + ".jpg"
        if (data_root / poster_rel).is_file():
            payload["highlight_poster_rel"] = poster_rel
    return payload


def _enrich_films(data_root: Path, films: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in films:
        item = dict(f)
        mrel = item.get("manifest_rel")
        if mrel and (data_root / str(mrel)).is_file():
            try:
                mobj = json.loads((data_root / str(mrel)).read_text(encoding="utf-8"))
                if isinstance(mobj, dict):
                    item["shots"] = mobj.get("shots") or []
                    item["story"] = mobj.get("story")
                    item["generated_at"] = mobj.get("generated_at") or item.get("updated_at")
                    item["period_title_zh"] = mobj.get("period_title_zh") or item.get("title_zh")
                    item["duration_s"] = mobj.get("duration_s", item.get("duration_s") or 0.0)
                    item["shot_count"] = mobj.get("shot_count", item.get("shot_count") or len(item["shots"]))
            except Exception:
                pass
        item.setdefault("shots", [])
        out.append(item)
    return out


def _extract_curator_status(daemon: Any, snap: dict[str, Any]) -> str:
    getter = getattr(daemon, "get_curator_status", None)
    if callable(getter):
        try:
            val = str(getter() or "").strip()
            if val:
                return val
        except Exception:
            pass
    aj = (snap.get("state") or {}).get("active_job") or {}
    if isinstance(aj, dict):
        cs = str(aj.get("curator_status") or "").strip()
        if cs:
            return cs
        det = str(aj.get("detail") or "")
        if "后台审片:" in det:
            return det.split("后台审片:", 1)[1].strip()
    return ""


def _build_home_boot(data_root: Path, db: ArchiveDB, cfg: dict[str, Any], daemon: Any = None) -> dict[str, Any]:
    snap = db.get_dashboard_snapshot()
    deliv = _load_deliverables(data_root)
    cam_names = _camera_names_map(cfg, snap)
    health_p = data_root / "v2" / "camera_health.json"
    if health_p.is_file():
        try:
            health_obj = json.loads(health_p.read_text(encoding="utf-8"))
        except Exception:
            health_obj = (snap.get("state") or {}).get("camera_health", {"health": []})
    else:
        health_obj = (snap.get("state") or {}).get("camera_health", {"health": []})

    ms_pack = _collect_milestones(data_root, db, cfg, snap, deliv)
    life = _build_lifecycle_snapshot(data_root, db, cfg, snap)

    days_list = list(snap.get("days") or [])
    has_1108_seed = (data_root / "day-20261108" / "highlight_20261108.mp4").is_file() or bool(deliv.get("photos"))
    if has_1108_seed and not any(d.get("day") == "2026-11-08" for d in days_list):
        days_list.append({
            "day": "2026-11-08",
            "priority_tier": "样板",
            "total_files": 0,
            "done_files": 0,
            "error_files": 0,
            "status": "featured_v3",
            "orig_bytes": 0,
            "t1_bytes": 0,
            "t2_bytes": 0,
            "highlight_rel": "day-20261108/highlight_20261108.mp4" if (data_root / "day-20261108" / "highlight_20261108.mp4").is_file() else None,
            "photos_count": len(deliv.get("photos") or []),
            "milestones_count": sum(1 for m in ms_pack["items"] if m.get("day") == "2026-11-08"),
            "speech_count": sum(1 for m in ms_pack["items"] if m.get("day") == "2026-11-08" and m.get("category") == "speech"),
            "seed_only": True,
        })
    days_list.sort(key=lambda d: str(d.get("day") or ""), reverse=True)

    # Pick Hero highlight day (prefer newest day with highlight_rel, else newest with photos, else 2026-11-08)
    hero_day_str = None
    for d in days_list:
        if d.get("highlight_rel"):
            hero_day_str = str(d["day"])
            break
    if not hero_day_str and days_list:
        for d in days_list:
            if int(d.get("photos_count") or 0) > 0:
                hero_day_str = str(d["day"])
                break
        if not hero_day_str:
            hero_day_str = str(days_list[0]["day"])

    hero_day = _build_day_payload(hero_day_str, data_root, db, cfg, snap, deliv) if hero_day_str else None
    featured_photos = _filter_existing_photos(data_root, (hero_day or {}).get("photos"))
    if not featured_photos and deliv.get("photos"):
        featured_photos = _filter_existing_photos(
            data_root,
            [
                {
                    "full": f"v2/photos/{p['full']}",
                    "framed": f"v2/photos/{p['framed']}",
                    "camera": p["camera"],
                    "zoom_x": p["zoom_x"],
                    "time": p["time"],
                    "score": p["score"],
                    "caption_zh": p["caption_zh"],
                    "people": list(p.get("people") or []),
                }
                for p in deliv.get("photos", [])
            ],
        )

    return {
        "project": {
            "name": cfg.get("project", {}).get("name", "时光成长馆 · 家庭 AI 数字档案馆"),
            "birthday": cfg.get("project", {}).get("birthday", "2025-12-17"),
            "data_root": str(cfg.get("project", {}).get("data_root", data_root)),
        },
        "paused": (
            bool(daemon.is_paused())
            if callable(getattr(daemon, "is_paused", None))
            else (data_root / "state" / "PAUSE").is_file()
        ),
        "curator_status": _extract_curator_status(daemon, snap),
        "totals": snap.get("totals", {}),
        "archive_counts": snap.get("archive_counts", {}),
        "token_totals": snap.get("token_totals", {}),
        "vlm_stats": snap.get("vlm_stats", []),
        "vlm_fleet": _build_vlm_fleet(cfg, snap),
        "state": {
            "active_job": (snap.get("state") or {}).get("active_job", {}),
            "runtime_caps": (snap.get("state") or {}).get("runtime_caps", {}),
        },
        "camera_names": cam_names,
        "camera_health": health_obj.get("health", []),
        "days": days_list,
        "hero_day": hero_day,
        "featured_photos": featured_photos[:8],
        "milestones_pack": ms_pack,
        "pending_reviews": db.list_vlm_reviews(status="pending_review", limit=100),
        "cinema": {
            "monthly_film": deliv.get("monthly_film"),
            "films": _enrich_films(data_root, db.list_film_productions()),
            "monthly_cards": deliv.get("monthly_cards") or [],
            "smooth_films": list((deliv.get("smooth_films") or {}).values()),
        },
        "lifecycle": life,
    }


_INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>时光成长馆 · 宝宝 AI 数字档案馆</title>
<style>
:root{
  --canvas:#FBF8F3;
  --surface:#FFFFFF;
  --surface-soft:#F4EFE6;
  --surface-warm:#FBF3EA;
  --line:#EAE0D4;
  --line-soft:#F0E8DE;
  --ink:#2B231F;
  --ink-2:#5A4E47;
  --muted:#877A72;
  --terra:#D4714E;
  --terra-hover:#BC5D3B;
  --terra-soft:#FCEEE8;
  --sage:#497B68;
  --sage-soft:#EAF3EE;
  --amber:#C88A32;
  --amber-soft:#FDF4E3;
  --shadow-sm:0 2px 10px rgba(72,50,36,.05);
  --shadow-md:0 10px 30px rgba(72,50,36,.08);
  --shadow-lg:0 20px 50px rgba(56,38,26,.14);
  --r-lg:20px;
  --r-md:14px;
  --r-sm:10px;
  --serif:"Songti SC","STSong","Noto Serif SC","Source Han Serif SC",Georgia,serif;
  --sans:"PingFang SC","HarmonyOS Sans SC","Microsoft YaHei UI",system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--canvas);color:var(--ink);font-family:var(--sans);line-height:1.6;-webkit-font-smoothing:antialiased}
body{background-image:radial-gradient(ellipse 90% 420px at 50% 0%, #FFF2E4 0%, rgba(251,248,243,0) 70%)}
a{color:var(--terra);text-decoration:none}
button,select,input{font-family:inherit}

/* Sticky Top Navigation */
.topbar{
  position:sticky;top:0;z-index:40;
  background:rgba(251,248,243,.88);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border-bottom:1px solid var(--line);
}
.topbar-in{
  max-width:1360px;margin:0 auto;padding:12px 24px;
  display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;
}
.brand{display:flex;align-items:center;gap:12px}
.brand-mark{
  width:42px;height:42px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#E28462,#C96745);color:#fff;font-size:20px;box-shadow:0 6px 16px rgba(212,113,78,.28);
  flex-shrink:0;
}
.brand-title{font-family:var(--serif);font-size:20px;font-weight:700;color:var(--ink);letter-spacing:.3px;line-height:1.2}
.age-pill{
  display:inline-flex;align-items:center;gap:6px;margin-top:3px;
  font-size:12px;color:var(--sage);background:var(--sage-soft);padding:2px 10px;border-radius:99px;font-weight:600;
}

.nav-tabs{
  display:flex;align-items:center;gap:6px;background:var(--surface-soft);padding:4px;border-radius:99px;border:1px solid var(--line);
  overflow-x:auto;max-width:100%;
}
.ntab{
  border:0;background:transparent;color:var(--ink-2);padding:7px 15px;border-radius:99px;cursor:pointer;
  font-size:13.5px;font-weight:500;white-space:nowrap;transition:all .2s ease;display:inline-flex;align-items:center;gap:6px;
}
.ntab:hover{color:var(--ink);background:rgba(255,255,255,.55)}
.ntab.active{background:var(--surface);color:var(--terra-hover);font-weight:600;box-shadow:var(--shadow-sm)}

.nav-right{display:flex;align-items:center;gap:8px}
.live-pill{
  display:inline-flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--line);
  border-radius:99px;padding:6px 12px;font-size:12.5px;color:var(--ink-2);cursor:pointer;box-shadow:var(--shadow-sm);
  transition:transform .15s ease,border-color .15s ease;
}
.live-pill:hover{border-color:var(--terra);transform:translateY(-1px)}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--sage);display:inline-block;box-shadow:0 0 0 3px rgba(73,123,104,.18)}
.sdot.paused{background:var(--amber);box-shadow:0 0 0 3px rgba(200,138,50,.2)}
.mon-link{
  font-size:12.5px;color:var(--ink-2);padding:6px 12px;border-radius:99px;background:var(--surface-soft);
  border:1px solid var(--line);font-weight:500;transition:all .15s;
}
.mon-link:hover{background:var(--terra-soft);color:var(--terra-hover);border-color:#EAC2B3}

/* Page Container & Views */
.wrap{max-width:1360px;margin:0 auto;padding:24px 24px 64px}
.view{display:none;animation:fadeUp .28s ease}
.view.active{display:block}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

.sec-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin:28px 0 14px;flex-wrap:wrap}
.sec-head:first-child{margin-top:0}
.sec-title{font-family:var(--serif);font-size:22px;font-weight:700;color:var(--ink);margin:0;display:flex;align-items:center;gap:8px}
.sec-sub{font-size:13px;color:var(--muted);margin:2px 0 0}
.link-btn{
  border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:99px;
  padding:6px 14px;font-size:13px;cursor:pointer;font-weight:500;transition:all .18s;
}
.link-btn:hover{border-color:var(--terra);color:var(--terra-hover);background:var(--terra-soft)}
.btn{
  border:0;background:var(--terra);color:#fff;border-radius:10px;padding:8px 16px;font-size:13px;
  font-weight:600;cursor:pointer;transition:background .18s,transform .12s;display:inline-flex;align-items:center;gap:6px;
}
.btn:hover{background:var(--terra-hover)}
.btn.sec{background:var(--surface-soft);color:var(--ink);border:1px solid var(--line)}
.btn.sec:hover{background:#EAE2D6}
.btn.sage{background:var(--sage)} .btn.sage:hover{background:#3A6554}
.btn.warn{background:var(--amber)}
.btn:disabled{opacity:.62;cursor:not-allowed;transform:none}
.btn.sec:disabled:hover{background:var(--surface-soft)}
.btn.sage:disabled:hover{background:var(--sage)}

/* Hero Split Card */
.hero-card{
  display:grid;grid-template-columns:1.55fr 1fr;gap:22px;
  background:var(--surface);border:1px solid var(--line);border-radius:24px;padding:18px;box-shadow:var(--shadow-md);
}
.hero-media{position:relative;border-radius:16px;overflow:hidden;background:#1A1512;aspect-ratio:16/9}
.hero-media video{width:100%;height:100%;object-fit:contain;display:block}
.hero-empty{
  width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:#EDE3D8;text-align:center;padding:24px;background:linear-gradient(135deg,#3A2E28,#221B18);
}
.hero-body{display:flex;flex-direction:column;justify-content:space-between;padding:8px 6px}
.hero-kicker{
  display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;
  color:var(--terra-hover);background:var(--terra-soft);padding:3px 11px;border-radius:99px;align-self:flex-start;
}
.hero-h1{font-family:var(--serif);font-size:26px;font-weight:700;margin:10px 0 8px;color:var(--ink);line-height:1.3}
.hero-desc{font-size:14px;color:var(--ink-2);margin:0 0 16px}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.stat-tile{background:var(--surface-warm);border:1px solid var(--line-soft);border-radius:14px;padding:10px 12px}
.stat-val{font-size:20px;font-weight:700;color:var(--ink);line-height:1.2}
.stat-lbl{font-size:12px;color:var(--muted);margin-top:2px}
.hero-actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center}

/* Photo Grid & Warm Photo Cards */
.photo-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.pcard{
  background:var(--surface);border:1px solid var(--line);border-radius:18px;overflow:hidden;
  box-shadow:var(--shadow-sm);cursor:pointer;transition:transform .22s ease,box-shadow .22s ease;display:flex;flex-direction:column;
}
.pcard:hover{transform:translateY(-3px);box-shadow:var(--shadow-md)}
.pcard-imgwrap{position:relative;aspect-ratio:16/10;background:var(--surface-soft);overflow:hidden}
.pcard-imgwrap img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s ease}
.pcard:hover .pcard-imgwrap img{transform:scale(1.03)}
.pcard-imgfallback{
  position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:16px;text-align:center;
  font-size:12.5px;color:var(--muted);background:var(--surface-soft);
}
.pcard-badge{
  position:absolute;top:10px;left:10px;background:rgba(36,28,24,.68);backdrop-filter:blur(6px);
  color:#FFF9F3;font-size:11.5px;padding:2px 9px;border-radius:99px;font-weight:500;
}
.pcard-score{
  position:absolute;top:10px;right:10px;background:rgba(255,250,242,.92);color:var(--amber);
  font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:99px;box-shadow:0 2px 6px rgba(0,0,0,.1);
}
.pcard-body{padding:12px 14px 14px;flex:1;display:flex;flex-direction:column;justify-content:space-between}
.pcard-cap{font-size:13.5px;color:var(--ink);font-weight:500;margin-bottom:6px}
.pcard-meta{font-size:12px;color:var(--muted);display:flex;justify-content:space-between}

/* Video / Milestone Cards */
.vid-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}
.mcard{
  background:var(--surface);border:1px solid var(--line);border-radius:18px;overflow:hidden;
  box-shadow:var(--shadow-sm);display:flex;flex-direction:column;
}
.mcard video{width:100%;aspect-ratio:16/9;background:#1A1512;display:block}
.mcard-body{padding:14px 16px 16px;flex:1;display:flex;flex-direction:column;gap:7px}
.mcard-top{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}
.cat-pill{
  font-size:11.5px;font-weight:600;padding:3px 10px;border-radius:99px;
  background:var(--terra-soft);color:var(--terra-hover);
}
.cat-pill.speech{background:var(--sage-soft);color:var(--sage)}
.cat-pill.daily_care{background:var(--amber-soft);color:#9B6515}
.mcard-cap{font-size:14px;color:var(--ink);font-weight:500}
.mcard-meta{font-size:12px;color:var(--muted);display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.t0-tag{
  font-size:11px;color:var(--sage);background:var(--sage-soft);padding:2px 8px;border-radius:6px;
  font-family:ui-monospace,Consolas,monospace;display:inline-flex;align-items:center;gap:4px;
}

/* Daily Calendar Strip */
.day-strip{
  display:flex;gap:10px;overflow-x:auto;padding:4px 2px 10px;scrollbar-width:thin;
}
.dchip{
  flex:0 0 auto;min-width:116px;background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:9px 12px;cursor:pointer;text-align:left;transition:all .18s;box-shadow:var(--shadow-sm);
}
.dchip:hover{border-color:var(--terra)}
.dchip.active{background:linear-gradient(135deg,#FFF4ED,#FCE8DF);border-color:var(--terra);box-shadow:0 4px 14px rgba(212,113,78,.18)}
.dchip-d{font-weight:700;font-size:14px;color:var(--ink)}
.dchip-s{font-size:11.5px;color:var(--muted);margin-top:3px;display:flex;align-items:center;gap:6px}
.day-toolbar{
  background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:12px 16px;
  display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:10px 0 18px;
}
select, .search-in{
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:7px 12px;
  font-size:13.5px;color:var(--ink);outline:none;transition:border-color .15s;
}
select:focus, .search-in:focus{border-color:var(--terra)}

/* Filter Pills Bar */
.filter-bar{
  background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:14px 16px;
  display:flex;flex-direction:column;gap:12px;margin-bottom:18px;box-shadow:var(--shadow-sm);
}
.fpills{display:flex;gap:8px;flex-wrap:wrap}
.fpill{
  border:1px solid var(--line);background:var(--surface-soft);color:var(--ink-2);
  border-radius:99px;padding:6px 13px;font-size:13px;cursor:pointer;transition:all .15s;font-weight:500;
}
.fpill:hover{border-color:var(--terra);color:var(--ink)}
.fpill.active{background:var(--terra);border-color:var(--terra);color:#fff}
.filter-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.search-in{flex:1;min-width:220px}

/* Cinema & Monthly Cards */
.cinema-box{
  background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:18px;box-shadow:var(--shadow-md);margin-bottom:22px;
}
.cinema-box video{width:100%;max-height:580px;border-radius:14px;background:#1A1512;display:block}
.schedule-banner{
  margin-top:14px;background:var(--amber-soft);border:1px solid #F0DFC2;border-radius:14px;padding:12px 16px;
  display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;font-size:13.5px;
}

/* System & Lifecycle Panels */
.sys-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:20px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow-sm)}
.panel-h{font-size:15.5px;font-weight:700;color:var(--ink);margin:0 0 10px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.gate-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px dashed var(--line);font-size:13px}
.gate-row:last-child{border-bottom:0}
.gbadge{padding:2px 9px;border-radius:99px;font-size:11.5px;font-weight:600}
.gbadge.pass{background:var(--sage-soft);color:var(--sage)}
.gbadge.lock{background:var(--amber-soft);color:#9B6515}
.gbadge.busy{background:var(--terra-soft);color:var(--terra-hover)}
.gbadge.err{background:#FDECEC;color:#B42318}
.film-qnote{margin-top:6px;padding:7px 10px;border-radius:10px;background:var(--amber-soft);border:1px solid #F0DFC2;font-size:12px;color:#7C4D0E;line-height:1.5}
.film-qnote.busy{background:var(--terra-soft);border-color:#F2C9BA;color:var(--terra-hover)}
.film-qnote.err{background:#FDECEC;border-color:#F5C2C0;color:#991B1B}
.toast-bar{position:fixed;right:20px;bottom:22px;z-index:120;max-width:440px;padding:11px 16px;border-radius:12px;background:rgba(43,35,31,.94);color:#FFF8F1;font-size:13px;line-height:1.5;box-shadow:var(--shadow-lg);opacity:0;transform:translateY(10px);pointer-events:none;transition:opacity .22s ease,transform .22s ease}
.toast-bar.show{opacity:1;transform:translateY(0)}
.toast-bar.err{background:rgba(153,27,27,.95)}
.pbar-wrap{background:var(--surface-soft);border:1px solid var(--line);border-radius:99px;height:10px;overflow:hidden;margin:8px 0}
.pbar-fill{background:linear-gradient(90deg,var(--sage),var(--terra));height:100%;transition:width .4s ease}
.vf-list{display:flex;flex-direction:column;gap:7px;margin-top:8px}
.vf-row{
  display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;
  background:var(--surface-soft);border:1px solid var(--line);border-radius:11px;padding:7px 10px;font-size:12px;
}
.vf-row.dim{opacity:.68;background:#FAF6F0}
.vf-main{display:flex;align-items:center;gap:6px;min-width:0;flex-wrap:wrap}
.vf-role{
  font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:99px;letter-spacing:.2px;flex-shrink:0;
}
.vf-role.pro{background:var(--amber-soft);color:#9B6515}
.vf-role.fast{background:var(--sage-soft);color:var(--sage)}
.vf-chan{font-size:11px;color:var(--muted)}
.vf-meta{display:flex;align-items:center;gap:8px;font-variant-numeric:tabular-nums;color:var(--ink-2)}
.vf-dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--sage)}
.vf-dot.rate_limited{background:var(--amber)} .vf-dot.quota_out{background:#C84B3E}
.hgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(265px,1fr));gap:14px}
.hcard{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:14px;border-top:4px solid var(--sage);box-shadow:var(--shadow-sm)}
.hcard.warn{border-top-color:var(--amber)} .hcard.off{border-top-color:#B8AFA8;opacity:.75}
.htop{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.htag{font-size:11.5px;padding:2px 9px;border-radius:99px;background:var(--sage-soft);color:var(--sage);font-weight:600}
.hcard.warn .htag{background:var(--amber-soft);color:#9B6515}

/* Family Face Recognition UI */
.fam-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.fam-card{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:var(--shadow-sm);display:flex;flex-direction:column;gap:10px}
.fam-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.fam-name{font-size:17px;font-weight:700;color:var(--ink);display:flex;align-items:center;gap:6px}
.fam-thumbs{display:flex;gap:8px;flex-wrap:wrap;min-height:68px;align-items:center}
.fam-th{position:relative;width:68px;height:68px;border-radius:12px;overflow:hidden;border:1px solid var(--line);background:var(--surface-soft);flex-shrink:0}
.fam-th img{width:100%;height:100%;object-fit:cover;display:block}
.fam-del{position:absolute;top:2px;right:2px;width:18px;height:18px;border:0;border-radius:50%;background:rgba(24,18,15,.76);color:#fff;font-size:11px;line-height:18px;padding:0;cursor:pointer;text-align:center}
.cand-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:12px}
.cand-card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:12px;box-shadow:var(--shadow-sm);display:flex;flex-direction:column;gap:8px}
.cand-card img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:12px;background:#1a1512}
.cand-btns{display:flex;flex-wrap:wrap;gap:5px}
.cbtn{border:1px solid var(--line);background:var(--surface-soft);color:var(--ink-2);border-radius:8px;padding:4px 8px;font-size:11.5px;cursor:pointer;font-weight:600}
.cbtn:hover{background:var(--terra);border-color:var(--terra);color:#fff}
.cbtn.ignore{background:#F6F1EA;color:var(--muted)}
.ppl-pill{display:inline-flex;align-items:center;gap:3px;font-size:11px;font-weight:600;padding:1px 7px;border-radius:99px;background:var(--sage-soft);color:var(--sage)}

/* Collapsible Details */
details.lapses{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:12px 16px;margin-top:18px}
details.lapses summary{cursor:pointer;font-weight:600;color:var(--ink-2);font-size:14px}

/* Fullscreen Lightbox */
.lightbox{
  position:fixed;inset:0;z-index:100;background:rgba(24,18,15,.92);backdrop-filter:blur(8px);
  display:none;flex-direction:column;align-items:center;justify-content:center;padding:18px;
}
.lightbox.open{display:flex}
.lb-stage{position:relative;max-width:94vw;max-height:76vh;display:flex;align-items:center;justify-content:center}
.lb-stage img{max-width:92vw;max-height:74vh;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.55);object-fit:contain;background:#000}
.lb-bar{
  margin-top:14px;max-width:920px;width:100%;background:rgba(44,35,31,.92);border:1px solid rgba(255,255,255,.12);
  border-radius:16px;padding:12px 16px;color:#FAF5EF;display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;
}
.lb-meta{font-size:13px;color:#DFD4CA}
.lb-meta b{color:#FFF}
.lb-ctrls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.lb-btn{
  border:1px solid rgba(255,255,255,.22);background:rgba(255,255,255,.08);color:#fff;
  border-radius:9px;padding:6px 12px;font-size:12.5px;cursor:pointer;transition:background .15s;text-decoration:none;display:inline-flex;align-items:center;gap:4px;
}
.lb-btn:hover{background:rgba(255,255,255,.18)}
.lb-btn.prim{background:var(--terra);border-color:var(--terra)}

.empty-note{
  background:var(--surface);border:1px dashed var(--line);border-radius:16px;padding:28px;
  text-align:center;color:var(--muted);font-size:14px;
}

@media (max-width:920px){
  .hero-card{grid-template-columns:1fr}
}
@media (max-width:640px){
  .topbar-in{padding:10px 14px}
  .wrap{padding:14px 14px 48px}
  .vid-grid,.photo-grid{grid-template-columns:1fr}
  .hero-h1{font-size:21px}
}
</style>
</head>
<body>

<header class="topbar">
  <div class="topbar-in">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">☀</div>
      <div>
        <div class="brand-title">时光成长馆 · 宝宝数字档案馆</div>
        <div class="age-pill" id="topAgePill">成长陪伴加载中…</div>
      </div>
    </div>

    <nav class="nav-tabs" aria-label="主导航">
      <button class="ntab active" data-tab="home" onclick="switchTab('home')">✨ 今日与精选</button>
      <button class="ntab" data-tab="daily" onclick="switchTab('daily')">📅 每日时光</button>
      <button class="ntab" data-tab="milestones" onclick="switchTab('milestones')">🌱 里程碑 · 童声</button>
      <button class="ntab" data-tab="cinema" onclick="switchTab('cinema')">🎞️ 流年大片</button>
      <button class="ntab" data-tab="family" onclick="switchTab('family')">👨‍👩‍👧 家人认脸</button>
      <button class="ntab" data-tab="system" onclick="switchTab('system')">🛡️ 守护与系统</button>
    </nav>

    <div class="nav-right">
      <button type="button" class="live-pill" onclick="switchTab('system')" title="查看源视频保护锁与实时引擎状态">
        <span id="topDot" class="sdot"></span>
        <span id="topPillTxt">加载中…</span>
      </button>
      <a class="mon-link" href="/monitor">监控大屏 ↗</a>
    </div>
  </div>
</header>

<main class="wrap">
  <!-- TAB 1: HOME -->
  <section id="view-home" class="view active">
    <div class="hero-card">
      <div class="hero-media" id="heroVideoBox"></div>
      <div class="hero-body">
        <div>
          <span class="hero-kicker" id="heroKicker">✦ 最新每日成长精华</span>
          <h1 class="hero-h1" id="heroTitle">成长的每一天都值得温柔珍藏</h1>
          <p class="hero-desc" id="heroDesc">NAS 本地无人值守精编 · 只读守护原始监控 · 纯童声与珍贵亲子瞬间自动留存。</p>
          <div class="stat-grid" id="heroStats"></div>
        </div>
        <div class="hero-actions">
          <button type="button" class="btn" id="heroDayBtn" onclick="openHeroDay()">查看该日完整档案 →</button>
          <button type="button" class="btn sec" onclick="switchTab('milestones')">纯童声与高光馆</button>
        </div>
      </div>
    </div>

    <div class="sec-head">
      <div>
        <h2 class="sec-title">📷 光影精选相册</h2>
        <p class="sec-sub">Gemini 4 严选正脸神态 · 点击任意照片可全屏欣赏或对比全景原画</p>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button type="button" class="link-btn" onclick="togglePhotos(this)">视图：智能构图版</button>
        <button type="button" class="link-btn" onclick="switchTab('daily')">浏览全部日期 →</button>
      </div>
    </div>
    <div class="photo-grid" id="homePhotos"></div>

    <div class="sec-head">
      <div>
        <h2 class="sec-title">🌱 近期里程碑与纯童声</h2>
        <p class="sec-sub">含「叫妈妈/叫爸爸」纯童声核验与「温馨日常照护」T0 原画留存</p>
      </div>
      <button type="button" class="link-btn" onclick="switchTab('milestones')">按分类检索全部 →</button>
    </div>
    <div class="vid-grid" id="homeMilestones"></div>
  </section>

  <!-- TAB 2: DAILY TIMELINE -->
  <section id="view-daily" class="view">
    <div class="sec-head">
      <div>
        <h2 class="sec-title">📅 每日时光档案馆</h2>
        <p class="sec-sub">P0 今日/昨日优先同步更新 · 点击日期卡片回看每日 1080p 精华、相册与全天延时</p>
      </div>
    </div>

    <div class="day-strip" id="dayStrip"></div>

    <div class="day-toolbar">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <label for="daySel" style="font-size:13px;font-weight:600;color:var(--ink-2)">快速跳转日期：</label>
        <select id="daySel" onchange="loadDay(this.value)"></select>
        <span id="dayMeta" style="font-size:13px;color:var(--muted)"></span>
      </div>
      <button type="button" class="btn sec" id="dailyModeBtn" onclick="togglePhotos(this)">视图：智能构图版</button>
    </div>

    <div id="dayHighlight"></div>

    <div class="sec-head">
      <div>
        <h3 class="sec-title" style="font-size:18px">当日精选生活照</h3>
      </div>
    </div>
    <div class="photo-grid" id="dayPhotos"></div>

    <div id="dayMsWrap" style="display:none">
      <div class="sec-head">
        <div><h3 class="sec-title" style="font-size:18px">当日高光里程碑与童声</h3></div>
      </div>
      <div class="vid-grid" id="dayMsGrid"></div>
    </div>

    <details class="lapses" id="dayLapsesWrap">
      <summary>⏱️ 展开当日各机位全天 1fps 浓缩延时回放</summary>
      <div class="vid-grid" id="dayLapses" style="margin-top:14px"></div>
    </details>
  </section>

  <!-- TAB 3: MILESTONES & SPEECH -->
  <section id="view-milestones" class="view">
    <div class="sec-head">
      <div>
        <h2 class="sec-title">🌱 成长里程碑与纯童声档案馆</h2>
        <p class="sec-sub">覆盖学步、站立、玩耍、叫妈妈/爸爸纯童声及洗澡擦身等温馨照护瞬间（局部隐私柔焦保护）</p>
      </div>
    </div>

    <div class="filter-bar">
      <div class="fpills" id="msCatPills"></div>
      <div class="filter-row">
        <select id="msCamSel" onchange="fetchMilestones()">
          <option value="all">全部机位</option>
        </select>
        <input id="msSearch" class="search-in" type="search" placeholder="搜索画面描述、日期（如 2026-11）、“妈妈”、“站立”、“照护”…" oninput="debounceMilestones()">
        <button type="button" class="btn sec" onclick="resetMsFilter()">重置筛选</button>
      </div>
    </div>

    <div id="pendingReviewBox" style="margin-bottom:14px"></div>
    <div class="vid-grid" id="msGrid"></div>
  </section>

  <!-- TAB 4: CINEMA & MONTHLY GROWTH -->
  <section id="view-cinema" class="view">
    <div class="sec-head">
      <div>
        <h2 class="sec-title">🎞️ 月龄成长流年与四季影院</h2>
        <p class="sec-sub">按时间轴记录宝宝从新生到周岁的外貌与动作蜕变 · 支持客厅电视与浏览器直接赏片</p>
      </div>
    </div>
    <div id="cinemaMain"></div>
    <div class="sec-head" style="margin-top:18px">
      <div><h3 class="sec-title" style="font-size:18px">🍂 四季影院 · 春夏秋冬成长大片（高思考导演编排）</h3></div>
    </div>
    <div class="vid-grid" id="seasonalCinemaGrid"></div>
    <div class="sec-head" style="margin-top:18px">
      <div><h3 class="sec-title" style="font-size:18px">🎬 月龄片与周岁片成片队列、导演剧本与逐镜溯源</h3></div>
    </div>
    <div class="vid-grid" id="filmsListGrid"></div>
    <div class="sec-head">
      <div><h3 class="sec-title" style="font-size:18px">✨ 1–11 个月成长定格卡（点击大图赏析）</h3></div>
    </div>
    <div class="photo-grid" id="monthlyCardsGrid"></div>
    <div class="sec-head">
      <div><h3 class="sec-title" style="font-size:18px">📽️ 四大固定视角平滑流年成片</h3></div>
    </div>
    <div class="vid-grid" id="smoothFilmsGrid"></div>
  </section>

  <!-- TAB 5: FAMILY FACE RECOGNITION -->
  <section id="view-family" class="view">
    <div class="sec-head">
      <div>
        <h2 class="sec-title">👨‍👩‍👧 家庭成员人脸识别库（防把奶奶认成妈妈）</h2>
        <p class="sec-sub">家里固定成员（妈妈/爸爸/奶奶/爷爷/姥姥/姥爷/小姨/月嫂，可随时增删）：上传生活照或直接从监控抓拍中点选标注，自动生成识别对照卡并锁定本地 ArcFace + 云端称呼</p>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <label style="font-size:12.5px;color:var(--ink-2);display:inline-flex;align-items:center;gap:5px">
          <input type="checkbox" id="famVisualOnlyChk"> 无正脸时也作为视觉参考录入
        </label>
        <button type="button" class="btn sec" onclick="discoverFamilyFaces(this)">🔍 扫描近期相册抓拍人脸</button>
      </div>
    </div>

    <div class="sys-grid" style="margin-bottom:16px">
      <div class="panel">
        <h3 class="panel-h">🧠 本地人脸引擎与称呼守卫 <span id="famEngBadge" class="gbadge pass">检测中</span></h3>
        <div id="famEngDesc" style="font-size:13px;color:var(--ink-2);line-height:1.65">加载中…</div>
        <div style="margin-top:12px;padding-top:10px;border-top:1px dashed var(--line);display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          <label for="famRecurateDay" style="font-size:12.5px;font-weight:600;color:var(--ink-2)">重编指定日期配字：</label>
          <input type="date" id="famRecurateDay" class="search-in" style="min-width:140px;flex:0 0 auto;padding:5px 10px">
          <button type="button" class="btn" onclick="recurateSelectedDay(this)">✨ 用最新人脸库重编配字</button>
        </div>
        <div id="famRecurateMsg" style="font-size:12px;color:var(--sage);margin-top:6px"></div>
      </div>

      <div class="panel">
        <h3 class="panel-h">🪪 云端 VLM 家庭识别对照卡预览 <span id="famSheetStat" class="gbadge lock">待录入</span></h3>
        <div id="famSheetBox" style="font-size:12.5px;color:var(--muted)">每位成员录入 1～3 张清晰照片后，系统会自动拼出带名签对照卡随监控画面发给 AI，并同步做本地 ArcFace 512 维特征校验。</div>
      </div>
    </div>

    <div class="sec-head" style="margin-top:10px">
      <div><h3 class="sec-title" style="font-size:18px">👤 家庭成员名单与参考照（建议每人传 1～3 张正面/日常照）</h3></div>
      <button type="button" class="btn sec" onclick="toggleAddMemberForm()">＋ 新增家庭成员</button>
    </div>
    <div id="addMemberBox" class="panel" style="display:none;margin-bottom:14px">
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
        <input id="newMemLabel" class="search-in" style="min-width:130px;flex:0 1 150px" placeholder="称呼（如：姑姑）">
        <select id="newMemGender">
          <option value="female">女性</option>
          <option value="male">男性</option>
          <option value="unknown">其他</option>
        </select>
        <input id="newMemFrom" type="date" title="生效开始日" class="search-in" style="min-width:130px;flex:0 0 auto">
        <input id="newMemUntil" type="date" title="生效结束日" class="search-in" style="min-width:130px;flex:0 0 auto">
        <input id="newMemNotes" class="search-in" style="min-width:180px;flex:1" placeholder="特征备忘（如：戴金丝眼镜、短发）">
        <button type="button" class="btn" onclick="saveNewMember()">保存成员</button>
      </div>
    </div>
    <div id="famUploadHint" style="font-size:12.5px;color:var(--sage);margin-bottom:10px"></div>
    <div class="fam-grid" id="famMembersGrid"></div>

    <div class="sec-head" style="margin-top:24px">
      <div>
        <h3 class="sec-title" style="font-size:18px">📸 从监控相册抓拍到的未知/待确认成人（点击下方姓名直接入库）</h3>
        <p class="sec-sub">监控俯拍角度与生活自拍略有差异，从这里点选 1～2 张监控抓拍图加入对应成员，识别准确率会大幅提升</p>
      </div>
    </div>
    <div id="famCandGrid" class="cand-grid"><div class="empty-note" style="grid-column:1/-1">点击上方「🔍 扫描近期相册抓拍人脸」自动提取待标注人脸。</div></div>
  </section>

  <!-- TAB 6: SYSTEM, STORAGE & PRIVACY GUARD -->
  <section id="view-system" class="view">
    <div class="sec-head">
      <div>
        <h2 class="sec-title">🛡️ 数据安全保护锁 · 百度网盘策略 · 实时引擎</h2>
        <p class="sec-sub">独立存储于 NAS /volume2/Growth_Archive_V3 · 彻底隔离原始监控录像</p>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <a class="btn sage" href="/monitor">打开实时进度监控大屏</a>
        <button type="button" id="sysPauseBtn" class="btn" onclick="togglePause(this)">暂停后台队列</button>
      </div>
    </div>

    <div class="sys-grid">
      <div class="panel">
        <h3 class="panel-h">🔒 源视频四重防误删保护锁 <span id="lifeLockBadge" class="gbadge lock">守护中</span></h3>
        <div id="lifeGuardText" style="font-size:13px;color:var(--ink-2);margin-bottom:10px">评估中…</div>
        <div class="gate-row"><span>1. 百度网盘首次源片备份完成</span><span id="g1" class="gbadge lock">校验中</span></div>
        <div class="gate-row"><span>2. 历史录像倒序归档 100% 追平</span><span id="g2" class="gbadge lock">校验中</span></div>
        <div class="gate-row"><span>3. 单日 T1/T2/精华 与 T0 校验完整</span><span id="g3" class="gbadge pass">强制启用</span></div>
        <div class="gate-row"><span>4. 原片本地保留期 &amp; 保险箱永久守护</span><span id="g4" class="gbadge pass">90 天 · Baby_Vault 永久</span></div>
      </div>

      <div class="panel">
        <h3 class="panel-h">☁️ 百度网盘平滑切换与本地留存 <span id="baiduPctBadge" class="gbadge lock">同步中</span></h3>
        <div id="baiduDesc" style="font-size:13px;color:var(--ink-2)">正在连接备份服务…</div>
        <div class="pbar-wrap"><div id="baiduBar" class="pbar-fill" style="width:0%"></div></div>
        <div id="baiduMeta" style="font-size:12px;color:var(--muted);margin-bottom:10px"></div>
        <div style="background:var(--surface-warm);border:1px solid var(--line-soft);border-radius:10px;padding:10px 12px;font-size:12.5px;color:var(--ink-2)">
          <div><b>源视频策略：</b><span id="sourcePolicyTxt">daily_enabled=false · 首次完成后原视频不再上百度云，在 NAS 本地保留 90 天</span></div>
          <div style="margin-top:4px"><b>数字档案同步：</b><span id="syncStatusTxt">待就绪</span></div>
        </div>
      </div>

      <div class="panel">
        <h3 class="panel-h">⚙️ 实时归档引擎与云端账单</h3>
        <div style="font-size:13.5px;color:var(--ink)">
          <b>当前状态：</b><span id="opsPhase">加载中</span><span id="opsDay"></span>
        </div>
        <div style="font-size:13px;color:var(--ink-2);margin-top:4px">
          已处理 <b><span id="opsDone">0</span>/<span id="opsTot">0</span>（<span id="opsPct">0</span>%）</b>
          · 原 <span id="opsOrig">0 MB</span> → 归档 <span id="opsArch">0 MB</span>
        </div>
        <div class="pbar-wrap"><div id="opsBar" class="pbar-fill" style="width:0%"></div></div>
        <div id="opsStep" style="font-size:12px;color:var(--muted);margin-bottom:6px">守护进程就绪中…</div>
        <div style="font-size:12.5px;color:var(--ink-2);margin-bottom:4px">
          NAS 可用内存 <b><span id="opsMem">—</span> GiB</b>（安全线 4.0 GiB）
          · 云端 Token <b><span id="opsTokTot">0</span></b>（入 <span id="opsTokIn">0</span> / 出 <span id="opsTokOut">0</span>）
        </div>
        <div id="vlmChips" class="vf-list"></div>
      </div>
    </div>

    <div class="sec-head">
      <div><h3 class="sec-title" style="font-size:18px">📹 摄像头在线状态与隐私守护策略</h3></div>
    </div>
    <div class="hgrid" id="camHealthGrid"></div>
  </section>
</main>

<!-- LIGHTBOX MODAL -->
<div id="lightbox" class="lightbox" role="dialog" aria-modal="true" onclick="if(event.target===this)closeLightbox()">
  <div class="lb-stage">
    <img id="lbImg" src="" alt="成长精选放大预览">
  </div>
  <div class="lb-bar">
    <div class="lb-meta" id="lbMeta"></div>
    <div class="lb-ctrls">
      <button type="button" class="lb-btn" onclick="stepLightbox(-1)">‹ 上一张</button>
      <button type="button" class="lb-btn" onclick="stepLightbox(1)">下一张 ›</button>
      <button type="button" class="lb-btn" id="lbSwitchBtn" onclick="toggleLightboxVariant()">切换：完整原画</button>
      <a class="lb-btn prim" id="lbDl" href="#" download>⬇ 下载高清图</a>
      <button type="button" class="lb-btn" onclick="closeLightbox()">✕ 关闭</button>
    </div>
  </div>
</div>
<div id="toastBar" class="toast-bar" role="status" aria-live="polite"></div>

<script>
const BOOT = __BOOT_JSON__;
const PHASE_ZH = __PHASE_JSON__;

let showFull = false;
let currentDay = (BOOT.hero_day && BOOT.hero_day.day) || (BOOT.days && BOOT.days[0] && BOOT.days[0].day) || '2026-11-08';
let activeCat = 'all';
let msTimer = null;
let lbItems = [];
let lbIndex = 0;
let lbUseFull = false;
let lbVer = '';
let _dayUpdatedMap = {};
let _photoErrRefreshTs = {};
let _dayPhotosSig = {};
let _statusDaySig = {};
let _poolVerMap = {};

function esc(v){
  return String(v == null ? '' : v)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function fmtGb(n){ const v = Number(n||0); return v >= 1e12 ? (v/1e12).toFixed(2)+' TB' : (v >= 1e9 ? (v/1e9).toFixed(2)+' GB' : (v/1e6).toFixed(1)+' MB'); }
function fmtTok(n){ const v = Math.round(Number(n||0)); return v >= 10000 ? (v/10000).toFixed(2)+' 万' : v.toLocaleString(); }
function mediaVerUrl(rel, ver){ if(!rel)return ''; return ver ? `${rel}${rel.includes('?') ? '&' : '?'}v=${encodeURIComponent(ver)}` : rel; }
function fmtFilmDur(sec){ const s = Math.round(Number(sec||0)); if(!s)return ''; const m = Math.floor(s/60), r = s%60; return m > 0 ? `${m}分${String(r).padStart(2,'0')}秒` : `${r}秒`; }

let familyMembersCache = [];

function switchTab(tab, updateUrl = true){
  document.querySelectorAll('.ntab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + tab));
  if (tab === 'family') loadFamilyMembers();
  if (updateUrl) {
    const u = new URL(location.href);
    u.searchParams.set('tab', tab);
    if (tab === 'daily' && currentDay) u.searchParams.set('day', currentDay);
    history.replaceState(null, '', u.pathname + u.search);
  }
}

async function fileToResizedJpegB64(file, maxEdge = 1600){
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('读取图片失败'));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('图片解码失败'));
      img.onload = () => {
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        const scale = Math.min(1, maxEdge / Math.max(w, h, 1));
        const cw = Math.max(1, Math.round(w * scale));
        const ch = Math.max(1, Math.round(h * scale));
        const canvas = document.createElement('canvas');
        canvas.width = cw;
        canvas.height = ch;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, cw, ch);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.90);
        resolve(dataUrl.split(',')[1] || '');
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

async function loadFamilyMembers(){
  try {
    const r = await fetch('/api/family/members', {cache: 'no-store'});
    if (!r.ok) return;
    const data = await r.json();
    renderFamilyData(data);
  } catch(e) {}
}

function renderFamilyData(data){
  const eng = data.engine || {};
  const members = data.members || [];
  familyMembersCache = members;
  const badge = document.getElementById('famEngBadge');
  if (badge) {
    badge.textContent = eng.engine_available ? 'SCRFD + ArcFace 就绪' : '仅云端对照模式';
    badge.className = 'gbadge ' + (eng.engine_available ? 'pass' : 'lock');
  }
  const desc = document.getElementById('famEngDesc');
  if (desc) {
    desc.innerHTML = `本地模型：<b>${esc(eng.det_model||'det_500m.onnx')}</b> + <b>${esc(eng.rec_model||'w600k_mbf.onnx')}</b>（512 维特征，纯本地运行不上云）<br>` +
      `防误称守卫：<b>${eng.strict_unverified_to_family ? '已开启（未确认身份绝不猜妈妈/爸爸，统一写“家人”）' : '温和模式'}</b> · ` +
      `云端视觉对照卡：<b>${eng.attach_reference_sheet ? '随图附带' : '关闭'}</b>`;
  }
  const recDay = document.getElementById('famRecurateDay');
  if (recDay && !recDay.value) recDay.value = currentDay || '';

  const sStat = document.getElementById('famSheetStat');
  const sBox = document.getElementById('famSheetBox');
  const totalSamples = members.reduce((acc, m) => acc + (m.sample_count || 0), 0);
  if (data.reference_sheet_rel && totalSamples > 0) {
    if (sStat) { sStat.textContent = `已包含 ${members.filter(m => m.sample_count > 0).length} 位成员`; sStat.className = 'gbadge pass'; }
    if (sBox) sBox.innerHTML = `<img src="${esc(mediaVerUrl(data.reference_sheet_rel, data.sheet_ts || Date.now()))}" style="width:100%;max-height:240px;object-fit:contain;border-radius:12px;background:#181513;display:block">`;
  } else {
    if (sStat) { sStat.textContent = '待录入样本'; sStat.className = 'gbadge lock'; }
    if (sBox) sBox.innerHTML = '暂未上传家庭成员参考照。上传后会自动生成【家庭成员识别对照卡】供 AI 配字核对；在此之前，防误称守卫已自动避免把照护人瞎猜成妈妈。';
  }

  const grid = document.getElementById('famMembersGrid');
  if (grid) {
    grid.innerHTML = members.map(m => {
      const ths = (m.samples || []).map(s => `
        <div class="fam-th" title="${esc(s.source_note || s.source)} · 检测分 ${Number(s.det_score||0).toFixed(2)}">
          <img src="${esc(s.thumb_rel)}" alt="${esc(m.label_zh)}">
          <button type="button" class="fam-del" title="删除样本" onclick="deleteFamilySample('${esc(s.sample_id)}')">×</button>
        </div>`).join('');
      const dateStr = (m.active_from || m.active_until)
        ? `生效期：${esc(m.active_from || '不限')} ~ ${esc(m.active_until || '长期')}`
        : '长期常驻成员';
      const gTag = m.gender === 'female' ? '女性' : (m.gender === 'male' ? '男性' : '成员');
      const enTag = m.enabled ? '' : '<span class="gbadge lock">已停用</span>';
      return `<article class="fam-card" style="${m.enabled ? '' : 'opacity:.65'}">
        <div class="fam-head">
          <div class="fam-name">${esc(m.label_zh)} <span class="cat-pill" style="font-size:11px;padding:1px 8px">${gTag}</span> ${enTag}</div>
          <span style="font-size:12px;color:var(--muted)">${m.sample_count || 0} 张参考（${m.embedding_count || 0} 个向量）</span>
        </div>
        <div style="font-size:12px;color:var(--muted)">${dateStr}${m.notes ? ' · ' + esc(m.notes) : ''}</div>
        <div class="fam-thumbs">${ths || '<span style="font-size:12px;color:var(--muted)">尚无参考照</span>'}</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:auto">
          <label class="btn" style="padding:6px 12px;font-size:12px">
            ＋ 上传照片
            <input type="file" accept="image/*" multiple style="display:none" onchange="uploadMemberFiles('${esc(m.member_id)}', this)">
          </label>
          <button type="button" class="btn sec" style="padding:6px 10px;font-size:12px" onclick="toggleMemberEdit('${esc(m.member_id)}')">编辑备注/日期</button>
        </div>
        <div id="edit-${esc(m.member_id)}" style="display:none;padding-top:8px;border-top:1px dashed var(--line);font-size:12px">
          <div style="display:flex;flex-direction:column;gap:6px">
            <input id="ed-lbl-${esc(m.member_id)}" class="search-in" style="min-width:0;padding:5px 8px" value="${esc(m.label_zh)}" placeholder="称呼">
            <input id="ed-note-${esc(m.member_id)}" class="search-in" style="min-width:0;padding:5px 8px" value="${esc(m.notes||'')}" placeholder="外貌备忘（如：戴黑框眼镜、短卷发）">
            <div style="display:flex;gap:6px">
              <input id="ed-af-${esc(m.member_id)}" type="date" class="search-in" style="min-width:0;flex:1;padding:4px 6px" value="${esc(m.active_from||'')}">
              <input id="ed-au-${esc(m.member_id)}" type="date" class="search-in" style="min-width:0;flex:1;padding:4px 6px" value="${esc(m.active_until||'')}">
            </div>
            <label style="display:flex;align-items:center;gap:5px"><input id="ed-en-${esc(m.member_id)}" type="checkbox" ${m.enabled ? 'checked' : ''}> 启用该成员</label>
            <button type="button" class="btn sage" style="padding:5px 10px;font-size:12px" onclick="saveMemberEdit('${esc(m.member_id)}','${esc(m.gender)}')">保存修改</button>
          </div>
        </div>
      </article>`;
    }).join('');
  }

  renderCandidatesGrid(data.candidates || [], members);
}

function renderCandidatesGrid(cands, members){
  const box = document.getElementById('famCandGrid');
  if (!box) return;
  if (!cands || !cands.length) {
    box.innerHTML = '<div class="empty-note" style="grid-column:1/-1">暂无待标注的抓拍人脸。点击上方「🔍 扫描近期相册抓拍人脸」可自动从监控精选照片中抽取尚未高置信匹配的面孔。</div>';
    return;
  }
  const activeMems = (members || familyMembersCache || []).filter(m => m.enabled);
  box.innerHTML = cands.map(c => {
    const guessTxt = c.guess_label ? `猜测：${esc(c.guess_label)} (${Number(c.guess_score||0).toFixed(2)})` : '待认领人脸';
    const btns = activeMems.map(m =>
      `<button type="button" class="cbtn" onclick="assignCandidate('${esc(c.cand_id)}','${esc(m.member_id)}',this)">${esc(m.label_zh)}</button>`
    ).join('');
    return `<article class="cand-card">
      <img src="${esc(c.thumb_rel)}" alt="抓拍人脸">
      <div style="font-size:12px;color:var(--ink-2)"><b>${guessTxt}</b><br><span style="color:var(--muted)">${esc(c.day||'')} · ${esc(c.source_file||'')}</span></div>
      <div class="cand-btns">
        ${btns}
        <button type="button" class="cbtn ignore" onclick="assignCandidate('${esc(c.cand_id)}','baby',this)">宝宝/忽略</button>
      </div>
    </article>`;
  }).join('');
}

function toggleMemberEdit(mid){
  const el = document.getElementById('edit-' + mid);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function toggleAddMemberForm(){
  const el = document.getElementById('addMemberBox');
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function saveNewMember(){
  const lbl = (document.getElementById('newMemLabel').value || '').trim();
  if (!lbl) return alert('请填写成员称呼');
  const mid = 'mem_' + Date.now().toString(36);
  const r = await fetch('/api/family/member', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      member_id: mid,
      label_zh: lbl,
      gender: document.getElementById('newMemGender').value || 'unknown',
      active_from: document.getElementById('newMemFrom').value || '',
      active_until: document.getElementById('newMemUntil').value || '',
      notes: document.getElementById('newMemNotes').value || '',
      enabled: true,
    }),
  });
  if (r.ok) {
    document.getElementById('newMemLabel').value = '';
    document.getElementById('newMemNotes').value = '';
    toggleAddMemberForm();
    loadFamilyMembers();
  }
}

async function saveMemberEdit(mid, gender){
  const r = await fetch('/api/family/member', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      member_id: mid,
      label_zh: (document.getElementById('ed-lbl-' + mid).value || '').trim(),
      gender,
      notes: (document.getElementById('ed-note-' + mid).value || '').trim(),
      active_from: document.getElementById('ed-af-' + mid).value || '',
      active_until: document.getElementById('ed-au-' + mid).value || '',
      enabled: Boolean(document.getElementById('ed-en-' + mid).checked),
    }),
  });
  if (r.ok) loadFamilyMembers();
}

async function uploadMemberFiles(mid, input){
  const files = Array.from(input.files || []);
  if (!files.length) return;
  const hint = document.getElementById('famUploadHint');
  const allowVis = Boolean(document.getElementById('famVisualOnlyChk') && document.getElementById('famVisualOnlyChk').checked);
  let okN = 0;
  const errs = [];
  for (const f of files) {
    if (hint) hint.textContent = `正在提取特征：${f.name}…`;
    try {
      const b64 = await fileToResizedJpegB64(f);
      const r = await fetch('/api/family/sample', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({member_id: mid, image_b64: b64, source_note: f.name, allow_visual_only: allowVis}),
      });
      const res = await r.json();
      if (r.ok) okN += 1;
      else errs.push(`${f.name}: ${res.error || '失败'}`);
    } catch(e) {
      errs.push(`${f.name}: ${e.message || '处理异常'}`);
    }
  }
  input.value = '';
  if (hint) {
    hint.textContent = errs.length
      ? `成功录入 ${okN} 张；未通过 ${errs.length} 张（${errs.join('；')}）`
      : `✅ 已成功录入 ${okN} 张参考照并刷新识别对照卡`;
  }
  loadFamilyMembers();
}

async function deleteFamilySample(sid){
  const r = await fetch('/api/family/sample/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sample_id: sid}),
  });
  if (r.ok) loadFamilyMembers();
}

async function discoverFamilyFaces(btn){
  if (btn) { btn.disabled = true; btn.textContent = '⏳ 正在扫描近期相册抓拍…'; }
  try {
    const r = await fetch('/api/family/discover', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({max_days: 14, max_photos: 60}),
    });
    if (r.ok) {
      const d = await r.json();
      renderCandidatesGrid(d.candidates || [], familyMembersCache);
    }
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '🔍 扫描近期相册抓拍人脸'; }
  }
}

async function assignCandidate(candId, memberId, btn){
  if (btn) btn.disabled = true;
  const r = await fetch('/api/family/candidate/assign', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cand_id: candId, member_id: memberId}),
  });
  if (r.ok) loadFamilyMembers();
  else if (btn) btn.disabled = false;
}

async function recurateSelectedDay(btn){
  const d = (document.getElementById('famRecurateDay').value || '').trim();
  const msg = document.getElementById('famRecurateMsg');
  if (!d) { if (msg) msg.textContent = '请先选择要重编的日期'; return; }
  if (btn) btn.disabled = true;
  try {
    const r = await fetch('/api/ctl/curate_day', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({day: d}),
    });
    if (r.ok) {
      if (msg) msg.textContent = `✅ 已将 ${d} 加入高思考重编队列，守护进程将在 1~3 分钟内更新配字与精华视频。`;
    } else if (msg) {
      msg.textContent = '⚠️ 触发失败，请确认守护进程在线';
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function togglePhotos(btn){
  showFull = !showFull;
  document.querySelectorAll('.phimg').forEach(img => {
    const nxt = showFull ? img.dataset.full : img.dataset.framed;
    if (nxt) img.src = nxt;
  });
  const label = showFull ? '视图：完整全景原画' : '视图：智能构图版';
  document.querySelectorAll('[onclick*="togglePhotos"]').forEach(b => { b.textContent = label; });
  if (btn) btn.textContent = label;
}

function photoCardHtml(p, idx, poolName, ver = ''){
  const framedUrl = mediaVerUrl(p.framed || p.full, ver);
  const fullUrl = mediaVerUrl(p.full || p.framed, ver);
  const src = showFull ? fullUrl : framedUrl;
  const pplHtml = (Array.isArray(p.people) && p.people.length)
    ? `<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:6px">${p.people.map(nm => `<span class="ppl-pill">👤 ${esc(nm)}</span>`).join('')}</div>`
    : '';
  return `<article class="pcard" onclick="openLightbox('${poolName}', ${idx})">
    <div class="pcard-imgwrap">
      <img class="phimg" loading="lazy" data-framed="${esc(framedUrl)}" data-full="${esc(fullUrl)}" src="${esc(src)}" alt="${esc(p.caption_zh||'成长照片')}" onerror="handlePhotoImgError(this, '${poolName}', ${idx})">
      <span class="pcard-badge">${esc(p.camera)} · ${esc(p.time||'')}</span>
      ${p.score ? `<span class="pcard-score">★ ${esc(p.score)}/5</span>` : ''}
    </div>
    <div class="pcard-body">
      ${pplHtml}
      <div class="pcard-cap">${esc(p.caption_zh || '温馨成长瞬间')}</div>
      <div class="pcard-meta"><span>智能缩放 ${esc(p.zoom_x || 1)}×</span><span>点击放大 / 对比原画</span></div>
    </div>
  </article>`;
}

function _showPhotoFallback(img, text){
  if (!img) return;
  img.style.display = 'none';
  const wrap = img.parentElement;
  if (!wrap) return;
  let fb = wrap.querySelector('.pcard-imgfallback');
  if (!fb) {
    fb = document.createElement('div');
    fb.className = 'pcard-imgfallback';
    wrap.insertBefore(fb, wrap.firstChild);
  }
  fb.textContent = text;
}

async function refreshDayPhotosOnly(dayStr){
  if (!dayStr) return;
  try {
    const r = await fetch('/api/day/' + encodeURIComponent(dayStr), {cache:'no-store'});
    if (!r.ok) return;
    const d = await r.json();
    if (d.day !== currentDay) return;
    const photos = d.photos || [];
    const sig = JSON.stringify(photos);
    _dayUpdatedMap[d.day] = d.updated_at || '';
    if (_dayPhotosSig[d.day] === sig) return;
    _dayPhotosSig[d.day] = sig;
    window._POOL_DAILY = photos;
    _poolVerMap._POOL_DAILY = d.updated_at || '';
    const box = document.getElementById('dayPhotos');
    if (box) {
      box.innerHTML = photos.length
        ? photos.map((p, i) => photoCardHtml(p, i, '_POOL_DAILY', d.updated_at || '')).join('')
        : '<div class="empty-note">该日期暂无达到“清晰正脸/优质构图”入选标准的精选照，或正在等待昼间录像处理。</div>';
    }
  } catch(_e) {}
}

async function refreshHomePhotosOnly(dayStr){
  if (!dayStr) return;
  try {
    const r = await fetch('/api/day/' + encodeURIComponent(dayStr), {cache:'no-store'});
    if (!r.ok) return;
    const d = await r.json();
    const photos = d.photos || [];
    if (!photos.length) return;
    BOOT.hero_day = d;
    BOOT.featured_photos = photos.slice(0, 8);
    window._POOL_HOME = BOOT.featured_photos;
    _poolVerMap._POOL_HOME = d.updated_at || '';
    const hp = document.getElementById('homePhotos');
    if (hp) {
      hp.innerHTML = window._POOL_HOME.map((p, i) => photoCardHtml(p, i, '_POOL_HOME', d.updated_at || '')).join('');
    }
  } catch(_e) {}
}

function handlePhotoImgError(img, poolName, _idx){
  if (!img) return;
  if (!img.dataset.triedAlt) {
    img.dataset.triedAlt = '1';
    const curAttr = img.getAttribute('src') || '';
    const altUrl = showFull ? (img.dataset.framed || '') : (img.dataset.full || '');
    if (altUrl && altUrl !== curAttr) {
      img.src = altUrl;
      return;
    }
  }
  const now = Date.now();
  let refreshing = false;
  if (poolName === '_POOL_DAILY' && currentDay) {
    if (now - (_photoErrRefreshTs['daily:' + currentDay] || 0) > 5000) {
      _photoErrRefreshTs['daily:' + currentDay] = now;
      refreshing = true;
      refreshDayPhotosOnly(currentDay);
    }
  } else if (poolName === '_POOL_HOME' && BOOT.hero_day && BOOT.hero_day.day) {
    const hdDay = BOOT.hero_day.day;
    if (now - (_photoErrRefreshTs['home:' + hdDay] || 0) > 5000) {
      _photoErrRefreshTs['home:' + hdDay] = now;
      refreshing = true;
      refreshHomePhotosOnly(hdDay);
    }
  }
  _showPhotoFallback(img, refreshing ? '照片已更新，正在刷新…' : '图片暂不可用');
}

function milestoneCardHtml(m){
  const cls = m.category === 'speech' ? 'speech' : (m.category === 'daily_care' ? 'daily_care' : '');
  const ageTxt = m.age_label ? ` · ${esc(m.age_label)}` : '';
  return `<article class="mcard">
    <video controls preload="${m.poster ? 'none' : 'metadata'}" poster="${esc(m.poster||'')}" src="${esc(m.video||'')}"></video>
    <div class="mcard-body">
      <div class="mcard-top">
        <span class="cat-pill ${cls}">${esc(m.cat_zh || m.category)}</span>
        <span style="font-size:12px;color:var(--muted)">${esc(m.time_str || m.day || '')}${ageTxt}</span>
      </div>
      <div class="mcard-cap">${esc(m.caption_zh || '')}</div>
      <div class="mcard-meta">
        <span>📹 ${esc(m.cam_zh || m.camera || '')}（${Number(m.zoom_x||1)}×）</span>
        ${m.speaker_note ? `<span>🎙️ ${esc(m.speaker_note)}</span>` : ''}
        ${m.t0_copy ? `<span class="t0-tag">🔐 T0原画: ${esc(m.t0_copy)}</span>` : ''}
      </div>
    </div>
  </article>`;
}

function renderHome(){
  const hd = BOOT.hero_day;
  const vbox = document.getElementById('heroVideoBox');
  const ac = BOOT.archive_counts || {};
  const tot = BOOT.totals || {};
  const msPack = BOOT.milestones_pack || {items:[], total:0, category_counts:{}};
  const life = BOOT.lifecycle || {};
  const age = life.age || {};

  if (hd && hd.highlight_rel) {
    const hdVer = hd.updated_at || '';
    const hdPoster = hd.highlight_poster_rel ? ` poster="${esc(mediaVerUrl(hd.highlight_poster_rel, hdVer))}"` : '';
    vbox.innerHTML = `<video controls preload="metadata"${hdPoster} src="${esc(mediaVerUrl(hd.highlight_rel, hdVer))}"></video>`;
    const st = hd.daily_story || null;
    const durTag = fmtFilmDur(hd.highlight_duration_s || (st && st.actual_duration_s) || (st && st.est_duration_s));
    const shotCnt = hd.highlight_shots_count || (st && st.shots && st.shots.length) || 0;
    const metaBadges = [durTag, shotCnt ? `${shotCnt}镜短剧` : ''].filter(Boolean).join(' · ');
    document.getElementById('heroKicker').textContent = `✦ ${hd.day} · 每日 1080p 成长短剧${metaBadges ? ' · ' + metaBadges : ''}`;
    document.getElementById('heroTitle').textContent = (st && st.film_title_zh) ? st.film_title_zh : `${hd.day} 的温暖成长故事已上线`;
    const hdAge = hd.age_info ? `宝宝当时 ${hd.age_info.label_zh}（第 ${hd.age_info.day_n} 天）。` : '';
    document.getElementById('heroDesc').textContent = (st && st.logline_zh)
      ? st.logline_zh
      : `${hdAge}已从全屋监控中精选高光片段并平滑过渡成片，支持点击切换每日档案或全屏赏片。`;
  } else {
    vbox.innerHTML = `<div class="hero-empty"><div style="font-size:28px;margin-bottom:8px">🎞️</div><div>最新每日精华正在自动剪辑中<br><span style="font-size:12px;opacity:.75">今日完成首批画面筛选与 Gemini 4 复核后即刻生成</span></div></div>`;
  }

  const archB = Number(tot.t1_bytes||0) + Number(tot.t2_bytes||0);
  const ratio = Number(tot.orig_bytes||0) > 0 ? (archB * 100 / Number(tot.orig_bytes)).toFixed(1) + '%' : '精简归档';
  document.getElementById('heroStats').innerHTML = `
    <div class="stat-tile"><div class="stat-val">${ac.days_count || (BOOT.days||[]).length} 天</div><div class="stat-lbl">已入库归档日期</div></div>
    <div class="stat-tile"><div class="stat-val">${msPack.total || 0} 段</div><div class="stat-lbl">高光里程碑（含 ${(msPack.category_counts||{}).speech || 0} 纯童声）</div></div>
    <div class="stat-tile"><div class="stat-val">${ac.photos_count || (BOOT.featured_photos||[]).length} 张</div><div class="stat-lbl">精选生活照片</div></div>
    <div class="stat-tile"><div class="stat-val">${ratio}</div><div class="stat-lbl">${fmtGb(tot.orig_bytes||0)} → ${fmtGb(archB)}</div></div>
  `;

  window._POOL_HOME = BOOT.featured_photos || [];
  const homeVer = (hd && hd.updated_at) || '';
  _poolVerMap._POOL_HOME = homeVer;
  const hp = document.getElementById('homePhotos');
  hp.innerHTML = window._POOL_HOME.length
    ? window._POOL_HOME.map((p, i) => photoCardHtml(p, i, '_POOL_HOME', homeVer)).join('')
    : '<div class="empty-note">首批精选照片将在今日白昼活动时段分析完成后自动呈现。</div>';

  const hm = document.getElementById('homeMilestones');
  const topMs = (msPack.items || []).slice(0, 6);
  hm.innerHTML = topMs.length
    ? topMs.map(milestoneCardHtml).join('')
    : '<div class="empty-note">正在检索高光动作、温馨照护与叫妈妈/爸爸纯童声片段…</div>';

  if (age.label_zh) {
    document.getElementById('topAgePill').textContent = `出生第 ${age.day_n} 天 · ${age.label_zh}`;
  }
}

function openHeroDay(){
  const hdDay = (BOOT.hero_day && BOOT.hero_day.day) || currentDay;
  switchTab('daily');
  loadDay(hdDay);
}

function renderDayStrip(){
  const strip = document.getElementById('dayStrip');
  const sel = document.getElementById('daySel');
  const days = BOOT.days || [];
  strip.innerHTML = '';
  sel.innerHTML = '';
  days.forEach(d => {
    const o = document.createElement('option');
    o.value = d.day;
    o.textContent = `${d.day}${d.priority_tier ? ' · '+d.priority_tier : ''}${d.highlight_rel ? ' · 有精华片' : ''}`;
    sel.appendChild(o);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'dchip' + (d.day === currentDay ? ' active' : '');
    btn.dataset.day = d.day;
    btn.onclick = () => loadDay(d.day);
    const icons = [
      d.highlight_rel ? '🎬 精华' : '⏳ 归档中',
      d.photos_count ? `📷 ${d.photos_count}` : '',
    ].filter(Boolean).join(' · ');
    btn.innerHTML = `<div class="dchip-d">${esc(d.day)}</div><div class="dchip-s">${esc(icons)}</div>`;
    strip.appendChild(btn);
  });
}

function renderDay(d){
  currentDay = d.day;
  const sel = document.getElementById('daySel');
  if (sel && sel.value !== d.day) sel.value = d.day;
  document.querySelectorAll('.dchip').forEach(c => c.classList.toggle('active', c.dataset.day === d.day));

  const ageStr = d.age_info ? `${d.age_info.label_zh}（第 ${d.age_info.day_n} 天）` : '';
  const filesStr = d.total_files ? `已处理 ${d.done_files||0}/${d.total_files||0} 档 · 原片 ${fmtGb(d.orig_bytes||0)}` : (d.status || '精选样板日');
  document.getElementById('dayMeta').textContent = [ageStr, filesStr].filter(Boolean).join(' · ');

  const hl = document.getElementById('dayHighlight');
  const st = d.daily_story || null;
  if (d.highlight_rel) {
    const hlVer = d.updated_at || '';
    const hlPoster = d.highlight_poster_rel ? ` poster="${esc(mediaVerUrl(d.highlight_poster_rel, hlVer))}"` : '';
    const durStr = fmtFilmDur(d.highlight_duration_s || (st && st.actual_duration_s) || (st && st.est_duration_s));
    const shotN = d.highlight_shots_count || (st && st.shots && st.shots.length) || 0;
    const headTitle = (st && st.film_title_zh) ? `🎬 ${st.film_title_zh}` : `🎬 ${d.day} 每日 1080p 高光短片`;
    const badgeStr = [durStr, shotN ? `${shotN} 镜` : ''].filter(Boolean).join(' · ');
    const scRows = (st && Array.isArray(st.shots) ? st.shots : []).map(sh => `
      <div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px dashed var(--line);font-size:12.5px;align-items:baseline;flex-wrap:wrap">
        <span style="font-weight:700;color:var(--terra-hover);min-width:34px">${esc(sh.shot_id || ('#'+sh.index))}</span>
        <span style="font-family:ui-monospace,Consolas,monospace;color:var(--muted);min-width:64px">${esc((sh.time||'').slice(0,8))}</span>
        <span class="cat-pill" style="font-size:11px;padding:1px 8px">${esc(sh.chapter_zh || '')}</span>
        ${Array.isArray(sh.people) && sh.people.length ? sh.people.map(nm => `<span class="ppl-pill">👤 ${esc(nm)}</span>`).join('') : ''}
        <span style="color:var(--ink);font-weight:500;flex:1">${esc(sh.lyric_zh || '')}</span>
        <span style="font-size:11.5px;color:var(--muted)">${esc(sh.camera || '')} · ${Number(sh.dur_s||0).toFixed(1)}s</span>
      </div>`).join('');
    const storyHtml = st ? `
      <div style="margin-top:14px;padding:14px 16px;background:var(--surface-warm);border:1px solid var(--line);border-radius:14px">
        ${st.logline_zh ? `<div style="font-size:14px;color:var(--ink);line-height:1.65">${esc(st.logline_zh)}</div>` : ''}
        <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;font-size:12.5px">
          ${st.joy_summary_zh ? `<div style="background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px 12px;color:var(--ink-2)"><b>😊 开心时刻：</b>${esc(st.joy_summary_zh)}</div>` : ''}
          ${st.growth_changes_zh ? `<div style="background:var(--sage-soft);border:1px solid rgba(73,123,104,.22);border-radius:10px;padding:6px 12px;color:var(--sage)"><b>🌱 成长发现：</b>${esc(st.growth_changes_zh)}</div>` : ''}
        </div>
        ${scRows ? `<details style="margin-top:10px"><summary style="cursor:pointer;font-size:12.5px;font-weight:600;color:var(--terra-hover)">📖 查看四幕导演分镜剧本（${st.shots.length} 镜）</summary><div style="margin-top:8px">${scRows}</div></details>` : ''}
      </div>` : '';
    hl.innerHTML = `<div class="cinema-box">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px">
        <div>
          <span style="font-size:14.5px;font-weight:700;color:var(--terra-hover)">${esc(headTitle)}</span>
          ${badgeStr ? `<span style="margin-left:8px;font-size:12px;color:var(--muted)">${esc(badgeStr)}</span>` : ''}
        </div>
        <button type="button" class="btn sec" style="padding:5px 12px;font-size:12px" onclick="recurateDay('${esc(d.day)}', this)">✨ 重新导演剪辑</button>
      </div>
      <video controls preload="metadata"${hlPoster} src="${esc(mediaVerUrl(d.highlight_rel, hlVer))}"></video>
      ${storyHtml}
    </div>`;
  } else {
    hl.innerHTML = `<div class="empty-note">${esc(d.day)} 每日精华短片正在自动合成中（完成当日批处理与高思考导演编排后出现）。</div>`;
  }

  window._POOL_DAILY = d.photos || [];
  const dayVer = d.updated_at || '';
  _dayUpdatedMap[d.day] = dayVer;
  _dayPhotosSig[d.day] = JSON.stringify(window._POOL_DAILY);
  _poolVerMap._POOL_DAILY = dayVer;
  document.getElementById('dayPhotos').innerHTML = window._POOL_DAILY.length
    ? window._POOL_DAILY.map((p, i) => photoCardHtml(p, i, '_POOL_DAILY', dayVer)).join('')
    : '<div class="empty-note">该日期暂无达到“清晰正脸/优质构图”入选标准的精选照，或正在等待昼间录像处理。</div>';

  const dms = d.milestones || [];
  const dmsWrap = document.getElementById('dayMsWrap');
  if (dms.length) {
    dmsWrap.style.display = 'block';
    document.getElementById('dayMsGrid').innerHTML = dms.map(milestoneCardHtml).join('');
  } else {
    dmsWrap.style.display = 'none';
  }

  const tls = d.timelapse_items || Object.entries(d.timelapses || {}).map(([cid, url]) => ({cid, name: (BOOT.camera_names||{})[cid] || cid, url}));
  const tlWrap = document.getElementById('dayLapsesWrap');
  if (tls.length) {
    tlWrap.style.display = 'block';
    document.getElementById('dayLapses').innerHTML = tls.map(t => `<article class="mcard"><video controls preload="metadata" src="${esc(t.url)}"></video><div class="mcard-body"><div class="mcard-cap">${esc(t.name)} · 全天 1fps 延时</div></div></article>`).join('');
  } else {
    tlWrap.style.display = 'none';
  }
}

async function loadDay(dayStr){
  if (!dayStr) return;
  const r = await fetch('/api/day/' + encodeURIComponent(dayStr), {cache:'no-store'});
  if (r.ok) renderDay(await r.json());
}
async function recurateDay(dayStr, btn){
  if (!dayStr) return;
  if (btn) { btn.disabled = true; btn.textContent = '⏳ 导演队列排期中…'; }
  try {
    const r = await fetch('/api/ctl/curate_day', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({day: dayStr}),
    });
    if (r.ok && btn) btn.textContent = '✅ 已触发高思考重剪，约1~3分钟刷新可见';
    else if (btn) { btn.disabled = false; btn.textContent = '⚠️ 触发失败，点击重试'; }
  } catch(e) {
    if (btn) { btn.disabled = false; btn.textContent = '⚠️ 网络异常，点击重试'; }
  }
}

function renderMilestonesUI(pack){
  const counts = pack.category_counts || {};
  const cats = pack.categories || {};
  const pillsBox = document.getElementById('msCatPills');
  const order = ['all', 'speech', 'standing', 'walking', 'daily_care', 'toy_play', 'happy_interaction', 'sofa_descent', 'kitchen_visit'];
  const allKeys = [...order, ...Object.keys(cats).filter(k => !order.includes(k))];
  pillsBox.innerHTML = '';
  allKeys.forEach(key => {
    const label = key === 'all' ? '全部瞬间' : (cats[key] || key);
    const cnt = counts[key] || 0;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'fpill' + (activeCat === key ? ' active' : '');
    b.textContent = `${label} (${cnt})`;
    b.onclick = () => { activeCat = key; fetchMilestones(); };
    pillsBox.appendChild(b);
  });

  const grid = document.getElementById('msGrid');
  const items = pack.items || [];
  grid.innerHTML = items.length
    ? items.map(milestoneCardHtml).join('')
    : '<div class="empty-note">当前筛选条件下暂无里程碑片段，可尝试切换分类或清空搜索词。</div>';
}

function populateMsCameras(){
  const sel = document.getElementById('msCamSel');
  const kall = document.createElement('option');
  kall.value = 'kitchen_all';
  kall.textContent = '🍳 全部厨房机位（旧小米 + G350）';
  sel.appendChild(kall);
  const names = BOOT.camera_names || {};
  Object.entries(names).forEach(([cid, cname]) => {
    const o = document.createElement('option');
    o.value = cid;
    o.textContent = cname;
    sel.appendChild(o);
  });
}

async function fetchMilestones(){
  const cam = document.getElementById('msCamSel').value;
  const q = document.getElementById('msSearch').value;
  const url = `/api/milestones?category=${encodeURIComponent(activeCat)}&camera=${encodeURIComponent(cam)}&q=${encodeURIComponent(q)}`;
  const r = await fetch(url);
  if (r.ok) renderMilestonesUI(await r.json());
}
function debounceMilestones(){ clearTimeout(msTimer); msTimer = setTimeout(fetchMilestones, 220); }
function resetMsFilter(){ activeCat = 'all'; document.getElementById('msCamSel').value = 'all'; document.getElementById('msSearch').value = ''; fetchMilestones(); }

const REVIEW_ST_ZH = {
  approved: '复核通过',
  pending_review: '待人工复核',
  rejected: '已退回排除',
  queued: '排队待编纂',
  rendering: '编纂渲染中',
  retry_wait: '失败待重试',
  failed: '生成失败',
  published: '已发布待核',
};

let _lastPendingReviewsSig = '';
const _cinemaSigs = {
  mainVideo: '',
  mainMeta: '',
  mainBanner: '',
  seasonal: '',
  coreFilms: '',
  monthlyCards: '',
  smoothFilms: '',
};

function _containerHasActiveVideo(root){
  if (!root) return false;
  return Array.from(root.querySelectorAll('video')).some(v => (!v.paused && !v.ended) || (v.currentTime > 0 && !v.ended));
}

function _filmRenderKey(f){
  if (!f) return null;
  return [
    f.film_id || '',
    f.status || '',
    f.review_status || '',
    f.video_rel || '',
    f.poster_rel || '',
    f.generated_at || f.updated_at || '',
    Number(f.duration_s || 0),
    Number(f.shot_count || (f.shots && f.shots.length) || 0),
    Number(f.attempts || 0),
    f.last_error || '',
    (f.story && f.story.film_title_zh) || f.title_zh || '',
    (f.story && f.story.logline_zh) || '',
  ];
}

function renderPendingReviews(force = false){
  const box = document.getElementById('pendingReviewBox');
  if (!box) return;
  const list = BOOT.pending_reviews || [];
  const sig = JSON.stringify(list);
  if (!force && sig === _lastPendingReviewsSig) return;
  _lastPendingReviewsSig = sig;
  if (!list.length) { box.innerHTML = ''; return; }
  box.innerHTML = `<div class="panel" style="border-color:var(--warn, #d97706)">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px">
      <b>🔍 云端拒答/历史待复核片段（${list.length} 条 · 未自动归为照护或里程碑）</b>
      <span style="font-size:12px;color:var(--muted)">需人工确认后才会进入里程碑与成片</span>
    </div>
    ${list.slice(0, 20).map(r => `<div class="gate-row" style="align-items:flex-start;gap:10px;padding:8px 0">
      <div style="font-size:12.5px;flex:1">
        <b>${esc(r.day || '')}</b> · ${esc(r.camera || '')} · <code>${esc(r.source_rel || '')}</code>
        <div style="color:var(--muted);font-size:12px">${esc(r.reason || 'content_policy_blocked')} · 建议分类 ${esc(r.suggested_category || 'daily_care')} · 片段 ${Number(r.start_s||0).toFixed(1)}s–${Number(r.end_s||0).toFixed(1)}s</div>
      </div>
      <div style="display:flex;gap:6px;flex-shrink:0">
        <button type="button" class="btn sage" style="padding:4px 10px;font-size:12px" onclick="decideVlmReview('${esc(r.review_id)}','approve')">通过归档</button>
        <button type="button" class="btn sec" style="padding:4px 10px;font-size:12px" onclick="decideVlmReview('${esc(r.review_id)}','reject')">驳回排除</button>
      </div>
    </div>`).join('')}
  </div>`;
}

async function decideVlmReview(reviewId, action){
  const res = await fetch('/api/ctl/vlm_review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({review_id: reviewId, action}),
  });
  if (res.ok) {
    const st = await fetch('/api/status', {cache: 'no-store'});
    if (st.ok) updateSystemStatus(await st.json());
    await fetchMilestones();
  }
}

let _toastTimer = null;
function showToast(msg, isErr = false){
  const el = document.getElementById('toastBar');
  if (!el) return;
  el.textContent = msg;
  el.className = 'toast-bar show' + (isErr ? ' err' : '');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.classList.remove('show'); }, 4500);
}

function _filmQueueOrder(films){
  const pri = {growth_chronicle: 0, monthly: 1, camera_chronicle: 2, seasonal: 3, annual: 4};
  return (films || [])
    .filter(f => f && f.status === 'queued')
    .slice()
    .sort((a, b) => {
      const pa = pri[a.kind] ?? 4, pb = pri[b.kind] ?? 4;
      if (pa !== pb) return pa - pb;
      const aa = Number(a.age_months || 0), ab = Number(b.age_months || 0);
      if (aa !== ab) return aa - ab;
      return Number(a.created_at || a.updated_at || 0) - Number(b.created_at || b.updated_at || 0);
    })
    .map(f => f.film_id);
}

function _filmQueueStatusUi(liveF, isPaused, curSt, queuedOrder){
  const st = (liveF && liveF.status) || '';
  const isQueued = st === 'queued';
  const isRendering = st === 'rendering';
  const isFailed = st === 'retry_wait' || st === 'failed';
  let badgeHtml = '';
  let noteHtml = '';
  if (isRendering) {
    badgeHtml = '<span class="gbadge busy">🎬 导演剪辑中</span>';
    const det = curSt ? `当前进度：${esc(curSt)}` : '正在运行高思考导演管线，成片完成后将自动刷新替换';
    noteHtml = `<div class="film-qnote busy">🎬 ${det}</div>`;
  } else if (isQueued) {
    const qIdx = queuedOrder.indexOf(liveF.film_id) + 1;
    badgeHtml = `<span class="gbadge busy">⏳ 排队中${qIdx > 0 ? ' · 第 ' + qIdx + ' 位' : ''}</span>`;
    let reason = '已加入导演编剪队列，即将自动开始剪辑…';
    if (isPaused) {
      reason = '已加入队列，但后台队列当前处于暂停状态，点击右上角「恢复后台队列」后将立即开始。';
    } else if (curSt) {
      reason = `已入队等待导演通道释放（当前后台审片：${esc(curSt)}）`;
    }
    noteHtml = `<div class="film-qnote">⏳ ${reason}</div>`;
  } else if (isFailed) {
    const att = Number((liveF && liveF.attempts) || 0);
    const errTxt = (liveF && liveF.last_error) ? `：${esc(liveF.last_error)}` : '';
    badgeHtml = `<span class="gbadge err">⚠️ ${esc(REVIEW_ST_ZH[st] || '生成失败')}</span>`;
    noteHtml = `<div class="film-qnote err">⚠️ 导演剪辑${att ? `（已尝试 ${att} 次）` : ''}异常${errTxt}</div>`;
  }
  return { status: st, isQueued, isRendering, isFailed, badgeHtml, noteHtml };
}

async function triggerFilmCompile(filmId, btn){
  if (!filmId) return;
  const orig = btn ? btn.textContent : '';
  if (btn) { btn.disabled = true; btn.textContent = '🎬 提交导演队列中…'; }
  try {
    const res = await fetch('/api/ctl/film/compile', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({film_id: filmId, force: true}),
    });
    let data = null;
    try { data = await res.json(); } catch(_je) {}
    if (res.ok) {
      BOOT.cinema = BOOT.cinema || {};
      if (data && typeof data.paused === 'boolean') BOOT.paused = data.paused;
      if (data && typeof data.curator_status === 'string') BOOT.curator_status = data.curator_status;
      try {
        const fr = await fetch('/api/films', {cache: 'no-store'});
        if (fr.ok) {
          const fd = await fr.json();
          BOOT.cinema.films = fd.films || [];
          if (fd.smooth_films) BOOT.cinema.smooth_films = fd.smooth_films;
          if (BOOT.lifecycle && fd.schedule) BOOT.lifecycle.monthly_schedule = fd.schedule;
          if (typeof fd.curator_status === 'string') BOOT.curator_status = fd.curator_status;
          if (typeof fd.paused === 'boolean') BOOT.paused = fd.paused;
        }
      } catch(_fe) {
        if (data && data.film) {
          const list = Array.isArray(BOOT.cinema.films) ? BOOT.cinema.films.slice() : [];
          const idx = list.findIndex(x => x && x.film_id === data.film.film_id);
          if (idx >= 0) list[idx] = data.film; else list.push(data.film);
          BOOT.cinema.films = list;
        }
      }
      renderCinema(false);
      const cur = String(BOOT.curator_status || '').trim();
      if (data && data.already_rendering) {
        showToast(`🎬 该视角正在导演剪辑中${cur ? '（' + cur + '）' : ''}，请勿重复提交`);
      } else if (BOOT.paused) {
        showToast('⏳ 已加入导演重剪队列（后台当前已暂停，恢复队列后自动开始）');
      } else if (cur) {
        showToast(`⏳ 已加入导演重剪队列，等待当前审片（${cur}）完成后开始`);
      } else {
        showToast('🎬 已加入导演编剪队列，后台正在启动高思考重剪…');
      }
    } else {
      showToast('⚠️ 排期失败：' + ((data && data.error) || ('HTTP ' + res.status)), true);
      if (btn) { btn.textContent = orig || '🎬 用导演管线重剪此视角'; btn.disabled = false; }
    }
  } catch(_e) {
    showToast('⚠️ 网络异常，未能提交导演重剪任务，请稍后重试', true);
    if (btn) { btn.textContent = orig || '🎬 用导演管线重剪此视角'; btn.disabled = false; }
  }
}

async function reviewFilm(filmId, action){
  const res = await fetch('/api/ctl/film/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({film_id: filmId, action}),
  });
  if (res.ok) {
    const fr = await fetch('/api/films', {cache: 'no-store'});
    if (fr.ok) {
      const fd = await fr.json();
      BOOT.cinema = BOOT.cinema || {};
      BOOT.cinema.films = fd.films || [];
      if (BOOT.lifecycle && fd.schedule) BOOT.lifecycle.monthly_schedule = fd.schedule;
      renderCinema(false);
    }
  }
}

function filmStoryBoxHtml(f){
  const st = f && f.story;
  if (!st) return '';
  return `<div style="margin-top:8px;padding:10px 12px;border-radius:10px;background:var(--surface-warm);border:1px solid var(--line-soft);font-size:12.5px;line-height:1.55">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap">
      <b>🎞️ ${esc(st.film_title_zh || f.title_zh || '')}</b>
      <span style="font-size:11px;color:var(--sage)">${esc(st.director_model || 'director')}</span>
    </div>
    ${st.logline_zh ? `<div style="margin-top:4px;color:var(--ink-2)">${esc(st.logline_zh)}</div>` : ''}
    ${st.joy_summary_zh ? `<div style="margin-top:3px;color:var(--muted)">😊 <b>开心点：</b>${esc(st.joy_summary_zh)}</div>` : ''}
    ${st.growth_changes_zh ? `<div style="margin-top:2px;color:var(--muted)">🌱 <b>新变化：</b>${esc(st.growth_changes_zh)}</div>` : ''}
  </div>`;
}

function _filmProvenanceDetailsHtml(f){
  const shots = (f && f.shots) || [];
  if (!shots.length) return '';
  const shotRows = shots.map(sh => `<tr>
    <td>#${sh.shot_index}</td>
    <td>${esc(sh.chapter_zh||'')}</td>
    <td>${esc(sh.day||'')}</td>
    <td>${esc(sh.camera_name||sh.camera||'')}</td>
    <td><code>${esc(sh.source_rel||'')}</code></td>
    <td>${esc(sh.lyric_zh||sh.caption_zh||'')}</td>
    <td>${Number(sh.source_start_s||0).toFixed(1)}–${Number(sh.source_end_s||0).toFixed(1)}s</td>
    <td>${Number(sh.film_offset_start_s||0).toFixed(1)}–${Number(sh.film_offset_end_s||0).toFixed(1)}s</td>
  </tr>`).join('');
  return `<details style="margin-top:8px;font-size:11.5px"><summary style="cursor:pointer;color:var(--muted)">查看分幕剧本与逐镜原片溯源（${shots.length} 镜）</summary>
    <div style="overflow-x:auto;margin-top:6px"><table style="width:100%;border-collapse:collapse;font-size:11px">
      <thead><tr><th>#</th><th>幕别</th><th>日期</th><th>机位</th><th>原片</th><th>台词手记</th><th>原片区间</th><th>成片区间</th></tr></thead>
      <tbody>${shotRows}</tbody>
    </table></div>
  </details>`;
}

function filmCardHtml(f){
  const rev = f.review_status || 'pending_review';
  const st = f.status || 'queued';
  const revCls = rev === 'approved' ? 'pass' : 'lock';
  const shots = f.shots || [];
  const fVer = f.generated_at || f.updated_at || '';
  const vidUrl = f.video_rel ? mediaVerUrl(f.video_rel, fVer) : '';
  const posUrl = f.poster_rel ? mediaVerUrl(f.poster_rel, fVer) : '';
  return `<article class="mcard">
    ${vidUrl ? `<video controls preload="${posUrl ? 'none' : 'metadata'}" ${posUrl ? `poster="${esc(posUrl)}"` : ''} src="${esc(vidUrl)}"></video>` : ''}
    <div class="mcard-body">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
        <b>${esc((f.story && f.story.film_title_zh) || f.title_zh || f.film_id)}</b>
        <span class="gbadge ${revCls}">${esc(REVIEW_ST_ZH[rev] || rev)} · ${esc(REVIEW_ST_ZH[st] || st)}</span>
      </div>
      <div style="font-size:12px;color:var(--muted);margin-top:4px">
        ${esc(f.period_title_zh || f.film_id)} · 周期 ${esc(f.period_start||'')} ~ ${esc(f.period_end||'')} · ${Number(f.shot_count||shots.length||0)} 镜 · ${fmtFilmDur(f.duration_s)}
        ${f.attempts ? ` · 重试 ${f.attempts} 次` : ''}
      </div>
      ${filmStoryBoxHtml(f)}
      ${f.last_error ? `<div style="font-size:12px;color:#b45309;margin-top:4px">最近错误：${esc(f.last_error)}</div>` : ''}
      <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
        ${f.video_rel ? `<button type="button" class="btn sage" style="padding:4px 10px;font-size:12px" onclick="reviewFilm('${esc(f.film_id)}','approve')">复核通过</button>` : ''}
        <button type="button" class="btn sec" style="padding:4px 10px;font-size:12px" onclick="reviewFilm('${esc(f.film_id)}','retry')">退回重编</button>
      </div>
      ${_filmProvenanceDetailsHtml(f)}
    </div>
  </article>`;
}

function renderCinema(force = false){
  const cin = BOOT.cinema || {};
  const life = BOOT.lifecycle || {};
  const sch = life.monthly_schedule || {};
  const mainBox = document.getElementById('cinemaMain');
  const films = cin.films || [];
  const filmById = Object.fromEntries(films.map(f => [f.film_id, f]));
  const curSt = String(BOOT.curator_status || '').trim();
  const isPaused = !!BOOT.paused;
  const queuedOrder = _filmQueueOrder(films);

  const pubFullSpan = films.find(f => f.video_rel && f.kind === 'growth_chronicle') || films.find(f => f.video_rel && f.kind === 'annual');
  const mf = cin.monthly_film;
  const pubFallback = films.find(f => f.video_rel && f.kind === 'monthly') || films.find(f => f.video_rel);
  const heroFilm = pubFullSpan || (mf ? null : pubFallback);

  let mainVidSlot = document.getElementById('cinemaMainVideoSlot');
  let mainMetaSlot = document.getElementById('cinemaMainMeta');
  let mainBanner = document.getElementById('cinemaScheduleBanner');
  if (!mainVidSlot || !mainMetaSlot || !mainBanner) {
    mainBox.innerHTML = `<div class="cinema-box"><div id="cinemaMainVideoSlot"></div><div id="cinemaMainMeta"></div><div class="schedule-banner" id="cinemaScheduleBanner"></div></div>`;
    mainVidSlot = document.getElementById('cinemaMainVideoSlot');
    mainMetaSlot = document.getElementById('cinemaMainMeta');
    mainBanner = document.getElementById('cinemaScheduleBanner');
  }

  let mainVidSrc = '';
  let mainPoster = '';
  if (heroFilm && heroFilm.video_rel) {
    const pvVer = heroFilm.generated_at || heroFilm.updated_at || '';
    mainVidSrc = mediaVerUrl(heroFilm.video_rel, pvVer);
    mainPoster = heroFilm.poster_rel ? mediaVerUrl(heroFilm.poster_rel, pvVer) : '';
  } else if (mf) {
    mainVidSrc = `v2/${mf.file}`;
    mainPoster = mf.poster ? `v2/${mf.poster}` : '';
  }
  const mainVidSig = JSON.stringify([mainVidSrc, mainPoster]);
  if (force || (_cinemaSigs.mainVideo !== mainVidSig && !_containerHasActiveVideo(mainVidSlot))) {
    mainVidSlot.innerHTML = mainVidSrc
      ? `<video controls preload="metadata"${mainPoster ? ` poster="${esc(mainPoster)}"` : ''} src="${esc(mainVidSrc)}"></video>`
      : '<div class="empty-note">全跨度成长汇总大片就绪后将在此呈现。</div>';
    _cinemaSigs.mainVideo = mainVidSig;
  }

  const gcLive = filmById['growth_chronicle'] || null;
  const gcSpec = sch.growth_chronicle || null;
  const gcUi = _filmQueueStatusUi(gcLive, isPaused, curSt, queuedOrder);
  const activeMetaFilm = heroFilm || gcLive;
  const mainMetaSig = JSON.stringify([
    isPaused,
    (gcUi.isQueued || gcUi.isRendering) ? curSt : '',
    _filmRenderKey(heroFilm),
    _filmRenderKey(gcLive),
    gcSpec ? gcSpec.title_zh : '',
    mf ? [mf.file, mf.shots, mf.duration_s] : null,
  ]);
  if (force || _cinemaSigs.mainMeta !== mainMetaSig) {
    const gcBusy = gcUi.isQueued || gcUi.isRendering;
    const compileBtn = gcBusy
      ? `<button type="button" class="btn sage" style="padding:5px 12px;font-size:12px" disabled>${gcUi.isRendering ? '🎬 全跨度大片导演剪辑中…' : '⏳ 已排队等待制作全跨度大片'}</button>`
      : `<button type="button" class="btn sage" style="padding:5px 12px;font-size:12px" onclick="triggerFilmCompile('growth_chronicle', this)">🎬 ${pubFullSpan ? '重新导演全跨度流年大片' : '导演制作 1–11 月全跨度流年大片'}</button>`;
    if (heroFilm && heroFilm.video_rel) {
      const hRev = heroFilm.review_status || 'pending_review';
      const hSt = heroFilm.status || 'queued';
      const hShots = heroFilm.shots || [];
      mainMetaSlot.innerHTML = `<div style="margin-top:12px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
          <div style="font-size:15px"><b>🌟 ${esc((heroFilm.story && heroFilm.story.film_title_zh) || heroFilm.title_zh || heroFilm.film_id)}</b>
            <span style="font-size:12px;color:var(--muted);margin-left:8px">${esc(heroFilm.period_start||'')} ~ ${esc(heroFilm.period_end||'')} · ${Number(heroFilm.shot_count||hShots.length||0)} 镜 · ${fmtFilmDur(heroFilm.duration_s)}</span>
          </div>
          <span class="gbadge ${hRev === 'approved' ? 'pass' : 'lock'}">${esc(REVIEW_ST_ZH[hRev] || hRev)} · ${esc(REVIEW_ST_ZH[hSt] || hSt)}</span>
        </div>
        ${filmStoryBoxHtml(heroFilm)}
        ${gcUi.noteHtml}
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
          ${compileBtn}
          <button type="button" class="btn sec" style="padding:5px 12px;font-size:12px" onclick="reviewFilm('${esc(heroFilm.film_id)}','approve')">复核通过</button>
          <button type="button" class="btn sec" style="padding:5px 12px;font-size:12px" onclick="reviewFilm('${esc(heroFilm.film_id)}','retry')">退回重编</button>
        </div>
        ${_filmProvenanceDetailsHtml(heroFilm)}
      </div>`;
    } else if (mf) {
      mainMetaSlot.innerHTML = `<div style="margin-top:12px;padding:10px 12px;border-radius:10px;background:var(--surface-warm);border:1px solid var(--line-soft)">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
          <div><b>🎞️ 宝宝 1–11 月龄跨月成长对比版</b> <span style="font-size:12px;color:var(--muted)">（${Number(mf.shots||12)} 镜 · ${fmtFilmDur(mf.duration_s)} · V2 经典版）</span></div>
          ${compileBtn}
        </div>
        ${gcUi.noteHtml}
      </div>`;
    } else if (activeMetaFilm) {
      mainMetaSlot.innerHTML = `<div style="margin-top:12px">${gcUi.noteHtml}<div style="margin-top:8px">${compileBtn}</div></div>`;
    } else {
      mainMetaSlot.innerHTML = `<div style="margin-top:10px">${compileBtn}</div>`;
    }
    _cinemaSigs.mainMeta = mainMetaSig;
  }

  const mainBannerSig = JSON.stringify([
    sch.compile_day || 18,
    sch.next_compile_date || '',
    sch.next_title_zh || '',
    sch.note_zh || '',
    sch.published_films_count || 0,
    sch.pending_review_films_count || 0,
    sch.monthly_cards_count || 0,
  ]);
  if (force || _cinemaSigs.mainBanner !== mainBannerSig) {
    mainBanner.innerHTML = `
      <div><b>📅 每月 ${sch.compile_day || 18} 日自动流年编纂计划（收齐 17 日满月/周岁全天录像归档后启动）：</b>下个节点 <b>${esc(sch.next_compile_date || '2026-12-18')}</b>（${esc(sch.next_title_zh || '月龄更新')}）</div>
      <div style="color:var(--muted)">${sch.note_zh ? esc(sch.note_zh) + ' · ' : ''}已发布 ${sch.published_films_count || 0} 部成片 · 待复核 ${sch.pending_review_films_count || 0} 部 · ${sch.monthly_cards_count || 0} 张月龄卡</div>
    `;
    _cinemaSigs.mainBanner = mainBannerSig;
  }
  const seaGrid = document.getElementById('seasonalCinemaGrid');
  if (seaGrid) {
    const seaCat = sch.seasonal_catalog || [];
    const anySeaBusy = seaCat.some(sc => {
      const lf = filmById[sc.film_id];
      const eff = (lf && lf.status) || sc.existing_status || '';
      return eff === 'queued' || eff === 'rendering';
    });
    const seaSig = JSON.stringify([
      isPaused,
      anySeaBusy ? curSt : '',
      seaCat.map(sc => [
        sc.film_id, sc.title_zh, sc.period_start, sc.period_end, sc.has_footage,
        sc.done_files || 0, sc.existing_status || '', _filmRenderKey(filmById[sc.film_id]),
      ]),
    ]);
    if (force || (_cinemaSigs.seasonal !== seaSig && !_containerHasActiveVideo(seaGrid))) {
      seaGrid.innerHTML = seaCat.length ? seaCat.map(sc => {
        const liveF = filmById[sc.film_id];
        if (liveF && liveF.video_rel) return filmCardHtml(liveF);
        const effSt = (liveF && liveF.status) || sc.existing_status || '';
        const qUi = _filmQueueStatusUi(liveF, isPaused, curSt, queuedOrder);
        const stTxt = effSt === 'rendering'
          ? '🎬 正在导演剪辑中…'
          : (effSt === 'queued'
            ? '⏳ 已排队等待导演剪辑'
            : (sc.has_footage ? `已归档 ${sc.done_files || 0} 档素材 · 待制作` : '等待对应季度历史录像归档'));
        const btnHtml = sc.has_footage
          ? ((effSt === 'rendering' || effSt === 'queued')
              ? `<div style="margin-top:10px"><button type="button" class="btn sage" style="padding:5px 12px;font-size:12px" disabled>${effSt === 'rendering' ? '🎬 导演管线剪辑中…' : '⏳ 已加入导演队列（排队中）'}</button></div>`
              : `<div style="margin-top:10px"><button type="button" class="btn sage" style="padding:5px 12px;font-size:12px" onclick="triggerFilmCompile('${esc(sc.film_id)}', this)">🎬 导演制作本季大片</button></div>`)
          : '';
        return `<article class="mcard"><div class="mcard-body">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
            <b>${esc(sc.title_zh)}</b>
            <span class="gbadge ${effSt === 'queued' || effSt === 'rendering' ? 'busy' : (sc.has_footage ? 'pass' : 'lock')}">${esc(stTxt)}</span>
          </div>
          <div style="font-size:12px;color:var(--muted);margin-top:6px">季节周期：${esc(sc.period_start)} ～ ${esc(sc.period_end)}</div>
          ${qUi.noteHtml}
          ${btnHtml}
        </div></article>`;
      }).join('') : '<div class="empty-note">四季影院编排目录加载中…</div>';
      _cinemaSigs.seasonal = seaSig;
    }
  }

  const filmsGrid = document.getElementById('filmsListGrid');
  if (filmsGrid) {
    const coreFilms = films.filter(f => f.kind !== 'seasonal' && f.kind !== 'camera_chronicle' && f.kind !== 'growth_chronicle');
    const coreSig = JSON.stringify(coreFilms.map(_filmRenderKey));
    if (force || (_cinemaSigs.coreFilms !== coreSig && !_containerHasActiveVideo(filmsGrid))) {
      filmsGrid.innerHTML = coreFilms.length
        ? coreFilms.map(filmCardHtml).join('')
        : '<div class="empty-note">月龄与周岁成片尚未排期生成。</div>';
      _cinemaSigs.coreFilms = coreSig;
    }
  }

  const mcList = cin.monthly_cards || [];
  const cardsBox = document.getElementById('monthlyCardsGrid');
  const cardsSig = JSON.stringify(mcList);
  if (cardsBox && (force || _cinemaSigs.monthlyCards !== cardsSig)) {
    window._POOL_MONTH = mcList.map(c => ({
      full: `v2/monthly_cards/${c.card}`,
      framed: `v2/monthly_cards/${c.card}`,
      camera: `${c.age_lbl}（${c.camera}）`,
      zoom_x: c.zoom_x || 1,
      time: `${c.month} · 第${c.day_n}天`,
      caption_zh: c.stage_zh,
    }));
    _poolVerMap._POOL_MONTH = '';
    cardsBox.innerHTML = window._POOL_MONTH.length
      ? window._POOL_MONTH.map((p, i) => photoCardHtml(p, i, '_POOL_MONTH')).join('')
      : '<div class="empty-note">月龄成长定格卡尚未生成。</div>';
    _cinemaSigs.monthlyCards = cardsSig;
  }

  const sfs = cin.smooth_films || [];
  const smoothGrid = document.getElementById('smoothFilmsGrid');
  if (smoothGrid) {
    const anySmoothBusy = sfs.some(sf => {
      const lf = filmById[sf.film_id || ('angle_' + (sf.angle || ''))];
      return lf && (lf.status === 'queued' || lf.status === 'rendering');
    });
    const smoothSig = JSON.stringify([
      isPaused,
      anySmoothBusy ? curSt : '',
      sfs.map(sf => {
        const angId = sf.film_id || ('angle_' + (sf.angle || ''));
        return [
          sf.file, sf.poster, sf.shots, sf.duration_s, sf.name, sf.zoom_x,
          !!sf.director_v3, sf.logline_zh || '', _filmRenderKey(filmById[angId]),
        ];
      }),
    ]);
    if (force || (_cinemaSigs.smoothFilms !== smoothSig && !_containerHasActiveVideo(smoothGrid))) {
      smoothGrid.innerHTML = sfs.length
        ? sfs.map(sf => {
            const angId = sf.film_id || ('angle_' + (sf.angle || ''));
            const liveF = filmById[angId];
            const qUi = _filmQueueStatusUi(liveF, isPaused, curSt, queuedOrder);
            const fVer = (liveF && (liveF.generated_at || liveF.updated_at)) || '';
            const vSrc = mediaVerUrl(`v2/${sf.file}`, fVer);
            const pSrc = mediaVerUrl(`v2/${sf.poster}`, fVer);
            const actionBtn = (sf.angle || sf.film_id)
              ? (qUi.isRendering
                  ? `<div style="margin-top:8px"><button type="button" class="btn sec" style="padding:4px 10px;font-size:12px" disabled>🎬 导演管线剪辑中…</button></div>`
                  : (qUi.isQueued
                      ? `<div style="margin-top:8px"><button type="button" class="btn sec" style="padding:4px 10px;font-size:12px" disabled>⏳ 已加入导演队列（排队中）</button></div>`
                      : `<div style="margin-top:8px"><button type="button" class="btn sec" style="padding:4px 10px;font-size:12px" onclick="triggerFilmCompile('${esc(angId)}', this)">${qUi.isFailed ? '🔄 立即重试导演重剪' : '🎬 用导演管线重剪此视角'}</button></div>`))
              : '';
            return `<article class="mcard"><video controls preload="none" poster="${esc(pSrc)}" src="${esc(vSrc)}"></video><div class="mcard-body">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap">
                <b>${esc(sf.name)}（${sf.zoom_x}× · ${sf.shots} 镜 · ${fmtFilmDur(sf.duration_s)}）</b>
                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
                  ${sf.director_v3 ? '<span class="gbadge pass">V3 导演流年版</span>' : '<span class="gbadge lock">V2 基础版</span>'}
                  ${qUi.badgeHtml}
                </div>
              </div>
              ${sf.logline_zh ? `<div style="font-size:12px;color:var(--ink-2);margin-top:4px">${esc(sf.logline_zh)}</div>` : ''}
              ${qUi.noteHtml}
              ${actionBtn}
            </div></article>`;
          }).join('')
        : '';
      _cinemaSigs.smoothFilms = smoothSig;
    }
  }
}

function updateSystemStatus(s){
  const paused = !!s.paused;
  const tot = s.totals || {};
  const j = (s.state && s.state.active_job) || {};
  const caps = (s.state && s.state.runtime_caps) || {};
  const life = s.lifecycle || BOOT.lifecycle || {};
  BOOT.paused = paused;
  if (s.state) BOOT.state = s.state;
  if (typeof s.curator_status === 'string') {
    BOOT.curator_status = s.curator_status;
  } else {
    const jCur = String(j.curator_status || '').trim();
    const detStr = String(j.detail || '');
    BOOT.curator_status = jCur || (detStr.includes('后台审片:') ? detStr.split('后台审片:')[1].trim() : '');
  }

  const done = Number(tot.done_files || 0), total = Number(tot.total_files || 0);
  const pct = total > 0 ? (done * 100 / total).toFixed(1) : '0.0';
  const phaseTxt = paused ? '已暂停' : (PHASE_ZH[j.phase] || j.phase || '待命扫描中');

  document.getElementById('topDot').className = 'sdot' + (paused ? ' paused' : '');
  document.getElementById('topPillTxt').textContent = `${paused ? '已暂停' : '运行中'} ${pct}% · ${fmtTok((s.token_totals||{}).total_tokens)} tok`;

  document.getElementById('opsPhase').textContent = phaseTxt;
  document.getElementById('opsDay').textContent = j.day ? ' · ' + j.day : '';
  document.getElementById('opsDone').textContent = done;
  document.getElementById('opsTot').textContent = total;
  document.getElementById('opsPct').textContent = pct;
  document.getElementById('opsBar').style.width = Math.min(100, Number(pct)) + '%';
  document.getElementById('opsOrig').textContent = fmtGb(tot.orig_bytes);
  document.getElementById('opsArch').textContent = fmtGb(Number(tot.t1_bytes||0) + Number(tot.t2_bytes||0));
  const memKib = Number(j.host_mem_kib || caps.host_mem_kib || 0);
  document.getElementById('opsMem').textContent = memKib ? (memKib / (1024*1024)).toFixed(2) : '—';

  const stepText = [j.step, j.detail].filter(Boolean).join(' · ') || (j.last_file ? j.last_file.split('/').pop() : '守护进程运行中…');
  document.getElementById('opsStep').textContent = stepText;

  const tt = s.token_totals || {};
  document.getElementById('opsTokTot').textContent = fmtTok(tt.total_tokens);
  document.getElementById('opsTokIn').textContent = fmtTok(tt.input_tokens);
  document.getElementById('opsTokOut').textContent = fmtTok(tt.output_tokens);

  const box = document.getElementById('vlmChips');
  box.textContent = '';
  const fleet = (s.vlm_fleet && s.vlm_fleet.length)
    ? s.vlm_fleet
    : (s.vlm_stats || []).map(v => ({
        model: v.model,
        role: 'fast',
        channel_zh: v.provider || '',
        status: Number(v.cooldown_until || 0) > Date.now() / 1000 ? 'rate_limited' : 'online',
        calls_ok: v.calls_ok || 0,
        calls_err: v.calls_err || 0,
        avg_latency_s: v.calls_ok ? Number((v.total_latency_s / v.calls_ok).toFixed(1)) : null,
        total_tokens: v.total_tokens || 0,
      }));
  if (fleet.length) {
    fleet.forEach(c => {
      const row = document.createElement('div');
      row.className = 'vf-row' + (c.status === 'online' ? '' : ' dim');
      const main = document.createElement('div');
      main.className = 'vf-main';
      const rb = document.createElement('span');
      rb.className = 'vf-role ' + (c.role === 'pro' ? 'pro' : 'fast');
      rb.textContent = (c.role_zh && c.role_zh.includes('全能')) ? 'PRO+FAST' : (c.role === 'pro' ? 'PRO 高光' : 'FAST 筛选');
      const nm = document.createElement('b');
      nm.textContent = c.model;
      main.appendChild(rb);
      main.appendChild(nm);
      if (c.channel_zh) {
        const ch = document.createElement('span');
        ch.className = 'vf-chan';
        ch.textContent = c.channel_zh;
        main.appendChild(ch);
      }
      const meta = document.createElement('div');
      meta.className = 'vf-meta';
      const dt = document.createElement('span');
      dt.className = 'vf-dot ' + (c.status || 'online');
      const latTxt = c.avg_latency_s != null ? `${c.avg_latency_s}s` : '待命';
      const errTxt = c.calls_err ? `/✗${c.calls_err}` : '';
      const txt = document.createElement('span');
      txt.textContent = `✓${c.calls_ok}${errTxt} · ${latTxt} · ${fmtTok(c.total_tokens)} tok`;
      meta.appendChild(dt);
      meta.appendChild(txt);
      row.appendChild(main);
      row.appendChild(meta);
      box.appendChild(row);
    });
  } else {
    const empty = document.createElement('div');
    empty.className = 'vf-row';
    empty.textContent = 'Carbon + GPT-6.2 共审高光 · Hydrogen / Qwen4 / DeepSeek-4.1 自领筛选就绪';
    box.appendChild(empty);
  }

  const pbtn = document.getElementById('sysPauseBtn');
  pbtn.textContent = paused ? '恢复后台队列' : '暂停后台队列';
  pbtn.className = 'btn ' + (paused ? 'sage' : 'warn');
  if (s.lifecycle) BOOT.lifecycle = s.lifecycle;
  if (Array.isArray(s.smooth_films)) {
    BOOT.cinema = BOOT.cinema || {};
    BOOT.cinema.smooth_films = s.smooth_films;
  }
  if (Array.isArray(s.pending_reviews)) {
    BOOT.pending_reviews = s.pending_reviews;
    renderPendingReviews(false);
  }
  if (Array.isArray(s.films) || Array.isArray(s.smooth_films) || (s.lifecycle && s.lifecycle.monthly_schedule)) {
    BOOT.cinema = BOOT.cinema || {};
    if (Array.isArray(s.films)) BOOT.cinema.films = s.films;
    renderCinema(false);
  }
  if (Array.isArray(s.days)) {
    s.days.forEach(d => {
      if (!d || !d.day) return;
      const sig = `${d.updated_at ?? ''}|${d.photos_count ?? ''}|${d.highlight_rel || ''}|${d.done_files || 0}`;
      const prevSig = _statusDaySig[d.day];
      _statusDaySig[d.day] = sig;
      if (prevSig !== undefined && prevSig !== sig && d.day === currentDay) {
        if (!_containerHasActiveVideo(document.getElementById('view-daily'))) {
          loadDay(currentDay);
        } else {
          refreshDayPhotosOnly(currentDay);
        }
      }
    });
  }

  // Lifecycle & Baidu Status
  const gates = life.gates || {};
  const lockBadge = document.getElementById('lifeLockBadge');
  lockBadge.textContent = life.can_prune ? '满足放行条件' : '只读锁定不删源片';
  lockBadge.className = 'gbadge ' + (life.can_prune ? 'pass' : 'lock');
  document.getElementById('lifeGuardText').textContent = life.guard_summary_zh || '保护锁生效中';

  const g1 = document.getElementById('g1');
  g1.textContent = gates.source_backup_done ? '已完成' : '上传中（锁定）';
  g1.className = 'gbadge ' + (gates.source_backup_done ? 'pass' : 'lock');

  const g2 = document.getElementById('g2');
  g2.textContent = gates.archive_backfill_done ? '已全部追平' : `追平中（待办 ${gates.hist_queued ?? '—'} 档）`;
  g2.className = 'gbadge ' + (gates.archive_backfill_done ? 'pass' : 'lock');

  const bd = life.baidu_backup || {};
  const bpct = Number(bd.pct || 0).toFixed(1);
  document.getElementById('baiduPctBadge').textContent = bd.initial_done ? '首次已完成' : (bd.available ? `${bpct}%` : '连接中');
  document.getElementById('baiduPctBadge').className = 'gbadge ' + (bd.initial_done ? 'pass' : 'lock');
  document.getElementById('baiduBar').style.width = Math.min(100, Number(bpct)) + '%';
  if (bd.available) {
    document.getElementById('baiduDesc').textContent = bd.initial_done
      ? `首次原始录像备份已全部完成（累计 ${Number(bd.uploaded_files||0).toLocaleString()} 档 / ${fmtGb(bd.uploaded_bytes)}）。`
      : `cctv-baidu-backup 正在补传历史原片：${Number(bd.uploaded_files||0).toLocaleString()} / ${Number(bd.scanned_files||0).toLocaleString()} 档（${fmtGb(bd.uploaded_bytes)} / ${fmtGb(bd.scanned_bytes)}）`;
    document.getElementById('baiduMeta').textContent = bd.current_file ? `当前上传：${bd.current_file}` : (bd.heartbeat ? `心跳：${bd.heartbeat}` : '');
  } else {
    document.getElementById('baiduDesc').textContent = '暂未获取到实时备份状态（保护锁默认闭锁，绝不删除任何源文件）。';
  }
  const rDays = life.retention_days || 90;
  document.getElementById('sourcePolicyTxt').textContent = `daily_enabled=${Boolean(bd.daily_enabled)} · 全部完成后原视频不上百度云，仅在 NAS 本地滚动保留 ${rDays} 天（当前模式: ${life.retention_mode || 'observe_only'}）`;
  const cs = life.archive_cloud_sync || {};
  document.getElementById('syncStatusTxt').textContent = cs.note_zh || '精编数字档案独立保存于 /volume2/Growth_Archive_V3';
}

function renderCameraHealth(){
  const grid = document.getElementById('camHealthGrid');
  const list = BOOT.camera_health || [];
  grid.innerHTML = list.map(h => {
    const cls = h.status === 'stale_warning' ? 'warn' : ((h.status === 'online' || h.status === 'historical') ? 'ok' : 'off');
    const tag = h.status === 'stale_warning' ? '停更告警' : (h.status === 'historical' ? '历史归档已纳管' : (h.status === 'online' ? '正常在线' : '已停用'));
    const z = h.enabled ? ` · ${Number(h.zoom_factor||1)}× / ${h.t1_res||720}p15` : '';
    return `<div class="hcard ${cls}">
      <div class="htop"><b>${esc(h.name)}</b><span class="htag">${tag}</span></div>
      <div style="font-size:12px;color:var(--muted)">ID: ${esc(h.id)}${esc(z)} · 最新: ${esc(h.latest_ts||'—')}</div>
      <div style="font-size:12.5px;color:var(--ink-2);margin-top:6px">${esc(h.note||'')}</div>
    </div>`;
  }).join('');
}

async function togglePause(btn){
  const paused = btn && btn.textContent.includes('恢复');
  const res = await fetch('/api/ctl/' + (paused ? 'resume' : 'pause'), {method:'POST'});
  if (res.ok) {
    const st = await fetch('/api/status');
    if (st.ok) updateSystemStatus(await st.json());
  }
}

/* Lightbox */
function openLightbox(poolName, index){
  const list = window[poolName] || [];
  if (!list.length) return;
  lbItems = list;
  lbIndex = Math.max(0, Math.min(index, list.length - 1));
  lbUseFull = showFull;
  lbVer = _poolVerMap[poolName] || '';
  renderLightbox();
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox(){ document.getElementById('lightbox').classList.remove('open'); }
function stepLightbox(delta){
  if (!lbItems.length) return;
  lbIndex = (lbIndex + delta + lbItems.length) % lbItems.length;
  renderLightbox();
}
function toggleLightboxVariant(){ lbUseFull = !lbUseFull; renderLightbox(); }
function renderLightbox(){
  const p = lbItems[lbIndex];
  if (!p) return;
  const framedUrl = mediaVerUrl(p.framed || p.full, lbVer);
  const fullUrl = mediaVerUrl(p.full || p.framed, lbVer);
  const hasFull = p.full && p.framed && p.full !== p.framed;
  const src = lbUseFull ? fullUrl : framedUrl;
  const lbImg = document.getElementById('lbImg');
  lbImg.dataset.triedAlt = '';
  lbImg.onerror = () => {
    if (lbImg.dataset.triedAlt) return;
    lbImg.dataset.triedAlt = '1';
    const fallbackUrl = lbUseFull ? framedUrl : fullUrl;
    if (fallbackUrl && fallbackUrl !== src) lbImg.src = fallbackUrl;
  };
  lbImg.src = src;
  document.getElementById('lbDl').href = fullUrl || src;
  const swBtn = document.getElementById('lbSwitchBtn');
  swBtn.style.display = hasFull ? 'inline-flex' : 'none';
  swBtn.textContent = lbUseFull ? '切换：智能构图版' : '切换：完整原画';
  document.getElementById('lbMeta').innerHTML = `<b>${esc(p.camera || '')}</b> · ${esc(p.time || '')} ${p.score ? '· ★ '+esc(p.score)+'/5' : ''} (${lbIndex+1}/${lbItems.length})<br>${esc(p.caption_zh || '')}`;
}
document.addEventListener('keydown', e => {
  if (!document.getElementById('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  else if (e.key === 'ArrowLeft') stepLightbox(-1);
  else if (e.key === 'ArrowRight') stepLightbox(1);
});

(function init(){
  renderHome();
  renderDayStrip();
  populateMsCameras();
  renderPendingReviews(true);
  renderMilestonesUI(BOOT.milestones_pack || {items:[], category_counts:{}, categories:{}});
  renderCinema(true);
  renderCameraHealth();
  updateSystemStatus({
    paused: BOOT.paused,
    curator_status: BOOT.curator_status,
    totals: BOOT.totals,
    token_totals: BOOT.token_totals,
    vlm_stats: BOOT.vlm_stats,
    vlm_fleet: BOOT.vlm_fleet,
    state: BOOT.state,
    lifecycle: BOOT.lifecycle,
    days: BOOT.days,
  });

  const sp = new URLSearchParams(location.search);
  const qDay = sp.get('day');
  const qTab = sp.get('tab');
  if (qDay) {
    loadDay(qDay);
    switchTab(qTab || 'daily', false);
  } else {
    if (BOOT.hero_day) renderDay(BOOT.hero_day);
    else if (currentDay) loadDay(currentDay);
    if (qTab && document.getElementById('view-' + qTab)) switchTab(qTab, false);
  }
})();

setInterval(async () => {
  try {
    const r = await fetch('/api/status', {cache:'no-store'});
    if (r.ok) updateSystemStatus(await r.json());
  } catch(e) {}
}, 5000);
</script>
</body>
</html>"""


def render_live_html(data_root: Path, db: ArchiveDB, cfg: dict[str, Any], daemon: Any = None) -> str:
    boot = _build_home_boot(data_root, db, cfg, daemon=daemon)
    boot_json = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    phase_json = json.dumps(_PHASE_ZH, ensure_ascii=False).replace("</", "<\\/")
    return _INDEX_HTML.replace("__BOOT_JSON__", boot_json).replace("__PHASE_JSON__", phase_json)


_MONITOR_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时进度与工作监控大屏 · 时光成长馆 V3</title>
<style>
:root{--bg:#0E1319;--card:#161D26;--card2:#1D2632;--bd:#283444;--tx:#F5F1EB;--sub:#9AA6B5;--ok:#4E9F80;--warn:#E09A42;--err:#DF6456;--acc:#E07A5F;--gold:#F0B35E}
*{box-sizing:border-box}
body{margin:0;font-family:"PingFang SC","HarmonyOS Sans SC","Microsoft YaHei UI",system-ui,sans-serif;background:var(--bg);color:var(--tx);line-height:1.5;font-variant-numeric:tabular-nums}
a{color:#F29D82;text-decoration:none}
header{background:linear-gradient(120deg,#161E28,#222C39);padding:16px 28px;border-bottom:1px solid var(--bd);display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
header h1{margin:0;font-size:20px;display:flex;align-items:center;gap:10px}
.pulse{width:10px;height:10px;border-radius:50%;background:var(--ok);display:inline-block;box-shadow:0 0 0 0 rgba(78,159,128,.7);animation:pul 1.8s infinite}
.pulse.paused{background:var(--warn);animation:none}
@keyframes pul{0%{box-shadow:0 0 0 0 rgba(78,159,128,.7)}70%{box-shadow:0 0 0 10px rgba(78,159,128,0)}100%{box-shadow:0 0 0 0 rgba(78,159,128,0)}}
.btn{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:8px 14px;cursor:pointer;font-size:13px;display:inline-block;font-weight:600}
.btn.sec{background:#242E3B;color:#F5F1EB;border:1px solid var(--bd)} .btn.warn{background:#B86E1D} .btn.ok{background:#3A7D63}
main{padding:18px 28px 28px;max-width:1460px;margin:0 auto}
.livebox{background:linear-gradient(135deg,#1A232E,#151C25);border:1px solid #E07A5F44;border-radius:16px;padding:14px 20px;margin-bottom:16px;display:flex;flex-wrap:wrap;gap:14px;justify-content:space-between;align-items:center}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:15px 16px}
.kpi-card{display:flex;flex-direction:column;justify-content:space-between;min-height:126px;border-top:2px solid #2F3E52}
.klabel{font-size:12px;color:var(--sub);margin-bottom:4px;letter-spacing:.2px}
.kval{font-size:22px;font-weight:700;margin-bottom:4px;line-height:1.25}
.ksub{font-size:12px;color:var(--sub);line-height:1.45}
.barwrap{background:#0D1117;border-radius:99px;height:8px;overflow:hidden;border:1px solid var(--bd);margin:7px 0}
.bar{background:linear-gradient(90deg,#4E9F80,#E07A5F);height:100%;transition:width .4s}
.bar.sm{height:6px;margin:4px 0}
.fleet-panel{margin-bottom:16px;background:linear-gradient(180deg,#18202B,#151C25)}
.fleet-legend{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.leg-pill{font-size:11.5px;padding:3px 10px;border-radius:99px;border:1px solid var(--bd);background:var(--card2);color:var(--sub)}
.leg-pill.pro{border-color:rgba(240,179,94,.45);color:#F7CD88;background:rgba(240,179,94,.1)}
.leg-pill.fast{border-color:rgba(78,159,128,.4);color:#8CE3BF;background:rgba(78,159,128,.1)}
.fleet-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px}
.m-card{
  background:var(--card2);border:1px solid var(--bd);border-radius:14px;padding:13px 14px;
  display:flex;flex-direction:column;justify-content:space-between;gap:10px;transition:transform .15s,border-color .15s;
}
.m-card:hover{transform:translateY(-1px);border-color:#3B4D63}
.m-card.pro{border-color:rgba(240,179,94,.45);border-top:3px solid var(--gold);background:linear-gradient(180deg,rgba(240,179,94,.08),rgba(29,38,50,.96))}
.m-card.fast{border-top:3px solid var(--ok)}
.m-card.dim{opacity:.72;border-top-color:var(--warn)}
.m-card.err-dim{opacity:.62;border-top-color:var(--err)}
.m-top{display:flex;align-items:center;justify-content:space-between;gap:6px}
.r-pill{font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px}
.r-pill.pro{background:rgba(240,179,94,.16);color:#FAD392}
.r-pill.fast{background:rgba(78,159,128,.16);color:#89E0BC}
.st-pill{font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;display:inline-flex;align-items:center;gap:4px}
.st-pill.online{background:rgba(78,159,128,.14);color:#82DFBA}
.st-pill.warn{background:rgba(224,154,66,.18);color:#FCD38A}
.st-pill.err{background:rgba(223,100,86,.2);color:#FCA5A5}
.m-name{font-size:15px;font-weight:700;color:#FFF9F2;letter-spacing:.15px;word-break:break-word}
.m-chan{font-size:11.5px;color:var(--sub);margin-top:2px;display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.sub-tag{background:#121820;border:1px solid #2A3748;border-radius:6px;padding:1px 6px;font-size:10.5px;color:#CBD5E1}
.sub-tag.cool{border-color:rgba(224,154,66,.45);color:#FCD38A}
.m-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;background:#121821;border:1px solid #232E3D;border-radius:10px;padding:8px}
.mm-lbl{font-size:10.5px;color:var(--sub)}
.mm-val{font-size:13.5px;font-weight:700;color:var(--tx);margin-top:1px}
.mm-val.err{color:#FCA5A5}
.m-foot{display:flex;flex-direction:column;gap:4px}
.m-share{font-size:11px;color:var(--sub);display:flex;justify-content:space-between}
.m-bar{background:#0E1319;border-radius:99px;height:5px;overflow:hidden}
.m-barfill{height:100%;border-radius:99px;background:linear-gradient(90deg,#4E9F80,#E07A5F);transition:width .4s}
.m-card.pro .m-barfill{background:linear-gradient(90deg,#E8A848,#E07A5F)}
.cols2{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px;margin-bottom:16px}
.panel-title{font-size:15.5px;font-weight:600;margin:0 0 12px;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 8px;border-bottom:1px solid var(--bd);text-align:left;vertical-align:middle}
th{color:var(--sub);font-weight:500;font-size:12px}
.feed{max-height:420px;overflow-y:auto}
.feed thead th{position:sticky;top:0;background:var(--card);z-index:2}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600}
.pill.ok{background:#183D32;color:#82DFBA} .pill.run{background:#3A2721;color:#F8B39C}
.pill.warn{background:#473114;color:#FCD38A} .pill.err{background:#4C1D1D;color:#FCA5A5}
.pill.off{background:#2D3542;color:#CBD5E1}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
@media (max-width:640px){main{padding:12px} .cols2,.fleet-grid{grid-template-columns:1fr}}
</style></head><body>
<header>
  <div>
    <h1><span id="dot" class="pulse"></span>成长档案 V3 · 实时进度与工作监控大屏</h1>
    <div id="hdrSub" style="font-size:12px;color:#B8C0CC;margin-top:4px">每 4 秒同步 NAS 6 路并发引擎 · Carbon + GPT-6.2-Sol 共审高光 · 多模型自领日常筛选 · 源视频保护锁与百度网盘状态</div>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <a class="btn sec" href="/">← 返回时光成长馆</a>
    <button id="pauseBtn" class="btn warn" onclick="togglePause()">暂停后台队列</button>
  </div>
</header>
<main>
  <div class="livebox">
    <div>
      <div style="font-size:12px;color:#F29D82">当前正在执行</div>
      <div id="curTitle" style="font-size:18px;font-weight:700;margin:2px 0">加载中…</div>
      <div id="curDetail" style="font-size:13px;color:#D6DCE5"></div>
    </div>
    <div style="text-align:right">
      <div id="curHeartbeat" style="font-size:13px;color:#82DFBA"></div>
      <div id="curFile" class="mono" style="color:var(--sub);margin-top:4px"></div>
    </div>
  </div>

  <div class="kpis">
    <div class="card kpi-card">
      <div>
        <div class="klabel">总体处理进度</div>
        <div class="kval"><span id="kDone">0</span> / <span id="kTot">0</span> <span id="kPct" style="font-size:15px;color:var(--acc)">(0%)</span></div>
        <div class="barwrap"><div id="kBar" class="bar" style="width:0%"></div></div>
      </div>
      <div id="kSub1" class="ksub">待处理: 0 · 异常隔离: 0</div>
    </div>
    <div class="card kpi-card">
      <div>
        <div class="klabel">实时吞吐速率</div>
        <div class="kval"><span id="kRate">0</span> <span style="font-size:15px;font-weight:500">档/小时</span></div>
      </div>
      <div id="kSub2" class="ksub">吞吐: 0 GB/h · 预计剩余: —</div>
    </div>
    <div class="card kpi-card">
      <div>
        <div class="klabel">空间压缩效率</div>
        <div class="kval"><span id="kOrig">0 MB</span> → <span id="kArch">0 MB</span></div>
      </div>
      <div id="kSub3" class="ksub">T1 精彩片段 + T2 全天 1fps 延时占比 —</div>
    </div>
    <div class="card kpi-card">
      <div>
        <div class="klabel">NAS 内存与硬件加速</div>
        <div class="kval"><span id="kMem">— GiB</span> <span style="font-size:13.5px;font-weight:500;color:var(--ok)">可用 (安全线 4.0 GiB)</span></div>
        <div class="barwrap"><div id="kMemBar" class="bar" style="width:0%"></div></div>
      </div>
      <div id="kMemSub" class="ksub">6 路并发池 + Intel VAAPI 硬件编码就绪</div>
    </div>
    <div class="card kpi-card">
      <div>
        <div class="klabel">源片安全锁 · 云端总账</div>
        <div class="kval" id="kLifeGuard" style="font-size:16.5px">🔒 只读守护中</div>
        <div id="kBaidu" class="ksub">百度首次源片上传：读取中…</div>
      </div>
      <div>
        <div id="kTok" class="ksub" style="color:#F29D82">云端累计 Token: —</div>
        <div id="kSync" class="ksub" style="margin-top:2px;color:#82DFBA"></div>
      </div>
    </div>
  </div>

  <section class="card fleet-panel">
    <div class="panel-title">
      <span>云端大模型工作池 · 专职分工与实时状态</span>
      <div class="fleet-legend">
        <span class="leg-pill pro">🎯 DIRECTOR/PRO 最高思考导演精剪 · Carbon HIGH + GPT-6.2 High</span>
        <span class="leg-pill fast">⚡ FAST 日常筛选 · 全主力抢单 · 无额度自动退出</span>
      </div>
    </div>
    <div id="vlmFleetGrid" class="fleet-grid"></div>
  </section>

  <div class="cols2">
    <div class="card">
      <div class="panel-title"><span>按日期实时队列进度（P0 抢占 + 多日滚动并行）</span><span class="ksub" id="dayCnt"></span></div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>日期</th><th>层级 / 状态</th><th>进度</th><th>最新时间点</th><th>归档产出品</th></tr></thead>
          <tbody id="tbDays"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="panel-title"><span>各机位监控进展与隐私门控统计</span><span class="ksub" id="camCnt"></span></div>
      <div style="overflow-x:auto">
        <table>
          <thead><tr><th>机位</th><th>在线</th><th>进度</th><th>有效活动 / 夜间静音</th><th>原片 → T1</th></tr></thead>
          <tbody id="tbCams"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="cols2">
    <div class="card">
      <div class="panel-title"><span>最近完成的 20 档监控录像（流水日志）</span><span class="ksub">按完结时间倒序</span></div>
      <div class="feed">
        <table>
          <thead><tr><th>完成时间</th><th>机位 · 录像时间</th><th>活动 / 时长</th><th>体积</th><th>状态</th></tr></thead>
          <tbody id="tbFiles"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="panel-title"><span>云端审片实况（Carbon + GPT-6.2 审高光 · 多模型自领日常筛选）</span><span class="ksub">最近 12 次复核</span></div>
      <div class="feed">
        <table>
          <thead><tr><th>时间</th><th>模型 / 类型</th><th>耗时 / 评分</th><th>画面客观描述</th></tr></thead>
          <tbody id="tbVlm"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>
<script>
const BOOT = __BOOT_JSON__;
const PHASE_ZH = __PHASE_JSON__;

function fmtGb(n){ const v=Number(n||0); return v>=1e12?(v/1e12).toFixed(2)+' TB':(v>=1e9?(v/1e9).toFixed(2)+' GB':(v>=1e6?(v/1e6).toFixed(1)+' MB':Math.round(v/1024)+' KB')); }
function fmtTok(n){ const v=Math.round(Number(n||0)); return v>=10000?(v/10000).toFixed(2)+' 万':v.toLocaleString(); }
function fmtDur(s){ const v=Math.round(Number(s||0)); return v>=3600?`${(v/3600).toFixed(1)}小时`:(v>=60?`${Math.round(v/60)}分`:`${v}秒`); }
function fmtClock(ts){ if(!ts)return '—'; const d=new Date(Number(ts)*1000); return d.toTimeString().slice(0,8); }
function el(tag, props, children){
  const n=document.createElement(tag);
  if(props) Object.entries(props).forEach(([k,v])=>{ if(k==='className')n.className=v; else if(k==='textContent')n.textContent=v; else if(k==='href')n.href=v; else if(k==='style')n.setAttribute('style',v); else n[k]=v; });
  (children||[]).forEach(c=>{ if(c!=null) n.appendChild(typeof c==='string'?document.createTextNode(c):c); });
  return n;
}

function renderFleet(m){
  const grid = document.getElementById('vlmFleetGrid');
  if(!grid) return;
  grid.textContent = '';
  const fleet = (m.vlm_fleet && m.vlm_fleet.length)
    ? m.vlm_fleet
    : (m.vlm_stats || []).map(v => ({
        model: v.model,
        role: 'fast',
        role_zh: '云端审片',
        channel_zh: v.provider || '',
        status: Number(v.cooldown_until || 0) > Date.now() / 1000 ? 'rate_limited' : 'online',
        status_zh: Number(v.cooldown_until || 0) > Date.now() / 1000 ? '冷却切备' : (v.calls_ok ? '在线活跃' : '在线就绪'),
        calls_ok: v.calls_ok || 0,
        calls_err: v.calls_err || 0,
        avg_latency_s: v.calls_ok ? Number((v.total_latency_s / v.calls_ok).toFixed(1)) : null,
        total_tokens: v.total_tokens || 0,
        token_share_pct: 0,
        sub_channels: [],
      }));

  if(!fleet.length){
    grid.appendChild(el('div', {className:'ksub', textContent:'等待首轮云端大模型审片结果…'}));
    return;
  }

  fleet.forEach(c => {
    const isPro = c.role === 'pro';
    const st = c.status || 'online';
    const cardCls = `m-card ${isPro ? 'pro' : 'fast'}` + (st === 'quota_out' ? ' err-dim' : (st === 'rate_limited' ? ' dim' : ''));
    const stCls = st === 'online' ? 'online' : (st === 'quota_out' ? 'err' : 'warn');
    const stIcon = st === 'online' ? '● ' : (st === 'quota_out' ? '⛔ ' : '⏸ ');

    const subEls = [el('span', {textContent: c.channel_zh || ''})];
    (c.sub_channels || []).forEach(sc => {
      subEls.push(el('span', {
        className: 'sub-tag' + (sc.cooling ? ' cool' : ''),
        textContent: `${sc.label} ✓${sc.ok || 0}`,
      }));
    });

    const callVal = c.calls_err ? `✓${c.calls_ok} / ✗${c.calls_err}` : `✓${c.calls_ok}`;
    const latVal = c.avg_latency_s != null ? `${c.avg_latency_s}s` : '待命';
    const sharePct = Number(c.token_share_pct || 0);
    const barW = c.total_tokens ? Math.max(4, Math.min(100, sharePct)) : 0;

    grid.appendChild(el('article', {className: cardCls}, [
      el('div', {className:'m-top'}, [
        el('span', {className:`r-pill ${isPro ? 'pro' : 'fast'}`, textContent: c.role_zh || (isPro ? '高光精审 · 独占' : '日常筛选 · 抢单')}),
        el('span', {className:`st-pill ${stCls}`, textContent: stIcon + (c.status_zh || '在线就绪')}),
      ]),
      el('div', {}, [
        el('div', {className:'m-name', textContent: c.model}),
        el('div', {className:'m-chan'}, subEls),
      ]),
      el('div', {className:'m-metrics'}, [
        el('div', {}, [
          el('div', {className:'mm-lbl', textContent:'调用成功'}),
          el('div', {className:'mm-val' + (c.calls_err ? ' err' : ''), textContent: callVal}),
        ]),
        el('div', {}, [
          el('div', {className:'mm-lbl', textContent:'平均耗时'}),
          el('div', {className:'mm-val', textContent: latVal}),
        ]),
        el('div', {}, [
          el('div', {className:'mm-lbl', textContent:'Token 消耗'}),
          el('div', {className:'mm-val', textContent: fmtTok(c.total_tokens)}),
        ]),
      ]),
      el('div', {className:'m-foot'}, [
        el('div', {className:'m-share'}, [
          el('span', {textContent: sharePct > 0 ? `占云端 Token ${sharePct}%` : '待命中 · 随任务自动领单'}),
          el('span', {textContent: c.slots ? `${c.slots} 并发槽` : ''}),
        ]),
        el('div', {className:'m-bar'}, [el('div', {className:'m-barfill', style:`width:${barW}%`})]),
      ]),
    ]));
  });
}

function render(m){
  const paused = !!m.paused;
  const job = (m.state && m.state.active_job) || {};
  const caps = (m.state && m.state.runtime_caps) || {};
  const tot = m.totals || {};
  const tp = m.throughput || {};
  const camNames = m.camera_names || {};
  const healthList = ((m.state && m.state.camera_health) || {}).health || [];
  const healthMap = {};
  healthList.forEach(h => { healthMap[h.id] = h; });

  const dot = document.getElementById('dot');
  dot.className = 'pulse' + (paused ? ' paused' : '');
  const pbtn = document.getElementById('pauseBtn');
  pbtn.textContent = paused ? '恢复无人值守' : '暂停后台队列';
  pbtn.className = 'btn ' + (paused ? 'ok' : 'warn');

  const phaseText = paused ? '已暂停' : (PHASE_ZH[job.phase] || job.phase || '启动扫描中');
  document.getElementById('curTitle').textContent = phaseText + (job.day ? ` · ${job.day}` : '') + (job.step ? ` · 【${job.step}】` : '');
  document.getElementById('curDetail').textContent = job.detail || (job.last_file ? `最近完成: ${job.last_file}` : '等待新录像稳定或扫描中…');
  let hb = '心跳正常';
  if (job.updated_at) {
    const dtMs = Date.now() - Date.parse(job.updated_at);
    const sAgo = Math.max(0, Math.round(dtMs / 1000));
    hb = sAgo < 60 ? `心跳: ${sAgo} 秒前 (${job.updated_at.slice(11,19)})` : `心跳: ${Math.round(sAgo/60)} 分钟前 (${job.updated_at.slice(11,19)})`;
  }
  document.getElementById('curHeartbeat').textContent = hb + ` · VAAPI: ${caps.vaapi_ok ? '开启' : '关闭'}`;
  document.getElementById('curFile').textContent = job.last_file ? `最新文件: ${job.last_file}` : '';

  const done = Number(tot.done_files || 0), total = Number(tot.total_files || 0), queued = Number(tot.queued_files || 0), errs = Number(tot.error_files || 0);
  const pct = total > 0 ? (done * 100 / total).toFixed(1) : '0.0';
  document.getElementById('kDone').textContent = done;
  document.getElementById('kTot').textContent = total;
  document.getElementById('kPct').textContent = `(${pct}%)`;
  document.getElementById('kBar').style.width = Math.min(100, Number(pct)) + '%';
  document.getElementById('kSub1').textContent = `待处理: ${queued} 档 · 异常隔离: ${errs} 档`;

  document.getElementById('kRate').textContent = tp.files_per_hour || 0;
  document.getElementById('kSub2').textContent = `吞吐 ${tp.gb_per_hour || 0} GB/h · 近15分钟 ${tp.done_15m || 0} 档 · 预计清空: ${tp.eta_hours != null ? tp.eta_hours + ' 小时' : '即将完成/待命中'}`;

  const origB = Number(tot.orig_bytes || 0), archB = Number(tot.t1_bytes || 0) + Number(tot.t2_bytes || 0);
  const ratio = origB > 0 ? (archB * 100 / origB).toFixed(1) : '0.0';
  document.getElementById('kOrig').textContent = fmtGb(origB);
  document.getElementById('kArch').textContent = fmtGb(archB);
  document.getElementById('kSub3').textContent = `T1 ${fmtGb(tot.t1_bytes)} + T2 ${fmtGb(tot.t2_bytes)}（占原片 ${ratio}%）`;

  const memKib = Number(job.host_mem_kib || caps.host_mem_kib || 0);
  const memGib = memKib ? (memKib / (1024*1024)) : 0;
  document.getElementById('kMem').textContent = memKib ? memGib.toFixed(2) + ' GiB' : '— GiB';
  document.getElementById('kMemBar').style.width = memGib ? Math.min(100, Math.max(8, memGib * 100 / 16.0)).toFixed(1) + '%' : '0%';
  document.getElementById('kMemSub').textContent = `高于 4.0 GiB 安全线 · VAAPI ${caps.vaapi_ok ? '开启' : '关闭'}`;

  const tt = m.token_totals || {};
  document.getElementById('kTok').textContent = `云端累计 ${fmtTok(tt.total_tokens)} tok（入 ${fmtTok(tt.input_tokens)} / 出 ${fmtTok(tt.output_tokens)}）`;
  renderFleet(m);

  const life = m.lifecycle || {};
  const bd = life.baidu_backup || {};
  document.getElementById('kLifeGuard').textContent = life.can_prune ? '✅ 满足放行条件' : '🔒 只读锁定（绝不删源）';
  document.getElementById('kBaidu').textContent = bd.available
    ? `百度首次原片备份 ${bd.pct || 0}%（${Number(bd.uploaded_files||0).toLocaleString()} / ${Number(bd.scanned_files||0).toLocaleString()}） · 完成后停传原片`
    : '百度备份状态读取中 · 完成后原片不上云（本地留 ' + (life.retention_days || 90) + ' 天）';
  const cs = life.archive_cloud_sync || {};
  document.getElementById('kSync').textContent = cs.note_zh || '';

  const tbDays = document.getElementById('tbDays');
  tbDays.textContent = '';
  const dqs = m.day_queues || [];
  document.getElementById('dayCnt').textContent = `共 ${dqs.length} 天已入库`;
  dqs.forEach(d => {
    const dpct = d.total > 0 ? (d.done * 100 / d.total).toFixed(1) : '0.0';
    const isDone = d.queued === 0;
    const stCls = isDone ? 'ok' : (d.done > 0 ? 'run' : 'off');
    const stTxt = `${d.priority_tier} · ${isDone ? '已完成' : (d.summary_status === 'preview' ? '预览已出/追平中' : '队列中')}`;
    const barBox = el('div', {}, [
      el('div', {textContent: `${d.done}/${d.total} (${dpct}%) · 剩${d.queued}` + (d.errs ? ` · 错${d.errs}` : '')}),
      el('div', {className:'barwrap sm'}, [el('div', {className:'bar', style:`width:${Math.min(100, Number(dpct))}%`})]),
    ]);
    const prod = [];
    if (d.highlight_rel) {
      prod.push(el('a', {href:`/?tab=daily&day=${encodeURIComponent(d.day)}`, textContent:'▶ 看当天精华'}));
    }
    if (d.photos_count) {
      prod.push(el('div', {className:'ksub', textContent:`精选照 ${d.photos_count} · ${fmtGb((d.t1_bytes||0)+(d.t2_bytes||0))}`}));
    } else if (!d.highlight_rel) {
      prod.push(el('span', {className:'ksub', textContent: d.done ? `归档 ${fmtGb((d.t1_bytes||0)+(d.t2_bytes||0))}` : '等待处理'}));
    }
    tbDays.appendChild(el('tr', {}, [
      el('td', {className:'mono', textContent: d.day}),
      el('td', {}, [el('span', {className:`pill ${stCls}`, textContent: stTxt})]),
      el('td', {}, [barBox]),
      el('td', {className:'mono', textContent: d.latest_done_ts ? d.latest_done_ts.slice(5, 19).replace('T', ' ') : '—'}),
      el('td', {}, prod),
    ]));
  });

  const tbCams = document.getElementById('tbCams');
  tbCams.textContent = '';
  const cqs = m.camera_queues || [];
  document.getElementById('camCnt').textContent = `${cqs.length} 路已纳管机位`;
  cqs.forEach(c => {
    const h = healthMap[c.camera] || {};
    const cname = camNames[c.camera] || h.name || c.camera;
    const cpct = c.total > 0 ? (c.done * 100 / c.total).toFixed(1) : '0.0';
    const hcCls = (h.status === 'online' || h.status === 'historical') ? 'ok' : (h.status === 'stale_warning' ? 'warn' : 'off');
    const hcTxt = h.status === 'historical' ? '历史纳管' : (h.status === 'online' ? '在线' : (h.status === 'stale_warning' ? '停更告警' : '队列'));
    const barBox = el('div', {}, [
      el('div', {textContent: `${c.done}/${c.total} (${cpct}%)`}),
      el('div', {className:'barwrap sm'}, [el('div', {className:'bar', style:`width:${Math.min(100, Number(cpct))}%`})]),
    ]);
    tbCams.appendChild(el('tr', {}, [
      el('td', {}, [el('b', {textContent:cname}), el('div', {className:'ksub mono', textContent:c.camera})]),
      el('td', {}, [el('span', {className:`pill ${hcCls}`, textContent:hcTxt})]),
      el('td', {}, [barBox]),
      el('td', {className:'ksub', textContent:`活动 ${fmtDur(c.active_s)} / 静音 ${fmtDur(c.night_sleep_s + c.static_suppressed_s)}`}),
      el('td', {className:'ksub', textContent:`${fmtGb(c.orig_bytes)} → ${fmtGb(c.t1_bytes)}`}),
    ]));
  });

  const tbFiles = document.getElementById('tbFiles');
  tbFiles.textContent = '';
  (m.recent_files || []).forEach(f => {
    const cname = camNames[f.camera] || f.camera;
    const stCls = f.status === 'done' ? 'ok' : 'err';
    const stTxt = f.status === 'done' ? '完成' : (f.error || '异常');
    const fname = (f.source_rel || '').split('/').pop();
    tbFiles.appendChild(el('tr', {}, [
      el('td', {className:'mono', textContent: fmtClock(f.updated_at)}),
      el('td', {}, [el('div', {textContent: cname}), el('div', {className:'ksub mono', textContent:`${(f.wall_ts||'').slice(5,19).replace('T',' ')} · ${fname}`})]),
      el('td', {className:'ksub', textContent:`${f.active_s||0}s / ${f.duration||0}s`}),
      el('td', {className:'ksub', textContent:`${fmtGb(f.size)} → ${fmtGb((f.t1_bytes||0)+(f.t2_bytes||0))}`}),
      el('td', {}, [el('span', {className:`pill ${stCls}`, textContent: stTxt})]),
    ]));
  });

  const tbVlm = document.getElementById('tbVlm');
  tbVlm.textContent = '';
  const kindZh = {ph:'照片筛选', ms:'高光里程碑', kit:'厨房核验', sp:'童声画面', hlc:'高光粗筛', dl_edit:'导演精剪', dir_script:'导演编剧'};
  (m.recent_vlm || []).forEach(v => {
    tbVlm.appendChild(el('tr', {}, [
      el('td', {className:'mono', textContent: fmtClock(v.created_at)}),
      el('td', {}, [el('div', {textContent:v.model}), el('div', {className:'ksub', textContent: kindZh[v.kind] || v.kind})]),
      el('td', {className:'ksub', textContent:`${v.latency_s}s` + (v.score != null ? ` · ${v.score}/5` : '') + (v.child_clear ? ` (${v.child_clear})` : '') + (v.total_tokens ? ` · ${v.total_tokens} tok` : '')}),
      el('td', {textContent: v.caption_zh || v.source_hint || '—'}),
    ]));
  });
}

async function togglePause(){
  const btn = document.getElementById('pauseBtn');
  const r = await fetch('/api/ctl/' + (btn.textContent.includes('恢复') ? 'resume' : 'pause'), {method:'POST'});
  if (r.ok) refresh();
}
async function refresh(){
  try {
    const r = await fetch('/api/monitor', {cache:'no-store'});
    if (r.ok) render(await r.json());
  } catch(e) {}
}
render(BOOT);
setInterval(refresh, 4000);
</script>
</body></html>"""


def render_monitor_html(data_root: Path, db: ArchiveDB, cfg: dict[str, Any]) -> str:
    mon = db.get_live_monitor_data()
    boot = {
        **mon,
        "vlm_fleet": _build_vlm_fleet(cfg, mon),
        "camera_names": _camera_names_map(cfg, mon),
        "paused": (data_root / "state" / "PAUSE").is_file(),
        "lifecycle": _build_lifecycle_snapshot(data_root, db, cfg, mon),
    }
    boot_json = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    phase_json = json.dumps(_PHASE_ZH, ensure_ascii=False).replace("</", "<\\/")
    return _MONITOR_HTML.replace("__BOOT_JSON__", boot_json).replace("__PHASE_JSON__", phase_json)


def create_server(host: str, port: int, data_root: Path, db: ArchiveDB, daemon: Any, cfg: dict[str, Any]) -> ThreadingHTTPServer:
    _fallback_face_svc: FamilyFaceService | None = None

    def _face_svc() -> FamilyFaceService:
        nonlocal _fallback_face_svc
        svc = getattr(daemon, "face_service", None)
        if svc is not None:
            return svc
        if _fallback_face_svc is None:
            _fallback_face_svc = FamilyFaceService(db, data_root, cfg)
        return _fallback_face_svc

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _is_lan(self) -> bool:
            try:
                ip = ipaddress.ip_address(self.client_address[0])
                return ip.is_private or ip.is_loopback
            except ValueError:
                return False

        def _read_json_body(self, max_bytes: int = 65536) -> dict[str, Any]:
            try:
                n = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                n = 0
            if n <= 0:
                return {}
            if n > max_bytes:
                raise ValueError("payload_too_large")
            raw = self.rfile.read(n)
            try:
                obj = json.loads(raw.decode("utf-8"))
                return obj if isinstance(obj, dict) else {}
            except ValueError as exc:
                raise ValueError("invalid_json") from exc

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if not self._is_lan():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "lan_only"})
                return
            origin = self.headers.get("Origin")
            host_header = self.headers.get("Host", "").lower()
            if origin and origin.lower() not in (f"http://{host_header}", f"https://{host_header}"):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "cross_origin"})
                return
            if path == "/api/ctl/pause":
                daemon.set_paused(True)
                self._send_json(200, {"paused": True})
                return
            if path == "/api/ctl/resume":
                daemon.set_paused(False)
                self._send_json(200, {"paused": False})
                return
            if path == "/api/ctl/curate_day":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                day_str = str(body.get("day") or "").strip()
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", day_str):
                    self._send_json(400, {"error": "invalid_day"})
                    return
                try:
                    dt.date.fromisoformat(day_str)
                except ValueError:
                    self._send_json(400, {"error": "invalid_day"})
                    return
                req_fn = getattr(daemon, "request_day_curation", None)
                if not callable(req_fn):
                    self._send_json(503, {"error": "curation_unavailable"})
                    return
                res = req_fn(day_str, body.get("tier"))
                self._send_json(200, res)
                return
            if path == "/api/ctl/film/compile":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                film_id = str(body.get("film_id") or "").strip()
                if not film_id:
                    self._send_json(400, {"error": "missing_film_id"})
                    return
                req_film_fn = getattr(daemon, "request_film_compile", None)
                if not callable(req_film_fn):
                    self._send_json(503, {"error": "film_compile_unavailable"})
                    return
                try:
                    res = req_film_fn(film_id, force=bool(body.get("force", True)))
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(200, res)
                return
            if path == "/api/ctl/film/review":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                film_id = str(body.get("film_id") or "").strip()
                action = str(body.get("action") or "").strip()
                if not film_id or not action:
                    self._send_json(400, {"error": "missing_film_id_or_action"})
                    return
                try:
                    updated = db.review_film_production(
                        film_id,
                        action,
                        note=str(body.get("note") or ""),
                        exclude_shots=body.get("exclude_shots") if isinstance(body.get("exclude_shots"), list) else None,
                    )
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                if updated is None:
                    self._send_json(404, {"error": "film_not_found"})
                    return
                fc.update_published_film_manifest_review(data_root, db, updated)
                if action in ("retry", "regenerate"):
                    wake_ev = getattr(daemon, "_wake", None)
                    if wake_ev is not None and hasattr(wake_ev, "set"):
                        wake_ev.set()
                enriched = _enrich_films(data_root, [updated])[0]
                self._send_json(200, {"film": enriched})
                return
            if path == "/api/ctl/vlm_review":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                review_id = str(body.get("review_id") or "").strip()
                raw_action = str(body.get("action") or body.get("status") or "").strip()
                status_map = {
                    "approve": "approved",
                    "approved": "approved",
                    "reject": "rejected",
                    "rejected": "rejected",
                    "pending_review": "pending_review",
                }
                mapped = status_map.get(raw_action)
                if not review_id or not mapped:
                    self._send_json(400, {"error": "invalid_review_request"})
                    return
                updated = db.decide_vlm_review(
                    review_id,
                    mapped,
                    category=(str(body["category"]) if body.get("category") else None),
                    note=str(body.get("note") or ""),
                )
                if updated is None:
                    self._send_json(404, {"error": "review_not_found"})
                    return
                self._send_json(200, {"review": updated})
                return
            if path == "/api/ctl/reload_config":
                reloader = getattr(daemon, "reload_config_if_changed", None)
                reloaded = bool(reloader(force=True)) if callable(reloader) else False
                self._send_json(200, {
                    "reloaded": reloaded,
                    "camera_names": _camera_names_map(cfg, db.get_dashboard_snapshot()),
                    "workers": int((cfg.get("scheduler") or {}).get("workers", 4)),
                })
                return
            if path == "/api/family/member":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                try:
                    updated = db.upsert_family_member(body)
                    _face_svc().invalidate_cache()
                    _face_svc().get_reference_sheet(day=None)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(200, {"member": updated})
                return
            if path == "/api/family/sample":
                try:
                    body = self._read_json_body(max_bytes=12 * 1024 * 1024)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                mid = str(body.get("member_id") or "").strip()
                raw_b64 = str(body.get("image_b64") or "").strip()
                if "," in raw_b64 and raw_b64.startswith("data:"):
                    raw_b64 = raw_b64.split(",", 1)[1]
                if not mid or not raw_b64:
                    self._send_json(400, {"error": "缺少 member_id 或图片数据"})
                    return
                try:
                    img_bytes = base64.b64decode(raw_b64, validate=True)
                except Exception:
                    self._send_json(400, {"error": "图片 base64 格式无效"})
                    return
                try:
                    res = _face_svc().enroll_image_bytes(
                        mid,
                        img_bytes,
                        source=str(body.get("source") or "upload"),
                        source_note=str(body.get("source_note") or ""),
                        allow_visual_only=bool(body.get("allow_visual_only", False)),
                        face_index=int(body.get("face_index", 0)),
                    )
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(200, {**res, "members": db.list_family_members()})
                return
            if path == "/api/family/sample/delete":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                sid = str(body.get("sample_id") or "").strip()
                if not sid:
                    self._send_json(400, {"error": "missing_sample_id"})
                    return
                ok = _face_svc().delete_sample(sid)
                if not ok:
                    self._send_json(404, {"error": "sample_not_found"})
                    return
                self._send_json(200, {"deleted": True, "sample_id": sid})
                return
            if path == "/api/family/discover":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                cands = _face_svc().discover_candidates_from_archive(
                    max_days=max(1, min(60, int(body.get("max_days", 14)))),
                    max_photos=max(1, min(160, int(body.get("max_photos", 60)))),
                )
                self._send_json(200, {"candidates": cands})
                return
            if path == "/api/family/candidate/assign":
                try:
                    body = self._read_json_body()
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                cid = str(body.get("cand_id") or "").strip()
                mid = str(body.get("member_id") or "").strip()
                if not cid or not mid:
                    self._send_json(400, {"error": "missing_cand_or_member"})
                    return
                try:
                    res = _face_svc().assign_discovered_candidate(cid, mid)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(200, res)
                return
            self._send_json(404, {"error": "not_found"})

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = parse_qs(parsed.query)

            if path == "/health.json":
                self._send_json(200, {"ok": True, "service": "cctv-growth-v3"})
                return
            if not self._is_lan():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "lan_only"})
                return
            reloader = getattr(daemon, "reload_config_if_changed", None)
            if path in ("/api/status", "/api/monitor", "/", "/index.html", "/monitor", "/monitor.html") and callable(reloader):
                reloader()
            if path == "/api/status":
                snap = db.get_dashboard_snapshot()
                deliv_s = _load_deliverables(data_root)
                self._send_json(200, {
                    **snap,
                    "vlm_fleet": _build_vlm_fleet(cfg, snap),
                    "paused": daemon.is_paused(),
                    "curator_status": _extract_curator_status(daemon, snap),
                    "pending_reviews": db.list_vlm_reviews(status="pending_review", limit=100),
                    "films": _enrich_films(data_root, db.list_film_productions()),
                    "smooth_films": list((deliv_s.get("smooth_films") or {}).values()),
                    "lifecycle": _build_lifecycle_snapshot(data_root, db, cfg, snap),
                })
                return
            if path == "/api/films":
                snap = db.get_dashboard_snapshot()
                deliv_f = _load_deliverables(data_root)
                self._send_json(200, {
                    "films": _enrich_films(data_root, db.list_film_productions()),
                    "smooth_films": list((deliv_f.get("smooth_films") or {}).values()),
                    "schedule": _build_lifecycle_snapshot(data_root, db, cfg, snap)["monthly_schedule"],
                    "curator_status": _extract_curator_status(daemon, snap),
                    "paused": bool(daemon.is_paused()) if callable(getattr(daemon, "is_paused", None)) else False,
                })
                return
            if path == "/api/reviews":
                st = (qs.get("status") or ["pending_review"])[0]
                day_f = (qs.get("day") or [""])[0]
                self._send_json(200, {
                    "items": db.list_vlm_reviews(status=st, day=day_f, limit=200),
                })
                return
            if path == "/api/monitor":
                mon = db.get_live_monitor_data()
                self._send_json(200, {
                    **mon,
                    "vlm_fleet": _build_vlm_fleet(cfg, mon),
                    "camera_names": _camera_names_map(cfg, mon),
                    "paused": daemon.is_paused(),
                    "lifecycle": _build_lifecycle_snapshot(data_root, db, cfg, mon),
                    "server_ts": time.time(),
                })
                return
            if path == "/api/days":
                self._send_json(200, db.get_dashboard_snapshot()["days"])
                return
            if path == "/api/family/members":
                fsvc = _face_svc()
                sheet_rel = "family_faces/reference_sheet_latest.jpg"
                has_sheet = fsvc.get_reference_sheet(day=None) is not None and (data_root / sheet_rel).is_file()
                self._send_json(200, {
                    "engine": fsvc.engine_status(),
                    "members": db.list_family_members(include_samples=True),
                    "reference_sheet_rel": sheet_rel if has_sheet else None,
                    "sheet_ts": int((data_root / sheet_rel).stat().st_mtime) if has_sheet else 0,
                    "candidates": fsvc.list_discovered_candidates(),
                })
                return
            if path == "/api/milestones":
                cat = (qs.get("category") or [""])[0]
                cam = (qs.get("camera") or [""])[0]
                q = (qs.get("q") or [""])[0]
                self._send_json(200, _collect_milestones(data_root, db, cfg, category=cat, camera=cam, q=q))
                return
            if path.startswith("/api/day/"):
                day_str = path[len("/api/day/") :].strip("/")
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", day_str):
                    self._send_json(400, {"error": "invalid_day"})
                    return
                try:
                    dt.date.fromisoformat(day_str)
                except ValueError:
                    self._send_json(400, {"error": "invalid_day"})
                    return
                self._send_json(200, _build_day_payload(day_str, data_root, db, cfg))
                return
            if path in ("/", "/index.html", "/monitor", "/monitor.html"):
                html_text = render_monitor_html(data_root, db, cfg) if path in ("/monitor", "/monitor.html") else render_live_html(data_root, db, cfg, daemon=daemon)
                body = html_text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
                return

            # Static media serving strictly within /data while blocking /data/state & secrets
            rel = path.lstrip("/")
            if not rel or rel.startswith("state") or ".." in Path(rel).parts:
                self._send_json(403, {"error": "forbidden"})
                return
            try:
                target = (data_root / rel).resolve()
                root_res = data_root.resolve()
                if not target.is_relative_to(root_res) or target.is_relative_to(root_res / "state") or not target.is_file():
                    self._send_json(404, {"error": "not_found"})
                    return
            except Exception:
                self._send_json(404, {"error": "not_found"})
                return

            size = target.stat().st_size
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range")
            if rng:
                m = RANGE_RE.match(rng.strip())
                if not m:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                s_raw, e_raw = m.group(1), m.group(2)
                start = int(s_raw) if s_raw else max(0, size - int(e_raw or 0))
                end = int(e_raw) if (s_raw and e_raw) else (size - 1 if s_raw else size - 1)
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                if self.command != "HEAD":
                    try:
                        with target.open("rb") as fh:
                            fh.seek(start)
                            rem = length
                            while rem > 0:
                                blk = fh.read(min(262144, rem))
                                if not blk:
                                    break
                                self.wfile.write(blk)
                                rem -= len(blk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if self.command != "HEAD":
                try:
                    with target.open("rb") as fh:
                        while True:
                            blk = fh.read(262144)
                            if not blk:
                                break
                            self.wfile.write(blk)
                except (BrokenPipeError, ConnectionResetError):
                    return

    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), Handler)
