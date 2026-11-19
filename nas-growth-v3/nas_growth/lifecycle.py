"""Source retention safety gates, Baidu Cloud status reader, archive cloud sync, and baby age/monthly scheduler."""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import os
import re
import stat
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests

from .cloud_vlm import redact_secrets
from .db import ArchiveDB, is_confirmed_milestone_record


def add_months(d: dt.date, months: int) -> dt.date:
    """Shift a calendar date by N months, clamping day to month length."""
    total_m = (d.year * 12 + (d.month - 1)) + int(months)
    year = total_m // 12
    month = (total_m % 12) + 1
    max_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, min(d.day, max_day))


def compute_age_info(birthday_str: str, target_date: dt.date | None = None) -> dict[str, Any]:
    """Compute exact calendar months, remainder days, and cumulative life day from birthday."""
    bday = dt.date.fromisoformat(birthday_str)
    d = target_date or dt.date.today()
    if d < bday:
        return {
            "birthday": birthday_str,
            "date": d.isoformat(),
            "months": 0,
            "days": 0,
            "day_n": 0,
            "label_zh": "即将降生",
            "short_zh": "出生前",
        }
    total_days = (d - bday).days
    day_n = total_days + 1
    months = (d.year - bday.year) * 12 + (d.month - bday.month)
    if d.day < bday.day:
        months -= 1
        prev_m = d.month - 1 or 12
        prev_y = d.year if d.month > 1 else d.year - 1
        days_in_prev = calendar.monthrange(prev_y, prev_m)[1]
        rem_days = (days_in_prev - min(bday.day, days_in_prev)) + d.day
    else:
        rem_days = d.day - bday.day

    if months == 0:
        label_zh = f"新生第 {day_n} 天"
        short_zh = f"{total_days} 天"
    elif rem_days == 0:
        label_zh = f"满 {months} 个月"
        short_zh = f"{months} 月整"
    else:
        label_zh = f"{months} 个月 {rem_days} 天"
        short_zh = f"{months} 个月 {rem_days} 天"
    return {
        "birthday": birthday_str,
        "date": d.isoformat(),
        "months": months,
        "days": rem_days,
        "day_n": day_n,
        "label_zh": label_zh,
        "short_zh": short_zh,
    }


CANONICAL_CHRONICLE_ANGLES: dict[str, dict[str, Any]] = {
    "bedroom_ceiling": {"name": "宝宝房顶部", "roles": ("bedroom_ceiling",), "zoom_x": 1.4},
    "bedside_crib": {"name": "宝宝床头", "roles": ("bedside_crib",), "zoom_x": 1.25},
    "living_room": {"name": "客厅", "roles": ("living_room",), "zoom_x": 1.2},
    "kitchen_4k": {"name": "厨房（旧小米 + G350）", "roles": ("kitchen_4k", "kitchen"), "zoom_x": 1.0},
}


def build_seasonal_film_spec(
    bday: dt.date,
    season_idx: int,
    compile_day: int = 18,
) -> dict[str, Any]:
    """Build canonical film production metadata for a 3-month developmental season (Spring/Summer/Autumn/Winter)."""
    s_idx = max(1, int(season_idx))
    p_start = add_months(bday, (s_idx - 1) * 3)
    p_end = add_months(bday, s_idx * 3)
    mid_dt = p_start + dt.timedelta(days=max(1, (p_end - p_start).days // 2))
    m_mid = mid_dt.month
    if m_mid in (3, 4, 5):
        slug, label = "spring", "春日萌芽"
    elif m_mid in (6, 7, 8):
        slug, label = "summer", "盛夏探索"
    elif m_mid in (9, 10, 11):
        slug, label = "autumn", "金秋初语"
    else:
        slug, label = "winter", "冬日初生"
    max_d = calendar.monthrange(p_end.year, p_end.month)[1]
    comp_dt = dt.date(p_end.year, p_end.month, min(compile_day, max_d))
    return {
        "film_id": f"season_{s_idx:02d}_{slug}",
        "kind": "seasonal",
        "age_months": s_idx * 3,
        "season_slug": slug,
        "season_label_zh": label,
        "title_zh": f"第 {s_idx} 季 · {label}（{(s_idx - 1) * 3}–{s_idx * 3}月龄）",
        "period_start": p_start.isoformat(),
        "period_end": p_end.isoformat(),
        "compile_date": comp_dt.isoformat(),
    }


def build_camera_chronicle_spec(
    bday: dt.date,
    angle: str,
    now_date: dt.date,
) -> dict[str, Any]:
    """Build director-curated chronicle film spec for one canonical fixed viewpoint."""
    ang = str(angle or "").strip()
    if ang not in CANONICAL_CHRONICLE_ANGLES:
        raise ValueError(f"Unsupported angle: {angle}")
    meta = CANONICAL_CHRONICLE_ANGLES[ang]
    age_m = max(1, int(compute_age_info(bday.isoformat(), now_date)["months"]))
    return {
        "film_id": f"angle_{ang}",
        "kind": "camera_chronicle",
        "age_months": age_m,
        "angle": ang,
        "title_zh": f"{meta['name']} · 成长流年影集",
        "period_start": bday.isoformat(),
        "period_end": now_date.isoformat(),
        "compile_date": now_date.isoformat(),
    }


def build_growth_chronicle_spec(
    bday: dt.date,
    now_date: dt.date,
) -> dict[str, Any]:
    """Build canonical spec for the full-span growth chronicle film covering birth through current age."""
    age_m = max(1, int(compute_age_info(bday.isoformat(), now_date)["months"]))
    return {
        "film_id": "growth_chronicle",
        "kind": "growth_chronicle",
        "age_months": age_m,
        "title_zh": f"宝宝成长全纪实（出生～{age_m}月龄流年大片）",
        "period_start": bday.isoformat(),
        "period_end": now_date.isoformat(),
        "compile_date": now_date.isoformat(),
    }


def build_film_spec(
    bday: dt.date,
    age_months: int,
    kind: str,
    compile_day: int = 18,
) -> dict[str, Any]:
    """Build canonical film production metadata for a monthly (1..N), seasonal (3m), or annual (12*y) window."""
    if kind == "seasonal":
        return build_seasonal_film_spec(bday, max(1, (int(age_months) + 2) // 3), compile_day=compile_day)
    if kind == "annual":
        years = max(1, int(age_months) // 12)
        eff_months = years * 12
        p_start = add_months(bday, 12 * (years - 1))
        p_end = add_months(bday, eff_months)
        film_id = f"annual_{years:02d}y"
        title_zh = f"宝宝 {years} 周岁成长纪念大片"
    else:
        eff_months = max(1, int(age_months))
        p_start = add_months(bday, eff_months - 1)
        p_end = add_months(bday, eff_months)
        film_id = f"month_{eff_months:02d}m"
        title_zh = f"第 {eff_months} 个月成长月龄片"
    max_d = calendar.monthrange(p_end.year, p_end.month)[1]
    comp_dt = dt.date(p_end.year, p_end.month, min(compile_day, max_d))
    return {
        "film_id": film_id,
        "kind": "annual" if kind == "annual" else "monthly",
        "age_months": eff_months,
        "title_zh": title_zh,
        "period_start": p_start.isoformat(),
        "period_end": p_end.isoformat(),
        "compile_date": comp_dt.isoformat(),
    }


def resolve_film_spec_by_id(
    birthday_str: str,
    film_id: str,
    now_date: dt.date,
    compile_day: int = 18,
) -> dict[str, Any]:
    """Resolve canonical production spec from any monthly, annual, seasonal, or camera chronicle ID."""
    bday = dt.date.fromisoformat(birthday_str)
    fid = str(film_id or "").strip()
    m_mon = re.match(r"^month_(\d+)m$", fid)
    if m_mon:
        return build_film_spec(bday, int(m_mon.group(1)), "monthly", compile_day=compile_day)
    m_ann = re.match(r"^annual_(\d+)y$", fid)
    if m_ann:
        return build_film_spec(bday, int(m_ann.group(1)) * 12, "annual", compile_day=compile_day)
    m_sea = re.match(r"^season_(\d+)_(spring|summer|autumn|winter)$", fid)
    if m_sea:
        spec = build_seasonal_film_spec(bday, int(m_sea.group(1)), compile_day=compile_day)
        if spec["film_id"] != fid:
            raise ValueError(f"Seasonal film_id mismatch: {fid} != {spec['film_id']}")
        return spec
    if fid == "growth_chronicle":
        return build_growth_chronicle_spec(bday, now_date)
    if fid.startswith("angle_"):
        return build_camera_chronicle_spec(bday, fid[len("angle_"):], now_date)
    raise ValueError(f"Unknown film_id: {film_id}")


def _covered_age_months_from_cards(cards: list[Any], bday: dt.date, compile_day: int) -> set[int]:
    """Return full age-months (>= 1) actually covered by monthly_cards, ignoring legacy mid-month V2 seeds."""
    legacy_days = [
        str(c.get("day") or "")
        for c in cards
        if isinstance(c, dict)
        and not str(c.get("film_id") or "").startswith("month_")
        and re.match(r"^\d{4}-\d{2}-\d{2}$", str(c.get("day") or ""))
    ]
    max_legacy_day = max(legacy_days) if legacy_days else ""
    covered: set[int] = set()
    generic_count = 0
    for c in cards:
        if not isinstance(c, dict):
            continue
        fid = str(c.get("film_id") or "")
        if fid.startswith("month_") and fid[6:-1].isdigit() and fid.endswith("m"):
            if c.get("review_status") != "rejected":
                covered.add(int(fid[6:-1]))
            continue
        m_str = str(c.get("month") or "")
        if re.match(r"^\d{4}-\d{2}$", m_str):
            my, mm = int(m_str[:4]), int(m_str[5:7])
            age_m = (my - bday.year) * 12 + (mm - bday.month)
            if age_m <= 0:
                continue
            sp_m = build_film_spec(bday, age_m, "monthly", compile_day=compile_day)
            # Only treat a legacy V2 card as covering month `age_m` if V2 reached its cutoff `period_end`
            if not max_legacy_day or max_legacy_day >= sp_m["period_end"]:
                covered.add(age_m)
            continue
        generic_count += 1
        covered.add(generic_count)
    return covered


def compute_monthly_schedule(
    birthday_str: str,
    now_date: dt.date,
    data_root: Path,
    compile_day: int = 18,
    db: ArchiveDB | None = None,
) -> dict[str, Any]:
    """Compute next monthly/birthday compilation milestone on the 18th after collecting all 17th recordings."""
    bday = dt.date.fromisoformat(birthday_str)
    deliv_p = data_root / "v2" / "deliverables.json"
    deliv: dict[str, Any] = {}
    if deliv_p.is_file():
        try:
            deliv = json.loads(deliv_p.read_text(encoding="utf-8"))
        except Exception:
            deliv = {}

    existing_films = {fp["film_id"]: fp for fp in db.list_film_productions()} if db is not None else {}
    cards = deliv.get("monthly_cards") or []
    cards_cnt = len(cards)
    covered_months = _covered_age_months_from_cards(cards, bday, compile_day)
    for fp in existing_films.values():
        if fp.get("kind") == "monthly" and fp.get("status") == "published":
            covered_months.add(int(fp.get("age_months") or 0))
    cur_months = (now_date.year - bday.year) * 12 + (now_date.month - bday.month)
    if cur_months > 0:
        cur_max_d = calendar.monthrange(now_date.year, now_date.month)[1]
        cur_comp = dt.date(now_date.year, now_date.month, min(compile_day, cur_max_d))
        cur_mid = f"month_{cur_months:02d}m"
        cur_aid = f"annual_{cur_months // 12:02d}y" if cur_months % 12 == 0 else ""
        needs_cur_film = (
            db is not None
            and cur_comp == now_date
            and (
                (cur_mid in existing_films and existing_films[cur_mid].get("status") != "published")
                or (bool(cur_aid) and cur_aid in existing_films and existing_films[cur_aid].get("status") != "published")
            )
        )
    else:
        needs_cur_film = False

    cur_month_covered = (cur_months in covered_months) and not needs_cur_film
    due_today = now_date.day == compile_day and cur_months > 0 and not cur_month_covered

    if now_date.day >= compile_day and not due_today:
        ny = now_date.year + (1 if now_date.month == 12 else 0)
        nm = 1 if now_date.month == 12 else now_date.month + 1
    else:
        ny, nm = now_date.year, now_date.month
    max_day = calendar.monthrange(ny, nm)[1]
    next_dt = dt.date(ny, nm, min(compile_day, max_day))
    cutoff_dt = dt.date(ny, nm, min(bday.day, max_day))
    cutoff_iso = cutoff_dt.isoformat()

    next_months = (next_dt.year - bday.year) * 12 + (next_dt.month - bday.month)
    is_birthday = next_months > 0 and next_months % 12 == 0
    title = f"宝宝 {next_months // 12} 周岁纪念大片" if is_birthday else f"满 {next_months} 个月成长流年更新"

    day_st = db.count_day_stats(cutoff_iso) if db is not None else {"total": 0, "done": 0, "queued": 0}
    cutoff_ready = now_date > cutoff_dt and (db is None or day_st["total"] > 0 and day_st["done"] == day_st["total"])
    if due_today and not cutoff_ready:
        remaining = day_st["total"] - day_st["done"]
        detail = f"（剩余 {remaining} 档）" if remaining else "（尚无已登记录像）"
        note_zh = f"待 {cutoff_dt.month}月{cutoff_dt.day}日 全天录像收齐并归档成功{detail}后启动本轮流年编纂"
    elif due_today:
        note_zh = f"已收齐 {cutoff_dt.month}月{cutoff_dt.day}日 全天录像，可于今日 ({next_dt.isoformat()}) 启动本轮流年编纂"
    else:
        note_zh = f"收齐 {cutoff_dt.month}月{cutoff_dt.day}日 全天录像并完成 P0 归档后，于 {next_dt.isoformat()} 启动"

    def _period_stats(p_start: str, p_end: str, comp_date: str) -> tuple[bool, int, bool]:
        if db is None:
            return False, 0, False
        p_counts = db.get_historical_queue_counts(p_start, p_end)
        c_counts = db.count_day_stats(p_end)
        has_ms = bool(db.query_milestones(min_day=p_start, max_day=p_end, limit=1))
        done_cnt = int(p_counts.get("done") or 0)
        has_ft = (done_cnt > 0) or has_ms
        comp_dt = dt.date.fromisoformat(comp_date)
        cutoff_ok = int(c_counts.get("queued") or 0) == 0 and (
            int(c_counts.get("done") or 0) > 0 or has_ms or now_date >= comp_dt + dt.timedelta(days=2)
        )
        return has_ft, done_cnt, (has_ft and cutoff_ok)

    due_productions: list[dict[str, Any]] = []
    if cur_months > 0:
        for m in range(1, cur_months + 1):
            specs = [build_film_spec(bday, m, "monthly", compile_day=compile_day)]
            if m % 3 == 0:
                specs.append(build_seasonal_film_spec(bday, m // 3, compile_day=compile_day))
            if m >= 12 and m % 12 == 0:
                specs.append(build_film_spec(bday, m, "annual", compile_day=compile_day))
            for sp in specs:
                if dt.date.fromisoformat(sp["compile_date"]) > now_date:
                    continue
                fp_existing = existing_films.get(sp["film_id"])
                if fp_existing and (fp_existing.get("status") == "published" or fp_existing.get("review_status") == "rejected"):
                    continue
                if sp["kind"] == "monthly" and m in covered_months and fp_existing is None:
                    continue
                has_footage, _done_cnt, ready_for_build = _period_stats(
                    sp["period_start"], sp["period_end"], sp["compile_date"]
                )
                if sp["kind"] == "seasonal" and not has_footage:
                    continue
                due_productions.append({
                    **sp,
                    "ready": ready_for_build,
                    "has_footage": has_footage,
                    "existing_status": fp_existing.get("status") if fp_existing else "scheduled",
                })

    max_seasons = max(4, (max(1, cur_months) + 2) // 3)
    card_months = {
        str(c.get("month") or "")
        for c in cards
        if isinstance(c, dict) and re.match(r"^\d{4}-\d{2}$", str(c.get("month") or ""))
    }
    seasonal_catalog: list[dict[str, Any]] = []
    for s_i in range(1, max_seasons + 1):
        ssp = build_seasonal_film_spec(bday, s_i, compile_day=compile_day)
        fp_s = existing_films.get(ssp["film_id"])
        has_ft, done_cnt, ready_s = _period_stats(ssp["period_start"], ssp["period_end"], ssp["compile_date"])
        if not has_ft and dt.date.fromisoformat(ssp["period_start"]) <= now_date:
            s_ym0, s_ym1 = ssp["period_start"][:7], ssp["period_end"][:7]
            if any(s_ym0 <= cm <= s_ym1 for cm in card_months):
                has_ft = True
        seasonal_catalog.append({
            **ssp,
            "has_footage": has_ft,
            "done_files": done_cnt,
            "ready": ready_s,
            "existing_status": fp_s.get("status") if fp_s else ("ready_manual" if has_ft else "scheduled"),
            "review_status": fp_s.get("review_status") if fp_s else "",
            "duration_s": fp_s.get("duration_s") if fp_s else 0.0,
            "shots_count": fp_s.get("shots_count") if fp_s else 0,
        })

    total_done_cnt = int(db.get_historical_queue_counts(bday.isoformat(), now_date.isoformat()).get("done") or 0) if db is not None else 0
    smooth_map = deliv.get("smooth_films") or {}
    camera_chronicles: list[dict[str, Any]] = []
    for ang, ang_meta in CANONICAL_CHRONICLE_ANGLES.items():
        asp = build_camera_chronicle_spec(bday, ang, now_date)
        fp_a = existing_films.get(asp["film_id"])
        legacy_sf = smooth_map.get(ang) or {}
        camera_chronicles.append({
            **asp,
            "name": ang_meta["name"],
            "zoom_x": ang_meta["zoom_x"],
            "has_footage": (total_done_cnt > 0) or bool(legacy_sf.get("file")),
            "existing_status": fp_a.get("status") if fp_a else ("legacy_v2" if legacy_sf.get("file") else "scheduled"),
            "review_status": fp_a.get("review_status") if fp_a else "",
            "director_v3": bool((fp_a and fp_a.get("status") == "published") or legacy_sf.get("director_v3")),
        })

    gc_spec = build_growth_chronicle_spec(bday, now_date)
    fp_gc = existing_films.get("growth_chronicle")
    growth_chronicle_summary = {
        **gc_spec,
        "has_footage": (total_done_cnt > 0) or bool(card_months) or bool(deliv.get("monthly_film")),
        "existing_status": fp_gc.get("status") if fp_gc else ("ready_manual" if (total_done_cnt > 0 or card_months) else "scheduled"),
        "review_status": fp_gc.get("review_status") if fp_gc else "",
        "duration_s": fp_gc.get("duration_s") if fp_gc else 0.0,
        "shots_count": fp_gc.get("shots_count") if fp_gc else 0,
    }

    return {
        "compile_day": compile_day,
        "next_compile_date": next_dt.isoformat(),
        "milestone_day_cutoff": cutoff_iso,
        "cutoff_day_queued": int(day_st.get("queued", 0)),
        "cutoff_day_done": int(day_st.get("done", 0)),
        "cutoff_day_ready": cutoff_ready,
        "due_today": due_today,
        "next_months": next_months,
        "next_title_zh": title,
        "note_zh": note_zh,
        "monthly_cards_count": cards_cnt,
        "smooth_films_count": len(smooth_map),
        "has_monthly_film": bool(deliv.get("monthly_film")) or any(
            fp.get("status") == "published" for fp in existing_films.values()
        ),
        "due_productions": due_productions,
        "seasonal_catalog": seasonal_catalog,
        "camera_chronicles": camera_chronicles,
        "growth_chronicle": growth_chronicle_summary,
        "published_films_count": sum(1 for fp in existing_films.values() if fp.get("status") == "published"),
        "pending_review_films_count": sum(
            1 for fp in existing_films.values()
            if fp.get("status") == "published" and fp.get("review_status") == "pending_review"
        ),
    }


_HTTP_CACHE_LOCK = threading.Lock()
_HTTP_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def read_baidu_backup_status(status_source: str | Path, *, use_cache: bool = True) -> dict[str, Any]:
    """Read cctv-baidu-backup status.json from a local path or HTTP endpoint without exposing secrets."""
    src_str = str(status_source or "").strip()
    is_http = src_str.startswith(("http://", "https://"))
    if is_http and use_cache:
        with _HTTP_CACHE_LOCK:
            cached = _HTTP_CACHE.get(src_str)
            if cached and (time.monotonic() - cached[0]) < 15.0:
                return dict(cached[1])
    base = {
        "available": False,
        "phase": "unknown",
        "initial_complete": None,
        "initial_done": False,
        "daily_enabled": False,
        "uploaded_files": 0,
        "uploaded_bytes": 0,
        "pending_files": 0,
        "pending_bytes": 0,
        "missing_files": 0,
        "failed_files": 0,
        "scanned_files": 0,
        "scanned_bytes": 0,
        "pct": 0.0,
        "current_file": None,
        "heartbeat": None,
    }
    raw: dict[str, Any] | None = None
    if is_http:
        try:
            sess = requests.Session()
            sess.trust_env = False
            resp = sess.get(src_str, timeout=2.5)
            resp.raise_for_status()
            parsed = resp.json()
            if isinstance(parsed, dict):
                raw = parsed
        except Exception:
            return base
    else:
        p = Path(src_str)
        if not p.is_file():
            return base
        try:
            parsed = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                raw = parsed
        except Exception:
            return base
    if not raw:
        return base

    totals = raw.get("totals") or {}
    up = totals.get("uploaded") or {}
    pend = totals.get("pending") or {}
    miss = totals.get("missing") or {}
    fail = totals.get("failed") or {}

    up_files = int(raw.get("uploaded_files_total") or up.get("files") or 0)
    up_bytes = int(raw.get("uploaded_bytes_total") or up.get("bytes") or 0)
    sc_files = int(raw.get("scanned_files") or (up_files + int(pend.get("files") or 0)))
    sc_bytes = int(raw.get("scanned_bytes") or (up_bytes + int(pend.get("bytes") or 0)))
    pct = round((up_bytes * 100.0 / sc_bytes), 1) if sc_bytes > 0 else 0.0
    init_comp = raw.get("initial_complete")

    out = {
        "available": True,
        "phase": str(raw.get("phase") or "unknown"),
        "initial_complete": init_comp,
        "initial_done": bool(init_comp) or str(raw.get("phase") or "") in ("completed", "completed_with_missing"),
        "daily_enabled": bool(raw.get("daily_enabled", False)),
        "uploaded_files": up_files,
        "uploaded_bytes": up_bytes,
        "pending_files": int(pend.get("files") or 0),
        "pending_bytes": int(pend.get("bytes") or 0),
        "missing_files": int(miss.get("files") or raw.get("missing_files") or 0),
        "failed_files": int(fail.get("files") or 0),
        "scanned_files": sc_files,
        "scanned_bytes": sc_bytes,
        "pct": min(100.0, pct),
        "current_file": raw.get("current_file"),
        "heartbeat": raw.get("heartbeat"),
    }
    if is_http and use_cache:
        with _HTTP_CACHE_LOCK:
            _HTTP_CACHE[src_str] = (time.monotonic(), dict(out))
    return out


def _file_valid(p: Path) -> bool:
    try:
        return p.is_file() and not p.is_symlink() and p.stat().st_size > 0
    except OSError:
        return False


def _resolve_baidu_status(lcfg: dict[str, Any]) -> dict[str, Any]:
    p = lcfg.get("baidu_status_path")
    if p and Path(str(p)).is_file():
        return read_baidu_backup_status(p)
    url = lcfg.get("baidu_status_url") or lcfg.get("baidu_status_path") or "http://127.0.0.1:8791/status.json"
    return read_baidu_backup_status(url)


def evaluate_source_retention(
    db: ArchiveDB,
    data_root: Path,
    cctv_root: Path,
    cfg: dict[str, Any],
    now_dt: dt.datetime | None = None,
) -> dict[str, Any]:
    """Evaluate 4 hard safety gates before any source video file can ever become eligible for local cleanup."""
    tz = ZoneInfo(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
    now = now_dt or dt.datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)

    scfg = cfg.get("scheduler", {})
    lcfg = cfg.get("lifecycle", {})
    retention_days = max(1, int(lcfg.get("source_retention_days", 90)))
    mode = str(lcfg.get("retention_mode", "observe_only")).lower()
    if mode not in ("observe_only", "enforce"):
        mode = "observe_only"
    protect_vault_forever = bool(lcfg.get("protect_vault_forever", True))
    baidu_status = _resolve_baidu_status(lcfg)

    # Gate 1: Initial source backup to Baidu Cloud must be completed (unless explicitly confirmed)
    backup_clean = (
        bool(baidu_status.get("initial_done")) and baidu_status.get("phase") != "completed_with_missing"
        and not any(baidu_status.get(key) for key in ("missing_files", "pending_files", "failed_files"))
    )
    gate1_ok = backup_clean or bool(lcfg.get("manual_backup_confirmed", False))

    # Gate 2: Full historical backfill in CCTV Growth Archive V3 must be 100% done
    min_day = str(scfg.get("p1_end_day", "2025-12-17"))
    max_day = str(scfg.get("p1_start_day", "2026-11-16"))
    hist_counts = db.get_historical_queue_counts(min_day, max_day)
    backfill_flag = bool(db.get_state("p1_backfill_complete", False))
    gate2_ok = backfill_flag and hist_counts["total"] == hist_counts["done"] and hist_counts["done"] > 0

    cutoff_date = (now.date() - dt.timedelta(days=retention_days)).isoformat()
    cutoff_mtime = now.timestamp() - (retention_days * 86400.0)
    pre_cutoff_counts = db.get_historical_queue_counts("1970-01-01", cutoff_date)
    if pre_cutoff_counts["total"] != pre_cutoff_counts["done"]:
        gate2_ok = False

    eligible_items: list[dict[str, Any]] = []
    verified_days_count = 0
    cctv_writable = bool(cctv_root.is_dir() and os.access(cctv_root, os.W_OK))

    # Gate 3 & 4 only scan candidates cleanly (zero side effects!)
    for day in db.get_verified_days_before(cutoff_date):
        sum_path = data_root / "days" / day / "summary.json"
        if not sum_path.is_file():
            continue
        try:
            sum_obj = json.loads(sum_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if sum_obj.get("status") != "done":
            continue
        hl_rel = sum_obj.get("highlight_rel")
        if hl_rel and not _file_valid(data_root / str(hl_rel)):
            continue

        day_rows = db.get_day_rows(day)
        if not day_rows or any(r.get("status") != "done" for r in day_rows):
            continue

        day_ms = db.get_day_milestones(day)
        if any(
            not ms.get("video_rel") or not _file_valid(data_root / str(ms["video_rel"]))
            or not ms.get("t0_rel") or not _file_valid(data_root / str(ms["t0_rel"]))
            for ms in day_ms
        ):
            continue

        verified_days_count += 1
        for r in day_rows:
            if r.get("source_root") != "cctv":
                continue
            if not r.get("t2_rel") or not _file_valid(data_root / str(r["t2_rel"])):
                continue
            if r.get("t1_rel") and not _file_valid(data_root / str(r["t1_rel"])):
                continue
            if not r.get("t1_rel") and r.get("t1_mode") != "verified_child_only":
                if json.loads(r.get("audio_summary_json") or "{}").get("segments"):
                    continue
            rel = Path(r["source_rel"])
            if rel.is_absolute() or ".." in rel.parts:
                continue
            src = cctv_root / rel
            try:
                if not src.is_file() or src.resolve() != cctv_root.resolve() / rel:
                    continue
                st = src.stat()
            except OSError:
                continue
            if st.st_mtime <= cutoff_mtime and st.st_size == r["size"] and st.st_mtime == r["mtime"]:
                eligible_items.append({
                    "source_rel": r["source_rel"],
                    "day": day,
                    "path": str(src),
                    "size": int(st.st_size),
                    "mtime": float(st.st_mtime),
                    "mtime_ns": st.st_mtime_ns,
                    "dev": st.st_dev,
                    "ino": st.st_ino,
                })

    can_prune = gate1_ok and gate2_ok
    eligible_bytes = sum(i["size"] for i in eligible_items)

    if not gate1_ok and not gate2_ok:
        guard_zh = "保护锁全开启：首次源片备份与历史全量归档均在进行中，绝不删除任何源视频"
    elif not gate1_ok:
        guard_zh = "保护锁生效中：等待百度网盘首次源片备份全部完成后才允许超期清理"
    elif not gate2_ok:
        guard_zh = "保护锁生效中：等待历史录像倒序归档全部追平完成，绝不提前删除源片"
    elif mode != "enforce" or not cctv_writable:
        guard_zh = f"已就绪（只读观察模式）：已锁定 {retention_days} 天保留期，期满可释放 {len(eligible_items)} 档"
    else:
        guard_zh = f"自动留存巡检生效中：保留近 {retention_days} 天源视频（原视频不上百度云）"

    return {
        "retention_days": retention_days,
        "retention_mode": mode,
        "cctv_writable": cctv_writable,
        "protect_vault_forever": protect_vault_forever,
        "cutoff_date": cutoff_date,
        "can_prune": can_prune,
        "gates": {
            "source_backup_done": gate1_ok,
            "archive_backfill_done": gate2_ok,
            "backfill_flag": backfill_flag,
            "hist_done": hist_counts["done"],
            "hist_queued": hist_counts["queued"],
        },
        "verified_expired_days": verified_days_count,
        "eligible_files": len(eligible_items),
        "eligible_bytes": eligible_bytes,
        "eligible_items": eligible_items,
        "baidu_backup": baidu_status,
        "guard_summary_zh": guard_zh,
    }


def run_retention_pass(
    db: ArchiveDB,
    data_root: Path,
    cctv_root: Path,
    cfg: dict[str, Any],
    now_dt: dt.datetime | None = None,
) -> dict[str, Any]:
    """Execute retention cleanup ONLY if all 4 safety gates pass, mode=='enforce', and cctv_root is writable."""
    eval_res = evaluate_source_retention(db, data_root, cctv_root, cfg, now_dt=now_dt)
    deleted_files = 0
    deleted_bytes = 0
    errors = 0
    items = eval_res.pop("eligible_items", [])

    if eval_res["can_prune"] and eval_res["retention_mode"] == "enforce" and eval_res["cctv_writable"]:
        retention_days = eval_res["retention_days"]
        tz = ZoneInfo(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
        now = now_dt or dt.datetime.now(tz)
        cutoff_mtime = now.timestamp() - (retention_days * 86400.0)
        for it in items:
            p = Path(it["path"])
            try:
                if not p.is_file() or p.resolve() != cctv_root.resolve() / it["source_rel"]:
                    continue
                st = p.stat()
                if (st.st_mtime > cutoff_mtime or st.st_size != it["size"] or st.st_mtime_ns != it["mtime_ns"]
                        or st.st_dev != it["dev"] or st.st_ino != it["ino"]):
                    continue
                sz = int(st.st_size)
                p.unlink()
                deleted_files += 1
                deleted_bytes += sz
            except OSError:
                errors += 1

    prev_audit = db.get_state("retention_audit", {}) or {}
    audit = {
        "checked_at": dt.datetime.now(ZoneInfo(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))).isoformat(timespec="seconds"),
        "can_prune": eval_res["can_prune"],
        "retention_mode": eval_res["retention_mode"],
        "last_deleted_files": deleted_files,
        "last_deleted_bytes": deleted_bytes,
        "total_deleted_files": int(prev_audit.get("total_deleted_files", 0)) + deleted_files,
        "total_deleted_bytes": int(prev_audit.get("total_deleted_bytes", 0)) + deleted_bytes,
        "errors": errors,
    }
    db.set_state("retention_audit", audit)
    return {**eval_res, "audit": audit}


class ArchiveCloudSync:
    """Syncs /volume2/Growth_Archive_V3 curated outputs (never /cctv raw sources) to Baidu Cloud via OpenList."""

    def __init__(self, data_root: Path, db: ArchiveDB, cfg: dict[str, Any]) -> None:
        self.data_root = data_root
        self.db = db
        self.cfg = cfg
        self.lcfg = cfg.get("lifecycle", {})
        self._token = ""
        self._known_dirs: set[str] = {"/"}
        self._sess = requests.Session()
        self._sess.trust_env = False

    def _base_url(self) -> str:
        return str(self.lcfg.get("openlist_url", "http://127.0.0.1:5244")).rstrip("/")

    def _read_creds(self) -> dict[str, Any] | None:
        p = Path(str(self.lcfg.get("openlist_credentials", "/run/secrets/openlist.json")))
        if not p.is_file():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) and obj.get("username") and obj.get("password") else None
        except Exception:
            return None

    def _login(self) -> None:
        creds = self._read_creds()
        if not creds:
            raise RuntimeError("OpenList credentials unavailable")
        r = self._sess.post(f"{self._base_url()}/api/auth/login", json=creds, timeout=20)
        r.raise_for_status()
        res = r.json()
        if res.get("code") != 200 or not (res.get("data") or {}).get("token"):
            raise RuntimeError(f"OpenList login code={res.get('code')}")
        self._token = str(res["data"]["token"])

    def _api(self, path: str, payload: dict[str, Any]) -> Any:
        if not self._token:
            self._login()
        for attempt in range(2):
            r = self._sess.post(
                f"{self._base_url()}{path}",
                json=payload,
                headers={"Authorization": self._token, "Content-Type": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
            res = r.json()
            code = res.get("code")
            if code in (401, 403) and attempt == 0:
                self._login()
                continue
            if code != 200:
                raise RuntimeError(f"OpenList {path} code={code}")
            return res.get("data")
        raise RuntimeError(f"OpenList {path} unauthorized")

    def _mkdir(self, remote_dir: str) -> None:
        norm = str(PurePosixPath(remote_dir))
        if norm in self._known_dirs:
            return
        self._mkdir(str(PurePosixPath(norm).parent))
        try:
            item = self._api("/api/fs/get", {"path": norm, "password": ""})
        except RuntimeError:
            self._api("/api/fs/mkdir", {"path": norm})
            item = self._api("/api/fs/get", {"path": norm, "password": ""})
        if not (item or {}).get("is_dir"):
            raise RuntimeError("Destination directory occupied by a file")
        self._known_dirs.add(norm)

    def _iter_curated_files(self) -> list[tuple[str, Path, os.stat_result]]:
        out: list[tuple[str, Path, os.stat_result]] = []
        bedside_timelapses = {
            f"timelapse_{camera}.mp4" for camera, settings in self.cfg.get("cameras", {}).items()
            if settings.get("role") == "bedside_crib"
        }
        blocked_highlights: set[str] = set()
        approved_films: set[str] = set()
        try:
            all_ms = self.db.query_milestones(limit=500, include_pending=True)
            for mr in all_ms:
                if not is_confirmed_milestone_record(mr):
                    if mr.get("video_rel"):
                        blocked_highlights.add(Path(str(mr["video_rel"])).stem)
                    blocked_highlights.add(str(mr["id"]))
            for fp in self.db.list_film_productions():
                if fp.get("status") == "published" and fp.get("review_status") == "approved":
                    approved_films.add(str(fp["film_id"]))
        except Exception:
            pass

        def collect(
            directory: Path,
            patterns: tuple[str, ...],
            *,
            allowed_stems: set[str] | None = None,
            blocked_stems: set[str] | None = None,
        ) -> None:
            # Check each directory component before descending into gallery assets.
            current = self.data_root
            for part in directory.relative_to(self.data_root).parts:
                current /= part
                if current.is_symlink() or not current.is_dir():
                    return
            for p in sorted(directory.iterdir()):
                if (p.name.startswith(".") or p.name in bedside_timelapses
                        or p.name.endswith((".tmp_concat.mp4", ".tmp_v.mp4", ".tmp.jpg"))):
                    continue
                if not any(p.match(pattern) for pattern in patterns):
                    continue
                stem = p.name[: -len(".manifest.json")] if p.name.endswith(".manifest.json") else p.stem
                if allowed_stems is not None and stem not in allowed_stems:
                    continue
                if blocked_stems is not None and stem in blocked_stems:
                    continue
                try:
                    st = p.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(st.st_mode) and st.st_size > 0:
                    out.append((p.relative_to(self.data_root).as_posix(), p, st))

        # T0 keepers are exact source recordings. Publish only named gallery
        # products, including the original pilot day's finished exports.
        v2 = self.data_root / "v2"
        collect(v2, ("deliverables.json", "growth_smooth_*.mp4", "growth_smooth_*.jpg",
                     "smooth_*.mp4", "smooth_*.jpg", "monthly_growth_*.mp4", "monthly_growth_*.jpg"))
        for sub in ("photos", "monthly_cards"):
            collect(v2 / sub, ("*.jpg",))
        collect(v2 / "highlights", ("*.jpg", "*.mp4"), blocked_stems=blocked_highlights)
        if approved_films:
            collect(v2 / "films", ("*.mp4", "*.jpg", "*.manifest.json"), allowed_stems=approved_films)
        day_dirs = [self.data_root / "day-20261108"]
        days = self.data_root / "days"
        if not days.is_symlink() and days.is_dir():
            day_dirs.extend(sorted(days.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]")))
        for directory in day_dirs:
            collect(directory, ("summary.json", "highlight_*.mp4", "timelapse_*.mp4", "compare.jpg"))
            collect(directory / "photos", ("*.jpg",))
        return out

    def sync_incremental(self, max_files: int = 30) -> dict[str, Any]:
        tz = ZoneInfo(self.cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
        now_iso = dt.datetime.now(tz).isoformat(timespec="seconds")
        enabled = bool(self.lcfg.get("archive_cloud_sync_enabled", True))
        if not enabled:
            st = {"status": "disabled", "note_zh": "精编档案网盘同步已关闭", "updated_at": now_iso}
            self.db.set_state("archive_cloud_sync", st)
            return st

        baidu_status = _resolve_baidu_status(self.lcfg)
        defer = bool(self.lcfg.get("defer_archive_sync_until_source_backup_done", True))
        if defer and not baidu_status.get("initial_done"):
            st = {
                "status": "deferred_for_initial_backup",
                "note_zh": f"礼让首次源片上云（{baidu_status.get('pct', 0)}%），完成后自动同步精编馆至百度网盘 {self.lcfg.get('openlist_destination', '/Growth_Archive_V3')}",
                "destination": self.lcfg.get("openlist_destination", "/Growth_Archive_V3"),
                "updated_at": now_iso,
            }
            self.db.set_state("archive_cloud_sync", st)
            return st

        if not self._read_creds():
            st = {
                "status": "no_credentials",
                "note_zh": "未检测到 OpenList 同步凭据，当前仅保存于 NAS 独立目录 /volume2/Growth_Archive_V3",
                "updated_at": now_iso,
            }
            self.db.set_state("archive_cloud_sync", st)
            return st

        dest_root = str(self.lcfg.get("openlist_destination", "/Growth_Archive_V3")).rstrip("/") or "/Growth_Archive_V3"
        files = self._iter_curated_files()
        synced_map = self.db.get_cloud_sync_map()
        uploaded = 0
        uploaded_bytes = 0
        already_synced = 0
        failures = 0

        for rel, path, st in files:
            prev = synced_map.get(rel)
            if prev and int(prev["size"]) == int(st.st_size) and int(prev["mtime_ns"]) == int(st.st_mtime_ns):
                already_synced += 1
                continue
            if uploaded >= max_files:
                continue
            remote_path = f"{dest_root}/{rel}"
            try:
                self._mkdir(str(PurePosixPath(remote_path).parent))
                if not self._token:
                    self._login()
                h = hashlib.sha256()
                with path.open("rb") as fh:
                    while True:
                        blk = fh.read(1024 * 1024)
                        if not blk:
                            break
                        h.update(blk)
                with path.open("rb") as fh:
                    r = self._sess.put(
                        f"{self._base_url()}/api/fs/put",
                        data=fh,
                        headers={
                            "Authorization": self._token,
                            "File-Path": quote(remote_path, safe="/"),
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(st.st_size),
                            "As-Task": "false",
                            "Overwrite": "true",
                            "Last-Modified": str(st.st_mtime_ns // 1_000_000),
                        },
                        timeout=300,
                    )
                    r.raise_for_status()
                    res = r.json()
                    if res.get("code") != 200:
                        raise RuntimeError(f"put code={res.get('code')}")
                self.db.mark_cloud_synced(rel, int(st.st_size), int(st.st_mtime_ns), h.hexdigest())
                uploaded += 1
                uploaded_bytes += int(st.st_size)
                already_synced += 1
            except Exception:
                failures += 1
                if failures >= 3:
                    break

        pending = max(0, len(files) - already_synced)
        st = {
            "status": "synced" if pending == 0 and failures == 0 else ("syncing" if failures == 0 else "retry_pending"),
            "destination": dest_root,
            "total_curated_files": len(files),
            "synced_files": already_synced,
            "pending_files": pending,
            "last_uploaded": uploaded,
            "last_uploaded_bytes": uploaded_bytes,
            "failures": failures,
            "note_zh": f"已同步精编档案 {already_synced}/{len(files)} 档至百度网盘 {dest_root}" if pending == 0 else f"增量同步中：{already_synced}/{len(files)} 档（剩余 {pending} 档）",
            "updated_at": now_iso,
        }
        self.db.set_state("archive_cloud_sync", st)
        return st
