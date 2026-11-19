"""P0-first (Today/Yesterday -> Reverse Historical Backfill) unattended NAS Growth V3 daemon."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from . import detectors as det
from . import asr_worker
from . import daily_director as dl_dir
from . import film_compiler as fc
from . import lifecycle as lc
from . import media_nas as med
from .cloud_vlm import (
    PROMPT_KITCHEN_CHILD,
    PROMPT_MILESTONE,
    PROMPT_PHOTO,
    CloudVLMRouter,
    extract_jpegs_ram,
    is_cloud_allowed,
    redact_secrets,
)
from .db import ArchiveDB, is_unconfirmed_refusal
from .curation_queue import CurationQueue
from .family_faces import FamilyFaceService


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


def _resolve_privacy_box(ans: dict[str, Any], fallback_box: list[float] | None) -> list[float] | None:
    pb = ans.get("privacy_box")
    if isinstance(pb, (list, tuple)) and len(pb) == 4 and float(pb[2]) > float(pb[0]) and float(pb[3]) > float(pb[1]):
        return [float(v) for v in pb]
    if (
        ans.get("local_care_fallback") in ("yes", "pending_review", True)
        or ans.get("vlm_refused")
        or ans.get("privacy_exposed") == "yes"
    ):
        return med.estimate_pelvic_privacy_box(fallback_box)
    return None


def read_host_mem_available_kib() -> int:
    """Read Linux host MemAvailable KiB; returns high default on non-Linux dev hosts."""
    p = Path("/proc/meminfo")
    if not p.is_file():
        return 16_777_216
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return 16_777_216


def read_container_headroom_kib(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Estimate allocatable cgroup v2 memory, including reclaimable file cache."""
    try:
        limit = (root / "memory.max").read_text().strip()
        if limit == "max":
            return None
        usage = int((root / "memory.current").read_text())
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        reclaimable = int(stats.get("inactive_file", 0)) + int(stats.get("active_file", 0)) // 2
        return max(0, int(limit) - usage + reclaimable) // 1024
    except (OSError, ValueError):
        return None


def _atomic_swap_dir(parent_dir: Path, stage_dir: Path, target_dir: Path) -> bool:
    """Atomically replace target_dir with stage_dir, preserving prior contents on any failure."""
    bak_dir = parent_dir / f".{target_dir.name}_bak_{os.getpid()}_{int(time.time() * 1000)}"
    if bak_dir.exists():
        shutil.rmtree(bak_dir, ignore_errors=True)
    had_prev = target_dir.exists()
    try:
        if had_prev:
            os.replace(target_dir, bak_dir)
        os.replace(stage_dir, target_dir)
    except OSError:
        if had_prev and not target_dir.exists() and bak_dir.exists():
            try:
                os.replace(bak_dir, target_dir)
            except OSError:
                pass
        shutil.rmtree(stage_dir, ignore_errors=True)
        return False
    if bak_dir.exists():
        shutil.rmtree(bak_dir, ignore_errors=True)
    return True


def _shanghai_now(cfg: dict[str, Any]) -> dt.datetime:
    return dt.datetime.now(ZoneInfo(cfg["project"].get("timezone", "Asia/Shanghai")))


def _parse_cloud_g3_ts(path: Path, tz: ZoneInfo) -> dt.datetime | None:
    stem = path.stem
    m = re.match(r"camera_(\d{10,16})(?:\D.*)?$", stem, re.I) or re.search(r"(?<!\d)(\d{13}|\d{10})(?!\d)", stem)
    if m:
        val = int(m.group(1))
        if val > 10**14:
            sec = val / 1e6
        elif val > 10**11:
            sec = val / 1e3
        else:
            sec = float(val)
        try:
            cand = dt.datetime.fromtimestamp(sec, tz=tz)
            if 2024 <= cand.year <= 2035:
                return cand
        except (OSError, ValueError):
            pass

    m_ymd = re.search(r"(?P<y>20\d{2})[-_]?(?P<mo>\d{2})[-_]?(?P<d>\d{2})[_ T-]?(?P<h>\d{2})[-:]?(?P<mi>\d{2})[-:]?(?P<s>\d{2})", stem)
    if m_ymd:
        try:
            return dt.datetime(
                int(m_ymd.group("y")), int(m_ymd.group("mo")), int(m_ymd.group("d")),
                int(m_ymd.group("h")), int(m_ymd.group("mi")), int(m_ymd.group("s")),
                tzinfo=tz,
            )
        except ValueError:
            pass

    for part in reversed(path.parts[:-1]):
        m_dir = re.fullmatch(r"(20\d{2})-?(\d{2})-?(\d{2})", part)
        if m_dir:
            y, mo, d = int(m_dir.group(1)), int(m_dir.group(2)), int(m_dir.group(3))
            m_hms = re.search(r"(?<!\d)(\d{2})[-:]?(\d{2})[-:]?(\d{2})(?!\d)", stem)
            hh, mm, ss = (int(m_hms.group(1)), int(m_hms.group(2)), int(m_hms.group(3))) if m_hms else (12, 0, 0)
            try:
                return dt.datetime(y, mo, d, min(23, hh), min(59, mm), min(59, ss), tzinfo=tz)
            except ValueError:
                break

    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=tz)
    except OSError:
        return None


def _iter_cloud_backfill_mp4s(cb_dir: Path) -> list[Path]:
    """Recursively yield settled .mp4/.MP4 files under cloud_backfill_dir, skipping hidden/partial uploads."""
    if not cb_dir.is_dir():
        return []
    skip_suffixes = (".part", ".tmp", ".crdownload", ".download", ".partial")
    out: list[Path] = []
    for p in cb_dir.rglob("*"):
        try:
            rel_parts = p.relative_to(cb_dir).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        low = p.name.lower()
        if any(low.endswith(sfx) for sfx in skip_suffixes):
            continue
        if p.suffix.lower() == ".mp4" and p.is_file():
            out.append(p)
    out.sort(key=lambda p: p.as_posix())
    return out


def _parse_xiaomi_ts(path: Path, tz: ZoneInfo) -> dt.datetime | None:
    parts = path.stem.split("_")
    for idx in (1, 2):
        if len(parts) > idx and len(parts[idx]) == 14 and parts[idx].isdigit():
            try:
                return dt.datetime.strptime(parts[idx], "%Y%m%d%H%M%S").replace(tzinfo=tz)
            except ValueError:
                pass
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=tz)
    except OSError:
        return None


def _parse_xiaomi_minute_ts(path: Path, tz: ZoneInfo) -> dt.datetime | None:
    m = re.match(r"^(?P<m>\d{2})M(?P<s>\d{2})S_(?P<epoch>\d{10})$", path.stem, re.I)
    if m:
        try:
            return dt.datetime.fromtimestamp(int(m.group("epoch")), tz=tz)
        except (OSError, ValueError):
            pass
    pdir = path.parent.name
    m2 = re.match(r"^(?P<m>\d{2})M(?P<s>\d{2})S", path.stem, re.I)
    if len(pdir) == 10 and pdir.isdigit():
        try:
            mm = int(m2.group("m")) if m2 else 0
            ss = int(m2.group("s")) if m2 else 0
            return dt.datetime.strptime(pdir, "%Y%m%d%H").replace(minute=min(59, mm), second=min(59, ss), tzinfo=tz)
        except ValueError:
            pass
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=tz)
    except OSError:
        return None


class GrowthDaemon:
    def __init__(self, cfg: dict[str, Any], db: ArchiveDB, ff_bin: str = "ffmpeg", fp_bin: str = "ffprobe", *, config_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        self._cfg_lock = threading.Lock()
        self._last_cfg_check = 0.0
        self._cfg_sig: tuple[int, int] | None = None
        if self.config_path and self.config_path.is_file():
            try:
                st = self.config_path.stat()
                self._cfg_sig = (int(st.st_mtime_ns), int(st.st_size))
            except OSError:
                self._cfg_sig = None
        self.db = db
        self.ff = ff_bin
        self.fp = fp_bin
        self.cctv_root = Path(cfg["project"]["cctv_root"])
        self.vault_root = Path(cfg["project"].get("vault_root", "/vault"))
        self.data_root = Path(cfg["project"]["data_root"])
        self.tz = ZoneInfo(cfg["project"].get("timezone", "Asia/Shanghai"))
        self.pause_flag = self.data_root / "state" / "PAUSE"

        mcfg = cfg.get("media", {})
        self.vaapi_dev = mcfg.get("vaapi_device", "/dev/dri/renderD128")
        self.use_vaapi = bool(mcfg.get("prefer_vaapi", True)) and med.check_vaapi(self.ff, self.vaapi_dev)
        self.font_path = med.resolve_font(mcfg.get("font_candidates", []))
        self.vlm = CloudVLMRouter(cfg, db)
        self.face_service = FamilyFaceService(db, self.data_root, cfg, font_path=self.font_path)
        active_vlm_provs = set(((cfg.get("vlm") or {}).get("providers") or {}).keys())
        if active_vlm_provs:
            self.db.prune_retired_vlm_stats(active_vlm_provs)

        acfg = cfg["asr"]
        yolo_pool = max(3, min(6, int(cfg.get("scheduler", {}).get("workers", 4))))
        self.yolo = det.YoloOnnxDetector(acfg["yolo_onnx"], intra_threads=1, pool_size=yolo_pool)
        self.yamnet = det.YamnetLite(acfg["yamnet_model"], acfg["yamnet_class_map"])
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.curation_queue: CurationQueue | None = None
        self._p1_cursor: dt.date | None = None
        self._first_live_day: dt.date | None = None
        self._next_t2_retry_check = 0.0
        self._exhausted_days: set[str] = set()
        self._cloud_backfill_sig: dict[str, tuple[int, int]] = {}
        self._job_lock = threading.Lock()
        self._curator_status: str = ""
        self._telemetry: dict[str, Any] = {
            "host_mem_yields": 0,
            "host_mem_throttles": 0,
            "container_headroom_throttles": 0,
            "heavy_cap_deferrals": 0,
            "min_host_mem_kib": read_host_mem_available_kib(),
            "max_inflight": 0,
        }
        self._last_telemetry_flush = 0.0
        self.cloud_sync = lc.ArchiveCloudSync(self.data_root, self.db, cfg)
        self.db.set_state("runtime_caps", {
            "vaapi_ok": self.use_vaapi,
            "vaapi_device": self.vaapi_dev,
            "font_path": self.font_path,
            "host_mem_kib": read_host_mem_available_kib(),
        })

    def _flush_telemetry(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_telemetry_flush) < 5.0:
            return
        self._last_telemetry_flush = now
        self.db.set_state("scheduler_telemetry", dict(self._telemetry))

    def compile_film(
        self,
        spec: dict[str, Any],
        *,
        max_shots: int | None = None,
        force: bool = False,
        out_w: int = 1920,
        out_h: int = 1080,
    ) -> dict[str, Any]:
        """Synchronously enqueue, compile, verify, and atomically publish a monthly, seasonal, viewpoint, or annual film."""
        cams = self._effective_cameras()
        if force:
            self.db.enqueue_film_production(spec, force=True)

        def _film_prog(stage: str) -> None:
            with self._job_lock:
                self._curator_status = f"{spec['film_id']} · {stage}"

        return fc.render_and_publish_film(
            self.cfg,
            self.db,
            self.data_root,
            self.cctv_root,
            self.vault_root,
            self.ff,
            self.fp,
            self.font_path,
            self.use_vaapi,
            self.vaapi_dev,
            spec,
            cams,
            qp=int(self.cfg.get("media", {}).get("smooth_qp", 19)),
            max_shots=max_shots,
            out_w=out_w,
            out_h=out_h,
            vlm=self.vlm,
            face_service=self.face_service,
            progress_fn=_film_prog,
        )

    def request_film_compile(self, film_id: str, *, force: bool = True) -> dict[str, Any]:
        """Enqueue any monthly, seasonal, camera_chronicle, or annual film and wake the daemon immediately."""
        now_tz = _shanghai_now(self.cfg)
        bday = str(self.cfg.get("project", {}).get("birthday", "2025-12-17"))
        compile_day = int(self.cfg.get("lifecycle", {}).get("monthly_compile_day", 18))
        spec = lc.resolve_film_spec_by_id(bday, film_id, now_tz.date(), compile_day=compile_day)
        job = self.db.enqueue_film_production(spec, force=force, preserve_rendering=True)
        already_rendering = str(job.get("status") or "") == "rendering"
        self._wake.set()
        return {
            "queued": True,
            "already_rendering": already_rendering,
            "film": job,
            "spec": spec,
            "curator_status": self.get_curator_status(),
            "paused": self.is_paused(),
        }

    def sync_and_compile_films(self, *, compile_one: bool = True) -> list[dict[str, Any]]:
        """Enqueue due monthly/annual film jobs from schedule and optionally render the next ready job."""
        now_tz = _shanghai_now(self.cfg)
        bday = str(self.cfg.get("project", {}).get("birthday", "2025-12-17"))
        compile_day = int(self.cfg.get("lifecycle", {}).get("monthly_compile_day", 18))
        sch = lc.compute_monthly_schedule(bday, now_tz.date(), self.data_root, compile_day=compile_day, db=self.db)
        for dp in sch.get("due_productions") or []:
            if dp.get("ready"):
                self.db.enqueue_film_production(dp, force=False)
        results: list[dict[str, Any]] = []
        if not compile_one or self.is_paused():
            return results
        min_mem_kib = int(self.cfg["scheduler"].get("min_host_mem_available_kib", 4194304))
        if read_host_mem_available_kib() < min_mem_kib:
            return results
        job = self.db.next_ready_film_production()
        if job is not None:
            results.append(self.compile_film(job))
        return results

    def refresh_lifecycle_status(self) -> dict[str, Any]:
        """Update 4-gate source retention guard, Baidu backup status, archive cloud sync, and baby age schedule."""
        now_tz = _shanghai_now(self.cfg)
        bday = str(self.cfg.get("project", {}).get("birthday", "2025-12-17"))
        compile_day = int(self.cfg.get("lifecycle", {}).get("monthly_compile_day", 18))
        ret = lc.run_retention_pass(self.db, self.data_root, self.cctv_root, self.cfg, now_dt=now_tz)
        age = lc.compute_age_info(bday, now_tz.date())
        monthly = lc.compute_monthly_schedule(bday, now_tz.date(), self.data_root, compile_day=compile_day, db=self.db)
        for dp in monthly.get("due_productions") or []:
            if dp.get("ready"):
                self.db.enqueue_film_production(dp, force=False)
        sync_st = self.cloud_sync.sync_incremental(max_files=24)
        combined = {**ret, "archive_cloud_sync": sync_st, "age": age, "monthly_schedule": monthly}
        self.db.set_state("lifecycle_status", combined)
        return combined

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def request_day_curation(self, day: str, tier: str | None = None) -> dict[str, Any]:
        """Force-enqueue immediate re-curation of a day and wake the scheduler loop."""
        dt.date.fromisoformat(day)
        stats = self.db.count_day_stats(day)
        now_d = _shanghai_now(self.cfg).date()
        p0_days = {now_d.isoformat(), (now_d - dt.timedelta(days=1)).isoformat()}
        eff_tier = tier or ("P0_LIVE" if day in p0_days else "P1_BACKFILL")
        is_final = (stats["queued"] == 0 and stats["done"] > 0)
        q = self.curation_queue if self.curation_queue is not None else CurationQueue(self.db)
        job = q.force_request(day, eff_tier, is_final, int(stats["done"]))
        self._wake.set()
        return {"queued": True, "day": day, **job}

    def reload_config_if_changed(self, *, force: bool = False) -> bool:
        """Hot-reload TOML configuration in-place when config_path changes without restarting the container."""
        if not self.config_path or not self.config_path.is_file():
            return False
        now_m = time.monotonic()
        with self._cfg_lock:
            if not force and (now_m - self._last_cfg_check) < 2.0:
                return False
            self._last_cfg_check = now_m
            try:
                st = self.config_path.stat()
                sig = (int(st.st_mtime_ns), int(st.st_size))
            except OSError:
                return False
            if not force and sig == self._cfg_sig:
                return False
            try:
                with self.config_path.open("rb") as fh:
                    new_cfg = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                return False
            for k in list(self.cfg.keys()):
                if k not in new_cfg:
                    del self.cfg[k]
            for k, v in new_cfg.items():
                self.cfg[k] = v
            if hasattr(self, "vlm"):
                self.vlm.vlm_cfg = self.cfg.get("vlm", {})
            if hasattr(self, "cloud_sync"):
                self.cloud_sync.cfg = self.cfg
                self.cloud_sync.lcfg = self.cfg.get("lifecycle", {})
            if hasattr(self, "face_service"):
                self.face_service.cfg = self.cfg
                self.face_service.ensure_seeded()
                self.face_service.invalidate_cache()
            self._cfg_sig = sig
        try:
            self.refresh_camera_health()
        except Exception:
            pass
        return True

    def is_paused(self) -> bool:
        return self.pause_flag.is_file()

    def get_curator_status(self) -> str:
        with self._job_lock:
            return str(self._curator_status or "")

    def set_paused(self, paused: bool) -> None:
        self.pause_flag.parent.mkdir(parents=True, exist_ok=True)
        if paused:
            self.pause_flag.write_text(f"paused_at={dt.datetime.now(self.tz).isoformat()}\n", encoding="utf-8")
        else:
            self.pause_flag.unlink(missing_ok=True)

    def _effective_cameras(self) -> dict[str, dict[str, Any]]:
        ad = self.cfg.get("auto_discovery", {})
        ignore = set(ad.get("ignore_ids", []))
        cams = {cid: {**c, "id": cid, "discovered": False} for cid, c in self.cfg.get("cameras", {}).items()}
        if ad.get("enabled", True):
            for root_rel in ad.get("roots", ["aqara_video", "xiaomi_camera_videos"]):
                for base in (self.cctv_root, self.vault_root):
                    rdir = base / root_rel
                    if not rdir.is_dir():
                        continue
                    fam = "aqara" if "aqara" in root_rel else "xiaomi_minute"
                    for entry in sorted(rdir.iterdir(), key=lambda p: p.name):
                        if entry.is_dir() and entry.name not in ignore and entry.name not in cams:
                            cams[entry.name] = {
                                "id": entry.name,
                                "name": f"新发现摄像头 ({entry.name})",
                                "family": fam,
                                "relative_root": f"{root_rel}/{entry.name}",
                                "enabled": True,
                                "role": ad.get("default_role", "study_or_room"),
                                "zoom_factor": float(ad.get("default_zoom_factor", 1.35)),
                                "t1_res": int(ad.get("default_t1_res", 720)),
                                "milestone_watch": True,
                                "daytime_photos": True,
                                "discovered": True,
                            }
        return cams

    def refresh_camera_health(self) -> dict[str, Any]:
        cams = self._effective_cameras()
        stale_days = float(self.cfg.get("auto_discovery", {}).get("stale_warning_days", 1.5))
        now = _shanghai_now(self.cfg).replace(tzinfo=None)
        health: list[dict[str, Any]] = []

        for cid, c in cams.items():
            if not c.get("enabled", True):
                health.append({
                    "id": cid, "name": c["name"], "role": c["role"], "enabled": False,
                    "zoom_factor": c.get("zoom_factor", 1.0), "t1_res": c.get("t1_res", 720),
                    "status": "disabled", "latest_ts": None, "days_stale": None, "note": "已停用历史目录",
                })
                continue

            lts: dt.datetime | None = None
            count = 0
            for base in (self.cctv_root, self.vault_root):
                cdir = base / c["relative_root"]
                if not cdir.is_dir():
                    continue
                if c["family"] == "aqara":
                    ddirs = sorted([d for d in cdir.iterdir() if d.is_dir() and len(d.name) == 8 and d.name.isdigit()], key=lambda d: d.name, reverse=True)
                    for dd in ddirs[:2]:
                        mp4s = sorted(dd.glob("*.mp4"))
                        if mp4s:
                            count = len(mp4s)
                            try:
                                ts = dt.datetime.strptime(f"{dd.name}{mp4s[-1].stem[:6]}", "%Y%m%d%H%M%S")
                            except ValueError:
                                ts = dt.datetime.fromtimestamp(mp4s[-1].stat().st_mtime)
                            if lts is None or ts > lts:
                                lts = ts
                            break
                elif c["family"] == "xiaomi_bundle":
                    mp4s = sorted(cdir.glob("00_*.mp4"))
                    if mp4s:
                        count = len(mp4s)
                        ts_z = _parse_xiaomi_ts(mp4s[-1], self.tz)
                        if ts_z:
                            ts = ts_z.replace(tzinfo=None)
                            if lts is None or ts > lts:
                                lts = ts
                elif c["family"] == "xiaomi_minute":
                    hdirs = sorted([d for d in cdir.iterdir() if d.is_dir() and len(d.name) == 10 and d.name.isdigit()], key=lambda d: d.name, reverse=True)
                    for hd in hdirs[:3]:
                        mp4s = sorted(hd.glob("*.mp4"))
                        if mp4s:
                            count = max(count, len(mp4s))
                            for p in mp4s[-5:]:
                                ts_z = _parse_xiaomi_minute_ts(p, self.tz)
                                if ts_z:
                                    ts = ts_z.replace(tzinfo=None)
                                    if lts is None or ts > lts:
                                        lts = ts
                            break

            cb_info: dict[str, Any] | None = None
            cdir_note = ""
            if c.get("cloud_backfill_dir"):
                cb_dir = self.cctv_root / c["cloud_backfill_dir"]
                start_d, end_d = self._cloud_backfill_bounds(c, now.date())
                closed_cd = float(self.cfg["scheduler"].get("closed_file_cooldown_seconds", 900))
                now_s = time.time()
                cb_ready = 0
                cb_cooling = 0
                cb_days_set: set[str] = set()
                for p in _iter_cloud_backfill_mp4s(cb_dir):
                    ts_cb = _parse_cloud_g3_ts(p, self.tz)
                    if not ts_cb:
                        continue
                    d_cb = ts_cb.strftime("%Y-%m-%d")
                    if d_cb < start_d or d_cb > end_d:
                        continue
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_size < 4096:
                        continue
                    cb_days_set.add(d_cb)
                    if (now_s - st.st_mtime) < closed_cd:
                        cb_cooling += 1
                    else:
                        cb_ready += 1
                cb_days = sorted(cb_days_set)
                cb_info = {
                    "dir": str(c["cloud_backfill_dir"]),
                    "start_day": start_d,
                    "end_day": end_d,
                    "ready_files": cb_ready,
                    "cooling_files": cb_cooling,
                    "days": cb_days,
                }
                span_lbl = f"{start_d[5:]}~{end_d[5:]}"
                cool_txt = f"，冷却中 {cb_cooling} 档" if cb_cooling else ""
                cdir_note = f" · {c['cloud_backfill_dir']} 补录监听中（{span_lbl} 已就绪 {cb_ready} 档{cool_txt}，后续补入自动纳管重编）"
            days_stale = round((now - lts).total_seconds() / 86400.0, 2) if lts else None
            is_hist = bool(c.get("historical_archive"))
            stale = (not is_hist) and (days_stale is not None) and (days_stale > stale_days)
            status = "historical" if (is_hist and lts) else ("stale_warning" if stale else ("online" if lts else "empty"))
            note = (f"SMB 已停更 {days_stale:.1f} 天{cdir_note}" if stale else ("正常写入" + cdir_note))
            if is_hist and lts:
                note = f"历史厨房机位（已归档至 {lts:%Y-%m-%d}）· 仅保留 Gemini 核验有宝宝片段进 T1，其余转 1fps 延时"
            elif c["role"] == "kitchen_4k":
                note = "4K 新机位：仅 Gemini 验证有宝宝片段留 T1，其余转 1fps 延时"
            elif c["role"] == "bedside_crib":
                note = "白天屏蔽静态床单，夜间纯睡眠进 T2 延时"
            elif c.get("discovered"):
                note = "自动发现的新机位，已按默认室内策略纳入"
            cam_h: dict[str, Any] = {
                "id": cid, "name": c["name"], "role": c["role"], "enabled": True,
                "zoom_factor": float(c.get("zoom_factor", 1.35)), "t1_res": int(c.get("t1_res", 720)),
                "status": status, "latest_ts": lts.isoformat(timespec="seconds") if lts else None,
                "latest_files": count, "days_stale": days_stale, "note": note,
            }
            if cb_info is not None:
                cam_h["cloud_backfill"] = cb_info
            health.append(cam_h)

        out = {"checked_at": now.isoformat(timespec="seconds"), "cameras": cams, "health": health}
        hp = self.data_root / "v2" / "camera_health.json"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db.set_state("camera_health", out)
        return out

    def discover_day_files(self, day_str: str) -> int:
        """Scan cctv_root (with vault_root fallback + Aqara_Cloud_G3) for a single YYYY-MM-DD and enqueue stabilized files."""
        self.reload_config_if_changed()
        cams = self._effective_cameras()
        day_compact = day_str.replace("-", "")
        now_s = time.time()
        scfg = self.cfg["scheduler"]
        active_cd = float(scfg.get("active_file_cooldown_seconds", 5400))
        closed_cd = float(scfg.get("closed_file_cooldown_seconds", 900))
        seen_rel: dict[str, dict[str, Any]] = {}

        for cid, c in cams.items():
            if not c.get("enabled", True):
                continue
            for root_name, root_path in (("cctv", self.cctv_root), ("vault", self.vault_root)):
                cdir = root_path / c["relative_root"]
                if not cdir.is_dir():
                    continue
                if c["family"] == "aqara":
                    ddir = cdir / day_compact
                    if not ddir.is_dir():
                        continue
                    mp4s = sorted(ddir.glob("*.mp4"), key=lambda p: p.name)
                    last_name = mp4s[-1].name if mp4s else ""
                    for p in mp4s:
                        rel = f"{c['relative_root']}/{day_compact}/{p.name}"
                        if rel in seen_rel:
                            continue
                        try:
                            st = p.stat()
                        except OSError:
                            continue
                        req_cd = active_cd if p.name == last_name else closed_cd
                        if st.st_size < 4096 or (now_s - st.st_mtime) < req_cd:
                            continue
                        try:
                            wdt = dt.datetime.strptime(f"{day_compact}{p.stem[:6]}", "%Y%m%d%H%M%S")
                            wall_ts = wdt.isoformat(timespec="seconds")
                        except ValueError:
                            wall_ts = dt.datetime.fromtimestamp(st.st_mtime, tz=self.tz).replace(tzinfo=None).isoformat(timespec="seconds")
                        seen_rel[rel] = {
                            "source_rel": rel, "source_root": root_name, "day": day_str, "camera": cid,
                            "wall_ts": wall_ts, "size": st.st_size, "mtime": st.st_mtime,
                        }
                elif c["family"] == "xiaomi_minute":
                    target_d = dt.date.fromisoformat(day_str)
                    cand_hours = [f"{(target_d - dt.timedelta(days=1)).strftime('%Y%m%d')}23"]
                    cand_hours.extend(f"{day_compact}{hh:02d}" for hh in range(24))
                    cand_hours.append(f"{(target_d + dt.timedelta(days=1)).strftime('%Y%m%d')}00")
                    for hname in cand_hours:
                        hdir = cdir / hname
                        if not hdir.is_dir():
                            continue
                        mp4s = sorted(hdir.glob("*.mp4"), key=lambda p: p.name)
                        for p in mp4s:
                            ts_z = _parse_xiaomi_minute_ts(p, self.tz)
                            if not ts_z or ts_z.strftime("%Y-%m-%d") != day_str:
                                continue
                            rel = f"{c['relative_root']}/{hname}/{p.name}"
                            if rel in seen_rel:
                                continue
                            try:
                                st = p.stat()
                            except OSError:
                                continue
                            if st.st_size < 4096 or (now_s - st.st_mtime) < closed_cd:
                                continue
                            seen_rel[rel] = {
                                "source_rel": rel, "source_root": root_name, "day": day_str, "camera": cid,
                                "wall_ts": ts_z.replace(tzinfo=None).isoformat(timespec="seconds"),
                                "size": st.st_size, "mtime": st.st_mtime,
                            }
                elif c["family"] == "xiaomi_bundle":
                    mp4s = sorted(cdir.glob(f"00_{day_compact}*.mp4"), key=lambda p: p.name)
                    # Also include cross-midnight bundles starting on previous day ending on this day if short
                    last_name = mp4s[-1].name if mp4s else ""
                    for p in mp4s:
                        rel = f"{c['relative_root']}/{p.name}"
                        if rel in seen_rel:
                            continue
                        try:
                            st = p.stat()
                        except OSError:
                            continue
                        req_cd = active_cd if p.name == last_name else closed_cd
                        if st.st_size < 4096 or (now_s - st.st_mtime) < req_cd:
                            continue
                        ts_z = _parse_xiaomi_ts(p, self.tz)
                        wall_ts = ts_z.replace(tzinfo=None).isoformat(timespec="seconds") if ts_z else f"{day_str}T12:00:00"
                        seen_rel[rel] = {
                            "source_rel": rel, "source_root": root_name, "day": day_str, "camera": cid,
                            "wall_ts": wall_ts, "size": st.st_size, "mtime": st.st_mtime,
                        }

            # Optional Aqara_Cloud_G3 backfill mapping onto living_room camera
            if c.get("cloud_backfill_dir"):
                start_d, end_d = self._cloud_backfill_bounds(c, _shanghai_now(self.cfg).date())
                if start_d <= day_str <= end_d:
                    cb_dir = self.cctv_root / c["cloud_backfill_dir"]
                    lan_prefix = f"{c['relative_root']}/"
                    lan_epochs: list[float] = []
                    for rec in seen_rel.values():
                        if rec.get("camera") == cid and str(rec.get("source_rel", "")).startswith(lan_prefix) and rec.get("wall_ts"):
                            try:
                                lan_epochs.append(dt.datetime.fromisoformat(str(rec["wall_ts"])).replace(tzinfo=self.tz).timestamp())
                            except ValueError:
                                pass
                    for p in _iter_cloud_backfill_mp4s(cb_dir):
                        ts_cb = _parse_cloud_g3_ts(p, self.tz)
                        if not ts_cb or ts_cb.strftime("%Y-%m-%d") != day_str:
                            continue
                        try:
                            st = p.stat()
                        except OSError:
                            continue
                        if st.st_size < 4096 or (now_s - st.st_mtime) < closed_cd:
                            continue
                        cb_ep = ts_cb.timestamp()
                        if any((le - 2.0) <= cb_ep <= (le + 55.0) for le in lan_epochs):
                            continue
                        rel = f"{c['cloud_backfill_dir']}/{p.relative_to(cb_dir).as_posix()}"
                        if rel not in seen_rel:
                            seen_rel[rel] = {
                                "source_rel": rel, "source_root": "cctv", "day": day_str, "camera": cid,
                                "wall_ts": ts_cb.replace(tzinfo=None).isoformat(timespec="seconds"),
                                "size": st.st_size, "mtime": st.st_mtime,
                            }

        return self.db.upsert_discovered_files(list(seen_rel.values()))

    def _cloud_backfill_bounds(self, cam_cfg: dict[str, Any], today_d: dt.date) -> tuple[str, str]:
        start_d = str(cam_cfg.get("cloud_backfill_start_day") or "").strip() or (today_d - dt.timedelta(days=14)).isoformat()
        end_d = str(cam_cfg.get("cloud_backfill_end_day") or "").strip() or today_d.isoformat()
        return start_d, end_d

    def sync_cloud_backfill_window(self) -> list[str]:
        """Watch cloud_backfill_dir (e.g. Aqara_Cloud_G3 11-11~11-17) for newly settled clips and reopen completed days."""
        cams = self._effective_cameras()
        today_d = _shanghai_now(self.cfg).date()
        now_s = time.time()
        closed_cd = float(self.cfg["scheduler"].get("closed_file_cooldown_seconds", 900))
        p0_set = {today_d.isoformat(), (today_d - dt.timedelta(days=1)).isoformat()}
        reopened: list[str] = []

        for cid, c in cams.items():
            if not c.get("enabled", True) or not c.get("cloud_backfill_dir"):
                continue
            cb_dir = self.cctv_root / str(c["cloud_backfill_dir"])
            if not cb_dir.is_dir():
                continue
            start_d, end_d = self._cloud_backfill_bounds(c, today_d)
            ready_cnt = 0
            max_mtime_ns = 0
            cand_days: set[str] = set()
            for p in _iter_cloud_backfill_mp4s(cb_dir):
                ts_cb = _parse_cloud_g3_ts(p, self.tz)
                if not ts_cb:
                    continue
                d_cb = ts_cb.strftime("%Y-%m-%d")
                if d_cb < start_d or d_cb > end_d:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                if st.st_size < 4096 or (now_s - st.st_mtime) < closed_cd:
                    continue
                ready_cnt += 1
                max_mtime_ns = max(max_mtime_ns, int(st.st_mtime_ns))
                cand_days.add(d_cb)

            sig = (ready_cnt, max_mtime_ns)
            first_sync = cid not in self._cloud_backfill_sig
            if self._cloud_backfill_sig.get(cid) == sig:
                continue
            self._cloud_backfill_sig[cid] = sig

            scan_days = set(cand_days)
            if first_sync:
                try:
                    sd = dt.date.fromisoformat(start_d)
                    ed = min(dt.date.fromisoformat(end_d), today_d)
                    while sd <= ed:
                        scan_days.add(sd.isoformat())
                        sd += dt.timedelta(days=1)
                except ValueError:
                    pass

            for d_str in sorted(scan_days, reverse=True):
                ins = self.discover_day_files(d_str)
                if ins <= 0:
                    continue
                self._exhausted_days.discard(d_str)
                sum_path = self.data_root / "days" / d_str / "summary.json"
                if sum_path.is_file():
                    try:
                        s_obj = json.loads(sum_path.read_text(encoding="utf-8"))
                        if isinstance(s_obj, dict) and s_obj.get("status") == "done":
                            s_obj["status"] = "preview"
                            s_obj["updated_at"] = dt.datetime.now(self.tz).isoformat(timespec="seconds")
                            tmp_s = sum_path.parent / f".summary_{os.getpid()}.tmp"
                            tmp_s.write_text(json.dumps(s_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                            os.replace(tmp_s, sum_path)
                            self.db.upsert_day_summary(d_str, "P0_LIVE" if d_str in p0_set else "P1_BACKFILL", "preview", s_obj)
                    except Exception:
                        pass
                reopened.append(d_str)
        return sorted(set(reopened))

    def _resolve_source_path(self, row: dict[str, Any]) -> Path | None:
        primary = (self.cctv_root if row.get("source_root", "cctv") == "cctv" else self.vault_root) / row["source_rel"]
        if primary.is_file():
            return primary
        alt = (self.vault_root if row.get("source_root", "cctv") == "cctv" else self.cctv_root) / row["source_rel"]
        return alt if alt.is_file() else None

    def _process_one_file(self, row: dict[str, Any], cams: dict[str, Any]) -> None:
        rel = row["source_rel"]
        cam = row["camera"]
        day = row["day"]
        c_cfg = cams.get(cam, {})
        src = self._resolve_source_path(row)
        if src is None:
            self.db.mark_file_done(rel, {"status": "missing", "error": "source missing"})
            return

        info = med.probe_media(self.fp, str(src))
        if info["duration"] <= 0.5 or info["width"] <= 0:
            self.db.mark_file_done(rel, {"status": "error", "error": "unreadable_media"})
            return

        mcfg = self.cfg["media"]
        hour_int = 12
        if row.get("wall_ts") and len(row["wall_ts"]) >= 13:
            try:
                hour_int = int(row["wall_ts"][11:13])
            except ValueError:
                pass
        role = c_cfg.get("role", "study_or_room")
        d_lo, d_hi = c_cfg.get("day_hours", [8, 20])
        is_day_hr = int(d_lo) <= hour_int < int(d_hi)
        is_bedside_night = (role == "bedside_crib") and not is_day_hr

        day_dir = self.data_root / "days" / day
        stem = src.stem
        t2_rel = f"days/{day}/t2/{cam}/{stem}.mp4"
        t2_path = self.data_root / t2_rel
        scan_bursts = bool(info["has_audio"]) and (6 <= hour_int <= 22)

        if info["has_audio"] and float(info["duration"]) >= 25.0:
            aud_pool = ThreadPoolExecutor(max_workers=1)
            fut_aud = aud_pool.submit(
                det.analyze_audio_and_bursts,
                self.ff, str(src), rel, cam, self.yamnet, scan_bursts=scan_bursts,
            )
            try:
                vis = med.stream_1fps_t2_and_analyze(
                    self.ff, str(src), t2_path, info["width"], info["height"], info["duration"],
                    self.yolo, det.PERSON_CONF, self.use_vaapi, self.vaapi_dev,
                    int(mcfg.get("t2_height", 720)), int(mcfg.get("t2_qp", 30)),
                    yolo_min_motion=0.005 if is_bedside_night else 0.003,
                    yolo_interval=0 if is_bedside_night else 5,
                )
                if vis is None:
                    try:
                        fut_aud.result()
                    except Exception:
                        pass
                    self.db.mark_file_done(rel, {"status": "error", "error": "t2_decode_failed"})
                    return
                aud = fut_aud.result()
            finally:
                aud_pool.shutdown(wait=True)
        else:
            vis = med.stream_1fps_t2_and_analyze(
                self.ff, str(src), t2_path, info["width"], info["height"], info["duration"],
                self.yolo, det.PERSON_CONF, self.use_vaapi, self.vaapi_dev,
                int(mcfg.get("t2_height", 720)), int(mcfg.get("t2_qp", 30)),
                yolo_min_motion=0.005 if is_bedside_night else 0.003,
                yolo_interval=0 if is_bedside_night else 5,
            )
            if vis is None:
                self.db.mark_file_done(rel, {"status": "error", "error": "t2_decode_failed"})
                return
            aud = (
                det.analyze_audio_and_bursts(self.ff, str(src), rel, cam, self.yamnet, scan_bursts=scan_bursts)
                if info["has_audio"]
                else {"active_secs": [], "cry_or_baby_secs": [], "max_child_voc": 0.0, "max_laughter": 0.0, "max_cry": 0.0, "bursts": []}
            )

        if not t2_path.is_file() or t2_path.stat().st_size == 0:
            self.db.mark_file_done(rel, {"status": "error", "error": "t2_output_missing"})
            return

        n = vis["n_frames"]
        boxes = vis["boxes"]
        gmot = vis["gmot"]
        bmot = vis["bmot"]
        pconf = vis["pconf"]
        cands = vis["cands"]

        audio_mask = np.zeros(n, dtype=bool)
        for s_i in aud["active_secs"]:
            if 0 <= s_i < n:
                audio_mask[s_i] = True
        cry_mask = np.zeros(n, dtype=bool)
        for s_i in aud["cry_or_baby_secs"]:
            if 0 <= s_i < n:
                cry_mask[s_i] = True

        static_suppressed = 0
        night_sleep = 0
        t1_mode = c_cfg.get("t1_mode", "motion_or_person")

        if role == "bedside_crib":
            sup_mask = det.compute_bedside_static_mask(boxes, bmot, min_run_s=12)
            static_suppressed = int(sup_mask.sum())
            if is_day_hr:
                active = (((pconf >= det.PERSON_CONF) & (~sup_mask) & (bmot >= det.BOX_MOTION)) | audio_mask)
            else:
                active = cry_mask | ((pconf >= 0.45) & (~sup_mask) & (bmot >= 0.06) & audio_mask)
                night_sleep = int((~active).sum())
        else:
            person_moving = (pconf >= det.PERSON_CONF) & (bmot >= det.BOX_MOTION)
            unexplained = (pconf < det.PERSON_CONF) & (gmot >= det.GLOBAL_MOTION)
            active = person_moving | unexplained | audio_mask

        segs = det.segments_from_mask(active, info["duration"]) if active.any() else []
        person_s = int((pconf >= det.PERSON_CONF).sum())

        t1_rel = None
        t1_bytes = 0
        if segs and t1_mode != "verified_child_only":
            t1_rel = f"days/{day}/t1/{cam}/{stem}.mp4"
            t1_path = self.data_root / t1_rel
            ok_t1, _ = med.encode_t1_segments(
                self.ff, str(src), t1_path, segs, info["duration"], info["has_audio"],
                self.use_vaapi, self.vaapi_dev, int(c_cfg.get("t1_res", mcfg.get("t1_height", 720))),
                int(mcfg.get("t1_fps", 15)), int(mcfg.get("t1_qp", 28)),
                pts_jump=bool(info.get("pts_jump")),
            )
            t1_bytes = t1_path.stat().st_size if ok_t1 and t1_path.is_file() else 0
            if t1_bytes == 0:
                self.db.mark_file_done(rel, {"status": "error", "error": "t1_encode_failed", "t2_rel": t2_rel, "t2_bytes": t2_path.stat().st_size})
                return

        self.db.mark_file_done(rel, {
            "status": "done",
            "duration": info["duration"],
            "width": info["width"],
            "height": info["height"],
            "has_audio": info["has_audio"],
            "active_s": int(active.sum()),
            "person_s": person_s,
            "static_suppressed_s": static_suppressed,
            "night_sleep_s": night_sleep,
            "t1_mode": t1_mode,
            "t1_rel": t1_rel,
            "t1_bytes": t1_bytes,
            "t2_rel": t2_rel,
            "t2_bytes": t2_path.stat().st_size if t2_path.is_file() else 0,
            "max_box_score": max((c["score"] for c in cands), default=0.0),
            "audio_summary": {
                "max_child_voc": aud["max_child_voc"],
                "max_laughter": aud["max_laughter"],
                "max_cry": aud["max_cry"],
                "segments": segs,
            },
            "candidates": cands,
            "bursts": aud["bursts"],
        })

    def _set_job(self, phase: str, day: str = "", step: str = "", detail: str = "", camera: str = "", last_file: str = "") -> None:
        with self._job_lock:
            cur_st = self._curator_status
            shown_detail = f"{detail} | 后台审片: {cur_st}" if (cur_st and step == "并发视频分析") else detail
            self.db.set_state("active_job", {
                "phase": phase,
                "day": day,
                "step": step,
                "detail": shown_detail,
                "curator_status": cur_st,
                "camera": camera,
                "last_file": last_file,
                "updated_at": dt.datetime.now(self.tz).isoformat(timespec="seconds"),
                "host_mem_kib": read_host_mem_available_kib(),
            })

    def _update_curate_progress(self, tier: str, day: str, step: str, detail: str, background: bool) -> None:
        if background:
            with self._job_lock:
                self._curator_status = f"{day} · {step}"
        else:
            self._set_job(tier, day=day, step=step, detail=detail)

    def _extract_cloud_frames(
        self, row: dict[str, Any], src: Path, offsets: list[float], cams: dict[str, Any], max_edge: int | None = None,
    ) -> list[bytes]:
        cam_cfg = cams.get(row["camera"])
        if not cam_cfg:
            return []
        wall_ts = row.get("wall_ts")
        if cam_cfg.get("role") == "bedside_crib" and cam_cfg.get("family") == "xiaomi_bundle":
            # Older rows may contain a copy time, an end time or a noon placeholder.
            match = re.match(r"^00_(\d{14})(?:_|$)", src.stem)
            if not match:
                return []
            try:
                wall_ts = dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").isoformat()
            except ValueError:
                return []
        if not is_cloud_allowed(cam_cfg, wall_ts, offsets):
            return []
        edge = int(max_edge) if max_edge is not None else 768
        return extract_jpegs_ram(self.ff, str(src), offsets, max_edge=edge)

    def _handle_vlm_refusal(
        self,
        kind: str,
        review_id: str,
        day: str,
        row: dict[str, Any],
        start_s: float,
        end_s: float,
        ans: dict[str, Any],
        box_fallback: list[float] | None = None,
    ) -> str | dict[str, Any] | None:
        """Route content-policy VLM refusals to pending_review unless already decided by human reviewer."""
        rev = self.db.get_vlm_review(review_id)
        if rev and rev.get("status") == "approved":
            cat = str(rev.get("suggested_category") or ans.get("suggested_category") or "daily_care")
            pbox = med.estimate_pelvic_privacy_box(box_fallback)
            return {
                "child_clear": "yes",
                "face_or_expression_clear": "yes",
                "composition_good": "yes",
                "adult_back_or_awkward": "no",
                "osd_box_on_child": "no",
                "categories": [cat],
                "independence": "assisted",
                "excitement_score": 4,
                "photo_score": 4,
                "child_box": box_fallback or [0.0, 0.0, 0.0, 0.0],
                "privacy_exposed": "yes",
                "privacy_box": pbox,
                "review_status": "approved",
                "vlm_refused": True,
                "caption_zh": str(rev.get("reviewer_note") or "人工复核通过 · 温馨亲子日常照护"),
                "_provider": ans.get("_provider") or "human_review",
                "_model": ans.get("_model") or "vlm_refusal_approved",
            }
        if rev and rev.get("status") == "rejected":
            return None
        self.db.upsert_vlm_review({
            "review_id": review_id,
            "day": day,
            "camera": row["camera"],
            "kind": kind,
            "source_rel": row["source_rel"],
            "start_s": float(start_s),
            "end_s": float(end_s),
            "suggested_category": str(ans.get("suggested_category") or "daily_care"),
            "status": "pending_review",
            "reason": str(ans.get("refusal_reason") or "content_policy_blocked"),
            "meta": {
                "provider": ans.get("_provider"),
                "model": ans.get("_model") or "vlm_refusal_pending",
                "child_box": box_fallback or ans.get("child_box"),
            },
        })
        return "pending_review"

    @staticmethod
    def _effective_row_wall_ts(row: dict[str, Any], cam_cfg: dict[str, Any]) -> str | None:
        wall_ts = row.get("wall_ts")
        if cam_cfg.get("role") == "bedside_crib" and cam_cfg.get("family") == "xiaomi_bundle":
            stem = Path(str(row.get("source_rel") or "")).stem
            match = re.match(r"^00_(\d{14})(?:_|$)", stem)
            if not match:
                return None
            try:
                return dt.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").isoformat()
            except ValueError:
                return None
        return str(wall_ts) if wall_ts else None

    @staticmethod
    def _parse_wall_epoch(wall_ts: str | None, offset_s: float = 0.0) -> float | None:
        if not wall_ts:
            return None
        try:
            return dt.datetime.fromisoformat(str(wall_ts)).timestamp() + float(offset_s)
        except ValueError:
            return None

    def _expand_kitchen_candidates(self, kr: dict[str, Any]) -> list[tuple[float, list[float] | None, float]]:
        cands = json.loads(kr.get("candidates_json") or "[]")
        aud_s = json.loads(kr.get("audio_summary_json") or "{}")
        dur = float(kr.get("duration") or 0.0)
        pts: list[tuple[float, list[float] | None, float]] = []
        for c in sorted(cands, key=lambda x: (-float(x.get("score") or 0.0), float(x.get("t") or 0.0))):
            t_c = round(max(0.0, float(c.get("t") or 0.0)), 1)
            if any(abs(t_c - pt[0]) < 4.0 for pt in pts):
                continue
            pts.append((t_c, c.get("box"), float(c.get("score") or 0.0)))
        if not pts and aud_s.get("max_child_voc", 0.0) >= 0.25:
            pts.append((round(max(0.5, dur * 0.4), 1), None, float(aud_s.get("max_child_voc", 0.0))))
        return pts

    def finalize_day(
        self,
        day: str,
        tier: str,
        final: bool = True,
        *,
        background: bool = False,
        force_curate: bool = False,
    ) -> dict[str, Any]:
        """Run Cloud VLM (Gemini 4 + Qwen4-Max parallel), Kitchen T1 gating, ephemeral ASR worker, T0 promotion, daily highlight, photos, and timelapses."""
        rows = [r for r in self.db.get_day_rows(day) if r["status"] == "done"]
        if not rows:
            return {}
        cams = self._effective_cameras()
        day_dir = self.data_root / "days" / day
        day_dir.mkdir(parents=True, exist_ok=True)
        mcfg = self.cfg["media"]
        vlm_cfg = self.cfg.get("vlm", {})

        prev_summary: dict[str, Any] = {}
        sum_path = day_dir / "summary.json"
        if sum_path.is_file():
            try:
                loaded = json.loads(sum_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    prev_summary = loaded
            except Exception:
                prev_summary = {}
        prev_photos: list[dict[str, Any]] = [
            p for p in (prev_summary.get("photos") or [])
            if isinstance(p, dict)
            and not is_unconfirmed_refusal(p)
            and p.get("full") and p.get("framed")
            and (self.data_root / str(p["full"])).is_file() and (self.data_root / str(p["full"])).stat().st_size > 0
            and (self.data_root / str(p["framed"])).is_file() and (self.data_root / str(p["framed"])).stat().st_size > 0
        ]
        prev_hl = prev_summary.get("highlight_rel")
        if not (isinstance(prev_hl, str) and prev_hl and (self.data_root / prev_hl).is_file() and (self.data_root / prev_hl).stat().st_size > 0):
            prev_hl = None

        # 1. Concat daily T2 timelapses per camera
        self._update_curate_progress(tier, day, "合成全天延时", f"正在拼接 {day} 各机位 1fps 延时视频", background)
        timelapses: dict[str, str] = {}
        by_cam: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_cam.setdefault(r["camera"], []).append(r)
        prev_tl_map = prev_summary.get("timelapses") if isinstance(prev_summary.get("timelapses"), dict) else {}
        for cid, crows in by_cam.items():
            t2_paths = [self.data_root / cr["t2_rel"] for cr in crows if cr.get("t2_rel")]
            tl_name = f"timelapse_{cid}.mp4"
            tl_path = day_dir / tl_name
            expected_rel = f"days/{day}/{tl_name}"
            if len(t2_paths) == len(crows):
                can_reuse_tl = False
                if prev_tl_map.get(cid) == expected_rel and tl_path.is_file() and tl_path.stat().st_size > 1024:
                    try:
                        tl_mtime = tl_path.stat().st_mtime
                        latest_t2_mtime = max((p.stat().st_mtime for p in t2_paths if p.is_file()), default=0.0)
                        can_reuse_tl = latest_t2_mtime > 0.0 and tl_mtime >= latest_t2_mtime
                    except OSError:
                        can_reuse_tl = False
                if can_reuse_tl or med.concat_mp4_list(self.ff, t2_paths, tl_path, self.fp):
                    timelapses[cid] = expected_rel

        # 2. Persistent per-candidate Kitchen T1 gating (paginated batch_limit=14, dedup, refusal pending_review, restart-safe)
        kit_rows = [
            r for r in rows
            if not r.get("t1_rel") and r.get("t1_mode") == "verified_child_only"
        ]
        kit_rows_sorted = sorted(
            kit_rows,
            key=lambda x: (
                -(float(x.get("max_box_score") or 0.0) + float(json.loads(x.get("audio_summary_json") or "{}").get("max_child_voc", 0.0))),
                str(x.get("wall_ts") or ""),
                str(x.get("source_rel") or ""),
            ),
        )
        batch_limit = max(1, int(vlm_cfg.get("kitchen_batch_size", 14)))
        existing_kit_reviews = self.db.get_kitchen_reviews_for_day(day)
        vlm_rev_by_id = {rv["review_id"]: rv for rv in self.db.list_vlm_reviews(status="all", day=day, limit=500)}
        row_by_rel = {r["source_rel"]: r for r in rows}

        seen_wall_slots: list[tuple[str, float, str]] = []
        for rv in existing_kit_reviews.values():
            rm = row_by_rel.get(rv["source_rel"])
            if rm and int(rv.get("source_size") or 0) == int(rm.get("size") or 0) and abs(float(rv.get("source_mtime") or 0.0) - float(rm.get("mtime") or 0.0)) <= 2.0:
                wep = self._parse_wall_epoch(rm.get("wall_ts"), float(rv.get("t_s") or 0.0))
                if wep is not None:
                    seen_wall_slots.append((rm["camera"], wep, str(rv["cand_key"])))

        kit_cands: list[tuple[str, dict[str, Any], Path, float, list[float] | None, list[Any], dict[str, Any]]] = []
        manual_approved_kit: list[tuple[str, dict[str, Any], Path, float, dict[str, Any]]] = []
        seen_keys_in_scan: set[str] = set()
        encoded_srcs: set[str] = set()
        kit_reviewed_this_pass = 0

        for kr in kit_rows_sorted:
            cands = json.loads(kr.get("candidates_json") or "[]")
            aud_s = json.loads(kr.get("audio_summary_json") or "{}")
            src = self._resolve_source_path(kr)
            if not src:
                continue
            pts = self._expand_kitchen_candidates(kr)
            for t0, box0, _sc0 in pts:
                cand_key = f"kit|{kr['source_rel']}|{t0:.1f}"
                if cand_key in seen_keys_in_scan:
                    continue
                seen_keys_in_scan.add(cand_key)
                prev_rev = existing_kit_reviews.get(cand_key)
                same_ver = bool(
                    prev_rev
                    and int(prev_rev.get("source_size") or 0) == int(kr.get("size") or 0)
                    and abs(float(prev_rev.get("source_mtime") or 0.0) - float(kr.get("mtime") or 0.0)) <= 2.0
                )
                if same_ver and prev_rev and prev_rev.get("status") in ("confirmed", "rejected", "pending_review"):
                    man_rev = vlm_rev_by_id.get(cand_key)
                    if prev_rev.get("status") == "pending_review" and man_rev and man_rev.get("status") == "approved":
                        manual_approved_kit.append((cand_key, kr, src, t0, {"child_clear": "yes", "review_status": "approved", "child_box": box0, "caption_zh": man_rev.get("reviewer_note") or "人工复核确认"}))
                    continue

                wep = self._parse_wall_epoch(kr.get("wall_ts"), t0)
                if wep is not None:
                    dup_of = next((ck for cam0, ep0, ck in seen_wall_slots if cam0 == kr["camera"] and abs(ep0 - wep) < 4.0), None)
                    if dup_of:
                        self.db.upsert_kitchen_review({
                            "cand_key": cand_key, "day": day, "camera": kr["camera"],
                            "source_rel": kr["source_rel"], "t_s": t0,
                            "source_size": kr["size"], "source_mtime": kr["mtime"],
                            "frame_sha256": None, "status": "rejected",
                            "provider": "dedup", "model": "wall_ts_dedup",
                            "response": {"child_clear": "no", "dedup_of": dup_of},
                        })
                        existing_kit_reviews[cand_key] = {"cand_key": cand_key, "source_size": kr["size"], "source_mtime": kr["mtime"], "status": "rejected"}
                        kit_reviewed_this_pass += 1
                        continue
                    seen_wall_slots.append((kr["camera"], wep, cand_key))

                if len(kit_cands) < batch_limit:
                    kit_cands.append((cand_key, kr, src, t0, box0, cands, aud_s))

        # Encode any human-approved pending kitchen reviews first
        for cand_key, kr, src, t0, man_ans in manual_approved_kit:
            if kr["source_rel"] in encoded_srcs:
                self.db.upsert_kitchen_review({
                    "cand_key": cand_key, "day": day, "camera": kr["camera"],
                    "source_rel": kr["source_rel"], "t_s": t0,
                    "source_size": kr["size"], "source_mtime": kr["mtime"],
                    "status": "rejected", "provider": "dedup", "model": "same_clip_t1_already_encoded",
                    "response": {"dedup_of_source": kr["source_rel"]},
                })
                continue
            aud_s = json.loads(kr.get("audio_summary_json") or "{}")
            cands = json.loads(kr.get("candidates_json") or "[]")
            segs = aud_s.get("segments") or [[max(0.0, t0 - 2.0), min(kr["duration"], t0 + 4.0)]]
            t1_rel = f"days/{day}/t1/{kr['camera']}/{src.stem}.mp4"
            ok_t1, _ = med.encode_t1_segments(
                self.ff, str(src), self.data_root / t1_rel, segs, kr["duration"], bool(kr["has_audio"]),
                self.use_vaapi, self.vaapi_dev, int(mcfg.get("t1_height", 720)), int(mcfg.get("t1_fps", 15)), int(mcfg.get("t1_qp", 28)),
            )
            if ok_t1 and (self.data_root / t1_rel).is_file() and (self.data_root / t1_rel).stat().st_size > 0:
                encoded_srcs.add(kr["source_rel"])
                kr["t1_rel"] = t1_rel
                kr["t1_bytes"] = (self.data_root / t1_rel).stat().st_size
                self.db.mark_file_done(kr["source_rel"], {
                    **kr, "t1_mode": "verified_child_confirmed",
                    "audio_summary": {**aud_s, "kitchen_vlm": man_ans},
                    "candidates": cands, "bursts": json.loads(kr.get("bursts_json") or "[]"),
                })
                self.db.upsert_kitchen_review({
                    "cand_key": cand_key, "day": day, "camera": kr["camera"],
                    "source_rel": kr["source_rel"], "t_s": t0,
                    "source_size": kr["size"], "source_mtime": kr["mtime"],
                    "status": "confirmed", "provider": "human_review", "model": "vlm_refusal_approved",
                    "response": man_ans,
                })
                kit_reviewed_this_pass += 1

        if kit_cands:
            self._update_curate_progress(tier, day, "厨房宝宝核验", f"多模型自领任务核验 {len(kit_cands)} 段厨房画面（Hydrogen / Qwen4 / DeepSeek-4.1 等）", background)

            def _prep_kit(item: tuple[int, tuple[str, dict[str, Any], Path, float, list[float] | None, list[Any], dict[str, Any]]]):
                idx_k, (ck_i, kr_i, src_i, t0_i, box_i, cands_i, aud_i) = item
                offsets = [max(0.2, t0_i - 0.8), t0_i, min(max(0.5, kr_i["duration"] - 0.5), t0_i + 0.9)]
                imgs = self._extract_cloud_frames(kr_i, src_i, offsets, cams)
                if len(imgs) < 2:
                    return (idx_k, ck_i, kr_i, src_i, t0_i, box_i, cands_i, aud_i, [], "")
                raw_len = sum(len(b) for b in imgs)
                fsha = hashlib.sha256(b"".join(imgs)).hexdigest() if raw_len >= 512 else ""
                return (idx_k, ck_i, kr_i, src_i, t0_i, box_i, cands_i, aud_i, imgs, fsha)

            with ThreadPoolExecutor(max_workers=min(4, len(kit_cands))) as kpool:
                prepped_kit_all = list(kpool.map(_prep_kit, list(enumerate(kit_cands))))

            batch_sha_seen: dict[str, str] = {}
            prepped_kit: list[tuple[int, str, dict[str, Any], Path, float, list[float] | None, list[Any], dict[str, Any], list[bytes], str]] = []
            for item_p in prepped_kit_all:
                idx_k, cand_key, kr, src, t0, _box0, _cands, _aud_s, imgs, fsha = item_p
                if len(imgs) < 2:
                    self.db.upsert_kitchen_review({
                        "cand_key": cand_key, "day": day, "camera": kr["camera"],
                        "source_rel": kr["source_rel"], "t_s": t0,
                        "source_size": kr["size"], "source_mtime": kr["mtime"],
                        "frame_sha256": None, "status": "rejected",
                        "provider": "local_prep", "model": "frame_unavailable",
                        "response": {"child_clear": "no", "reason": "insufficient_frames"},
                    })
                    kit_reviewed_this_pass += 1
                    continue
                if fsha:
                    prior_sha_row = self.db.get_kitchen_review_by_frame_sha(day, fsha)
                    dup_target = batch_sha_seen.get(fsha) or (prior_sha_row.get("cand_key") if prior_sha_row else None)
                    if dup_target:
                        dup_status = "pending_review" if (prior_sha_row and prior_sha_row.get("status") == "pending_review") else "rejected"
                        self.db.upsert_kitchen_review({
                            "cand_key": cand_key, "day": day, "camera": kr["camera"],
                            "source_rel": kr["source_rel"], "t_s": t0,
                            "source_size": kr["size"], "source_mtime": kr["mtime"],
                            "frame_sha256": fsha, "status": dup_status,
                            "provider": "dedup", "model": "frame_sha256_dedup",
                            "response": {"child_clear": "no", "dedup_of": dup_target},
                        })
                        kit_reviewed_this_pass += 1
                        continue
                    batch_sha_seen[fsha] = cand_key
                prepped_kit.append(item_p)

            kit_face_active = bool(self.face_service.enabled)
            kit_ref_bytes: bytes | None = None
            if kit_face_active and bool((self.face_service.fcfg or {}).get("attach_reference_sheet", True)):
                try:
                    kit_ref_bytes = self.face_service.get_reference_sheet(day)
                except Exception:
                    kit_ref_bytes = None
            kit_ref_imgs = [("家庭成员识别对照卡", kit_ref_bytes)] if kit_ref_bytes else None

            kit_tasks = []
            kit_face_matches: dict[int, dict[str, Any] | None] = {}
            for idx_k, cand_key, _kr_i, _src_i, _t0_i, _b0, _cands_i, _aud_i, imgs, _fsha in prepped_kit:
                k_face_m = None
                if kit_face_active:
                    try:
                        k_face_m = self.face_service.identify_images(imgs, day=day, child_box=_b0)
                    except Exception:
                        k_face_m = None
                kit_face_matches[idx_k] = k_face_m
                fam_blk_k = (
                    "\n\n" + self.face_service.build_vlm_family_prompt_block(day, k_face_m, has_ref_sheet=bool(kit_ref_imgs))
                    if kit_face_active
                    else ""
                )
                k_prompt = f"{PROMPT_KITCHEN_CHILD}{fam_blk_k}"
                if kit_ref_imgs:
                    kit_tasks.append((idx_k, cand_key, imgs, k_prompt, kit_ref_imgs))
                else:
                    kit_tasks.append((idx_k, cand_key, imgs, k_prompt))
            kit_answers = self.vlm.ask_batch(kit_tasks, tier="fast") if kit_tasks else {}

            for idx_k, cand_key, kr, src, t0, box0, cands, aud_s, _imgs, fsha in prepped_kit:
                ans = kit_answers.get(idx_k)
                if not ans:
                    continue
                if is_unconfirmed_refusal(ans):
                    rev_res = self._handle_vlm_refusal(
                        "kitchen", cand_key, day, kr, max(0.0, t0 - 1.5), min(float(kr["duration"]), t0 + 2.5),
                        ans, box_fallback=box0,
                    )
                    if rev_res == "pending_review":
                        self.db.upsert_kitchen_review({
                            "cand_key": cand_key, "day": day, "camera": kr["camera"],
                            "source_rel": kr["source_rel"], "t_s": t0,
                            "source_size": kr["size"], "source_mtime": kr["mtime"],
                            "frame_sha256": fsha or None, "status": "pending_review",
                            "provider": ans.get("_provider"), "model": ans.get("_model") or "vlm_refusal_pending",
                            "response": ans,
                        })
                        kit_reviewed_this_pass += 1
                        continue
                    if rev_res is None:
                        self.db.upsert_kitchen_review({
                            "cand_key": cand_key, "day": day, "camera": kr["camera"],
                            "source_rel": kr["source_rel"], "t_s": t0,
                            "source_size": kr["size"], "source_mtime": kr["mtime"],
                            "frame_sha256": fsha or None, "status": "rejected",
                            "provider": "human_review", "model": "vlm_refusal_rejected",
                            "response": {"child_clear": "no"},
                        })
                        kit_reviewed_this_pass += 1
                        continue
                    ans = rev_res

                if ans.get("child_clear") in ("yes", "partial"):
                    k_face_m = kit_face_matches.get(idx_k)
                    if ans.get("caption_zh"):
                        clean_k_cap, kit_people = self.face_service.sanitize_text(
                            str(ans["caption_zh"]), day=day, match_res=k_face_m, vlm_ans=ans,
                        )
                        ans["caption_zh"] = clean_k_cap
                        if kit_people:
                            ans["_people"] = kit_people
                    if kr["source_rel"] in encoded_srcs:
                        self.db.upsert_kitchen_review({
                            "cand_key": cand_key, "day": day, "camera": kr["camera"],
                            "source_rel": kr["source_rel"], "t_s": t0,
                            "source_size": kr["size"], "source_mtime": kr["mtime"],
                            "frame_sha256": fsha or None, "status": "rejected",
                            "provider": "dedup", "model": "same_clip_t1_already_encoded",
                            "response": ans,
                        })
                        kit_reviewed_this_pass += 1
                        continue
                    segs = aud_s.get("segments") or [[max(0.0, t0 - 2.0), min(kr["duration"], t0 + 4.0)]]
                    t1_rel = f"days/{day}/t1/{kr['camera']}/{src.stem}.mp4"
                    ok_t1, _ = med.encode_t1_segments(
                        self.ff, str(src), self.data_root / t1_rel, segs, kr["duration"], bool(kr["has_audio"]),
                        self.use_vaapi, self.vaapi_dev, int(mcfg.get("t1_height", 720)), int(mcfg.get("t1_fps", 15)), int(mcfg.get("t1_qp", 28)),
                    )
                    if ok_t1 and (self.data_root / t1_rel).is_file() and (self.data_root / t1_rel).stat().st_size > 0:
                        encoded_srcs.add(kr["source_rel"])
                        kr["t1_rel"] = t1_rel
                        kr["t1_bytes"] = (self.data_root / t1_rel).stat().st_size
                        self.db.mark_file_done(kr["source_rel"], {
                            **kr, "t1_mode": "verified_child_confirmed",
                            "audio_summary": {**aud_s, "kitchen_vlm": ans},
                            "candidates": cands, "bursts": json.loads(kr.get("bursts_json") or "[]"),
                        })
                        self.db.upsert_kitchen_review({
                            "cand_key": cand_key, "day": day, "camera": kr["camera"],
                            "source_rel": kr["source_rel"], "t_s": t0,
                            "source_size": kr["size"], "source_mtime": kr["mtime"],
                            "frame_sha256": fsha or None, "status": "confirmed",
                            "provider": ans.get("_provider"), "model": ans.get("_model"),
                            "response": ans,
                        })
                        kit_reviewed_this_pass += 1
                else:
                    self.db.upsert_kitchen_review({
                        "cand_key": cand_key, "day": day, "camera": kr["camera"],
                        "source_rel": kr["source_rel"], "t_s": t0,
                        "source_size": kr["size"], "source_mtime": kr["mtime"],
                        "frame_sha256": fsha or None, "status": "rejected",
                        "provider": ans.get("_provider"), "model": ans.get("_model"),
                        "response": ans,
                    })
                    kit_reviewed_this_pass += 1

        # Recompute remaining unreviewed kitchen candidates accurately after persisting this batch
        final_kit_reviews = self.db.get_kitchen_reviews_for_day(day)
        kit_unreviewed_remaining = 0
        for kr in [r for r in self.db.get_day_rows(day) if r["status"] == "done" and not r.get("t1_rel") and r.get("t1_mode") == "verified_child_only"]:
            src = self._resolve_source_path(kr)
            if not src:
                continue
            for t0, _b0, _sc0 in self._expand_kitchen_candidates(kr):
                ck = f"kit|{kr['source_rel']}|{t0:.1f}"
                prev_r = final_kit_reviews.get(ck)
                same_v = bool(
                    prev_r
                    and int(prev_r.get("source_size") or 0) == int(kr.get("size") or 0)
                    and abs(float(prev_r.get("source_mtime") or 0.0) - float(kr.get("mtime") or 0.0)) <= 2.0
                    and prev_r.get("status") in ("confirmed", "rejected", "pending_review")
                )
                if not same_v:
                    kit_unreviewed_remaining += 1

        has_complete_preview = bool(
            prev_hl
            and isinstance(prev_summary.get("daily_story"), dict)
            and (prev_summary.get("daily_story") or {}).get("shots")
            and int(prev_summary.get("done_files") or 0) == len(rows)
        )
        if not force_curate and kit_unreviewed_remaining > 0 and has_complete_preview:
            st_mid = self.db.count_day_stats(day)
            pending_rev_mid = len(self.db.list_vlm_reviews(status="pending_review", day=day, limit=200))
            summary_mid = {
                **prev_summary,
                "day": day,
                "tier": tier,
                "status": "preview",
                "total_files": st_mid["total"],
                "done_files": st_mid["done"],
                "orig_bytes": st_mid["orig_bytes"],
                "t1_bytes": st_mid["t1_bytes"],
                "t2_bytes": st_mid["t2_bytes"],
                "highlight_rel": prev_hl,
                "photos": prev_photos,
                "timelapses": timelapses,
                "missing_timelapse_cameras": sorted(by_cam.keys() - timelapses.keys()),
                "pending_review_count": pending_rev_mid,
                "kitchen_reviewed_this_pass": kit_reviewed_this_pass,
                "kitchen_unreviewed_remaining": kit_unreviewed_remaining,
                "updated_at": dt.datetime.now(self.tz).isoformat(timespec="seconds"),
            }
            sum_tmp = day_dir / f".summary_{os.getpid()}.tmp"
            sum_tmp.write_text(json.dumps(summary_mid, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(sum_tmp, sum_path)
            self.db.upsert_day_summary(day, tier, "preview", summary_mid)
            return summary_mid

        # 3. Curate Today/Day's Selected Photos via work-stealing fast pool (all non-Carbon models)
        photo_pool: list[dict[str, Any]] = []
        for r in rows:
            c_cfg = cams.get(r["camera"], {})
            if not c_cfg.get("daytime_photos", True):
                continue
            eff_wts = self._effective_row_wall_ts(r, c_cfg)
            for pc in json.loads(r.get("candidates_json") or "[]"):
                if pc.get("luma", 0) >= 70 and pc.get("sat", 0) >= 16 and pc.get("sharp", 0) >= 35:
                    if not is_cloud_allowed(c_cfg, eff_wts, [float(pc.get("t") or 0.0)]):
                        continue
                    ep = self._parse_wall_epoch(eff_wts, float(pc.get("t") or 0.0))
                    photo_pool.append({**pc, "row": r, "_eff_wts": eff_wts, "_ep": ep})
        photo_pool.sort(key=lambda x: -x["score"])
        spread_ph: list[dict[str, Any]] = []
        for pc in photo_pool:
            ep1 = pc.get("_ep")
            if ep1 is not None and any(
                q["row"].get("camera") == pc["row"]["camera"]
                and q.get("_ep") is not None
                and abs(ep1 - q["_ep"]) < 180.0
                for q in spread_ph
            ):
                continue
            spread_ph.append(pc)
            if len(spread_ph) >= 24:
                break

        face_active = bool(self.face_service.enabled)
        ref_sheet_bytes: bytes | None = None
        if face_active and bool((self.face_service.fcfg or {}).get("attach_reference_sheet", True)):
            try:
                ref_sheet_bytes = self.face_service.get_reference_sheet(day)
            except Exception:
                ref_sheet_bytes = None
        ref_imgs = [("家庭成员识别对照卡", ref_sheet_bytes)] if ref_sheet_bytes else None
        hi_res_edge = int((self.face_service.fcfg or {}).get("hi_res_edge", 1920))
        use_hires_face = bool(face_active and self.face_service.has_enrolled_members(day))

        vlm_photos: list[dict[str, Any]] = []
        vlm_photos_3plus: list[dict[str, Any]] = []
        if spread_ph:
            self._update_curate_progress(tier, day, "精选昼间照片", f"多模型自领任务筛选 {len(spread_ph)} 张高清候选照片（Hydrogen / Qwen4 / DeepSeek-4.1 等）", background)

            def _prep_photo(item: tuple[int, dict[str, Any]]):
                idx_p, pc = item
                r_p = pc["row"]
                src_p = self._resolve_source_path(r_p)
                if not src_p:
                    return None
                face_match: dict[str, Any] | None = None
                if use_hires_face:
                    hires = self._extract_cloud_frames(r_p, src_p, [pc["t"]], cams, max_edge=hi_res_edge)
                    if not hires:
                        return None
                    try:
                        face_match = self.face_service.identify_images(hires, day=day, child_box=pc.get("box"))
                    except Exception:
                        face_match = None
                    imgs = dl_dir._resize_jpegs_max_edge(hires, 768)
                else:
                    imgs = self._extract_cloud_frames(r_p, src_p, [pc["t"]], cams)
                    if not imgs:
                        return None
                    if face_active:
                        try:
                            face_match = self.face_service.identify_images(imgs, day=day, child_box=pc.get("box"))
                        except Exception:
                            face_match = None
                return idx_p, pc, r_p, imgs, face_match

            with ThreadPoolExecutor(max_workers=min(4, len(spread_ph))) as phpool:
                prepped_ph = [x for x in phpool.map(_prep_photo, list(enumerate(spread_ph))) if x is not None]

            ph_tasks: list[tuple[Any, ...]] = []
            for idx_p, pc, r_p, imgs, face_match in prepped_ph:
                fam_blk = (
                    "\n\n" + self.face_service.build_vlm_family_prompt_block(day, face_match, has_ref_sheet=bool(ref_imgs))
                    if face_active
                    else ""
                )
                ph_prompt = f"{PROMPT_PHOTO}{fam_blk}"
                if ref_imgs:
                    ph_tasks.append((idx_p, f"ph|{r_p['source_rel']}|{pc['t']:.1f}", imgs, ph_prompt, ref_imgs))
                else:
                    ph_tasks.append((idx_p, f"ph|{r_p['source_rel']}|{pc['t']:.1f}", imgs, ph_prompt))
            ph_answers = self.vlm.ask_batch(ph_tasks, tier="fast") if ph_tasks else {}

            for idx_p, pc, r_p, _imgs, face_match in prepped_ph:
                ans = ph_answers.get(idx_p)
                if not ans:
                    continue
                if is_unconfirmed_refusal(ans):
                    rev_res = self._handle_vlm_refusal(
                        "photo", f"ph|{r_p['source_rel']}|{pc['t']:.1f}", day, r_p,
                        float(pc["t"]), float(pc["t"]) + 1.0, ans, box_fallback=pc.get("box"),
                    )
                    if not isinstance(rev_res, dict):
                        continue
                    ans = rev_res
                if (
                    ans.get("child_clear") == "yes"
                    and ans.get("face_or_expression_clear") == "yes"
                    and ans.get("composition_good", "yes") == "yes"
                    and ans.get("adult_back_or_awkward") == "no"
                    and ans.get("osd_box_on_child", "no") != "yes"
                ):
                    clean_cap, people_list = self.face_service.sanitize_text(
                        str(ans.get("caption_zh") or ""),
                        day=day,
                        match_res=face_match,
                        vlm_ans=ans,
                    )
                    ans_clean = {**ans, "caption_zh": clean_cap, "_people": people_list}
                    pscore = float(ans_clean.get("photo_score", 0))
                    if pscore >= 4:
                        vlm_photos.append({**pc, "vlm": ans_clean, "_people": people_list})
                    elif pscore >= 3:
                        vlm_photos_3plus.append({**pc, "vlm": ans_clean, "_people": people_list})
            if not vlm_photos and vlm_photos_3plus and any(float(p.get("zoom_x", 1.0)) >= 1.95 for p in prev_photos):
                vlm_photos = vlm_photos_3plus

        vlm_photos.sort(key=lambda p: -(float(p["vlm"].get("photo_score", 0)) * 2 + p["score"]))
        chosen_ph: list[dict[str, Any]] = []
        per_cam_ph: dict[str, int] = {}
        for vp in vlm_photos:
            cam = vp["row"]["camera"]
            if per_cam_ph.get(cam, 0) >= 5:
                continue
            chosen_ph.append(vp)
            per_cam_ph[cam] = per_cam_ph.get(cam, 0) + 1
            if len(chosen_ph) >= 10:
                break
        chosen_ph.sort(key=lambda p: p["row"].get("wall_ts") or "")

        pdir = day_dir / "photos"
        for old_st in day_dir.glob(".photos_staging_*"):
            shutil.rmtree(old_st, ignore_errors=True)
        saved_photos: list[dict[str, Any]] = list(prev_photos)
        pending_photo_stage: Path | None = None
        pending_staged_photos: list[dict[str, Any]] | None = None
        if chosen_ph:
            stage_dir = day_dir / f".photos_staging_{os.getpid()}_{int(time.time() * 1000)}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            staged_photos: list[dict[str, Any]] = []
            gen_failed = False
            try:
                for idx, vp in enumerate(chosen_ph, 1):
                    r = vp["row"]
                    src = self._resolve_source_path(r)
                    if not src:
                        gen_failed = True
                        break
                    cam = r["camera"]
                    c_cfg = cams.get(cam, {})
                    zoom = float(c_cfg.get("zoom_factor", 1.35))
                    cname = c_cfg.get("name", cam)
                    wts = vp.get("_eff_wts") or self._effective_row_wall_ts(r, c_cfg) or r.get("wall_ts") or f"{day}T12:00:00"
                    t_lbl = (dt.datetime.fromisoformat(wts) + dt.timedelta(seconds=vp["t"])).strftime("%H:%M:%S")
                    full_name = f"full_{idx:02d}_{cname}_{t_lbl.replace(':', '')}.jpg"
                    framed_name = f"framed_{idx:02d}_{cname}_{t_lbl.replace(':', '')}.jpg"
                    box = vp["vlm"].get("child_box")
                    if not box or box == [0.0, 0.0, 0.0, 0.0]:
                        box = vp.get("box")
                    pbox = _resolve_privacy_box(vp["vlm"], box)
                    f_out = stage_dir / full_name
                    fr_out = stage_dir / framed_name
                    if not med.save_photo_pair(self.ff, str(src), vp["t"], r["width"], r["height"], box, zoom, f_out, fr_out, privacy_box=pbox):
                        gen_failed = True
                        break
                    if not (f_out.is_file() and f_out.stat().st_size > 0 and fr_out.is_file() and fr_out.stat().st_size > 0):
                        gen_failed = True
                        break
                    staged_photos.append({
                        "full": f"days/{day}/photos/{full_name}",
                        "framed": f"days/{day}/photos/{framed_name}",
                        "camera": cname, "zoom_x": zoom, "time": t_lbl,
                        "score": vp["vlm"].get("photo_score", 4),
                        "caption_zh": vp["vlm"].get("caption_zh", ""),
                        "people": list(vp.get("_people") or vp["vlm"].get("_people") or []),
                    })
            except Exception:
                gen_failed = True

            if not gen_failed and len(staged_photos) == len(chosen_ph) and len(staged_photos) > 0:
                pending_photo_stage = stage_dir
                pending_staged_photos = staged_photos
            else:
                shutil.rmtree(stage_dir, ignore_errors=True)

        try:
            # 4. Ephemeral child-only '妈妈/妈' & '爸爸/爸' ASR worker (only if bursts exist & MemAvailable high!)
            all_bursts: list[dict[str, Any]] = []
            for r in rows:
                bs = json.loads(r.get("bursts_json") or "[]")
                if bs:
                    src = self._resolve_source_path(r)
                    if src:
                        for b in bs:
                            all_bursts.append({**b, "source_path": str(src), "wall_ts": r.get("wall_ts"), "row": r})
            speech_hits: list[dict[str, Any]] = []
            asr_min_kib = int(self.cfg["scheduler"].get("asr_min_host_mem_available_kib", 4456448))
            asr_enabled = self.cfg["asr"].get("enabled", True)
            asr_status = "disabled" if not asr_enabled else "no_candidates"
            if all_bursts and asr_enabled:
                all_bursts.sort(key=lambda b: -(b.get("snr_ratio", 0) * 0.4 + b.get("y_child_babble", 0) * 2.0))
                payload = [{k: v for k, v in b.items() if k != "row"} for b in all_bursts[:30]]
                model_dir = self.cfg["asr"]["model_dir"]
                checkpoints = self.data_root / "state" / "asr" / day
                try:
                    jobs = asr_worker.checkpoint_jobs(model_dir, payload)
                    processed, speech_hits = asr_worker.read_checkpoint(checkpoints, jobs)
                    headroom = read_container_headroom_kib()
                    asr_reserve = int(self.cfg["scheduler"].get("asr_min_container_headroom_kib", 1310720))
                    if len(processed) == len(jobs):
                        asr_status = "complete"
                    elif read_host_mem_available_kib() < asr_min_kib or (headroom is not None and headroom < asr_reserve):
                        asr_status = "deferred_memory"
                    else:
                        self._update_curate_progress(tier, day, "纯童声语音核验", f"本地 Whisper INT8 核验剩余 {len(jobs) - len(processed)} 段童声突发", background)
                        with tempfile.TemporaryDirectory(prefix="growth-asr-") as asr_dir:
                            in_p = Path(asr_dir) / "in.json"
                            in_p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                            try:
                                proc = subprocess.run(
                                    [sys.executable, "-m", "nas_growth.asr_worker", "--model-dir", model_dir,
                                     "--ffmpeg", self.ff, "--input", str(in_p), "--output", str(Path(asr_dir) / "out.json"),
                                     "--threads", "4", "--checkpoint-dir", str(checkpoints)],
                                    timeout=300, check=False, capture_output=True,
                                )
                                exited_ok = proc.returncode == 0
                            except (OSError, subprocess.SubprocessError):
                                exited_ok = False
                        jobs = asr_worker.checkpoint_jobs(model_dir, payload)
                        processed, speech_hits = asr_worker.read_checkpoint(checkpoints, jobs)
                        asr_status = "complete" if exited_ok and len(processed) == len(jobs) else "failed"
                except (OSError, ValueError):
                    asr_status = "failed"

            speech_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for sh in speech_hits[:4]:
                r_match = next((r for r in rows if r["source_rel"] == sh["source_rel"]), None)
                if r_match:
                    speech_items.append((sh, r_match))

            speech_vlm: dict[int, dict[str, Any]] = {}
            speech_face_matches: dict[int, dict[str, Any] | None] = {}
            if speech_items:
                def _prep_sp(item: tuple[int, tuple[dict[str, Any], dict[str, Any]]]):
                    idx_s, (sh_s, rm_s) = item
                    offsets = [max(0.2, sh_s["utterance_t_s"] - 0.5), sh_s["utterance_t_s"], min(max(0.5, rm_s["duration"] - 0.4), sh_s["utterance_t_s"] + 0.8)]
                    face_m: dict[str, Any] | None = None
                    if use_hires_face:
                        hires = self._extract_cloud_frames(rm_s, Path(sh_s["source_path"]), offsets, cams, max_edge=hi_res_edge)
                        if not hires:
                            return None
                        try:
                            face_m = self.face_service.identify_images(hires, day=day)
                        except Exception:
                            face_m = None
                        imgs = dl_dir._resize_jpegs_max_edge(hires, 768)
                    else:
                        imgs = self._extract_cloud_frames(rm_s, Path(sh_s["source_path"]), offsets, cams)
                        if not imgs:
                            return None
                        if face_active:
                            try:
                                face_m = self.face_service.identify_images(imgs, day=day)
                            except Exception:
                                face_m = None
                    fam_blk = (
                        "\n\n" + self.face_service.build_vlm_family_prompt_block(day, face_m, has_ref_sheet=bool(ref_imgs))
                        if face_active
                        else ""
                    )
                    sp_prompt = f"{PROMPT_MILESTONE}{fam_blk}"
                    if ref_imgs:
                        task_sp: tuple[Any, ...] = (idx_s, f"sp|{sh_s['source_rel']}|{sh_s['utterance_t_s']:.2f}", imgs, sp_prompt, ref_imgs)
                    else:
                        task_sp = (idx_s, f"sp|{sh_s['source_rel']}|{sh_s['utterance_t_s']:.2f}", imgs, sp_prompt)
                    return idx_s, task_sp, face_m

                with ThreadPoolExecutor(max_workers=min(4, len(speech_items))) as spool:
                    prepped_sp = [x for x in spool.map(_prep_sp, list(enumerate(speech_items))) if x is not None]
                sp_tasks = [t for _i, t, _fm in prepped_sp]
                for i_sp, _t, fm in prepped_sp:
                    speech_face_matches[i_sp] = fm
                if sp_tasks:
                    speech_vlm = self.vlm.ask_batch(sp_tasks, tier="pro")

            speech_count = 0
            for idx_s, (sh, r_match) in enumerate(speech_items):
                src = Path(sh["source_path"])
                c_cfg = cams.get(sh["camera"], {})
                vlm_ans: dict[str, Any] = speech_vlm.get(idx_s, {})
                if idx_s in speech_vlm and is_unconfirmed_refusal(vlm_ans):
                    rev_res = self._handle_vlm_refusal(
                        "speech", f"sp|{sh['source_rel']}|{sh['utterance_t_s']:.2f}", day, r_match,
                        float(sh["start_s"]), float(sh["end_s"]), vlm_ans, box_fallback=vlm_ans.get("child_box"),
                    )
                    if not isinstance(rev_res, dict):
                        continue
                    vlm_ans = rev_res
                # Audio-first preservation: even if camera missed child framing, keep full frame with clear label!
                visual_ok = vlm_ans.get("child_clear") in ("yes", "partial")
                zoom = float(c_cfg.get("zoom_factor", 1.35)) if visual_ok else 1.0
                box = vlm_ans.get("child_box") if visual_ok else None
                pbox = _resolve_privacy_box(vlm_ans, box or vlm_ans.get("child_box"))
                ms_id = f"speech_{day}_{Path(sh['source_rel']).stem}_{int(sh['utterance_t_s'])}"
                vid_rel = f"v2/highlights/{ms_id}.mp4"
                t0_rel = f"v2/t0_keepers/{sh['camera']}_{day}_{src.name}"
                ok_t0, sha_t0 = med.promote_t0_keeper(src, self.data_root / t0_rel)
                rel_t = round(sh["burst_start_s"] - sh["start_s"], 1)
                cname = c_cfg.get("name", sh["camera"])
                t_lbl = (dt.datetime.fromisoformat(r_match["wall_ts"]) + dt.timedelta(seconds=sh["start_s"])).strftime("%H:%M:%S") if r_match.get("wall_ts") else ""
                title_zh = f"纯童声 · 清晰「{sh['matched_word']}」（第 {rel_t:.1f} 秒处）"
                sub_zh = f"{cname} · {day} {t_lbl}" + ("" if visual_ok else " · 原声留存")
                if med.render_single_clip_with_badge(
                    self.ff, str(src), sh["start_s"], max(4.5, sh["end_s"] - sh["start_s"]),
                    r_match["width"], r_match["height"], box, zoom, bool(r_match["has_audio"]),
                    title_zh, sub_zh, self.data_root / vid_rel, self.font_path, self.use_vaapi, self.vaapi_dev,
                    privacy_box=pbox,
                ):
                    sp_note = f"基频 {sh['f0_median_hz']} Hz · 信噪比 {sh['snr_ratio']:.1f}× · 置信度 {sh['phonetic_prob']:.2f}"
                    raw_vlm_cap = str(vlm_ans.get("caption_zh") or "").strip()
                    if raw_vlm_cap:
                        raw_vlm_cap, sp_people = self.face_service.sanitize_text(
                            raw_vlm_cap, day=day, match_res=speech_face_matches.get(idx_s), vlm_ans=vlm_ans
                        )
                    else:
                        sp_people = []
                    cap = f"{title_zh}（{sp_note}）" + (f" · {raw_vlm_cap}" if raw_vlm_cap else " · 纯童声原声留存")
                    self.db.upsert_milestone({
                        "id": ms_id, "day": day, "camera": sh["camera"], "category": "speech",
                        "source_rel": sh["source_rel"], "start_s": sh["start_s"], "end_s": sh["end_s"],
                        "caption_zh": cap, "speaker_note": sp_note, "independence": "independent",
                        "score": 5.0 + sh["phonetic_prob"], "video_rel": vid_rel,
                        "poster_rel": f"v2/highlights/{ms_id}.jpg", "t0_rel": t0_rel if ok_t0 else None,
                        "t0_sha256": sha_t0 if ok_t0 else None, "review_status": "approved",
                        "meta": {"speech": sh, "zoom_x": zoom, "cam_zh": cname, "child_box": box, "privacy_box": pbox, "people": sp_people},
                    })
                    speech_count += 1

            # 5. Director-grade 4-stage Daily Highlight Short Film & Visual Milestones (HIGH Thinking)
            bday_str = str(self.cfg.get("project", {}).get("birthday", "2025-12-17"))
            age_info = lc.compute_age_info(bday_str, dt.date.fromisoformat(day))
            highlight_shots, story_doc, milestone_cands = dl_dir.curate_daily_highlight_and_milestones(
                day=day,
                tier=tier,
                background=background,
                rows=rows,
                cams=cams,
                age_info=age_info,
                vlm=self.vlm,
                vlm_cfg=vlm_cfg,
                extract_frames_fn=self._extract_cloud_frames,
                handle_refusal_fn=self._handle_vlm_refusal,
                effective_wall_ts_fn=self._effective_row_wall_ts,
                parse_wall_epoch_fn=self._parse_wall_epoch,
                resolve_source_path_fn=self._resolve_source_path,
                resolve_privacy_box_fn=_resolve_privacy_box,
                progress_fn=self._update_curate_progress,
                speech_items=speech_items,
                vlm_photos=vlm_photos or vlm_photos_3plus,
                kit_reviews=final_kit_reviews,
                face_service=self.face_service,
                extract_hires_frames_fn=lambda r_h, s_h, offs_h, c_h: self._extract_cloud_frames(
                    r_h, s_h, offs_h, c_h, max_edge=hi_res_edge
                ),
            )

            milestones_added = 0
            for mc in milestone_cands:
                r_mc = mc["row"]
                src_mc = mc["src"]
                cat = mc["cat"]
                ms_id = f"{cat}_{day}_{src_mc.stem}"
                vid_rel = f"v2/highlights/{ms_id}.mp4"
                t0_rel = f"v2/t0_keepers/{mc['cam']}_{day}_{src_mc.name}"
                ok_t0, sha_t0 = med.promote_t0_keeper(src_mc, self.data_root / t0_rel)
                if med.render_single_clip_with_badge(
                    self.ff, str(src_mc), mc["s0"], max(4.5, mc["dur_s"]), r_mc["width"], r_mc["height"],
                    mc["box"], mc["zoom"], bool(r_mc["has_audio"]),
                    f"{mc['cname']} · {CATEGORY_ZH.get(cat, cat)}", f"{day} {mc['t_lbl']}",
                    self.data_root / vid_rel, self.font_path, self.use_vaapi, self.vaapi_dev,
                    privacy_box=mc["pbox"],
                ):
                    self.db.upsert_milestone({
                        "id": ms_id, "day": day, "camera": mc["cam"], "category": cat, "source_rel": r_mc["source_rel"],
                        "start_s": mc["s0"], "end_s": mc["s0"] + mc["dur_s"], "caption_zh": mc["caption_zh"],
                        "independence": mc["independence"], "score": mc["exc"],
                        "video_rel": vid_rel, "poster_rel": f"v2/highlights/{ms_id}.jpg",
                        "t0_rel": t0_rel if ok_t0 else None, "t0_sha256": sha_t0 if ok_t0 else None,
                        "review_status": "approved",
                        "meta": {
                            "zoom_x": mc["zoom"], "cam_zh": mc["cname"], "provider": mc.get("provider"),
                            "model": mc.get("model"), "child_box": mc["box"], "privacy_box": mc["pbox"],
                            "people": list(mc.get("people") or []),
                            "review_status": "approved",
                        },
                    })
                    milestones_added += 1

            # Protect seeded day-20261108 fluent highlight untouched!
            hl_rel = None
            hl_meta: dict[str, Any] | None = None
            if day == "2026-11-08" and (self.data_root / "day-20261108" / "highlight_20261108.mp4").is_file():
                hl_rel = "day-20261108/highlight_20261108.mp4"
            elif highlight_shots:
                hl_name = f"highlight_{day.replace('-', '')}.mp4"
                hl_path = day_dir / hl_name
                self._update_curate_progress(tier, day, "渲染每日精华", f"核显 VAAPI 合成 {day} 1080p 电影级每日短片 ({len(highlight_shots)} 镜)", background)
                film_header = (
                    {
                        "kicker_zh": f"成长第 {age_info['day_n']} 天 · {age_info['label_zh']}",
                        "title_zh": str(story_doc.get("film_title_zh") or f"{day} · 每日成长短剧"),
                        "sub_zh": f"{day} · 四幕每日珍藏记",
                    }
                    if story_doc
                    else None
                )
                hl_meta = med.render_smooth_film_nas(
                    self.ff, highlight_shots, hl_path, self.font_path, self.use_vaapi, self.vaapi_dev,
                    fade_s=0.55, fps=30, out_w=1920, out_h=1080, qp=int(mcfg.get("highlight_qp", 19)),
                    film_header=film_header,
                    closing_epilogue_zh=story_doc.get("closing_epilogue_zh") if story_doc else None,
                )
                if hl_meta:
                    hl_rel = f"days/{day}/{hl_name}"
                    if story_doc:
                        story_doc["actual_duration_s"] = hl_meta.get("duration_s")
                        story_doc["shots_count"] = hl_meta.get("shots", len(highlight_shots))
                        sc_tmp = day_dir / f".script_{os.getpid()}.tmp"
                        sc_tmp.write_text(json.dumps(story_doc, ensure_ascii=False, indent=2), encoding="utf-8")
                        os.replace(sc_tmp, day_dir / "daily_script.json")

            saved_story = story_doc if (hl_meta and story_doc) else (
                prev_summary.get("daily_story") if (hl_rel is None and prev_hl) else (story_doc or None)
            )
            hl_dur_s = (
                float(hl_meta.get("duration_s") or 0.0)
                if hl_meta
                else float(prev_summary.get("highlight_duration_s") or (story_doc.get("est_duration_s") if story_doc else 0.0) or 0.0)
            )
            hl_shots_cnt = (
                int(hl_meta.get("shots") or len(highlight_shots))
                if hl_meta
                else int(prev_summary.get("highlight_shots_count") or len(highlight_shots) or 0)
            )
            if hl_rel is None and prev_hl:
                hl_rel = prev_hl

            st = self.db.count_day_stats(day)
            pending_rev_day = len(self.db.list_vlm_reviews(status="pending_review", day=day, limit=200))
            summary = {
                "day": day,
                "tier": tier,
                "status": "done" if final and len(rows) == st["total"] == st["done"] else "preview",
                "total_files": st["total"],
                "done_files": st["done"],
                "orig_bytes": st["orig_bytes"],
                "t1_bytes": st["t1_bytes"],
                "t2_bytes": st["t2_bytes"],
                "highlight_rel": hl_rel,
                "highlight_duration_s": round(hl_dur_s, 1) if hl_dur_s else 0.0,
                "highlight_shots_count": hl_shots_cnt,
                "daily_story": saved_story,
                "photos": saved_photos,
                "timelapses": timelapses,
                "missing_timelapse_cameras": sorted(by_cam.keys() - timelapses.keys()),
                "milestones_count": milestones_added,
                "speech_count": speech_count,
                "pending_review_count": pending_rev_day,
                "kitchen_reviewed_this_pass": kit_reviewed_this_pass,
                "kitchen_unreviewed_remaining": kit_unreviewed_remaining,
                "asr_status": asr_status,
                "updated_at": dt.datetime.now(self.tz).isoformat(timespec="seconds"),
            }
            if summary["missing_timelapse_cameras"] or asr_status in {"deferred_memory", "failed"} or kit_unreviewed_remaining > 0:
                summary["status"] = "preview"
            if pending_photo_stage is not None and pending_staged_photos:
                if _atomic_swap_dir(day_dir, pending_photo_stage, pdir):
                    saved_photos = pending_staged_photos
                else:
                    shutil.rmtree(pending_photo_stage, ignore_errors=True)
                pending_photo_stage = None
            summary["photos"] = saved_photos
            sum_tmp = day_dir / f".summary_{os.getpid()}.tmp"
            sum_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(sum_tmp, sum_path)
            self.db.upsert_day_summary(day, tier, summary["status"], summary)
            return summary
        finally:
            if pending_photo_stage is not None and pending_photo_stage.exists():
                shutil.rmtree(pending_photo_stage, ignore_errors=True)

    def _run_day_batch(self, day: str, tier: str, max_files: int, cams: dict[str, Any]) -> int:
        batch = self.db.get_queued_files_for_day(day, limit=max_files, daytime_first=True)
        if not batch:
            return 0
        workers = max(1, int(self.cfg["scheduler"].get("workers", 6)))
        heavy_permits = threading.BoundedSemaphore(max(1, int(self.cfg["scheduler"].get("max_heavy_workers", 2))))

        def process(row: dict[str, Any]) -> None:
            if int(row["size"]) > 60_000_000:
                with heavy_permits:
                    self._process_one_file(row, cams)
            else:
                self._process_one_file(row, cams)

        done_n = 0
        self._set_job(tier, day=day, step="并发视频分析", detail=f"正在分析本批 {len(batch)} 档监控（VAAPI 抽帧 + YOLO26s + YAMNet）")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process, r): r for r in batch}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    fut.result()
                except Exception as exc:
                    self.db.mark_file_done(r["source_rel"], {"status": "error", "error": redact_secrets(repr(exc))})
                done_n += 1
                self._set_job(tier, day=day, step="并发视频分析", detail=f"已完成 {Path(r['source_rel']).name}", camera=r["camera"], last_file=r["source_rel"])
        return done_n

    def ensure_rolling_days(self, today_d: dt.date) -> tuple[list[str], bool, list[str]]:
        """Keep at most `rolling_active_days` newest non-today days active in strict reverse-chronological order."""
        scfg = self.cfg["scheduler"]
        p1_start = dt.date.fromisoformat(scfg.get("p1_start_day", "2026-11-16"))
        p1_end = dt.date.fromisoformat(scfg.get("p1_end_day", "2025-12-17"))
        rolling_days = max(2, int(scfg.get("rolling_active_days", 2)))
        today_str = today_d.isoformat()
        yesterday_d = today_d - dt.timedelta(days=1)
        if self._first_live_day is None:
            self._first_live_day = dt.date.fromisoformat(
                self.db.first_live_day(p1_start.isoformat(), yesterday_d.isoformat())
            )
        scan_floor = min(p1_end, self._first_live_day)
        now_mono = time.monotonic()
        if now_mono >= self._next_t2_retry_check:
            self.db.requeue_due_t2_failures(scan_floor.isoformat(), today_str)
            self._next_t2_retry_check = now_mono + 30

        incomplete_set = set(self.db.get_incomplete_days(scan_floor.isoformat(), today_str))
        picked_non_today: list[str] = []
        newly_empty_with_done: list[str] = []

        cur_d = yesterday_d
        while cur_d >= scan_floor and len(picked_non_today) < rolling_days:
            if p1_start < cur_d < self._first_live_day:
                cur_d = p1_start
                continue
            ds = cur_d.isoformat()
            if p1_end <= cur_d <= p1_start or cur_d >= self._first_live_day:
                if ds in incomplete_set:
                    picked_non_today.append(ds)
                elif ds not in self._exhausted_days:
                    self.discover_day_files(ds)
                    st = self.db.count_day_stats(ds)
                    if st["queued"] > 0:
                        incomplete_set.add(ds)
                        picked_non_today.append(ds)
                    else:
                        self._exhausted_days.add(ds)
                        if st["done"] > 0:
                            newly_empty_with_done.append(ds)
            cur_d -= dt.timedelta(days=1)

        if len(picked_non_today) < rolling_days:
            for d in sorted(incomplete_set - {today_str} - set(picked_non_today), reverse=True):
                picked_non_today.append(d)
                if len(picked_non_today) >= rolling_days:
                    break

        workers = max(1, int(scfg.get("workers", 6)))
        max_heavy = max(1, int(scfg.get("max_heavy_workers", 2)))
        light_target = max(1, workers - max_heavy)
        spillover_cap = max(0, int(scfg.get("light_spillover_days", 2)))
        if picked_non_today and spillover_cap > 0 and hasattr(self.db, "get_day_queue_weight_counts"):
            w_map = self.db.get_day_queue_weight_counts(picked_non_today)
            tot_heavy = sum(v["heavy"] for v in w_map.values())
            tot_light = sum(v["light"] for v in w_map.values())
            max_active = rolling_days + spillover_cap
            while tot_heavy > 0 and tot_light < light_target and cur_d >= scan_floor and len(picked_non_today) < max_active:
                if p1_start < cur_d < self._first_live_day:
                    cur_d = p1_start
                    continue
                ds = cur_d.isoformat()
                if (p1_end <= cur_d <= p1_start or cur_d >= self._first_live_day) and ds not in picked_non_today:
                    if ds in incomplete_set:
                        picked_non_today.append(ds)
                        dw = self.db.get_day_queue_weight_counts([ds]).get(ds, {"light": 0, "heavy": 0})
                        tot_light += dw["light"]
                        tot_heavy += dw["heavy"]
                    elif ds not in self._exhausted_days:
                        self.discover_day_files(ds)
                        st = self.db.count_day_stats(ds)
                        if st["queued"] > 0:
                            incomplete_set.add(ds)
                            picked_non_today.append(ds)
                            dw = self.db.get_day_queue_weight_counts([ds]).get(ds, {"light": 0, "heavy": 0})
                            tot_light += dw["light"]
                            tot_heavy += dw["heavy"]
                        else:
                            self._exhausted_days.add(ds)
                            if st["done"] > 0:
                                newly_empty_with_done.append(ds)
                cur_d -= dt.timedelta(days=1)

        any_non_today = bool(incomplete_set - {today_str}) or bool(picked_non_today)
        all_p1_done = False
        if not any_non_today:
            hist_counts = self.db.get_historical_queue_counts(p1_end.isoformat(), p1_start.isoformat())
            all_p1_done = hist_counts["done"] > 0 and hist_counts["done"] == hist_counts["total"]
        if self.db.get_state("p1_backfill_complete") != all_p1_done:
            self.db.set_state("p1_backfill_complete", all_p1_done)

        active_days = ([today_str] if today_str in incomplete_set else []) + picked_non_today
        return active_days, all_p1_done, newly_empty_with_done

    def pick_next_batch_files(
        self,
        active_days: list[str],
        today_str: str,
        slots: int,
        inflight_rels: set[str],
        inflight_heavy: int,
        max_heavy: int = 2,
    ) -> list[dict[str, Any]]:
        """Pick up to `slots` files: Today (P0) strictly first, then 2:1 interleaved multi-day backfill with heavy-bundle cap."""
        if slots <= 0 or not active_days:
            return []
        fetch_lim = max(slots * 4, 32)
        cands_by_day: dict[str, list[dict[str, Any]]] = {}
        heavy_slots = max(0, max_heavy - inflight_heavy)
        for d in active_days:
            rows = self.db.get_queued_files_for_day(d, limit=fetch_lim, light_only=heavy_slots == 0)
            filtered = [r for r in rows if r["source_rel"] not in inflight_rels]
            light_count = sum(int(r["size"]) <= 60_000_000 for r in filtered)
            if heavy_slots and len(rows) == fetch_lim and light_count < slots:
                # Daytime bundles can occupy the entire first page ahead of night-time light files.
                seen = inflight_rels | {r["source_rel"] for r in filtered}
                light = self.db.get_queued_files_for_day(d, limit=fetch_lim, light_only=True)
                filtered.extend(r for r in light if r["source_rel"] not in seen)
            if filtered:
                cands_by_day[d] = filtered

        ordered_cands: list[dict[str, Any]] = []
        if today_str in cands_by_day:
            ordered_cands.extend(cands_by_day.pop(today_str))

        other_days = [d for d in active_days if d != today_str and d in cands_by_day]
        if other_days:
            queues = [list(cands_by_day[d]) for d in other_days]
            weights = [2 if idx == 0 else 1 for idx in range(len(queues))]
            while any(queues):
                for idx, q in enumerate(queues):
                    for _ in range(weights[idx]):
                        if q:
                            ordered_cands.append(q.pop(0))

        picked: list[dict[str, Any]] = []
        cur_heavy = inflight_heavy
        for r in ordered_cands:
            is_heavy = int(r.get("size") or 0) > 60_000_000
            if is_heavy and cur_heavy >= max_heavy:
                if hasattr(self, "_telemetry"):
                    self._telemetry["heavy_cap_deferrals"] += 1
                continue
            picked.append(r)
            if is_heavy:
                cur_heavy += 1
            if len(picked) >= slots:
                break

        return picked

    def _compute_admitted_slots(self, mem_kib: int, inflight_len: int) -> tuple[int, int]:
        """Compute effective worker cap and admissible new slots under host & cgroup memory guards."""
        scfg = self.cfg["scheduler"]
        min_mem_kib = int(scfg.get("min_host_mem_available_kib", 4194304))
        workers = max(1, int(scfg.get("workers", 6)))
        full_cushion = int(scfg.get("mem_full_workers_cushion_kib", 1_572_864))
        mid_cushion = int(scfg.get("mem_mid_workers_cushion_kib", 786_432))

        if mem_kib < min_mem_kib:
            return 0, 0
        if mem_kib >= min_mem_kib + full_cushion:
            eff_workers = workers
        elif mem_kib >= min_mem_kib + mid_cushion:
            eff_workers = min(workers, 4)
        else:
            eff_workers = min(workers, 2)

        if eff_workers < workers:
            self._telemetry["host_mem_throttles"] += 1

        slots = max(0, eff_workers - inflight_len)
        headroom = read_container_headroom_kib()
        if headroom is not None and slots > 0:
            per_worker = max(1, int(scfg.get("worker_container_headroom_kib", 393216)))
            cg_slots = headroom // per_worker
            if cg_slots < slots:
                self._telemetry["container_headroom_throttles"] += 1
            slots = min(slots, cg_slots)
        return eff_workers, slots

    def run_queue_until_idle(
        self,
        *,
        discover_days: list[str] | None = None,
        max_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Drain queued files using the production admission/heavy-cap scheduler (used for isolated concurrency benchmarks)."""
        scfg = self.cfg["scheduler"]
        workers = max(1, int(scfg.get("workers", 6)))
        max_heavy = max(1, int(scfg.get("max_heavy_workers", 2)))
        min_mem_kib = int(scfg.get("min_host_mem_available_kib", 4194304))
        cams = self._effective_cameras()
        if discover_days:
            for d_str in discover_days:
                self.discover_day_files(d_str)

        t_start = time.monotonic()
        inflight: dict[Any, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as worker_pool:
            while not self._stop.is_set() and (time.monotonic() - t_start) < max_seconds:
                done_futs = [f for f in inflight if f.done()]
                for fut in done_futs:
                    r = inflight.pop(fut)
                    try:
                        fut.result()
                    except Exception as exc:
                        self.db.mark_file_done(r["source_rel"], {"status": "error", "error": redact_secrets(repr(exc))})

                mem_kib = read_host_mem_available_kib()
                cur_min = self._telemetry.get("min_host_mem_kib")
                self._telemetry["min_host_mem_kib"] = mem_kib if cur_min is None else min(int(cur_min), mem_kib)
                eff_workers, slots = self._compute_admitted_slots(mem_kib, len(inflight))
                if mem_kib < min_mem_kib:
                    self._telemetry["host_mem_yields"] += 1

                active_days = self.db.get_incomplete_days("1970-01-01", "2099-12-31")
                if not active_days and not inflight:
                    break

                today_str = active_days[0] if active_days else _shanghai_now(self.cfg).date().isoformat()
                if slots > 0 and active_days:
                    inflight_rels = {r["source_rel"] for r in inflight.values()}
                    inflight_heavy = sum(1 for r in inflight.values() if int(r.get("size") or 0) > 60_000_000)
                    new_batch = self.pick_next_batch_files(
                        active_days, today_str, slots, inflight_rels, inflight_heavy, max_heavy=max_heavy
                    )
                    for r in new_batch:
                        fut = worker_pool.submit(self._process_one_file, r, cams)
                        inflight[fut] = r
                    self._telemetry["max_inflight"] = max(int(self._telemetry["max_inflight"]), len(inflight))

                self._flush_telemetry()
                if inflight:
                    wait(list(inflight.keys()), timeout=1.0, return_when=FIRST_COMPLETED)
                else:
                    self._stop.wait(1.0)

        self._flush_telemetry(force=True)
        elapsed_s = round(time.monotonic() - t_start, 3)
        with self.db.session() as con:
            agg = con.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                          SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued,
                          SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                          COALESCE(SUM(CASE WHEN status='done' THEN duration ELSE 0 END), 0.0) AS source_duration_s,
                          COALESCE(SUM(CASE WHEN status='done' THEN size ELSE 0 END), 0) AS orig_bytes,
                          COALESCE(SUM(CASE WHEN status='done' THEN t1_bytes ELSE 0 END), 0) AS t1_bytes,
                          COALESCE(SUM(CASE WHEN status='done' THEN t2_bytes ELSE 0 END), 0) AS t2_bytes
                   FROM files"""
            ).fetchone()
        tot_d = dict(agg) if agg else {}
        src_s = float(tot_d.get("source_duration_s") or 0.0)
        return {
            "workers": workers,
            "max_heavy_workers": max_heavy,
            "elapsed_s": elapsed_s,
            "source_duration_s": round(src_s, 2),
            "source_hours_per_wall_hour": round((src_s / elapsed_s), 2) if elapsed_s > 0 else 0.0,
            "total": int(tot_d.get("total") or 0),
            "done": int(tot_d.get("done") or 0),
            "queued": int(tot_d.get("queued") or 0),
            "errors": int(tot_d.get("errors") or 0),
            "orig_bytes": int(tot_d.get("orig_bytes") or 0),
            "t1_bytes": int(tot_d.get("t1_bytes") or 0),
            "t2_bytes": int(tot_d.get("t2_bytes") or 0),
            "telemetry": dict(self._telemetry),
        }

    def _curate_day_async(self, day: str, tier: str, final: bool, force_curate: bool = False) -> dict[str, Any]:
        try:
            return self.finalize_day(day, tier, final=final, background=True, force_curate=force_curate)
        finally:
            with self._job_lock:
                self._curator_status = ""

    def _compile_film_async(self, spec: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.compile_film(spec)
        finally:
            with self._job_lock:
                self._curator_status = ""

    def run_forever(self) -> None:
        scfg = self.cfg["scheduler"]
        p0_interval = int(scfg.get("p0_interval_seconds", 900))
        min_mem_kib = int(scfg.get("min_host_mem_available_kib", 4194304))
        workers = max(1, int(scfg.get("workers", 6)))
        max_heavy = max(1, int(scfg.get("max_heavy_workers", 2)))
        preview_thr = int(scfg.get("p0_preview_active_threshold", 20))
        last_p0_scan = 0.0
        last_lifecycle_scan = 0.0
        last_cam_refresh = 0.0
        cams: dict[str, Any] = {}
        active_days: list[str] = []
        published_days = self.db.get_dashboard_snapshot().get("days", [])
        last_requested_count = {d["day"]: int(d["done_files"]) for d in published_days}
        preview_days = {d["day"] for d in published_days if d["status"] == "preview"}
        inflight: dict[Any, dict[str, Any]] = {}
        pending_curate = CurationQueue(self.db)
        self.curation_queue = pending_curate
        self.db.reset_stale_rendering_films()
        last_requested_count.update({d: job["done_files"] for d, job in pending_curate.jobs.items()})
        curating_fut: Any = None
        curating_kind: str = "day"
        c_day: str = ""
        c_job: dict[str, Any] = {}
        lifecycle_fut: Any = None
        init_tz = _shanghai_now(self.cfg).date()
        for p0d in (init_tz.isoformat(), (init_tz - dt.timedelta(days=1)).isoformat()):
            if p0d in pending_curate.jobs:
                pending_curate.clear_backoff(p0d)
            st0 = self.db.count_day_stats(p0d)
            if st0["done"] < preview_thr:
                continue
            sp0 = self.data_root / "days" / p0d / "summary.json"
            needs_refresh = not sp0.is_file()
            if not needs_refresh:
                try:
                    sobj = json.loads(sp0.read_text(encoding="utf-8"))
                    prev_done = int(sobj.get("done_files") or 0)
                    needs_refresh = (
                        any(float(p.get("zoom_x", 1.0)) >= 1.95 for p in (sobj.get("photos") or []))
                        or not isinstance(sobj.get("daily_story"), dict)
                        or str(sobj.get("updated_at") or "") < "2026-11-18T10:05:00"
                        or (st0["done"] - prev_done >= 40)
                        or (st0["queued"] == 0 and prev_done < st0["done"])
                    )
                except Exception:
                    needs_refresh = True
            if needs_refresh:
                pending_curate.request(p0d, "P0_LIVE", st0["queued"] == 0, st0["done"])

        worker_pool = ThreadPoolExecutor(max_workers=max(workers, 6))
        curator_pool = ThreadPoolExecutor(max_workers=1)
        lifecycle_pool = ThreadPoolExecutor(max_workers=1)
        try:
            while not self._stop.is_set():
                try:
                    if self.reload_config_if_changed():
                        scfg = self.cfg["scheduler"]
                        p0_interval = int(scfg.get("p0_interval_seconds", 900))
                        min_mem_kib = int(scfg.get("min_host_mem_available_kib", 4194304))
                        max_heavy = max(1, int(scfg.get("max_heavy_workers", 2)))
                        preview_thr = int(scfg.get("p0_preview_active_threshold", 20))
                        cams = self._effective_cameras()
                        last_p0_scan = 0.0
                    now_tz = _shanghai_now(self.cfg)
                    today_d = now_tz.date()
                    today_str = today_d.isoformat()
                    yesterday_str = (today_d - dt.timedelta(days=1)).isoformat()
                    p0_days = {today_str, yesterday_str}

                    if curating_fut is not None and curating_fut.done():
                        if curating_kind == "film":
                            try:
                                curating_fut.result()
                            except Exception:
                                pass
                        else:
                            try:
                                result = curating_fut.result()
                            except Exception as exc:
                                pending_curate.finish(c_day, c_job, redact_secrets(repr(exc)))
                            else:
                                asr_incomplete = result.get("asr_status") in {"deferred_memory", "failed"}
                                kit_rem = int(result.get("kitchen_unreviewed_remaining") or 0)
                                kit_prog = int(result.get("kitchen_reviewed_this_pass") or 0)
                                retry_reason = ""
                                requeue_kit_now = False
                                if c_job["final"] and result.get("status") == "preview":
                                    stats = self.db.count_day_stats(c_day)
                                    if stats["total"] > 0 and stats["total"] == stats["done"]:
                                        if kit_rem > 0 and kit_prog > 0:
                                            requeue_kit_now = True
                                        elif kit_rem > 0:
                                            retry_reason = "kitchen_vlm_deferred"
                                        elif asr_incomplete:
                                            retry_reason = "asr_" + result["asr_status"]
                                        elif result.get("missing_timelapse_cameras"):
                                            retry_reason = "daily_timelapse_incomplete"
                                        else:
                                            retry_reason = "source_snapshot_changed"
                                pending_curate.finish(c_day, c_job, retry_reason)
                                if c_job.get("final"):
                                    last_lifecycle_scan = 0.0
                                if requeue_kit_now:
                                    stats = self.db.count_day_stats(c_day)
                                    pending_curate.request(c_day, c_job["tier"], c_job["final"], stats["done"])
                        curating_fut = None

                    done_futs = [f for f in inflight if f.done()]
                    for fut in done_futs:
                        r = inflight.pop(fut)
                        try:
                            fut.result()
                        except Exception as exc:
                            self.db.mark_file_done(r["source_rel"], {"status": "error", "error": redact_secrets(repr(exc))})
                        r_day = r["day"]
                        st_d = self.db.count_day_stats(r_day)
                        d_tier = "P0_LIVE" if r_day in p0_days else "P1_BACKFILL"
                        prev_n = last_requested_count.get(r_day, 0)
                        if st_d["queued"] == 0 or (st_d["done"] >= preview_thr and st_d["done"] - prev_n >= 40) or (prev_n == 0 and st_d["done"] >= preview_thr):
                            pending_curate.request(r_day, d_tier, st_d["queued"] == 0, st_d["done"])
                            last_requested_count[r_day] = st_d["done"]
                        days_lbl = ",".join(active_days) if active_days else r_day
                        self._set_job(
                            d_tier,
                            day=r_day,
                            step="并发视频分析",
                            detail=f"{len(inflight)}/{workers} 路并发 · 活跃日期 {days_lbl} · 完成 {Path(r['source_rel']).name}",
                            camera=r["camera"],
                            last_file=r["source_rel"],
                        )

                    if self.is_paused():
                        self.db.set_state("active_job", {"phase": "PAUSED", "updated_at": dt.datetime.now(self.tz).isoformat(timespec="seconds")})
                        wait_t = [f for f in list(inflight.keys()) + ([curating_fut] if curating_fut is not None else [])]
                        if wait_t:
                            wait(wait_t, timeout=5.0, return_when=FIRST_COMPLETED)
                        else:
                            self._stop.wait(10.0)
                        continue

                    mem_kib = read_host_mem_available_kib()
                    cur_min = self._telemetry.get("min_host_mem_kib")
                    self._telemetry["min_host_mem_kib"] = mem_kib if cur_min is None else min(int(cur_min), mem_kib)
                    if mem_kib < min_mem_kib:
                        self._telemetry["host_mem_yields"] += 1
                        self._flush_telemetry(force=True)
                        self.db.set_state("active_job", {"phase": "YIELD_LOW_MEM", "host_mem_kib": mem_kib, "min_kib": min_mem_kib})
                        wait_t = [f for f in list(inflight.keys()) + ([curating_fut] if curating_fut is not None else [])]
                        if wait_t:
                            wait(wait_t, timeout=5.0, return_when=FIRST_COMPLETED)
                        else:
                            self._stop.wait(30.0)
                        continue

                    if lifecycle_fut is not None and lifecycle_fut.done():
                        try:
                            lifecycle_fut.result()
                        except Exception as exc:
                            self.db.set_state("lifecycle_error", {"error": redact_secrets(repr(exc)), "at": time.time()})
                        lifecycle_fut = None
                    if lifecycle_fut is None and time.time() - last_lifecycle_scan >= 1800:
                        lifecycle_fut = lifecycle_pool.submit(self.refresh_lifecycle_status)
                        last_lifecycle_scan = time.time()

                    if not cams or (time.time() - last_cam_refresh >= 300):
                        cams = self.refresh_camera_health()["cameras"]
                        for rd in self.sync_cloud_backfill_window():
                            last_requested_count.pop(rd, None)
                        last_cam_refresh = time.time()

                    if time.time() - last_p0_scan >= p0_interval:
                        for pd in (today_str, yesterday_str):
                            self._exhausted_days.discard(pd)
                            self.discover_day_files(pd)
                        last_p0_scan = time.time()

                    eff_workers, slots = self._compute_admitted_slots(mem_kib, len(inflight))

                    if len(inflight) < eff_workers:
                        active_days, _all_p1_done, empty_done_days = self.ensure_rolling_days(today_d)
                        for ed in empty_done_days:
                            st_ed = self.db.count_day_stats(ed)
                            if st_ed["done"] > 0 and (ed in preview_days or st_ed["done"] != last_requested_count.get(ed)):
                                d_tier = "P0_LIVE" if ed in p0_days else "P1_BACKFILL"
                                pending_curate.request(ed, d_tier, True, st_ed["done"])
                                preview_days.discard(ed)
                                last_requested_count[ed] = st_ed["done"]
                        for pd in (today_str, yesterday_str):
                            st_p0 = self.db.count_day_stats(pd)
                            if (st_p0["queued"] == 0 and st_p0["done"] > 0
                                    and (pd in preview_days or st_p0["done"] != last_requested_count.get(pd))):
                                pending_curate.request(pd, "P0_LIVE", True, st_p0["done"])
                                preview_days.discard(pd)
                                last_requested_count[pd] = st_p0["done"]

                        eff_workers, slots = self._compute_admitted_slots(mem_kib, len(inflight))
                        inflight_rels = {r["source_rel"] for r in inflight.values()}
                        inflight_heavy = sum(1 for r in inflight.values() if int(r.get("size") or 0) > 60_000_000)
                        new_batch = self.pick_next_batch_files(
                            active_days, today_str, slots, inflight_rels, inflight_heavy, max_heavy=max_heavy
                        )
                        for r in new_batch:
                            fut = worker_pool.submit(self._process_one_file, r, cams)
                            inflight[fut] = r
                        self._telemetry["max_inflight"] = max(int(self._telemetry["max_inflight"]), len(inflight))
                        if new_batch:
                            top_day = active_days[0] if active_days else new_batch[0]["day"]
                            top_tier = "P0_LIVE" if top_day in p0_days else "P1_BACKFILL"
                            self._set_job(
                                top_tier,
                                day=top_day,
                                step="并发视频分析",
                                detail=f"{len(inflight)}/{eff_workers} 路并发运行中 · 活跃日期 {','.join(active_days)}",
                            )

                    if curating_fut is None:
                        ready = pending_curate.next_ready()
                        forced_day = bool(ready and ready[1].get("force_ts"))
                        film_job = None if forced_day else self.db.next_ready_film_production()
                        if film_job is not None:
                            curating_kind = "film"
                            with self._job_lock:
                                self._curator_status = f"{film_job['film_id']} · 流年编纂"
                            curating_fut = curator_pool.submit(self._compile_film_async, film_job)
                        elif ready:
                            c_day, c_job = ready
                            c_tier, c_final = c_job["tier"], c_job["final"]
                            curating_kind = "day"
                            with self._job_lock:
                                self._curator_status = f"{c_day} · {'终审' if c_final else '预览'}"
                            curating_fut = curator_pool.submit(
                                self._curate_day_async,
                                c_day,
                                c_tier,
                                c_final,
                                bool(c_job.get("force_ts")),
                            )

                    self._flush_telemetry()
                    wait_targets = list(inflight.keys()) + ([curating_fut] if curating_fut is not None else [])
                    if wait_targets:
                        wait(wait_targets, timeout=2.0, return_when=FIRST_COMPLETED)
                    else:
                        self.db.set_state("active_job", {
                            "phase": ("YIELD_CONTAINER_MEMORY" if active_days and slots == 0
                                      else "CURATION_RETRY_WAIT" if pending_curate.jobs
                                      else "IDLE_CAUGHT_UP"),
                            "updated_at": dt.datetime.now(self.tz).isoformat(timespec="seconds"),
                            "host_mem_kib": mem_kib,
                        })
                        self._wake.wait(60.0)
                        if not self._stop.is_set():
                            self._wake.clear()
                except Exception as exc:
                    self.db.set_state("active_job", {
                        "phase": "RECOVERING",
                        "error": redact_secrets(repr(exc)),
                        "updated_at": dt.datetime.now(self.tz).isoformat(timespec="seconds"),
                    })
                    self._stop.wait(15.0)
        finally:
            self._flush_telemetry(force=True)
            lifecycle_pool.shutdown(wait=True, cancel_futures=True)
            worker_pool.shutdown(wait=True, cancel_futures=True)
            curator_pool.shutdown(wait=True, cancel_futures=True)
