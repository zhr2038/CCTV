"""Growth archive V2: camera auto-discovery, privacy-safe kinship speech spotting,
kitchen 4K & bedside day/night filtering, and local VLM (Qwen3.8-27B) verification.

NAS shares stay strictly read-only; no cloud media uploads occur.
"""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .audio_events import YamnetDetector
from .media import resolve_ffmpeg, resolve_ffprobe

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = PROJECT / "config/growth_archive.toml"
FF = resolve_ffmpeg(None)
FP = resolve_ffprobe(None)


def load_v2_config(path: str | Path = DEFAULT_CFG) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def ensure_v2_dirs(cfg: dict[str, Any]) -> Path:
    root = PROJECT / cfg["project"]["pilot_output"]
    v2 = root / "v2"
    for sub in ("highlights", "t0_keepers", "photos", "monthly_cards", "quality_lab", "work", "kitchen_20261115", "bedside_v2"):
        (v2 / sub).mkdir(parents=True, exist_ok=True)
    return root


def run_cmd(cmd: list[str], timeout: float = 900, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)


def probe_media(path: str) -> dict[str, Any]:
    r = run_cmd([FP, "-v", "error", "-show_entries",
                 "format=duration:stream=codec_type,codec_name,width,height",
                 "-of", "json", path], text=True)
    info = json.loads(r.stdout or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    dur = float(info.get("format", {}).get("duration") or 0)
    return {
        "duration": dur,
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
        "vcodec": v.get("codec_name") if v else None,
        "has_audio": a is not None,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- Step 1: Camera Discovery & Health

def _latest_aqara_file(cam_dir: Path) -> tuple[str | None, dt.datetime | None, int]:
    date_dirs = sorted([d for d in cam_dir.iterdir() if d.is_dir() and len(d.name) == 8 and d.name.isdigit()],
                       key=lambda d: d.name, reverse=True)
    for d in date_dirs:
        mp4s = sorted(d.glob("*.mp4"), key=lambda p: p.name)
        if mp4s:
            last = mp4s[-1]
            try:
                ts = dt.datetime.strptime(f"{d.name}{last.stem[:6]}", "%Y%m%d%H%M%S")
            except ValueError:
                ts = dt.datetime.fromtimestamp(last.stat().st_mtime)
            return d.name, ts, len(mp4s)
    return None, None, 0


def _latest_xiaomi_file(cam_dir: Path) -> tuple[str | None, dt.datetime | None, int]:
    mp4s = sorted(cam_dir.glob("*.mp4"), key=lambda p: p.name)
    if mp4s:
        last = mp4s[-1]
        parts = last.stem.split("_")
        ts = None
        for idx in (2, 1):
            if len(parts) > idx and len(parts[idx]) == 14 and parts[idx].isdigit():
                try:
                    ts = dt.datetime.strptime(parts[idx], "%Y%m%d%H%M%S")
                    break
                except ValueError:
                    pass
        if ts is None:
            ts = dt.datetime.fromtimestamp(last.stat().st_mtime)
        return ts.strftime("%Y%m%d"), ts, len(mp4s)
    # Check hourly folders (like old kitchen 50ec503b63bf)
    subdirs = sorted([d for d in cam_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: d.name, reverse=True)
    for d in subdirs:
        mp4s = sorted(d.glob("*.mp4"), key=lambda p: p.name)
        if mp4s:
            ts = dt.datetime.fromtimestamp(mp4s[-1].stat().st_mtime)
            return ts.strftime("%Y%m%d"), ts, len(mp4s)
    return None, None, 0


def discover_cameras(cfg: dict[str, Any], out_root: Path | None = None) -> dict[str, Any]:
    nas = Path(cfg["project"]["nas_root"])
    ad = cfg.get("auto_discovery", {})
    configured = dict(cfg.get("cameras", {}))
    ignore_ids = set(ad.get("ignore_ids", []))
    stale_days = float(ad.get("stale_warning_days", 1.5))
    now = dt.datetime.now()

    cameras: dict[str, dict[str, Any]] = {}
    for cid, c in configured.items():
        cameras[cid] = {**c, "id": cid, "discovered": False}

    if ad.get("enabled", True):
        for root_rel in ad.get("roots", ["aqara_video", "xiaomi_camera_videos"]):
            rdir = nas / root_rel
            if not rdir.is_dir():
                continue
            family = "aqara" if "aqara" in root_rel else "xiaomi"
            for entry in sorted(rdir.iterdir(), key=lambda p: p.name):
                if not entry.is_dir():
                    continue
                cid = entry.name
                if cid in ignore_ids:
                    continue
                if cid not in cameras:
                    cameras[cid] = {
                        "id": cid,
                        "name": f"新发现摄像头 ({cid})",
                        "family": family,
                        "relative_root": f"{root_rel}/{cid}",
                        "enabled": True,
                        "role": ad.get("default_role", "study_or_room"),
                        "zoom_policy": ad.get("default_zoom_policy", "gentle_if_wide"),
                        "zoom_factor": float(ad.get("default_zoom_factor", 1.35)),
                        "t1_res": int(ad.get("default_t1_res", 720)),
                        "milestone_watch": bool(ad.get("default_milestone_watch", True)),
                        "daytime_photos": True,
                        "discovered": True,
                    }

    health: list[dict[str, Any]] = []
    for cid, c in cameras.items():
        cdir = nas / c["relative_root"]
        if not c.get("enabled", True):
            health.append({
                "id": cid, "name": c["name"], "role": c["role"], "enabled": False,
                "status": "disabled", "discovered": c["discovered"],
                "latest_date": None, "latest_ts": None, "days_stale": None,
                "note": "已停用历史目录，不纳入日常处理",
            })
            continue
        if not cdir.is_dir():
            health.append({
                "id": cid, "name": c["name"], "role": c["role"], "enabled": True,
                "status": "missing_dir", "discovered": c["discovered"],
                "latest_date": None, "latest_ts": None, "days_stale": None,
                "note": "NAS 目录不存在",
            })
            continue
        if c["family"] == "aqara":
            ldate, lts, count = _latest_aqara_file(cdir)
        else:
            ldate, lts, count = _latest_xiaomi_file(cdir)
        days_stale = round((now - lts).total_seconds() / 86400.0, 2) if lts else None
        stale = days_stale is not None and days_stale > stale_days
        status = "stale_warning" if stale else ("online" if lts else "empty")
        note = ""
        if stale:
            note = f"已停更 {days_stale:.1f} 天（最后文件 {lts:%Y-%m-%d %H:%M}），请检查摄像头或 microSD/NAS 同步"
        elif c["discovered"]:
            note = "自动发现的新摄像头目录，已按默认室内策略纳入"
        elif c["role"] == "kitchen_4k":
            note = "4K 新机位，仅儿童确认片段进 T1，其余转 1fps 延时"
        elif c["role"] == "bedside_crib":
            note = "白天屏蔽静态床单误报，夜间纯睡眠进 T2 延时，仅哭声/抱起/醒来留 T1"
        else:
            note = "正常写入"
        health.append({
            "id": cid, "name": c["name"], "role": c["role"], "enabled": True,
            "zoom_policy": c.get("zoom_policy", "never"),
            "zoom_factor": float(c.get("zoom_factor", ad.get("default_zoom_factor", 1.0))),
            "t1_res": c.get("t1_res", 720),
            "status": status, "discovered": c["discovered"],
            "latest_date": ldate, "latest_ts": lts.isoformat(timespec="seconds") if lts else None,
            "latest_dir_files": count, "days_stale": days_stale, "note": note,
        })

    result = {
        "checked_at": now.isoformat(timespec="seconds"),
        "nas_root": str(nas),
        "cameras": cameras,
        "health": health,
        "study_g100_ready": True,
    }
    if out_root is not None:
        p = out_root / "v2" / "camera_health.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- Step 2: Explicit ASR Download & Keyword Spotting

def resolve_local_asr_dir(cfg: dict[str, Any]) -> tuple[Path, str]:
    asr = cfg["asr"]
    primary = PROJECT / asr["model_dir"]
    if (primary / "config.json").is_file() and (
        (primary / "model.safetensors").is_file() or (primary / "pytorch_model.bin").is_file()
    ):
        return primary, asr["model_id"]
    fallback = PROJECT / asr["fallback_model_dir"]
    if (fallback / "config.json").is_file() and (
        (fallback / "model.safetensors").is_file() or (fallback / "pytorch_model.bin").is_file()
    ):
        return fallback, asr["fallback_model_id"]
    raise FileNotFoundError(
        f"Local ASR model not found at {primary} or {fallback}; run `growth_archive download-asr` explicitly."
    )


def cmd_download_asr(args) -> None:
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cfg = load_v2_config(args.config)
    asr = cfg["asr"]
    use_fallback = bool(getattr(args, "fallback", False))
    model_id = asr["fallback_model_id"] if use_fallback else asr["model_id"]
    target_dir = PROJECT / (asr["fallback_model_dir"] if use_fallback else asr["model_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {target_dir} ...", flush=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model*", "tf_model*", "*.bin"],
    )
    # Verify offline load works immediately
    _ = WhisperProcessor.from_pretrained(str(target_dir), local_files_only=True)
    _ = WhisperForConditionalGeneration.from_pretrained(str(target_dir), local_files_only=True)
    print(f"Verified local_files_only load OK: {target_dir}", flush=True)


def _decode_pcm_window(path: str, start_s: float, duration_s: float) -> np.ndarray | None:
    cmd = [
        FF, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.2f}", "-t", f"{duration_s:.2f}",
        "-i", path, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    r = run_cmd(cmd, timeout=60)
    if r.returncode != 0 or not r.stdout:
        return None
    arr = np.frombuffer(r.stdout, dtype="<f4").copy()
    if arr.size < 4000:
        return None
    return arr


_KINSHIP_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("call_mama", "妈妈", re.compile(r"(妈妈|麻麻|妈咪)")),
    ("call_papa", "爸爸", re.compile(r"(爸爸|粑粑|爸比)")),
]
_SHORT_MAMA = re.compile(r"(?<![干他她它去买找你我])妈(?![烦了呢呀啊嘛])")
_SHORT_PAPA = re.compile(r"(?<![干他她它去买找你我])爸(?![烦了呢呀啊])")


def _match_kinship_in_memory(text: str, child_voc: float) -> tuple[str, str] | None:
    clean = re.sub(r"[\s，。！？、,.!?\"'“”‘’]+", "", text or "")
    if not clean:
        return None
    for ktype, label, pat in _KINSHIP_RULES:
        m = pat.search(clean)
        if m:
            return ktype, m.group(1)
    if len(clean) <= 6 and child_voc >= 0.25:
        if _SHORT_MAMA.search(clean):
            return "call_mama", "妈"
        if _SHORT_PAPA.search(clean):
            return "call_papa", "爸"
    return None


_yamnet_tls = threading.local()


def _scan_audio_windows(path: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    aud = cfg["audio"]
    det = getattr(_yamnet_tls, "det", None)
    if det is None:
        det = YamnetDetector(
            model_path=PROJECT / aud["yamnet_model"],
            class_map_path=PROJECT / aud["yamnet_class_map"],
            event_labels=aud["labels"],
            minimum_confidence=float(aud["minimum_confidence"]),
            merge_gap_seconds=1.0,
        )
        _yamnet_tls.det = det
    try:
        res = det.analyze(path, ffmpeg=FF, timeout_seconds=120)
    except Exception:
        return None
    if not res.events:
        return None
    # Group nearby speech/child_vocalization/laughter events into 3.5-8.0s candidate windows
    target = [e for e in res.events if e.label in ("child_vocalization", "laughter", "speech")]
    if not target:
        return None
    target.sort(key=lambda e: e.media_start_seconds)
    clusters: list[dict[str, Any]] = []
    for e in target:
        s = max(0.0, e.media_start_seconds - 1.0)
        en = e.media_end_seconds + 1.0
        if clusters and s - clusters[-1]["end"] <= 1.5 and (en - clusters[-1]["start"]) <= 8.0:
            c = clusters[-1]
            c["end"] = min(c["start"] + 8.0, max(c["end"], en))
            c[e.label] = max(c[e.label], e.confidence)
        else:
            if en - s < 3.5:
                en = s + 3.5
            clusters.append({
                "start": round(s, 2),
                "end": round(min(s + 8.0, en), 2),
                "child_vocalization": e.confidence if e.label == "child_vocalization" else 0.0,
                "laughter": e.confidence if e.label == "laughter" else 0.0,
                "speech": e.confidence if e.label == "speech" else 0.0,
            })
    return {"path": path, "windows": clusters}


def run_asr_milestones(cfg: dict[str, Any], out_root: Path, max_windows: int = 240) -> dict[str, Any]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    out_json = out_root / "v2" / "speech_milestones.json"
    nas = Path(cfg["project"]["nas_root"])

    # Collect high-yield candidate files across 2026-11-08, recent living room (11-10/11-11), and 2026-11-15
    candidates: list[tuple[str, str, float]] = []  # (camera_id, path, priority)
    fjsonl = out_root / "day-20261108" / "files.jsonl"
    if fjsonl.is_file():
        by_src = {}
        for line in fjsonl.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if "error" not in r:
                    by_src[r["source"]] = r
            except Exception:
                pass
        for r in by_src.values():
            cam = r["camera"]
            has_aud = bool(r.get("audio_events"))
            max_child = max((p["child"] for p in r.get("photo_candidates", [])), default=0.0)
            if cam == "B88880620430":
                if has_aud:
                    candidates.append((cam, r["source"], 1.5 + max_child))
            elif has_aud or max_child >= 0.50:
                # Daytime living room & bedroom priority
                st = r.get("start_hint", "")
                hour = int(st[11:13]) if len(st) >= 13 else 12
                day_bonus = 1.0 if 7 <= hour <= 22 else 0.2
                candidates.append((cam, r["source"], day_bonus + (1.0 if has_aud else 0.0) + max_child))

    # Also add daytime clips from living room 2026-11-10 & 2026-11-11, and bedroom/kitchen 2026-11-15
    extra_dirs = [
        ("lumi1.54ef4482e358", nas / "aqara_video/lumi1.54ef4482e358/20261111", 80),
        ("lumi1.54ef4482e358", nas / "aqara_video/lumi1.54ef4482e358/20261110", 60),
        ("lumi3.8e2ce0d05d9c251e", nas / "aqara_video/lumi3.8e2ce0d05d9c251e/20261115", 80),
        ("lumi3.932ece08fac40c05", nas / "aqara_video/lumi3.932ece08fac40c05/20261115", 60),
    ]
    seen_paths = {p for _, p, _ in candidates}
    for cam, d, cap in extra_dirs:
        if not d.is_dir():
            continue
        mp4s = [p for p in sorted(d.glob("*.mp4")) if "080000" <= p.stem <= "213000" and p.stat().st_size >= 1_500_000]
        if len(mp4s) > cap:
            step = len(mp4s) / cap
            mp4s = [mp4s[int(i * step)] for i in range(cap)]
        for p in mp4s:
            sp = str(p)
            if sp not in seen_paths:
                candidates.append((cam, sp, 1.4))
                seen_paths.add(sp)

    candidates.sort(key=lambda x: -x[2])
    scan_list = candidates[:420]
    print(f"[ASR] Scanning audio with YAMNet on {len(scan_list)} priority clips...", flush=True)

    win_cands: list[dict[str, Any]] = []
    cam_by_path = {p: cam for cam, p, _ in scan_list}
    with ThreadPoolExecutor(8) as pool:
        futs = [pool.submit(_scan_audio_windows, p, cfg) for _, p, _ in scan_list]
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            p = res["path"]
            cam = cam_by_path[p]
            for w in res["windows"]:
                score = w["child_vocalization"] * 2.5 + w["laughter"] * 1.8 + w["speech"] * 0.6
                win_cands.append({
                    "camera": cam,
                    "source": p,
                    **w,
                    "priority": score,
                })

    win_cands.sort(key=lambda x: -x["priority"])
    selected = win_cands[:max_windows]
    print(f"[ASR] Selected {len(selected)} vocalization windows (from {len(win_cands)} detected). Loading local Whisper...", flush=True)

    model_dir, model_id = resolve_local_asr_dir(cfg)
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float16
    ).to("cuda").eval()

    hits: list[dict[str, Any]] = []
    scanned_windows = 0
    try:
        # Decode audio windows in parallel CPU workers, transcribe sequentially in small batches
        with ThreadPoolExecutor(6) as pool:
            pcm_items = list(pool.map(
                lambda w: (w, _decode_pcm_window(w["source"], w["start"], w["end"] - w["start"])),
                selected,
            ))
        valid = [(w, arr) for w, arr in pcm_items if arr is not None]
        for i in range(0, len(valid), 8):
            batch = valid[i:i + 8]
            arrays = [arr for _, arr in batch]
            inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True)
            feats = inputs.input_features.to("cuda", dtype=torch.float16)
            with torch.inference_mode():
                gen_ids = model.generate(
                    feats,
                    language=cfg["asr"].get("language", "zh"),
                    task="transcribe",
                    max_new_tokens=48,
                )
            texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
            for (w, _), text in zip(batch, texts):
                scanned_windows += 1
                matched = _match_kinship_in_memory(text, w["child_vocalization"])
                # STRICT PRIVACY: immediately drop `text`; never store full transcript anywhere!
                del text
                if matched:
                    ktype, word = matched
                    hits.append({
                        "camera": w["camera"],
                        "source": w["source"],
                        "start_s": w["start"],
                        "end_s": w["end"],
                        "keyword_type": ktype,
                        "matched_word": word,
                        "child_vocalization_score": round(w["child_vocalization"], 3),
                        "laughter_score": round(w["laughter"], 3),
                        "speech_score": round(w["speech"], 3),
                        "speaker_hint": (
                            "child_vocalization_detected"
                            if w["child_vocalization"] >= 0.25
                            else "interaction_near_child_unverified_speaker"
                        ),
                    })
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    # Keep deduplicated best hits per file/time
    hits.sort(key=lambda h: -(h["child_vocalization_score"] * 2 + h["laughter_score"] + h["speech_score"]))
    deduped: list[dict[str, Any]] = []
    for h in hits:
        if any(d["source"] == h["source"] and abs(d["start_s"] - h["start_s"]) < 6.0 for d in deduped):
            continue
        deduped.append(h)

    payload = {
        "model_id": model_id,
        "local_dir": str(model_dir),
        "privacy_note": "On-demand memory-only matching; non-matching transcripts discarded without storage.",
        "files_scanned": len(scan_list),
        "candidate_windows_detected": len(win_cands),
        "windows_transcribed": scanned_windows,
        "matched_count": len(deduped),
        "hits": deduped,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ASR] Done: {scanned_windows} windows checked -> {len(deduped)} kinship hits saved to {out_json}", flush=True)
    return payload


def cmd_run_v2(args) -> None:
    from .growth_v2_pipeline import optimize_bedside_v2, process_kitchen_day, run_vlm_pipeline
    from .growth_v2_media import build_v2_deliverables, build_v2_page

    cfg = load_v2_config(args.config)
    out_root = ensure_v2_dirs(cfg)
    stage = args.stage

    if stage in ("all", "health"):
        discover_cameras(cfg, out_root)
    if stage in ("all", "kitchen"):
        process_kitchen_day(cfg, out_root, day_str="2026-11-15", workers=args.workers)
    if stage in ("all", "bedside"):
        optimize_bedside_v2(cfg, out_root, day_str="2026-11-08")
    if stage in ("all", "asr"):
        run_asr_milestones(cfg, out_root, max_windows=args.max_asr_windows)
    if stage in ("all", "vlm"):
        run_vlm_pipeline(cfg, out_root)
    if stage in ("all", "render"):
        build_v2_deliverables(cfg, out_root)
    if stage in ("all", "page"):
        build_v2_page(cfg, out_root)


def cmd_page_v2(args) -> None:
    from .growth_v2_media import build_v2_page

    cfg = load_v2_config(args.config)
    out_root = ensure_v2_dirs(cfg)
    discover_cameras(cfg, out_root)
    build_v2_page(cfg, out_root)
