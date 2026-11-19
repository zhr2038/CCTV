"""Growth archive V2 pipeline stages:
- Kitchen G350 (4K) 1fps T2 + candidate extraction & VLM-confirmed T1 encoding
- Bedside day/night optimization with static-box suppression
- Local Llama-server Qwen3.8-27B verification for milestones, photos, monthly cards, and smooth montage
"""
from __future__ import annotations

import base64
import datetime as dt
import gc
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from . import growth_archive as ga
from . import growth_v2 as v2
from .audio_events import YamnetDetector

PROJECT = v2.PROJECT
FF = v2.FF
FP = v2.FP

T1_CAPPED_VIDEO = ["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                   "-cq", "39", "-b:v", "0", "-maxrate", "650k", "-bufsize", "1300k", "-g", "300"]


def _box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
    bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _decode_t2_pair(path: str, t2_path: Path, dur: float, w: int, h: int) -> list[np.ndarray]:
    t2_path.parent.mkdir(parents=True, exist_ok=True)
    limit_s = max(180.0, dur * 0.5)
    for hw in (True, False):
        cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if hw:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-i", path, "-an", "-vf", f"fps=1,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        dec = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        enc = subprocess.Popen(
            [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0", *ga.T2_VIDEO,
             "-pix_fmt", "yuv420p", str(t2_path)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        timed_out = threading.Event()

        def kill_pair() -> None:
            timed_out.set()
            for p in (dec, enc):
                try:
                    p.kill()
                except OSError:
                    pass

        timer = threading.Timer(limit_s, kill_pair)
        timer.daemon = True
        timer.start()
        size = w * h * 3
        frames: list[np.ndarray] = []
        try:
            while True:
                buf = dec.stdout.read(size)
                if len(buf) < size:
                    break
                frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy())
                enc.stdin.write(buf)
        except OSError:
            pass
        finally:
            timer.cancel()
            try:
                enc.stdin.close()
            except OSError:
                pass
            dec.wait()
            enc.wait()
        if not timed_out.is_set() and dec.returncode == 0 and enc.returncode == 0 and frames:
            return frames
        if t2_path.exists():
            t2_path.unlink()
    return []


# --------------------------------------------------------------------------- Step 3A: Kitchen G350 (2026-11-15)

def _kitchen_one_file(models: ga.Models, cfg: dict[str, Any], path: str, day: dt.date, kdir: Path) -> dict[str, Any]:
    import cv2
    t0 = time.time()
    info = ga.probe(path)
    rec: dict[str, Any] = {"camera": "lumi3.932ece08fac40c05", "source": path, "size": os.path.getsize(path), **info}
    if info["duration"] <= 0 or not info["width"]:
        rec["error"] = "unreadable"
        return rec
    stem = Path(path).stem
    w = 1280
    h = ga.even(info["height"] * w / info["width"])
    t2 = kdir / "t2" / f"{stem}.mp4"
    frames = _decode_t2_pair(path, t2, info["duration"], w, h)
    if not frames:
        rec["error"] = "no frames decoded"
        return rec
    n = len(frames)
    boxes = models.persons(frames)
    person_s = sum(1 for bx in boxes if any(b[0] >= ga.PERSON_CONF for b in bx))

    # Audio scan with YAMNet (child_vocalization, laughter, speech, baby_cry)
    aud_res = v2._scan_audio_windows(path, cfg) if info["has_audio"] else None
    max_cvoc = max((w_["child_vocalization"] for w_ in aud_res["windows"]), default=0.0) if aud_res else 0.0
    max_laugh = max((w_["laughter"] for w_ in aud_res["windows"]), default=0.0) if aud_res else 0.0
    max_speech = max((w_["speech"] for w_ in aud_res["windows"]), default=0.0) if aud_res else 0.0

    # Check child probability on up to 3 person frames (prefer frames with smaller/multiple boxes or non-close-up)
    cands = []
    for i in range(n):
        good = [b for b in boxes[i] if b[0] >= ga.PERSON_CONF]
        if not good:
            continue
        # Kitchen counter has adult backs occupying >35% area; also score frames with smaller/mid boxes
        best_b = max(good, key=lambda b: b[0])
        area = max((b[3] - b[1]) * (b[4] - b[2]) for b in good)
        f = frames[i]
        x1, y1, x2, y2 = int(best_b[1] * w), int(best_b[2] * h), int(best_b[3] * w), int(best_b[4] * h)
        crop = cv2.cvtColor(f[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        score_rank = best_b[0] * min(1.0, sharp / 120.0) * (1.2 if len(good) >= 2 else 1.0)
        cands.append((score_rank, i, good, area, sharp))
    cands.sort(key=lambda c: -c[0])
    picked = []
    for c in cands:
        if all(abs(c[1] - p[1]) >= 4 for p in picked):
            picked.append(c)
        if len(picked) == 3:
            break

    photo_cands = []
    cdir = kdir / "work" / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    for _, i, good, area, sharp in picked:
        cs = models.score_frame(frames[i], good)
        jp = cdir / f"{stem}_{i:04d}.jpg"
        cv2.imwrite(str(jp), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 86])
        photo_cands.append({"t": float(i), "child": round(cs, 4), "area": round(area, 3),
                            "n_boxes": len(good), "sharp": round(sharp, 1), "jpg": str(jp)})

    st = ga.file_start("lumi3.932ece08fac40c05", path, day)
    rec.update({
        "frames_1fps": n,
        "start_hint": st.isoformat() if st else None,
        "person_seconds": person_s,
        "max_child": max((p["child"] for p in photo_cands), default=0.0),
        "audio_max_child_voc": round(max_cvoc, 3),
        "audio_max_laughter": round(max_laugh, 3),
        "audio_max_speech": round(max_speech, 3),
        "photo_candidates": photo_cands,
        "t2": {"file": str(t2), "size": t2.stat().st_size if t2.exists() else 0},
        "elapsed": round(time.time() - t0, 1),
    })
    return rec


def process_kitchen_day(cfg: dict[str, Any], out_root: Path, day_str: str = "2026-11-15", workers: int = 6) -> dict[str, Any]:
    day = dt.date.fromisoformat(day_str)
    kdir = out_root / "v2" / f"kitchen_{day:%Y%m%d}"
    kdir.mkdir(parents=True, exist_ok=True)
    manifest = kdir / "files.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:
                pass
    nas_dir = Path(cfg["project"]["nas_root"]) / "aqara_video/lumi3.932ece08fac40c05" / f"{day:%Y%m%d}"
    all_files = sorted(str(p) for p in nas_dir.glob("*.mp4")) if nas_dir.is_dir() else []
    jobs = [f for f in all_files if f not in done]
    print(f"[Kitchen] {len(jobs)} files to scan for {day_str} ({len(done)} already done)", flush=True)
    if jobs:
        models = ga.Models(siglip=True)
        lock = threading.Lock()
        t0 = time.time()
        try:
            with ThreadPoolExecutor(workers) as pool, manifest.open("a", encoding="utf-8") as mf:
                futs = {pool.submit(_kitchen_one_file, models, cfg, f, day, kdir): f for f in jobs}
                for k, fut in enumerate(as_completed(futs), 1):
                    try:
                        rec = fut.result()
                    except Exception as exc:
                        rec = {"source": futs[fut], "error": repr(exc)}
                    with lock:
                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        mf.flush()
                    if k % 50 == 0 or k == len(jobs):
                        print(f"[Kitchen] {k}/{len(jobs)} files, {time.time() - t0:.0f}s", flush=True)
        finally:
            del models
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    rows = [r for r in ga.load_jsonl(manifest) if "error" not in r]
    rows.sort(key=lambda r: r.get("start_hint") or "")
    lst = kdir / "work" / "t2_list.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("".join(f"file '{Path(r['t2']['file']).as_posix()}'\n" for r in rows if r["t2"]["size"]), encoding="utf-8")
    lapse = kdir / f"timelapse_kitchen_g350_{day:%Y%m%d}.mp4"
    if not lapse.exists():
        v2.run_cmd([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(lapse)])

    orig_bytes = sum(r["size"] for r in rows)
    t2_bytes = sum(r["t2"]["size"] for r in rows)
    hours = sum(r["duration"] for r in rows) / 3600.0
    person_files = sum(1 for r in rows if r.get("person_seconds", 0) > 0)
    cands = [r for r in rows if r.get("max_child", 0) >= 0.45 or r.get("audio_max_child_voc", 0) >= 0.25
             or r.get("audio_max_laughter", 0) >= 0.25]
    summary = {
        "day": day_str,
        "camera": "lumi3.932ece08fac40c05",
        "files": len(rows),
        "errors": sum(1 for r in ga.load_jsonl(manifest) if "error" in r),
        "hours": round(hours, 2),
        "orig_bytes": orig_bytes,
        "t2_bytes": t2_bytes,
        "person_files": person_files,
        "candidate_child_files": len(cands),
        "timelapse": lapse.name if lapse.exists() else None,
        "t1_bytes": 0,
        "confirmed_child_files": 0,
    }
    (kdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Kitchen] Scanned {len(rows)} files ({orig_bytes/1e9:.2f} GB) -> T2={t2_bytes/1e9:.3f} GB, candidates={len(cands)}", flush=True)
    return summary


# --------------------------------------------------------------------------- Step 3B: Bedside Day/Night V2 Optimization

_yamnet_bed_tls = threading.local()


def _bedside_audio_per_sec(path: str, n_sec: int, cfg: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    aud = cfg["audio"]
    det = getattr(_yamnet_bed_tls, "det", None)
    if det is None:
        det = YamnetDetector(
            model_path=PROJECT / aud["yamnet_model"],
            class_map_path=PROJECT / aud["yamnet_class_map"],
            event_labels=["baby_cry", "child_vocalization", "laughter", "speech"],
            minimum_confidence=0.25,
            merge_gap_seconds=1.0,
        )
        _yamnet_bed_tls.det = det
    mask = np.zeros(n_sec, bool)
    ev_list: list[dict[str, Any]] = []
    try:
        res = det.analyze(path, ffmpeg=FF, timeout_seconds=180)
        for e in res.events:
            # Bedside crib camera ignores generic adult speech (room ceiling camera covers room dialogue)
            trigger = (
                (e.label == "baby_cry" and e.confidence >= 0.30)
                or (e.label == "child_vocalization" and e.confidence >= 0.28)
                or (e.label == "laughter" and e.confidence >= 0.30)
            )
            if trigger:
                s = max(0, int(e.media_start_seconds))
                en = min(n_sec, int(np.ceil(e.media_end_seconds)))
                mask[s:en] = True
                ev_list.append({"label": e.label, "s": round(e.media_start_seconds, 2),
                                "e": round(e.media_end_seconds, 2), "conf": round(e.confidence, 3)})
    except Exception:
        pass
    return mask, ev_list


def _merge_segs(active: np.ndarray, duration: float, pad: float = 2.0, gap: float = 6.0, min_seg: float = 3.0) -> list[list[float]]:
    segs: list[list[float]] = []
    for i in np.flatnonzero(active):
        s, e = max(0.0, i - pad), min(duration, i + 1 + pad)
        if segs and s - segs[-1][1] <= gap:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    out = []
    for s, e in segs:
        if e - s < min_seg:
            e = min(duration, s + min_seg)
        out.append([round(s, 2), round(e, 2)])
    return out


def optimize_bedside_v2(cfg: dict[str, Any], out_root: Path, day_str: str = "2026-11-08") -> dict[str, Any]:
    import cv2
    import torch

    bdir = out_root / "v2" / "bedside_v2"
    (bdir / "t1").mkdir(parents=True, exist_ok=True)
    summary_p = bdir / "summary.json"
    rows = [r for r in ga.load_jsonl(out_root / f"day-{day_str.replace('-', '')}" / "files.jsonl")
            if "error" not in r and r["camera"] == "B88880620430"]
    rows.sort(key=lambda r: r.get("start_hint") or "")
    print(f"[Bedside V2] Analyzing {len(rows)} bedside bundles with static-box suppression + day/night rules...", flush=True)

    models = ga.Models(siglip=True)
    results: list[dict[str, Any]] = []
    day_hrs = cfg["cameras"]["B88880620430"].get("day_hours", [8, 20])
    try:
        # Run YAMNet across bedside files in parallel first
        aud_map: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
        with ThreadPoolExecutor(6) as pool:
            futs = {pool.submit(_bedside_audio_per_sec, r["source"], int(r["frames_1fps"]), cfg): r["source"]
                    for r in rows if r.get("has_audio")}
            for fut in as_completed(futs):
                aud_map[futs[fut]] = fut.result()

        for idx, r in enumerate(rows, 1):
            t2_path = r["t2"]["file"]
            cap = cv2.VideoCapture(t2_path)
            frames: list[np.ndarray] = []
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                frames.append(frm)
            cap.release()
            n = len(frames)
            if n == 0:
                continue
            h, w = frames[0].shape[:2]
            boxes = models.persons(frames)
            grays = [cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (320, ga.even(320 * h / w))), cv2.COLOR_BGR2GRAY), (5, 5), 0)
                     for f in frames]
            gh, gw = grays[0].shape
            diffs = [None] + [cv2.absdiff(grays[i], grays[i - 1]) > 12 for i in range(1, n)]
            gmot = np.array([0.0] + [float(d.mean()) for d in diffs[1:]])
            if n > 1:
                gmot[0] = gmot[1]

            # Rolling 45s static-box suppression
            good_boxes = [[b for b in boxes[i] if b[0] >= ga.PERSON_CONF] for i in range(n)]
            nonstatic_boxes: list[list[tuple[float, ...]]] = [[] for _ in range(n)]
            nonstatic_bmot = np.zeros(n)
            static_suppressed_s = 0

            for i in range(n):
                lo, hi = max(0, i - 22), min(n, i + 23)
                win_len = max(1, hi - lo)
                d_mat = diffs[i] if i > 0 else (diffs[1] if n > 1 else None)
                had_static_only = False
                for b in good_boxes[i]:
                    xa, ya = int(b[1] * gw), int(b[2] * gh)
                    xb, yb = max(xa + 1, int(b[3] * gw)), max(ya + 1, int(b[4] * gh))
                    bm = float(d_mat[ya:yb, xa:xb].mean()) if d_mat is not None else 0.0
                    same_cnt = sum(1 for j in range(lo, hi) if any(_box_iou(b, b2) >= 0.80 for b2 in good_boxes[j]))
                    if same_cnt / win_len >= 0.60 and bm < 0.025:
                        had_static_only = True
                        continue
                    nonstatic_boxes[i].append(b)
                    nonstatic_bmot[i] = max(nonstatic_bmot[i], bm)
                if had_static_only and not nonstatic_boxes[i]:
                    static_suppressed_s += 1

            aud_mask, aud_evs = aud_map.get(r["source"], (np.zeros(n, bool), []))
            if len(aud_mask) < n:
                pad = np.zeros(n, bool)
                pad[:len(aud_mask)] = aud_mask
                aud_mask = pad

            st = dt.datetime.fromisoformat(r["start_hint"]) if r.get("start_hint") else dt.datetime(2026, 11, 8, 0, 0)
            active = np.zeros(n, bool)
            day_child_check_idx = [
                i for i in range(n)
                if (day_hrs[0] <= (st + dt.timedelta(seconds=i)).hour < day_hrs[1])
                and nonstatic_boxes[i] and nonstatic_bmot[i] >= 0.055 and not aud_mask[i]
            ]
            # Check SigLIP child score ONLY on candidate daytime non-static active frames
            day_child_ok: set[int] = set()
            for i in day_child_check_idx[::2]:  # check every 2s for fast daytime confirmation
                if models.score_frame(frames[i], nonstatic_boxes[i]) >= 0.60:
                    day_child_ok.update((i, min(n - 1, i + 1)))

            night_sleep_t2_s = 0
            day_filtered_s = 0
            for i in range(n):
                hr = (st + dt.timedelta(seconds=i)).hour
                is_day = day_hrs[0] <= hr < day_hrs[1]
                if aud_mask[i]:
                    active[i] = True
                elif is_day:
                    if i in day_child_ok:
                        active[i] = True
                    elif good_boxes[i]:
                        day_filtered_s += 1
                else:
                    if (nonstatic_boxes[i] and nonstatic_bmot[i] >= 0.060) or (not nonstatic_boxes[i] and gmot[i] >= 0.035):
                        active[i] = True
                    else:
                        night_sleep_t2_s += 1

            segs = _merge_segs(active, r["duration"])
            stem = Path(r["source"]).stem
            dst = bdir / "t1" / f"{stem}.mp4"
            t1_s = sum(e - s for s, e in segs)

            results.append({
                "source": r["source"],
                "start_hint": r.get("start_hint"),
                "duration": r["duration"],
                "orig_size": r["size"],
                "v1_t1_size": r["t1"].get("size", 0),
                "v1_t1_s": r["t1"].get("seconds", 0.0),
                "v2_t1_size": dst.stat().st_size if dst.exists() else 0,
                "v2_t1_s": round(t1_s, 2),
                "t2_size": r["t2"]["size"],
                "static_suppressed_s": static_suppressed_s,
                "day_filtered_s": day_filtered_s,
                "night_sleep_t2_s": night_sleep_t2_s,
                "segments": segs,
                "audio_triggers": aud_evs[:20],
                "v2_t1_file": str(dst) if segs else None,
            })
            if idx % 10 == 0 or idx == len(rows):
                print(f"[Bedside V2] analyzed {idx}/{len(rows)} bundles", flush=True)
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()

    # Parallel FFmpeg T1 encoding across 6 workers
    def _encode_item(item: dict[str, Any]) -> None:
        segs = item["segments"]
        if not segs or not item["v2_t1_file"]:
            return
        dst = Path(item["v2_t1_file"])
        if dst.exists() and dst.stat().st_size > 0:
            item["v2_t1_size"] = dst.stat().st_size
            return
        tmp_dst = dst.with_suffix(".tmp.mp4")
        whole = len(segs) == 1 and segs[0][0] <= 0.5 and segs[0][1] >= item["duration"] - 0.5
        cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "cuda", "-i", item["source"],
               "-map", "0:v:0", "-map", "0:a:0?"]
        if whole:
            cmd += ["-vf", ga.T1_FILTER]
        else:
            expr = "+".join(f"between(t\\,{s:.2f}\\,{e:.2f})" for s, e in segs)
            cmd += ["-vf", f"select={expr},setpts=N/FRAME_RATE/TB,{ga.T1_FILTER}",
                    "-af", f"aselect={expr},asetpts=N/SR/TB"]
        cmd += [*T1_CAPPED_VIDEO, "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "24k", "-ac", "1", str(tmp_dst)]
        res_p = v2.run_cmd(cmd, timeout=900)
        if res_p.returncode != 0 or not tmp_dst.exists() or tmp_dst.stat().st_size == 0:
            res_p = v2.run_cmd([c for c in cmd if c not in ("-hwaccel", "cuda")], timeout=900)
        if tmp_dst.exists() and tmp_dst.stat().st_size > 0:
            os.replace(tmp_dst, dst)
            item["v2_t1_size"] = dst.stat().st_size
        elif tmp_dst.exists():
            tmp_dst.unlink()

    to_enc = [x for x in results if x["segments"]]
    print(f"[Bedside V2] Encoding {len(to_enc)} non-empty T1 bundles across 6 workers...", flush=True)
    with ThreadPoolExecutor(6) as pool:
        futs = [pool.submit(_encode_item, it) for it in to_enc]
        for k, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if k % 10 == 0 or k == len(to_enc):
                print(f"[Bedside V2] encoded {k}/{len(to_enc)} T1 files", flush=True)
        torch.cuda.empty_cache()

    v1_t1 = sum(x["v1_t1_size"] for x in results)
    v2_t1 = sum(x["v2_t1_size"] for x in results)
    t2_tot = sum(x["t2_size"] for x in results)
    orig_tot = sum(x["orig_size"] for x in results)
    payload = {
        "day": day_str,
        "files": len(results),
        "orig_bytes": orig_tot,
        "t2_bytes": t2_tot,
        "v1_t1_bytes": v1_t1,
        "v1_t1_hours": round(sum(x["v1_t1_s"] for x in results) / 3600, 2),
        "v1_total_bytes": v1_t1 + t2_tot,
        "v1_ratio": round(orig_tot / max(1, v1_t1 + t2_tot), 2),
        "v2_t1_bytes": v2_t1,
        "v2_t1_hours": round(sum(x["v2_t1_s"] for x in results) / 3600, 2),
        "v2_total_bytes": v2_t1 + t2_tot,
        "v2_ratio": round(orig_tot / max(1, v2_t1 + t2_tot), 2),
        "day_filtered_blanket_hours": round(sum(x["day_filtered_s"] for x in results) / 3600, 2),
        "night_sleep_to_t2_hours": round(sum(x["night_sleep_t2_s"] for x in results) / 3600, 2),
        "static_suppressed_hours": round(sum(x["static_suppressed_s"] for x in results) / 3600, 2),
        "items": results,
    }
    summary_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Bedside V2] {orig_tot/1e9:.2f}GB -> V1 { (v1_t1+t2_tot)/1e9:.2f}GB ({payload['v1_ratio']}x) "
          f"-> V2 {(v2_t1+t2_tot)/1e9:.2f}GB ({payload['v2_ratio']}x)", flush=True)
    return payload


# --------------------------------------------------------------------------- Step 4: Local Qwen3.8-27B VLM Server & Verification

def _parse_json_obj(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        a = s.find("{")
        if a < 0:
            raise
        depth, in_str, esc, end = 0, False, False, -1
        for idx, ch in enumerate(s[a:], start=a):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
            elif not in_str and ch == "{":
                depth += 1
            elif not in_str and ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end < 0:
            raise
        obj = json.loads(s[a:end])
    if not isinstance(obj, dict):
        raise ValueError("VLM output is not a JSON object")
    return obj


class LlamaVLM:
    def __init__(self, cfg: dict[str, Any], cache_path: Path, use_fallback: bool = False) -> None:
        self.cfg = cfg
        self.v = cfg["vlm"]
        self.cache_path = cache_path
        self.use_fallback = use_fallback
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["response"]
                except Exception:
                    pass
        self._proc: subprocess.Popen | None = None
        self._session = None
        self._url = f"http://127.0.0.1:{int(self.v['port'])}"

    @contextmanager
    def running(self):
        import requests
        port = int(self.v["port"])
        with socket.socket() as sk:
            try:
                sk.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"VLM port {port} is already occupied") from exc

        runtime = PROJECT / self.v["runtime"]
        if not self.use_fallback:
            model = PROJECT / self.v["model"]
            proj = PROJECT / self.v["projector"]
            if self.v.get("model_sha256") and v2.file_sha256(model) != self.v["model_sha256"]:
                raise RuntimeError("Pinned Qwen3.8-27B model sha256 mismatch")
            if self.v.get("projector_sha256") and v2.file_sha256(proj) != self.v["projector_sha256"]:
                raise RuntimeError("Pinned Qwen3.8-27B projector sha256 mismatch")
        else:
            model = PROJECT / self.v["fallback_model"]
            proj = PROJECT / self.v["fallback_projector"]

        args = [
            str(runtime), "-m", str(model), "--mmproj", str(proj), "-ngl", "99",
            "-c", str(self.v["context_tokens"]), "--parallel", "1",
            "--host", "127.0.0.1", "--port", str(port),
            "--flash-attn", "on", "--batch-size", "1024", "--ubatch-size", "256", "--no-webui",
        ]
        log_dir = self.cache_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "llama_vlm.out.log").open("w", encoding="utf-8") as so, \
             (log_dir / "llama_vlm.err.log").open("w", encoding="utf-8") as se:
            try:
                self._proc = subprocess.Popen(
                    args, cwd=runtime.parent, stdout=so, stderr=se,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                sess = requests.Session()
                sess.trust_env = False
                self._session = sess
                for _ in range(150):
                    if self._proc.poll() is not None:
                        raise RuntimeError(f"llama-server exited early (code={self._proc.returncode})")
                    try:
                        if sess.get(f"{self._url}/health", timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("llama-server startup timed out")
                print(f"[VLM] llama-server ready ({model.name}) on {self._url}", flush=True)
                yield self
            finally:
                if self._proc is not None and self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=10)
                print("[VLM] llama-server stopped and VRAM released", flush=True)

    def ask(self, prompt_id: str, frames_jpeg: list[bytes], prompt: str) -> dict[str, Any]:
        key_h = hashlib.sha256(prompt_id.encode() + prompt.encode())
        for jb in frames_jpeg:
            key_h.update(jb)
        key = key_h.hexdigest()
        if key in self.cache:
            return self.cache[key]

        content: list[dict[str, Any]] = []
        for idx, jb in enumerate(frames_jpeg):
            content.append({"type": "text", "text": f"frame_{idx + 1}"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jb).decode()}})
        content.append({"type": "text", "text": prompt})
        payload = {
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": int(self.v["max_output_tokens"]),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self._session.post(f"{self._url}/v1/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        raw_txt = data["choices"][0]["message"]["content"]
        parsed = _parse_json_obj(raw_txt)
        self.cache[key] = parsed
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "prompt_id": prompt_id, "response": parsed}, ensure_ascii=False) + "\n")
        return parsed


def extract_jpegs(path: str, times_s: list[float], max_edge: int = 768) -> list[bytes]:
    from PIL import Image
    import cv2

    out: list[bytes] = []
    for t in times_s:
        r = v2.run_cmd(
            [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}", "-i", path,
             "-frames:v", "1", "-vf", f"scale='min({max_edge},iw)':'min({max_edge},ih)':force_original_aspect_ratio=decrease",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"],
            timeout=45,
        )
        if r.returncode == 0 and r.stdout:
            out.append(r.stdout)
    if not out:
        cap = cv2.VideoCapture(path)
        for t in times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frm = cap.read()
            if ok:
                im = Image.fromarray(frm[:, :, ::-1])
                im.thumbnail((max_edge, max_edge))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=88)
                out.append(buf.getvalue())
        cap.release()
    return out


# --------------------------------------------------------------------------- Step 4 Sub-pipelines

PROMPT_MILESTONE = """客观观察这组按时间顺序排列的家庭监控连续画面（只写画面里直接可见的事实，不要编造未出现的动作）。
注意区分成人和婴幼儿（约0-14个月宝宝）：成人弯腰/蹲下或画面远处的大人不能算宝宝；成人抱起/搀扶的动作必须在independence填assisted。
只返回单个JSON对象：
{
  "child_clear": "yes|partial|blocked|no",
  "child_box": [x1, y1, x2, y2],
  "categories": ["standing"|"walking"|"sofa_descent"|"toy_play"|"happy_interaction"|"kitchen_visit"|"none"],
  "independence": "independent|assisted|unclear",
  "excitement_score": 0,
  "private_care": "yes|no",
  "caption_zh": "简短客观中文描述，写清宝宝在做什么及是否有成人搀扶/互动"
}
坐标x1,y1,x2,y2为0到1之间归一化数值，框住宝宝主体（无宝宝时填[0,0,0,0]）。
standing=双脚站立或扶物站立；walking=迈步走/扶走；sofa_descent=从沙发/床沿下地或爬上爬下；toy_play=宝宝专注抓握/玩耍/推玩玩具；happy_interaction=明显开心互动/逗笑；kitchen_visit=宝宝出现在厨房。private_care=换尿布/洗浴/裸露照护。"""

PROMPT_PHOTO = """评估这张家庭监控原画截图是否适合作为“今日成长精选照片”。
只返回单个JSON对象：
{
  "child_clear": "yes|blocked|no",
  "face_or_expression_clear": "yes|side|no",
  "child_box": [x1, y1, x2, y2],
  "adult_back_or_awkward": "yes|no",
  "meaningful_action": "yes|no",
  "private_care": "yes|no",
  "photo_score": 0,
  "caption_zh": "简短客观中文图注"
}
如果宝宝被大人背影遮挡、在角落边缘只露出一角、画面杂乱无意义（空床/大人背影走过/只拍到枕头被子），adult_back_or_awkward填yes或meaningful_action填no，photo_score给0-2。
只有当宝宝主体清晰、姿态自然（如玩玩具、抬头、站立、爬行、开心互动）且无遮挡时，photo_score才给4或5。坐标x1,y1,x2,y2归一化到0..1。"""

PROMPT_MONTH = """这是同一个宝宝某个月份的3张连续监控帧。请判断宝宝是否清晰可见且未被大人背影或物体大面积遮挡。
只返回单个JSON对象：
{
  "child_clear": "yes|blocked|no",
  "child_box": [x1, y1, x2, y2],
  "private_care": "yes|no",
  "clarity_score": 0,
  "stage_zh": "10字以内客观姿态描述，如：仰卧挥动小手、俯卧抬头、独坐玩玩具、扶站迈步"
}
坐标x1,y1,x2,y2归一化到0..1框住宝宝主体（多保留一点周边动作空间）。换尿布/裸露填private_care=yes。"""


def _mine_milestone_candidates(cfg: dict[str, Any], out_root: Path) -> list[dict[str, Any]]:
    """Mine candidate 6-8s windows across 11-08 (living/bedroom/bedside), 11-09..11-11 (living), 11-15 (kitchen/bedroom), and ASR hits."""
    nas = Path(cfg["project"]["nas_root"])
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    def add_cand(cam: str, src: str, start_s: float, end_s: float, hint: str, priority: float, time_str: str = "") -> None:
        key = (src, int(start_s // 5))
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append({
            "camera": cam, "source": src, "start_s": round(max(0.0, start_s), 2),
            "end_s": round(end_s, 2), "hint": hint, "priority": round(priority, 3), "time_str": time_str,
        })

    # 1. All kinship speech hits first!
    asr_p = out_root / "v2" / "speech_milestones.json"
    if asr_p.is_file():
        asr_data = json.loads(asr_p.read_text(encoding="utf-8"))
        for h in asr_data.get("hits", [])[:15]:
            s = max(0.0, h["start_s"] - 1.5)
            e = max(s + 6.0, h["end_s"] + 1.5)
            add_cand(h["camera"], h["source"], s, e, f"speech:{h['keyword_type']}:{h['matched_word']}",
                     10.0 + h["child_vocalization_score"] * 3, "")

    # 2. 2026-11-08 living room + bedroom ceiling from files.jsonl (high child score + active motion + laughter)
    fjsonl = out_root / "day-20261108" / "files.jsonl"
    if fjsonl.is_file():
        rows = [r for r in ga.load_jsonl(fjsonl) if "error" not in r]
        living_best = []
        bed_best = []
        for r in rows:
            max_c = max((p["child"] for p in r.get("photo_candidates", [])), default=0.0)
            pcs = r.get("photo_candidates") or [{"t": r["duration"] / 2, "child": 0.0, "quality": 0.0}]
            best_p = max(pcs, key=lambda p: p["child"] * (0.4 + 0.6 * min(1.0, p.get("quality", 0) / 0.5)))
            st = r.get("start_hint", "")
            hr = int(st[11:13]) if len(st) >= 13 else 12
            if not (7 <= hr <= 22):
                continue
            sc = max_c * 2.0 + min(1.0, r.get("active_seconds", 0) / 15.0) + (0.6 if r.get("audio_events") else 0.0)
            if r["camera"] == "lumi1.54ef4482e358" and max_c >= 0.48:
                living_best.append((sc, r, best_p["t"]))
            elif r["camera"] == "lumi3.8e2ce0d05d9c251e" and max_c >= 0.58:
                bed_best.append((sc, r, best_p["t"]))
        living_best.sort(key=lambda x: -x[0])
        bed_best.sort(key=lambda x: -x[0])
        for sc, r, t in living_best[:24]:
            s = max(0.0, t - 3.5)
            e = min(r["duration"], s + 8.0)
            if e - s >= 4.0:
                add_cand(r["camera"], r["source"], s, e, "living_active", 5.0 + sc, ga.wall_label(r, s))
        for sc, r, t in bed_best[:12]:
            s = max(0.0, t - 3.5)
            e = min(r["duration"], s + 8.0)
            if e - s >= 4.0:
                add_cand(r["camera"], r["source"], s, e, "bedroom_active", 4.5 + sc, ga.wall_label(r, s))

    # 3. 2026-11-15 Kitchen G350 top child candidates
    kjsonl = out_root / "v2" / "kitchen_20261115" / "files.jsonl"
    if kjsonl.is_file():
        krows = [r for r in ga.load_jsonl(kjsonl) if "error" not in r]
        kcands = [r for r in krows if r.get("max_child", 0) >= 0.45 or r.get("audio_max_child_voc", 0) >= 0.25 or r.get("audio_max_laughter", 0) >= 0.25]
        kcands.sort(key=lambda r: -(r.get("max_child", 0) * 2 + r.get("audio_max_child_voc", 0) * 2 + r.get("audio_max_laughter", 0)))
        for r in kcands[:24]:
            t = r["photo_candidates"][0]["t"] if r.get("photo_candidates") else r["duration"] / 2
            s = max(0.0, t - 3.5)
            e = min(r["duration"], s + 8.0)
            add_cand("lumi3.932ece08fac40c05", r["source"], s, e, "kitchen_candidate",
                     6.0 + r.get("max_child", 0), ga.wall_label(r, s))

    # 4. Also sample 2026-11-09..11-11 daytime living room clips via quick YOLO+SigLIP (for sofa descent / walking / standing!)
    l_extra: list[str] = []
    for dstr in ("20261111", "20261110", "20261109"):
        ldir = nas / "aqara_video/lumi1.54ef4482e358" / dstr
        if ldir.is_dir():
            day_mp4s = [p for p in sorted(ldir.glob("*.mp4")) if "090000" <= p.stem <= "203000" and p.stat().st_size >= 3_000_000]
            if len(day_mp4s) > 18:
                step = len(day_mp4s) / 18
                day_mp4s = [day_mp4s[int(i * step)] for i in range(18)]
            l_extra.extend(str(p) for p in day_mp4s)
    if l_extra:
        models = ga.Models(siglip=True)
        try:
            with ThreadPoolExecutor(6) as pool:
                grabs = list(pool.map(lambda p: (p, ga.grab(p, 3.5, 1280)), l_extra))
            valid = [(p, im) for p, im in grabs if im is not None]
            if valid:
                bxs = models.persons([im for _, im in valid])
                for (p, im), bx in zip(valid, bxs):
                    good = [b for b in bx if b[0] >= ga.PERSON_CONF]
                    if not good:
                        continue
                    cs = models.score_frame(im, good)
                    if cs >= 0.55:
                        info = ga.probe(p)
                        dur = info["duration"]
                        e = min(dur, 8.5)
                        if e >= 4.0:
                            stem = Path(p).stem
                            dname = Path(p).parent.name
                            tstr = f"{dname[:4]}-{dname[4:6]}-{dname[6:8]} {stem[:2]}:{stem[2:4]}:{stem[4:6]}"
                            add_cand("lumi1.54ef4482e358", p, 0.5, e, "living_recent_day", 5.2 + cs, tstr)
        finally:
            del models
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    candidates.sort(key=lambda c: -c["priority"])
    return candidates


def _mine_photo_candidates(out_root: Path) -> list[dict[str, Any]]:
    """Strict pre-filter BEFORE VLM: no bedside daytime, away from edges, high crop sharpness, color."""
    import cv2

    rows = [r for r in ga.load_jsonl(out_root / "day-20261108" / "files.jsonl")
            if "error" not in r and r["camera"] != "B88880620430"]
    cands = []
    for r in rows:
        st = dt.datetime.fromisoformat(r["start_hint"]) if r.get("start_hint") else None
        for pc in r.get("photo_candidates", []):
            if pc["child"] < 0.50 or pc["sat"] < 20.0 or pc["sharp"] < 40.0:
                continue
            jp = Path(pc["jpg"])
            if not jp.is_file():
                continue
            when = (st + dt.timedelta(seconds=pc["t"])) if st else None
            cands.append({
                "camera": r["camera"], "source": r["source"], "t": pc["t"],
                "child": pc["child"], "quality": pc["quality"], "sat": pc["sat"], "sharp": pc["sharp"],
                "jpg": str(jp), "when": when.isoformat() if when else "", "time_str": ga.wall_label(r, pc["t"]),
            })
    cands.sort(key=lambda c: -(c["child"] * (0.4 + 0.6 * min(1.0, c["quality"] / 0.5)) * min(1.0, c["sat"] / 30.0)))
    # Spread temporally so VLM checks diverse candidates (~45 frames)
    picked = []
    for c in cands:
        if c["when"] and any(p["when"] and p["camera"] == c["camera"]
                             and abs((dt.datetime.fromisoformat(c["when"]) - dt.datetime.fromisoformat(p["when"])).total_seconds()) < 300
                             for p in picked):
            continue
        picked.append(c)
        if len(picked) >= 45:
            break
    return picked


def _mine_monthly_candidates(cfg: dict[str, Any], out_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Sample 5 unobstructed daytime candidates per month 2025-12 .. 2026-11."""
    nas = Path(cfg["project"]["nas_root"])
    osd = out_root / "one-second-a-day" / "days.json"
    days_state = json.loads(osd.read_text(encoding="utf-8")) if osd.exists() else {}
    by_month: dict[str, list[dict[str, Any]]] = {}
    for d, info in sorted(days_state.items()):
        if info.get("status") != "ok":
            continue
        m = d[:7]
        by_month.setdefault(m, []).append({"day": d, "day_n": info["day_n"], "source": info["source"],
                                           "camera": "lumi3.8e2ce0d05d9c251e", "score": info.get("score", 0.5)})

    models = ga.Models(siglip=True)
    result: dict[str, list[dict[str, Any]]] = {}
    try:
        for m, items in sorted(by_month.items()):
            items.sort(key=lambda x: -x["score"])
            picks = items[:10]
            # Also add 2 living-room daytime clips for later months if available
            if m >= "2026-06":
                ldir = nas / "aqara_video/lumi1.54ef4482e358"
                for d_item in items[:3]:
                    dd = ldir / d_item["day"].replace("-", "")
                    if dd.is_dir():
                        mp4s = [p for p in sorted(dd.glob("*.mp4")) if "093000" <= p.stem <= "190000" and p.stat().st_size >= 4_000_000]
                        if mp4s:
                            picks.append({"day": d_item["day"], "day_n": d_item["day_n"],
                                          "source": str(mp4s[len(mp4s) // 2]), "camera": "lumi1.54ef4482e358", "score": 0.9})
            # Rank by child score and no giant occluding adult center box
            scored = []
            for it in picks:
                im = ga.grab(it["source"], 2.5, 1280)
                if im is None:
                    continue
                bx = [b for b in models.persons([im])[0] if b[0] >= ga.PERSON_CONF]
                if not bx:
                    continue
                cs = models.score_frame(im, bx)
                scored.append({**it, "child": round(cs, 3)})
            scored.sort(key=lambda x: -x["child"])
            result[m] = scored[:6]
    finally:
        del models
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    return result


def run_vlm_pipeline(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    v2_dir = out_root / "v2"
    cache_p = v2_dir / "work" / "vlm_cache.jsonl"
    milestone_cands = _mine_milestone_candidates(cfg, out_root)
    photo_cands = _mine_photo_candidates(out_root)
    month_cands = _mine_monthly_candidates(cfg, out_root)

    # Also grab ~26 smooth progression candidates evenly spaced across 2025-12 .. 2026-11 from days.json
    days_state = json.loads((out_root / "one-second-a-day" / "days.json").read_text(encoding="utf-8"))
    ok_days = [(d, info) for d, info in sorted(days_state.items()) if info.get("status") == "ok"]
    prog_picks = []
    if ok_days:
        step = max(1.0, len(ok_days) / 36.0)
        idxs = sorted({min(len(ok_days) - 1, int(i * step)) for i in range(36)})
        prog_picks = [{"day": ok_days[i][0], "day_n": ok_days[i][1]["day_n"],
                       "source": ok_days[i][1]["source"], "camera": "lumi3.8e2ce0d05d9c251e"} for i in idxs]

    def _run_with_server(vlm: LlamaVLM) -> dict[str, Any]:
        # 1. Verify milestone & kitchen candidates (up to 72 windows)
        print(f"[VLM] Verifying {min(72, len(milestone_cands))} milestone/kitchen windows...", flush=True)
        verified_milestones = []
        for c in milestone_cands[:72]:
            dur = max(0.5, c["end_s"] - c["start_s"])
            times = [c["start_s"] + dur * f for f in (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)]
            imgs = extract_jpegs(c["source"], times, max_edge=768)
            if len(imgs) < 3:
                continue
            try:
                ans = vlm.ask(f"ms|{c['source']}|{c['start_s']}-{c['end_s']}", imgs, PROMPT_MILESTONE)
                verified_milestones.append({**c, "vlm": ans})
            except Exception as exc:
                print(f"[VLM] milestone err {Path(c['source']).name}: {exc}", flush=True)

        # Targeted living-room recall boost if any key milestone (standing/walking/sofa_descent) is still missing
        found_cats = {
            cat for m in verified_milestones
            if m["vlm"].get("child_clear") in ("yes", "partial") and m["vlm"].get("private_care") != "yes"
            for cat in m["vlm"].get("categories", [])
        }
        missing_key = {"standing", "walking", "sofa_descent"} - found_cats
        if missing_key:
            already_srcs = {m["source"] for m in verified_milestones}
            extra_living: list[dict[str, Any]] = []
            fjsonl = out_root / "day-20261108" / "files.jsonl"
            if fjsonl.is_file():
                for r in ga.load_jsonl(fjsonl):
                    if "error" in r or r["camera"] != "lumi1.54ef4482e358" or r["source"] in already_srcs:
                        continue
                    if r.get("duration", 0) >= 12 and r.get("active_seconds", 0) >= 8:
                        st = r.get("start_hint", "")
                        hr = int(st[11:13]) if len(st) >= 13 else 12
                        if 8 <= hr <= 21:
                            extra_living.append({
                                "camera": "lumi1.54ef4482e358", "source": r["source"],
                                "start_s": 0.5, "end_s": min(r["duration"], 8.5),
                                "hint": "living_boost", "time_str": ga.wall_label(r, 0.5),
                            })
            nas = Path(cfg["project"]["nas_root"])
            for dstr in ("20261111", "20261110", "20261109"):
                ldir = nas / "aqara_video/lumi1.54ef4482e358" / dstr
                if not ldir.is_dir():
                    continue
                mp4s = [p for p in sorted(ldir.glob("*.mp4"))
                        if "090000" <= p.stem <= "203000" and str(p) not in already_srcs and p.stat().st_size >= 4_500_000]
                if len(mp4s) > 12:
                    step = len(mp4s) / 12
                    mp4s = [mp4s[int(i * step)] for i in range(12)]
                for p in mp4s:
                    stem = p.stem
                    extra_living.append({
                        "camera": "lumi1.54ef4482e358", "source": str(p),
                        "start_s": 0.5, "end_s": 8.5,
                        "hint": "living_boost_recent", "time_str": f"{dstr[:4]}-{dstr[4:6]}-{dstr[6:8]} {stem[:2]}:{stem[2:4]}:{stem[4:6]}",
                    })
            print(f"[VLM] Missing {sorted(missing_key)}; boosting with {min(40, len(extra_living))} living-room clips...", flush=True)
            for c in extra_living[:40]:
                dur = max(0.5, c["end_s"] - c["start_s"])
                times = [c["start_s"] + dur * f for f in (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)]
                imgs = extract_jpegs(c["source"], times, max_edge=768)
                if len(imgs) < 3:
                    continue
                try:
                    ans = vlm.ask(f"ms|{c['source']}|{c['start_s']}-{c['end_s']}", imgs, PROMPT_MILESTONE)
                    verified_milestones.append({**c, "vlm": ans})
                except Exception as exc:
                    print(f"[VLM] boost err {Path(c['source']).name}: {exc}", flush=True)

        # 2. Encode Kitchen G350 T1 ONLY for VLM-confirmed child clips!
        kdir = v2_dir / "kitchen_20261115"
        if (kdir / "summary.json").is_file():
            ksum = json.loads((kdir / "summary.json").read_text(encoding="utf-8"))
            (kdir / "t1").mkdir(exist_ok=True)
            confirmed_srcs = {
                m["source"] for m in verified_milestones
                if m["camera"] == "lumi3.932ece08fac40c05"
                and m["vlm"].get("child_clear") in ("yes", "partial")
                and m["vlm"].get("private_care") != "yes"
            }
            kt1_bytes = 0
            for src in sorted(confirmed_srcs):
                dst = kdir / "t1" / f"{Path(src).stem}.mp4"
                if not dst.exists():
                    v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                                "-map", "0:v:0", "-map", "0:a:0?", "-vf", ga.T1_FILTER, *T1_CAPPED_VIDEO,
                                "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "24k", "-ac", "1", str(dst)], timeout=300)
                if dst.exists():
                    kt1_bytes += dst.stat().st_size
            ksum["confirmed_child_files"] = len(confirmed_srcs)
            ksum["t1_bytes"] = kt1_bytes
            ksum["total_kept_bytes"] = kt1_bytes + ksum["t2_bytes"]
            ksum["compression_ratio"] = round(ksum["orig_bytes"] / max(1, ksum["total_kept_bytes"]), 2)
            (kdir / "summary.json").write_text(json.dumps(ksum, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Kitchen] VLM confirmed {len(confirmed_srcs)} child clips -> T1={kt1_bytes/1e6:.2f}MB, "
                  f"total ratio={ksum['compression_ratio']}x", flush=True)

        # 3. Verify V2 photo candidates
        print(f"[VLM] Verifying {len(photo_cands)} V2 photo candidates...", flush=True)
        verified_photos = []
        for pc in photo_cands:
            jb = Path(pc["jpg"]).read_bytes()
            try:
                ans = vlm.ask(f"ph|{pc['source']}|{pc['t']}", [jb], PROMPT_PHOTO)
                verified_photos.append({**pc, "vlm": ans})
            except Exception as exc:
                print(f"[VLM] photo err: {exc}", flush=True)

        # 4. Verify 1-11 month comparison candidates
        print("[VLM] Verifying 1-11 month growth cards...", flush=True)
        monthly_best: dict[str, dict[str, Any]] = {}
        for m, clist in sorted(month_cands.items()):
            best_item = None
            for it in clist:
                imgs = extract_jpegs(it["source"], [1.0, 3.0, 5.0], max_edge=768)
                if len(imgs) < 2:
                    continue
                try:
                    ans = vlm.ask(f"mo|{it['source']}", imgs, PROMPT_MONTH)
                except Exception:
                    continue
                if ans.get("child_clear") != "yes" or ans.get("private_care") == "yes":
                    continue
                score = float(ans.get("clarity_score", 0))
                if best_item is None or score > best_item["score"]:
                    best_item = {**it, "vlm": ans, "score": score}
                if score >= 4.5:
                    break
            if best_item:
                monthly_best[m] = best_item

        # 5. Verify smooth progression clips (every ~10 days, unobstructed)
        print(f"[VLM] Verifying {len(prog_picks)} smooth-progression clips...", flush=True)
        prog_verified = []
        for it in prog_picks:
            imgs = extract_jpegs(it["source"], [1.0, 2.5, 4.0], max_edge=768)
            if len(imgs) < 2:
                continue
            try:
                ans = vlm.ask(f"mo|{it['source']}", imgs, PROMPT_MONTH)
            except Exception:
                continue
            if ans.get("child_clear") == "yes" and ans.get("private_care") != "yes" and float(ans.get("clarity_score", 0)) >= 3:
                prog_verified.append({**it, "vlm": ans})

        return {
            "milestones": verified_milestones,
            "photos": verified_photos,
            "monthly_best": monthly_best,
            "progression": prog_verified,
        }

    try:
        with LlamaVLM(cfg, cache_p, use_fallback=False).running() as vlm:
            out_data = _run_with_server(vlm)
    except Exception as primary_exc:
        print(f"[VLM] Primary Qwen3.8-27B failed ({primary_exc}); trying fallback MiniCPM-V 35B...", flush=True)
        with LlamaVLM(cfg, cache_p, use_fallback=True).running() as vlm:
            out_data = _run_with_server(vlm)

    (v2_dir / "vlm_verified.json").write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_data
