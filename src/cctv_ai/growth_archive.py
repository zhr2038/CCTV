"""Growth archive pilot.

Turns one day of raw CCTV recordings into a tiered archive and family-facing outputs:

* T2 "full negative": every original file -> 1 fps, 720p time-lapse (always kept).
* T1 "growth segments": seconds with a moving person or human sound -> 720p15 clip with audio.
* Daily photos: best child frames extracted from the ORIGINAL resolution before any downgrade.
* Daily highlight: short H.264 film made from the best T1 moments.
* One second a day: one 1-second clip per day since the birthday from a fixed camera.

The NAS share is only read. Nothing here deletes or rewrites originals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .media import resolve_ffmpeg, resolve_ffprobe

NAS_ROOT = os.environ.get("CCTV_NAS_ROOT", r"\\192.168.1.100\cctv")
PROJECT = Path(__file__).resolve().parents[2]
BIRTHDAY = dt.date(2025, 12, 17)
CAMERAS: dict[str, tuple[str, str]] = {
    "lumi1.54ef4482e358": ("客厅", "aqara_video/lumi1.54ef4482e358"),
    "lumi3.8e2ce0d05d9c251e": ("宝宝房顶部", "aqara_video/lumi3.8e2ce0d05d9c251e"),
    "lumi3.932ece08fac40c05": ("厨房", "aqara_video/lumi3.932ece08fac40c05"),
    "B88880620430": ("宝宝床头", "xiaomi_camera_videos/B88880620430"),
}
FF = resolve_ffmpeg(None)
FP = resolve_ffprobe(None)
FONT = "C\\:/Windows/Fonts/msyh.ttc"

# T1 / T2 encoder settings (chosen from the 2026-11-16 compression pilot).
T1_VIDEO = ["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "38",
            "-b:v", "0", "-g", "300"]
T1_FILTER = "scale=-2:720,hqdn3d=1.5:1.5:4:4,fps=15"
T2_VIDEO = ["-c:v", "av1_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "36", "-b:v", "0", "-g", "300"]

# Activity rules.
PERSON_CONF = 0.30
BOX_MOTION = 0.03      # share of changed pixels inside a person box
GLOBAL_MOTION = 0.010  # share of changed pixels in the whole frame (protects YOLO misses)
PAD_BEFORE, PAD_AFTER, MERGE_GAP, MIN_SEG = 2.0, 2.0, 8.0, 3.0


# --------------------------------------------------------------------------- helpers

def run(cmd: list[str], timeout: float = 900, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)


def probe(path: str) -> dict[str, Any]:
    r = run([FP, "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name,width,height",
             "-of", "json", path], text=True)
    info = json.loads(r.stdout or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    dur = float(info.get("format", {}).get("duration") or 0)
    return {"duration": dur, "width": int(v["width"]) if v else 0, "height": int(v["height"]) if v else 0,
            "vcodec": v.get("codec_name") if v else None, "has_audio": a is not None}


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def list_day(camera: str, day: dt.date) -> list[str]:
    rel = CAMERAS[camera][1]
    base = Path(NAS_ROOT) / rel
    if camera.startswith("B888"):
        return sorted(str(p) for p in base.glob(f"00_{day:%Y%m%d}*.mp4"))
    d = base / f"{day:%Y%m%d}"
    return sorted(str(p) for p in d.glob("*.mp4")) if d.is_dir() else []


def file_start(camera: str, path: str, day: dt.date) -> dt.datetime | None:
    name = Path(path).stem
    try:
        if camera.startswith("B888"):
            return dt.datetime.strptime(name.split("_")[1], "%Y%m%d%H%M%S")
        return dt.datetime.combine(day, dt.datetime.strptime(name, "%H%M%S").time())
    except (ValueError, IndexError):
        return None


class Models:
    """YOLO + SigLIP shared across worker threads (GPU calls are serialised)."""

    CHILD = ["a photo of a baby", "a photo of a toddler", "a little child playing"]
    OTHER = ["a photo of an adult", "an empty room", "an empty bed"]

    def __init__(self, siglip: bool = True) -> None:
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.lock = threading.Lock()
        self.yolo = YOLO(str(PROJECT / ".data/models/yolo26s.pt"))
        self.siglip = None
        if siglip:
            from transformers import AutoModel, AutoProcessor
            cache = str(PROJECT / ".data/models/huggingface")
            name = "google/siglip2-base-patch16-224"
            self.proc = AutoProcessor.from_pretrained(name, cache_dir=cache, use_fast=True, local_files_only=True)
            self.siglip = AutoModel.from_pretrained(name, cache_dir=cache, dtype=torch.float16,
                                                    local_files_only=True).to("cuda").eval()
            t = self.proc(text=self.CHILD + self.OTHER, padding="max_length", truncation=True, return_tensors="pt")
            with torch.inference_mode():
                tf = self.siglip.get_text_features(**{k: v.to("cuda") for k, v in t.items()
                                                      if k in ("input_ids", "attention_mask")}).float()
            self.text = tf / tf.norm(dim=-1, keepdim=True)
            self.scale = self.siglip.logit_scale.float().exp()
            self.bias = self.siglip.logit_bias.float()

    def persons(self, frames: list[np.ndarray]) -> list[list[tuple[float, float, float, float, float]]]:
        out: list[list[tuple[float, float, float, float, float]]] = []
        with self.lock:
            for i in range(0, len(frames), 32):
                res = self.yolo.predict(frames[i:i + 32], imgsz=640, conf=0.20, classes=[0],
                                        verbose=False, half=True, device=0)
                for r in res:
                    h, w = r.orig_shape
                    boxes = []
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        boxes.append((float(b.conf[0]), x1 / w, y1 / h, x2 / w, y2 / h))
                    out.append(boxes)
        return out

    def child_prob(self, images_rgb: list[np.ndarray]) -> list[float]:
        """Sigmoid SigLIP probability that the image shows a baby/toddler (max over child prompts)."""
        if not images_rgb:
            return []
        torch = self.torch
        probs: list[float] = []
        with self.lock:
            for i in range(0, len(images_rgb), 64):
                inp = self.proc(images=images_rgb[i:i + 64], return_tensors="pt")
                with torch.inference_mode():
                    f = self.siglip.get_image_features(pixel_values=inp["pixel_values"].to("cuda", torch.float16)).float()
                f = f / f.norm(dim=-1, keepdim=True)
                logits = f @ self.text.T * self.scale + self.bias
                p = torch.sigmoid(logits)
                child = p[:, :len(self.CHILD)].max(dim=1).values
                other = p[:, len(self.CHILD):].max(dim=1).values
                # Relative score keeps "adult only" frames low even when absolute sigmoid is small.
                probs.extend((child / (child + other + 1e-6)).tolist())
        return probs

    def score_frame(self, bgr: np.ndarray, boxes) -> float:
        """Child score of a frame using full view plus crops around the biggest person boxes."""
        rgb = bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        views = [np.ascontiguousarray(rgb)]
        for conf, x1, y1, x2, y2 in sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[4] - b[2]))[:2]:
            cx, cy, bw, bh = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h, (x2 - x1) * w, (y2 - y1) * h
            s = max(bw, bh) * 1.4
            xa, ya = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            xb, yb = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
            if xb - xa > 32 and yb - ya > 32:
                views.append(np.ascontiguousarray(rgb[ya:yb, xa:xb]))
        return max(self.child_prob(views))


# --------------------------------------------------------------------------- per-file analysis

_audio_local = threading.local()


def audio_events(path: str) -> list[tuple[float, float, str]]:
    from .audio_events import YamnetDetector
    det = getattr(_audio_local, "det", None)
    if det is None:
        det = YamnetDetector(model_path=PROJECT / ".data/models/yamnet.tflite",
                             class_map_path=PROJECT / ".data/models/yamnet_class_map.csv",
                             event_labels=["speech", "baby_cry", "yell", "scream"],
                             minimum_confidence=0.35, merge_gap_seconds=1.0)
        _audio_local.det = det
    try:
        res = det.analyze(path, ffmpeg=FF, timeout_seconds=600)
    except Exception:
        return []
    return [(e.media_start_seconds, e.media_end_seconds, e.label) for e in res.events]


def segments_from(active: np.ndarray, duration: float) -> list[list[float]]:
    segs: list[list[float]] = []
    idx = np.flatnonzero(active)
    for i in idx:
        s, e = max(0.0, i - PAD_BEFORE), min(duration, i + 1 + PAD_AFTER)
        if segs and s - segs[-1][1] <= MERGE_GAP:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    out = []
    for s, e in segs:
        if e - s < MIN_SEG:
            e = min(duration, s + MIN_SEG)
        out.append([round(s, 2), round(e, 2)])
    return out


def process_file(models: Models, camera: str, path: str, day: dt.date, out: Path) -> dict[str, Any]:
    import cv2
    t0 = time.time()
    info = probe(path)
    rec: dict[str, Any] = {"camera": camera, "source": path, "size": os.path.getsize(path), **info}
    if info["duration"] <= 0 or not info["width"]:
        rec["error"] = "unreadable"
        return rec
    stem = Path(path).stem
    w = 1280
    h = even(info["height"] * w / info["width"])
    t2 = out / "t2" / camera / f"{stem}.mp4"
    t2.parent.mkdir(parents=True, exist_ok=True)

    # One decode at 1 fps: feeds the time-lapse encoder and the detectors.
    def decoder(hw: bool) -> subprocess.Popen:
        cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if hw:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-i", path, "-an", "-vf", f"fps=1,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames: list[np.ndarray] = []
    limit_s = max(180.0, info["duration"] * 0.5)
    for hw in (True, False):
        dec = decoder(hw)
        enc = subprocess.Popen([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0", *T2_VIDEO,
                                "-pix_fmt", "yuv420p", str(t2)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
        frames = []
        try:
            while True:
                buf = dec.stdout.read(size)
                if len(buf) < size:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                frames.append(frame)
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
            break
        frames = []
        if t2.exists():
            t2.unlink()
    if not frames:
        rec["error"] = "no frames decoded"
        return rec
    n = len(frames)

    boxes = models.persons(frames)
    grays = [cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (320, even(320 * h / w))), cv2.COLOR_BGR2GRAY), (5, 5), 0)
             for f in frames]
    gh, gw = grays[0].shape
    gmot = np.zeros(n)
    bmot = np.zeros(n)
    pconf = np.zeros(n)
    for i in range(n):
        pconf[i] = max((b[0] for b in boxes[i]), default=0.0)
        if i == 0:
            continue
        diff = cv2.absdiff(grays[i], grays[i - 1]) > 12
        gmot[i] = diff.mean()
        for c, x1, y1, x2, y2 in boxes[i]:
            if c < PERSON_CONF:
                continue
            xa, ya, xb, yb = int(x1 * gw), int(y1 * gh), max(int(x2 * gw), int(x1 * gw) + 1), max(int(y2 * gh), int(y1 * gh) + 1)
            bmot[i] = max(bmot[i], diff[ya:yb, xa:xb].mean())
    gmot[0] = gmot[1] if n > 1 else 0
    bmot[0] = bmot[1] if n > 1 else 0

    aev = audio_events(path) if info["has_audio"] else []
    audio_mask = np.zeros(n, bool)
    for s, e, _ in aev:
        audio_mask[int(s):min(n, int(np.ceil(e)))] = True

    person_moving = (pconf >= PERSON_CONF) & (bmot >= BOX_MOTION)
    unexplained_motion = (pconf < PERSON_CONF) & (gmot >= GLOBAL_MOTION)
    active = person_moving | unexplained_motion | audio_mask
    segs = segments_from(active, info["duration"])

    # Photo candidates: frames with a person, ranked by box size * confidence * sharpness * colour.
    cands = []
    for i in range(n):
        good = [b for b in boxes[i] if b[0] >= PERSON_CONF]
        if not good:
            continue
        big = max(good, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
        area = (big[3] - big[1]) * (big[4] - big[2])
        f = frames[i]
        x1, y1, x2, y2 = int(big[1] * w), int(big[2] * h), int(big[3] * w), int(big[4] * h)
        crop = cv2.cvtColor(f[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        sat = float(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        q = big[0] * min(1.0, area / 0.04) * min(1.0, sharp / 150.0) * (0.5 + 0.5 * min(1.0, sat / 40.0))
        cands.append((q, i, good, sharp, sat))
    cands.sort(key=lambda c: -c[0])
    picked = []
    for c in cands:
        if all(abs(c[1] - p[1]) >= 8 for p in picked):
            picked.append(c)
        if len(picked) == 3:
            break
    photo_cands = []
    cdir = out / "work" / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    for q, i, good, sharp, sat in picked:
        score = models.score_frame(frames[i], good) if models.siglip is not None else 0.0
        jp = cdir / f"{camera}_{stem}_{i:05d}.jpg"
        cv2.imwrite(str(jp), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 88])
        photo_cands.append({"t": float(i), "quality": round(q, 4), "child": round(score, 4),
                            "sharp": round(sharp, 1), "sat": round(sat, 1), "jpg": str(jp)})

    # T1 encode: ONE file per original containing only its active segments back to back.
    # "map" keeps [source_start, source_end, t1_offset] so every second stays traceable to the original.
    t1: dict[str, Any] = {"size": 0, "map": []}
    if segs:
        t1dir = out / "t1" / camera
        t1dir.mkdir(parents=True, exist_ok=True)
        dst = t1dir / f"{stem}.mp4"
        whole = len(segs) == 1 and segs[0][0] <= 0.5 and segs[0][1] >= info["duration"] - 0.5
        cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "cuda", "-i", path,
               "-map", "0:v:0", "-map", "0:a:0?"]
        if whole:
            cmd += ["-vf", T1_FILTER]
        else:
            expr = "+".join(f"between(t\\,{s:.2f}\\,{e:.2f})" for s, e in segs)
            cmd += ["-vf", f"select={expr},setpts=N/FRAME_RATE/TB,{T1_FILTER}",
                    "-af", f"aselect={expr},asetpts=N/SR/TB"]
        cmd += [*T1_VIDEO, "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "24k", "-ac", "1", str(dst)]
        r = run(cmd)
        if r.returncode != 0 or not dst.exists():
            r = run([c for c in cmd if c not in ("-hwaccel", "cuda")])
        if r.returncode == 0 and dst.exists():
            off = 0.0
            for s, e in segs:
                t1["map"].append([s, e, round(off, 2)])
                off += e - s
            t1.update({"file": str(dst), "size": dst.stat().st_size, "seconds": round(off, 2)})
        else:
            t1["error"] = r.stderr.decode("utf-8", "replace")[-300:]

    st = file_start(camera, path, day)
    rec.update({
        "frames_1fps": n, "start_hint": st.isoformat() if st else None,
        "person_seconds": int((pconf >= PERSON_CONF).sum()), "active_seconds": int(active.sum()),
        "audio_events": aev[:50], "segments": segs, "t1": t1,
        "t2": {"file": str(t2), "size": t2.stat().st_size if t2.exists() else 0},
        "photo_candidates": photo_cands, "elapsed": round(time.time() - t0, 1),
    })
    return rec


# --------------------------------------------------------------------------- day pipeline

def cmd_day(args) -> None:
    day = dt.date.fromisoformat(args.day)
    out = Path(args.output) / f"day-{day:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "files.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:
                pass
    jobs = []
    for cam in args.camera or list(CAMERAS):
        files = list_day(cam, day)
        if args.limit:
            files = files[: args.limit]
        jobs += [(cam, f) for f in files if f not in done]
    print(f"{len(jobs)} files to process ({len(done)} already done)", flush=True)
    models = Models(siglip=True)
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool, manifest.open("a", encoding="utf-8") as mf:
        futs = {pool.submit(process_file, models, cam, f, day, out): f for cam, f in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # keep going; record the failure
                rec = {"source": futs[fut], "error": repr(exc)}
            with lock:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
            if k % 25 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} files, {time.time() - t0:.0f}s", flush=True)
    finalize_day(models, day, out)


def load_jsonl(p: Path) -> list[dict]:
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            rows[r["source"]] = r
        except Exception:
            pass
    return list(rows.values())


def wall_label(rec: dict, t: float) -> str:
    if not rec.get("start_hint"):
        return ""
    ts = dt.datetime.fromisoformat(rec["start_hint"]) + dt.timedelta(seconds=t)
    approx = "约" if rec["camera"].startswith("B888") else ""
    return f"{approx}{ts:%H:%M:%S}"


def finalize_day(models: Models | None, day: dt.date, out: Path) -> None:
    rows = [r for r in load_jsonl(out / "files.jsonl") if "error" not in r]
    # ---- storage summary
    summary: dict[str, Any] = {"day": day.isoformat(), "cameras": {}}
    for r in rows:
        c = summary["cameras"].setdefault(r["camera"], {"name": CAMERAS[r["camera"]][0], "files": 0, "orig": 0,
                                                       "t1": 0, "t2": 0, "hours": 0.0, "active_s": 0, "person_s": 0,
                                                       "t1_s": 0.0})
        c["files"] += 1
        c["orig"] += r["size"]
        c["t2"] += r["t2"]["size"]
        c["t1"] += r["t1"].get("size", 0)
        c["t1_s"] += r["t1"].get("seconds", 0.0)
        c["hours"] += r["duration"] / 3600
        c["active_s"] += r["active_seconds"]
        c["person_s"] += r["person_seconds"]

    # ---- photos: pick best child frames spread across the day, then re-extract at original resolution
    cands = []
    for r in rows:
        for pc in r["photo_candidates"]:
            st = dt.datetime.fromisoformat(r["start_hint"]) if r.get("start_hint") else None
            when = (st + dt.timedelta(seconds=pc["t"])) if st else None
            cands.append({**pc, "rec": r, "when": when})
    def photo_rank(c):  # child first, then sharp/large, and strongly prefer colour (daylight) frames
        colour = 0.3 + 0.7 * min(1.0, c["sat"] / 30.0)
        return c["child"] * (0.4 + 0.6 * min(1.0, c["quality"] / 0.5)) * colour

    cands.sort(key=lambda c: -photo_rank(c))
    photos = []
    per_cam: dict[str, int] = {}
    for c in cands:
        if c["child"] < 0.5:
            continue
        cam = c["rec"]["camera"]
        if per_cam.get(cam, 0) >= 5:
            continue
        if c["when"] and any(p["when"] and abs((c["when"] - p["when"]).total_seconds()) < 1200 for p in photos):
            continue
        photos.append(c)
        per_cam[cam] = per_cam.get(cam, 0) + 1
        if len(photos) == 12:
            break
    photos.sort(key=lambda c: c["when"] or dt.datetime.min)
    pdir = out / "photos"
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    photo_list = []
    for k, c in enumerate(photos, 1):
        r = c["rec"]
        dst = pdir / f"{k:02d}_{CAMERAS[r['camera']][0]}_{wall_label(r, c['t']).replace(':', '')}.jpg"
        run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{c['t']:.2f}", "-i", r["source"],
             "-frames:v", "1", "-q:v", "2", str(dst)])
        photo_list.append({"file": dst.name, "camera": CAMERAS[r["camera"]][0], "time": wall_label(r, c["t"]),
                           "child": c["child"], "bytes": dst.stat().st_size if dst.exists() else 0,
                           "source": r["source"], "offset": c["t"]})

    # ---- highlight: best moments inside T1 segments, 8 s each, at least 15 min apart
    moments = []
    for c in cands:
        r = c["rec"]
        seg = next((s for s in r["segments"] if s[0] <= c["t"] <= s[1]), None)
        if seg is None or c["child"] < 0.55:
            continue
        if c["when"] and any(m["when"] and abs((c["when"] - m["when"]).total_seconds()) < 900 for m in moments):
            continue
        s = max(seg[0], c["t"] - 4.0)
        e = min(r["duration"], s + 8.0)
        if e - s < 3:
            continue
        moments.append({**c, "s": s, "e": e})
        if len(moments) == 8:
            break
    moments.sort(key=lambda m: m["when"] or dt.datetime.min)
    hdir = out / "work" / "highlight"
    hdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, m in enumerate(moments):
        r = m["rec"]
        dst = hdir / f"part{k:02d}.mp4"
        d = m["e"] - m["s"]
        label = f"{CAMERAS[r['camera']][0]}  {wall_label(r, m['s'])}".replace(":", "\\:")
        vf = (f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,"
              f"fade=t=in:st=0:d=0.4,fade=t=out:st={d - 0.4:.2f}:d=0.4,"
              f"drawtext=fontfile='{FONT}':text='{label}':x=40:y=h-80:fontsize=40:fontcolor=white:"
              f"box=1:boxcolor=black@0.45:boxborderw=12")
        af = f"aresample=48000,afade=t=in:st=0:d=0.4,afade=t=out:st={d - 0.4:.2f}:d=0.4"
        cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{m['s']:.2f}", "-i", r["source"],
               "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{d:.2f}", "-vf", vf]
        if r.get("has_audio"):
            cmd += ["-map", "0:v:0", "-map", "0:a:0", "-af", af]
        else:
            cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-ac", "1", "-ar", "48000", str(dst)]
        if run(cmd).returncode == 0:
            parts.append(dst)
    highlight = out / f"highlight_{day:%Y%m%d}.mp4"
    if parts:
        lst = hdir / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", "-movflags", "+faststart", str(highlight)])

    # ---- per-camera full-day time-lapse (concatenate per-file T2 proxies, stream copy)
    lapses = {}
    for cam in summary["cameras"]:
        files = sorted((r for r in rows if r["camera"] == cam), key=lambda r: r.get("start_hint") or "")
        lst = out / "work" / f"t2_{cam}.txt"
        lst.write_text("".join(f"file '{Path(r['t2']['file']).as_posix()}'\n" for r in files if r["t2"]["size"]),
                       encoding="utf-8")
        dst = out / f"timelapse_{cam}.mp4"
        run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", "-movflags", "+faststart", str(dst)])
        lapses[cam] = dst.name

    # ---- quality comparison frame (original vs T1 proxy) from the top photo
    compare = None
    for c in photos:
        r = c["rec"]
        seg = next((m for m in r["t1"].get("map", []) if m[0] <= c["t"] <= m[1]), None)
        if not seg or not r["t1"].get("file"):
            continue
        dst = out / "compare.jpg"
        rel_t = seg[2] + (c["t"] - seg[0])
        run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{c['t']:.2f}", "-i", r["source"],
             "-ss", f"{rel_t:.2f}", "-i", r["t1"]["file"], "-filter_complex",
             "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack", "-frames:v", "1", "-q:v", "3", str(dst)])
        compare = dst.name
        break

    timeline = {}
    for r in rows:
        if not r.get("start_hint"):
            continue
        st = dt.datetime.fromisoformat(r["start_hint"])
        base = dt.datetime.combine(day, dt.time())
        o = (st - base).total_seconds()
        tl = timeline.setdefault(r["camera"], {"rec": [], "t1": []})
        tl["rec"].append([round(o), round(o + r["duration"])])
        tl["t1"] += [[round(o + s), round(o + e)] for s, e in r["segments"]]
    summary.update({"photos": photo_list, "highlight": highlight.name if parts else None,
                    "highlight_moments": len(parts), "timelapses": lapses, "compare": compare,
                    "timeline": timeline,
                    "errors": sum(1 for r in load_jsonl(out / "files.jsonl") if "error" in r)})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k in ("cameras", "highlight_moments", "errors")},
                     ensure_ascii=False, indent=1))


def cmd_finalize(args) -> None:
    day = dt.date.fromisoformat(args.day)
    finalize_day(None, day, Path(args.output) / f"day-{day:%Y%m%d}")


# --------------------------------------------------------------------------- one second a day

def grab(path: str, t: float, width: int = 1280) -> np.ndarray | None:
    r = run([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
             "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "pipe:1"])
    if r.returncode or not r.stdout:
        return None
    import cv2
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def cmd_one_second(args) -> None:
    import cv2
    cam = args.camera
    start = dt.date.fromisoformat(args.since)
    end = dt.date.fromisoformat(args.until)
    out = Path(args.output) / "one-second-a-day"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    state_p = out / "days.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}
    models = Models(siglip=True)
    days = [start + dt.timedelta(d) for d in range((end - start).days + 1)]

    def candidates(day: dt.date) -> list[str]:
        files = [f for f in list_day(cam, day) if "080000" <= Path(f).stem <= "200000"]
        if len(files) > args.per_day:
            step = len(files) / args.per_day
            files = [files[int(i * step)] for i in range(args.per_day)]
        return files

    with ThreadPoolExecutor(args.workers) as pool:
        for day in days:
            key = day.isoformat()
            if key in state:
                continue
            files = candidates(day)
            frames = list(pool.map(lambda f: (f, grab(f, 2.0)), files))
            frames = [(f, im) for f, im in frames if im is not None]
            best = None
            if frames:
                boxes = models.persons([im for _, im in frames])
                for (f, im), bx in zip(frames, boxes):
                    good = [b for b in bx if b[0] >= PERSON_CONF]
                    if not good:
                        continue
                    child = models.score_frame(im, good)
                    sat = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                    score = child * (0.6 + 0.4 * min(1.0, sat / 40.0))
                    if best is None or score > best[0]:
                        best = (score, f, im, child)
            if best is None or best[3] < 0.5:
                state[key] = {"status": "no_child_frame", "candidates": len(frames)}
            else:
                score, f, im, child = best
                n = (day - BIRTHDAY).days + 1
                clip = out / "clips" / f"{day:%Y%m%d}.mp4"
                label = f"{day:%Y.%m.%d}   第 {n} 天"
                vf = (f"scale=1920:-2,crop=1920:1080,fps=30,drawtext=fontfile='{FONT}':text='{label}':"
                      f"x=w-tw-50:y=h-90:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=14")
                r = run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1.5", "-i", f, "-t", "1.0",
                         "-vf", vf, "-an", "-c:v", "h264_nvenc", "-preset", "p6", "-cq", "23", "-pix_fmt", "yuv420p",
                         "-r", "30", str(clip)])
                cv2.imwrite(str(out / "frames" / f"{day:%Y%m%d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                state[key] = {"status": "ok" if r.returncode == 0 else "clip_failed", "source": f,
                              "child": round(child, 3), "score": round(score, 3), "day_n": n,
                              "candidates": len(frames)}
            state_p.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
            print(key, state[key].get("status"), state[key].get("child"), flush=True)

    ok = [d for d in sorted(state) if state[d].get("status") == "ok"]
    lst = out / "list.txt"
    lst.write_text("".join(f"file '{(out / 'clips' / (d.replace('-', '') + '.mp4')).as_posix()}'\n" for d in ok),
                   encoding="utf-8")
    film = out / "one_second_a_day.mp4"
    run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", str(film)])
    # Monthly growth strip: best frame of each month.
    months: dict[str, tuple[float, str]] = {}
    for d in ok:
        m = d[:7]
        if m not in months or state[d]["score"] > months[m][0]:
            months[m] = (state[d]["score"], d)
    strip = [{"month": m, "day": d, "day_n": state[d]["day_n"], "frame": f"frames/{d.replace('-', '')}.jpg"}
             for m, (_, d) in sorted(months.items())]
    (out / "summary.json").write_text(json.dumps({"camera": cam, "days": len(state), "ok": len(ok),
                                                  "film": film.name, "months": strip}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"one-second film: {len(ok)} days -> {film}")


# --------------------------------------------------------------------------- report page

def cmd_one_second_zoom(args) -> None:
    """Re-render the chosen daily seconds with a 16:9 crop centred on the child (same source moments)."""
    out = Path(args.output) / "one-second-a-day"
    state = json.loads((out / "days.json").read_text(encoding="utf-8"))
    zdir = out / "clips_zoom"
    zdir.mkdir(exist_ok=True)
    models = Models(siglip=True)
    ok = [d for d in sorted(state) if state[d].get("status") == "ok"]

    def child_box(im: np.ndarray):
        boxes = [b for b in models.persons([im])[0] if b[0] >= PERSON_CONF]
        if not boxes:
            return None
        rgb = im[:, :, ::-1]
        h, w = rgb.shape[:2]
        crops, keep = [], []
        for b in boxes:
            x1, y1, x2, y2 = int(b[1] * w), int(b[2] * h), int(b[3] * w), int(b[4] * h)
            if x2 - x1 > 16 and y2 - y1 > 16:
                crops.append(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                keep.append(b)
        if not crops:
            return None
        probs = models.child_prob(crops)
        k = int(np.argmax(probs))
        return keep[k], probs[k]

    for d in ok:
        st = state[d]
        dst = zdir / f"{d.replace('-', '')}.mp4"
        if dst.exists():
            continue
        info = probe(st["source"])
        W, H = info["width"], info["height"]
        im = grab(st["source"], 2.0, 1280)
        found = child_box(im) if im is not None else None
        if found is None or found[1] < 0.4:
            cw, ch, cx, cy = W, H, W / 2, H / 2
        else:
            b = found[0]
            bw, bh = (b[3] - b[1]) * W, (b[4] - b[2]) * H
            cw = min(W, max(W * 0.5, bw * 3.0, bh * 3.0 * 16 / 9))
            ch = cw * 9 / 16
            if ch > H:
                ch, cw = H, H * 16 / 9
            cx, cy = (b[1] + b[3]) / 2 * W, (b[2] + b[4]) / 2 * H
        x = int(min(max(0, cx - cw / 2), W - cw))
        y = int(min(max(0, cy - ch / 2), H - ch))
        label = f"{d.replace('-', '.')}   第 {st['day_n']} 天"
        vf = (f"crop={even(cw)}:{even(ch)}:{x}:{y},scale=1920:1080:flags=lanczos,fps=30,"
              f"drawtext=fontfile='{FONT}':text='{label}':x=w-tw-50:y=h-90:fontsize=48:fontcolor=white:"
              f"box=1:boxcolor=black@0.45:boxborderw=14")
        run([FF, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1.5", "-i", st["source"], "-t", "1.0",
             "-vf", vf, "-an", "-c:v", "h264_nvenc", "-preset", "p6", "-cq", "24", "-pix_fmt", "yuv420p",
             "-r", "30", str(dst)])
        print(d, "zoom" if found and found[1] >= 0.4 else "full", flush=True)
    lst = out / "list_zoom.txt"
    lst.write_text("".join(f"file '{(zdir / (d.replace('-', '') + '.mp4')).as_posix()}'\n" for d in ok
                           if (zdir / (d.replace('-', '') + '.mp4')).exists()), encoding="utf-8")
    run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", str(out / "one_second_a_day_zoom.mp4")])
    print("zoom film done")


def gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB" if x >= 1e8 else f"{x / 1e6:.1f} MB"


def cmd_page(args) -> None:
    root = Path(args.output)
    day = dt.date.fromisoformat(args.day)
    dd = root / f"day-{day:%Y%m%d}"
    s = json.loads((dd / "summary.json").read_text(encoding="utf-8"))
    osd = root / "one-second-a-day" / "summary.json"
    one = json.loads(osd.read_text(encoding="utf-8")) if osd.exists() else None
    rel = f"day-{day:%Y%m%d}"
    tot = {k: sum(c[k] for c in s["cameras"].values()) for k in ("orig", "t1", "t2", "hours", "t1_s")}
    photo_bytes = sum(p["bytes"] for p in s["photos"])
    kept = tot["t1"] + tot["t2"] + photo_bytes
    ratio = tot["orig"] / max(1, kept)
    rows = ""
    for cam, c in s["cameras"].items():
        k = c["t1"] + c["t2"]
        rows += (f"<tr><td>{html.escape(c['name'])}</td><td>{c['files']}</td><td>{c['hours']:.1f} h</td>"
                 f"<td>{gb(c['orig'])}</td><td>{c['t1_s'] / 3600:.1f} h · {gb(c['t1'])}</td><td>{gb(c['t2'])}</td>"
                 f"<td><b>{c['orig'] / max(1, k):.1f}×</b></td></tr>")
    photos = "".join(
        f"<figure><a href='{rel}/photos/{html.escape(p['file'])}' target='_blank'>"
        f"<img loading='lazy' src='{rel}/photos/{html.escape(p['file'])}'></a>"
        f"<figcaption>{html.escape(p['camera'])} {html.escape(p['time'])}</figcaption></figure>"
        for p in s["photos"])
    # timeline bars
    tl = ""
    for cam, t in s["timeline"].items():
        bars = "".join(f"<i class='rec' style='left:{a / 864:.3f}%;width:{max(0.05, (b - a) / 864):.3f}%'></i>"
                       for a, b in t["rec"])
        bars += "".join(f"<i class='t1' style='left:{a / 864:.3f}%;width:{max(0.08, (b - a) / 864):.3f}%'></i>"
                        for a, b in t["t1"])
        tl += f"<div class='tlrow'><span>{html.escape(CAMERAS[cam][0])}</span><div class='tl'>{bars}</div></div>"
    hours = "".join(f"<span style='left:{h / 24 * 100:.2f}%'>{h:02d}</span>" for h in range(0, 24, 3))
    lapses = "".join(
        f"<figure><video controls preload='metadata' src='{rel}/{v}'></video>"
        f"<figcaption>{html.escape(CAMERAS[c][0])} · 全天 1 帧/秒</figcaption></figure>"
        for c, v in s["timelapses"].items())
    one_html = ""
    if one:
        strip = "".join(f"<figure><img src='one-second-a-day/{m['frame']}'><figcaption>{m['month']} · 第 {m['day_n']} 天"
                        f"</figcaption></figure>" for m in one["months"])
        zoom = (root / "one-second-a-day" / "one_second_a_day_zoom.mp4").exists()
        films = (f"<div class='grid2'><figure><video controls preload='metadata' src='one-second-a-day/one_second_a_day_zoom.mp4'>"
                 f"</video><figcaption>跟随孩子放大版</figcaption></figure>" if zoom else "<div class='grid2'>")
        films += (f"<figure><video controls preload='metadata' src='one-second-a-day/{one['film']}'></video>"
                  f"<figcaption>全景版（同一机位）</figcaption></figure></div>")
        one_html = (f"<section><h2>每天一秒 · 从出生到现在</h2><p class='note'>宝宝房顶部摄像头，每天自动挑 1 秒孩子入镜最好的画面，"
                    f"共 {one['ok']} 天，{one['ok']} 秒（{one['days'] - one['ok']} 天没有找到合适画面，多为出生头几天还没回家）。</p>"
                    f"{films}"
                    f"<h3>每月成长对比</h3><div class='strip'>{strip}</div></section>")
    hl = (f"<video controls preload='metadata' src='{rel}/{s['highlight']}'></video>" if s.get("highlight")
          else "<p>当天没有足够的精华片段。</p>")
    cmp_html = (f"<img class='wide' src='{rel}/{s['compare']}'><p class='note'>左：原片 · 右：保存的 720p 成长片段</p>"
                if s.get("compare") else "")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>成长档案试点 {day}</title>
<style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#faf8f5;color:#2b2b2b}}
header{{background:#2f4858;color:#fff;padding:28px 40px}} header h1{{margin:0 0 6px;font-size:26px}}
section{{padding:18px 40px;border-bottom:1px solid #eee}} h2{{font-size:20px;margin:6px 0 12px}}
.kpi{{display:flex;gap:18px;flex-wrap:wrap}} .kpi div{{background:#fff;border-radius:10px;padding:14px 20px;box-shadow:0 1px 3px #0001}}
.kpi b{{display:block;font-size:26px;color:#2f4858}}
table{{border-collapse:collapse;background:#fff}} td,th{{padding:7px 14px;border-bottom:1px solid #eee;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
figure{{margin:0}} figure img,figure video{{width:100%;border-radius:8px;display:block}}
figcaption{{font-size:13px;color:#666;padding:4px 2px}}
video{{width:100%;max-width:1100px;border-radius:8px;background:#000}}
.strip{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}
.tlrow{{display:flex;align-items:center;margin:6px 0}} .tlrow span{{width:90px;font-size:13px}}
.tl{{position:relative;flex:1;height:22px;background:#eee;border-radius:4px;overflow:hidden}}
.tl i{{position:absolute;top:0;bottom:0}} .tl i.rec{{background:#c9d6df}} .tl i.t1{{background:#e07a5f;top:5px;bottom:5px}}
.hours{{position:relative;margin-left:90px;height:16px;font-size:11px;color:#888}} .hours span{{position:absolute}}
.note{{color:#666;font-size:14px}} img.wide{{max-width:1100px;width:100%;border-radius:8px}}
</style></head><body>
<header><h1>成长档案试点 · {day:%Y年%m月%d日}（孩子出生第 {(day - BIRTHDAY).days + 1} 天）</h1>
<div>一天的监控录像自动整理成：精选照片 · 当日精华 · 带声音的成长片段 · 全天延时底片。原片只读，没有删除或改动任何文件。</div></header>
<section><h2>体积对比</h2><div class="kpi">
<div>原片<b>{gb(tot['orig'])}</b>{tot['hours']:.1f} 小时</div>
<div>整理后<b>{gb(kept)}</b>成长片段 + 延时 + 照片</div>
<div>压缩倍数<b>{ratio:.1f}×</b>节省 {100 * (1 - kept / tot['orig']):.0f}%</div>
<div>按此推算 7 TB 历史<b>≈ {7000 / ratio:.0f} GB</b>每天 22 GB → {22 / ratio:.1f} GB</div></div>
<p></p><table><tr><th>摄像头</th><th>文件</th><th>录像时长</th><th>原片</th><th>成长片段（有人活动/有声音）</th><th>全天延时</th><th>倍数</th></tr>{rows}</table>
<p class="note">成长片段：画面里有人在动、或有说话/哭/喊等声音的时段，前后各留 2 秒，转成 720p 15 帧带声音；
其余时间（睡觉、静坐、空房）只留每秒 1 帧的延时底片，任何时刻仍然看得到。</p></section>
<section><h2>当天精选照片（原画分辨率）</h2><div class="grid">{photos}</div></section>
<section><h2>当天精华（自动剪辑）</h2>{hl}</section>
{one_html}
<section><h2>全天时间轴</h2><div class="hours">{hours}</div>{tl}
<p class="note">浅蓝：摄像头有录像 · 橙色：保存为成长片段</p></section>
<section><h2>全天延时底片</h2><div class="grid">{lapses}</div></section>
<section><h2>画质对比</h2>{cmp_html}</section>
<section><h2>说明</h2><p class="note">本页全部由程序自动生成，没有人工挑选。NAS 原片、保险箱和百度备份都没有被删除或改动。
现在 NAS 上的空间实际被 Baby_Vault 保险箱（reflink）和每日只读快照占着，所以只删 CCTV 共享里的原片腾不出空间。
要真正释放空间，得在保险箱这一层用整理后的档案替换已验收日期的原片，同时清理旧快照。这一步等您确认后再做。</p></section>
</body></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
    print(root / "index.html")


def main() -> None:
    p = argparse.ArgumentParser(prog="growth_archive")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("day")
    d.add_argument("--day", required=True)
    d.add_argument("--output", default=str(PROJECT / "output/growth-archive-pilot-20261117"))
    d.add_argument("--camera", action="append")
    d.add_argument("--limit", type=int, default=0)
    d.add_argument("--workers", type=int, default=4)
    d.set_defaults(func=cmd_day)
    f = sub.add_parser("finalize")
    f.add_argument("--day", required=True)
    f.add_argument("--output", default=str(PROJECT / "output/growth-archive-pilot-20261117"))
    f.set_defaults(func=cmd_finalize)
    o = sub.add_parser("one-second")
    o.add_argument("--camera", default="lumi3.8e2ce0d05d9c251e")
    o.add_argument("--since", default=BIRTHDAY.isoformat())
    o.add_argument("--until", required=True)
    o.add_argument("--per-day", type=int, default=12)
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--output", default=str(PROJECT / "output/growth-archive-pilot-20261117"))
    o.set_defaults(func=cmd_one_second)
    z = sub.add_parser("one-second-zoom")
    z.add_argument("--output", default=str(PROJECT / "output/growth-archive-pilot-20261117"))
    z.set_defaults(func=cmd_one_second_zoom)
    g = sub.add_parser("page")
    g.add_argument("--day", required=True)
    g.add_argument("--output", default=str(PROJECT / "output/growth-archive-pilot-20261117"))
    g.set_defaults(func=cmd_page)

    from .growth_v2 import cmd_download_asr, cmd_page_v2, cmd_run_v2
    da = sub.add_parser("download-asr")
    da.add_argument("--config", default=str(PROJECT / "config/growth_archive.toml"))
    da.add_argument("--fallback", action="store_true")
    da.set_defaults(func=cmd_download_asr)

    rv = sub.add_parser("run-v2")
    rv.add_argument("--config", default=str(PROJECT / "config/growth_archive.toml"))
    rv.add_argument("--stage", default="all", choices=["all", "health", "kitchen", "bedside", "asr", "vlm", "render", "page"])
    rv.add_argument("--workers", type=int, default=6)
    rv.add_argument("--max-asr-windows", type=int, default=240)
    rv.set_defaults(func=cmd_run_v2)

    pv = sub.add_parser("page-v2")
    pv.add_argument("--config", default=str(PROJECT / "config/growth_archive.toml"))
    pv.set_defaults(func=cmd_page_v2)

    from .growth_v3 import cmd_run_v3
    r3 = sub.add_parser("run-v3")
    r3.add_argument("--config", default=str(PROJECT / "config/growth_archive.toml"))
    r3.add_argument("--stage", default="all", choices=["all", "health", "asr", "mine", "vlm", "render", "page"])
    r3.add_argument("--max-asr-windows", type=int, default=800)
    r3.set_defaults(func=cmd_run_v3)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
