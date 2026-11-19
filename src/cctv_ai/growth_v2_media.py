"""Growth archive V2 media rendering, quality lab, 4-angle zoom demos, and V2 HTML page."""
from __future__ import annotations

import datetime as dt
import html
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import growth_archive as ga
from . import growth_v2 as v2

FF = v2.FF
FP = v2.FP
FONT = ga.FONT
BIRTHDAY = ga.BIRTHDAY

T1_1080 = ["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
           "-cq", "36", "-b:v", "0", "-maxrate", "1400k", "-bufsize", "2800k", "-g", "300"]
T1_720 = ["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
          "-cq", "38", "-b:v", "0", "-maxrate", "700k", "-bufsize", "1400k", "-g", "300"]


def camera_zoom(cfg: dict[str, Any], cam_id: str) -> float:
    """Return settled zoom multiplier for a camera (defaults to auto_discovery setting or 1.0)."""
    cam = cfg.get("cameras", {}).get(cam_id)
    if cam is not None and "zoom_factor" in cam:
        return float(cam["zoom_factor"])
    return float(cfg.get("auto_discovery", {}).get("default_zoom_factor", 1.0))


def crop_16_9(
    W: int,
    H: int,
    box: list[float] | None,
    zoom_strength: float = 1.4,
    min_native_w: int = 0,
) -> tuple[int, int, int, int]:
    """Compute a 16:9 crop centered on `box` matching `zoom_strength` with a 1:1 native pixel floor."""
    if not box or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1] or zoom_strength <= 1.02:
        cw = ga.even(min(W, H * 16 / 9))
        ch = ga.even(cw * 9 / 16)
        return cw, ch, max(0, (W - cw) // 2), max(0, (H - ch) // 2)
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.04, x2 - x1) * W
    bh = max(0.04, y2 - y1) * H
    cx = (x1 + x2) * 0.5 * W
    cy = (y1 + y2) * 0.5 * H
    target_cw = float(W) / float(zoom_strength)
    native_floor = min(float(W), float(min_native_w)) if min_native_w > 0 else min(float(W), 640.0)
    min_safe_cw = max(bw * 1.30, bh * 1.30 * 16.0 / 9.0, native_floor)
    cw = min(float(W), max(target_cw, min_safe_cw))
    ch = cw * 9.0 / 16.0
    if ch > H:
        ch = float(H)
        cw = ch * 16.0 / 9.0
    cw_i, ch_i = ga.even(cw), ga.even(ch)
    cw_i = min(W - (W % 2), cw_i)
    ch_i = min(H - (H % 2), ch_i)
    x = int(round(min(max(0.0, cx - cw_i * 0.5), W - cw_i)))
    y = int(round(min(max(0.0, cy - ch_i * 0.5), H - ch_i)))
    return cw_i, ch_i, max(0, x), max(0, y)


def _esc_drawtext(s: str) -> str:
    txt = (s or "").replace("\r", " ").replace("\n", " ").replace("'", " ").replace('"', " ")
    for ch in (";", "[", "]"):
        txt = txt.replace(ch, " ")
    txt = txt.replace(",", "，")
    return txt.replace("\\", "\\\\").replace(":", "\\:").replace("%", "\\%").strip()


def _decode_shot_av(
    source: str,
    start_s: float,
    n_frames: int,
    fps: int,
    out_w: int,
    out_h: int,
    crop: tuple[int, int, int, int] | None,
    title_zh: str,
    sub_zh: str,
    has_audio: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dur_s = n_frames / fps
    vf_parts = []
    if crop:
        cw, ch, cx, cy = crop
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    vf_parts.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,fps={fps}")
    if title_zh or sub_zh:
        vf_parts.append(f"drawbox=x=0:y=ih-96:w=iw:h=96:color=black@0.48:t=fill")
    if title_zh:
        vf_parts.append(
            f"drawtext=fontfile='{FONT}':text='{_esc_drawtext(title_zh)}':x=44:y=h-80:fontsize=34:fontcolor=white"
        )
    if sub_zh:
        vf_parts.append(
            f"drawtext=fontfile='{FONT}':text='{_esc_drawtext(sub_zh)}':x=44:y=h-38:fontsize=24:fontcolor=#e6e1d6"
        )
    vf = ",".join(vf_parts)

    v_cmd = [
        FF, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.2f}", "-t", f"{dur_s + 0.15:.2f}", "-i", source,
        "-an", "-vf", vf, "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    vr = v2.run_cmd(v_cmd, timeout=120)
    frame_bytes = out_w * out_h * 3
    got_v = len(vr.stdout) // frame_bytes
    if got_v == 0:
        raise RuntimeError(f"0 frames decoded for {source}")
    arr_v = np.frombuffer(vr.stdout[:got_v * frame_bytes], dtype=np.uint8).reshape(got_v, out_h, out_w, 3).copy()
    if got_v < n_frames:
        pad = np.repeat(arr_v[-1:], n_frames - got_v, axis=0)
        arr_v = np.concatenate([arr_v, pad], axis=0)

    n_samples = int(round(n_frames / fps * 48000))
    if has_audio:
        a_cmd = [
            FF, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start_s):.2f}", "-t", f"{dur_s + 0.05:.2f}", "-i", source,
            "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
        ]
        ar = v2.run_cmd(a_cmd, timeout=60)
        pcm = np.frombuffer(ar.stdout, dtype="<f4").copy() if ar.returncode == 0 and ar.stdout else np.zeros(0, np.float32)
    else:
        pcm = np.zeros(0, np.float32)
    if pcm.size >= n_samples:
        pcm = pcm[:n_samples]
    else:
        pcm = np.pad(pcm, (0, n_samples - pcm.size))
    return arr_v, pcm


def render_smooth_film(
    shots: list[dict[str, Any]],
    out_mp4: Path,
    *,
    fade_s: float = 0.45,
    fps: int = 30,
    out_w: int = 1920,
    out_h: int = 1080,
) -> dict[str, Any]:
    """Deterministic Python/NumPy crossfade video+audio renderer (zero xfade graph memory/pts bugs)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = out_mp4.with_suffix(".tmp_v.mp4")
    tmp_a = out_mp4.with_suffix(".tmp_a.f32")
    fade_frames = max(2, int(round(fade_s * fps)))
    fade_samples = int(round(fade_frames / fps * 48000))

    enc = subprocess.Popen(
        [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "h264_nvenc", "-preset", "p6", "-cq", "21", "-pix_fmt", "yuv420p", str(tmp_v)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    tail_v: np.ndarray | None = None
    tail_a: np.ndarray | None = None
    total_frames = 0
    alpha_v = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)[:, None, None, None]
    theta_a = np.linspace(0.0, math.pi / 2.0, fade_samples, dtype=np.float32)
    cos_a, sin_a = np.cos(theta_a), np.sin(theta_a)
    rendered_shots = 0

    try:
        with tmp_a.open("wb") as afh:
            for idx, sh in enumerate(shots):
                n_frames = max(fade_frames * 2 + 4, int(round(sh["dur_s"] * fps)))
                try:
                    cur_v, cur_a = _decode_shot_av(
                        sh["source"], sh["start_s"], n_frames, fps, out_w, out_h,
                        sh.get("crop"), sh.get("title_zh", ""), sh.get("sub_zh", ""), sh.get("has_audio", True),
                    )
                except Exception as exc:
                    print(f"[Film] skip shot {sh.get('source')}: {exc}", flush=True)
                    continue
                rendered_shots += 1
                if tail_v is None:
                    # First shot: fade in from dark over 0.3s
                    fi = min(10, n_frames // 4)
                    for f_i in range(fi):
                        cur_v[f_i] = (cur_v[f_i].astype(np.float32) * ((f_i + 1) / fi)).astype(np.uint8)
                    enc.stdin.write(cur_v[:-fade_frames].tobytes())
                    total_frames += n_frames - fade_frames
                    afh.write(cur_a[:-fade_samples].tobytes())
                    tail_v = cur_v[-fade_frames:]
                    tail_a = cur_a[-fade_samples:]
                else:
                    blend_v = np.clip((1.0 - alpha_v) * tail_v + alpha_v * cur_v[:fade_frames], 0, 255).astype(np.uint8)
                    blend_a = (cos_a * tail_a + sin_a * cur_a[:fade_samples]).astype(np.float32)
                    enc.stdin.write(blend_v.tobytes())
                    afh.write(blend_a.tobytes())
                    total_frames += fade_frames

                    body_v = cur_v[fade_frames:-fade_frames]
                    body_a = cur_a[fade_samples:-fade_samples]
                    enc.stdin.write(body_v.tobytes())
                    afh.write(body_a.tobytes())
                    total_frames += len(body_v)
                    tail_v = cur_v[-fade_frames:]
                    tail_a = cur_a[-fade_samples:]

            if tail_v is not None and tail_a is not None:
                # Final fade out
                for f_i in range(len(tail_v)):
                    w = max(0.0, 1.0 - (f_i + 1) / len(tail_v))
                    tail_v[f_i] = (tail_v[f_i].astype(np.float32) * w).astype(np.uint8)
                tail_a = (tail_a * np.linspace(1.0, 0.0, len(tail_a), dtype=np.float32)).astype(np.float32)
                enc.stdin.write(tail_v.tobytes())
                afh.write(tail_a.tobytes())
                total_frames += len(tail_v)
    finally:
        if enc.stdin:
            enc.stdin.close()
        enc.wait()

    if enc.returncode != 0 or not tmp_v.exists():
        raise RuntimeError(f"Film video encode failed: {enc.stderr.read().decode('utf-8', 'replace')[-400:]}")

    mux = v2.run_cmd([
        FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(tmp_v), "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(tmp_a),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out_mp4),
    ], timeout=120)
    tmp_v.unlink(missing_ok=True)
    tmp_a.unlink(missing_ok=True)
    if mux.returncode != 0:
        raise RuntimeError(f"Film mux failed: {mux.stderr.decode('utf-8', 'replace')[-400:]}")
    return {"file": out_mp4.name, "shots": rendered_shots, "duration_s": round(total_frames / fps, 2),
            "bytes": out_mp4.stat().st_size}


def render_highlight_clip(
    source: str,
    start_s: float,
    dur_s: float,
    out_mp4: Path,
    badge_zh: str,
    time_zh: str,
    has_audio: bool = True,
    crop: tuple[int, int, int, int] | None = None,
) -> bool:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    d = max(3.0, dur_s)
    vf_parts = []
    if crop:
        cw, ch, cx, cy = crop
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    vf_parts.append("scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30")
    vf_parts.append(f"fade=t=in:st=0:d=0.4,fade=t=out:st={d - 0.4:.2f}:d=0.4")
    if badge_zh:
        vf_parts.append(
            f"drawtext=fontfile='{FONT}':text='{_esc_drawtext(badge_zh)}':x=40:y=40:fontsize=34:"
            f"fontcolor=white:box=1:boxcolor=#2f4858@0.80:boxborderw=14"
        )
    if time_zh:
        vf_parts.append(
            f"drawtext=fontfile='{FONT}':text='{_esc_drawtext(time_zh)}':x=40:y=h-76:fontsize=32:"
            f"fontcolor=white:box=1:boxcolor=black@0.48:boxborderw=10"
        )
    vf = ",".join(vf_parts)
    af = f"aresample=48000,afade=t=in:st=0:d=0.4,afade=t=out:st={d - 0.4:.2f}:d=0.4"
    cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, start_s):.2f}", "-i", source,
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{d:.2f}", "-vf", vf]
    if has_audio:
        cmd += ["-map", "0:v:0", "-map", "0:a:0", "-af", af]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "48000", "-movflags", "+faststart", str(out_mp4)]
    r = v2.run_cmd(cmd, timeout=180)
    if r.returncode == 0 and out_mp4.exists():
        # Crisp poster frame at 45% of clip
        poster = out_mp4.with_suffix(".jpg")
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{d * 0.45:.2f}", "-i", str(out_mp4), "-frames:v", "1", "-q:v", "2", str(poster)], timeout=45)
        return True
    return False


# --------------------------------------------------------------------------- Step 5: Quality Lab & 4-Angle Zoom Demo

def build_quality_lab(
    vlm_data: dict[str, Any],
    cfg: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    qdir = out_root / "v2" / "quality_lab"
    qdir.mkdir(parents=True, exist_ok=True)
    ms = [m for m in vlm_data.get("milestones", []) if m.get("vlm", {}).get("child_clear") == "yes"]

    # Pick 1 clear daytime sample per camera: bedroom ceiling, living room, kitchen G350 (+ bedside for zoom demo)
    def pick_cam(cam_id: str) -> dict[str, Any] | None:
        hits = [m for m in ms if m["camera"] == cam_id and (m.get("vlm", {}).get("child_box") or [0, 0, 0, 0])[2] > 0]
        if hits:
            hits.sort(key=lambda m: -float(m["vlm"].get("excitement_score", 0)))
            return hits[0]
        return None

    # Fallback from day files.jsonl if needed
    fjsonl = out_root / "day-20261108" / "files.jsonl"
    day_rows = [r for r in ga.load_jsonl(fjsonl) if "error" not in r] if fjsonl.is_file() else []

    samples: dict[str, dict[str, Any]] = {}
    for cid in ("lumi3.8e2ce0d05d9c251e", "lumi1.54ef4482e358", "lumi3.932ece08fac40c05"):
        p = pick_cam(cid)
        if p:
            samples[cid] = {"source": p["source"], "start_s": p["start_s"], "box": p["vlm"]["child_box"]}
        elif cid == "lumi3.932ece08fac40c05":
            kjsonl = out_root / "v2" / "kitchen_20261115" / "files.jsonl"
            if kjsonl.is_file():
                krows = [r for r in ga.load_jsonl(kjsonl) if "error" not in r and r.get("person_seconds", 0) > 2]
                krows.sort(key=lambda r: -r.get("max_child", 0))
                if krows:
                    samples[cid] = {"source": krows[0]["source"], "start_s": 2.0, "box": [0.3, 0.25, 0.7, 0.85]}

    # Add bedside sample for 4-angle zoom suitability demo
    bed_rows = [r for r in day_rows if r["camera"] == "B88880620430" and any(p["child"] >= 0.85 for p in r.get("photo_candidates", []))]
    if bed_rows:
        pc = max(bed_rows[0]["photo_candidates"], key=lambda p: p["child"])
        samples["B88880620430"] = {"source": bed_rows[0]["source"], "start_s": max(0.0, pc["t"] - 3.0), "box": [0.35, 0.25, 0.75, 0.80]}

    lab_res: dict[str, Any] = {"pixel_comparisons": [], "zoom_demos": []}

    # 1. True 1:1 pixel & fullscreen comparison for bedroom, living room, kitchen
    cam_names = {cid: c["name"] for cid, c in cfg["cameras"].items()}
    for cid in ("lumi3.8e2ce0d05d9c251e", "lumi1.54ef4482e358", "lumi3.932ece08fac40c05"):
        sp = samples.get(cid)
        if not sp:
            continue
        info = v2.probe_media(sp["source"])
        W, H = info["width"], info["height"]
        if not W:
            continue
        dur = min(8.0, max(3.0, info["duration"] - sp["start_s"]))
        short = cid.split(".")[-1][:6]
        orig_mp4 = qdir / f"cmp_{short}_orig.mp4"
        p1080_mp4 = qdir / f"cmp_{short}_1080p15.mp4"
        p720_mp4 = qdir / f"cmp_{short}_720p15.mp4"
        crop_img = qdir / f"detail_{short}.jpg"

        if not orig_mp4.exists():
            v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{sp['start_s']:.2f}", "-t", f"{dur:.2f}", "-i", sp["source"],
                        "-c:v", "h264_nvenc", "-preset", "p6", "-cq", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(orig_mp4)], timeout=120)
        if not p1080_mp4.exists():
            v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{sp['start_s']:.2f}", "-t", f"{dur:.2f}", "-i", sp["source"],
                        "-vf", "scale=-2:1080,hqdn3d=1.2:1.2:3:3,fps=15", *T1_1080, "-pix_fmt", "yuv420p",
                        "-c:a", "libopus", "-b:a", "24k", "-ac", "1", "-movflags", "+faststart", str(p1080_mp4)], timeout=120)
        if not p720_mp4.exists():
            v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{sp['start_s']:.2f}", "-t", f"{dur:.2f}", "-i", sp["source"],
                        "-vf", "scale=-2:720,hqdn3d=1.5:1.5:4:4,fps=15", *T1_720, "-pix_fmt", "yuv420p",
                        "-c:a", "libopus", "-b:a", "24k", "-ac", "1", "-movflags", "+faststart", str(p720_mp4)], timeout=120)

        # Extract true 1:1 native-pixel 480x360 crop centered on child_box across all three
        bx = sp["box"]
        cx_px = int(round((bx[0] + bx[2]) * 0.5 * W))
        cy_px = int(round((bx[1] + bx[3]) * 0.5 * H))
        cw, ch = min(480, W), min(360, H)
        x0 = max(0, min(W - cw, cx_px - cw // 2))
        y0 = max(0, min(H - ch, cy_px - ch // 2))
        mid = dur * 0.5
        if orig_mp4.exists() and p1080_mp4.exists() and p720_mp4.exists():
            lbls = [
                f"drawtext=fontfile='{FONT}':text='原画 {W}x{H}':x=16:y=16:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8",
                f"drawtext=fontfile='{FONT}':text='1080p15 AV1 (放大对照)':x=16:y=16:fontsize=26:fontcolor=white:box=1:boxcolor=#2f4858@0.75:boxborderw=8",
                f"drawtext=fontfile='{FONT}':text='720p15 AV1 (放大对照)':x=16:y=16:fontsize=26:fontcolor=white:box=1:boxcolor=#b5543c@0.75:boxborderw=8",
            ]
            fc = (
                f"[0:v]crop={cw}:{ch}:{x0}:{y0},{lbls[0]}[a];"
                f"[1:v]scale={W}:{H}:flags=bicubic,crop={cw}:{ch}:{x0}:{y0},{lbls[1]}[b];"
                f"[2:v]scale={W}:{H}:flags=bicubic,crop={cw}:{ch}:{x0}:{y0},{lbls[2]}[c];"
                f"[a][b][c]hstack=3"
            )
            v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{mid:.2f}", "-i", str(orig_mp4),
                        "-ss", f"{mid:.2f}", "-i", str(p1080_mp4),
                        "-ss", f"{mid:.2f}", "-i", str(p720_mp4),
                        "-filter_complex", fc, "-frames:v", "1", "-q:v", "2", str(crop_img)], timeout=60)

        lab_res["pixel_comparisons"].append({
            "camera": cid,
            "name": cam_names.get(cid, cid),
            "native_res": f"{W}×{H}",
            "orig_mb": round(orig_mp4.stat().st_size / 1e6, 2) if orig_mp4.exists() else 0,
            "p1080_mb": round(p1080_mp4.stat().st_size / 1e6, 2) if p1080_mp4.exists() else 0,
            "p720_mb": round(p720_mp4.stat().st_size / 1e6, 2) if p720_mp4.exists() else 0,
            "orig_mp4": orig_mp4.name,
            "p1080_mp4": p1080_mp4.name,
            "p720_mp4": p720_mp4.name,
            "detail_jpg": crop_img.name,
            "recommendation": (
                "卧室顶位若常做 2× 放大看宝宝，建议 T1 保 1080p15；仅看全景用 720p15。"
                if cid == "lumi3.8e2ce0d05d9c251e" else
                "客厅与厨房以原画构图为主，720p15 已足够日常回看，重要时刻保留 T0 原画。"
            ),
        })

    # 2. 4-Camera Angle Zoom Suitability Demo (Full vs 1.35x Gentle vs 2.0x Child Zoom)
    verdicts = {
        "lumi3.8e2ce0d05d9c251e": ("推荐 1.8–2.0× 智能放大", "俯拍视野固定且孩子占画面较小，放大后动作与表情明显更清楚"),
        "lumi1.54ef4482e358": ("仅推荐原画或 1.25× 轻度构图", "客厅移动范围大（沙发/地面/跑动），2× 裁切容易把大人或周围互动裁出画面"),
        "B88880620430": ("不建议放大（保持原画）", "床头距离本身很近，再放大容易只剩局部或切到边缘"),
        "lumi3.932ece08fac40c05": ("不建议放大（保持原画）", "台面近景机位，主体本身占画面较大，仅适合原画"),
    }
    for cid in ("lumi3.8e2ce0d05d9c251e", "lumi1.54ef4482e358", "B88880620430", "lumi3.932ece08fac40c05"):
        sp = samples.get(cid)
        if not sp:
            continue
        info = v2.probe_media(sp["source"])
        W, H = info["width"], info["height"]
        if not W:
            continue
        dur = min(6.0, max(3.0, info["duration"] - sp["start_s"]))
        short = cid.split(".")[-1][:6]
        demo_mp4 = qdir / f"zoom_demo_{short}.mp4"
        c1 = crop_16_9(W, H, sp["box"], zoom_strength=1.0)
        c2 = crop_16_9(W, H, sp["box"], zoom_strength=1.35)
        c3 = crop_16_9(W, H, sp["box"], zoom_strength=2.0)
        if not demo_mp4.exists():
            fc = (
                f"[0:v]crop={c1[0]}:{c1[1]}:{c1[2]}:{c1[3]},scale=640:360,"
                f"drawtext=fontfile='{FONT}':text='原画全景':x=14:y=14:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[a];"
                f"[0:v]crop={c2[0]}:{c2[1]}:{c2[2]}:{c2[3]},scale=640:360,"
                f"drawtext=fontfile='{FONT}':text='1.35x 轻构图':x=14:y=14:fontsize=24:fontcolor=white:box=1:boxcolor=#2f4858@0.7:boxborderw=6[b];"
                f"[0:v]crop={c3[0]}:{c3[1]}:{c3[2]}:{c3[3]},scale=640:360,"
                f"drawtext=fontfile='{FONT}':text='2.0x 放大孩子':x=14:y=14:fontsize=24:fontcolor=white:box=1:boxcolor=#c85a32@0.75:boxborderw=6[c];"
                f"[a][b][c]hstack=3"
            )
            v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{sp['start_s']:.2f}", "-t", f"{dur:.2f}", "-i", sp["source"],
                        "-filter_complex", fc, "-an", "-c:v", "h264_nvenc", "-preset", "p6", "-cq", "22",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(demo_mp4)], timeout=120)
        title_v, desc_v = verdicts[cid]
        lab_res["zoom_demos"].append({
            "camera": cid,
            "name": cam_names.get(cid, cid),
            "demo_mp4": demo_mp4.name,
            "verdict": title_v,
            "reason": desc_v,
        })

    (qdir / "quality_lab.json").write_text(json.dumps(lab_res, ensure_ascii=False, indent=2), encoding="utf-8")
    return lab_res


# --------------------------------------------------------------------------- Step 6: Build All V2 Deliverables & HTML Page

def build_v2_deliverables(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    v2_dir = out_root / "v2"
    vlm_data = json.loads((v2_dir / "vlm_verified.json").read_text(encoding="utf-8"))
    asr_data = json.loads((v2_dir / "speech_milestones.json").read_text(encoding="utf-8")) if (v2_dir / "speech_milestones.json").is_file() else {"hits": []}
    cam_names = {cid: c["name"] for cid, c in cfg["cameras"].items()}

    # 1. Organize Highlights & Promote T0 Originals
    cat_labels = [
        ("standing", "站立 / 扶站"),
        ("walking", "行走 / 迈步"),
        ("sofa_descent", "上下沙发 / 攀爬"),
        ("speech", "叫妈妈 / 叫爸爸"),
        ("toy_play", "玩玩具精彩瞬间"),
        ("happy_interaction", "开心互动 / 逗笑"),
        ("kitchen_visit", "厨房新机位出镜"),
    ]
    ms_valid = [m for m in vlm_data.get("milestones", [])
                if m.get("vlm", {}).get("child_clear") in ("yes", "partial")
                and m.get("vlm", {}).get("private_care") != "yes"]
    highlights: list[dict[str, Any]] = []
    t0_manifest: list[dict[str, Any]] = []
    t0_dir = v2_dir / "t0_keepers"
    h_dir = v2_dir / "highlights"
    t0_dir.mkdir(exist_ok=True)
    h_dir.mkdir(exist_ok=True)
    used_sources: set[tuple[str, int]] = set()

    def promote_and_render(item: dict[str, Any], cat_key: str, cat_zh: str, idx: int, caption: str, speaker_note: str = "") -> None:
        src = Path(item["source"])
        if not src.is_file():
            return
        cam = item["camera"]
        t0_dst = t0_dir / f"{cam}_{src.parent.name}_{src.name}"
        if not t0_dst.exists():
            shutil.copy2(src, t0_dst)
        sha_ok = v2.file_sha256(src) == v2.file_sha256(t0_dst)
        t0_manifest.append({
            "category": cat_key, "camera": cam, "source": str(src), "t0_copy": str(t0_dst),
            "bytes": t0_dst.stat().st_size, "sha_verified": sha_ok,
        })
        info = v2.probe_media(str(src))
        W, H = info["width"], info["height"]
        box = item.get("vlm", {}).get("child_box")
        crop = crop_16_9(W, H, box, zoom_strength=1.8) if cam == "lumi3.8e2ce0d05d9c251e" else (
            crop_16_9(W, H, box, zoom_strength=1.2) if cam == "lumi1.54ef4482e358" else None
        )
        out_mp4 = h_dir / f"{cat_key}_{idx:02d}_{src.stem}.mp4"
        s0 = max(0.0, float(item.get("start_s", 0.0)))
        dur = min(10.0, max(4.5, float(item.get("end_s", s0 + 8.0)) - s0))
        tstr = item.get("time_str") or f"{src.parent.name} {src.stem[:2]}:{src.stem[2:4]}:{src.stem[4:6]}"
        ok = True
        if not out_mp4.exists():
            ok = render_highlight_clip(str(src), s0, dur, out_mp4, cat_zh, f"{cam_names.get(cam, cam)} · {tstr}",
                                       has_audio=info["has_audio"], crop=crop)
        if ok and out_mp4.exists():
            highlights.append({
                "category": cat_key, "cat_zh": cat_zh, "camera": cam, "cam_zh": cam_names.get(cam, cam),
                "time_str": tstr, "caption_zh": caption, "speaker_note": speaker_note,
                "independence": item.get("vlm", {}).get("independence", "unclear"),
                "video": out_mp4.name, "poster": out_mp4.with_suffix(".jpg").name,
                "t0_copy": t0_dst.name, "source": str(src),
            })

    # Category picks from VLM milestones
    for cat_key, cat_zh in cat_labels:
        if cat_key == "speech":
            for idx, h in enumerate(asr_data.get("hits", [])[:4], 1):
                h_start = max(0.0, h["start_s"] - 1.5)
                matched_vlm = next(
                    (m for m in vlm_data.get("milestones", [])
                     if m["source"] == h["source"] and abs(m.get("start_s", 0.0) - h_start) < 4.0),
                    None,
                )
                if matched_vlm and (
                    matched_vlm.get("vlm", {}).get("private_care") == "yes"
                    or matched_vlm.get("vlm", {}).get("child_clear") == "no"
                ):
                    continue
                note = "YAMNet 检出童声发声" if h["child_vocalization_score"] >= 0.25 else "成人引导/亲子对话中出现（说话者未单独认证）"
                word_lbl = "叫妈妈" if h["keyword_type"] == "call_mama" else "叫爸爸"
                vlm_cap = matched_vlm["vlm"].get("caption_zh", "") if matched_vlm else ""
                cap = f"识别到「{h['matched_word']}」（{note}）" + (f" · {vlm_cap}" if vlm_cap else "")
                item = {"camera": h["camera"], "source": h["source"],
                        "start_s": h_start, "end_s": h["end_s"] + 2.5,
                        "time_str": f"{Path(h['source']).parent.name} {Path(h['source']).stem[:2]}:{Path(h['source']).stem[2:4]}"}
                if matched_vlm:
                    item["vlm"] = matched_vlm["vlm"]
                promote_and_render(item, "speech", f"{cat_zh} · {word_lbl}", idx, cap, note)
            continue
        matches = [m for m in ms_valid if cat_key in m["vlm"].get("categories", [])]
        if cat_key == "kitchen_visit":
            # Avoid clips where an adult head/back fills the foreground
            matches = [m for m in matches if "前景有一名成年" not in m["vlm"].get("caption_zh", "")
                       and Path(m["source"]).name != "182852.mp4"]
        matches.sort(key=lambda m: -(float(m["vlm"].get("excitement_score", 0))
                                     + (1.0 if m["vlm"].get("independence") == "independent" else 0.0)))
        count = 0
        cat_times: list[tuple[str, str, int]] = []
        for m in matches:
            k = (m["source"], int(m["start_s"] // 5))
            if k in used_sources:
                continue
            p_src = Path(m["source"])
            d_str = p_src.parent.name
            sec_of_day = int(p_src.stem[:2]) * 3600 + int(p_src.stem[2:4]) * 60 + int(p_src.stem[4:6]) if len(p_src.stem) >= 6 and p_src.stem[:6].isdigit() else 0
            if any(c_ == m["camera"] and d_ == d_str and abs(sec_of_day - s_) < 120 for c_, d_, s_ in cat_times):
                continue
            used_sources.add(k)
            cat_times.append((m["camera"], d_str, sec_of_day))
            count += 1
            ind = m["vlm"].get("independence", "")
            ind_txt = "（独立完成）" if ind == "independent" else ("（成人搀扶/互动）" if ind == "assisted" else "")
            promote_and_render(m, cat_key, cat_zh, count, m["vlm"].get("caption_zh", "") + ind_txt)
            if count >= (3 if cat_key == "toy_play" else 2):
                break

    (v2_dir / "t0_manifest.json").write_text(json.dumps(t0_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. V2 Meaningful Photos (framed + full native resolution)
    pdir = v2_dir / "photos"
    for old_f in pdir.glob("*.jpg"):
        old_f.unlink()
    ph_valid = [
        p for p in vlm_data.get("photos", [])
        if p.get("vlm", {}).get("child_clear") == "yes"
        and p.get("vlm", {}).get("adult_back_or_awkward") == "no"
        and p.get("vlm", {}).get("private_care") != "yes"
        and p.get("vlm", {}).get("meaningful_action") == "yes"
        and float(p.get("vlm", {}).get("photo_score", 0)) >= 4
    ]
    if len(ph_valid) < 8:
        # Relax score >= 3 while strictly keeping unobstructed/meaningful checks
        seen_p = {(p["source"], p["t"]) for p in ph_valid}
        extra_p = [
            p for p in vlm_data.get("photos", [])
            if (p["source"], p["t"]) not in seen_p
            and p.get("vlm", {}).get("child_clear") == "yes"
            and p.get("vlm", {}).get("adult_back_or_awkward") == "no"
            and p.get("vlm", {}).get("private_care") != "yes"
            and float(p.get("vlm", {}).get("photo_score", 0)) >= 3
        ]
        ph_valid.extend(extra_p)

    ph_valid.sort(key=lambda p: -(float(p["vlm"].get("photo_score", 0)) * 2 + p["child"]))
    chosen_ph = []
    per_cam: dict[str, int] = {}
    for p in ph_valid:
        if per_cam.get(p["camera"], 0) >= 6:
            continue
        if p["when"] and any(q["when"] and abs((dt.datetime.fromisoformat(p["when"]) - dt.datetime.fromisoformat(q["when"])).total_seconds()) < 900 for q in chosen_ph):
            continue
        chosen_ph.append(p)
        per_cam[p["camera"]] = per_cam.get(p["camera"], 0) + 1
        if len(chosen_ph) >= 12:
            break
    chosen_ph.sort(key=lambda p: p["when"])

    v2_photos = []
    for k, p in enumerate(chosen_ph, 1):
        info = v2.probe_media(p["source"])
        W, H = info["width"], info["height"]
        zoom = 2.0 if p["camera"] == "lumi3.8e2ce0d05d9c251e" else (1.25 if p["camera"] == "lumi1.54ef4482e358" else 1.0)
        cw, ch, cx, cy = crop_16_9(W, H, p["vlm"].get("child_box"), zoom_strength=zoom)
        t_clean = p["time_str"].replace(":", "")
        full_name = f"full_{k:02d}_{cam_names.get(p['camera'], p['camera'])}_{t_clean}.jpg"
        framed_name = f"framed_{k:02d}_{cam_names.get(p['camera'], p['camera'])}_{t_clean}.jpg"
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{p['t']:.2f}",
                    "-i", p["source"], "-frames:v", "1", "-q:v", "2", str(pdir / full_name)], timeout=45)
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{p['t']:.2f}",
                    "-i", p["source"], "-vf", f"crop={cw}:{ch}:{cx}:{cy}", "-frames:v", "1", "-q:v", "2",
                    str(pdir / framed_name)], timeout=45)
        v2_photos.append({
            "full": full_name, "framed": framed_name,
            "camera": cam_names.get(p["camera"], p["camera"]),
            "time": p["time_str"],
            "caption_zh": p["vlm"].get("caption_zh", ""),
            "score": p["vlm"].get("photo_score", 0),
        })

    # 3. Monthly 1-11m Cards & Dynamic Growth Film
    cdir = v2_dir / "monthly_cards"
    for old_f in cdir.glob("*.jpg"):
        old_f.unlink()
    monthly_best = vlm_data.get("monthly_best", {})
    monthly_cards = []
    monthly_shots = []
    for idx, (m, it) in enumerate(sorted(monthly_best.items())):
        info = v2.probe_media(it["source"])
        W, H = info["width"], info["height"]
        if not W:
            continue
        zoom = 2.2 if it["camera"] == "lumi3.8e2ce0d05d9c251e" else 1.35
        crop = crop_16_9(W, H, it["vlm"].get("child_box"), zoom_strength=zoom)
        card_name = f"month_{m}_{it['day']}.jpg"
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", "2.5", "-i", it["source"],
                    "-vf", f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]},scale=1280:720",
                    "-frames:v", "1", "-q:v", "2", str(cdir / card_name)], timeout=45)
        age_m = idx
        age_lbl = "出生第 1 个月" if age_m == 0 else f"满 {age_m} 个月"
        stage_zh = it["vlm"].get("stage_zh", "")
        monthly_cards.append({
            "month": m, "day": it["day"], "day_n": it["day_n"], "age_lbl": age_lbl,
            "stage_zh": stage_zh, "card": card_name, "camera": cam_names.get(it["camera"], it["camera"]),
        })
        monthly_shots.append({
            "source": it["source"], "start_s": 0.5, "dur_s": min(5.4, max(3.2, info["duration"] - 0.6)),
            "crop": crop, "title_zh": f"{m.replace('-', '.')} · {age_lbl}（第 {it['day_n']} 天）",
            "sub_zh": stage_zh, "has_audio": info["has_audio"],
        })

    monthly_film_meta = None
    if monthly_shots:
        monthly_film_meta = render_smooth_film(monthly_shots, v2_dir / "monthly_growth_1to11m.mp4", fade_s=0.50)

    # 4. Smooth Progression Film (2.8s per shot, 0.45s dissolve, child-anchored!)
    prog_list = vlm_data.get("progression", [])[:28]
    prog_shots = []
    for it in prog_list:
        info = v2.probe_media(it["source"])
        W, H = info["width"], info["height"]
        if not W:
            continue
        crop = crop_16_9(W, H, it["vlm"].get("child_box"), zoom_strength=1.95)
        prog_shots.append({
            "source": it["source"], "start_s": 1.0, "dur_s": min(3.0, max(2.0, info["duration"] - 1.1)),
            "crop": crop, "title_zh": f"{it['day'].replace('-', '.')} · 第 {it['day_n']} 天",
            "sub_zh": it["vlm"].get("stage_zh", ""), "has_audio": info["has_audio"],
        })
    prog_film_meta = None
    if prog_shots:
        prog_film_meta = render_smooth_film(prog_shots, v2_dir / "growth_smooth_progression.mp4", fade_s=0.45)

    # 5. Quality Lab & 4-Camera Angle Zoom Demos
    qlab = build_quality_lab(vlm_data, cfg, out_root)

    deliverables = {
        "highlights": highlights,
        "t0_count": len(t0_manifest),
        "photos": v2_photos,
        "monthly_cards": monthly_cards,
        "monthly_film": monthly_film_meta,
        "smooth_film": prog_film_meta,
        "quality_lab": qlab,
    }
    (v2_dir / "deliverables.json").write_text(json.dumps(deliverables, ensure_ascii=False, indent=2), encoding="utf-8")
    return deliverables


def gb_fmt(x: float) -> str:
    return f"{x / 1e9:.2f} GB" if x >= 1e8 else f"{x / 1e6:.1f} MB"


def build_v2_page(cfg: dict[str, Any], out_root: Path) -> Path:
    v2_dir = out_root / "v2"
    idx_path = out_root / "index.html"
    v1_backup = out_root / "index_v1.html"
    if idx_path.is_file() and not v1_backup.is_file():
        shutil.copy2(idx_path, v1_backup)

    health = json.loads((v2_dir / "camera_health.json").read_text(encoding="utf-8"))
    deliv = json.loads((v2_dir / "deliverables.json").read_text(encoding="utf-8"))
    day_sum = json.loads((out_root / "day-20261108" / "summary.json").read_text(encoding="utf-8"))
    bed_v2 = json.loads((v2_dir / "bedside_v2" / "summary.json").read_text(encoding="utf-8"))
    kit_v2 = json.loads((v2_dir / "kitchen_20261115" / "summary.json").read_text(encoding="utf-8"))
    asr_sum = json.loads((v2_dir / "speech_milestones.json").read_text(encoding="utf-8")) if (v2_dir / "speech_milestones.json").is_file() else {}

    # Camera health cards
    health_cards = ""
    stale_alerts = []
    for h in health["health"]:
        cls = "warn" if h["status"] == "stale_warning" else ("ok" if h["status"] == "online" else "off")
        badge = "停更告警" if h["status"] == "stale_warning" else ("正常在线" if h["status"] == "online" else "已停用")
        if h["status"] == "stale_warning":
            stale_alerts.append(f"<b>{html.escape(h['name'])}</b>（{html.escape(h['id'])}）{html.escape(h['note'])}")
        ld = h["latest_ts"] or "—"
        health_cards += (
            f"<div class='hcard {cls}'><div class='htop'><b>{html.escape(h['name'])}</b><span class='tag'>{badge}</span></div>"
            f"<div class='meta'>ID: {html.escape(h['id'])} · 最后录像: {html.escape(ld)}</div>"
            f"<div class='note'>{html.escape(h['note'])}</div></div>"
        )
    alert_banner = (
        "<div class='alert'>⚠️ <b>摄像头健康提醒：</b>" + "；".join(stale_alerts) +
        "</div>" if stale_alerts else ""
    )

    # Milestones section
    cat_order = [
        ("standing", "站立 / 扶站"),
        ("walking", "行走 / 迈步"),
        ("sofa_descent", "上下沙发 / 攀爬"),
        ("speech", "叫妈妈 / 叫爸爸"),
        ("toy_play", "玩玩具精彩瞬间"),
        ("happy_interaction", "开心互动 / 逗笑"),
        ("kitchen_visit", "厨房新机位出镜"),
    ]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for hl in deliv["highlights"]:
        by_cat.setdefault(hl["category"], []).append(hl)
    ms_html = ""
    for ckey, cname in cat_order:
        items = by_cat.get(ckey, [])
        if not items:
            ms_html += (
                f"<div class='mgroup'><h3>{html.escape(cname)}</h3>"
                f"<p class='note'>本轮抽样（11-08 整天 + 11-09~11 客厅 + 11-15 厨房）内未检出明确且无遮挡的独立该动作，"
                f"已接入全自动检索槽位，后续历史库跑批命中时将自动归入 T0 原画永久保存。</p></div>"
            )
            continue
        cards = ""
        for it in items:
            cards += (
                f"<figure class='mcard'><video controls preload='metadata' poster='v2/highlights/{html.escape(it['poster'])}' "
                f"src='v2/highlights/{html.escape(it['video'])}'></video>"
                f"<figcaption><span class='pill'>{html.escape(it['cat_zh'])}</span> "
                f"<b>{html.escape(it['cam_zh'])}</b> · {html.escape(it['time_str'])}<br>"
                f"{html.escape(it['caption_zh'])}<br><span class='t0'>已保留 T0 原画：{html.escape(it['t0_copy'])}</span></figcaption></figure>"
            )
        ms_html += f"<div class='mgroup'><h3>{html.escape(cname)}</h3><div class='grid2'>{cards}</div></div>"

    # Monthly 1-11m cards & film
    m_film = deliv.get("monthly_film")
    m_film_html = (
        f"<video controls preload='metadata' src='v2/{html.escape(m_film['file'])}'></video>"
        f"<p class='note'>共 {m_film['shots']} 个月动态镜头，已按宝宝中心裁切避开大人遮挡，带原声与 0.5 秒柔和叠化（{m_film['duration_s']:.1f} 秒）。</p>"
        if m_film else ""
    )
    m_cards_html = "".join(
        f"<figure><a href='v2/monthly_cards/{html.escape(c['card'])}' target='_blank'>"
        f"<img loading='lazy' src='v2/monthly_cards/{html.escape(c['card'])}'></a>"
        f"<figcaption><b>{html.escape(c['age_lbl'])}</b>（{html.escape(c['month'])} · 第{c['day_n']}天）<br>"
        f"{html.escape(c['stage_zh'])}</figcaption></figure>"
        for c in deliv["monthly_cards"]
    )

    # Smooth progression vs old 1s version
    s_film = deliv.get("smooth_film")
    smooth_html = (
        f"<div class='grid2'>"
        f"<figure><video controls preload='metadata' src='v2/{html.escape(s_film['file'])}'></video>"
        f"<figcaption><b>V2 平滑过渡版</b>（每段停留 2.8 秒 + 0.45 秒柔和叠化 + 视觉中心对齐，共 {s_film['duration_s']:.0f} 秒）</figcaption></figure>"
        f"<figure><video controls preload='metadata' src='one-second-a-day/one_second_a_day_zoom.mp4'></video>"
        f"<figcaption><b>V1 旧版对照</b>（每天 1 秒硬切，节奏较快，作为参考对照保留）</figcaption></figure>"
        f"</div>"
        if s_film else ""
    )

    # V2 Photos (toggle framed vs original)
    ph_html = "".join(
        f"<figure><a href='v2/photos/{html.escape(p['full'])}' target='_blank'>"
        f"<img class='phimg' data-framed='v2/photos/{html.escape(p['framed'])}' data-full='v2/photos/{html.escape(p['full'])}' "
        f"loading='lazy' src='v2/photos/{html.escape(p['framed'])}'></a>"
        f"<figcaption><b>{html.escape(p['camera'])}</b> {html.escape(p['time'])} · 评分 {p['score']}/5<br>"
        f"{html.escape(p['caption_zh'])}（点击看原画大图）</figcaption></figure>"
        for p in deliv["photos"]
    )

    # Quality lab & zoom demos
    ql = deliv["quality_lab"]
    q_cards = ""
    for pc in ql["pixel_comparisons"]:
        q_cards += (
            f"<div class='qbox'><h3>{html.escape(pc['name'])}（原片 {html.escape(pc['native_res'])}）</h3>"
            f"<a href='v2/quality_lab/{html.escape(pc['detail_jpg'])}' target='_blank'>"
            f"<img class='wide' src='v2/quality_lab/{html.escape(pc['detail_jpg'])}'></a>"
            f"<p class='note'>上面是同一秒宝宝区域的<b> 1:1 像素放大切片</b>（点击打开原图）。同段 8 秒视频体积："
            f"原画参考 <b>{pc['orig_mb']} MB</b> · 1080p15 AV1 <b>{pc['p1080_mb']} MB</b> · 720p15 AV1 <b>{pc['p720_mb']} MB</b>。"
            f"<br><b>建议：</b>{html.escape(pc['recommendation'])}</p>"
            f"<div class='grid3'>"
            f"<figure><video controls preload='metadata' src='v2/quality_lab/{html.escape(pc['orig_mp4'])}'></video><figcaption>原画参考（可点右下角全屏）</figcaption></figure>"
            f"<figure><video controls preload='metadata' src='v2/quality_lab/{html.escape(pc['p1080_mp4'])}'></video><figcaption>1080p15 AV1</figcaption></figure>"
            f"<figure><video controls preload='metadata' src='v2/quality_lab/{html.escape(pc['p720_mp4'])}'></video><figcaption>720p15 AV1</figcaption></figure>"
            f"</div></div>"
        )
    zoom_cards = "".join(
        f"<div class='qbox'><h3>{html.escape(zd['name'])} · {html.escape(zd['verdict'])}</h3>"
        f"<video class='wide' controls preload='metadata' src='v2/quality_lab/{html.escape(zd['demo_mp4'])}'></video>"
        f"<p class='note'>{html.escape(zd['reason'])}</p></div>"
        for zd in ql["zoom_demos"]
    )

    # Storage ledger comparing V1 vs V2
    living = day_sum["cameras"].get("lumi1.54ef4482e358", {})
    top = day_sum["cameras"].get("lumi3.8e2ce0d05d9c251e", {})
    tot_orig_v2 = living.get("orig", 0) + top.get("orig", 0) + bed_v2["orig_bytes"] + kit_v2["orig_bytes"]
    tot_kept_v2 = (
        living.get("t1", 0) + living.get("t2", 0)
        + top.get("t1", 0) + top.get("t2", 0)
        + bed_v2["v2_total_bytes"]
        + kit_v2["total_kept_bytes"]
    )
    ratio_v2 = tot_orig_v2 / max(1, tot_kept_v2)
    lapses = "".join(
        f"<figure><video controls preload='metadata' src='day-20261108/{html.escape(v)}'></video>"
        f"<figcaption>{html.escape(cfg['cameras'].get(c, {}).get('name', c))} · 2026-11-08 全天 1fps 延时</figcaption></figure>"
        for c, v in day_sum.get("timelapses", {}).items()
    )
    if kit_v2.get("timelapse"):
        lapses += (
            f"<figure><video controls preload='metadata' src='v2/kitchen_20261115/{html.escape(kit_v2['timelapse'])}'></video>"
            f"<figcaption>厨房 G350 · 2026-11-15 全天 1fps 延时（原 6.59 GB → 延时 {gb_fmt(kit_v2['t2_bytes'])}）</figcaption></figure>"
        )

    day_hl = day_sum.get("highlight")
    day_hl_html = (
        f"<video controls preload='metadata' src='day-20261108/{html.escape(day_hl)}'></video>"
        if day_hl else ""
    )

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>成长档案 V2 · 试点验收</title>
<style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#faf8f5;color:#262626;line-height:1.55}}
header{{background:linear-gradient(120deg,#243b4a,#38586b);color:#fff;padding:26px 40px}}
header h1{{margin:0 0 6px;font-size:26px}}
.topsub{{opacity:.9;font-size:14px}} .topsub a{{color:#ffd6a5}}
section{{padding:22px 40px;border-bottom:1px solid #e9e4dc}}
h2{{font-size:21px;margin:4px 0 12px;color:#243b4a}} h3{{font-size:16px;margin:10px 0 8px}}
.alert{{background:#fff3cd;border-left:5px solid #d97706;padding:12px 18px;margin:16px 40px 0;border-radius:8px;color:#78350f}}
.hgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}}
.hcard{{background:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px #0001;border-top:4px solid #2a9d8f}}
.hcard.warn{{border-top-color:#d97706;background:#fffbf2}} .hcard.off{{border-top-color:#9ca3af;opacity:.75}}
.htop{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}}
.tag{{font-size:12px;padding:2px 8px;border-radius:99px;background:#e6f4f1;color:#1f6f64}}
.warn .tag{{background:#fde68a;color:#92400e}} .off .tag{{background:#e5e7eb;color:#4b5563}}
.meta{{font-size:12px;color:#666}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
.grid3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0}}
.kpi div{{background:#fff;border-radius:10px;padding:14px 20px;box-shadow:0 1px 3px #0001}}
.kpi b{{display:block;font-size:24px;color:#243b4a}}
figure{{margin:0;background:#fff;padding:8px;border-radius:10px;box-shadow:0 1px 3px #0001}}
figure img,figure video{{width:100%;border-radius:8px;display:block;background:#000}}
figcaption{{font-size:13px;color:#444;padding:8px 4px 2px}}
.pill{{display:inline-block;background:#e07a5f;color:#fff;font-size:12px;padding:1px 8px;border-radius:99px;margin-right:4px}}
.t0{{font-size:11px;color:#2a9d8f}}
video.wide,img.wide{{width:100%;max-width:1120px;border-radius:8px;display:block}}
.qbox{{background:#fff;padding:14px;border-radius:10px;box-shadow:0 1px 3px #0001;margin-bottom:14px}}
table{{border-collapse:collapse;background:#fff;width:100%;max-width:1120px;border-radius:8px;overflow:hidden}}
td,th{{padding:8px 14px;border-bottom:1px solid #eee;text-align:right;font-size:14px}}
td:first-child,th:first-child{{text-align:left}}
.note{{color:#666;font-size:13px}}
button.toggle{{background:#243b4a;color:#fff;border:0;border-radius:8px;padding:7px 14px;cursor:pointer;margin-bottom:10px}}
</style></head><body>
<header><h1>成长档案 V2 · 多机位试点验收（2026-11-17）</h1>
<div class="topsub">本地 Qwen3.8-27B + Whisper 中文关键词检索 + 昼夜分治瘦身 · 原片只读未删 · <a href="index_v1.html">对比查看 V1 旧版</a></div>
</header>
{alert_banner}
<section><h2>摄像头在线状态与新机位自动发现</h2>
<div class="hgrid">{health_cards}</div>
<p class="note">书房新增 Aqara G100 接入 NAS 后，目录扫描器会自动识别并按室内默认规则纳入（支持重要时刻、精选照片与昼夜压缩），无需改代码。</p>
</section>

<section><h2>一、重要成长时刻与精彩玩耍（自动保留 T0 原画）</h2>
<p class="note">本机 Whisper 检查 {asr_sum.get('windows_transcribed', 0)} 个童声/说话候选窗（未命中对话当场丢弃，绝不落盘），命中 {asr_sum.get('matched_count', 0)} 条亲属称呼候选；视觉片段经本地 Qwen3.8-27B 逐段复核并自动复制原片至 T0 珍藏库。</p>
{ms_html}</section>

<section><h2>二、1–11 个月成长变化（动态短片 + 无遮挡近景月度卡）</h2>
{m_film_html}
<h3>每月宝宝近景对照（已剔除大人背影与遮挡）</h3>
<div class="grid">{m_cards_html}</div></section>

<section><h2>三、平滑过渡成长流年（解决“闪得抓不住重点”）</h2>
{smooth_html}</section>

<section><h2>四、当天精华（流畅保留）与 V2 精选照片</h2>
<h3>2026-11-08 当天精华（流畅版）</h3>
{day_hl_html}
<h3>V2 精选照片（已过滤空床/大人背影/边缘残缺，共 {len(deliv['photos'])} 张）</h3>
<button class="toggle" onclick="togglePhotos(this)">切换为：2.7K/2K 完整原画视图</button>
<div class="grid">{ph_html}</div></section>

<section><h2>五、多视角放大适配度与 1:1 像素画质对比</h2>
<h3>1. 四个机位到底适不适合放大？</h3>
{zoom_cards}
<h3>2. 1:1 局部像素放大对比与全屏视频对照（原画 vs 1080p15 vs 720p15）</h3>
{q_cards}</section>

<section><h2>六、空间优化账本与全天延时底片</h2>
<div class="kpi">
<div>四路样本原片<b>{gb_fmt(tot_orig_v2)}</b>含 2026-11-15 厨房 4K</div>
<div>V2 优化后<b>{gb_fmt(tot_kept_v2)}</b>床头昼夜分离 + 厨房滤除</div>
<div>综合压缩比<b>{ratio_v2:.1f}×</b>节省 {100 * (1 - tot_kept_v2 / max(1, tot_orig_v2)):.1f}%</div>
<div>床头单机压缩<b>{bed_v2['v1_ratio']}× → {bed_v2['v2_ratio']}×</b>T1 {bed_v2['v1_t1_hours']}h → {bed_v2['v2_t1_hours']}h</div>
</div>
<table>
<tr><th>摄像头（试点日）</th><th>原片</th><th>V1 / 旧规则保留</th><th>V2 优化后</th><th>实际压缩倍数</th><th>关键优化</th></tr>
<tr><td>宝宝床头（11-08 24h）</td><td>{gb_fmt(bed_v2['orig_bytes'])}</td><td>{gb_fmt(bed_v2['v1_total_bytes'])} ({bed_v2['v1_t1_hours']}h T1)</td><td>{gb_fmt(bed_v2['v2_total_bytes'])} ({bed_v2['v2_t1_hours']}h T1)</td><td><b>{bed_v2['v2_ratio']}×</b></td><td>屏蔽静态被褥 {bed_v2['static_suppressed_hours']}h，夜间纯睡眠入 T2</td></tr>
<tr><td>厨房 G350 4K（11-15）</td><td>{gb_fmt(kit_v2['orig_bytes'])}</td><td>若全保留约 {gb_fmt(kit_v2['orig_bytes']*0.35)}</td><td>{gb_fmt(kit_v2['total_kept_bytes'])}（T1 {kit_v2['confirmed_child_files']} 段 + T2 {gb_fmt(kit_v2['t2_bytes'])}）</td><td><b>{kit_v2['compression_ratio']}×</b></td><td>{kit_v2['files']} 段中仅宝宝确认出镜留 T1，余转 1fps T2</td></tr>
<tr><td>宝宝房顶部（11-08）</td><td>{gb_fmt(top.get('orig',0))}</td><td>{gb_fmt(top.get('t1',0)+top.get('t2',0))}</td><td>{gb_fmt(top.get('t1',0)+top.get('t2',0))}</td><td><b>{top.get('orig',0)/max(1,top.get('t1',0)+top.get('t2',0)):.2f}×</b></td><td>建议 T1 升 1080p15 兼顾放大清晰度</td></tr>
<tr><td>客厅（11-08）</td><td>{gb_fmt(living.get('orig',0))}</td><td>{gb_fmt(living.get('t1',0)+living.get('t2',0))}</td><td>{gb_fmt(living.get('t1',0)+living.get('t2',0))}</td><td><b>{living.get('orig',0)/max(1,living.get('t1',0)+living.get('t2',0)):.2f}×</b></td><td>活动最多；若仅儿童确认入 T1 还可再省约 40%</td></tr>
</table>
<h3>全天 1fps 延时底片</h3>
<div class="grid">{lapses}</div></section>
<script>
let showFull = false;
function togglePhotos(btn) {{
  showFull = !showFull;
  document.querySelectorAll('.phimg').forEach(img => {{ img.src = showFull ? img.dataset.full : img.dataset.framed; }});
  btn.textContent = showFull ? '切换为：智能构图视图' : '切换为：2.7K/2K 完整原画视图';
}}
</script>
</body></html>"""
    idx_path.write_text(page, encoding="utf-8")
    print(f"[Page V2] Wrote {idx_path}", flush=True)
    return idx_path
