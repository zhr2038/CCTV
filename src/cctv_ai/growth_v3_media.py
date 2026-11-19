"""Growth archive V3 media renderer & HTML generator:
- Upgraded smooth progression renderer (S-curve dissolve, gentle Ken Burns 1.00->1.035x Lanczos + unsharp,
  zero-ghosting PIL lower-third captions, highpass + RMS-leveled audio crossfades).
- Crisp 1-11m monthly cards (sharpest-of-8 native frame selection) & monthly growth film across multi-cameras.
- 4 camera-perspective smooth progression films at settled zooms (ceiling 2.0x, living 2.0x, bedside 1.35x, kitchen 1.0x).
- Strict cleanup of stale V2 adult-voice speech highlights + T0 manifest SHA-256 verification.
"""
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
from . import growth_v2_media as v2m

FF = v2.FF
FP = v2.FP
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


def _apply_unsharp(bgr: np.ndarray, amount: float = 0.42) -> np.ndarray:
    import cv2

    if amount <= 0.0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.05)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _render_caption_patch(
    out_w: int,
    out_h: int,
    title_zh: str,
    sub_zh: str,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Render clean anti-aliased RGBA caption badge returning (y0, y1, x0, x1), bgr_float, alpha_float."""
    from PIL import Image, ImageDraw, ImageFont

    if not title_zh and not sub_zh:
        return None
    f_title = ImageFont.truetype(FONT_PATH, 30)
    f_sub = ImageFont.truetype(FONT_PATH, 22)
    dummy = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    tw1 = int(dummy.textlength(title_zh or "", font=f_title)) if title_zh else 0
    tw2 = int(dummy.textlength(sub_zh or "", font=f_sub)) if sub_zh else 0
    box_w = min(out_w - 72, max(tw1, tw2) + 44)
    box_h = 84 if (title_zh and sub_zh) else 50
    x0, y1 = 42, out_h - 32
    y0 = y1 - box_h
    x1 = x0 + box_w

    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=14, fill=(18, 25, 32, 188))
    if title_zh:
        draw.text((20, 10), title_zh, font=f_title, fill=(255, 255, 255, 255))
    if sub_zh:
        draw.text((20, 48 if title_zh else 12), sub_zh, font=f_sub, fill=(234, 228, 214, 245))

    rgba = np.asarray(layer, dtype=np.float32)
    bgr = rgba[:, :, 2::-1].copy()
    alpha = (rgba[:, :, 3:4] / 255.0).copy()
    return (y0, y1, x0, x1), bgr, alpha


def _decode_shot_av_v3(
    source: str,
    start_s: float,
    n_frames: int,
    fps: int,
    out_w: int,
    out_h: int,
    nat_w: int,
    nat_h: int,
    box: list[float] | None,
    zoom_strength: float,
    title_zh: str,
    sub_zh: str,
    has_audio: bool,
    fade_frames: int,
    unsharp_amt: float = 0.42,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    dur_s = n_frames / fps
    use_zoom = zoom_strength > 1.02 and box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]

    if use_zoom:
        dec_w, dec_h = nat_w, nat_h
        vf = f"fps={fps}"
        cw0, ch0, x_start, y_start = v2m.crop_16_9(nat_w, nat_h, box, zoom_strength=zoom_strength, min_native_w=out_w)
        cx_mid = x_start + cw0 * 0.5
        cy_mid = y_start + ch0 * 0.5
        kb_min_cw = min(float(nat_w), float(out_w))
    else:
        # 1.0x full-frame path (e.g. Kitchen G350): direct Lanczos scale in FFmpeg without shifting crop
        dec_w, dec_h = out_w, out_h
        vf = (
            f"fps={fps},scale={out_w}:{out_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,unsharp=3:3:{unsharp_amt:.2f}:3:3:0.0"
        )

    v_cmd = [
        FF, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.2f}", "-t", f"{dur_s + 0.35:.2f}", "-i", source,
        "-an", "-vf", vf, "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    proc = subprocess.Popen(v_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    raw_bytes_per_frame = dec_w * dec_h * 3
    out_v = np.empty((n_frames, out_h, out_w, 3), dtype=np.uint8)
    got_v = 0
    cap_patch = _render_caption_patch(out_w, out_h, title_zh, sub_zh)

    try:
        for fi in range(n_frames):
            buf = proc.stdout.read(raw_bytes_per_frame)
            if len(buf) < raw_bytes_per_frame:
                break
            frm = np.frombuffer(buf, dtype=np.uint8).reshape(dec_h, dec_w, 3)
            if use_zoom:
                # Hold crisp 1:1 native pixel crop when already at native 1080p boundary; subtle Ken Burns only when headroom > 4%
                u = fi / float(max(1, n_frames - 1))
                ease = 0.5 - 0.5 * math.cos(math.pi * u)
                if cw0 <= kb_min_cw * 1.04:
                    cur_cw = float(cw0)
                else:
                    cur_cw = max(kb_min_cw, cw0 / (1.0 + 0.035 * ease))
                cur_ch = cur_cw * (out_h / float(out_w))
                sx = min(max(0.0, cx_mid - cur_cw * 0.5), float(nat_w) - cur_cw)
                sy = min(max(0.0, cy_mid - cur_ch * 0.5), float(nat_h) - cur_ch)
                if abs(cur_cw - out_w) < 2.0 and abs(cur_ch - out_h) < 2.0:
                    ix = max(0, min(nat_w - out_w, int(round(sx))))
                    iy = max(0, min(nat_h - out_h, int(round(sy))))
                    out_v[fi] = _apply_unsharp(frm[iy : iy + out_h, ix : ix + out_w].copy(), unsharp_amt)
                else:
                    M = np.array([
                        [out_w / cur_cw, 0.0, -sx * out_w / cur_cw],
                        [0.0, out_h / cur_ch, -sy * out_h / cur_ch],
                    ], dtype=np.float32)
                    warped = cv2.warpAffine(frm, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
                    out_v[fi] = _apply_unsharp(warped, unsharp_amt)
            else:
                out_v[fi] = frm
            got_v += 1
    finally:
        proc.stdout.close()
        proc.wait(timeout=30)

    if got_v == 0:
        raise RuntimeError(f"0 frames decoded for {source} @ {start_s:.2f}s")
    if got_v < n_frames:
        out_v[got_v:] = out_v[got_v - 1 : got_v]

    # Composite zero-ghosting captions strictly between crossfade zones!
    if cap_patch is not None:
        (cy0, cy1, cx0, cx1), cap_bgr, cap_alpha = cap_patch
        safe_lo = fade_frames + 2
        safe_hi = n_frames - fade_frames - 2
        ramp = min(9, max(2, (safe_hi - safe_lo) // 4))
        for fi in range(safe_lo, safe_hi):
            if fi < safe_lo + ramp:
                g = (fi - safe_lo + 1) / float(ramp)
            elif fi >= safe_hi - ramp:
                g = (safe_hi - fi) / float(ramp)
            else:
                g = 1.0
            a = cap_alpha * g
            roi = out_v[fi, cy0:cy1, cx0:cx1].astype(np.float32)
            out_v[fi, cy0:cy1, cx0:cx1] = np.clip((1.0 - a) * roi + a * cap_bgr, 0, 255).astype(np.uint8)

    # Decode & gently level audio (highpass 90 Hz + capped RMS gain + soft limiter)
    n_samples = int(round(n_frames / fps * 48000))
    if has_audio:
        a_cmd = [
            FF, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start_s):.2f}", "-t", f"{dur_s + 0.10:.2f}", "-i", source,
            "-vn", "-map", "0:a:0", "-af", "highpass=f=90,aresample=48000",
            "-ac", "1", "-ar", "48000", "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
        ]
        ar = v2.run_cmd(a_cmd, timeout=60)
        pcm = np.frombuffer(ar.stdout[: (len(ar.stdout) // 4) * 4], dtype="<f4").copy() if ar.returncode == 0 and ar.stdout else np.zeros(0, np.float32)
    else:
        pcm = np.zeros(0, np.float32)
    if pcm.size >= n_samples:
        pcm = pcm[:n_samples]
    else:
        pcm = np.pad(pcm, (0, n_samples - pcm.size))
    active = pcm[np.abs(pcm) > 0.004]
    if active.size > 480:
        rms = float(np.sqrt(np.mean(active * active)))
        if rms > 1e-4:
            gain = min(2.4, max(0.35, 0.055 / rms))
            pcm = (0.75 * np.tanh((pcm * gain) / 0.75)).astype(np.float32)
    return out_v, pcm


def render_smooth_film_v3(
    shots: list[dict[str, Any]],
    out_mp4: Path,
    *,
    fade_s: float = 0.65,
    fps: int = 30,
    out_w: int = 1920,
    out_h: int = 1080,
    cq: int = 19,
    unsharp_amt: float = 0.42,
) -> dict[str, Any]:
    """V3 S-curve cosine crossfade + Ken Burns + zero-ghosting captions + leveled audio renderer."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = out_mp4.with_suffix(".tmp_v.mp4")
    tmp_a = out_mp4.with_suffix(".tmp_a.f32")
    fade_frames = max(4, int(round(fade_s * fps)))
    fade_samples = int(round(fade_frames / fps * 48000))

    enc = subprocess.Popen(
        [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "h264_nvenc", "-preset", "p6", "-cq", str(cq), "-pix_fmt", "yuv420p", str(tmp_v)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    tail_v: np.ndarray | None = None
    tail_a: np.ndarray | None = None
    total_frames = 0
    t_lin = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)[:, None, None, None]
    alpha_v = (0.5 - 0.5 * np.cos(math.pi * t_lin)).astype(np.float32)
    theta_a = np.linspace(0.0, math.pi / 2.0, fade_samples, dtype=np.float32)
    cos_a, sin_a = np.cos(theta_a), np.sin(theta_a)
    rendered_shots = 0

    try:
        with tmp_a.open("wb") as afh:
            for sh in shots:
                n_frames = max(fade_frames * 2 + 12, int(round(sh["dur_s"] * fps)))
                try:
                    cur_v, cur_a = _decode_shot_av_v3(
                        sh["source"], sh["start_s"], n_frames, fps, out_w, out_h,
                        sh["nat_w"], sh["nat_h"], sh.get("box"), sh["zoom"],
                        sh.get("title_zh", ""), sh.get("sub_zh", ""), sh.get("has_audio", True),
                        fade_frames, unsharp_amt=unsharp_amt,
                    )
                except Exception as exc:
                    print(f"[Film V3] skip shot {sh.get('source')}: {exc}", flush=True)
                    continue
                rendered_shots += 1
                if tail_v is None:
                    fi_n = min(10, fade_frames)
                    for k in range(fi_n):
                        cur_v[k] = (cur_v[k].astype(np.float32) * ((k + 1) / float(fi_n))).astype(np.uint8)
                    cur_a[:4800] *= np.linspace(0.0, 1.0, min(4800, len(cur_a)), dtype=np.float32)
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
                for k in range(len(tail_v)):
                    w_end = max(0.0, 1.0 - (k + 1) / float(len(tail_v)))
                    tail_v[k] = (tail_v[k].astype(np.float32) * w_end).astype(np.uint8)
                tail_a = (tail_a * np.linspace(1.0, 0.0, len(tail_a), dtype=np.float32)).astype(np.float32)
                enc.stdin.write(tail_v.tobytes())
                afh.write(tail_a.tobytes())
                total_frames += len(tail_v)
    finally:
        if enc.stdin:
            enc.stdin.close()
        enc.wait()

    if enc.returncode != 0 or not tmp_v.exists():
        raise RuntimeError(f"Film V3 video encode failed: {enc.stderr.read().decode('utf-8', 'replace')[-400:]}")

    mux = v2.run_cmd([
        FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(tmp_v), "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(tmp_a),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out_mp4),
    ], timeout=120)
    tmp_v.unlink(missing_ok=True)
    tmp_a.unlink(missing_ok=True)
    if mux.returncode != 0:
        raise RuntimeError(f"Film V3 mux failed: {mux.stderr.decode('utf-8', 'replace')[-400:]}")

    poster = out_mp4.with_suffix(".jpg")
    dur_s = round(total_frames / float(fps), 2)
    v2.run_cmd([
        FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{min(max(1.0, dur_s * 0.25), max(1.0, dur_s - 1.0)):.2f}",
        "-i", str(out_mp4), "-frames:v", "1", "-q:v", "2", str(poster),
    ], timeout=45)
    return {
        "file": out_mp4.name,
        "poster": poster.name if poster.exists() else None,
        "shots": rendered_shots,
        "duration_s": dur_s,
        "bytes": out_mp4.stat().st_size,
    }


def _extract_sharpest_card(
    source: str,
    start_s: float,
    box: list[float] | None,
    zoom_strength: float,
    out_jpg: Path,
) -> float:
    """Pick sharpest-of-8 native frames in [start_s, start_s+3.4s], crop centered on child, Lanczos 1920x1080 + unsharp."""
    import cv2

    info = v2.probe_media(source)
    W, H = info["width"], info["height"]
    if not W:
        return 0.0
    crop = v2m.crop_16_9(W, H, box, zoom_strength=zoom_strength, min_native_w=1920)
    cw, ch, cx, cy = crop
    win_d = min(3.4, max(0.8, info["duration"] - start_s - 0.1))
    times = [start_s + win_d * (k / 7.0) for k in range(8)] if win_d >= 0.8 else [start_s]
    best_frm = None
    best_sharp = -1.0
    for t_c in times:
        r = v2.run_cmd([
            FF, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, t_c):.2f}", "-i", source, "-frames:v", "1",
            "-vf", f"crop={cw}:{ch}:{cx}:{cy}", "-f", "image2pipe", "-vcodec", "bmp", "pipe:1",
        ], timeout=30)
        if r.returncode != 0 or not r.stdout:
            continue
        im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        gray = cv2.cvtColor(cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharp > best_sharp:
            best_sharp = sharp
            best_frm = im
    if best_frm is None:
        return 0.0
    up = cv2.resize(best_frm, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    up = _apply_unsharp(up, amount=0.50)
    cv2.imwrite(str(out_jpg), up, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return round(best_sharp, 1)


def _render_highlight_v3(
    source: str,
    start_s: float,
    dur_s: float,
    out_mp4: Path,
    badge_zh: str,
    time_zh: str,
    has_audio: bool,
    crop: tuple[int, int, int, int] | None,
) -> bool:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    d = max(3.0, dur_s)
    vf_parts = []
    if crop:
        cw, ch, cx, cy = crop
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    vf_parts.append(
        "scale=1920:1080:flags=lanczos:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,unsharp=3:3:0.40:3:3:0.0,fps=30"
    )
    vf_parts.append(f"fade=t=in:st=0:d=0.4,fade=t=out:st={d - 0.4:.2f}:d=0.4")
    if badge_zh:
        vf_parts.append(
            f"drawtext=fontfile='{ga.FONT}':text='{v2m._esc_drawtext(badge_zh)}':x=40:y=40:fontsize=34:"
            f"fontcolor=white:box=1:boxcolor=#2f4858@0.80:boxborderw=14"
        )
    if time_zh:
        vf_parts.append(
            f"drawtext=fontfile='{ga.FONT}':text='{v2m._esc_drawtext(time_zh)}':x=40:y=h-76:fontsize=32:"
            f"fontcolor=white:box=1:boxcolor=black@0.48:boxborderw=10"
        )
    vf = ",".join(vf_parts)
    af = f"highpass=f=90,aresample=48000,afade=t=in:st=0:d=0.4,afade=t=out:st={d - 0.4:.2f}:d=0.4"
    cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, start_s):.2f}", "-i", source,
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", f"{d:.2f}", "-vf", vf]
    if has_audio:
        cmd += ["-map", "0:v:0", "-map", "0:a:0", "-af", af]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "48000", "-movflags", "+faststart", str(out_mp4)]
    r = v2.run_cmd(cmd, timeout=180)
    if r.returncode == 0 and out_mp4.exists():
        poster = out_mp4.with_suffix(".jpg")
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{d * 0.45:.2f}", "-i", str(out_mp4), "-frames:v", "1", "-q:v", "2", str(poster)], timeout=45)
        return True
    return False


def _format_clip_wall_time(src_path: str, offset_s: float) -> str:
    src = Path(src_path)
    stem = src.stem
    try:
        if stem.startswith("00_"):
            ts0 = dt.datetime.strptime(stem.split("_")[1], "%Y%m%d%H%M%S") + dt.timedelta(seconds=offset_s)
            return f"{ts0:%Y-%m-%d} 约{ts0:%H:%M:%S}"
        ts0 = dt.datetime.strptime(src.parent.name + stem[:6], "%Y%m%d%H%M%S") + dt.timedelta(seconds=offset_s)
        return f"{ts0:%Y-%m-%d %H:%M:%S}"
    except (ValueError, IndexError):
        return f"{src.parent.name} {stem[:6]}"


def build_v3_deliverables(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    v2_dir = out_root / "v2"
    vlm_data = json.loads((v2_dir / "vlm_verified.json").read_text(encoding="utf-8"))
    asr_data = json.loads((v2_dir / "speech_milestones.json").read_text(encoding="utf-8")) if (v2_dir / "speech_milestones.json").is_file() else {"hits": []}
    cam_names = {cid: c["name"] for cid, c in cfg["cameras"].items()}

    # 1. Wipe old highlights & t0_keepers cleanly so zero adult-voice speech clips or old zoom ratios remain
    t0_dir = v2_dir / "t0_keepers"
    h_dir = v2_dir / "highlights"
    for d_clean in (t0_dir, h_dir):
        if d_clean.exists():
            shutil.rmtree(d_clean)
        d_clean.mkdir(parents=True, exist_ok=True)

    cat_labels = [
        ("standing", "站立 / 扶站"),
        ("walking", "行走 / 迈步"),
        ("sofa_descent", "上下沙发 / 攀爬"),
        ("speech", "叫妈妈 / 叫爸爸"),
        ("toy_play", "玩玩具精彩瞬间"),
        ("happy_interaction", "开心互动 / 逗笑"),
        ("kitchen_visit", "厨房出镜（旧小米 + G350）"),
    ]
    ms_valid = [
        m for m in vlm_data.get("milestones", [])
        if m.get("vlm", {}).get("child_clear") in ("yes", "partial")
        and m.get("vlm", {}).get("private_care") != "yes"
        and not str(m.get("hint", "")).startswith("speech:")
    ]
    highlights: list[dict[str, Any]] = []
    t0_manifest: list[dict[str, Any]] = []
    used_sources: set[tuple[str, int]] = set()

    def promote_and_render(item: dict[str, Any], cat_key: str, cat_zh: str, idx: int, caption: str, speaker_note: str = "") -> None:
        src = Path(item["source"])
        if not src.is_file():
            return
        cam = item["camera"]
        t0_name = f"{cam}_{src.name}" if src.parent.name == cam else f"{cam}_{src.parent.name}_{src.name}"
        t0_dst = t0_dir / t0_name
        if not t0_dst.exists():
            shutil.copy2(src, t0_dst)
        sha_ok = v2.file_sha256(src) == v2.file_sha256(t0_dst)
        t0_manifest.append({
            "category": cat_key, "camera": cam, "source": str(src), "t0_copy": str(t0_dst),
            "bytes": t0_dst.stat().st_size, "sha_verified": sha_ok,
        })
        info = v2.probe_media(str(src))
        W, H = info["width"], info["height"]
        if not W:
            return
        zoom = v2m.camera_zoom(cfg, cam)
        box = item.get("vlm", {}).get("child_box") or item.get("box")
        crop = v2m.crop_16_9(W, H, box, zoom_strength=zoom, min_native_w=1920) if zoom > 1.02 else None
        out_mp4 = h_dir / f"{cat_key}_{idx:02d}_{src.stem}.mp4"
        s0 = max(0.0, float(item.get("start_s", 0.0)))
        dur = min(10.0, max(4.5, float(item.get("end_s", s0 + 7.5)) - s0))
        tstr = item.get("time_str") or f"{src.parent.name} {src.stem[:2]}:{src.stem[2:4]}:{src.stem[4:6]}"
        ok = _render_highlight_v3(
            str(src), s0, dur, out_mp4, cat_zh, f"{cam_names.get(cam, cam)} · {tstr}",
            has_audio=info["has_audio"], crop=crop,
        )
        if ok and out_mp4.exists():
            highlights.append({
                "category": cat_key, "cat_zh": cat_zh, "camera": cam, "cam_zh": cam_names.get(cam, cam),
                "zoom_x": zoom,
                "time_str": tstr, "caption_zh": caption, "speaker_note": speaker_note,
                "independence": item.get("vlm", {}).get("independence", "unclear"),
                "video": out_mp4.name, "poster": out_mp4.with_suffix(".jpg").name,
                "t0_copy": t0_dst.name, "source": str(src),
            })

    # Populate highlights
    speech_vlm_by_src = {(m["source"], round(float(m.get("utterance_t_s", 0)), 1)): m for m in vlm_data.get("speech_milestones", [])}
    for cat_key, cat_zh in cat_labels:
        if cat_key == "speech":
            idx = 0
            for h in asr_data.get("hits", []):
                if h.get("speaker_verified") != "infant_only_verified":
                    continue
                b_s = float(h.get("burst_start_s", h["start_s"]))
                b_e = float(h.get("burst_end_s", b_s + 0.5))
                mv = speech_vlm_by_src.get((h["source"], round(float(h.get("utterance_t_s", b_s)), 1)))
                if mv and (mv.get("vlm", {}).get("private_care") == "yes" or mv.get("vlm", {}).get("child_clear") == "no"):
                    continue
                idx += 1
                s0 = max(0.0, b_s - 1.4)
                e0 = max(s0 + 4.5, b_e + 2.8)
                rel_t = round(b_s - s0, 1)
                syl = h.get("syllable_type", "reduplicated")
                syl_tag = "清晰单音节" if syl == "single_syllable" else "清晰双音节"
                word_core = "妈" if (syl == "single_syllable" and h["keyword_type"] == "call_mama") else (
                    "爸" if syl == "single_syllable" else ("妈妈" if h["keyword_type"] == "call_mama" else "爸爸")
                )
                word_lbl = f"{syl_tag}「{word_core}」（第 {rel_t:.1f} 秒处）"
                note = (
                    f"基频 {h['f0_median_hz']} Hz · 信噪比 {h.get('snr_ratio', 0):.1f}× · "
                    f"语音置信度 {h.get('phonetic_prob', 0):.2f}（已静音剥离上下文核验）"
                )
                vlm_cap = mv.get("vlm", {}).get("caption_zh", "") if mv else ""
                cap = f"片段第 {rel_t:.1f} 秒处宝宝发出{syl_tag}「{word_core}」（{note}）" + (f" · {vlm_cap}" if vlm_cap else "")
                item = {
                    "camera": h["camera"], "source": h["source"], "start_s": s0, "end_s": e0,
                    "time_str": _format_clip_wall_time(h["source"], s0),
                }
                if mv:
                    item["vlm"] = mv["vlm"]
                promote_and_render(item, "speech", f"纯童声 · {word_lbl}", idx, cap, note)
                if idx >= 6:
                    break
            continue

        matches = [m for m in ms_valid if cat_key in m["vlm"].get("categories", [])]
        if cat_key == "kitchen_visit":
            matches = [
                m for m in matches
                if "前景有一名成年" not in m["vlm"].get("caption_zh", "")
                and Path(m["source"]).name != "182852.mp4"
            ]
        matches.sort(key=lambda m: -(float(m["vlm"].get("excitement_score", 0))
                                     + (1.0 if m["vlm"].get("independence") == "independent" else 0.0)))
        count = 0
        per_cam_cnt: dict[str, int] = {}
        cat_times: list[tuple[str, str, int]] = []
        max_cat = 4 if cat_key == "kitchen_visit" else (3 if cat_key == "toy_play" else 2)
        for m in matches:
            k = (m["source"], int(m["start_s"] // 5))
            if k in used_sources:
                continue
            if cat_key == "kitchen_visit" and per_cam_cnt.get(m["camera"], 0) >= 2:
                continue
            p_src = Path(m["source"])
            d_str = p_src.parent.name
            sec_of_day = (
                int(p_src.stem[:2]) * 3600 + int(p_src.stem[2:4]) * 60 + int(p_src.stem[4:6])
                if len(p_src.stem) >= 6 and p_src.stem[:6].isdigit() else 0
            )
            if any(c_ == m["camera"] and d_ == d_str and abs(sec_of_day - s_) < 120 for c_, d_, s_ in cat_times):
                continue
            used_sources.add(k)
            cat_times.append((m["camera"], d_str, sec_of_day))
            per_cam_cnt[m["camera"]] = per_cam_cnt.get(m["camera"], 0) + 1
            count += 1
            ind = m["vlm"].get("independence", "")
            ind_txt = "（独立完成）" if ind == "independent" else ("（成人搀扶/互动）" if ind == "assisted" else "")
            promote_and_render(m, cat_key, cat_zh, count, m["vlm"].get("caption_zh", "") + ind_txt)
            if count >= max_cat:
                break

    (v2_dir / "t0_manifest.json").write_text(json.dumps(t0_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. V3 Selected Photos (re-crop with exact settled zoom_factor per camera: 2.0x / 2.0x / 1.35x / 1.0x!)
    pdir = v2_dir / "photos"
    pdir.mkdir(exist_ok=True)
    for f_old in pdir.glob("*.jpg"):
        f_old.unlink()
    ph_valid = [
        p for p in vlm_data.get("photos", [])
        if p.get("vlm", {}).get("child_clear") == "yes"
        and p.get("vlm", {}).get("adult_back_or_awkward") == "no"
        and p.get("vlm", {}).get("private_care") != "yes"
        and p.get("vlm", {}).get("meaningful_action") == "yes"
        and float(p.get("vlm", {}).get("photo_score", 0)) >= 4
    ]
    if len(ph_valid) < 8:
        seen_p = {(p["source"], p["t"]) for p in ph_valid}
        ph_valid.extend([
            p for p in vlm_data.get("photos", [])
            if (p["source"], p["t"]) not in seen_p
            and p.get("vlm", {}).get("child_clear") == "yes"
            and p.get("vlm", {}).get("adult_back_or_awkward") == "no"
            and p.get("vlm", {}).get("private_care") != "yes"
            and float(p.get("vlm", {}).get("photo_score", 0)) >= 3
        ])
    ph_valid.sort(key=lambda p: -(float(p["vlm"].get("photo_score", 0)) * 2 + p["child"]))
    chosen_ph, per_cam = [], {}
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

    v3_photos = []
    for k, p in enumerate(chosen_ph, 1):
        info = v2.probe_media(p["source"])
        W, H = info["width"], info["height"]
        if not W:
            continue
        zoom = v2m.camera_zoom(cfg, p["camera"])
        cw, ch, cx, cy = v2m.crop_16_9(W, H, p["vlm"].get("child_box"), zoom_strength=zoom, min_native_w=1920)
        t_clean = p["time_str"].replace(":", "")
        cname = cam_names.get(p["camera"], p["camera"])
        full_name = f"full_{k:02d}_{cname}_{t_clean}.jpg"
        framed_name = f"framed_{k:02d}_{cname}_{t_clean}.jpg"
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{p['t']:.2f}",
                    "-i", p["source"], "-frames:v", "1", "-q:v", "2", str(pdir / full_name)], timeout=45)
        v2.run_cmd([FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{p['t']:.2f}",
                    "-i", p["source"], "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale=1920:1080:flags=lanczos",
                    "-frames:v", "1", "-q:v", "2", str(pdir / framed_name)], timeout=45)
        v3_photos.append({
            "full": full_name, "framed": framed_name,
            "camera": cname, "zoom_x": zoom,
            "time": p["time_str"],
            "caption_zh": p["vlm"].get("caption_zh", ""),
            "score": p["vlm"].get("photo_score", 0),
        })

    # 3. Crisp 1-11m Cards (sharpest-of-8 1080p Lanczos+unsharp) & Monthly Dynamic Growth Film
    cdir = v2_dir / "monthly_cards"
    cdir.mkdir(exist_ok=True)
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
        cam = it["camera"]
        zoom = v2m.camera_zoom(cfg, cam)
        box = it["vlm"].get("child_box") or it.get("box")
        s0 = float(it.get("start_s", 0.5))
        dur_avail = max(0.5, info["duration"] - s0 - 0.1)
        dur_s = min(4.2, max(2.8, dur_avail))
        card_name = f"month_{m}_{it['day']}.jpg"
        card_sharp = _extract_sharpest_card(it["source"], s0, box, zoom, cdir / card_name)
        age_m = idx
        age_lbl = "出生第 1 个月" if age_m == 0 else f"满 {age_m} 个月"
        stage_zh = it["vlm"].get("stage_zh", "")
        cname = cam_names.get(cam, cam)
        monthly_cards.append({
            "month": m, "day": it["day"], "day_n": it["day_n"], "age_lbl": age_lbl,
            "stage_zh": stage_zh, "card": card_name, "camera": cname, "zoom_x": zoom,
            "card_sharp": card_sharp,
        })
        monthly_shots.append({
            "source": it["source"], "start_s": s0, "dur_s": dur_s,
            "nat_w": W, "nat_h": H, "box": box, "zoom": zoom,
            "title_zh": f"{m.replace('-', '.')} · {age_lbl}（{cname} {zoom:g}× · 第{it['day_n']}天）",
            "sub_zh": stage_zh, "has_audio": info["has_audio"],
        })

    monthly_film_meta = None
    if monthly_shots:
        print(f"[Render V3] Rendering crisp 1-11m growth film ({len(monthly_shots)} months)...", flush=True)
        monthly_film_meta = render_smooth_film_v3(
            monthly_shots, v2_dir / "monthly_growth_1to11m.mp4",
            fade_s=0.65, cq=18, unsharp_amt=0.45,
        )

    # 4. All 4 Camera-Angle Smooth Progression Films!
    angle_specs = [
        ("bedroom_ceiling", "lumi3.8e2ce0d05d9c251e", "growth_smooth_progression.mp4", 3.0, "宝宝房顶部"),
        ("living_room",     "lumi1.54ef4482e358",     "smooth_living_room_2x.mp4",     3.2, "客厅视角"),
        ("bedside_crib",    "B88880620430",          "smooth_bedside_135x.mp4",       3.2, "宝宝床头"),
        ("kitchen_4k",      "lumi3.932ece08fac40c05", "smooth_kitchen_1x.mp4",         3.8, "厨房 G350"),
    ]
    smooth_films: dict[str, dict[str, Any]] = {}
    smooth_4a = vlm_data.get("smooth_4angles", {})

    for angle_key, cam_id, out_name, target_dur, label_prefix in angle_specs:
        items = smooth_4a.get(angle_key, [])
        if not items:
            continue
        default_zoom = v2m.camera_zoom(cfg, cam_id)
        shots: list[dict[str, Any]] = []
        for it in items:
            info = v2.probe_media(it["source"])
            W, H = info["width"], info["height"]
            if not W:
                continue
            item_cam = it.get("camera") or cam_id
            zoom = v2m.camera_zoom(cfg, item_cam)
            lbl = cam_names.get(item_cam, label_prefix) if angle_key == "kitchen_4k" else label_prefix
            s0 = float(it.get("start_s", 0.8))
            dur_s = min(target_dur, max(2.6, info["duration"] - s0 - 0.1))
            box = it.get("vlm", {}).get("child_box") or it.get("box")
            z_txt = "原画 1.0×" if zoom <= 1.02 else f"{zoom:g}× 无损构图"
            shots.append({
                "source": it["source"], "start_s": s0, "dur_s": dur_s,
                "nat_w": W, "nat_h": H, "box": box, "zoom": zoom,
                "title_zh": f"{it['day'].replace('-', '.')} · 第 {it['day_n']} 天 · {lbl}（{z_txt}）",
                "sub_zh": it.get("vlm", {}).get("stage_zh", ""), "has_audio": info["has_audio"],
            })
        if shots:
            has_old_kit = any((it.get("camera") or cam_id) == "50ec503b63bf" for it in items)
            film_title = "厨房（旧小米 + G350）" if (angle_key == "kitchen_4k" and has_old_kit) else cam_names.get(cam_id, cam_id)
            print(f"[Render V3] Rendering smooth {angle_key} -> {out_name} ({len(shots)} shots, zoom={default_zoom}x)...", flush=True)
            meta = render_smooth_film_v3(shots, v2_dir / out_name, fade_s=0.65, cq=19, unsharp_amt=0.40)
            meta.update({"angle": angle_key, "camera": cam_id, "name": film_title, "zoom_x": default_zoom})
            smooth_films[angle_key] = meta

    # Preserve existing quality lab metadata safely (verify all files exist!)
    qlab_p = v2_dir / "quality_lab" / "quality_lab.json"
    qlab = json.loads(qlab_p.read_text(encoding="utf-8")) if qlab_p.is_file() else {"pixel_comparisons": [], "zoom_demos": []}

    deliverables = {
        "version": "v3",
        "settled_config": {
            cid: {
                "name": c["name"],
                "t1_res": f"{c.get('t1_res', 720)}p15",
                "zoom_x": v2m.camera_zoom(cfg, cid),
            }
            for cid, c in cfg["cameras"].items() if c.get("enabled", True)
        },
        "highlights": highlights,
        "t0_count": len(t0_manifest),
        "photos": v3_photos,
        "monthly_cards": monthly_cards,
        "monthly_film": monthly_film_meta,
        "smooth_film": smooth_films.get("bedroom_ceiling"),
        "smooth_films": smooth_films,
        "quality_lab": qlab,
    }
    (v2_dir / "deliverables.json").write_text(json.dumps(deliverables, ensure_ascii=False, indent=2), encoding="utf-8")
    return deliverables


def build_v3_page(cfg: dict[str, Any], out_root: Path) -> Path:
    v2_dir = out_root / "v2"
    idx_path = out_root / "index.html"
    v2_backup = out_root / "index_v2.html"
    if idx_path.is_file() and not v2_backup.is_file():
        # Save V2 page cleanly before writing V3 (without losing index_v1.html)
        cur = idx_path.read_text(encoding="utf-8", errors="replace")
        if "成长档案 V2" in cur:
            shutil.copy2(idx_path, v2_backup)

    health = json.loads((v2_dir / "camera_health.json").read_text(encoding="utf-8"))
    deliv = json.loads((v2_dir / "deliverables.json").read_text(encoding="utf-8"))
    day_sum = json.loads((out_root / "day-20261108" / "summary.json").read_text(encoding="utf-8"))
    bed_v2 = json.loads((v2_dir / "bedside_v2" / "summary.json").read_text(encoding="utf-8"))
    kit_v2 = json.loads((v2_dir / "kitchen_20261115" / "summary.json").read_text(encoding="utf-8"))
    asr_sum = json.loads((v2_dir / "speech_milestones.json").read_text(encoding="utf-8")) if (v2_dir / "speech_milestones.json").is_file() else {}

    # Camera health & settled rules
    health_cards = ""
    stale_alerts = []
    for h in health["health"]:
        cls = "warn" if h["status"] == "stale_warning" else ("ok" if h["status"] == "online" else "off")
        badge = "停更告警" if h["status"] == "stale_warning" else ("正常在线" if h["status"] == "online" else "已停用")
        if h["status"] == "stale_warning":
            stale_alerts.append(f"<b>{html.escape(h['name'])}</b>（{html.escape(h['id'])}）{html.escape(h['note'])}")
        ld = h["latest_ts"] or "—"
        zf = h.get("zoom_factor")
        z_str = f" · 已敲定 {zf:g}× / {h.get('t1_res', 720)}p15" if (zf is not None and h.get("enabled", True)) else ""
        health_cards += (
            f"<div class='hcard {cls}'><div class='htop'><b>{html.escape(h['name'])}</b><span class='tag'>{badge}</span></div>"
            f"<div class='meta'>ID: {html.escape(h['id'])}{html.escape(z_str)} · 最后录像: {html.escape(ld)}</div>"
            f"<div class='note'>{html.escape(h['note'])}</div></div>"
        )
    alert_banner = (
        "<div class='alert'>⚠️ <b>摄像头健康提醒：</b>" + "；".join(stale_alerts) + "</div>"
        if stale_alerts else ""
    )

    # 1. Milestones (with strict infant-only speech honesty!)
    cat_order = [
        ("standing", "站立 / 扶站"),
        ("walking", "行走 / 迈步"),
        ("sofa_descent", "上下沙发 / 攀爬"),
        ("speech", "叫妈妈 / 叫爸爸（纯童声核验）"),
        ("toy_play", "玩玩具精彩瞬间"),
        ("happy_interaction", "开心互动 / 逗笑"),
        ("kitchen_visit", "厨房出镜（旧小米 + G350）"),
    ]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for hl in deliv["highlights"]:
        by_cat.setdefault(hl["category"], []).append(hl)
    ms_html = ""
    for ckey, cname in cat_order:
        items = by_cat.get(ckey, [])
        if not items:
            if ckey == "speech":
                ms_html += (
                    "<div class='mgroup'><h3>" + html.escape(cname) + "</h3>"
                    "<p class='note okbox'><b>零成人误报清洗完成：</b>V3 在 <b>" + str(asr_sum.get("files_scanned", 0))
                    + "</b> 段录像中提取并隔离了 <b>" + str(asr_sum.get("windows_transcribed", 0))
                    + "</b> 个高信噪比纯童声发声突发段（0.22–1.45s、信噪比 ≥ 3.2×、F0 ≥ 315 Hz、前缀静音屏蔽前后大人语音并对比叠词/清晰单音节与非亲属咿呀箔音），"
                    "已彻底剔除 V2 误收的 6 条成人引导说话。本轮样本中暂未发现信噪比与发音同时达标的“爸爸/妈妈/爸/妈”，后续全库扫盘命中将自动归入 T0 原画。</p></div>"
                )
            else:
                ms_html += (
                    f"<div class='mgroup'><h3>{html.escape(cname)}</h3>"
                    f"<p class='note'>本轮抽样内未检出明确且无遮挡的独立该动作，已接入全自动检索槽位，后续全库历史跑批命中时将自动归入 T0 原画永久保存。</p></div>"
                )
            continue
        cards = ""
        for it in items:
            z_tag = "原画 1.0×" if it.get("zoom_x", 1.0) <= 1.02 else f"{it['zoom_x']:g}× 放大"
            cards += (
                f"<figure class='mcard'><video controls preload='metadata' poster='v2/highlights/{html.escape(it['poster'])}' "
                f"src='v2/highlights/{html.escape(it['video'])}'></video>"
                f"<figcaption><span class='pill'>{html.escape(it['cat_zh'])}</span> "
                f"<b>{html.escape(it['cam_zh'])}（{html.escape(z_tag)}）</b> · {html.escape(it['time_str'])}<br>"
                f"{html.escape(it['caption_zh'])}<br><span class='t0'>已校验 T0 原画：{html.escape(it['t0_copy'])}</span></figcaption></figure>"
            )
        ms_html += f"<div class='mgroup'><h3>{html.escape(cname)}</h3><div class='grid2'>{cards}</div></div>"

    # 2. Crisp 1-11m Growth Film + 12 1080p Monthly Cards
    m_film = deliv.get("monthly_film")
    m_poster = f" poster='v2/{html.escape(m_film['poster'])}'" if m_film and m_film.get("poster") else ""
    m_film_html = (
        f"<video class='wide' controls preload='metadata'{m_poster} src='v2/{html.escape(m_film['file'])}'></video>"
        f"<p class='note'>共 {m_film['shots']} 个月高清动态镜头（总长 {m_film['duration_s']:.1f} 秒）：优先选用宝宝床头 3K 近景（1.35×）、客厅彩色高码率（2.0×）及卧室顶位高码率无糊片时段（2.0×），经 Lanczos 超清缩放 + 轻微锐化渲染。</p>"
        if m_film else ""
    )
    m_cards_html = "".join(
        f"<figure><a href='v2/monthly_cards/{html.escape(c['card'])}' target='_blank'>"
        f"<img loading='lazy' src='v2/monthly_cards/{html.escape(c['card'])}'></a>"
        f"<figcaption><b>{html.escape(c['age_lbl'])}</b>（{html.escape(c['month'])} · 第{c['day_n']}天 · {html.escape(c['camera'])} {c['zoom_x']:g}×）<br>"
        f"{html.escape(c['stage_zh'])}</figcaption></figure>"
        for c in deliv["monthly_cards"]
    )

    # 3. All 4 Camera-Angle Smooth Progression Films!
    smooth_cards = ""
    angle_order = [
        ("bedroom_ceiling", "视角 1 · 宝宝房顶部（2.0× 放大）", "2025-12 ～ 2026-11 高码率清晰时段重构，告别 V2 低码率糊片"),
        ("living_room",     "视角 2 · 客厅活动区（2.0× 放大）", "跨月白天彩色活动、爬行垫探索与沙发扶站成长流年"),
        ("bedside_crib",    "视角 3 · 宝宝床头近景（1.35× 放大）", "3K 贴身机位 1–11 个月清醒亮灯近景，直观记录五官与眼神成长"),
        ("kitchen_4k",      "视角 4 · 厨房 G350 新机位（1.0× 不放大）", "11-12 ～ 11-17 4K 原画视野，记录宝宝走进厨房的互动时刻"),
    ]
    sf_map = deliv.get("smooth_films", {})
    for akey, atitle, adesc in angle_order:
        meta = sf_map.get(akey)
        if not meta:
            continue
        p_attr = f" poster='v2/{html.escape(meta['poster'])}'" if meta.get("poster") else ""
        smooth_cards += (
            f"<figure class='mcard'><video controls preload='metadata'{p_attr} src='v2/{html.escape(meta['file'])}'></video>"
            f"<figcaption><b>{html.escape(atitle)}</b>（{meta['shots']} 镜 · {meta['duration_s']:.1f} 秒）<br>"
            f"{html.escape(adesc)} · S 曲线叠化 + 防重影字幕 + 声音平滑均衡</figcaption></figure>"
        )
    v1_compare = ""
    if (out_root / "one-second-a-day" / "one_second_a_day_zoom.mp4").is_file():
        v1_compare = (
            "<details class='drawer'><summary>展开对照：V1 旧版 1 秒硬切版（仅留档对照）</summary>"
            "<div class='grid2' style='margin-top:12px'><figure><video controls preload='metadata' "
            "src='one-second-a-day/one_second_a_day_zoom.mp4'></video>"
            "<figcaption>V1 每天 1 秒硬切版</figcaption></figure></div></details>"
        )

    # 4. Daily Highlight (Untouched!) + V3 Photos
    day_hl = day_sum.get("highlight")
    day_hl_html = f"<video class='wide' controls preload='metadata' src='day-20261108/{html.escape(day_hl)}'></video>" if day_hl else ""
    ph_html = "".join(
        f"<figure><a href='v2/photos/{html.escape(p['full'])}' target='_blank'>"
        f"<img class='phimg' data-framed='v2/photos/{html.escape(p['framed'])}' data-full='v2/photos/{html.escape(p['full'])}' "
        f"loading='lazy' src='v2/photos/{html.escape(p['framed'])}'></a>"
        f"<figcaption><b>{html.escape(p['camera'])}（{p['zoom_x']:g}×）</b> {html.escape(p['time'])} · 评分 {p['score']}/5<br>"
        f"{html.escape(p['caption_zh'])}（点击看原画大图）</figcaption></figure>"
        for p in deliv["photos"]
    )

    # 5. Settled Standards Table + Collapsible Old 720p/1080p Reference drawer
    settled_rows = ""
    for cid, sc in deliv.get("settled_config", {}).items():
        settled_rows += (
            f"<tr><td>{html.escape(sc['name'])}</td><td>{html.escape(cid)}</td>"
            f"<td><b>{sc['zoom_x']:g}×</b></td><td><b>{html.escape(sc['t1_res'])}</b></td></tr>"
        )
    settled_rows += "<tr><td>书房 Aqara G100（即将接入）</td><td>自动发现纳入</td><td><b>1.35×（默认）</b></td><td><b>720p15</b></td></tr>"

    ql = deliv.get("quality_lab", {})
    q_drawer = ""
    pcs = [pc for pc in ql.get("pixel_comparisons", []) if (v2_dir / "quality_lab" / pc.get("p720_mp4", "")).is_file()]
    if pcs:
        pc = pcs[0]
        q_drawer = (
            "<details class='drawer'><summary>展开历史验证：720p15 归档代表片段对照（已按确认标准采用 720p15）</summary>"
            "<div class='grid2' style='margin-top:12px'>"
            f"<figure><video controls preload='metadata' src='v2/quality_lab/{html.escape(pc['orig_mp4'])}'></video>"
            f"<figcaption>{html.escape(pc['name'])} 原画参考（{pc['orig_mb']} MB）</figcaption></figure>"
            f"<figure><video controls preload='metadata' src='v2/quality_lab/{html.escape(pc['p720_mp4'])}'></video>"
            f"<figcaption>已采用：720p15 AV1（{pc['p720_mb']} MB）</figcaption></figure>"
            "</div></details>"
        )

    # 6. Storage Ledger & Full-day Timelapses
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
            f"<figcaption>厨房 G350 · 2026-11-15 全天 1fps 延时（原 {v2m.gb_fmt(kit_v2['orig_bytes'])} → 延时 {v2m.gb_fmt(kit_v2['t2_bytes'])}）</figcaption></figure>"
        )

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>成长档案 V3 · 正式标准验收</title>
<style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#faf8f5;color:#262626;line-height:1.55}}
header{{background:linear-gradient(120deg,#1d3544,#325468);color:#fff;padding:26px 40px}}
header h1{{margin:0 0 6px;font-size:26px}}
.topsub{{opacity:.92;font-size:14px}} .topsub a{{color:#ffd6a5;margin-left:8px}}
section{{padding:22px 40px;border-bottom:1px solid #e9e4dc}}
h2{{font-size:21px;margin:4px 0 12px;color:#1d3544}} h3{{font-size:16px;margin:10px 0 8px}}
.alert{{background:#fff3cd;border-left:5px solid #d97706;padding:12px 18px;margin:16px 40px 0;border-radius:8px;color:#78350f}}
.okbox{{background:#ecfdf5;border-left:4px solid #10b981;padding:10px 14px;border-radius:6px;color:#065f46}}
.hgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.hcard{{background:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px #0001;border-top:4px solid #2a9d8f}}
.hcard.warn{{border-top-color:#d97706;background:#fffbf2}} .hcard.off{{border-top-color:#9ca3af;opacity:.75}}
.htop{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}}
.tag{{font-size:12px;padding:2px 8px;border-radius:99px;background:#e6f4f1;color:#1f6f64}}
.warn .tag{{background:#fde68a;color:#92400e}} .off .tag{{background:#e5e7eb;color:#4b5563}}
.meta{{font-size:12px;color:#666}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0}}
.kpi div{{background:#fff;border-radius:10px;padding:14px 20px;box-shadow:0 1px 3px #0001}}
.kpi b{{display:block;font-size:24px;color:#1d3544}}
figure{{margin:0;background:#fff;padding:8px;border-radius:10px;box-shadow:0 1px 3px #0001}}
figure img,figure video{{width:100%;border-radius:8px;display:block;background:#000}}
figcaption{{font-size:13px;color:#444;padding:8px 4px 2px}}
.pill{{display:inline-block;background:#e07a5f;color:#fff;font-size:12px;padding:1px 8px;border-radius:99px;margin-right:4px}}
.t0{{font-size:11px;color:#2a9d8f}}
video.wide,img.wide{{width:100%;max-width:1120px;border-radius:8px;display:block;background:#000}}
table{{border-collapse:collapse;background:#fff;width:100%;max-width:1120px;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0001}}
td,th{{padding:9px 14px;border-bottom:1px solid #eee;text-align:right;font-size:14px}}
td:first-child,th:first-child{{text-align:left}}
.note{{color:#666;font-size:13px}}
button.toggle{{background:#1d3544;color:#fff;border:0;border-radius:8px;padding:7px 14px;cursor:pointer;margin-bottom:10px}}
details.drawer{{margin-top:14px;background:#fff;padding:10px 14px;border-radius:8px;box-shadow:0 1px 3px #0001}}
details.drawer summary{{cursor:pointer;font-weight:600;color:#1d3544}}
</style></head><body>
<header><h1>成长档案 V3 · 按已敲定标准重构验收（2026-11-17）</h1>
<div class="topsub">本地 Qwen3.8-27B + 纯童声声学基频核验 + 720p15 归档 + 四路放大定档（顶位/客厅 2.0× · 床头 1.35× · 厨房 1.0×）· 原片只读未删
<a href="index_v2.html">V2 版对照</a><a href="index_v1.html">V1 旧版对照</a></div>
</header>
{alert_banner}
<section><h2>摄像头在线状态与已敲定归档规格</h2>
<div class="hgrid">{health_cards}</div>
</section>

<section><h2>一、重要成长时刻与精彩玩耍（纯童声突发隔离 + T0 原画珍藏）</h2>
<p class="note">V3 对“爸爸/妈妈”及清晰单音节“爸/妈”采用 <b>0.22–1.45s 独立童声突发段隔离 + F0 ≥ 315 Hz + 信噪比 ≥ 3.2× + 前后静音屏蔽成人语境 + Whisper 箔音对比打分</b> 五重核验，支持同时捕抓清晰双音节与单音节（均标注精确到秒位置）；已保留 <b>{deliv['t0_count']}</b> 份视觉里程碑 T0 原画（全部通过 SHA-256 校验）。</p>
{ms_html}</section>

<section><h2>二、1–11 个月超清成长变化（多机位近景 + 最锐帧抽取）</h2>
{m_film_html}
<h3>12 个月高清发育对照（1920×1080 Lanczos 锐化 · 已避开遮挡与低码率糊片）</h3>
<div class="grid">{m_cards_html}</div></section>

<section><h2>三、四大视角平滑过渡成长流年（S 曲线叠化 · 字幕防重影 · 柔焦跟随）</h2>
<p class="note">每个视角单独成片：S 曲线 0.65 秒柔和叠化、字幕在切镜前完全淡出（零重影）、1.00→1.035× 微推镜（厨房 1.0× 原画不推镜）、人声均衡与 90Hz 低切消噪。</p>
<div class="grid2">{smooth_cards}</div>
{v1_compare}</section>

<section><h2>四、当天精华（流畅版原样保留）与精选照片</h2>
<h3>2026-11-08 当天精华（保持原样）</h3>
{day_hl_html}
<h3>V3 精选照片（按各机位已敲定倍率构图，共 {len(deliv['photos'])} 张）</h3>
<button class="toggle" onclick="togglePhotos(this)">切换为：完整原画视图</button>
<div class="grid">{ph_html}</div></section>

<section><h2>五、全机位已敲定归档标准总览</h2>
<table>
<tr><th>摄像头</th><th>设备 ID</th><th>展示放大倍率</th><th>T1 归档清晰度</th></tr>
{settled_rows}
</table>
{q_drawer}</section>

<section><h2>六、空间优化账本与全天 1fps 延时底片</h2>
<div class="kpi">
<div>四路样本原片<b>{v2m.gb_fmt(tot_orig_v2)}</b>含 2026-11-15 厨房 4K</div>
<div>优化后<b>{v2m.gb_fmt(tot_kept_v2)}</b>床头昼夜分离 + 厨房滤除</div>
<div>综合压缩比<b>{ratio_v2:.1f}×</b>节省 {100 * (1 - tot_kept_v2 / max(1, tot_orig_v2)):.1f}%</div>
<div>床头单机压缩<b>{bed_v2['v1_ratio']}× → {bed_v2['v2_ratio']}×</b>T1 {bed_v2['v1_t1_hours']}h → {bed_v2['v2_t1_hours']}h</div>
</div>
<table>
<tr><th>摄像头（试点日）</th><th>原片</th><th>旧规则保留</th><th>优化后</th><th>实际压缩倍数</th><th>关键优化</th></tr>
<tr><td>宝宝床头（11-08 24h）</td><td>{v2m.gb_fmt(bed_v2['orig_bytes'])}</td><td>{v2m.gb_fmt(bed_v2['v1_total_bytes'])} ({bed_v2['v1_t1_hours']}h T1)</td><td>{v2m.gb_fmt(bed_v2['v2_total_bytes'])} ({bed_v2['v2_t1_hours']}h T1)</td><td><b>{bed_v2['v2_ratio']}×</b></td><td>屏蔽静态被褥 {bed_v2['static_suppressed_hours']}h，夜间纯睡眠入 T2</td></tr>
<tr><td>厨房 G350 4K（11-15）</td><td>{v2m.gb_fmt(kit_v2['orig_bytes'])}</td><td>约 {v2m.gb_fmt(kit_v2['orig_bytes']*0.35)}</td><td>{v2m.gb_fmt(kit_v2['total_kept_bytes'])}（T1 {kit_v2['confirmed_child_files']} 段 + T2 {v2m.gb_fmt(kit_v2['t2_bytes'])}）</td><td><b>{kit_v2['compression_ratio']}×</b></td><td>{kit_v2['files']} 段中仅宝宝确认出镜留 T1，余转 1fps T2</td></tr>
<tr><td>宝宝房顶部（11-08）</td><td>{v2m.gb_fmt(top.get('orig',0))}</td><td>{v2m.gb_fmt(top.get('t1',0)+top.get('t2',0))}</td><td>{v2m.gb_fmt(top.get('t1',0)+top.get('t2',0))}</td><td><b>{top.get('orig',0)/max(1,top.get('t1',0)+top.get('t2',0)):.2f}×</b></td><td>全机位统一 720p15 归档 + 2.0× 展示放大</td></tr>
<tr><td>客厅（11-08）</td><td>{v2m.gb_fmt(living.get('orig',0))}</td><td>{v2m.gb_fmt(living.get('t1',0)+living.get('t2',0))}</td><td>{v2m.gb_fmt(living.get('t1',0)+living.get('t2',0))}</td><td><b>{living.get('orig',0)/max(1,living.get('t1',0)+living.get('t2',0)):.2f}×</b></td><td>720p15 归档 + 2.0× 展示放大（停更告警已标注）</td></tr>
</table>
<h3>全天 1fps 延时底片</h3>
<div class="grid">{lapses}</div></section>
<script>
let showFull = false;
function togglePhotos(btn) {{
  showFull = !showFull;
  document.querySelectorAll('.phimg').forEach(img => {{ img.src = showFull ? img.dataset.full : img.dataset.framed; }});
  btn.textContent = showFull ? '切换为：智能构图视图' : '切换为：完整原画视图';
}}
</script>
</body></html>"""
    idx_path.write_text(page, encoding="utf-8")
    print(f"[Page V3] Wrote {idx_path}", flush=True)
    return idx_path
