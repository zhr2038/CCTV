"""Hardware VAAPI / software x264 media helpers, T0 reflink promotion, daily highlights, and V3 smooth films."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@contextmanager
def _staging_path(destination: Path):
    """Keep partial files off the public path and rename on the same filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid.uuid4().hex}{destination.suffix}")
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def open_fast_clip_window(
    ff_bin: str,
    source_path: str,
    start_s: float,
    span_s: float,
    *,
    include_audio: bool = True,
    threshold_s: float = 20.0,
) -> Iterator[tuple[str, float]]:
    """Slice a short Matroska window via stream-copy output seek so deep offsets in fMP4 bundles never decode from 0s."""
    s0 = max(0.0, float(start_s))
    if s0 < threshold_s or not os.path.isfile(source_path):
        yield source_path, s0
        return
    with tempfile.TemporaryDirectory(prefix="growth-slice-") as td:
        slice_mkv = Path(td) / "slice.mkv"
        lead_s = max(0.0, s0 - 5.0)
        clip_span = (s0 - lead_s) + max(1.5, float(span_s)) + 2.0
        cmd = [
            ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", source_path, "-ss", f"{lead_s:.2f}", "-t", f"{clip_span:.2f}",
            "-map", "0:v:0", *(["-map", "0:a:0?"] if include_audio else ["-an"]),
            "-c", "copy", str(slice_mkv),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            r = None
        if r is not None and r.returncode == 0 and slice_mkv.is_file() and slice_mkv.stat().st_size > 1024:
            yield str(slice_mkv), round(s0 - lead_s, 3)
        else:
            yield source_path, s0


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def _parse_fps(rate_str: str | None, default: float = 20.0) -> float:
    if not rate_str or "/" not in str(rate_str):
        return default
    try:
        n_s, d_s = str(rate_str).split("/", 1)
        d = float(d_s)
        return float(n_s) / d if d > 0 else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def probe_media(ffprobe_bin: str, path: str) -> dict[str, Any]:
    r = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, timeout=30,
    )
    info = json.loads(r.stdout or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    raw_dur = float(info.get("format", {}).get("duration") or 0.0)
    v_nb = _safe_int(v.get("nb_frames") if v else 0, 0)
    a_nb = _safe_int(a.get("nb_frames") if a else 0, 0)
    r_fps = _parse_fps(v.get("r_frame_rate") if v else None, 20.0)

    pts_jump = False
    eff_dur = raw_dur
    if v_nb > 0:
        est_dur = v_nb / max(10.0, min(60.0, r_fps or 20.0))
        if raw_dur > max(30.0, est_dur * 4.0):
            pts_jump = True
            eff_dur = round(max(1.0, est_dur), 2)
    has_audio = (a is not None) and (not pts_jump) and not (0 < a_nb <= 3 and raw_dur > 15.0)
    return {
        "duration": eff_dur,
        "raw_duration": raw_dur,
        "pts_jump": pts_jump,
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
        "vcodec": v.get("codec_name") if v else None,
        "has_audio": has_audio,
    }


def check_vaapi(ff_bin: str, device: str = "/dev/dri/renderD128") -> bool:
    if not os.path.exists(device):
        return False
    cmd = [
        ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-init_hw_device", f"vaapi=va:{device}", "-f", "lavfi", "-i", "testsrc=size=128x72:rate=15",
        "-t", "0.3", "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "28", "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def resolve_font(candidates: list[str]) -> str | None:
    for c in candidates:
        if Path(c).is_file():
            return c
    return None


def crop_16_9(
    W: int,
    H: int,
    box: list[float] | None,
    zoom_strength: float = 1.4,
    min_native_w: int = 0,
) -> tuple[int, int, int, int]:
    """Compute a 16:9 crop matching `zoom_strength` centered on child while enforcing a 1:1 native pixel floor."""
    if not box or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1] or zoom_strength <= 1.02:
        cw = even(min(W, H * 16.0 / 9.0))
        ch = even(cw * 9.0 / 16.0)
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
    cw_i = min(W - (W % 2), even(cw))
    ch_i = min(H - (H % 2), even(ch))
    x = int(round(min(max(0.0, cx - cw_i * 0.5), W - cw_i)))
    y = int(round(min(max(0.0, cy - ch_i * 0.5), H - ch_i)))
    return cw_i, ch_i, max(0, x), max(0, y)


def _apply_unsharp(bgr: np.ndarray, amount: float = 0.30) -> np.ndarray:
    if amount <= 0.0:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def estimate_pelvic_privacy_box(child_box: list[float] | None) -> list[float]:
    """Estimate lower-torso/diaper sub-box inside a child bounding box when upstream safety filter triggers."""
    if not child_box or len(child_box) != 4 or child_box[2] <= child_box[0] or child_box[3] <= child_box[1]:
        return [0.32, 0.38, 0.68, 0.78]
    x1, y1, x2, y2 = [float(v) for v in child_box]
    bw, bh = x2 - x1, y2 - y1
    if bh >= bw * 1.15:
        return [round(x1 + 0.18 * bw, 4), round(y1 + 0.45 * bh, 4), round(x2 - 0.18 * bw, 4), round(min(1.0, y1 + 0.84 * bh), 4)]
    return [round(x1 + 0.20 * bw, 4), round(y1 + 0.26 * bh, 4), round(x2 - 0.20 * bw, 4), round(max(y1 + 0.45 * bh, y2 - 0.12 * bh), 4)]


def apply_privacy_mosaic(frm: np.ndarray, pbox: list[float] | None, pad_ratio: float = 0.14) -> np.ndarray:
    """Apply an elliptical feathered soft mosaic strictly over exposed private anatomy while keeping face/scene clear."""
    if not pbox or len(pbox) != 4 or pbox[2] <= pbox[0] or pbox[3] <= pbox[1]:
        return frm
    out = frm if frm.flags.writeable else frm.copy()
    H, W = out.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in pbox]
    pad_x = max(0.008, (x2 - x1) * pad_ratio)
    pad_y = max(0.008, (y2 - y1) * pad_ratio)
    xa = max(0, min(W - 1, int(math.floor((x1 - pad_x) * W))))
    ya = max(0, min(H - 1, int(math.floor((y1 - pad_y) * H))))
    xb = max(xa + 1, min(W, int(math.ceil((x2 + pad_x) * W))))
    yb = max(ya + 1, min(H, int(math.ceil((y2 + pad_y) * H))))
    rw, rh = xb - xa, yb - ya
    if rw < 6 or rh < 6:
        return out

    roi = out[ya:yb, xa:xb]
    blk = max(8, min(rw, rh) // 6)
    gw, gh = max(4, rw // blk), max(4, rh // blk)
    small = cv2.resize(roi, (gw, gh), interpolation=cv2.INTER_AREA)
    mos = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    mos = cv2.GaussianBlur(mos, (9, 9), 0)

    yy, xx = np.ogrid[:rh, :rw]
    nx = ((xx - (rw - 1) * 0.5) / max(1.0, rw * 0.5))
    ny = ((yy - (rh - 1) * 0.5) / max(1.0, rh * 0.5))
    dist = np.sqrt(nx * nx + ny * ny)
    alpha = np.clip((1.05 - dist) / 0.28, 0.0, 1.0).astype(np.float32)[:, :, None]

    blended = np.clip((1.0 - alpha) * roi.astype(np.float32) + alpha * mos.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    out[ya:yb, xa:xb] = blended
    return out


def _fit_pil_text(draw: Any, text: str, font: Any, max_w: int) -> str:
    if not text or draw.textlength(text, font=font) <= max_w:
        return text
    out = text
    while len(out) > 1 and draw.textlength(out + "…", font=font) > max_w:
        out = out[:-1]
    return (out + "…") if out else text[:1]


def _layer_to_patch(layer: Any, x0: int, y0: int) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    box_w, box_h = layer.size
    rgba = np.asarray(layer, dtype=np.float32)
    bgr = rgba[:, :, 2::-1].copy()
    alpha = (rgba[:, :, 3:4] / 255.0).copy()
    return (y0, y0 + box_h, x0, x0 + box_w), bgr, alpha


def _patch_envelope(fi: int, lo: int, hi: int, ramp: int) -> float:
    if fi < lo or fi >= hi or hi <= lo:
        return 0.0
    if fi < lo + ramp:
        return (fi - lo + 1) / float(max(1, ramp))
    if fi >= hi - ramp:
        return (hi - fi) / float(max(1, ramp))
    return 1.0


def _blend_patch(
    frm: np.ndarray,
    patch: tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None,
    gain: float,
) -> None:
    if patch is None or gain <= 0.005:
        return
    (cy0, cy1, cx0, cx1), cbgr, calpha = patch
    H, W = frm.shape[:2]
    y0, y1 = max(0, cy0), min(H, cy1)
    x0, x1 = max(0, cx0), min(W, cx1)
    if y1 <= y0 or x1 <= x0:
        return
    py0, px0 = y0 - cy0, x0 - cx0
    py1, px1 = py0 + (y1 - y0), px0 + (x1 - x0)
    a = calpha[py0:py1, px0:px1] * float(gain)
    roi = frm[y0:y1, x0:x1].astype(np.float32)
    frm[y0:y1, x0:x1] = np.clip((1.0 - a) * roi + a * cbgr[py0:py1, px0:px1], 0, 255).astype(np.uint8)


def render_caption_patch(
    out_w: int,
    out_h: int,
    title_zh: str,
    sub_zh: str,
    font_path: str | None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    from PIL import Image, ImageDraw, ImageFont

    if (not title_zh and not sub_zh) or not font_path:
        return None
    f_title = ImageFont.truetype(font_path, 30)
    f_sub = ImageFont.truetype(font_path, 22)
    dummy = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    t1 = _fit_pil_text(dummy, title_zh or "", f_title, max(120, out_w - 116))
    t2 = _fit_pil_text(dummy, sub_zh or "", f_sub, max(120, out_w - 116))
    tw1 = int(dummy.textlength(t1, font=f_title)) if t1 else 0
    tw2 = int(dummy.textlength(t2, font=f_sub)) if t2 else 0
    box_w = min(out_w - 72, max(tw1, tw2) + 44)
    box_h = 84 if (t1 and t2) else 50
    x0, y1 = 42, out_h - 32
    y0 = y1 - box_h

    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=14, fill=(18, 25, 32, 192))
    if t1:
        draw.text((20, 10), t1, font=f_title, fill=(255, 255, 255, 255))
    if t2:
        draw.text((20, 48 if t1 else 12), t2, font=f_sub, fill=(234, 228, 214, 245))
    return _layer_to_patch(layer, x0, y0)


def render_cinema_subtitle_patch(
    out_w: int,
    out_h: int,
    lyric_zh: str,
    meta_zh: str,
    font_path: str | None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Centered warm cinema-style dual-line subtitle pill at the bottom of the frame."""
    from PIL import Image, ImageDraw, ImageFont

    if (not lyric_zh and not meta_zh) or not font_path:
        return None
    f_lyric = ImageFont.truetype(font_path, max(18, int(round(28 * (out_h / 1080.0)))))
    f_meta = ImageFont.truetype(font_path, max(13, int(round(18 * (out_h / 1080.0)))))
    dummy = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    max_txt_w = max(140, out_w - 140)
    t_lyric = _fit_pil_text(dummy, lyric_zh or "", f_lyric, max_txt_w)
    t_meta = _fit_pil_text(dummy, meta_zh or "", f_meta, max_txt_w)
    tw1 = int(dummy.textlength(t_lyric, font=f_lyric)) if t_lyric else 0
    tw2 = int(dummy.textlength(t_meta, font=f_meta)) if t_meta else 0
    pad_x = 34
    box_w = min(out_w - 72, max(tw1, tw2) + pad_x * 2)
    box_h = 84 if (t_lyric and t_meta) else 50
    x0 = max(20, (out_w - box_w) // 2)
    y0 = max(16, out_h - 36 - box_h)

    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=16, fill=(16, 14, 13, 174))
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=16, outline=(240, 198, 138, 65), width=1)
    if t_lyric:
        draw.text((max(10, (box_w - tw1) // 2), 11), t_lyric, font=f_lyric, fill=(255, 251, 244, 255))
    if t_meta:
        draw.text((max(10, (box_w - tw2) // 2), 49 if t_lyric else 13), t_meta, font=f_meta, fill=(228, 212, 184, 236))
    return _layer_to_patch(layer, x0, y0)


def render_chapter_pill_patch(
    out_w: int,
    out_h: int,
    chapter_zh: str,
    font_path: str | None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Top-right delicate act/chapter badge placed clear of top-left camera OSD timestamps."""
    from PIL import Image, ImageDraw, ImageFont

    if not chapter_zh or not font_path:
        return None
    f_ch = ImageFont.truetype(font_path, max(14, int(round(21 * (out_h / 1080.0)))))
    dummy = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    t_ch = _fit_pil_text(dummy, chapter_zh, f_ch, max(120, out_w // 2))
    tw = int(dummy.textlength(t_ch, font=f_ch))
    box_w = min(out_w - 96, tw + 56)
    box_h = 44
    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=16, fill=(20, 18, 16, 180))
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=16, outline=(240, 179, 94, 108), width=1)
    draw.ellipse([16, 16, 26, 26], fill=(240, 179, 94, 255))
    draw.text((34, 9), t_ch, font=f_ch, fill=(255, 247, 234, 255))
    x0 = max(20, out_w - box_w - 48)
    return _layer_to_patch(layer, x0, 38)


def render_film_title_patch(
    out_w: int,
    out_h: int,
    kicker_zh: str,
    title_zh: str,
    sub_zh: str,
    font_path: str | None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Warm film opening title card placed in lower-third so the baby in center frame stays unobstructed."""
    from PIL import Image, ImageDraw, ImageFont

    if not title_zh or not font_path:
        return None
    scale = out_h / 1080.0
    f_kick = ImageFont.truetype(font_path, max(13, int(round(19 * scale))))
    f_tit = ImageFont.truetype(font_path, max(20, int(round(34 * scale))))
    f_sub = ImageFont.truetype(font_path, max(14, int(round(20 * scale))))
    dummy = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    max_w = max(200, out_w - 160)
    tk = _fit_pil_text(dummy, kicker_zh or "", f_kick, max_w)
    tt = _fit_pil_text(dummy, title_zh, f_tit, max_w)
    ts = _fit_pil_text(dummy, sub_zh or "", f_sub, max_w)
    twk = int(dummy.textlength(tk, font=f_kick)) if tk else 0
    twt = int(dummy.textlength(tt, font=f_tit))
    tws = int(dummy.textlength(ts, font=f_sub)) if ts else 0
    box_w = min(out_w - 84, max(twk, twt, tws, 320) + 84)
    box_h = 54 + (32 if tk else 0) + (34 if ts else 0) + 28
    x0 = max(20, (out_w - box_w) // 2)
    y0 = max(40, min(out_h - box_h - 44, int(out_h * 0.72) - box_h // 2))

    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=20, fill=(16, 14, 13, 186))
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=20, outline=(240, 179, 94, 110), width=1)
    y_cur = 18
    if tk:
        draw.text((max(16, (box_w - twk) // 2), y_cur), tk, font=f_kick, fill=(242, 186, 108, 250))
        y_cur += 32
    draw.text((max(16, (box_w - twt) // 2), y_cur), tt, font=f_tit, fill=(255, 255, 255, 255))
    y_cur += 46
    draw.line([(box_w // 4, y_cur - 4), (3 * box_w // 4, y_cur - 4)], fill=(240, 179, 94, 120), width=1)
    if ts:
        draw.text((max(16, (box_w - tws) // 2), y_cur + 4), ts, font=f_sub, fill=(232, 221, 204, 240))
    return _layer_to_patch(layer, x0, y0)


def render_closing_epilogue_patch(
    out_w: int,
    out_h: int,
    epilogue_zh: str,
    font_path: str | None,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    from PIL import Image, ImageDraw, ImageFont

    if not epilogue_zh or not font_path:
        return None
    f_ep = ImageFont.truetype(font_path, max(16, int(round(24 * (out_h / 1080.0)))))
    dummy = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    t_ep = _fit_pil_text(dummy, epilogue_zh, f_ep, max(160, out_w - 160))
    tw = int(dummy.textlength(t_ep, font=f_ep))
    box_w = min(out_w - 84, tw + 68)
    box_h = 56
    x0 = max(20, (out_w - box_w) // 2)
    y0 = max(24, out_h - 48 - box_h)
    layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=18, fill=(16, 14, 13, 184))
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=18, outline=(240, 179, 94, 96), width=1)
    draw.text((max(12, (box_w - tw) // 2), 14), t_ep, font=f_ep, fill=(255, 244, 222, 255))
    return _layer_to_patch(layer, x0, y0)


def t2_decode_timeout_s(dur: float) -> float:
    """Timeout budget for streaming 1fps T2 decode + encode under heavy concurrent load."""
    return max(360.0, float(dur or 0.0) * 1.15)


def stream_1fps_t2_and_analyze(
    ff_bin: str,
    source_path: str,
    t2_path: Path,
    orig_w: int,
    orig_h: int,
    dur: float,
    yolo_det: Any,
    person_conf: float,
    use_vaapi: bool,
    vaapi_dev: str = "/dev/dri/renderD128",
    t2_h: int = 720,
    t2_qp: int = 30,
    yolo_min_motion: float = 0.0,
    yolo_interval: int = 5,
) -> dict[str, Any] | None:
    """Zero-accumulation streaming 1fps T2 encoder + person/motion/sharpness analyzer (< 20 MB RAM even on 40-min bundles)."""
    w = even(orig_w * (t2_h / float(max(1, orig_h))))
    h = even(t2_h)
    gw, gh = 320, even(320 * h / float(max(1, w)))
    limit_s = t2_decode_timeout_s(dur)
    modes = [(True, True), (False, True), (False, False)] if use_vaapi else [(False, False)]

    sel_1fps = "select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,0.98)+lt(t\\,prev_selected_t)'"
    with _staging_path(t2_path) as staged:
        for hw_dec, hw_enc in modes:
            dec_cmd = [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error"]
            if hw_dec:
                dec_cmd += [
                    "-hwaccel", "vaapi", "-hwaccel_device", vaapi_dev, "-hwaccel_output_format", "vaapi",
                    "-i", source_path, "-an", "-vsync", "0",
                    "-vf", f"{sel_1fps},scale_vaapi=w={w}:h={h}:format=nv12,hwdownload,format=nv12,format=bgr24",
                ]
            else:
                dec_cmd += ["-i", source_path, "-an", "-vsync", "0", "-vf", f"{sel_1fps},scale={w}:{h}"]
            dec_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

            enc_cmd = [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            if hw_enc:
                enc_cmd += ["-init_hw_device", f"vaapi=va:{vaapi_dev}"]
            enc_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0"]
            if hw_enc:
                enc_cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", str(t2_qp)]
            else:
                enc_cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p"]
            enc_cmd += ["-g", "150", "-movflags", "+faststart", str(staged)]
            enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except OSError:
                enc.kill()
                enc.stdin.close()
                enc.wait()
                raise

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
            fsize = w * h * 3

            boxes_list: list[list[tuple[float, float, float, float, float]]] = []
            gmot_list: list[float] = []
            bmot_list: list[float] = []
            pconf_list: list[float] = []
            raw_cands: list[tuple[float, float, list[float], float, float, float, float]] = []
            prev_gray: np.ndarray | None = None
            last_boxes: list[tuple[float, float, float, float, float]] = []
            fi = 0
            complete = False

            try:
                assert dec.stdout is not None and enc.stdin is not None
                while True:
                    buf = dec.stdout.read(fsize)
                    if len(buf) < fsize:
                        complete = not buf
                        break
                    enc.stdin.write(buf)
                    frm = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                    small_bgr = cv2.resize(frm, (gw, gh))
                    gray = cv2.GaussianBlur(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

                    if prev_gray is None:
                        diff_mask = None
                        gm = 0.0
                    else:
                        diff_mask = cv2.absdiff(gray, prev_gray) > 12
                        gm = float(diff_mask.mean())
                    prev_gray = gray

                    if fi == 0 or gm >= max(0.0025, yolo_min_motion) or (yolo_interval > 0 and fi % yolo_interval == 0):
                        last_boxes = yolo_det.detect_persons([frm], conf_thr=0.22)[0]
                    bxs = last_boxes
                    boxes_list.append(bxs)

                    pc = max((b[0] for b in bxs), default=0.0)
                    pconf_list.append(pc)
                    bm = 0.0
                    if diff_mask is not None:
                        for sc, x1, y1, x2, y2 in bxs:
                            if sc < person_conf:
                                continue
                            xa, ya = int(x1 * gw), int(y1 * gh)
                            xb, yb = max(xa + 1, int(x2 * gw)), max(ya + 1, int(y2 * gh))
                            bm = max(bm, float(diff_mask[ya:yb, xa:xb].mean()))
                    gmot_list.append(gm)
                    bmot_list.append(bm)

                    good = [b for b in bxs if b[0] >= person_conf]
                    if good and (fi == 0 or gm >= 0.002 or fi % 4 == 0):
                        big = max(good, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
                        xa, ya = max(0, int(big[1] * w)), max(0, int(big[2] * h))
                        xb, yb = min(w, max(xa + 4, int(big[3] * w))), min(h, max(ya + 4, int(big[4] * h)))
                        crop_g = cv2.cvtColor(frm[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
                        sharp = float(cv2.Laplacian(crop_g, cv2.CV_64F).var())
                        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
                        sat = float(hsv[:, :, 1].mean())
                        luma = float(hsv[:, :, 2].mean())
                        bscore = big[0] * min(1.0, sharp / 120.0) * (0.4 + 0.6 * min(1.0, sat / 35.0)) * (1.2 if len(good) >= 2 else 1.0)
                        raw_cands.append((bscore, float(fi), [round(float(v), 4) for v in big[1:5]], round(sharp, 1), round(sat, 1), round(luma, 1), round(bm, 3)))
                    fi += 1
            except OSError:
                pass
            finally:
                try:
                    if not complete:
                        dec.kill()
                        enc.kill()
                    dec.stdout.close()
                    try:
                        enc.stdin.close()
                    except OSError:
                        pass
                    dec.wait()
                    enc.wait()
                finally:
                    timer.cancel()
                    timer.join()

            if complete and not timed_out.is_set() and dec.returncode == 0 and enc.returncode == 0 and fi > 0 and staged.is_file() and staged.stat().st_size > 0:
                if fi > 1:
                    gmot_list[0] = gmot_list[1]
                    bmot_list[0] = bmot_list[1]
                raw_cands.sort(key=lambda x: -x[0])
                cands: list[dict[str, Any]] = []
                for sc, ts_off, box4, sh_v, sat_v, lu_v, bm_v in raw_cands:
                    if all(abs(ts_off - c["t"]) >= 5.0 for c in cands):
                        cands.append({"t": ts_off, "score": round(sc, 3), "box": box4, "sharp": sh_v, "sat": sat_v, "luma": lu_v, "motion": bm_v})
                    if len(cands) >= 2:
                        break
                os.replace(staged, t2_path)
                return {
                    "n_frames": fi,
                    "boxes": boxes_list,
                    "gmot": np.asarray(gmot_list, dtype=np.float32),
                    "bmot": np.asarray(bmot_list, dtype=np.float32),
                    "pconf": np.asarray(pconf_list, dtype=np.float32),
                    "cands": cands,
                }
            logging.getLogger(__name__).warning(
                "T2 attempt failed for %s: hw_decode=%s hw_encode=%s decoder_rc=%s encoder_rc=%s frames=%d timeout=%s",
                Path(source_path).name, hw_dec, hw_enc, dec.returncode, enc.returncode, fi, timed_out.is_set(),
            )
    return None


def encode_t1_segments(
    ff_bin: str,
    source_path: str,
    t1_path: Path,
    segs: list[list[float]],
    dur: float,
    has_audio: bool,
    use_vaapi: bool,
    vaapi_dev: str = "/dev/dri/renderD128",
    t1_h: int = 720,
    t1_fps: int = 15,
    t1_qp: int = 28,
    pts_jump: bool = False,
) -> tuple[bool, float]:
    """Encode consolidated T1 720p15 clip for a file's active segments."""
    if not segs:
        return False, 0.0
    whole = pts_jump or (len(segs) == 1 and segs[0][0] <= 0.5 and segs[0][1] >= max(0.0, dur - 0.5))
    t1_secs = dur if pts_jump else sum(max(0.0, e - s) for s, e in segs)
    max_out_frames = max(45, int(t1_secs * t1_fps) + 60)
    use_audio = bool(has_audio) and not pts_jump
    end_cap = max(float(e) for _, e in segs) + 1.0
    min_s = min(float(s) for s, _ in segs)
    lead_s = max(0.0, min_s - 4.0) if (not whole and not pts_jump and min_s >= 8.0) else 0.0
    modes = [(True, True), (False, True), (False, False)] if use_vaapi else [(False, False)]

    with _staging_path(t1_path) as staged:
        for hw_dec, hw_enc in modes:
            cmd = [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            zero_copy_vaapi = bool(hw_dec and hw_enc and not pts_jump)
            if hw_enc or hw_dec:
                cmd += ["-init_hw_device", f"vaapi=va:{vaapi_dev}", "-filter_hw_device", "va"]
            if hw_dec:
                cmd += ["-hwaccel", "vaapi", "-hwaccel_device", "va"]
                if zero_copy_vaapi:
                    cmd += ["-hwaccel_output_format", "vaapi"]
            if lead_s > 0.0:
                span_s = max(2.0, end_cap - lead_s)
                cmd += ["-ss", f"{lead_s:.2f}", "-t", f"{span_s:.2f}"]
            elif not whole and end_cap < dur - 2.0:
                cmd += ["-to", f"{end_cap:.2f}"]
            cmd += ["-i", source_path, "-map", "0:v:0"]
            if use_audio:
                cmd += ["-map", "0:a:0"]
            vf_base = (
                f"fps={t1_fps},scale_vaapi=w=-2:h={t1_h}:format=nv12"
                if zero_copy_vaapi
                else f"fps={t1_fps},scale=-2:{t1_h}"
            )
            if pts_jump:
                # Restore source-frame timing before conversion to the archive frame rate.
                # Assigning one 15fps output frame to every 20fps source frame slows and truncates recordings.
                source_rate = "if(isnan(FRAME_RATE)+lte(FRAME_RATE,0),20,clip(FRAME_RATE,10,60))"
                vf = f"setpts='N/(({source_rate})*TB)',fps={t1_fps},scale=-2:{t1_h}"
            elif not whole:
                expr = "+".join(
                    f"between(t\\,{max(0.0, float(s) - lead_s):.2f}\\,{max(0.0, float(e) - lead_s):.2f})"
                    for s, e in segs
                )
                vf = f"select={expr},setpts=N/FRAME_RATE/TB,{vf_base}"
                if use_audio:
                    cmd += ["-af", f"aselect={expr},asetpts=N/SR/TB"]
            else:
                vf = vf_base
            if hw_enc:
                vf_out = vf if zero_copy_vaapi else f"{vf},format=nv12,hwupload"
                cmd += ["-vf", vf_out, "-c:v", "h264_vaapi", "-qp", str(t1_qp)]
            else:
                cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"]
            if use_audio:
                cmd += ["-c:a", "aac", "-b:a", "32k", "-ac", "1", "-ar", "16000"]
            cmd += ["-frames:v", str(max_out_frames), "-g", "150", "-movflags", "+faststart", str(staged)]
            r = subprocess.run(cmd, capture_output=True, timeout=max(180.0, dur * 1.5))
            if r.returncode == 0 and staged.is_file() and staged.stat().st_size > 1000:
                os.replace(staged, t1_path)
                return True, round(t1_secs, 2)
    return False, 0.0


def promote_t0_keeper(src: Path, dst: Path) -> tuple[bool, str]:
    """Promote an original recording into T0 keepers using Btrfs reflink when possible + SHA-256 verification."""
    src_sha = file_sha256(src)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and file_sha256(dst) == src_sha:
        return True, src_sha
    with _staging_path(dst) as staged:
        try:
            subprocess.run(["cp", "--reflink=auto", "--", str(src), str(staged)], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            shutil.copy2(src, staged)
        dst_sha = file_sha256(staged)
        if dst_sha != src_sha:
            return False, dst_sha
        os.replace(staged, dst)
        return True, dst_sha


def _open_raw_h264_encoder(
    ff_bin: str, out_tmp: Path, out_w: int, out_h: int, fps: int,
    use_vaapi: bool, vaapi_dev: str, qp: int,
) -> subprocess.Popen:
    cmd = [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if use_vaapi:
        cmd += ["-init_hw_device", f"vaapi=va:{vaapi_dev}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0"]
    if use_vaapi:
        cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", str(qp)]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(max(18, qp - 1)), "-pix_fmt", "yuv420p"]
    cmd += [str(out_tmp)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _transition_fade_frames(sh: dict[str, Any], default_fade_s: float, fps: int) -> int:
    tr = str(sh.get("transition") or "").strip().lower()
    if tr == "soft_cut":
        sec = 0.20
    elif tr == "chapter_dissolve":
        sec = 0.80
    else:
        sec = float(default_fade_s)
    return max(4, int(round(sec * fps)))


def render_smooth_film_nas(
    ff_bin: str,
    shots: list[dict[str, Any]],
    out_mp4: Path,
    font_path: str | None,
    use_vaapi: bool,
    vaapi_dev: str = "/dev/dri/renderD128",
    *,
    fade_s: float = 0.60,
    fps: int = 30,
    out_w: int = 1920,
    out_h: int = 1080,
    qp: int = 19,
    film_header: dict[str, str] | None = None,
    closing_epilogue_zh: str | None = None,
) -> dict[str, Any] | None:
    """Smooth S-curve crossfade + anti-ghosting PIL CJK captions + leveled audio film (H.264 VAAPI/x264 +faststart)."""
    if not shots:
        return None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    default_fade_frames = max(4, int(round(fade_s * fps)))
    end_fade_frames = (
        max(default_fade_frames, int(round(0.85 * fps)))
        if (film_header or closing_epilogue_zh)
        else default_fade_frames
    )

    with _staging_path(out_mp4) as tmp_v, _staging_path(out_mp4.with_suffix(".f32")) as tmp_a:
        for hw in ((True, False) if use_vaapi else (False,)):
            try:
                enc = _open_raw_h264_encoder(ff_bin, tmp_v, out_w, out_h, fps, hw, vaapi_dev, qp)
            except OSError:
                return None
            tail_v: list[np.ndarray] | None = None
            tail_a: np.ndarray | None = None
            total_frames = 0
            rendered_shots = 0

            try:
                assert enc.stdin is not None
                with tmp_a.open("wb") as afh:
                    for sh_idx, sh in enumerate(shots):
                        is_first_candidate = tail_v is None
                        is_last_shot = sh_idx == len(shots) - 1
                        in_fade = len(tail_v) if tail_v is not None else default_fade_frames
                        out_fade = (
                            end_fade_frames
                            if is_last_shot
                            else _transition_fade_frames(shots[sh_idx + 1], fade_s, fps)
                        )
                        req_frames = int(round(sh["dur_s"] * fps))
                        if is_first_candidate and film_header:
                            req_frames = max(req_frames, out_fade + int(round(3.8 * fps)))
                        n_frames = max(in_fade + out_fade + 12, req_frames)
                        dur_s = n_frames / float(fps)
                        nat_w, nat_h = int(sh["nat_w"]), int(sh["nat_h"])
                        zoom = float(sh.get("zoom", 1.0))
                        box = sh.get("box")
                        pbox = sh.get("privacy_box")
                        has_pbox = bool(pbox and len(pbox) == 4 and pbox[2] > pbox[0] and pbox[3] > pbox[1])
                        use_zoom = zoom > 1.02 and box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]
                        pan_style = str(sh.get("pan_style") or "gentle_push")
                        mapped_pbox = None
                        if use_zoom:
                            dec_w, dec_h = nat_w, nat_h
                            vf = f"fps={fps}"
                            cw0, ch0, x_st, y_st = crop_16_9(nat_w, nat_h, box, zoom_strength=zoom, min_native_w=out_w)
                            cx_m, cy_m = x_st + cw0 * 0.5, y_st + ch0 * 0.5
                            kb_min_cw = min(float(nat_w), float(out_w))
                        else:
                            dec_w, dec_h = out_w, out_h
                            vf = f"fps={fps},scale={out_w}:{out_h}:flags=lanczos:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,unsharp=5:5:0.28:3:3:0.0"
                            if has_pbox:
                                sc_m = min(out_w / float(max(1, nat_w)), out_h / float(max(1, nat_h)))
                                ox_m = (out_w - nat_w * sc_m) * 0.5
                                oy_m = (out_h - nat_h * sc_m) * 0.5
                                mapped_pbox = [
                                    (ox_m + pbox[0] * nat_w * sc_m) / float(out_w),
                                    (oy_m + pbox[1] * nat_h * sc_m) / float(out_h),
                                    (ox_m + pbox[2] * nat_w * sc_m) / float(out_w),
                                    (oy_m + pbox[3] * nat_h * sc_m) / float(out_h),
                                ]

                        if sh.get("cinema_style"):
                            cap = render_cinema_subtitle_patch(
                                out_w, out_h, sh.get("title_zh", ""), sh.get("sub_zh", ""), font_path
                            )
                        else:
                            cap = render_caption_patch(
                                out_w, out_h, sh.get("title_zh", ""), sh.get("sub_zh", ""), font_path
                            )
                        hdr_patch = (
                            render_film_title_patch(
                                out_w,
                                out_h,
                                film_header.get("kicker_zh", ""),
                                film_header.get("title_zh", ""),
                                film_header.get("sub_zh", ""),
                                font_path,
                            )
                            if (is_first_candidate and film_header)
                            else None
                        )
                        chap_patch = (
                            render_chapter_pill_patch(out_w, out_h, str(sh.get("chapter_zh") or ""), font_path)
                            if sh.get("chapter_zh")
                            else None
                        )
                        epi_patch = (
                            render_closing_epilogue_patch(out_w, out_h, str(closing_epilogue_zh or ""), font_path)
                            if (is_last_shot and closing_epilogue_zh)
                            else None
                        )

                        h_lo = int(round(0.35 * fps))
                        h_hi = min(n_frames - out_fade - 4, int(round(3.10 * fps)))
                        h_ramp = min(10, max(2, (h_hi - h_lo) // 5))
                        c_lo = in_fade + 4 if tail_v is not None else 4
                        c_hi = min(n_frames - out_fade - 4, c_lo + int(round(2.7 * fps)))
                        c_ramp = min(8, max(2, (c_hi - c_lo) // 5))

                        slo = max(in_fade + 2, (h_hi + 3) if hdr_patch is not None else (in_fade + 2))
                        shi = n_frames - out_fade - 2
                        e_hi = n_frames - max(4, out_fade // 2)
                        e_lo = max(slo + 8, e_hi - int(round(2.4 * fps)))
                        e_ramp = min(10, max(2, (e_hi - e_lo) // 5))
                        if epi_patch is not None and shi > e_lo - 2:
                            shi = max(slo + 6, e_lo - 2)
                        ramp = min(8, max(2, (shi - slo) // 4))

                        in_alpha_v: np.ndarray | None = None
                        if tail_v is not None:
                            t_lin = np.linspace(0.0, 1.0, in_fade, dtype=np.float32)[:, None, None, None]
                            in_alpha_v = (0.5 - 0.5 * np.cos(math.pi * t_lin)).astype(np.float32)

                        fb = dec_w * dec_h * 3
                        new_tail: list[np.ndarray] = []
                        last_out: np.ndarray | None = None
                        with open_fast_clip_window(
                            ff_bin,
                            sh["source"],
                            max(0.0, float(sh["start_s"])),
                            dur_s + 0.5,
                            include_audio=bool(sh.get("has_audio", True)),
                        ) as (clip_src, clip_start_s):
                            try:
                                v_proc = subprocess.Popen(
                                    [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                                     "-ss", f"{max(0.0, clip_start_s):.2f}", "-t", f"{dur_s + 0.35:.2f}", "-i", clip_src,
                                     "-an", "-vf", vf, "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                )
                            except OSError:
                                continue

                            def _kill_vproc() -> None:
                                try:
                                    v_proc.kill()
                                except OSError:
                                    pass

                            v_timer = threading.Timer(60.0, _kill_vproc)
                            v_timer.daemon = True
                            v_timer.start()
                            assert v_proc.stdout is not None
                            try:
                                for fi in range(n_frames):
                                    buf = v_proc.stdout.read(fb)
                                    if len(buf) == fb:
                                        frm = np.frombuffer(buf, dtype=np.uint8).reshape(dec_h, dec_w, 3)
                                        if use_zoom:
                                            if has_pbox:
                                                frm = apply_privacy_mosaic(frm.copy(), pbox)
                                            u = fi / float(max(1, n_frames - 1))
                                            ease = 0.5 - 0.5 * math.cos(math.pi * u)
                                            # When already at 1:1 native pixel boundary, hold crisp 1:1 without upscaling past native resolution
                                            if cw0 <= kb_min_cw * 1.04 or pan_style == "steady":
                                                cur_cw = float(cw0)
                                            elif pan_style == "gentle_pull":
                                                cur_cw = max(kb_min_cw, cw0 / (1.0 + 0.035 * (1.0 - ease)))
                                            else:
                                                cur_cw = max(kb_min_cw, cw0 / (1.0 + 0.035 * ease))
                                            cur_ch = cur_cw * (out_h / float(out_w))
                                            sx = min(max(0.0, cx_m - cur_cw * 0.5), float(nat_w) - cur_cw)
                                            sy = min(max(0.0, cy_m - cur_ch * 0.5), float(nat_h) - cur_ch)
                                            if abs(cur_cw - out_w) < 2.0 and abs(cur_ch - out_h) < 2.0:
                                                ix = max(0, min(nat_w - out_w, int(round(sx))))
                                                iy = max(0, min(nat_h - out_h, int(round(sy))))
                                                out_frm = _apply_unsharp(frm[iy : iy + out_h, ix : ix + out_w].copy(), 0.28)
                                            else:
                                                M = np.array([[out_w / cur_cw, 0.0, -sx * out_w / cur_cw], [0.0, out_h / cur_ch, -sy * out_h / cur_ch]], dtype=np.float32)
                                                out_frm = _apply_unsharp(
                                                    cv2.warpAffine(frm, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE),
                                                    0.28,
                                                )
                                        else:
                                            out_frm = frm.copy()
                                            if mapped_pbox is not None:
                                                out_frm = apply_privacy_mosaic(out_frm, mapped_pbox)
                                        last_out = out_frm
                                    elif last_out is not None:
                                        out_frm = last_out.copy()
                                    else:
                                        break

                                    _blend_patch(out_frm, hdr_patch, _patch_envelope(fi, h_lo, h_hi, h_ramp))
                                    _blend_patch(out_frm, chap_patch, _patch_envelope(fi, c_lo, c_hi, c_ramp))
                                    _blend_patch(out_frm, cap, _patch_envelope(fi, slo, shi, ramp))
                                    _blend_patch(out_frm, epi_patch, _patch_envelope(fi, e_lo, e_hi, e_ramp))

                                    if tail_v is None and fi < in_fade:
                                        fi_n = min(10, in_fade)
                                        if fi < fi_n:
                                            w_f = (fi + 1) / float(fi_n)
                                            enc.stdin.write(np.clip(out_frm.astype(np.float32) * w_f, 0, 255).astype(np.uint8).tobytes())
                                        else:
                                            enc.stdin.write(out_frm.tobytes())
                                        total_frames += 1
                                    elif tail_v is not None and in_alpha_v is not None and fi < in_fade:
                                        a_f = float(in_alpha_v[fi, 0, 0, 0])
                                        bl = np.clip((1.0 - a_f) * tail_v[fi] + a_f * out_frm, 0, 255).astype(np.uint8)
                                        enc.stdin.write(bl.tobytes())
                                        total_frames += 1
                                    elif fi < n_frames - out_fade:
                                        enc.stdin.write(out_frm.tobytes())
                                        total_frames += 1
                                    else:
                                        new_tail.append(out_frm)
                            finally:
                                v_timer.cancel()
                                v_proc.stdout.close()
                                try:
                                    v_proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    _kill_vproc()
                                    v_proc.wait()

                            if last_out is None or len(new_tail) < out_fade:
                                continue

                            n_samp = int(round(n_frames / float(fps) * 48000))
                            out_samp = int(round(out_fade / float(fps) * 48000))
                            if sh.get("has_audio", True):
                                try:
                                    ar = subprocess.run(
                                        [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                                         "-ss", f"{max(0.0, clip_start_s):.2f}", "-t", f"{dur_s + 0.10:.2f}", "-i", clip_src,
                                         "-vn", "-map", "0:a:0", "-af", "highpass=f=90,aresample=48000",
                                         "-ac", "1", "-ar", "48000", "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1"],
                                        capture_output=True, timeout=60,
                                    )
                                except (OSError, subprocess.SubprocessError):
                                    ar = None
                                cur_a = np.frombuffer(ar.stdout[: (len(ar.stdout) // 4) * 4], dtype="<f4").copy() if ar is not None and ar.returncode == 0 and ar.stdout else np.zeros(0, np.float32)
                            else:
                                cur_a = np.zeros(0, np.float32)
                        cur_a = cur_a[:n_samp] if cur_a.size >= n_samp else np.pad(cur_a, (0, n_samp - cur_a.size))
                        act = cur_a[np.abs(cur_a) > 0.004]
                        if act.size > 480:
                            rms = float(np.sqrt(np.mean(act * act)))
                            if rms > 1e-4:
                                tgt_rms = 0.082 if sh.get("boost_vocal") else 0.055
                                max_gain = 2.8 if sh.get("boost_vocal") else 2.4
                                cur_a = (0.78 * np.tanh((cur_a * min(max_gain, max(0.35, tgt_rms / rms))) / 0.78)).astype(np.float32)

                        rendered_shots += 1
                        if tail_a is None:
                            cur_a[:4800] *= np.linspace(0.0, 1.0, min(4800, len(cur_a)), dtype=np.float32)
                            afh.write(cur_a[:-out_samp].tobytes())
                        else:
                            in_samp = len(tail_a)
                            theta_a = np.linspace(0.0, math.pi / 2.0, in_samp, dtype=np.float32)
                            blend_a = (np.cos(theta_a) * tail_a + np.sin(theta_a) * cur_a[:in_samp]).astype(np.float32)
                            afh.write(blend_a.tobytes())
                            afh.write(cur_a[in_samp:-out_samp].tobytes())
                        tail_v = new_tail
                        tail_a = cur_a[-out_samp:]

                    if tail_v is not None and tail_a is not None:
                        for k, tfrm in enumerate(tail_v):
                            w_end = max(0.0, 1.0 - (k + 1) / float(len(tail_v)))
                            enc.stdin.write(np.clip(tfrm.astype(np.float32) * w_end, 0, 255).astype(np.uint8).tobytes())
                        tail_a = (tail_a * np.linspace(1.0, 0.0, len(tail_a), dtype=np.float32)).astype(np.float32)
                        afh.write(tail_a.tobytes())
                        total_frames += len(tail_v)
            except BrokenPipeError:
                # A failed VAAPI encoder can close its pipe while the decoder is still writing.
                pass
            finally:
                try:
                    enc.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    enc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    enc.kill()
                    enc.wait()

            if enc.returncode == 0 and rendered_shots > 0 and tmp_v.is_file():
                poster = out_mp4.with_suffix(".jpg")
                with _staging_path(out_mp4) as movie, _staging_path(poster) as photo:
                    try:
                        mux = subprocess.run(
                            [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
                             "-i", str(tmp_v), "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", str(tmp_a),
                             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(movie)],
                            capture_output=True, timeout=300,
                        )
                    except subprocess.TimeoutExpired:
                        return None
                    if mux.returncode or not movie.is_file() or movie.stat().st_size <= 1000:
                        return None
                    dur_out = round(total_frames / float(fps), 2)
                    try:
                        thumb = subprocess.run(
                            [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                             "-ss", f"{min(dur_out * 0.5, max(0.8, dur_out * 0.25)):.2f}",
                             "-i", str(movie), "-frames:v", "1", "-q:v", "3", str(photo)],
                            capture_output=True, timeout=30,
                        )
                    except subprocess.TimeoutExpired:
                        return None
                    if thumb.returncode or not photo.is_file() or photo.stat().st_size == 0:
                        return None
                    os.replace(movie, out_mp4)
                    os.replace(photo, poster)
                    return {
                        "file": out_mp4.name,
                        "poster": poster.name,
                        "shots": rendered_shots,
                        "duration_s": dur_out,
                        "bytes": out_mp4.stat().st_size,
                    }
    return None


def render_single_clip_with_badge(
    ff_bin: str,
    source_path: str,
    start_s: float,
    dur_s: float,
    nat_w: int,
    nat_h: int,
    box: list[float] | None,
    zoom: float,
    has_audio: bool,
    title_zh: str,
    sub_zh: str,
    out_mp4: Path,
    font_path: str | None,
    use_vaapi: bool,
    vaapi_dev: str = "/dev/dri/renderD128",
    privacy_box: list[float] | None = None,
) -> bool:
    """Render a standalone 1080p30 milestone/speech clip using single-shot smooth renderer (ensuring valid CJK badge & crispness)."""
    res = render_smooth_film_nas(
        ff_bin,
        [{
            "source": source_path, "start_s": start_s, "dur_s": max(3.5, dur_s),
            "nat_w": nat_w, "nat_h": nat_h, "box": box, "zoom": zoom,
            "has_audio": has_audio, "title_zh": title_zh, "sub_zh": sub_zh,
            "privacy_box": privacy_box,
        }],
        out_mp4, font_path, use_vaapi, vaapi_dev, fade_s=0.40, fps=30, out_w=1920, out_h=1080, qp=19,
    )
    return res is not None and out_mp4.is_file()


def save_photo_pair(
    ff_bin: str,
    source_path: str,
    t_s: float,
    nat_w: int,
    nat_h: int,
    box: list[float] | None,
    zoom: float,
    full_path: Path,
    framed_path: Path,
    privacy_box: list[float] | None = None,
) -> bool:
    """Render full + child-framed JPEGs atomically via temporary staging files (with elliptical privacy mosaic when needed)."""
    full_path.parent.mkdir(parents=True, exist_ok=True)
    framed_path.parent.mkdir(parents=True, exist_ok=True)
    cw, ch, cx, cy = crop_16_9(nat_w, nat_h, box, zoom_strength=zoom, min_native_w=1920)
    full_tmp = full_path.with_suffix(".tmp.jpg")
    framed_tmp = framed_path.with_suffix(".tmp.jpg")
    has_pbox = bool(privacy_box and len(privacy_box) == 4 and privacy_box[2] > privacy_box[0] and privacy_box[3] > privacy_box[1])
    try:
        with open_fast_clip_window(ff_bin, source_path, max(0.0, float(t_s)), 1.5, include_audio=False) as (clip_src, clip_t_s):
            if has_pbox and nat_w > 0 and nat_h > 0:
                raw_r = subprocess.run(
                    [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{max(0.0, clip_t_s):.2f}", "-i", clip_src, "-frames:v", "1",
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                    capture_output=True, timeout=30,
                )
                fb = nat_w * nat_h * 3
                if raw_r.returncode == 0 and len(raw_r.stdout) >= fb:
                    frm = np.frombuffer(raw_r.stdout[:fb], dtype=np.uint8).reshape(nat_h, nat_w, 3).copy()
                    frm = apply_privacy_mosaic(frm, privacy_box)
                    cropped = frm[cy : cy + ch, cx : cx + cw]
                    framed_img = cropped if (cw == 1920 and ch == 1080) else cv2.resize(cropped, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
                    framed_img = _apply_unsharp(framed_img, 0.30)
                    ok1, enc1 = cv2.imencode(".jpg", frm, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    ok2, enc2 = cv2.imencode(".jpg", framed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if ok1 and ok2:
                        full_tmp.write_bytes(enc1.tobytes())
                        framed_tmp.write_bytes(enc2.tobytes())
            else:
                r1 = subprocess.run(
                    [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", f"{max(0.0, clip_t_s):.2f}", "-i", clip_src, "-frames:v", "1", "-q:v", "2", str(full_tmp)],
                    capture_output=True, timeout=30,
                )
                vf_crop = (
                    f"crop={cw}:{ch}:{cx}:{cy},unsharp=5:5:0.30:3:3:0.0"
                    if (cw == 1920 and ch == 1080)
                    else f"crop={cw}:{ch}:{cx}:{cy},scale=1920:1080:flags=lanczos,unsharp=5:5:0.35:3:3:0.0"
                )
                r2 = subprocess.run(
                    [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", f"{max(0.0, clip_t_s):.2f}", "-i", clip_src, "-vf", vf_crop,
                     "-frames:v", "1", "-q:v", "2", str(framed_tmp)],
                    capture_output=True, timeout=30,
                )
                if r1.returncode != 0 or r2.returncode != 0:
                    full_tmp.unlink(missing_ok=True)
                    framed_tmp.unlink(missing_ok=True)
                    return False
        if (
            full_tmp.is_file() and full_tmp.stat().st_size > 0
            and framed_tmp.is_file() and framed_tmp.stat().st_size > 0
        ):
            os.replace(full_tmp, full_path)
            os.replace(framed_tmp, framed_path)
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    full_tmp.unlink(missing_ok=True)
    framed_tmp.unlink(missing_ok=True)
    return False


def concat_mp4_list(ff_bin: str, paths: list[Path], out_mp4: Path, ffprobe_bin: str = "ffprobe") -> bool:
    """Publish a daily timelapse only after every input and the output container can be read."""
    try:
        if not paths or any(not p.is_file() or p.stat().st_size <= 1000 for p in paths):
            return False
        with _staging_path(out_mp4.with_suffix(".txt")) as lst, _staging_path(out_mp4) as tmp:
            names = [p.resolve().as_posix().replace("'", "'\\''") for p in paths]
            if any("\n" in name or "\r" in name for name in names):
                return False
            lst.write_text("".join(f"file '{name}'\n" for name in names), encoding="utf-8")
            r = subprocess.run(
                [ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
                 "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(tmp)],
                capture_output=True, timeout=120,
            )
            if r.returncode or r.stderr.strip() or not tmp.is_file() or tmp.stat().st_size <= 1000:
                return False
            probe = subprocess.run(
                [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
                 "-of", "default=nw=1:nk=1", str(tmp)], capture_output=True, timeout=30,
            )
            if probe.returncode or _safe_int(probe.stdout) < len(paths):
                return False
            os.replace(tmp, out_mp4)
            return True
    except (OSError, subprocess.SubprocessError):
        return False
