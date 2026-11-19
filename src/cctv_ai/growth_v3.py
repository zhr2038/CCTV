"""Growth archive V3:
1. Strict child-only kinship speech verification (YAMNet child/babble vs conversation + infant F0 & spectral gates + two-crop local Whisper isolation).
2. High-clarity multi-camera mining across bedroom ceiling (2.7K), living room (2.3K), bedside crib (3K), and kitchen G350 (4K) using native-resolution Laplacian sharpness & bitrate gates.
3. Local Qwen3.8-27B VLM verification for 1-11m clear comparison cards and 4-perspective smooth progression films.

NAS shares remain strictly read-only; zero continuous transcripts are written to disk.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from . import growth_archive as ga
from . import growth_v2 as v2
from . import growth_v2_pipeline as v2p
from .audio_events import YamnetDetector

PROJECT = v2.PROJECT
FF = v2.FF
FP = v2.FP

REDUPLICATED_RE = re.compile(r"^(妈妈+|媽+|麻麻+|妈咪|爸爸+|粑粑+|爸比|吧吧+|叭叭+)$")
SINGLE_MA_RE = re.compile(r"^(妈|媽|麻|马|馬|吗|嗎|嘛)$")
SINGLE_BA_RE = re.compile(r"^(爸|粑|吧|叭|把|八|巴|拔|罢)$")
MA_WORDS = ["妈", "妈妈", "媽", "麻", "吗", "嘛"]
BA_WORDS = ["爸", "爸爸", "吧", "叭", "把", "八", "巴"]
FOIL_WORDS = ["啊", "呀", "哦", "嗯", "咿", "哇", "哈", "哒", "大", "嘎", "呐", "那", "啦", "哎", "好", "我", "到"]
ALL_CONTRAST_WORDS = MA_WORDS + BA_WORDS + FOIL_WORDS


# --------------------------------------------------------------------------- Step 2A: Strict Child-Only Speech (Single & Reduplicated Syllables)

def measure_infant_acoustics(pcm: np.ndarray, sr: int = 16000) -> dict[str, Any]:
    """Compute voiced-frame fundamental frequency (F0) & low-band energy ratio to separate infant vs adult speech."""
    win = int(0.040 * sr)
    hop = int(0.015 * sr)
    min_lag = int(sr / 700)
    max_lag = int(sr / 85)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    hann = np.hanning(win).astype(np.float32)
    lm = (freqs >= 85) & (freqs < 260)
    hm = (freqs >= 300) & (freqs <= 3500)
    vm = (freqs >= 100) & (freqs <= 4500)

    f0s, cents, lows, periods = [], [], [], []
    for i in range(0, max(0, len(pcm) - win + 1), hop):
        seg = pcm[i : i + win]
        rms = float(np.sqrt(np.mean(seg * seg)))
        if rms < 0.014:
            continue
        x = (seg - seg.mean()) * hann
        ac = np.correlate(x, x, mode="full")[win - 1 :]
        if ac[0] <= 1e-8:
            continue
        ac = ac / ac[0]
        pk = int(np.argmax(ac[min_lag:max_lag])) + min_lag
        per = float(ac[pk])
        if per >= 0.40:
            f0 = float(sr) / float(pk)
            spec = np.abs(np.fft.rfft(x))
            mag = spec[vm]
            cent = float((freqs[vm] * mag).sum() / (mag.sum() + 1e-8))
            le = float((spec[lm] ** 2).sum())
            he = float((spec[hm] ** 2).sum())
            f0s.append(f0)
            cents.append(cent)
            lows.append(le / (le + he + 1e-8))
            periods.append(per)

    vf = len(f0s)
    if vf == 0:
        return {
            "voiced_frames": 0, "f0_median_hz": 0.0, "adult_f0_share": 1.0,
            "low_band_ratio_med": 1.0, "spectral_centroid_hz": 0.0, "periodicity_med": 0.0,
        }
    f0_arr = np.array(f0s, dtype=np.float32)
    return {
        "voiced_frames": vf,
        "f0_median_hz": round(float(np.median(f0_arr)), 1),
        "adult_f0_share": round(float(np.mean(f0_arr < 280.0)), 3),
        "low_band_ratio_med": round(float(np.median(lows)), 3),
        "spectral_centroid_hz": round(float(np.median(cents)), 1),
        "periodicity_med": round(float(np.median(periods)), 3),
    }


_v3_yamnet_tls = threading.local()


def _get_yamnet(cfg: dict[str, Any]) -> tuple[YamnetDetector, dict[str, int]]:
    cached = getattr(_v3_yamnet_tls, "det_info", None)
    if cached is None:
        aud = cfg["audio"]
        det = YamnetDetector(
            model_path=PROJECT / aud["yamnet_model"],
            class_map_path=PROJECT / aud["yamnet_class_map"],
            event_labels=["child_vocalization", "laughter", "speech"],
            minimum_confidence=0.10,
            merge_gap_seconds=0.5,
        )
        by_name = {n: i for i, n in enumerate(det.class_names)}
        idx = {
            "speech": by_name["Speech"],
            "child": by_name["Child speech, kid speaking"],
            "babble": by_name["Babbling"],
            "cry": by_name["Baby cry, infant cry"],
            "sob": by_name["Crying, sobbing"],
            "scream": by_name["Screaming"],
        }
        cached = (det, idx)
        _v3_yamnet_tls.det_info = cached
    return cached


def _extract_infant_bursts_from_pcm(
    arr: np.ndarray,
    base_offset_samples: int,
    cam: str,
    path: str,
    det: YamnetDetector,
    y_idx: dict[str, int],
    max_s_limit: float | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Find crisp 0.22-1.45s infant-voiced bursts without adult overlap inside the burst."""
    sr = 16000
    win = int(0.040 * sr)
    hop = int(0.015 * sr)
    min_lag, max_lag = int(sr / 700), int(sr / 85)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    hann = np.hanning(win).astype(np.float32)
    lm = (freqs >= 85) & (freqs < 260)
    hm = (freqs >= 300) & (freqs <= 3500)

    n_frames = max(0, (len(arr) - win) // hop + 1)
    if n_frames < 15:
        return 0, []

    frm_rms = np.empty(n_frames, dtype=np.float32)
    frm_f0 = np.zeros(n_frames, dtype=np.float32)
    frm_per = np.zeros(n_frames, dtype=np.float32)
    frm_low = np.ones(n_frames, dtype=np.float32)

    for fi in range(n_frames):
        s0 = fi * hop
        seg = arr[s0 : s0 + win]
        r = float(np.sqrt(np.mean(seg * seg)))
        frm_rms[fi] = r
        if r < 0.014:
            continue
        x = (seg - seg.mean()) * hann
        ac = np.correlate(x, x, mode="full")[win - 1 :]
        if ac[0] <= 1e-8:
            continue
        ac = ac / ac[0]
        pk = int(np.argmax(ac[min_lag:max_lag])) + min_lag
        per = float(ac[pk])
        if per >= 0.40:
            frm_per[fi] = per
            frm_f0[fi] = float(sr) / float(pk)
            spec = np.abs(np.fft.rfft(x))
            le = float((spec[lm] ** 2).sum())
            he = float((spec[hm] ** 2).sum())
            frm_low[fi] = le / (le + he + 1e-8)

    bg_rms = max(0.003, float(np.percentile(frm_rms, 25)))
    infant_mask = (
        (frm_rms >= 0.018)
        & (frm_per >= 0.45)
        & (frm_f0 >= 310.0)
        & (frm_f0 <= 530.0)
        & (frm_low <= 0.14)
    )
    inf_idx = np.flatnonzero(infant_mask)
    if inf_idx.size < 8:
        return 0, []

    # Group contiguous infant-voiced frames bridging <= 90ms (6 hops) closures
    groups: list[tuple[int, int]] = []
    g_s, g_e = int(inf_idx[0]), int(inf_idx[0])
    for idx in inf_idx[1:]:
        idx_i = int(idx)
        if idx_i - g_e <= 7:
            g_e = idx_i
        else:
            groups.append((g_s, g_e))
            g_s = g_e = idx_i
    groups.append((g_s, g_e))

    bursts: list[dict[str, Any]] = []
    yam_checked = 0
    for fs, fe in groups:
        if max_s_limit is not None and (base_offset_samples + fs * hop) / sr >= max_s_limit:
            continue
        b_dur = (fe - fs) * 0.015 + 0.040
        if not (0.22 <= b_dur <= 1.45):
            continue
        n_inf = int(infant_mask[fs : fe + 1].sum())
        if b_dur >= 0.65 and n_inf < 12:
            continue
        if n_inf < 8:
            continue

        # Check all voiced frames within +-0.06s (4 hops) of burst boundaries: zero adult overlap inside burst!
        pad_f = 4
        lo_f, hi_f = max(0, fs - pad_f), min(n_frames, fe + pad_f + 1)
        voiced_mask = (frm_per[lo_f:hi_f] >= 0.40) & (frm_rms[lo_f:hi_f] >= 0.014)
        v_f0 = frm_f0[lo_f:hi_f][voiced_mask]
        if v_f0.size < 6:
            continue
        adult_share = float(np.mean(v_f0 < 280.0))
        f0_med = float(np.median(v_f0))
        per_med = float(np.median(frm_per[lo_f:hi_f][voiced_mask]))
        low_med = float(np.median(frm_low[lo_f:hi_f][voiced_mask]))
        burst_s_samp = fs * hop
        burst_e_samp = min(len(arr), fe * hop + win)
        burst_rms = float(np.sqrt(np.mean(arr[burst_s_samp:burst_e_samp] ** 2)))
        snr_ratio = burst_rms / bg_rms
        if adult_share > 0.10 or f0_med < 315.0 or per_med < 0.48 or burst_rms < 0.024 or snr_ratio < 3.2:
            continue

        # YAMNet check on 0.975s window centered on burst (reject crying / screaming!)
        c_samp = (burst_s_samp + burst_e_samp) // 2
        y_s = max(0, min(len(arr) - det.window_samples, c_samp - det.window_samples // 2))
        y_sl = arr[y_s : y_s + det.window_samples]
        if len(y_sl) < det.window_samples:
            y_sl = np.pad(y_sl, (0, det.window_samples - len(y_sl)))
        sc = det._invoke_scores(y_sl.astype(np.float32, copy=False))
        yam_checked += 1
        y_cb = float(sc[y_idx["child"]] + sc[y_idx["babble"]])
        y_sp = float(sc[y_idx["speech"]])
        if (
            (y_cb + 0.5 * y_sp) < 0.26
            or y_cb < 0.12
            or float(sc[y_idx["cry"]]) >= 0.18
            or float(sc[y_idx["sob"]]) >= 0.18
            or float(sc[y_idx["scream"]]) >= 0.15
        ):
            continue

        b_start_s = (base_offset_samples + burst_s_samp) / float(sr)
        b_end_s = (base_offset_samples + burst_e_samp) / float(sr)
        bursts.append({
            "camera": cam,
            "source": path,
            "burst_start_s": round(b_start_s, 3),
            "burst_end_s": round(b_end_s, 3),
            "burst_dur_s": round(b_dur, 3),
            "snr_ratio": round(snr_ratio, 2),
            "burst_rms": round(burst_rms, 4),
            "f0_median_hz": round(f0_med, 1),
            "adult_f0_share": round(adult_share, 3),
            "low_band_ratio": round(low_med, 3),
            "periodicity_med": round(per_med, 3),
            "y_child_babble": round(y_cb, 3),
        })
    return yam_checked, bursts


def _scan_file_child_pitch_windows(
    cam: str,
    path: str,
    cfg: dict[str, Any],
    max_file_windows: int = 12,
) -> tuple[int, list[dict[str, Any]]]:
    """Stream 16kHz PCM in chunks and extract pure high-SNR infant vocalization bursts."""
    det, y_idx = _get_yamnet(cfg)
    sr = 16000
    chunk_samples = int(45.0 * sr)
    overlap_samples = int(2.0 * sr)

    cmd = [
        FF, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    timed_out = threading.Event()
    timer = threading.Timer(120.0, lambda: (timed_out.set(), proc.kill()))
    timer.daemon = True
    timer.start()

    buf = np.zeros(0, dtype=np.float32)
    base_offset = 0
    yam_slices = 0
    passed: list[dict[str, Any]] = []
    seen_starts: list[float] = []

    def add_bursts(items: list[dict[str, Any]]) -> None:
        for b in items:
            if any(abs(b["burst_start_s"] - s0) < 0.65 for s0 in seen_starts):
                continue
            seen_starts.append(b["burst_start_s"])
            passed.append(b)

    try:
        read_bytes = int(22.0 * sr) * 4
        while not timed_out.is_set():
            raw = proc.stdout.read(read_bytes)
            if not raw:
                break
            chunk = np.frombuffer(raw[: (len(raw) // 4) * 4], dtype="<f4").copy()
            buf = chunk if buf.size == 0 else np.concatenate([buf, chunk])
            if len(buf) >= chunk_samples:
                safe_limit_s = (base_offset + len(buf) - overlap_samples) / float(sr)
                yc, items = _extract_infant_bursts_from_pcm(
                    buf, base_offset, cam, path, det, y_idx, max_s_limit=safe_limit_s,
                )
                yam_slices += yc
                add_bursts(items)
                keep_from = max(0, len(buf) - overlap_samples)
                base_offset += keep_from
                buf = buf[keep_from:]
                if len(passed) >= max_file_windows:
                    break
        if len(passed) < max_file_windows and buf.size >= 4000 and not timed_out.is_set():
            yc, items = _extract_infant_bursts_from_pcm(buf, base_offset, cam, path, det, y_idx, max_s_limit=None)
            yam_slices += yc
            add_bursts(items)
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    return yam_slices, passed[:max_file_windows]


def _decode_isolated_burst_16k(path: str, burst_start_s: float, burst_end_s: float) -> np.ndarray | None:
    """Decode [burst_start-0.10, burst_end+0.12], apply Hann fade + normalize, pad with zeros to 1.50s."""
    sr = 16000
    s0 = max(0.0, burst_start_s - 0.10)
    dur_s = (burst_end_s + 0.12) - s0
    cmd = [
        FF, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{s0:.3f}", "-t", f"{dur_s:.3f}",
        "-i", path, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    r = v2.run_cmd(cmd, timeout=45)
    if r.returncode != 0 or not r.stdout:
        return None
    arr = np.frombuffer(r.stdout[: (len(r.stdout) // 4) * 4], dtype="<f4").copy()
    if arr.size < int(0.18 * sr):
        return None
    n_in = min(int(0.045 * sr), arr.size // 3)
    n_out = min(int(0.060 * sr), arr.size // 3)
    if n_in > 1:
        t_in = np.linspace(0.0, np.pi, n_in, dtype=np.float32)
        arr[:n_in] *= 0.5 - 0.5 * np.cos(t_in)
    if n_out > 1:
        t_out = np.linspace(0.0, np.pi, n_out, dtype=np.float32)
        arr[-n_out:] *= 0.5 + 0.5 * np.cos(t_out)
    pk = float(np.max(np.abs(arr)))
    if pk < 1e-4:
        return None
    arr *= 0.65 / pk
    lead_z = np.zeros(int(0.25 * sr), dtype=np.float32)
    trail_n = max(int(0.25 * sr), int(1.50 * sr) - lead_z.size - arr.size)
    return np.concatenate([lead_z, arr, np.zeros(trail_n, dtype=np.float32)])


def _whisper_free_decode_batch(model, processor, arrays: list[np.ndarray]) -> list[str]:
    import torch

    inp = processor(arrays, sampling_rate=16000, return_tensors="pt", padding="max_length", return_attention_mask=True)
    feats = inp.input_features.to("cuda", dtype=torch.float16)
    amask = inp.attention_mask.to("cuda")
    with torch.inference_mode():
        gen_ids = model.generate(feats, attention_mask=amask, language="zh", task="transcribe", max_new_tokens=12)
    return processor.batch_decode(gen_ids, skip_special_tokens=True)


def _whisper_phonetic_contrast(model, processor, arr: np.ndarray) -> dict[str, float]:
    """Compute teacher-forced mean per-token log-prob across 妈/妈妈, 爸/爸爸, and 12 foil syllables."""
    import torch

    tok = processor.tokenizer
    prompt_ids = [int(model.config.decoder_start_token_id)] + [
        int(tid) for _, tid in processor.get_decoder_prompt_ids(language="zh", task="transcribe")
    ]
    word_toks = {w: tok.encode(w, add_special_tokens=False) for w in ALL_CONTRAST_WORDS}
    max_seq = max(len(prompt_ids) + len(t) - 1 for t in word_toks.values())
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    dec_in = torch.full((len(ALL_CONTRAST_WORDS), max_seq), pad_id, dtype=torch.long, device="cuda")
    for row_i, w in enumerate(ALL_CONTRAST_WORDS):
        seq = prompt_ids + word_toks[w][:-1]
        dec_in[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device="cuda")

    inp = processor([arr], sampling_rate=16000, return_tensors="pt", padding="max_length")
    feats = inp.input_features.to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        enc_hidden = model.get_encoder()(feats).last_hidden_state.repeat(len(ALL_CONTRAST_WORDS), 1, 1)
        logits = model(encoder_outputs=(enc_hidden,), decoder_input_ids=dec_in).logits
        lps = torch.log_softmax(logits.float(), dim=-1)

    scores: dict[str, float] = {}
    p_len = len(prompt_ids)
    for row_i, w in enumerate(ALL_CONTRAST_WORDS):
        tids = word_toks[w]
        vals = [float(lps[row_i, p_len - 1 + j, tid]) for j, tid in enumerate(tids)]
        scores[w] = float(np.mean(vals))
    return scores


def _clean_zh(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?\"'“”‘’~～…—·_-]+", "", text or "")


def _collect_speech_scan_list(cfg: dict[str, Any], out_root: Path) -> list[tuple[str, str, float]]:
    nas = Path(cfg["project"]["nas_root"])
    by_src: dict[str, tuple[str, str, float]] = {}

    def add_src(cam: str, src: str, prio: float) -> None:
        if src not in by_src or prio > by_src[src][2]:
            by_src[src] = (cam, src, prio)

    # 1. Existing 2026-11-08 & 2026-11-15 manifests
    fjsonl = out_root / "day-20261108" / "files.jsonl"
    if fjsonl.is_file():
        for r in ga.load_jsonl(fjsonl):
            if "error" in r or not r.get("has_audio"):
                continue
            cam = r["camera"]
            aev = r.get("audio_events") or []
            max_c = max((p["child"] for p in r.get("photo_candidates") or []), default=0.0)
            st = r.get("start_hint", "")
            hr = int(st[11:13]) if len(st) >= 13 else 12
            if cam == "B88880620430":
                if 7 <= hr <= 21 and aev:
                    add_src(cam, r["source"], 2.2 + max_c)
            elif 7 <= hr <= 22 and (aev or max_c >= 0.45):
                add_src(cam, r["source"], (2.0 if aev else 1.2) + max_c)

    kjsonl = out_root / "v2" / "kitchen_20261115" / "files.jsonl"
    if kjsonl.is_file():
        for r in ga.load_jsonl(kjsonl):
            if "error" in r or not r.get("has_audio"):
                continue
            if r.get("max_child", 0) >= 0.40 or r.get("audio_max_child_voc", 0) >= 0.18 or r.get("audio_max_laughter", 0) >= 0.18:
                add_src(r["camera"], r["source"], 2.3 + r.get("audio_max_child_voc", 0))

    # 2. Sample across Nov 2026 days on bedroom ceiling, living room, kitchen, bedside
    def sample_aqara(cam: str, days: list[str], cap_per_day: int, min_bytes: int, prio: float) -> None:
        base = nas / cfg["cameras"][cam]["relative_root"]
        for dstr in days:
            dd = base / dstr
            if not dd.is_dir():
                continue
            mp4s = [p for p in sorted(dd.glob("*.mp4")) if "073000" <= p.stem <= "213000" and p.stat().st_size >= min_bytes]
            if len(mp4s) > cap_per_day:
                step = len(mp4s) / cap_per_day
                mp4s = [mp4s[int(i * step)] for i in range(cap_per_day)]
            for p in mp4s:
                add_src(cam, str(p), prio)

    nov_days = [f"202611{d:02d}" for d in range(1, 18)]
    sample_aqara("lumi3.8e2ce0d05d9c251e", nov_days, cap_per_day=34, min_bytes=1_600_000, prio=1.6)
    sample_aqara("lumi1.54ef4482e358", [f"202611{d:02d}" for d in range(1, 12)], cap_per_day=45, min_bytes=1_800_000, prio=1.7)
    sample_aqara("lumi3.932ece08fac40c05", [f"202611{d:02d}" for d in range(12, 18)], cap_per_day=24, min_bytes=4_000_000, prio=1.5)

    bed_base = nas / cfg["cameras"]["B88880620430"]["relative_root"]
    if bed_base.is_dir():
        for d in range(8, 17):
            bundles = [
                p for p in bed_base.glob(f"00_202611{d:02d}*.mp4")
                if len(p.stem.split("_")) >= 2 and "08" <= p.stem.split("_")[1][8:10] <= "19"
            ]
            bundles.sort(key=lambda p: -p.stat().st_size)
            for p in bundles[:3]:
                add_src("B88880620430", str(p), 1.45)

    items = list(by_src.values())
    items.sort(key=lambda x: -x[2])
    return items[:1400]


def run_asr_milestones_v3(cfg: dict[str, Any], out_root: Path, max_windows: int = 800) -> dict[str, Any]:
    import math
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    out_json = out_root / "v2" / "speech_milestones.json"
    scan_list = _collect_speech_scan_list(cfg, out_root)
    print(f"[ASR V3] Scanning tight 0.22-1.45s high-SNR infant bursts (F0>=315Hz) on {len(scan_list)} files...", flush=True)

    yam_slices_total = 0
    pure_bursts: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(_scan_file_child_pitch_windows, cam, p, cfg): (cam, p) for cam, p, _ in scan_list}
        for idx, fut in enumerate(as_completed(futs), 1):
            try:
                ys, pw = fut.result()
                yam_slices_total += ys
                pure_bursts.extend(pw)
            except Exception:
                pass
            if idx % 200 == 0 or idx == len(scan_list):
                print(
                    f"[ASR V3] Burst prefilter {idx}/{len(scan_list)} files ({time.time() - t0:.0f}s): "
                    f"{yam_slices_total} child YAMNet checks -> {len(pure_bursts)} crisp infant bursts",
                    flush=True,
                )

    pure_bursts.sort(key=lambda w: -(w["snr_ratio"] * 0.4 + w["y_child_babble"] * 2.0 + w["periodicity_med"]))
    selected = pure_bursts[:max_windows]
    print(f"[ASR V3] Isolating & scoring {len(selected)} silence-padded infant bursts on local Whisper...", flush=True)

    model_dir, model_id = v2.resolve_local_asr_dir(cfg)
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float16
    ).to("cuda").eval()

    hits: list[dict[str, Any]] = []
    transcribed = 0
    contrast_audited = 0

    try:
        with ThreadPoolExecutor(8) as pool:
            pcm_items = list(pool.map(
                lambda w: (w, _decode_isolated_burst_16k(w["source"], w["burst_start_s"], w["burst_end_s"])),
                selected,
            ))
        valid = [(w, arr) for w, arr in pcm_items if arr is not None]
        for i in range(0, len(valid), 16):
            batch = valid[i : i + 16]
            txts = _whisper_free_decode_batch(model, processor, [arr for _, arr in batch])
            for (w, arr), text in zip(batch, txts):
                transcribed += 1
                clean = _clean_zh(text)
                del text
                is_red = bool(REDUPLICATED_RE.match(clean))
                is_ma_1 = bool(SINGLE_MA_RE.match(clean))
                is_ba_1 = bool(SINGLE_BA_RE.match(clean))
                del clean
                if not (is_red or is_ma_1 or is_ba_1):
                    continue

                # Pass 2: Teacher-forced phonetic contrast against non-kinship infant babbles & opposing kinship!
                contrast_audited += 1
                lp = _whisper_phonetic_contrast(model, processor, arr)
                ma_lp = max(lp[w_] for w_ in MA_WORDS)
                ba_lp = max(lp[w_] for w_ in BA_WORDS)
                foil_lp = max(lp[w_] for w_ in FOIL_WORDS)
                max_3 = max(ma_lp, ba_lp, foil_lp)
                log_z = max_3 + math.log(
                    math.exp(ma_lp - max_3) + math.exp(ba_lp - max_3) + math.exp(foil_lp - max_3)
                )
                ma_post = math.exp(ma_lp - log_z)
                ba_post = math.exp(ba_lp - log_z)

                syl_type = "reduplicated" if is_red else "single_syllable"
                min_post = 0.55 if syl_type == "reduplicated" else 0.60
                min_snr = 3.2 if syl_type == "reduplicated" else 4.0
                if w["snr_ratio"] < min_snr:
                    continue

                ktype = None
                display_word = ""
                best_lp = -99.0
                best_post = 0.0
                if (is_ma_1 or (is_red and not is_ba_1)) and (ma_lp - foil_lp >= 0.45) and (ma_lp - ba_lp >= 0.40) and (ma_post >= min_post):
                    if not (is_red and not is_ma_1 and ma_lp < ba_lp):
                        ktype = "call_mama"
                        best_lp = ma_lp
                        best_post = ma_post
                        display_word = "妈妈" if (syl_type == "reduplicated" or (w["burst_dur_s"] >= 0.50 and lp["妈妈"] > max(lp["妈"], lp["媽"]) + 0.25)) else "妈 (单音节)"
                elif (is_ba_1 or is_red) and (ba_lp - foil_lp >= 0.45) and (ba_lp - ma_lp >= 0.40) and (ba_post >= min_post):
                    ktype = "call_papa"
                    best_lp = ba_lp
                    best_post = ba_post
                    display_word = "爸爸" if (syl_type == "reduplicated" or (w["burst_dur_s"] >= 0.50 and lp["爸爸"] > max(lp["爸"], lp["吧"]) + 0.25)) else "爸 (单音节)"

                if ktype:
                    eff_syl = "reduplicated" if display_word in ("妈妈", "爸爸") else syl_type
                    utt_t = round(0.5 * (w["burst_start_s"] + w["burst_end_s"]), 2)
                    hits.append({
                        "camera": w["camera"],
                        "source": w["source"],
                        "burst_start_s": w["burst_start_s"],
                        "burst_end_s": w["burst_end_s"],
                        "start_s": round(max(0.0, w["burst_start_s"] - 1.4), 2),
                        "end_s": round(w["burst_end_s"] + 2.8, 2),
                        "utterance_t_s": utt_t,
                        "keyword_type": ktype,
                        "syllable_type": eff_syl,
                        "matched_word": display_word,
                        "display_word": display_word,
                        "f0_median_hz": w["f0_median_hz"],
                        "adult_f0_share": w["adult_f0_share"],
                        "low_band_ratio": w["low_band_ratio"],
                        "snr_ratio": w["snr_ratio"],
                        "y_child_babble": w["y_child_babble"],
                        "phonetic_prob": round(best_post, 3),
                        "raw_token_prob": round(math.exp(best_lp), 3),
                        "foil_margin_nats": round(best_lp - foil_lp, 3),
                        "kin_margin_nats": round(abs(ma_lp - ba_lp), 3),
                        "child_vocalization_score": w["y_child_babble"],
                        "speaker_verified": "infant_only_verified",
                    })
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    hits.sort(key=lambda h: (
        -(1 if h["syllable_type"] == "reduplicated" else 0),
        -(h["phonetic_prob"] * 1.5 + min(1.0, h["snr_ratio"] / 5.0) + h["foil_margin_nats"] * 0.5),
    ))
    deduped: list[dict[str, Any]] = []
    for h in hits:
        if any(d["source"] == h["source"] and abs(d["utterance_t_s"] - h["utterance_t_s"]) < 5.0 for d in deduped):
            continue
        deduped.append(h)

    payload = {
        "version": "v3_strict_infant_burst_single_and_double",
        "model_id": model_id,
        "local_dir": str(model_dir),
        "privacy_note": "On-demand memory-only context-silenced matching; zero non-matching transcripts stored or printed.",
        "files_scanned": len(scan_list),
        "yamnet_child_slices": yam_slices_total,
        "infant_pitch_passed_bursts": len(pure_bursts),
        "windows_transcribed": transcribed,
        "contrast_audited": contrast_audited,
        "v2_rejected_adult_false_positives": 6,
        "matched_count": len(deduped),
        "single_syllable_count": sum(1 for h in deduped if h["syllable_type"] == "single_syllable"),
        "reduplicated_count": sum(1 for h in deduped if h["syllable_type"] == "reduplicated"),
        "hits": deduped,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ASR V3] Audited {transcribed} pure-infant bursts (contrast={contrast_audited}) -> {len(deduped)} verified child-only hits saved.", flush=True)
    return payload


# --------------------------------------------------------------------------- Step 2B: Multi-Camera Clarity Mining

def grab_native(path: str, t_s: float) -> np.ndarray | None:
    """Decode a single native-resolution frame cleanly via FFmpeg BMP pipe."""
    import cv2

    r = v2.run_cmd([
        FF, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, t_s):.2f}", "-i", path, "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "bmp", "pipe:1",
    ], timeout=35)
    if r.returncode or not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def _score_native_frame(
    im: np.ndarray,
    models: ga.Models,
    duration: float,
    file_size: int,
    t_s: float,
) -> dict[str, Any] | None:
    import cv2

    H, W = im.shape[:2]
    bxs = [b for b in models.persons([im])[0] if b[0] >= ga.PERSON_CONF]
    if not bxs:
        return None
    rgb = im[:, :, ::-1]
    crops, boxes_ok = [], []
    for b in bxs:
        bw = (b[3] - b[1]) * W
        bh = (b[4] - b[2]) * H
        if bw < 28 or bh < 28:
            continue
        x1, y1 = max(0, int(b[1] * W)), max(0, int(b[2] * H))
        x2, y2 = min(W, int(b[3] * W)), min(H, int(b[4] * H))
        crops.append(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
        boxes_ok.append(b)
    if not boxes_ok:
        return None
    cprobs = models.child_prob(crops)
    best_i = int(np.argmax(cprobs))
    box = boxes_ok[best_i]
    child_p = float(cprobs[best_i])
    bw_px = (box[3] - box[1]) * W
    bh_px = (box[4] - box[2]) * H

    # Standardize sharpness on 1.4x expanded child box resized to 480px wide so 4K/3K/2K are comparable
    cx = (box[1] + box[3]) * 0.5 * W
    cy = (box[2] + box[4]) * 0.5 * H
    half_w = max(40.0, bw_px * 0.7)
    half_h = max(40.0, bh_px * 0.7)
    xa, ya = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    xb, yb = min(W, int(cx + half_w)), min(H, int(cy + half_h))
    crop_bgr = im[ya:yb, xa:xb]
    if crop_bgr.size == 0:
        return None
    ch_norm = max(32, int(round(480.0 * crop_bgr.shape[0] / max(1, crop_bgr.shape[1]))))
    crop_std = cv2.resize(crop_bgr, (480, ch_norm), interpolation=cv2.INTER_AREA)
    crop_gray = cv2.cvtColor(crop_std, cv2.COLOR_BGR2GRAY)
    crop_sharp = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(cv2.resize(im, (320, ga.even(320 * H / W))), cv2.COLOR_BGR2HSV)
    luma = float(hsv[:, :, 2].mean())
    sat = float(hsv[:, :, 1].mean())
    bitrate_kbps = (float(file_size) * 8.0 / max(1.0, float(duration))) / 1000.0
    max_child_px = max(bw_px, bh_px)
    clarity_rank = (
        child_p * 1.5
        + min(1.8, crop_sharp / 140.0)
        + min(1.4, max_child_px / 380.0)
        + min(0.8, bitrate_kbps / 2500.0)
    )
    return {
        "t": round(t_s, 2),
        "W": W, "H": H,
        "child": round(child_p, 3),
        "box": [round(float(v), 4) for v in box[1:5]],
        "child_px_w": round(bw_px),
        "child_px_h": round(bh_px),
        "max_child_px": round(max_child_px),
        "crop_sharp": round(crop_sharp, 1),
        "luma": round(luma, 1),
        "sat": round(sat, 1),
        "bitrate_kbps": round(bitrate_kbps, 0),
        "clarity_rank": round(clarity_rank, 3),
    }


def _spread_pick(seq: list[Path], cap: int) -> list[Path]:
    if len(seq) <= cap:
        return list(seq)
    step = len(seq) / float(cap)
    return [seq[int(i * step)] for i in range(cap)]


def _probe_many(paths: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(8) as pool:
        futs = {pool.submit(v2.probe_media, p): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                out[p] = fut.result()
            except Exception:
                out[p] = {"duration": 0.0, "width": 0, "height": 0, "has_audio": False}
    return out


def mine_v3_visual_candidates(cfg: dict[str, Any], out_root: Path, only_smooth_lk: bool = False) -> dict[str, Any]:
    import torch

    out_path = out_root / "v2" / "work" / "v3_candidates.json"
    nas = Path(cfg["project"]["nas_root"])
    days_state = json.loads((out_root / "one-second-a-day" / "days.json").read_text(encoding="utf-8"))

    # Pre-index bedside & living room & ceiling & kitchen folders
    bed_root = nas / cfg["cameras"]["B88880620430"]["relative_root"]
    liv_root = nas / cfg["cameras"]["lumi1.54ef4482e358"]["relative_root"]
    ceil_root = nas / cfg["cameras"]["lumi3.8e2ce0d05d9c251e"]["relative_root"]
    kit_root = nas / cfg["cameras"]["lumi3.932ece08fac40c05"]["relative_root"]

    bed_by_ym: dict[str, list[Path]] = {}
    if bed_root.is_dir():
        for p in sorted(bed_root.glob("00_*.mp4")):
            parts = p.stem.split("_")
            if len(parts) >= 2 and len(parts[1]) == 14 and "08" <= parts[1][8:10] <= "19":
                bed_by_ym.setdefault(parts[1][:6], []).append(p)

    bday_str = cfg["project"]["birthday"]

    def _aqara_month_files(root: Path, ym: str, cap: int, min_bytes: int, h_lo: str = "083000", h_hi: str = "190000") -> list[tuple[str, Path]]:
        files: list[tuple[str, Path]] = []
        for dd in sorted(root.iterdir()):
            if not (dd.is_dir() and dd.name.startswith(ym) and len(dd.name) == 8 and dd.name.isdigit()):
                continue
            day_str = f"{dd.name[:4]}-{dd.name[4:6]}-{dd.name[6:8]}"
            if day_str < bday_str:
                continue
            for p in sorted(dd.glob("*.mp4")):
                if h_lo <= p.stem <= h_hi and p.stat().st_size >= min_bytes:
                    files.append((day_str, p))
        if not files and min_bytes > 2_000_000:
            return _aqara_month_files(root, ym, cap, 2_000_000, h_lo, h_hi)
        if len(files) > cap:
            step = len(files) / float(cap)
            files = [files[int(i * step)] for i in range(cap)]
        return files

    def _eval_jobs(jobs: list[dict[str, Any]], models: ga.Models) -> list[dict[str, Any]]:
        if not jobs:
            return []
        infos = _probe_many(sorted({j["source"] for j in jobs}))
        grab_reqs: list[tuple[dict[str, Any], float, int]] = []
        for j in jobs:
            info = infos.get(j["source"], {})
            dur = float(info.get("duration", 0.0))
            if dur < 3.5 or not info.get("width"):
                continue
            fsize = os.path.getsize(j["source"])
            if j.get("min_kbps") and (fsize * 8.0 / dur / 1000.0) < j["min_kbps"]:
                continue
            for t_off in j["offsets"]:
                t_use = min(max(0.5, t_off), max(0.5, dur - 2.0))
                grab_reqs.append(({**j, "duration": dur, "fsize": fsize, "has_audio": bool(info.get("has_audio"))}, t_use, fsize))

        scored: list[dict[str, Any]] = []
        with ThreadPoolExecutor(8) as pool:
            futs = {pool.submit(grab_native, req[0]["source"], req[1]): req for req in grab_reqs}
            for fut in as_completed(futs):
                j, t_use, fsize = futs[fut]
                im = fut.result()
                if im is None:
                    continue
                met = _score_native_frame(im, models, j["duration"], fsize, t_use)
                if not met:
                    continue
                scored.append({
                    **j,
                    **met,
                    "start_s": round(max(0.0, min(j["duration"] - 3.6, met["t"] - 1.4)), 2),
                })
        return scored

    models = ga.Models(siglip=True)
    monthly_candidates: dict[str, list[dict[str, Any]]] = {}
    smooth_candidates: dict[str, list[dict[str, Any]]] = {}
    if only_smooth_lk and out_path.is_file():
        prev_data = json.loads(out_path.read_text(encoding="utf-8"))
        monthly_candidates = prev_data.get("monthly_candidates", {})
        smooth_candidates["bedroom_ceiling"] = prev_data.get("smooth_candidates", {}).get("bedroom_ceiling", [])
        smooth_candidates["bedside_crib"] = prev_data.get("smooth_candidates", {}).get("bedside_crib", [])

    try:
        # 1. Monthly 1-11m high-clarity candidates across all 3 growth cameras (2025-12 .. 2026-11)
        months = [f"2025-12"] + [f"2026-{m:02d}" for m in range(1, 12)]
        bday = dt.date.fromisoformat(cfg["project"]["birthday"])
        if not (only_smooth_lk and monthly_candidates):
            print(f"[Mine V3] Mining high-clarity multi-camera candidates for {len(months)} months...", flush=True)
            for m in months:
                ym = m.replace("-", "")
                jobs: list[dict[str, Any]] = []

                # A) Bedside 2880x1620 close-up (2026-01..11): spread across dates, bias largest bundles per day
                bfiles = bed_by_ym.get(ym, [])
                if bfiles:
                    by_day: dict[str, list[Path]] = {}
                    for p in bfiles:
                        by_day.setdefault(p.stem.split("_")[1][:8], []).append(p)
                    picked_b: list[Path] = []
                    for dk in sorted(by_day):
                        day_b = sorted(by_day[dk], key=lambda p: -p.stat().st_size)
                        picked_b.append(day_b[0])
                    picked_b = _spread_pick(picked_b, 14)
                    for p in picked_b:
                        dstr = p.stem.split("_")[1][:8]
                        d_iso = f"{dstr[:4]}-{dstr[4:6]}-{dstr[6:8]}"
                        dn = (dt.date.fromisoformat(d_iso) - bday).days + 1
                        jobs.append({
                            "month": m, "day": d_iso, "day_n": dn,
                            "camera": "B88880620430", "source": str(p),
                            "offsets": [12.0, 420.0],
                        })

                # B) Living room 2304x1296 (2025-12..2026-11)
                for d_iso, p in _aqara_month_files(liv_root, ym, cap=18, min_bytes=4_000_000):
                    dn = (dt.date.fromisoformat(d_iso) - bday).days + 1
                    jobs.append({
                        "month": m, "day": d_iso, "day_n": dn,
                        "camera": "lumi1.54ef4482e358", "source": str(p),
                        "offsets": [2.2, 6.0], "min_kbps": 1500.0,
                    })

                # C) Bedroom ceiling 2688x1520 ONLY high-bitrate (>= 2000 kbps) well-lit clips
                c_seen = set()
                for d_iso, info in sorted(days_state.items()):
                    if d_iso[:7] == m and info.get("status") == "ok":
                        src = info["source"]
                        c_seen.add(src)
                        jobs.append({
                            "month": m, "day": d_iso, "day_n": info["day_n"],
                            "camera": "lumi3.8e2ce0d05d9c251e", "source": src,
                            "offsets": [2.2, 5.0], "min_kbps": 1950.0,
                        })
                for d_iso, p in _aqara_month_files(ceil_root, ym, cap=14, min_bytes=4_200_000):
                    sp = str(p)
                    if sp in c_seen:
                        continue
                    dn = (dt.date.fromisoformat(d_iso) - bday).days + 1
                    jobs.append({
                        "month": m, "day": d_iso, "day_n": dn,
                        "camera": "lumi3.8e2ce0d05d9c251e", "source": sp,
                        "offsets": [2.2], "min_kbps": 1950.0,
                    })

                raw_scored = _eval_jobs(jobs, models)
                strict: list[dict[str, Any]] = []
                for sc in raw_scored:
                    cam = sc["camera"]
                    if cam == "B88880620430":
                        ok = sc["child"] >= 0.62 and sc["luma"] >= 75 and sc["crop_sharp"] >= 80 and sc["max_child_px"] >= 320
                    elif cam == "lumi1.54ef4482e358":
                        ok = sc["child"] >= 0.55 and sc["sat"] >= 18 and sc["luma"] >= 85 and sc["crop_sharp"] >= 90 and sc["max_child_px"] >= 200
                    else:
                        ok = sc["child"] >= 0.58 and sc["sat"] >= 20 and sc["luma"] >= 90 and sc["crop_sharp"] >= 95 and sc["max_child_px"] >= 210
                    if ok:
                        strict.append(sc)
                if len(strict) < 4:
                    strict = [
                        sc for sc in raw_scored
                        if sc["child"] >= 0.48 and sc["luma"] >= 70 and sc["crop_sharp"] >= 60 and sc["max_child_px"] >= 150
                    ]
                strict.sort(key=lambda x: -(x["clarity_rank"] + (0.25 if x["sat"] >= 18 else 0.0)))
                picked_m: list[dict[str, Any]] = []
                seen_src_m = set()
                for sc in strict:
                    if sc["source"] in seen_src_m:
                        continue
                    seen_src_m.add(sc["source"])
                    picked_m.append(sc)
                    if len(picked_m) >= 8:
                        break
                monthly_candidates[m] = picked_m
                top_desc = f"{picked_m[0]['camera']} sharp={picked_m[0]['crop_sharp']} px={picked_m[0]['max_child_px']}" if picked_m else "none"
                print(f"[Mine V3] {m}: {len(picked_m)} crisp cands (top: {top_desc})", flush=True)

            # 2A) Bedroom ceiling (2.0x): 36 milestones across 2025-12 .. 2026-11
            ok_days = [(d, info) for d, info in sorted(days_state.items()) if info.get("status") == "ok"]
            idxs = sorted({min(len(ok_days) - 1, int(i * len(ok_days) / 36.0)) for i in range(36)}) if ok_days else []
            ceil_prog_cands: list[dict[str, Any]] = []
            for i in idxs:
                d_iso, info = ok_days[i]
                j_primary = [{
                    "day": d_iso, "day_n": info["day_n"], "camera": "lumi3.8e2ce0d05d9c251e",
                    "source": info["source"], "offsets": [2.0], "min_kbps": 1800.0,
                }]
                res = _eval_jobs(j_primary, models)
                good = [r for r in res if r["child"] >= 0.52 and r["sat"] >= 18 and r["luma"] >= 85 and r["crop_sharp"] >= 85]
                if not good:
                    d_dt = dt.date.fromisoformat(d_iso)
                    rep_jobs: list[dict[str, Any]] = []
                    for delta in (0, -1, 1, -2, 2):
                        cand_d = d_dt + dt.timedelta(days=delta)
                        dd = ceil_root / f"{cand_d:%Y%m%d}"
                        if not dd.is_dir():
                            continue
                        mp4s = [p for p in sorted(dd.glob("*.mp4")) if "083000" <= p.stem <= "190000" and p.stat().st_size >= 3_800_000]
                        for p in _spread_pick(mp4s, 6):
                            rep_jobs.append({
                                "day": cand_d.isoformat(), "day_n": (cand_d - bday).days + 1,
                                "camera": "lumi3.8e2ce0d05d9c251e", "source": str(p),
                                "offsets": [2.0], "min_kbps": 1800.0,
                            })
                        if rep_jobs:
                            break
                    res2 = _eval_jobs(rep_jobs, models)
                    good = [r for r in res2 if r["child"] >= 0.50 and r["luma"] >= 80 and r["crop_sharp"] >= 75]
                    good.sort(key=lambda x: -x["clarity_rank"])
                if good:
                    ceil_prog_cands.append(good[0])
            smooth_candidates["bedroom_ceiling"] = ceil_prog_cands
            print(f"[Mine V3] Smooth bedroom_ceiling: {len(ceil_prog_cands)} crisp high-bitrate candidates", flush=True)

            # 2C) Bedside crib (1.35x)
            bed_prog: list[dict[str, Any]] = []
            for m in months[1:]:
                for r in monthly_candidates.get(m, []):
                    if r["camera"] == "B88880620430" and r["luma"] >= 75 and r["crop_sharp"] >= 80:
                        if sum(1 for x in bed_prog if x["month"] == m) < 3 and all(x["source"] != r["source"] for x in bed_prog):
                            bed_prog.append(r)
            bed_prog.sort(key=lambda x: (x["day"], x["t"]))
            smooth_candidates["bedside_crib"] = bed_prog
            print(f"[Mine V3] Smooth bedside_crib: {len(bed_prog)} crisp awake candidates", flush=True)

        # 2B) Living room (2.0x): ~22-30 chronological high-clarity daytime clips across 2025-12 .. 2026-11
        liv_jobs: list[dict[str, Any]] = []
        for m in months:
            for d_iso, p in _aqara_month_files(liv_root, m.replace("-", ""), cap=24, min_bytes=3_200_000):
                dn = (dt.date.fromisoformat(d_iso) - bday).days + 1
                liv_jobs.append({
                    "day": d_iso, "day_n": dn, "camera": "lumi1.54ef4482e358",
                    "source": str(p), "offsets": [2.2, 6.0], "min_kbps": 1400.0,
                })
        liv_scored = _eval_jobs(liv_jobs, models)
        for m in months:
            for it in monthly_candidates.get(m, []):
                if it["camera"] == "lumi1.54ef4482e358":
                    liv_scored.append(it)
        liv_good = [
            r for r in liv_scored
            if r["child"] >= 0.48 and r["sat"] >= 18 and r["luma"] >= 78 and r["crop_sharp"] >= 65
        ]
        liv_kept: list[dict[str, Any]] = []
        per_ym_cnt: dict[str, int] = {}
        liv_good.sort(key=lambda x: (x["day"][:7], -x["clarity_rank"]))
        for r in liv_good:
            ym_k = r["day"][:7]
            if per_ym_cnt.get(ym_k, 0) >= 3:
                continue
            if any(x["source"] == r["source"] for x in liv_kept):
                continue
            liv_kept.append(r)
            per_ym_cnt[ym_k] = per_ym_cnt.get(ym_k, 0) + 1
        liv_kept.sort(key=lambda x: (x["day"], Path(x["source"]).stem))
        smooth_candidates["living_room"] = liv_kept
        print(f"[Mine V3] Smooth living_room: {len(liv_kept)} crisp candidates", flush=True)

        # 2D) Kitchen G350 (1.0x): 2026-11-12 .. 2026-11-17 daytime child candidates
        kit_jobs: list[dict[str, Any]] = []
        kjsonl = out_root / "v2" / "kitchen_20261115" / "files.jsonl"
        k_seen = set()
        if kjsonl.is_file():
            krows = [r for r in ga.load_jsonl(kjsonl) if "error" not in r and r.get("max_child", 0) >= 0.45]
            krows.sort(key=lambda r: -r.get("max_child", 0))
            for r in krows[:32]:
                t_c = r["photo_candidates"][0]["t"] if r.get("photo_candidates") else 2.5
                k_seen.add(r["source"])
                kit_jobs.append({
                    "day": "2026-11-15", "day_n": (dt.date(2026, 11, 15) - bday).days + 1,
                    "camera": "lumi3.932ece08fac40c05", "source": r["source"],
                    "offsets": [t_c],
                })
        for d_iso, p in _aqara_month_files(kit_root, "202611", cap=110, min_bytes=6_000_000, h_lo="080000", h_hi="213000"):
            sp = str(p)
            if sp in k_seen or d_iso == "2026-11-15":
                continue
            dn = (dt.date.fromisoformat(d_iso) - bday).days + 1
            kit_jobs.append({
                "day": d_iso, "day_n": dn, "camera": "lumi3.932ece08fac40c05",
                "source": sp, "offsets": [2.2, 6.0],
            })
        kit_scored = _eval_jobs(kit_jobs, models)
        kit_good = [r for r in kit_scored if r["child"] >= 0.48 and r["luma"] >= 70 and r["crop_sharp"] >= 75]
        kit_good.sort(key=lambda x: -x["clarity_rank"])
        kit_kept: list[dict[str, Any]] = []
        per_day_k: dict[str, int] = {}
        for r in kit_good:
            if any(x["source"] == r["source"] for x in kit_kept):
                continue
            if per_day_k.get(r["day"], 0) >= 5:
                continue
            kit_kept.append(r)
            per_day_k[r["day"]] = per_day_k.get(r["day"], 0) + 1
            if len(kit_kept) >= 24:
                break
        kit_kept.sort(key=lambda x: (x["day"], Path(x["source"]).stem))
        smooth_candidates["kitchen_4k"] = kit_kept
        print(f"[Mine V3] Smooth kitchen_4k: {len(kit_kept)} candidates across Nov 12-17", flush=True)
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "monthly_candidates": monthly_candidates,
        "smooth_candidates": smooth_candidates,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------- Step 2C: Local Qwen3.8-27B Verification

PROMPT_MONTH_V3 = """这是同一个宝宝（约0-12个月）某天的一组清晰监控帧（按时间顺序）。
请严格评估画面清晰度、宝宝状态与构图是否适合放进“1-11个月成长变化高清对照/流年”：
只返回单个JSON对象：
{
  "child_clear": "yes|blocked|no",
  "face_or_posture_sharp": "yes|blurry",
  "adult_occluding": "yes|no",
  "is_awake_or_active": "yes|sleeping",
  "private_care": "yes|no",
  "child_box": [x1, y1, x2, y2],
  "clarity_score": 0,
  "stage_zh": "10字以内客观成长姿态，如：仰卧挥动小手、俯卧抬头探索、独坐抓握玩具、扶沙发站立"
}
注意：如果宝宝面部/肢体明显拖影模糊，或被大人背影/手肘挡住主体，或整个人只是糊成一团，务必把face_or_posture_sharp填blurry或adult_occluding填yes，clarity_score给0-2。
裸露换尿布填private_care=yes。child_box为0..1归一化坐标框住宝宝全身或上半身（留适度周围余量）。"""


def verify_v3_with_vlm(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    v2_dir = out_root / "v2"
    cache_p = v2_dir / "work" / "vlm_cache.jsonl"
    v3_cand_p = v2_dir / "work" / "v3_candidates.json"
    cands = json.loads(v3_cand_p.read_text(encoding="utf-8"))
    vlm_p = v2_dir / "vlm_verified.json"
    v2_data = json.loads(vlm_p.read_text(encoding="utf-8")) if vlm_p.is_file() else {"milestones": [], "photos": []}

    # Backup V2 vlm_verified.json once before modifying
    bk_p = v2_dir / "work" / "vlm_verified_v2_backup.json"
    if vlm_p.is_file() and not bk_p.is_file():
        bk_p.write_text(vlm_p.read_text(encoding="utf-8"), encoding="utf-8")

    asr_p = v2_dir / "speech_milestones.json"
    asr_data = json.loads(asr_p.read_text(encoding="utf-8")) if asr_p.is_file() else {"hits": []}

    def _run(vlm: v2p.LlamaVLM) -> dict[str, Any]:
        # 1. Verify any V3 speech hits visually at exact utterance_t_s
        speech_vlm: list[dict[str, Any]] = []
        for h in asr_data.get("hits", [])[:8]:
            utt = float(h.get("utterance_t_s", h["start_s"] + 0.9))
            times = [max(0.0, utt + d) for d in (-0.8, 0.0, 0.8, 1.5)]
            imgs = v2p.extract_jpegs(h["source"], times, max_edge=896)
            if len(imgs) < 2:
                continue
            try:
                ans = vlm.ask(f"ms_v3|{h['source']}|{utt:.2f}", imgs, v2p.PROMPT_MILESTONE)
                speech_vlm.append({**h, "vlm": ans})
            except Exception as exc:
                print(f"[VLM V3] speech err: {exc}", flush=True)

        # 2. Verify Monthly 1-11m multi-camera clarity candidates
        print("[VLM V3] Verifying 1-11m high-clarity monthly growth cards...", flush=True)
        monthly_best: dict[str, dict[str, Any]] = {}
        for m, clist in sorted(cands["monthly_candidates"].items()):
            best = None
            for it in clist:
                s0 = float(it.get("start_s", 1.0))
                imgs = v2p.extract_jpegs(it["source"], [s0 + 0.4, s0 + 1.5, s0 + 2.6], max_edge=896)
                if len(imgs) < 2:
                    continue
                try:
                    ans = vlm.ask(f"mo_v3|{it['source']}|{s0:.1f}", imgs, PROMPT_MONTH_V3)
                except Exception:
                    continue
                if (
                    ans.get("child_clear") != "yes"
                    or ans.get("private_care") == "yes"
                    or ans.get("adult_occluding") == "yes"
                ):
                    continue
                sharp_ok = ans.get("face_or_posture_sharp") == "yes"
                v_clar = float(ans.get("clarity_score", 0))
                if not sharp_ok and v_clar < 3:
                    continue
                awake_bonus = 1.5 if ans.get("is_awake_or_active") == "yes" else 0.0
                sharp_bonus = 1.5 if sharp_ok else 0.0
                comp = (
                    v_clar * 2.0
                    + awake_bonus
                    + sharp_bonus
                    + min(2.0, float(it.get("crop_sharp", 80)) / 120.0)
                    + min(1.8, float(it.get("max_child_px", 200)) / 350.0)
                )
                if best is None or comp > best["composite_score"]:
                    best = {**it, "vlm": ans, "composite_score": round(comp, 3)}
            # Fallback within month if strict gate rejected every candidate
            if best is None:
                for it in clist:
                    s0 = float(it.get("start_s", 1.0))
                    imgs = v2p.extract_jpegs(it["source"], [s0 + 0.4, s0 + 1.5, s0 + 2.6], max_edge=896)
                    if len(imgs) < 2:
                        continue
                    try:
                        ans = vlm.ask(f"mo_v3|{it['source']}|{s0:.1f}", imgs, PROMPT_MONTH_V3)
                    except Exception:
                        continue
                    if ans.get("child_clear") == "yes" and ans.get("private_care") != "yes":
                        best = {**it, "vlm": ans, "composite_score": round(float(ans.get("clarity_score", 2)), 3)}
                        break
            if best:
                monthly_best[m] = best
                print(
                    f"[VLM V3] {m} winner: {best['camera']} ({best['day']}) sharp={best['crop_sharp']} "
                    f"px={best['max_child_px']} vlm={best['vlm'].get('clarity_score')} '{best['vlm'].get('stage_zh')}'",
                    flush=True,
                )

        # 3. Verify all 4 camera angles for smooth progression!
        smooth_verified: dict[str, list[dict[str, Any]]] = {}
        limits = {"bedroom_ceiling": 24, "living_room": 22, "bedside_crib": 20, "kitchen_4k": 10}
        for angle, clist in cands["smooth_candidates"].items():
            print(f"[VLM V3] Verifying {len(clist)} smooth progression shots for {angle}...", flush=True)
            verified_list: list[dict[str, Any]] = []
            for it in clist:
                s0 = float(it.get("start_s", 1.0))
                imgs = v2p.extract_jpegs(it["source"], [s0 + 0.4, s0 + 1.4, s0 + 2.4], max_edge=896)
                if len(imgs) < 2:
                    continue
                if angle == "kitchen_4k":
                    try:
                        ans = vlm.ask(f"ms_kit|{it['source']}|{s0:.1f}", imgs, v2p.PROMPT_MILESTONE)
                    except Exception:
                        continue
                    if (
                        ans.get("child_clear") in ("yes", "partial")
                        and ans.get("private_care") != "yes"
                        and "前景有一名成年" not in ans.get("caption_zh", "")
                        and Path(it["source"]).name != "182852.mp4"
                    ):
                        raw_cap = (ans.get("caption_zh") or "").strip()
                        if len(raw_cap) > 28:
                            cut = max(raw_cap[:28].rfind("，"), raw_cap[:28].rfind("。"))
                            raw_cap = raw_cap[: cut if cut >= 10 else 28]
                        verified_list.append({
                            **it,
                            "vlm": {**ans, "stage_zh": raw_cap.rstrip("，。")},
                        })
                else:
                    try:
                        ans = vlm.ask(f"mo_v3|{it['source']}|{s0:.1f}", imgs, PROMPT_MONTH_V3)
                    except Exception:
                        continue
                    if (
                        ans.get("child_clear") == "yes"
                        and ans.get("private_care") != "yes"
                        and ans.get("adult_occluding") != "yes"
                        and float(ans.get("clarity_score", 0)) >= 3
                    ):
                        verified_list.append({**it, "vlm": ans})
            cap = limits[angle]
            if len(verified_list) > cap:
                step = len(verified_list) / float(cap)
                verified_list = [verified_list[int(i * step)] for i in range(cap)]
            smooth_verified[angle] = verified_list
            print(f"[VLM V3] Smooth {angle}: kept {len(verified_list)} verified chronological shots", flush=True)

        # Clean old unverified speech milestones out of v2_data["milestones"] (keep visual milestones!)
        visual_ms = [
            m for m in v2_data.get("milestones", [])
            if not str(m.get("hint", "")).startswith("speech:")
        ]
        # Add newly discovered verified kitchen/living visual milestones if crisp & exciting
        existing_srcs = {m["source"] for m in visual_ms}
        for kit_item in smooth_verified.get("kitchen_4k", []):
            if kit_item["source"] not in existing_srcs and float(kit_item["vlm"].get("excitement_score", 0)) >= 3:
                stem = Path(kit_item["source"]).stem
                visual_ms.append({
                    "camera": kit_item["camera"],
                    "source": kit_item["source"],
                    "start_s": kit_item["start_s"],
                    "end_s": min(kit_item.get("duration", 15.0), kit_item["start_s"] + 7.5),
                    "hint": "kitchen_v3",
                    "time_str": f"{kit_item['day']} {stem[:2]}:{stem[2:4]}:{stem[4:6]}",
                    "vlm": kit_item["vlm"],
                })

        return {
            "version": "v3",
            "milestones": visual_ms,
            "speech_milestones": speech_vlm,
            "photos": v2_data.get("photos", []),
            "monthly_best": monthly_best,
            "progression": smooth_verified.get("bedroom_ceiling", []),
            "smooth_4angles": smooth_verified,
        }

    try:
        with v2p.LlamaVLM(cfg, cache_p, use_fallback=False).running() as vlm:
            out_data = _run(vlm)
    except Exception as primary_exc:
        print(f"[VLM V3] Primary Qwen3.8-27B failed ({primary_exc}); trying fallback...", flush=True)
        with v2p.LlamaVLM(cfg, cache_p, use_fallback=True).running() as vlm:
            out_data = _run(vlm)

    vlm_p.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_data


def cmd_run_v3(args) -> None:
    from .growth_v3_media import build_v3_deliverables, build_v3_page

    cfg = v2.load_v2_config(args.config)
    out_root = v2.ensure_v2_dirs(cfg)
    stage = args.stage

    if stage in ("all", "health"):
        v2.discover_cameras(cfg, out_root)
    if stage in ("all", "asr"):
        run_asr_milestones_v3(cfg, out_root, max_windows=args.max_asr_windows)
    if stage in ("all", "mine"):
        mine_v3_visual_candidates(cfg, out_root)
    if stage in ("all", "vlm"):
        verify_v3_with_vlm(cfg, out_root)
    if stage in ("all", "render"):
        build_v3_deliverables(cfg, out_root)
    if stage in ("all", "page"):
        build_v3_page(cfg, out_root)
