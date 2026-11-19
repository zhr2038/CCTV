"""Lightweight CPU ONNX YOLO26 person detector, TFLite YAMNet audio detector, and pure-infant speech burst isolator."""
from __future__ import annotations

import csv
import subprocess
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PERSON_CONF = 0.28
BOX_MOTION = 0.030
GLOBAL_MOTION = 0.010
PAD_BEFORE, PAD_AFTER, MERGE_GAP, MIN_SEG = 2.0, 2.0, 8.0, 3.0


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
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


class YoloOnnxDetector:
    """Thread-safe sharded CPU ONNX Runtime person detector (YOLO26s end-to-end NMS-free 640x640)."""

    def __init__(self, onnx_path: str | Path, intra_threads: int = 1, pool_size: int = 3) -> None:
        import onnxruntime as ort

        n_slots = max(1, int(pool_size))
        self._slots: list[tuple[Any, threading.Lock]] = []
        for _ in range(n_slots):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, int(intra_threads))
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._slots.append((sess, threading.Lock()))
        self.sess = self._slots[0][0]
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self._rr = 0
        self._rr_lock = threading.Lock()

    def _acquire_slot(self) -> tuple[Any, threading.Lock]:
        n = len(self._slots)
        with self._rr_lock:
            start = self._rr
            self._rr = (start + 1) % n
        for offset in range(n):
            sess, lock = self._slots[(start + offset) % n]
            if lock.acquire(blocking=False):
                return sess, lock
        sess, lock = self._slots[start]
        lock.acquire()
        return sess, lock

    @staticmethod
    def _letterbox_640(bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        h, w = bgr.shape[:2]
        r = min(640.0 / w, 640.0 / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pw, ph = (640 - nw) / 2.0, (640 - nh) / 2.0
        top, bottom = int(round(ph - 0.1)), int(round(ph + 0.1))
        left, right = int(round(pw - 0.1)), int(round(pw + 0.1))
        canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        nchw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
        return nchw, r, float(left), float(top)

    def detect_persons(self, frames: list[np.ndarray], conf_thr: float = 0.20) -> list[list[tuple[float, float, float, float, float]]]:
        out: list[list[tuple[float, float, float, float, float]]] = []
        for frm in frames:
            oh, ow = frm.shape[:2]
            nchw, r, px, py = self._letterbox_640(frm)
            sess, lock = self._acquire_slot()
            try:
                raw = sess.run([self.out_name], {self.in_name: nchw})[0]
            finally:
                lock.release()
            arr = raw[0] if (raw.ndim == 3 and raw.shape[0] == 1) else raw
            boxes: list[tuple[float, float, float, float, float]] = []
            if arr.ndim == 2 and arr.shape[1] == 6:
                for x1, y1, x2, y2, sc, cls_id in arr:
                    if int(round(cls_id)) != 0 or float(sc) < conf_thr:
                        continue
                    nx1 = max(0.0, min(1.0, (float(x1) - px) / r / ow))
                    ny1 = max(0.0, min(1.0, (float(y1) - py) / r / oh))
                    nx2 = max(0.0, min(1.0, (float(x2) - px) / r / ow))
                    ny2 = max(0.0, min(1.0, (float(y2) - py) / r / oh))
                    if nx2 > nx1 and ny2 > ny1:
                        boxes.append((float(sc), nx1, ny1, nx2, ny2))
            boxes.sort(key=lambda b: -b[0])
            out.append(boxes)
        return out


class YamnetLite:
    """Lightweight TFLite YAMNet audio event & infant vocalization detector."""

    def __init__(self, model_path: str | Path, class_map_path: str | Path) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf  # type: ignore
                Interpreter = tf.lite.Interpreter

        self.model_path = Path(model_path)
        self.class_names: dict[int, str] = {}
        with Path(class_map_path).open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                self.class_names[int(row["index"])] = row["display_name"]
        by_name = {v: k for k, v in self.class_names.items()}
        self.idx = {
            "speech": by_name["Speech"],
            "child": by_name["Child speech, kid speaking"],
            "babble": by_name["Babbling"],
            "laughter": [by_name[n] for n in ("Laughter", "Baby laughter", "Giggle", "Chuckle, chortle") if n in by_name],
            "cry": by_name["Baby cry, infant cry"],
            "sob": by_name["Crying, sobbing"],
            "scream": by_name["Screaming"],
        }
        self._tls = threading.local()
        self._interp_cls = Interpreter
        self.window_samples = 15600  # 0.975s @ 16kHz

    def _get_interp(self):
        interp = getattr(self._tls, "interp", None)
        if interp is None:
            interp = self._interp_cls(model_path=str(self.model_path), num_threads=1)
            interp.allocate_tensors()
            in_det = interp.get_input_details()[0]
            if list(in_det["shape"]) != [self.window_samples]:
                interp.resize_tensor_input(in_det["index"], [self.window_samples])
                interp.allocate_tensors()
            self._tls.interp = interp
            self._tls.in_idx = interp.get_input_details()[0]["index"]
            self._tls.out_idx = interp.get_output_details()[0]["index"]
        return interp

    def score_window(self, pcm_15600: np.ndarray) -> np.ndarray:
        interp = self._get_interp()
        x = pcm_15600.astype(np.float32, copy=False)
        if x.shape[0] < self.window_samples:
            x = np.pad(x, (0, self.window_samples - x.shape[0]))
        else:
            x = x[: self.window_samples]
        interp.set_tensor(self._tls.in_idx, x)
        interp.invoke()
        out = interp.get_tensor(self._tls.out_idx)
        return out.mean(axis=0) if out.ndim == 2 else out


def extract_infant_bursts_from_pcm(
    arr: np.ndarray,
    base_offset_samples: int,
    cam: str,
    source_rel: str,
    yamnet: YamnetLite,
    max_s_limit: float | None = None,
) -> list[dict[str, Any]]:
    """Isolate pure 0.20-1.55s infant vocalization bursts supporting both nasal /m/ (妈/妈妈) and plosive /b/ (爸/爸爸)."""
    sr = 16000
    win = int(0.040 * sr)
    hop = int(0.015 * sr)
    min_lag, max_lag = int(sr / 700), int(sr / 85)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    hann = np.hanning(win).astype(np.float32)
    lm = (freqs >= 85) & (freqs < 250)
    hm = (freqs >= 290) & (freqs <= 3500)

    n_frames = max(0, (len(arr) - win) // hop + 1)
    if n_frames < 14:
        return []

    frames = np.lib.stride_tricks.sliding_window_view(arr, win)[::hop]
    frm_rms = np.sqrt(np.einsum("ij,ij->i", frames, frames) / win)
    frm_f0 = np.zeros(n_frames, dtype=np.float32)
    frm_per = np.zeros(n_frames, dtype=np.float32)
    frm_low = np.ones(n_frames, dtype=np.float32)

    for fi in np.flatnonzero(frm_rms >= 0.013):
        seg = frames[fi]
        x = (seg - seg.mean()) * hann
        ac = np.correlate(x, x, mode="full")[win - 1 :]
        if ac[0] <= 1e-8:
            continue
        ac = ac / ac[0]
        pk = int(np.argmax(ac[min_lag:max_lag])) + min_lag
        per = float(ac[pk])
        if per >= 0.38:
            frm_per[fi] = per
            frm_f0[fi] = float(sr) / float(pk)
            spec = np.abs(np.fft.rfft(x))
            le = float((spec[lm] ** 2).sum())
            he = float((spec[hm] ** 2).sum())
            frm_low[fi] = le / (le + he + 1e-8)

    bg_rms = max(0.003, float(np.percentile(frm_rms, 25)))
    infant_mask = (
        (frm_rms >= 0.016)
        & (frm_per >= 0.42)
        & (frm_f0 >= 305.0)
        & (frm_f0 <= 540.0)
        & (frm_low <= 0.22)
    )
    inf_idx = np.flatnonzero(infant_mask)
    if inf_idx.size < 7:
        return []

    groups: list[tuple[int, int]] = []
    g_s, g_e = int(inf_idx[0]), int(inf_idx[0])
    for idx in inf_idx[1:]:
        idx_i = int(idx)
        if idx_i - g_e <= 8:
            g_e = idx_i
        else:
            groups.append((g_s, g_e))
            g_s = g_e = idx_i
    groups.append((g_s, g_e))

    bursts: list[dict[str, Any]] = []
    y_idx = yamnet.idx
    for fs, fe in groups:
        if max_s_limit is not None and (base_offset_samples + fs * hop) / sr >= max_s_limit:
            continue
        b_dur = (fe - fs) * 0.015 + 0.040
        if not (0.20 <= b_dur <= 1.55):
            continue
        n_inf = int(infant_mask[fs : fe + 1].sum())
        if n_inf < (11 if b_dur >= 0.65 else 7):
            continue

        pad_f = 4
        lo_f, hi_f = max(0, fs - pad_f), min(n_frames, fe + pad_f + 1)
        voiced_mask = (frm_per[lo_f:hi_f] >= 0.38) & (frm_rms[lo_f:hi_f] >= 0.013)
        v_f0 = frm_f0[lo_f:hi_f][voiced_mask]
        if v_f0.size < 6:
            continue
        adult_share = float(np.mean(v_f0 < 265.0))
        f0_med = float(np.median(v_f0))
        per_med = float(np.median(frm_per[lo_f:hi_f][voiced_mask]))
        low_med = float(np.median(frm_low[lo_f:hi_f][voiced_mask]))
        bs_samp = fs * hop
        be_samp = min(len(arr), fe * hop + win)
        burst_rms = float(np.sqrt(np.mean(arr[bs_samp:be_samp] ** 2)))
        snr_ratio = burst_rms / bg_rms
        if adult_share > 0.10 or f0_med < 310.0 or per_med < 0.44 or burst_rms < 0.022 or snr_ratio < 3.0 or low_med > 0.22:
            continue

        c_samp = (bs_samp + be_samp) // 2
        y_s = max(0, min(len(arr) - yamnet.window_samples, c_samp - yamnet.window_samples // 2))
        sc = yamnet.score_window(arr[y_s : y_s + yamnet.window_samples])
        y_cb = float(sc[y_idx["child"]] + sc[y_idx["babble"]])
        y_sp = float(sc[y_idx["speech"]])
        if (
            (y_cb + 0.5 * y_sp) < 0.24
            or y_cb < 0.10
            or float(sc[y_idx["cry"]]) >= 0.18
            or float(sc[y_idx["sob"]]) >= 0.18
            or float(sc[y_idx["scream"]]) >= 0.15
        ):
            continue

        bursts.append({
            "camera": cam,
            "source_rel": source_rel,
            "burst_start_s": round((base_offset_samples + bs_samp) / float(sr), 3),
            "burst_end_s": round((base_offset_samples + be_samp) / float(sr), 3),
            "burst_dur_s": round(b_dur, 3),
            "snr_ratio": round(snr_ratio, 2),
            "f0_median_hz": round(f0_med, 1),
            "adult_f0_share": round(adult_share, 3),
            "low_band_ratio": round(low_med, 3),
            "periodicity_med": round(per_med, 3),
            "y_child_babble": round(y_cb, 3),
        })
    return bursts


def analyze_audio_and_bursts(
    ff_bin: str,
    source_path: str,
    source_rel: str,
    cam: str,
    yamnet: YamnetLite,
    scan_bursts: bool = True,
    max_bursts: int = 10,
) -> dict[str, Any]:
    """Single-pass 16kHz PCM stream extracting both YAMNet timeline events AND pure-infant speech bursts."""
    sr = 16000
    cmd = [
        ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", source_path, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None

    buf = np.zeros(0, dtype=np.float32)
    base_offset = 0
    chunk_samples = int(45.0 * sr)
    overlap_samples = int(2.0 * sr)

    active_audio_secs: set[int] = set()
    cry_secs: set[int] = set()
    max_cvoc = 0.0
    max_laugh = 0.0
    max_cry = 0.0
    bursts: list[dict[str, Any]] = []
    seen_b: list[float] = []
    y_idx = yamnet.idx

    def _scan_yamnet_chunk(pcm_chunk: np.ndarray, offset_samp: int, limit_samp: int) -> None:
        nonlocal max_cvoc, max_laugh, max_cry
        # score_window zero-pads the last partial window; retain at least 250 ms of signal.
        for pos in range(0, min(limit_samp, len(pcm_chunk) - sr // 4 + 1), sr):
            sc = yamnet.score_window(pcm_chunk[pos : pos + yamnet.window_samples])
            cvoc = float(sc[y_idx["child"]] + sc[y_idx["babble"]])
            laugh = max((float(sc[i]) for i in y_idx["laughter"]), default=0.0)
            cry = max(float(sc[y_idx["cry"]]), float(sc[y_idx["sob"]]))
            sp = max(float(sc[y_idx["speech"]]), cvoc)
            sec_i = int((offset_samp + pos) // sr)
            max_cvoc = max(max_cvoc, cvoc)
            max_laugh = max(max_laugh, laugh)
            max_cry = max(max_cry, cry)
            if sp >= 0.30 or laugh >= 0.28 or cry >= 0.28 or float(sc[y_idx["scream"]]) >= 0.25:
                active_audio_secs.add(sec_i)
            if cry >= 0.30 or cvoc >= 0.32 or laugh >= 0.30:
                cry_secs.add(sec_i)

    def _add_bursts(items: list[dict[str, Any]]) -> None:
        for b in items:
            if len(bursts) >= max_bursts:
                return
            if any(abs(b["burst_start_s"] - s0) < 0.65 for s0 in seen_b):
                continue
            seen_b.append(b["burst_start_s"])
            bursts.append(b)

    try:
        read_bytes = int(20.0 * sr) * 4
        while True:
            # Limit a stalled read, not the duration of a healthy long recording or local inference.
            timed_out = threading.Event()
            timer = threading.Timer(150.0, lambda: (timed_out.set(), proc.kill()))
            timer.daemon = True
            timer.start()
            try:
                raw = proc.stdout.read(read_bytes)
            finally:
                timer.cancel()
                timer.join()
            if timed_out.is_set():
                raise TimeoutError("audio_decode_stalled")
            if not raw:
                break
            chunk = np.frombuffer(raw[: (len(raw) // 4) * 4], dtype="<f4").copy()
            buf = chunk if buf.size == 0 else np.concatenate([buf, chunk])
            if len(buf) >= chunk_samples:
                safe_samp = len(buf) - overlap_samples
                _scan_yamnet_chunk(buf, base_offset, safe_samp)
                if scan_bursts and len(bursts) < max_bursts:
                    _add_bursts(extract_infant_bursts_from_pcm(
                        buf, base_offset, cam, source_rel, yamnet,
                        max_s_limit=(base_offset + safe_samp) / float(sr),
                    ))
                base_offset += safe_samp
                buf = buf[safe_samp:]
        if proc.wait(timeout=150.0) != 0:
            raise RuntimeError(f"audio_decode_failed (ffmpeg exit {proc.returncode})")
        if base_offset + buf.size == 0:
            raise RuntimeError("audio_decode_empty")
        if buf.size >= 4000:
            _scan_yamnet_chunk(buf, base_offset, len(buf))
            if scan_bursts and len(bursts) < max_bursts:
                _add_bursts(extract_infant_bursts_from_pcm(buf, base_offset, cam, source_rel, yamnet, None))
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    return {
        "active_secs": sorted(active_audio_secs),
        "cry_or_baby_secs": sorted(cry_secs),
        "max_child_voc": round(max_cvoc, 3),
        "max_laughter": round(max_laugh, 3),
        "max_cry": round(max_cry, 3),
        "bursts": bursts,
    }


def segments_from_mask(active: np.ndarray, duration: float) -> list[list[float]]:
    segs: list[list[float]] = []
    for i in np.flatnonzero(active):
        s, e = max(0.0, float(i) - PAD_BEFORE), min(duration, float(i) + 1.0 + PAD_AFTER)
        if segs and s - segs[-1][1] <= MERGE_GAP:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    out: list[list[float]] = []
    for s, e in segs:
        if e - s < MIN_SEG:
            e = min(duration, s + MIN_SEG)
        out.append([round(s, 2), round(e, 2)])
    return out


def compute_bedside_static_mask(
    boxes: list[list[tuple[float, float, float, float, float]]],
    bmot: np.ndarray,
    min_run_s: int = 12,
) -> np.ndarray:
    """Identify seconds where YOLO falsely latches onto static crib blankets/pillows (IoU >= 0.82, low motion)."""
    n = len(boxes)
    suppressed = np.zeros(n, dtype=bool)
    run_start = 0
    prev_top = None
    for i in range(n):
        good = [b for b in boxes[i] if b[0] >= PERSON_CONF]
        top = good[0] if good else None
        if top is not None and prev_top is not None and box_iou(top, prev_top) >= 0.82 and bmot[i] < 0.018:
            pass
        else:
            if i - run_start >= min_run_s and prev_top is not None:
                suppressed[run_start:i] = True
            run_start = i
        prev_top = top
    if n - run_start >= min_run_s and prev_top is not None:
        suppressed[run_start:n] = True
    return suppressed
