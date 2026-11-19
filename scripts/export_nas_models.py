"""Export lightweight NAS-ready ONNX, TFLite, and CTranslate2 INT8 models and verify equivalence."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT_MODELS = PROJECT / "nas-growth-v3" / "models"


def _letterbox_640(bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Ultralytics-compatible letterbox resize to 640x640, returning (blob_nchw, r, pad_x, pad_y)."""
    h, w = bgr.shape[:2]
    r = min(640.0 / w, 640.0 / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (640 - nw) / 2.0
    pad_h = (640 - nh) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    nchw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])
    return nchw, r, float(left), float(top)


def _decode_yolo_onnx(raw: np.ndarray, orig_w: int, orig_h: int, r: float, pad_x: float, pad_y: float, conf_thr: float = 0.20):
    """Parse ONNX detection output handling both (1, N, 6) end2end and (1, 84, 8400) layouts."""
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    boxes = []
    if arr.ndim == 2 and arr.shape[1] == 6:
        # End-to-end NMS-free layout: [x1, y1, x2, y2, score, class_id]
        for x1, y1, x2, y2, sc, cls_id in arr:
            if int(round(cls_id)) != 0 or float(sc) < conf_thr:
                continue
            nx1 = max(0.0, min(1.0, (float(x1) - pad_x) / r / orig_w))
            ny1 = max(0.0, min(1.0, (float(y1) - pad_y) / r / orig_h))
            nx2 = max(0.0, min(1.0, (float(x2) - pad_x) / r / orig_w))
            ny2 = max(0.0, min(1.0, (float(y2) - pad_y) / r / orig_h))
            if nx2 > nx1 and ny2 > ny1:
                boxes.append((float(sc), nx1, ny1, nx2, ny2))
    else:
        # Fallback YOLOv8-style (84, 8400) -> [cx, cy, w, h, cls0..cls79]
        if arr.shape[0] in (84, 85) and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        scores = arr[:, 4]
        mask = scores >= conf_thr
        cand = arr[mask]
        if cand.size:
            raw_boxes = []
            confs = []
            for row in cand:
                cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                x1, y1 = (cx - bw / 2.0 - pad_x) / r, (cy - bh / 2.0 - pad_y) / r
                w_o, h_o = bw / r, bh / r
                raw_boxes.append([x1, y1, w_o, h_o])
                confs.append(float(row[4]))
            idxs = cv2.dnn.NMSBoxes(raw_boxes, confs, conf_thr, 0.45)
            for idx in (np.array(idxs).flatten().tolist() if len(idxs) else []):
                x1, y1, w_o, h_o = raw_boxes[idx]
                nx1 = max(0.0, min(1.0, x1 / orig_w))
                ny1 = max(0.0, min(1.0, y1 / orig_h))
                nx2 = max(0.0, min(1.0, (x1 + w_o) / orig_w))
                ny2 = max(0.0, min(1.0, (y1 + h_o) / orig_h))
                boxes.append((confs[idx], nx1, ny1, nx2, ny2))
    boxes.sort(key=lambda b: -b[0])
    return boxes


def export_and_verify_yolo() -> None:
    import onnxruntime as ort
    from ultralytics import YOLO

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    pt_path = PROJECT / ".data" / "models" / "yolo26s.pt"
    onnx_dst = OUT_MODELS / "yolo26s.onnx"

    print(f"[Export] Loading {pt_path}...", flush=True)
    y_model = YOLO(str(pt_path))
    exported = Path(y_model.export(format="onnx", imgsz=640, opset=17, simplify=True, half=False, dynamic=False))
    if exported.resolve() != onnx_dst.resolve():
        shutil.copy2(exported, onnx_dst)
    print(f"[Export] Saved {onnx_dst} ({onnx_dst.stat().st_size / 1e6:.2f} MB)", flush=True)

    sess = ort.InferenceSession(str(onnx_dst), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    print(f"[Export] ONNX input={in_meta.name} {in_meta.shape}, output={out_meta.name} {out_meta.shape}", flush=True)

    sample_jpg = sorted((PROJECT / "output" / "growth-archive-pilot-20261117" / "v2" / "photos").glob("full_*.jpg"))[0]
    im = cv2.imread(str(sample_jpg))
    oh, ow = im.shape[:2]

    # Reference PyTorch prediction on CPU
    pt_res = y_model.predict([im], imgsz=640, conf=0.20, classes=[0], verbose=False, device="cpu")[0]
    pt_boxes = []
    for b in pt_res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        pt_boxes.append((round(float(b.conf[0]), 3), round(x1 / ow, 3), round(y1 / oh, 3), round(x2 / ow, 3), round(y2 / oh, 3)))

    nchw, r, px, py = _letterbox_640(im)
    t0 = time.time()
    out = sess.run([out_meta.name], {in_meta.name: nchw})[0]
    dt_ms = (time.time() - t0) * 1000.0
    ort_boxes = [(round(c, 3), round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
                 for c, x1, y1, x2, y2 in _decode_yolo_onnx(out, ow, oh, r, px, py, 0.20)]

    print(f"[Verify YOLO] ONNX raw shape={out.shape}, CPU infer={dt_ms:.1f}ms", flush=True)
    print(f"[Verify YOLO] PT  boxes: {pt_boxes}", flush=True)
    print(f"[Verify YOLO] ORT boxes: {ort_boxes}", flush=True)
    if not ort_boxes or abs(pt_boxes[0][0] - ort_boxes[0][0]) > 0.05:
        raise RuntimeError("YOLO ONNX box verification mismatch!")


def copy_yamnet() -> None:
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    for fn in ("yamnet.tflite", "yamnet_class_map.csv"):
        src = PROJECT / ".data" / "models" / fn
        dst = OUT_MODELS / fn
        shutil.copy2(src, dst)
        print(f"[Export] Copied {dst.name} ({dst.stat().st_size / 1e6:.2f} MB)", flush=True)


def convert_and_verify_whisper() -> None:
    import ctranslate2
    sys.path.insert(0, str(PROJECT / "src"))
    from cctv_ai.media import resolve_ffmpeg

    ff = resolve_ffmpeg(None)
    src_dir = PROJECT / ".data" / "models" / "whisper-large-v3-turbo"
    dst_dir = OUT_MODELS / "whisper-v3-turbo-int8"
    if not (dst_dir / "model.bin").is_file():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        print(f"[Export] Converting {src_dir} -> {dst_dir} (int8)...", flush=True)
        conv = ctranslate2.converters.TransformersConverter(
            str(src_dir),
            copy_files=["tokenizer.json", "preprocessor_config.json"],
        )
        conv.convert(str(dst_dir), quantization="int8", force=True)
    sz_mb = sum(f.stat().st_size for f in dst_dir.rglob("*") if f.is_file()) / 1e6
    print(f"[Export] Whisper INT8 dir size: {sz_mb:.1f} MB", flush=True)

    # Optional smoke test on local infant speech clips (JSON list via CCTV_WHISPER_TEST_CASES env var)
    cases_json = os.environ.get("CCTV_WHISPER_TEST_CASES", "").strip()
    sample_root = os.environ.get("CCTV_SAMPLE_VIDEO_ROOT", r"\\192.168.1.100\cctv")
    cases = json.loads(cases_json) if cases_json else []
    if not cases:
        print(f"[Export] Whisper INT8 dir ready (set CCTV_WHISPER_TEST_CASES / CCTV_SAMPLE_VIDEO_ROOT={sample_root} to run real-clip RSS check).", flush=True)
        return
    probe_code = rf"""
import ctypes, ctypes.wintypes, json, math, os, subprocess, sys
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
from tokenizers import Tokenizer

def rss_mb():
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return round(pmc.WorkingSetSize / (1024*1024), 1), round(pmc.PeakWorkingSetSize / (1024*1024), 1)
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            v = round(int(ln.split()[1]) / 1024.0, 1)
            return v, v
    return 0.0, 0.0

t_dir = {str(dst_dir)!r}
ff = {str(ff)!r}
fw = WhisperModel(t_dir, device="cpu", compute_type="int8", cpu_threads=2, local_files_only=True)
print("RSS after WhisperModel load:", rss_mb(), "MB")

tok = Tokenizer.from_file(t_dir + "/tokenizer.json")
prompt = [
    tok.token_to_id("<|startoftranscript|>"),
    tok.token_to_id("<|zh|>"),
    tok.token_to_id("<|transcribe|>"),
    tok.token_to_id("<|notimestamps|>"),
]

MA_WORDS = ["妈妈", "媽媽", "麻麻", "妈咪", "妈", "媽", "麻", "马", "吗", "嘛"]
BA_WORDS = ["爸爸", "粑粑", "爸", "吧", "叭", "把", "八", "巴"]
FOIL_WORDS = ["啊", "呀", "哦", "嗯", "咿", "哇", "哈", "哒", "大", "嘎", "呐", "那", "啦", "哎", "好", "我", "到"]
words = MA_WORDS + BA_WORDS + FOIL_WORDS
word_toks = {{w: tok.encode(w, add_special_tokens=False).ids for w in words}}

for expected, src, b_s, b_e in {cases!r}:
    s0 = max(0.0, b_s - 0.10)
    dur_s = (b_e + 0.12) - s0
    r = subprocess.run([
        ff, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{{s0:.3f}}", "-t", f"{{dur_s:.3f}}", "-i", src,
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1"
    ], capture_output=True)
    arr = np.frombuffer(r.stdout[:(len(r.stdout)//4)*4], dtype="<f4").copy()
    n_in, n_out = min(int(0.045*16000), arr.size//3), min(int(0.060*16000), arr.size//3)
    arr[:n_in] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in, dtype=np.float32))
    arr[-n_out:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n_out, dtype=np.float32))
    arr *= 0.65 / float(np.max(np.abs(arr)))
    lead = np.zeros(int(0.25*16000), dtype=np.float32)
    trail = np.zeros(max(int(0.25*16000), int(1.50*16000) - lead.size - arr.size), dtype=np.float32)
    wave = np.concatenate([lead, arr, trail])

    segs, _ = fw.transcribe(wave, language="zh", task="transcribe", beam_size=5, vad_filter=False, without_timestamps=True)
    txt = "".join(s.text for s in segs).strip()

    feat = fw.feature_extractor(wave)
    mel = np.zeros((feat.shape[0], 3000), dtype=np.float32)
    n_copy = min(3000, feat.shape[1])
    mel[:, :n_copy] = feat[:, :n_copy]
    sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(mel[None, ...]))
    enc = fw.model.encode(sv)

    scores = {{}}
    for w in words:
        tids = word_toks[w]
        lps = []
        cur_prompt = list(prompt)
        for tid in tids:
            res = fw.model.generate(enc, [cur_prompt], beam_size=1, max_length=len(cur_prompt)+1, suppress_tokens=[], suppress_blank=False, return_logits_vocab=True)[0]
            lv = res.logits[0]
            arr_l = np.array(lv.to_numpy() if hasattr(lv, "to_numpy") else lv, dtype=np.float64)
            logits = arr_l[0] if arr_l.ndim == 2 else arr_l
            logits -= np.max(logits)
            log_z = math.log(np.sum(np.exp(logits)))
            lps.append(float(logits[tid] - log_z))
            cur_prompt.append(tid)
        scores[w] = float(np.mean(lps))

    ma_lp = max(scores[w] for w in MA_WORDS)
    ba_lp = max(scores[w] for w in BA_WORDS)
    foil_lp = max(scores[w] for w in FOIL_WORDS)
    m3 = max(ma_lp, ba_lp, foil_lp)
    lz = m3 + math.log(math.exp(ma_lp-m3)+math.exp(ba_lp-m3)+math.exp(foil_lp-m3))
    print(f"CASE expected={{expected!r}} free_txt={{txt!r}} ma_lp={{ma_lp:.3f}} ba_lp={{ba_lp:.3f}} foil_lp={{foil_lp:.3f}} ma_post={{math.exp(ma_lp-lz):.3f}} ba_post={{math.exp(ba_lp-lz):.3f}} rss={{rss_mb()}}")
"""
    p = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True)
    print(p.stdout)
    if p.stderr:
        print("STDERR:", p.stderr[-500:])
    if p.returncode != 0:
        raise RuntimeError(f"Whisper INT8 verification failed ({p.returncode})")


if __name__ == "__main__":
    export_and_verify_yolo()
    copy_yamnet()
    convert_and_verify_whisper()
