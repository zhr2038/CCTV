"""Closed-set family face recognition (SCRFD-500M + ArcFace MBF ONNX), VLM reference sheet & kinship caption guard."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .db import ArchiveDB


DEFAULT_FAMILY_MEMBERS: list[dict[str, Any]] = [
    {"member_id": "mama", "label_zh": "妈妈", "gender": "female", "role_group": "parent", "notes": "宝宝妈妈", "sort_order": 10},
    {"member_id": "baba", "label_zh": "爸爸", "gender": "male", "role_group": "parent", "notes": "宝宝爸爸", "sort_order": 20},
    {"member_id": "nainai", "label_zh": "奶奶", "gender": "female", "role_group": "grandparent", "notes": "宝宝奶奶", "sort_order": 30},
    {"member_id": "yeye", "label_zh": "爷爷", "gender": "male", "role_group": "grandparent", "notes": "宝宝爷爷", "sort_order": 40},
    {"member_id": "laolao", "label_zh": "姥姥", "gender": "female", "role_group": "grandparent", "notes": "宝宝姥姥", "sort_order": 50},
    {"member_id": "laoye", "label_zh": "姥爷", "gender": "male", "role_group": "grandparent", "notes": "宝宝姥爷", "sort_order": 60},
    {
        "member_id": "yuesao",
        "label_zh": "月嫂",
        "gender": "female",
        "role_group": "caregiver",
        "active_from": "2025-12-17",
        "active_until": "2026-01-31",
        "notes": "宝宝首月照护阿姨",
        "sort_order": 70,
    },
    {"member_id": "xiaoyi", "label_zh": "小姨", "gender": "female", "role_group": "relative", "notes": "宝宝小姨（偶尔来访）", "sort_order": 80},
]

KINSHIP_ROLES: tuple[str, ...] = ("妈妈", "爸爸", "奶奶", "爷爷", "姥姥", "姥爷", "月嫂", "小姨")
FEMALE_ROLES = frozenset({"妈妈", "奶奶", "姥姥", "月嫂", "小姨"})
MALE_ROLES = frozenset({"爸爸", "爷爷", "姥爷"})

# Synonymous words that VLM may emit for each canonical family role
ROLE_ALIAS_MAP: dict[str, str] = {
    "妈妈": "妈妈",
    "母亲": "妈妈",
    "宝妈": "妈妈",
    "妈咪": "妈妈",
    "爸爸": "爸爸",
    "父亲": "爸爸",
    "宝爸": "爸爸",
    "奶奶": "奶奶",
    "祖母": "奶奶",
    "爷爷": "爷爷",
    "祖父": "爷爷",
    "姥姥": "姥姥",
    "外婆": "姥姥",
    "姥爷": "姥爷",
    "外公": "姥爷",
    "月嫂": "月嫂",
    "育儿嫂": "月嫂",
    "小姨": "小姨",
    "小姨妈": "小姨",
    "姨妈": "小姨",
}

ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Protect baby's own speech utterances like 「叫了一声“妈妈”」「冒出短短一声「爸」」 from adult kinship substitution
_SPEECH_PROTECT_RE = re.compile(
    r"(?:"
    r"(?:叫|喊|唤|学说)(?:了)?(?:一声|出)?[「“\"'《]?(?:妈妈|妈|爸爸|爸)[」”\"'》]?"
    r"|(?:冒出|发出|吐出|喊出|叫出)[^，。！？\n]{0,10}[「“\"'《]?(?:妈妈|妈|爸爸|爸)[」”\"'》]?"
    r"|(?:一声|清晰(?:的)?)[「“\"'《]?(?:妈妈|妈|爸爸|爸)[」”\"'》]?"
    r"|[「“\"'](?:妈妈|妈|爸爸|爸)[」”\"'](?:的)?(?:声|音|童声|发音|字眼)?"
    r")"
)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return v
    return (v / n).astype(np.float32)


def _cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size == 0 or va.size != vb.size:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _decode_image_bgr(raw_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG/WebP bytes with EXIF orientation applied and return OpenCV BGR uint8 array."""
    if not raw_bytes:
        return None
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im_raw:
            im = ImageOps.exif_transpose(im_raw).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def is_reliable_face_pose(
    face: dict[str, Any] | None,
    *,
    min_eye_dx_ratio: float = 0.20,
    min_mouth_dy_ratio: float = 0.18,
) -> bool:
    """Reject collapsed extreme-profile / back-of-head detections where 5-point ArcFace alignment forms hub false matches."""
    if not isinstance(face, dict):
        return False
    box = face.get("box_px")
    kps = face.get("kps")
    if not (isinstance(box, (list, tuple)) and len(box) == 4 and isinstance(kps, (list, tuple)) and len(kps) == 5):
        return False
    bw = float(box[2]) - float(box[0])
    bh = float(box[3]) - float(box[1])
    if bw <= 1.0 or bh <= 1.0:
        return False
    try:
        le, re_pt, nose, lm, rm = kps
        eye_dx_ratio = abs(float(re_pt[0]) - float(le[0])) / bw
        eye_y = (float(le[1]) + float(re_pt[1])) * 0.5
        mouth_y = (float(lm[1]) + float(rm[1])) * 0.5
        mouth_dy_ratio = (mouth_y - eye_y) / bh
        min_eye_y = min(float(le[1]), float(re_pt[1]))
        max_mouth_y = max(float(lm[1]), float(rm[1]))
        nose_y = float(nose[1])
    except (TypeError, ValueError, IndexError):
        return False
    if eye_dx_ratio < float(min_eye_dx_ratio):
        return False
    if mouth_dy_ratio < float(min_mouth_dy_ratio):
        return False
    if nose_y < (min_eye_y - 0.12 * bh) or nose_y > (max_mouth_y + 0.12 * bh):
        return False
    return True


def is_member_active_on_day(member: dict[str, Any], day: str | None) -> bool:
    if not member.get("enabled", True):
        return False
    d = str(day or "").strip()
    if not d:
        return True
    af = str(member.get("active_from") or "").strip()
    au = str(member.get("active_until") or "").strip()
    if af and d < af:
        return False
    if au and d > au:
        return False
    return True


def _resolve_model_path(cfg_path: str | None, default_container_path: str, filename: str) -> Path:
    raw = str(cfg_path or "").strip() or default_container_path
    p = Path(raw)
    if p.is_file():
        return p
    if raw == default_container_path:
        local_cand = Path(__file__).resolve().parents[1] / "models" / "face" / filename
        if local_cand.is_file():
            return local_cand
    return p


class ScrfdArcFaceEngine:
    """Thread-safe lazy CPU ONNX Runtime face detector (SCRFD-500M) + recognizer (ArcFace MBF)."""

    def __init__(self, det_onnx: str | Path, rec_onnx: str | Path) -> None:
        self.det_path = Path(det_onnx)
        self.rec_path = Path(rec_onnx)
        self._lock = threading.Lock()
        self._det_sess: Any = None
        self._rec_sess: Any = None
        self._det_in_name: str = ""
        self._rec_in_name: str = ""
        self._rec_out_name: str = ""
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self.det_path.is_file() and self.rec_path.is_file()

    def _ensure_loaded(self) -> bool:
        if self._det_sess is not None and self._rec_sess is not None:
            return True
        with self._lock:
            if self._det_sess is not None and self._rec_sess is not None:
                return True
            if not self.available:
                self._load_error = "face_models_missing"
                return False
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                det_sess = ort.InferenceSession(str(self.det_path), sess_options=opts, providers=["CPUExecutionProvider"])
                rec_sess = ort.InferenceSession(str(self.rec_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self._det_in_name = det_sess.get_inputs()[0].name
                self._rec_in_name = rec_sess.get_inputs()[0].name
                self._rec_out_name = rec_sess.get_outputs()[0].name
                self._det_sess = det_sess
                self._rec_sess = rec_sess
                self._load_error = None
                return True
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                return False

    def _anchor_centers(self, h: int, w: int, stride: int, num_anchors: int = 2) -> np.ndarray:
        key = (h, w, stride)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached
        anchor_centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
        self._center_cache[key] = anchor_centers
        return anchor_centers

    @staticmethod
    def _nms(dets: np.ndarray, iou_thr: float = 0.40) -> list[int]:
        if dets.shape[0] == 0:
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            iw = np.maximum(0.0, xx2 - xx1 + 1.0)
            ih = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = iw * ih
            ovr = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def detect_faces(
        self,
        bgr: np.ndarray,
        *,
        score_thr: float = 0.40,
        input_size: tuple[int, int] = (640, 640),
        min_face_px: int = 20,
    ) -> list[dict[str, Any]]:
        """Detect faces and 5 landmarks using SCRFD-500M."""
        if bgr is None or bgr.size == 0 or not self._ensure_loaded():
            return []
        oh, ow = bgr.shape[:2]
        in_w, in_h = input_size
        im_ratio = float(oh) / float(max(1, ow))
        model_ratio = float(in_h) / float(max(1, in_w))
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = max(1, int(new_h / im_ratio))
        else:
            new_w = in_w
            new_h = max(1, int(new_w * im_ratio))
        det_scale = float(new_h) / float(max(1, oh))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / 128.0,
            (in_w, in_h),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )

        with self._lock:
            net_outs = self._det_sess.run(None, {self._det_in_name: blob})

        strides = [8, 16, 32]
        num_anchors = 2
        by_stride: dict[int, dict[str, np.ndarray]] = {}
        for out_arr in net_outs:
            arr = np.asarray(out_arr)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                continue
            n_rows, ch = arr.shape
            for st in strides:
                fh, fw = in_h // st, in_w // st
                if n_rows == fh * fw * num_anchors:
                    slot = by_stride.setdefault(st, {})
                    if ch == 1:
                        slot["scores"] = arr.reshape(-1)
                    elif ch == 4:
                        slot["bboxes"] = arr
                    elif ch == 10:
                        slot["kps"] = arr
                    break

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []
        for st in strides:
            part = by_stride.get(st)
            if not part or "scores" not in part or "bboxes" not in part:
                continue
            sc = part["scores"]
            if sc.size and (sc.min() < 0.0 or sc.max() > 1.0):
                sc = 1.0 / (1.0 + np.exp(-np.clip(sc, -30.0, 30.0)))
            pos_inds = np.where(sc >= float(score_thr))[0]
            if pos_inds.size == 0:
                continue
            fh, fw = in_h // st, in_w // st
            centers = self._anchor_centers(fh, fw, st, num_anchors=num_anchors)
            b_pred = part["bboxes"] * float(st)
            cx, cy = centers[:, 0], centers[:, 1]
            boxes = np.stack(
                [cx - b_pred[:, 0], cy - b_pred[:, 1], cx + b_pred[:, 2], cy + b_pred[:, 3]],
                axis=-1,
            ) / det_scale
            scores_list.append(sc[pos_inds])
            bboxes_list.append(boxes[pos_inds])
            if "kps" in part:
                k_pred = part["kps"] * float(st)
                kpts = np.empty((centers.shape[0], 10), dtype=np.float32)
                for j in range(5):
                    kpts[:, 2 * j] = (cx + k_pred[:, 2 * j]) / det_scale
                    kpts[:, 2 * j + 1] = (cy + k_pred[:, 2 * j + 1]) / det_scale
                kpss_list.append(kpts[pos_inds])

        if not scores_list:
            return []

        scores_all = np.concatenate(scores_list, axis=0)
        bboxes_all = np.concatenate(bboxes_list, axis=0)
        kpss_all = np.concatenate(kpss_list, axis=0) if kpss_list else None
        order = scores_all.argsort()[::-1]
        pre_dets = np.hstack((bboxes_all, scores_all[:, None])).astype(np.float32)[order, :]
        if kpss_all is not None:
            kpss_all = kpss_all[order, :]

        keep = self._nms(pre_dets, iou_thr=0.40)
        results: list[dict[str, Any]] = []
        for idx in keep:
            x1, y1, x2, y2, sc_val = pre_dets[idx].tolist()
            cx1 = max(0.0, min(float(ow - 1), x1))
            cy1 = max(0.0, min(float(oh - 1), y1))
            cx2 = max(cx1 + 1.0, min(float(ow), x2))
            cy2 = max(cy1 + 1.0, min(float(oh), y2))
            bw, bh = cx2 - cx1, cy2 - cy1
            if bw < min_face_px or bh < min_face_px:
                continue
            kps = kpss_all[idx].reshape((5, 2)).tolist() if kpss_all is not None else []
            results.append({
                "score": round(float(sc_val), 4),
                "box_px": [round(cx1, 1), round(cy1, 1), round(cx2, 1), round(cy2, 1)],
                "box_norm": [
                    round(cx1 / ow, 4),
                    round(cy1 / oh, 4),
                    round(cx2 / ow, 4),
                    round(cy2 / oh, 4),
                ],
                "kps": [[round(float(kx), 2), round(float(ky), 2)] for kx, ky in kps],
            })
        return results

    def align_face_112(self, bgr: np.ndarray, kps: list[list[float]] | np.ndarray) -> np.ndarray | None:
        pts = np.asarray(kps, dtype=np.float32)
        if pts.shape != (5, 2):
            return None
        M, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_DST_112, method=cv2.LMEDS)
        if M is None:
            M, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_DST_112, method=cv2.RANSAC)
        if M is None:
            return None
        return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0.0)

    def extract_embedding(self, bgr: np.ndarray, kps: list[list[float]] | np.ndarray) -> list[float] | None:
        if not self._ensure_loaded():
            return None
        aligned = self.align_face_112(bgr, kps)
        if aligned is None:
            return None
        flipped = cv2.flip(aligned, 1)
        batch = cv2.dnn.blobFromImages(
            [aligned, flipped],
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        with self._lock:
            # Handle static batch_size=1 ONNX graphs cleanly by running two single inferences
            e0 = self._rec_sess.run([self._rec_out_name], {self._rec_in_name: batch[0:1]})[0][0]
            e1 = self._rec_sess.run([self._rec_out_name], {self._rec_in_name: batch[1:2]})[0][0]
        feat = _norm_vec(np.asarray(e0, dtype=np.float32) + np.asarray(e1, dtype=np.float32))
        if feat.size < 64 or float(np.linalg.norm(feat)) <= 1e-6:
            return None
        return [round(float(v), 6) for v in feat.tolist()]

    @staticmethod
    def crop_head_portrait_jpeg(bgr: np.ndarray, box_px: list[float] | None, size: int = 240, quality: int = 90) -> bytes:
        """Crop head + hair + shoulders portrait for human UI and VLM visual reference sheets."""
        oh, ow = bgr.shape[:2]
        if box_px and len(box_px) == 4 and box_px[2] > box_px[0] and box_px[3] > box_px[1]:
            x1, y1, x2, y2 = [float(v) for v in box_px]
            bw, bh = x2 - x1, y2 - y1
            nx1 = max(0, int(math.floor(x1 - bw * 0.48)))
            nx2 = min(ow, int(math.ceil(x2 + bw * 0.48)))
            ny1 = max(0, int(math.floor(y1 - bh * 0.55)))
            ny2 = min(oh, int(math.ceil(y2 + bh * 0.55)))
        else:
            side = min(ow, oh)
            nx1 = max(0, (ow - side) // 2)
            ny1 = max(0, (oh - side) // 2)
            nx2 = nx1 + side
            ny2 = ny1 + side
        crop = bgr[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            crop = bgr
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im = ImageOps.pad(im, (size, size), color=(28, 24, 22), method=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()


def match_embedding(
    emb: list[float] | np.ndarray,
    gallery: list[dict[str, Any]],
    *,
    high_thr: float = 0.40,
    med_thr: float = 0.32,
    min_margin: float = 0.06,
) -> dict[str, Any]:
    """Match one 512-d face embedding against enrolled active family members."""
    q = np.asarray(emb, dtype=np.float32)
    if q.size < 64 or float(np.linalg.norm(q)) <= 1e-6:
        return {"status": "unknown", "label_zh": None, "member_id": None, "best_score": 0.0, "margin": 0.0}
    q = _norm_vec(q)

    scored: list[dict[str, Any]] = []
    for member in gallery:
        if not member.get("enabled", True):
            continue
        embs = [
            _norm_vec(np.asarray(s["embedding"], dtype=np.float32))
            for s in (member.get("samples") or [])
            if s.get("has_embedding") and isinstance(s.get("embedding"), list) and len(s["embedding"]) == q.size
        ]
        if not embs:
            continue
        sims = sorted((float(np.dot(q, e)) for e in embs), reverse=True)
        top2_mean = sum(sims[:2]) / float(min(2, len(sims)))
        centroid = _norm_vec(np.mean(np.stack(embs, axis=0), axis=0))
        centroid_sim = float(np.dot(q, centroid))
        score = 0.65 * top2_mean + 0.35 * centroid_sim
        scored.append({
            "member_id": member["member_id"],
            "label_zh": member["label_zh"],
            "gender": member.get("gender", "unknown"),
            "score": round(score, 4),
            "max_sample_sim": round(sims[0], 4),
        })

    if not scored:
        return {"status": "unknown", "label_zh": None, "member_id": None, "best_score": 0.0, "margin": 0.0}

    scored.sort(key=lambda x: -x["score"])
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = round(best["score"] - (second["score"] if second else 0.0), 4)

    if best["score"] >= high_thr and (second is None or margin >= min_margin):
        status = "high"
    elif best["score"] >= med_thr and (second is None or margin >= min_margin * 0.5):
        status = "medium"
    else:
        status = "unknown"

    return {
        "status": status,
        "member_id": best["member_id"] if status != "unknown" else None,
        "label_zh": best["label_zh"] if status != "unknown" else None,
        "gender": best["gender"] if status != "unknown" else "unknown",
        "best_label_zh": best["label_zh"],
        "best_member_id": best["member_id"],
        "best_score": best["score"],
        "margin": margin,
        "runner_up": second["label_zh"] if second else None,
        "runner_up_score": second["score"] if second else 0.0,
    }


def sanitize_caption_kinship(
    text: str,
    *,
    verified_local_roles: set[str] | frozenset[str] = frozenset(),
    verified_vlm_roles: set[str] | frozenset[str] = frozenset(),
    active_roles: set[str] | frozenset[str] = frozenset(KINSHIP_ROLES),
    has_enrolled_ref: bool = False,
    strict_unverified: bool = True,
    day_confirmed_roles: set[str] | frozenset[str] | None = None,
) -> str:
    """Protect baby speech quotes while fixing/neutralizing hallucinated kinship words like 奶奶->妈妈."""
    raw = str(text or "")
    if not raw.strip():
        return raw

    # 1. Stash baby's own speech utterances so 「叫了一声“妈妈”」「冒出短短一声「爸」」 are never touched
    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        tok = f"\x00SPK{len(stashed)}\x00"
        stashed.append(m.group(0))
        return tok

    work = _SPEECH_PROTECT_RE.sub(_stash, raw)

    local_roles = {r for r in verified_local_roles if r in active_roles}
    vlm_roles = {r for r in verified_vlm_roles if r in active_roles} if has_enrolled_ref else set()
    day_confirmed = {r for r in (day_confirmed_roles or ()) if r in active_roles}

    # Same-group elder disambiguation using whole-day verified presence (e.g. 奶奶 present, 姥姥 absent -> map 姥姥->奶奶)
    day_promoted: set[str] = set()
    for present_role, absent_role, absent_pat in (
        ("奶奶", "姥姥", r"(?:姥姥|外婆)"),
        ("姥姥", "奶奶", r"(?:奶奶|祖母)"),
        ("爷爷", "姥爷", r"(?:姥爷|外公)"),
        ("姥爷", "爷爷", r"(?:爷爷|祖父)"),
    ):
        if day_confirmed and present_role in day_confirmed and absent_role not in day_confirmed and absent_role not in local_roles:
            vlm_roles.discard(absent_role)
            if has_enrolled_ref and present_role in active_roles:
                work = re.sub(absent_pat, present_role, work)
                day_promoted.add(present_role)
            else:
                work = re.sub(absent_pat, "家人", work)

    # Drop VLM same-gender hallucination when local ArcFace strongly identified another person of that gender
    local_females = local_roles & FEMALE_ROLES
    local_males = local_roles & MALE_ROLES
    if local_females and "妈妈" not in local_females:
        vlm_roles.discard("妈妈")
    if local_males and "爸爸" not in local_males:
        vlm_roles.discard("爸爸")

    # 2. Direct contradiction override when local ArcFace gives unambiguous gender role (e.g. 奶奶 instead of 妈妈)
    if local_females and "妈妈" not in local_females:
        repl_f = next(iter(local_females)) if len(local_females) == 1 else "家人"
        work = re.sub(r"(?:妈妈|宝妈|妈咪|母亲)", repl_f, work)
    if local_males and "爸爸" not in local_males:
        repl_m = next(iter(local_males)) if len(local_males) == 1 else "家人"
        work = re.sub(r"(?:爸爸|宝爸|父亲)", repl_m, work)

    allowed_roles = local_roles | vlm_roles | day_promoted

    # 3. Compound parent terms (爸爸妈妈 / 爸妈 / 父母)
    def _replace_compound(m: re.Match[str]) -> str:
        word = m.group(0)
        if not strict_unverified and has_enrolled_ref:
            # Even when non-strict, never allow a blocked gender parent term
            has_f = ("妈妈" in active_roles) and not (local_females and "妈妈" not in local_females)
            has_m = ("爸爸" in active_roles) and not (local_males and "爸爸" not in local_males)
            if has_f and has_m:
                return word
        if "爸爸" in allowed_roles and "妈妈" in allowed_roles:
            return word
        if "爸爸" in allowed_roles:
            return "爸爸" if not local_females else f"爸爸和{next(iter(local_females))}" if len(local_females) == 1 else "家人"
        if "妈妈" in allowed_roles:
            return "妈妈" if not local_males else f"妈妈和{next(iter(local_males))}" if len(local_males) == 1 else "家人"
        return "家人"

    if strict_unverified or local_females or local_males:
        work = re.sub(r"(?:爸爸妈妈|爸妈|父母)", _replace_compound, work)

    # 4. Individual kinship terms & synonyms (check longer words before shorter ones)
    alias_keys = sorted(ROLE_ALIAS_MAP.keys(), key=len, reverse=True)
    alias_pat = re.compile("|".join(re.escape(k) for k in alias_keys))

    def _replace_role(m: re.Match[str]) -> str:
        tok = m.group(0)
        canon = ROLE_ALIAS_MAP.get(tok, tok)
        if canon not in active_roles:
            return "家人"
        if strict_unverified and canon not in allowed_roles:
            return "家人"
        return tok

    work = alias_pat.sub(_replace_role, work)

    # 4b. Neutralize/resolve cold surveillance adult nouns (一名成年男性 / 两位女士 / 家长 / 大人)
    male_repl = next(iter(local_males)) if len(local_males) == 1 else "家人"
    fem_repl = next(iter(local_females)) if len(local_females) == 1 else "家人"
    adult_repl = next(iter(allowed_roles)) if len(allowed_roles) == 1 else "家人"
    work = re.sub(r"(?:两名|两位|一名|一位|一个|该)?(?:成年)?(?:男性|男子|男士)", male_repl, work)
    work = re.sub(r"(?:两名|两位|一名|一位|一个|该)?(?:成年)?(?:女性|女子|女士|妇女)", fem_repl, work)
    work = re.sub(r"(?:两名|两位|一名|一位|一个|该)?(?:家长|成年人|成人)", "家人", work)
    work = re.sub(r"(?:一名|一位|一个|该)?大人(?!们)", adult_repl, work)

    # 5. Tidy duplicate neutral placeholders created by compound replacements
    work = re.sub(r"家人(?:[和与及、]家人)+", "家人", work)
    work = re.sub(r"(?:家人){2,}", "家人", work)

    for idx, orig in enumerate(stashed):
        work = work.replace(f"\x00SPK{idx}\x00", orig)
    return work


class FamilyFaceService:
    """High-level coordinator for member enrollment, multi-frame local matching, VLM ref sheets & kinship guard."""

    def __init__(
        self,
        db: ArchiveDB,
        data_root: Path,
        cfg: dict[str, Any],
        *,
        font_path: str | None = None,
    ) -> None:
        self.db = db
        self.data_root = Path(data_root)
        self.cfg = cfg
        self.font_path = font_path
        self._sheet_lock = threading.Lock()
        self._sheet_cache: dict[str, tuple[str, bytes]] = {}
        self._engine: ScrfdArcFaceEngine | None = None
        self._engine_sig: tuple[str, str] | None = None
        self.ensure_seeded()

    @property
    def fcfg(self) -> dict[str, Any]:
        return self.cfg.get("face_id") or {}

    @property
    def enabled(self) -> bool:
        return bool(self.fcfg.get("enabled", True))

    def _get_engine(self) -> ScrfdArcFaceEngine:
        fcfg = self.fcfg
        det_p = _resolve_model_path(fcfg.get("det_onnx"), "/app/models/face/det_500m.onnx", "det_500m.onnx")
        rec_p = _resolve_model_path(fcfg.get("rec_onnx"), "/app/models/face/w600k_mbf.onnx", "w600k_mbf.onnx")
        sig = (str(det_p), str(rec_p))
        if self._engine is None or self._engine_sig != sig:
            self._engine = ScrfdArcFaceEngine(det_p, rec_p)
            self._engine_sig = sig
        return self._engine

    def engine_status(self) -> dict[str, Any]:
        eng = self._get_engine()
        return {
            "enabled": self.enabled,
            "engine_available": eng.available,
            "det_model": str(eng.det_path.name),
            "rec_model": str(eng.rec_path.name),
            "load_error": eng._load_error,
            "attach_reference_sheet": bool(self.fcfg.get("attach_reference_sheet", True)),
            "strict_unverified_to_family": bool(self.fcfg.get("strict_unverified_to_family", True)),
            "high_sim_threshold": float(self.fcfg.get("high_sim_threshold", 0.40)),
            "medium_sim_threshold": float(self.fcfg.get("medium_sim_threshold", 0.32)),
        }

    def ensure_seeded(self) -> None:
        cfg_members = self.cfg.get("family_members")
        defaults = cfg_members if isinstance(cfg_members, list) and cfg_members else DEFAULT_FAMILY_MEMBERS
        self.db.seed_default_family_members(defaults)

    def invalidate_cache(self) -> None:
        with self._sheet_lock:
            self._sheet_cache.clear()

    def _gallery_for_day(self, day: str | None) -> list[dict[str, Any]]:
        return self.db.get_family_face_gallery(day=day)

    def active_roles_for_day(self, day: str | None) -> set[str]:
        return {m["label_zh"] for m in self._gallery_for_day(day)}

    def has_enrolled_members(self, day: str | None = None) -> bool:
        for m in self._gallery_for_day(day):
            if m.get("samples"):
                return True
        return False

    def enroll_image_bytes(
        self,
        member_id: str,
        img_bytes: bytes,
        *,
        source: str = "upload",
        source_note: str = "",
        allow_visual_only: bool = False,
        face_index: int = 0,
    ) -> dict[str, Any]:
        member = self.db.get_family_member(member_id)
        if member is None:
            raise ValueError(f"未找到家庭成员: {member_id}")
        bgr = _decode_image_bgr(img_bytes)
        if bgr is None or bgr.size == 0:
            raise ValueError("无法解析图片文件，请上传清晰的 JPG/PNG 照片")

        eng = self._get_engine()
        det_thr = float(self.fcfg.get("enroll_det_threshold", 0.45))
        faces = eng.detect_faces(bgr, score_thr=det_thr, min_face_px=24) if (self.enabled and eng.available) else []
        if not faces and self.enabled and eng.available:
            faces = eng.detect_faces(bgr, score_thr=max(0.28, det_thr - 0.12), min_face_px=20)

        chosen_face: dict[str, Any] | None = None
        embedding: list[float] | None = None
        if faces:
            if 0 <= face_index < len(faces):
                chosen_face = faces[face_index]
            else:
                # Prefer largest reliable-pose face when multiple appear in a life photo
                reliable_faces = [f for f in faces if is_reliable_face_pose(f, min_eye_dx_ratio=0.18)]
                pool = reliable_faces or faces
                chosen_face = max(
                    pool,
                    key=lambda f: (f["box_px"][2] - f["box_px"][0]) * (f["box_px"][3] - f["box_px"][1]) * f["score"],
                )
            if chosen_face.get("kps"):
                if is_reliable_face_pose(chosen_face, min_eye_dx_ratio=0.18):
                    embedding = eng.extract_embedding(bgr, chosen_face["kps"])
                elif not allow_visual_only:
                    raise ValueError("照片中人脸偏转角度过大或为侧背影，容易引起误识别；请换一张正脸/斜正脸照，或勾选“无清晰正脸也仅作为视觉参考录入”")

        if chosen_face is None and not allow_visual_only:
            if eng.available:
                raise ValueError("照片中未检测到清晰正脸/侧脸，请换一张正面照或勾选“无清晰正脸也仅作为视觉参考录入”")
            raise ValueError("人脸模型未就绪，且未开启仅视觉参考录入")

        sid = f"fs_{uuid.uuid4().hex[:12]}"
        thumb_rel = f"family_faces/{member_id}/{sid}.jpg"
        thumb_path = self.data_root / thumb_rel
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        portrait_jpg = eng.crop_head_portrait_jpeg(bgr, chosen_face["box_px"] if chosen_face else None, size=240)
        thumb_path.write_bytes(portrait_jpg)

        sample = self.db.add_family_face_sample({
            "sample_id": sid,
            "member_id": member_id,
            "source": source,
            "source_note": source_note[:120],
            "thumb_rel": thumb_rel,
            "embedding": embedding or [],
            "det_score": float(chosen_face["score"]) if chosen_face else 0.0,
        })
        self.invalidate_cache()
        self.get_reference_sheet(day=None)
        return {
            "sample": sample,
            "faces_detected": len(faces),
            "has_embedding": bool(embedding),
        }

    def delete_sample(self, sample_id: str) -> bool:
        removed = self.db.delete_family_face_sample(sample_id)
        if not removed:
            return False
        trel = removed.get("thumb_rel")
        if trel:
            try:
                (self.data_root / str(trel)).unlink(missing_ok=True)
            except OSError:
                pass
        self.invalidate_cache()
        self.get_reference_sheet(day=None)
        return True

    def _load_sample_portrait(self, thumb_rel: str) -> Image.Image | None:
        if not thumb_rel:
            return None
        p = self.data_root / thumb_rel
        if not p.is_file():
            return None
        try:
            with Image.open(p) as im:
                return im.convert("RGB")
        except Exception:
            return None

    def _load_cjk_font(self, size: int) -> Any:
        cands = [self.font_path, *(self.cfg.get("media", {}).get("font_candidates") or [])]
        for cand in cands:
            if cand and Path(str(cand)).is_file():
                try:
                    return ImageFont.truetype(str(cand), size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def get_reference_sheet(self, day: str | None = None) -> bytes | None:
        """Build or return cached VLM family visual reference contact sheet for active enrolled members."""
        gallery = self._gallery_for_day(day)
        enrolled = [m for m in gallery if m.get("samples")]
        if not enrolled:
            latest_p = self.data_root / "family_faces" / "reference_sheet_latest.jpg"
            if day is None:
                latest_p.unlink(missing_ok=True)
            return None

        ranked_by_member: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        sig_parts = []
        for m in enrolled:
            # Rank clear frontal/embedded samples ahead of newer low-score side captures so VLM always sees crisp faces
            ranked = sorted(
                m["samples"],
                key=lambda s: (
                    int(bool(s.get("has_embedding"))),
                    float(s.get("det_score") or 0.0),
                    float(s.get("created_at") or 0.0),
                ),
                reverse=True,
            )
            ranked_by_member.append((m, ranked))
            s_ids = ",".join(str(s["sample_id"]) for s in ranked[:2])
            sig_parts.append(f"{m['member_id']}:{m['label_zh']}:{s_ids}")
        sig = hashlib.sha256("|".join(sig_parts).encode("utf-8")).hexdigest()
        cache_key = day or "__all__"

        with self._sheet_lock:
            cached = self._sheet_cache.get(cache_key)
            if cached and cached[0] == sig:
                return cached[1]

        tiles: list[tuple[str, str, list[Image.Image]]] = []
        for m, ranked in ranked_by_member:
            imgs: list[Image.Image] = []
            for s in ranked:
                im = self._load_sample_portrait(s.get("thumb_rel", ""))
                if im is not None:
                    imgs.append(im)
                if len(imgs) >= 2:
                    break
            if imgs:
                tiles.append((m["label_zh"], str(m.get("notes") or ""), imgs))

        if not tiles:
            return None

        cols = min(4, max(2, len(tiles)))
        rows = int(math.ceil(len(tiles) / float(cols)))
        cell_w, cell_h, hdr_h, gap = 252, 186, 44, 10
        banner_h = 34
        total_w = cols * cell_w + (cols + 1) * gap
        total_h = banner_h + rows * cell_h + (rows + 1) * gap
        canvas = Image.new("RGB", (total_w, total_h), (24, 21, 19))
        draw = ImageDraw.Draw(canvas)
        f_banner = self._load_cjk_font(17)
        f_title = self._load_cjk_font(20)
        f_sub = self._load_cjk_font(12)

        draw.rectangle([0, 0, total_w, banner_h], fill=(42, 34, 30))
        draw.text((12, 7), "家庭成员识别对照卡（按五官、发型、眼镜与日常衣着特征核对）", font=f_banner, fill=(255, 240, 214))

        for idx, (lbl, note, imgs) in enumerate(tiles):
            r_i, c_i = divmod(idx, cols)
            x0 = gap + c_i * (cell_w + gap)
            y0 = banner_h + gap + r_i * (cell_h + gap)
            draw.rounded_rectangle([x0, y0, x0 + cell_w, y0 + cell_h], radius=10, fill=(38, 33, 30), outline=(88, 76, 68))
            img_y = y0 + 6
            img_h = cell_h - hdr_h - 10
            if len(imgs) == 1:
                pw = cell_w - 12
                sq = ImageOps.fit(imgs[0], (pw, img_h), method=Image.Resampling.LANCZOS)
                canvas.paste(sq, (x0 + 6, img_y))
            else:
                pw = (cell_w - 16) // 2
                for j in range(2):
                    sq = ImageOps.fit(imgs[j], (pw, img_h), method=Image.Resampling.LANCZOS)
                    canvas.paste(sq, (x0 + 6 + j * (pw + 4), img_y))
            draw.rectangle([x0 + 2, y0 + cell_h - hdr_h, x0 + cell_w - 2, y0 + cell_h - 2], fill=(20, 17, 15))
            draw.text((x0 + 10, y0 + cell_h - hdr_h + 4), lbl, font=f_title, fill=(255, 232, 186))
            if note:
                draw.text((x0 + 10, y0 + cell_h - 18), note[:18], font=f_sub, fill=(198, 188, 176))

        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=88)
        sheet_bytes = buf.getvalue()

        with self._sheet_lock:
            self._sheet_cache[cache_key] = (sig, sheet_bytes)

        if day is None:
            try:
                latest_p = self.data_root / "family_faces" / "reference_sheet_latest.jpg"
                latest_p.parent.mkdir(parents=True, exist_ok=True)
                latest_p.write_bytes(sheet_bytes)
            except OSError:
                pass
        return sheet_bytes

    def identify_images(
        self,
        images_jpeg: list[bytes],
        *,
        day: str | None = None,
        child_box: list[float] | None = None,
    ) -> dict[str, Any]:
        """Run local SCRFD + ArcFace across one or more frames and aggregate member votes."""
        gallery = self._gallery_for_day(day)
        active_labels = [m["label_zh"] for m in gallery]
        all_members = self.db.list_family_members(include_samples=False)
        inactive_labels = [m["label_zh"] for m in all_members if m["label_zh"] not in active_labels]
        enrolled_with_emb = [
            m for m in gallery if any(s.get("has_embedding") for s in (m.get("samples") or []))
        ]
        enrolled_any = [m for m in gallery if m.get("samples")]

        base_res: dict[str, Any] = {
            "verified_roles": set(),
            "possible_roles": set(),
            "unknown_faces": 0,
            "details": [],
            "active_roster_labels": active_labels,
            "inactive_roster_labels": inactive_labels,
            "active_member_notes": [f"{m['label_zh']}({m['notes']})" for m in gallery if m.get("notes")],
            "has_enrolled_members": bool(enrolled_any),
            "has_enrolled_embeddings": bool(enrolled_with_emb),
        }
        eng = self._get_engine()
        if not self.enabled or not eng.available or not enrolled_with_emb or not images_jpeg:
            return base_res

        det_thr = float(self.fcfg.get("det_score_threshold", 0.42))
        min_face_px = int(self.fcfg.get("min_face_px", 24))
        min_eye_dx = float(self.fcfg.get("min_eye_dx_ratio", 0.20))
        high_thr = float(self.fcfg.get("high_sim_threshold", 0.40))
        med_thr = float(self.fcfg.get("medium_sim_threshold", 0.32))
        min_margin = float(self.fcfg.get("min_margin", 0.06))

        per_member_votes: dict[str, dict[str, Any]] = {}
        unknown_faces = 0

        for f_idx, jb in enumerate(images_jpeg):
            bgr = _decode_image_bgr(jb)
            if bgr is None:
                continue
            faces = eng.detect_faces(bgr, score_thr=det_thr, min_face_px=min_face_px)
            for fc in faces:
                kps = fc.get("kps")
                if not kps or not is_reliable_face_pose(fc, min_eye_dx_ratio=min_eye_dx):
                    continue
                emb = eng.extract_embedding(bgr, kps)
                if not emb:
                    continue
                m_res = match_embedding(emb, enrolled_with_emb, high_thr=high_thr, med_thr=med_thr, min_margin=min_margin)
                if m_res["status"] == "unknown":
                    # Avoid treating the baby's own face inside child_box as an unknown adult
                    bx = fc["box_norm"]
                    fcx, fcy = (bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5
                    if child_box and len(child_box) == 4 and child_box[2] > child_box[0]:
                        if child_box[0] <= fcx <= child_box[2] and child_box[1] <= fcy <= child_box[3]:
                            continue
                    unknown_faces += 1
                    continue
                lbl = str(m_res["label_zh"])
                slot = per_member_votes.setdefault(
                    lbl,
                    {
                        "label_zh": lbl,
                        "member_id": m_res["member_id"],
                        "scores": [],
                        "margins": [],
                        "frames": set(),
                        "statuses": [],
                    },
                )
                slot["scores"].append(float(m_res["best_score"]))
                slot["margins"].append(float(m_res["margin"]))
                slot["frames"].add(f_idx)
                slot["statuses"].append(m_res["status"])

        verified_roles: set[str] = set()
        possible_roles: set[str] = set()
        details: list[dict[str, Any]] = []
        for lbl, slot in per_member_votes.items():
            best_sc = max(slot["scores"])
            avg_sc = sum(slot["scores"]) / float(len(slot["scores"]))
            best_margin = max(slot["margins"])
            n_frames = len(slot["frames"])
            is_verified = (best_sc >= high_thr and best_margin >= min_margin) or (
                n_frames >= 2 and avg_sc >= max(med_thr + 0.03, 0.36) and best_margin >= min_margin
            )
            if is_verified:
                verified_roles.add(lbl)
            else:
                possible_roles.add(lbl)
            details.append({
                "label_zh": lbl,
                "member_id": slot["member_id"],
                "confidence": "high" if is_verified else "medium",
                "best_score": round(best_sc, 4),
                "avg_score": round(avg_sc, 4),
                "margin": round(best_margin, 4),
                "frames_hit": n_frames,
            })
        details.sort(key=lambda d: (-int(d["confidence"] == "high"), -d["best_score"]))

        base_res.update({
            "verified_roles": verified_roles,
            "possible_roles": possible_roles,
            "unknown_faces": unknown_faces,
            "details": details,
        })
        return base_res

    def build_vlm_family_prompt_block(
        self,
        day: str | None,
        match_res: dict[str, Any] | None,
        *,
        has_ref_sheet: bool = False,
        day_confirmed_roles: set[str] | frozenset[str] | None = None,
    ) -> str:
        res = match_res or {}
        active_lbls = res.get("active_roster_labels") or sorted(self.active_roles_for_day(day))
        inactive_lbls = res.get("inactive_roster_labels") or []
        notes = res.get("active_member_notes") or []
        details = res.get("details") or []
        day_conf = sorted(r for r in (day_confirmed_roles or ()) if r in active_lbls)

        lines = [
            "【家庭成员身份与称呼红线（极其重要）】",
            f"- 当日（{day or '近期'}）家里可能出镜的成员仅有：{'、'.join(active_lbls) if active_lbls else '妈妈、爸爸、奶奶、爷爷、姥姥、姥爷、小姨'}。",
        ]
        if inactive_lbls:
            lines.append(f"- 以下成员在当日不在有效照顾期，严禁写入称呼：{'、'.join(inactive_lbls)}。")
        if notes:
            lines.append("- 成员特征备忘：" + "；".join(notes) + "。")
        if has_ref_sheet:
            lines.append("- 附带的【家庭成员识别对照卡】为已录入的家庭成员标准照/监控抓拍照；对比画面中成人的五官、发型、眼镜与穿着判断是谁。")
        if day_conf:
            lines.append(
                f"- 本日全天已通过本地人脸高置信确认在家的成人有：{'、'.join(day_conf)}。若本段画面为侧脸/低头/背影但发型、眼镜或衣着与对照卡中某位今日在场成员高度一致，请优先对应到今日在场成员并把 caregivers_seen 填 high；严禁臆猜成今日未在场的其他长辈。"
            )
        if details:
            desc = []
            for d in details:
                lvl = "高置信确认" if d["confidence"] == "high" else "疑似候选"
                desc.append(f"{d['label_zh']}（{lvl}，相似度 {d['best_score']:.2f}）")
            lines.append("- 本地人脸识别引擎结果：" + "；".join(desc) + "。若已高置信确认是某位成员（如奶奶/爷爷/姥姥），描述中必须使用该准确称呼，严禁把奶奶或姥姥默认写成“妈妈”。")
        else:
            lines.append("- 本地人脸识别未高置信锁定具体成人（可能为背影、低头、侧脸或尚未录入照片）。")
        lines.append("- 铁律：严禁按“有女性照护=妈妈、有男性照护=爸爸”去猜！也严禁使用“大人/一名男子/一位女士/成年男性/成年女性”等冰冷词；没有看清特征对应具体成员时，一律写“家人”或省略称呼（如“在怀里”、“身旁照料”），caregivers_seen 填 [{\"role\":\"unknown\",\"confidence\":\"low\"}]。")
        return "\n".join(lines)

    def sanitize_text(
        self,
        text: str,
        *,
        day: str | None,
        match_res: dict[str, Any] | None = None,
        vlm_ans: dict[str, Any] | None = None,
        day_confirmed_roles: set[str] | frozenset[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Apply deterministic kinship sanitization & return (cleaned_text, confirmed_people_labels)."""
        active_roles = self.active_roles_for_day(day)
        res = match_res or {}
        verified_local = set(res.get("verified_roles") or set())
        verified_vlm = extract_high_conf_vlm_roles(vlm_ans)
        has_ref = bool(res.get("has_enrolled_members"))
        strict = bool(self.fcfg.get("strict_unverified_to_family", True))

        cleaned = sanitize_caption_kinship(
            text,
            verified_local_roles=verified_local,
            verified_vlm_roles=verified_vlm,
            active_roles=active_roles,
            has_enrolled_ref=has_ref,
            strict_unverified=strict,
            day_confirmed_roles=day_confirmed_roles,
        )

        # Compute clean list of confirmed people for metadata / UI pills
        eff_vlm = set(verified_vlm) if has_ref else set()
        day_conf = {r for r in (day_confirmed_roles or ()) if r in active_roles}
        for present_role, absent_role in (("奶奶", "姥姥"), ("姥姥", "奶奶"), ("爷爷", "姥爷"), ("姥爷", "爷爷")):
            if day_conf and present_role in day_conf and absent_role not in day_conf and absent_role not in verified_local:
                if absent_role in eff_vlm:
                    eff_vlm.discard(absent_role)
                    eff_vlm.add(present_role)
        lf = verified_local & FEMALE_ROLES
        lm = verified_local & MALE_ROLES
        if lf and "妈妈" not in lf:
            eff_vlm.discard("妈妈")
        if lm and "爸爸" not in lm:
            eff_vlm.discard("爸爸")
        people = sorted((verified_local | eff_vlm) & active_roles)
        return cleaned, people

    def _candidates_index_path(self) -> Path:
        return self.data_root / "family_faces" / "candidates" / "candidates_index.json"

    def list_discovered_candidates(self) -> list[dict[str, Any]]:
        idx_p = self._candidates_index_path()
        if not idx_p.is_file():
            return []
        try:
            raw = json.loads(idx_p.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
        except Exception:
            return []
        ignored = self.db.get_ignored_face_hashes()
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("face_hash") in ignored:
                continue
            rel = str(item.get("thumb_rel") or "")
            if rel and (self.data_root / rel).is_file():
                out.append({k: v for k, v in item.items() if k != "embedding"})
        return out

    def discover_candidates_from_archive(
        self,
        *,
        max_days: int = 14,
        max_photos: int = 60,
        max_candidates: int = 36,
    ) -> list[dict[str, Any]]:
        """Scan recent archived full-resolution photos to harvest cropped adult face candidates for 1-click labeling."""
        eng = self._get_engine()
        if not self.enabled or not eng.available:
            return []

        ignored = self.db.get_ignored_face_hashes()
        gallery = self._gallery_for_day(None)
        enrolled_embs = [
            _norm_vec(np.asarray(s["embedding"], dtype=np.float32))
            for m in gallery
            for s in (m.get("samples") or [])
            if s.get("has_embedding") and isinstance(s.get("embedding"), list) and len(s["embedding"]) >= 64
        ]

        photo_files: list[tuple[str, Path]] = []
        days_dir = self.data_root / "days"
        if days_dir.is_dir():
            day_dirs = sorted([d for d in days_dir.iterdir() if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)], key=lambda d: d.name, reverse=True)
            for dd in day_dirs[:max_days]:
                pdir = dd / "photos"
                if not pdir.is_dir():
                    continue
                for pf in sorted(pdir.glob("full_*.jpg"), reverse=True):
                    photo_files.append((dd.name, pf))
                    if len(photo_files) >= max_photos:
                        break
                if len(photo_files) >= max_photos:
                    break
        v2_pdir = self.data_root / "v2" / "photos"
        if len(photo_files) < max_photos and v2_pdir.is_dir():
            for pf in sorted(v2_pdir.glob("full_*.jpg"), reverse=True):
                photo_files.append(("", pf))
                if len(photo_files) >= max_photos:
                    break

        cand_dir = self.data_root / "family_faces" / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        for old in cand_dir.glob("cand_*.jpg"):
            old.unlink(missing_ok=True)

        accepted: list[dict[str, Any]] = []
        accepted_embs: list[np.ndarray] = []

        for day_s, pf in photo_files:
            if len(accepted) >= max_candidates:
                break
            try:
                bgr = _decode_image_bgr(pf.read_bytes())
            except OSError:
                continue
            if bgr is None:
                continue
            faces = eng.detect_faces(bgr, score_thr=0.55, min_face_px=32)
            for f_idx, fc in enumerate(faces):
                if len(accepted) >= max_candidates:
                    break
                if not fc.get("kps") or not is_reliable_face_pose(fc, min_eye_dx_ratio=0.22):
                    continue
                emb = eng.extract_embedding(bgr, fc["kps"])
                if not emb:
                    continue
                emb_arr = np.asarray(emb, dtype=np.float32)
                # Quantized signature hash so ignored faces stay suppressed across re-scans
                f_hash = hashlib.sha256(f"{pf.name}|{f_idx}|{fc['box_norm']}".encode("utf-8")).hexdigest()[:16]
                if f_hash in ignored:
                    continue
                # Skip faces already very confidently enrolled (>= 0.62 cosine) to keep review focused
                if any(float(np.dot(emb_arr, ee)) >= 0.62 for ee in enrolled_embs):
                    continue
                # Deduplicate near-identical captures across burst photos
                if any(float(np.dot(emb_arr, ae)) >= 0.84 for ae in accepted_embs):
                    continue

                m_guess = match_embedding(emb, gallery, high_thr=0.40, med_thr=0.30, min_margin=0.05) if gallery else {}
                cid = f"cand_{len(accepted) + 1:02d}_{f_hash[:8]}"
                thumb_rel = f"family_faces/candidates/{cid}.jpg"
                thumb_jpg = eng.crop_head_portrait_jpeg(bgr, fc["box_px"], size=220)
                (self.data_root / thumb_rel).write_bytes(thumb_jpg)

                accepted_embs.append(emb_arr)
                accepted.append({
                    "cand_id": cid,
                    "face_hash": f_hash,
                    "day": day_s,
                    "source_file": pf.name,
                    "thumb_rel": thumb_rel,
                    "det_score": fc["score"],
                    "guess_label": m_guess.get("best_label_zh") or "",
                    "guess_member_id": m_guess.get("best_member_id") or "",
                    "guess_score": float(m_guess.get("best_score") or 0.0),
                    "guess_status": str(m_guess.get("status") or "unknown"),
                    "embedding": emb,
                })

        self._candidates_index_path().write_text(json.dumps(accepted, ensure_ascii=False), encoding="utf-8")
        return [{k: v for k, v in it.items() if k != "embedding"} for it in accepted]

    def assign_discovered_candidate(self, cand_id: str, member_id: str) -> dict[str, Any]:
        idx_p = self._candidates_index_path()
        if not idx_p.is_file():
            raise ValueError("候选列表为空，请先点击扫描")
        items = json.loads(idx_p.read_text(encoding="utf-8"))
        target = next((it for it in items if it.get("cand_id") == cand_id), None)
        if target is None:
            raise ValueError(f"未找到候选人脸: {cand_id}")

        remaining = [it for it in items if it.get("cand_id") != cand_id]
        idx_p.write_text(json.dumps(remaining, ensure_ascii=False), encoding="utf-8")

        if member_id in ("ignore", "baby", "skip"):
            self.db.add_ignored_face(str(target.get("face_hash") or cand_id), reason=member_id)
            rel_old = str(target.get("thumb_rel") or "")
            if rel_old:
                (self.data_root / rel_old).unlink(missing_ok=True)
            return {"assigned": False, "ignored": True, "cand_id": cand_id}

        member = self.db.get_family_member(member_id)
        if member is None:
            raise ValueError(f"未知家庭成员: {member_id}")

        sid = f"fs_{uuid.uuid4().hex[:12]}"
        thumb_rel = f"family_faces/{member_id}/{sid}.jpg"
        dst_thumb = self.data_root / thumb_rel
        dst_thumb.parent.mkdir(parents=True, exist_ok=True)
        src_thumb = self.data_root / str(target.get("thumb_rel") or "")
        if src_thumb.is_file():
            dst_thumb.write_bytes(src_thumb.read_bytes())
            src_thumb.unlink(missing_ok=True)

        note = f"监控抓拍 {target.get('day') or ''} {target.get('source_file') or ''}".strip()
        sample = self.db.add_family_face_sample({
            "sample_id": sid,
            "member_id": member_id,
            "source": "cctv_capture",
            "source_note": note,
            "thumb_rel": thumb_rel,
            "embedding": target.get("embedding") or [],
            "det_score": float(target.get("det_score") or 0.0),
        })
        self.invalidate_cache()
        self.get_reference_sheet(day=None)
        return {"assigned": True, "sample": sample, "member_id": member_id}


def extract_high_conf_vlm_roles(ans: dict[str, Any] | None) -> set[str]:
    """Extract canonical kinship role names reported with high confidence by VLM."""
    if not isinstance(ans, dict):
        return set()
    raw = ans.get("caregivers_seen")
    if not isinstance(raw, list):
        return set()
    roles: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            role = str(item.get("role") or "").strip()
            conf = str(item.get("confidence") or "").strip().lower()
            canon = ROLE_ALIAS_MAP.get(role, role)
            if canon in KINSHIP_ROLES and conf == "high":
                roles.add(canon)
        elif isinstance(item, str):
            canon = ROLE_ALIAS_MAP.get(item.strip(), item.strip())
            if canon in KINSHIP_ROLES:
                roles.add(canon)
    return roles
