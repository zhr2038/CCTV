"""Two-tier Cloud VLM router (Gemini 4 primary + multi-cloud fallback) with strict privacy gates & zero secret leaks."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import inspect
import io
import json
import os
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
import urllib3.util.connection as urllib3_cn

from . import media_nas as med
from .db import (
    LEGACY_REFUSAL_CAPTIONS,
    REFUSAL_MODELS,
    ArchiveDB,
    is_unconfirmed_refusal,
    normalize_vlm_response,
)


# Enforce IPv4 connections so NAS container traffic via router gateway avoids broken IPv6/TLS EOF resets.
def _ipv4_only():
    return socket.AF_INET


urllib3_cn.allowed_gai_family = _ipv4_only

PROMPT_MILESTONE = """客观观察这组按时间顺序排列的家庭监控连续画面（只写画面里直接可见的事实，不要编造未出现的动作）。
注意区分成人和婴幼儿（约0-14个月宝宝）：成人弯腰/蹲下或画面远处的大人不能算宝宝；成人抱起/搀扶的动作必须在independence填assisted。
【身份称呼铁律】若画面中出现照护宝宝的大人，切勿按年龄或性别主观猜测为“妈妈/爸爸”！除非附带的家庭成员对照卡/本地人脸结果已明确高置信匹配某位具体成员（如奶奶/爷爷/姥姥/姥爷/小姨/妈妈/爸爸），否则 caption_zh 里一律写“家人”并把 caregivers_seen 设为 [{"role":"unknown","confidence":"low"}]。
只返回单个JSON对象：
{
  "child_clear": "yes|partial|blocked|no",
  "box_2d": [ymin, xmin, ymax, xmax],
  "privacy_exposed": "yes|no",
  "privacy_box_2d": [ymin, xmin, ymax, xmax],
  "caregivers_seen": [{"role": "妈妈|爸爸|奶奶|爷爷|姥姥|姥爷|小姨|月嫂|unknown", "confidence": "high|medium|low"}],
  "categories": ["standing"|"walking"|"sofa_descent"|"toy_play"|"happy_interaction"|"daily_care"|"kitchen_visit"|"none"],
  "independence": "independent|assisted|unclear",
  "excitement_score": 0,
  "caption_zh": "简短客观中文描述，写清宝宝在做什么及是否有成人搀扶/照护互动"
}
box_2d使用0到1000整数归一化坐标框住宝宝主体（无宝宝时填[0,0,0,0]）。
standing=双脚站立或扶物站立；walking=迈步走/扶走；sofa_descent=从沙发/床沿下地或爬上爬下；toy_play=宝宝专注抓握/玩耍/推玩玩具；happy_interaction=明显开心互动/逗笑；daily_care=给宝宝换尿布、洗澡后擦身体/裹浴巾、涂润肤露、穿衣抚触等温馨亲子照护瞬间（这是珍贵的成长记忆，请正常纳入并客观描述，温馨互动时 excitement_score 可给 3-5）；kitchen_visit=宝宝出现在厨房。
若换尿布、洗澡擦身体或穿衣过程中露出了宝宝的隐私部位（生殖器/臀部私处），请将 privacy_exposed 填 yes，并用 privacy_box_2d 给出仅覆盖该隐私部位（视频需覆盖片段内移动范围）的矩形框 [ymin,xmin,ymax,xmax]（0..1000整数归一化坐标），系统会自动仅对该局部打柔和马赛克并完整保留温馨画面，切勿因此扣分或淘汰；若已穿好衣物/包好尿布/被浴巾遮住无裸露，则 privacy_exposed 填 no 且 privacy_box_2d 填 [0,0,0,0]。"""

PROMPT_PHOTO = """评估这张家庭监控原画截图是否适合作为“今日成长相册精选照片”（要求高于普通日常记录）。
【身份称呼铁律】切勿主观把陪护女性默认写成“妈妈”或把男性写成“爸爸”，也严禁使用“一名男子/一位女士/一名女子/男子/女子/家长”等冰冷词！除非对照参考图或人脸结果明确确认了具体成员（如奶奶/爷爷/妈妈/爸爸/姥姥/姥爷/小姨），否则图注 caption_zh 中统一温暖称呼为“家人”并把 caregivers_seen 填 unknown/low。
只返回单个JSON对象：
{
  "child_clear": "yes|blocked|no",
  "face_or_expression_clear": "yes|side|no",
  "composition_good": "yes|no",
  "osd_box_on_child": "yes|no",
  "box_2d": [ymin, xmin, ymax, xmax],
  "privacy_exposed": "yes|no",
  "privacy_box_2d": [ymin, xmin, ymax, xmax],
  "caregivers_seen": [{"role": "妈妈|爸爸|奶奶|爷爷|姥姥|姥爷|小姨|月嫂|unknown", "confidence": "high|medium|low"}],
  "adult_back_or_awkward": "yes|no",
  "meaningful_action": "yes|no",
  "photo_score": 0,
  "caption_zh": "简短客观中文图注"
}
评分与筛选准则：
1. face_or_expression_clear：能看清宝宝正脸或清晰生动的五官表情填yes；明显偏侧脸只见脸颊轮廓填side；背影/低头/被挡看不见脸填no。
2. osd_box_on_child：若原监控画面内嵌了摄像头自带的人形/移动检测框（绿框/白框/黄框等）叠加在宝宝身上，填yes；无内嵌框填no。
3. composition_good：宝宝主体完整舒展、构图自然且未被边缘截断或被大人背影遮挡填yes，否则填no。
4. 给宝宝洗澡后裹浴巾擦身体、换尿布互动、抚触穿衣等温馨日常照护同样属于珍贵家庭记录，不要因照护场景扣分或排斥；只要满足五官表情清晰、构图自然、无内嵌检测框同样可给 4 或 5 分。
5. 若换尿布、洗澡擦身体或穿衣过程中露出了宝宝的隐私部位（生殖器/臀部私处），请将 privacy_exposed 填 yes，并用 privacy_box_2d 给出仅覆盖该隐私部位（视频需覆盖片段内移动范围）的矩形框 [ymin,xmin,ymax,xmax]（0..1000整数归一化坐标），系统会自动仅对该局部打柔和马赛克并完整保留温馨画面，切勿因此扣分或淘汰；若已穿好衣物/包好尿布/被浴巾遮住无裸露，则 privacy_exposed 填 no 且 privacy_box_2d 填 [0,0,0,0]。
6. 只有同时满足 child_clear=yes、face_or_expression_clear=yes、composition_good=yes、adult_back_or_awkward=no、osd_box_on_child=no 且姿态自然有神采时，photo_score 才给 4 或 5；偏侧脸、带内嵌检测框或构图一般的日常记录帧最多给 2-3；空床/遮挡/杂乱给 0-1。box_2d使用0到1000整数归一化坐标框住宝宝。"""

PROMPT_MONTH = """这是同一个宝宝（约0-14个月）某天的一组清晰监控帧（按时间顺序）。
请严格评估画面清晰度、宝宝状态与构图是否适合放进“月龄成长变化高清对照/流年”：
只返回单个JSON对象：
{
  "child_clear": "yes|blocked|no",
  "face_or_posture_sharp": "yes|blurry",
  "adult_occluding": "yes|no",
  "is_awake_or_active": "yes|sleeping",
  "box_2d": [ymin, xmin, ymax, xmax],
  "clarity_score": 0,
  "stage_zh": "10字以内客观成长姿态，如：仰卧挥动小手、俯卧抬头探索、独坐抓握玩具、扶沙发站立、浴后抚触穿衣"
}
注意：若宝宝面部/肢体明显拖影模糊或被大人挡住主体，face_or_posture_sharp填blurry或adult_occluding填yes，clarity_score给0-2。box_2d用0..1000整数坐标框住宝宝全身或上半身（留适度余量）。"""

PROMPT_KITCHEN_CHILD = """客观观察这组家庭厨房监控连续帧（按时间顺序）。
注意严厉区分成人和婴幼儿（约0-14个月宝宝）：如果画面里只有大人在做饭、走动、弯腰取物，没有明确清晰的宝宝本人出镜，child_clear必须填no。
【身份称呼铁律】切勿主观把大人写成“妈妈/爸爸”，也禁用“一名男子/一位女士/男子/女子/家长”等冰冷词；若已对照家庭成员参考卡/本地人脸结果高置信确认具体成员则用该准确称呼，否则一律写“家人”并把 caregivers_seen 填 unknown/low。
只返回单个JSON对象：
{
  "child_clear": "yes|partial|no",
  "box_2d": [ymin, xmin, ymax, xmax],
  "caregivers_seen": [{"role": "妈妈|爸爸|奶奶|爷爷|姥姥|姥爷|小姨|月嫂|unknown", "confidence": "high|medium|low"}],
  "excitement_score": 0,
  "caption_zh": "15字以内客观描述"
}
box_2d使用0..1000整数归一化坐标框住宝宝主体（无宝宝填[0,0,0,0]）。"""


class ContentPolicyBlockedError(RuntimeError):
    """Raised when an upstream VLM safety filter refuses to describe an innocent baby care clip."""


QUOTA_EXHAUSTED_RE = re.compile(
    r"(?i)(insufficient_quota|quota.*exceed|exceeded.*quota|arrearage|overdue|balance|credit.*exhaust|billing|payment_required|allocationquota|free_tier|plan.*expir)"
)
HARD_QUOTA_RE = re.compile(
    r"(?i)(insufficient_quota|billing_disabled|billing.*disabl|credit.*exhaust|arrearage|overdue|payment_required|allocationquota|free_tier|plan.*expir)"
)
TRANSIENT_RATE_RE = re.compile(
    r"(?i)(per[\s_-]*minute|per[\s_-]*second|rate_limit_exceeded|ratelimitexceeded|\brpm\b|\btpm\b)"
)


def _is_hard_quota_exhausted(code: int, raw_err: str) -> bool:
    if code == 402:
        return True
    if code not in (429, 403, 400):
        return False
    err_txt = str(raw_err or "")
    if HARD_QUOTA_RE.search(err_txt):
        return True
    if TRANSIENT_RATE_RE.search(err_txt):
        return False
    return bool(QUOTA_EXHAUSTED_RE.search(err_txt))


def redact_secrets(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"AIza[A-Za-z0-9_-]+", "[REDACTED_GKEY]", s)
    s = re.sub(r"ya29\.[A-Za-z0-9._-]+", "[REDACTED_GTOKEN]", s)
    s = re.sub(r"sk-[A-Za-z0-9_.-]+", "[REDACTED_SK]", s)
    s = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", s)
    s = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED_PKEY]", s)
    s = re.sub(r'"private_key"\s*:\s*"[^"]+"', '"private_key":"[REDACTED_PKEY]"', s)
    return s[:280]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _parse_service_account_json(raw: str | None, pcfg: dict[str, Any]) -> dict[str, str] | None:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    client_email = str(obj.get("client_email") or "").strip()
    private_key = str(obj.get("private_key") or "").strip()
    project_id = str(pcfg.get("project_id") or obj.get("project_id") or "").strip()
    token_uri = str(obj.get("token_uri") or "https://oauth2.googleapis.com/token").strip()
    if not client_email or not private_key or not project_id:
        return None
    return {
        "client_email": client_email,
        "private_key": private_key,
        "project_id": project_id,
        "token_uri": token_uri,
    }


def _sign_rs256_jwt(sa: dict[str, str], now: int | None = None) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    ts = int(time.time() if now is None else now)
    hdr = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    claims = _b64url(
        json.dumps(
            {
                "iss": sa["client_email"],
                "sub": sa["client_email"],
                "aud": sa["token_uri"],
                "scope": "https://www.googleapis.com/auth/cloud-platform",
                "iat": ts - 30,
                "exp": ts + 3540,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signing_input = f"{hdr}.{claims}".encode("ascii")
    priv_key = serialization.load_pem_private_key(sa["private_key"].encode("utf-8"), password=None)
    sig = _b64url(priv_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256()))
    return f"{hdr}.{claims}.{sig}"


def _build_vertex_url(pcfg: dict[str, Any], project_id: str) -> str:
    model = pcfg["model"]
    location = str(pcfg.get("location") or "global").strip()
    api_ver = str(pcfg.get("api_version") or "v1beta1").strip()
    explicit_base = str(pcfg.get("base_url") or "").strip().rstrip("/")
    if explicit_base:
        base = explicit_base
    elif location == "global":
        base = f"https://aiplatform.googleapis.com/{api_ver}"
    else:
        base = f"https://{location}-aiplatform.googleapis.com/{api_ver}"
    return f"{base}/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"


def _parse_gemini_like_response(r: requests.Response) -> dict[str, Any]:
    obj = r.json()
    block_reason = (obj.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        raise ContentPolicyBlockedError(f"Gemini blockReason={block_reason}")
    cands = obj.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini returned no candidates: {redact_secrets(r.text)}")
    finish_reason = str(cands[0].get("finishReason") or "")
    resp_parts = cands[0].get("content", {}).get("parts", [])
    raw_txt = "".join(p.get("text", "") for p in resp_parts if not p.get("thought"))
    if not raw_txt.strip():
        if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "BLOCKLIST", "OTHER"):
            raise ContentPolicyBlockedError(f"Gemini finishReason={finish_reason}")
        raise RuntimeError("Gemini returned empty non-thought text")
    um = obj.get("usageMetadata") or {}
    in_tok = int(um.get("promptTokenCount") or 0)
    out_tok = int(um.get("candidatesTokenCount") or 0) + int(um.get("thoughtsTokenCount") or 0)
    tot_tok = int(um.get("totalTokenCount") or (in_tok + out_tok))
    if tot_tok > out_tok and in_tok == 0:
        in_tok = tot_tok - out_tok
    res = parse_json_obj(raw_txt)
    res["_usage"] = {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": max(tot_tok, in_tok + out_tok)}
    return res


def normalize_named_box(obj: dict[str, Any], key_2d: str, key_xyxy: str) -> list[float]:
    """Normalize either Gemini [ymin,xmin,ymax,xmax] (0..1000) or [x1,y1,x2,y2] -> [x1,y1,x2,y2] (0..1)."""
    b2d = obj.get(key_2d)
    if isinstance(b2d, (list, tuple)) and len(b2d) == 4:
        try:
            vals = [float(v) for v in b2d]
            scale = 1000.0 if max(abs(v) for v in vals) > 1.5 else 1.0
            ymin, xmin, ymax, xmax = [max(0.0, min(1.0, v / scale)) for v in vals]
            if xmax > xmin and ymax > ymin:
                return [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)]
        except (TypeError, ValueError):
            pass
    cbox = obj.get(key_xyxy)
    if isinstance(cbox, (list, tuple)) and len(cbox) == 4:
        try:
            vals = [float(v) for v in cbox]
            scale = 1000.0 if max(abs(v) for v in vals) > 1.5 else 1.0
            x1, y1, x2, y2 = [max(0.0, min(1.0, v / scale)) for v in vals]
            if x2 > x1 and y2 > y1:
                return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
        except (TypeError, ValueError):
            pass
    return [0.0, 0.0, 0.0, 0.0]


def normalize_box(obj: dict[str, Any]) -> list[float]:
    """Normalize either Gemini box_2d [ymin,xmin,ymax,xmax] (0..1000) or child_box [x1,y1,x2,y2] -> [x1,y1,x2,y2] (0..1)."""
    return normalize_named_box(obj, "box_2d", "child_box")


def parse_json_obj(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in VLM response")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("VLM response is not a JSON object")
    obj["child_box"] = normalize_box(obj)
    pbox = normalize_named_box(obj, "privacy_box_2d", "privacy_box")
    if obj.get("privacy_exposed") == "yes" or pbox != [0.0, 0.0, 0.0, 0.0]:
        obj["privacy_box"] = pbox
    else:
        obj["privacy_box"] = [0.0, 0.0, 0.0, 0.0]
    return obj


def is_cloud_allowed(
    cam_cfg: dict[str, Any], wall_ts: str | None, frame_offsets: tuple[float, ...] | list[float] = (0.0,),
) -> bool:
    """Check every requested bedside frame against its local capture time."""
    if cam_cfg.get("role") != "bedside_crib":
        return True
    if not wall_ts or len(wall_ts) < 16 or not frame_offsets:
        return False
    try:
        start = dt.datetime.fromisoformat(wall_ts)
        day_lo, day_hi = map(int, cam_cfg.get("day_hours", [8, 20]))
        if not 0 <= day_lo < day_hi <= 24:
            return False
        for offset in frame_offsets:
            seconds = float(offset)
            if seconds < 0:
                return False
            frame_time = start + dt.timedelta(seconds=seconds)
            if not day_lo <= frame_time.hour < day_hi:
                return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def extract_jpegs_ram(ff_bin: str, source_path: str, times_s: list[float], max_edge: int = 768, quality: int = 85) -> list[bytes]:
    """Decode silent resized JPEGs purely into RAM."""
    import cv2
    from PIL import Image

    clean_times = [max(0.0, float(t)) for t in times_s]
    if not clean_times:
        return []
    min_t, max_t = min(clean_times), max(clean_times)
    out: list[bytes] = []
    timed_out = False
    with med.open_fast_clip_window(
        ff_bin, source_path, min_t, (max_t - min_t) + 1.0, include_audio=False
    ) as (clip_src, clip_min_t):
        shift = min_t - clip_min_t
        for t in clean_times:
            rel_t = max(0.0, t - shift)
            cmd = [
                ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", f"{rel_t:.2f}", "-i", clip_src, "-frames:v", "1",
                "-vf", f"scale='min({max_edge},iw)':'min({max_edge},ih)':force_original_aspect_ratio=decrease",
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "pipe:1",
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                timed_out = True
                break
            except (OSError, subprocess.SubprocessError):
                r = None
            if r is not None and r.returncode == 0 and r.stdout:
                out.append(r.stdout)
        if not out and not timed_out and os.path.isfile(clip_src):
            cap = cv2.VideoCapture(clip_src)
            try:
                for t in clean_times:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - shift) * 1000.0)
                    ok, frm = cap.read()
                    if ok:
                        im = Image.fromarray(frm[:, :, ::-1])
                        im.thumbnail((max_edge, max_edge))
                        buf = io.BytesIO()
                        im.save(buf, "JPEG", quality=quality)
                        out.append(buf.getvalue())
            finally:
                cap.release()
    return out


def _stitch_images_horizontal(images: list[bytes], max_edge: int = 1024, quality: int = 85) -> bytes:
    """Combine multiple JPEG frames horizontally into a single labeled collage image for single-image endpoints."""
    if len(images) <= 1:
        return images[0]
    from PIL import Image, ImageDraw

    decoded: list[Image.Image] = []
    for jb in images:
        decoded.append(Image.open(io.BytesIO(jb)).convert("RGB"))
    target_h = min(480, max(im.height for im in decoded))
    gap = 6
    tiles: list[Image.Image] = []
    for idx, im in enumerate(decoded):
        nw = max(32, int(round(im.width * (target_h / float(max(1, im.height))))))
        tile = im.resize((nw, target_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.rectangle([4, 4, 34, 22], fill=(16, 20, 26))
        draw.text((8, 7), f"F{idx + 1}", fill=(255, 255, 255))
        tiles.append(tile)
    total_w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (total_w, target_h), (24, 24, 24))
    x_off = 0
    for t in tiles:
        canvas.paste(t, (x_off, 0))
        x_off += t.width + gap
    if max(canvas.size) > max_edge:
        canvas.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class CloudVLMRouter:
    """Two-tier Gemini 4 primary + multi-cloud fallback VLM client with caching and circuit breakers."""

    def __init__(self, cfg: dict[str, Any], db: ArchiveDB, secret_overrides: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.vlm_cfg = cfg["vlm"]
        self.db = db
        self._overrides = secret_overrides or {}
        self._tls = threading.local()
        self._route_lock = threading.Lock()
        self._vertex_token_lock = threading.Lock()
        self._rr_counter: dict[str, int] = {}
        self._inflight: dict[str, int] = {}
        self._resolved_models: dict[str, tuple[str, float]] = {}
        self._vertex_tokens: dict[str, tuple[str, float]] = {}
        self.timeout_s = float(self.vlm_cfg.get("request_timeout_seconds", 45))
        self.cooldown_s = int(self.vlm_cfg.get("cooldown_seconds", 900))
        self.rate_limit_cooldown_s = int(self.vlm_cfg.get("rate_limit_cooldown_seconds", 120))
        self.quota_cooldown_s = int(self.vlm_cfg.get("quota_cooldown_seconds", 21600))
        self.max_edge = int(self.vlm_cfg.get("max_image_edge", 768))

    def _timeout_for(self, pcfg: dict[str, Any]) -> float:
        return float(pcfg.get("timeout_seconds") or self.timeout_s)

    def _effective_pcfg(self, pcfg: dict[str, Any], tier: str) -> dict[str, Any]:
        ov = (self.vlm_cfg.get("tier_overrides") or {}).get(tier) or {}
        if not isinstance(ov, dict) or not ov:
            return pcfg
        eff = dict(pcfg)
        if ov.get("timeout_seconds"):
            eff["timeout_seconds"] = float(ov["timeout_seconds"])
        kind = str(pcfg.get("kind") or "")
        if kind in ("gemini", "vertex"):
            if ov.get("thinking_level"):
                eff["thinking_level"] = str(ov["thinking_level"])
            gem_tok = ov.get("gemini_max_output_tokens") or ov.get("max_output_tokens")
            if gem_tok:
                eff["max_output_tokens"] = max(int(pcfg.get("max_output_tokens", 2048)), int(gem_tok))
        elif kind == "openai":
            oai_tok = ov.get("openai_max_output_tokens") or ov.get("max_output_tokens")
            if oai_tok:
                eff["max_output_tokens"] = max(int(pcfg.get("max_output_tokens", 1200)), int(oai_tok))
            if pcfg.get("supports_reasoning_effort") and ov.get("reasoning_effort"):
                eff["reasoning_effort"] = str(ov["reasoning_effort"])
        return eff

    @staticmethod
    def _logical_model(pcfg: dict[str, Any], name: str) -> str:
        return str(pcfg.get("display_model") or pcfg.get("model") or name)

    def _read_secret(self, path_str: str) -> str | None:
        if path_str in self._overrides:
            return self._overrides[path_str].strip()
        p = Path(path_str)
        if not p.is_file():
            return None
        val = p.read_text(encoding="utf-8", errors="ignore").strip()
        return val or None

    def _session_for(self, provider: str) -> requests.Session:
        sessions = getattr(self._tls, "sessions", None)
        if sessions is None:
            sessions = {}
            self._tls.sessions = sessions
        sess = sessions.get(provider)
        if sess is None:
            sess = requests.Session()
            sess.trust_env = False
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            sessions[provider] = sess
        return sess

    def _reset_session(self, provider: str) -> None:
        sessions = getattr(self._tls, "sessions", None)
        if sessions is not None:
            sessions.pop(provider, None)

    def _select_provider_chain(self, tier: str) -> list[str]:
        tiers_cfg = self.vlm_cfg.get("tiers", {})
        tier_list = tiers_cfg.get(tier)
        if not tier_list and tier == "director":
            tier_list = tiers_cfg.get("pro")
        full_chain = list(tier_list or tiers_cfg.get("fast", []))
        parallel_pool = list(self.vlm_cfg.get("parallel_tiers", {}).get(tier) or [])
        if not parallel_pool and tier == "director" and not tiers_cfg.get(tier):
            parallel_pool = list(self.vlm_cfg.get("parallel_tiers", {}).get("pro") or [])
        if not parallel_pool:
            return full_chain

        providers = self.vlm_cfg.get("providers", {})
        avail = [p for p in parallel_pool if p in providers and self._provider_available(p, providers[p])]
        if avail:
            grouped: dict[str, list[str]] = {}
            for p in avail:
                grouped.setdefault(self._logical_model(providers[p], p), []).append(p)
            groups = list(grouped.values())
            with self._route_lock:
                rr = self._rr_counter.get(tier, 0)
                self._rr_counter[tier] = rr + 1
            n_grp = len(groups)
            g_shift = rr % n_grp
            groups_rot = groups[g_shift:] + groups[:g_shift]
            sub_shift = rr // n_grp
            groups_inner = [g[sub_shift % len(g) :] + g[: sub_shift % len(g)] for g in groups_rot]
            max_g_len = max(len(g) for g in groups_inner)
            rotated = [g[turn] for turn in range(max_g_len) for g in groups_inner if turn < len(g)]
            with self._route_lock:
                avail_sorted = sorted(rotated, key=lambda p: self._inflight.get(p, 0))
        else:
            avail_sorted = []
        return avail_sorted + [p for p in full_chain if p not in avail_sorted]

    def _order_chain_for_refs(self, chain: list[str], has_ref: bool) -> list[str]:
        """Prefer multi-image providers over single-image collage models when reference images are attached."""
        if not has_ref:
            return chain
        providers = self.vlm_cfg.get("providers", {})
        multi = [p for p in chain if not (providers.get(p) or {}).get("single_image_collage")]
        collage = [p for p in chain if (providers.get(p) or {}).get("single_image_collage")]
        return multi + collage

    def _provider_available(self, name: str, pcfg: dict[str, Any]) -> bool:
        exp = pcfg.get("expires_at")
        if exp:
            try:
                if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(exp).astimezone(dt.timezone.utc):
                    return False
            except ValueError:
                pass
        secret_raw = self._read_secret(pcfg.get("secret_file", ""))
        if not secret_raw:
            return False
        if pcfg.get("kind") == "vertex" and _parse_service_account_json(secret_raw, pcfg) is None:
            return False
        cd = self.db.get_vlm_cooldown(name)
        return time.time() >= cd

    def _get_vertex_access_token(self, sa: dict[str, str], sess: requests.Session, force_refresh: bool = False) -> str:
        cache_key = sa["client_email"]
        with self._vertex_token_lock:
            if not force_refresh:
                cached = self._vertex_tokens.get(cache_key)
                if cached is not None and time.time() < cached[1] - 300:
                    return cached[0]
            assertion = _sign_rs256_jwt(sa)
            resp = sess.post(
                sa["token_uri"],
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": assertion,
                },
                timeout=(10, 20),
            )
            resp.raise_for_status()
            tok_obj = resp.json()
            token = str(tok_obj.get("access_token") or "").strip()
            if not token:
                raise RuntimeError("OAuth2 token response missing access_token")
            expires_in = int(tok_obj.get("expires_in") or 3600)
            self._vertex_tokens[cache_key] = (token, time.time() + max(60, expires_in))
            return token

    def _call_gemini(
        self,
        pcfg: dict[str, Any],
        key: str,
        sess: requests.Session,
        images: list[bytes],
        prompt: str,
        reference_images: list[tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        model = pcfg["model"]
        url = f"{pcfg['base_url'].rstrip('/')}/models/{model}:generateContent"
        parts: list[dict[str, Any]] = []
        for lbl, jb in (reference_images or []):
            parts.append({"text": str(lbl or "reference_sheet")})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jb).decode("ascii")}})
        for idx, jb in enumerate(images):
            parts.append({"text": f"frame_{idx + 1}"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jb).decode("ascii")}})
        parts.append({"text": prompt})
        gcfg: dict[str, Any] = {
            "temperature": 0,
            "maxOutputTokens": int(pcfg.get("max_output_tokens", 1536)),
            "responseMimeType": "application/json",
        }
        if pcfg.get("thinking_level"):
            gcfg["thinkingConfig"] = {"thinkingLevel": str(pcfg["thinking_level"])}
        payload = {"contents": [{"parts": parts}], "generationConfig": gcfg}

        r = sess.post(
            url,
            json=payload,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            timeout=(10, self._timeout_for(pcfg)),
        )
        r.raise_for_status()
        return _parse_gemini_like_response(r)

    def _call_vertex(
        self,
        pcfg: dict[str, Any],
        sa_raw: str,
        sess: requests.Session,
        images: list[bytes],
        prompt: str,
        reference_images: list[tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        sa = _parse_service_account_json(sa_raw, pcfg)
        if sa is None:
            raise ValueError("Invalid Vertex AI service account JSON")
        url = _build_vertex_url(pcfg, sa["project_id"])
        parts: list[dict[str, Any]] = []
        for lbl, jb in (reference_images or []):
            parts.append({"text": str(lbl or "reference_sheet")})
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jb).decode("ascii")}})
        for idx, jb in enumerate(images):
            parts.append({"text": f"frame_{idx + 1}"})
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jb).decode("ascii")}})
        parts.append({"text": prompt})
        gcfg: dict[str, Any] = {
            "temperature": 0,
            "maxOutputTokens": int(pcfg.get("max_output_tokens", 1536)),
            "responseMimeType": "application/json",
        }
        if pcfg.get("thinking_level"):
            gcfg["thinkingConfig"] = {"thinkingLevel": str(pcfg["thinking_level"])}
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gcfg}

        r: requests.Response | None = None
        for tok_attempt in range(2):
            token = self._get_vertex_access_token(sa, sess, force_refresh=(tok_attempt > 0))
            r = sess.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=(10, self._timeout_for(pcfg)),
            )
            if r.status_code == 401 and tok_attempt == 0:
                with self._vertex_token_lock:
                    self._vertex_tokens.pop(sa["client_email"], None)
                continue
            r.raise_for_status()
            return _parse_gemini_like_response(r)
        assert r is not None
        r.raise_for_status()
        return _parse_gemini_like_response(r)

    def _call_openai_compat(
        self,
        prov_name: str,
        pcfg: dict[str, Any],
        key: str,
        sess: requests.Session,
        images: list[bytes],
        prompt: str,
        reference_images: list[tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        use_collage = bool(pcfg.get("single_image_collage") and len(images) > 1)
        send_images = [_stitch_images_horizontal(images)] if use_collage else images
        ref_list = [] if pcfg.get("single_image_collage") else list(reference_images or [])
        url = f"{pcfg['base_url'].rstrip('/')}/chat/completions"
        content: list[dict[str, Any]] = []
        for lbl, jb in ref_list:
            content.append({"type": "text", "text": str(lbl or "reference_sheet")})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jb).decode("ascii")}})
        if len(send_images) == 1 and not ref_list:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(send_images[0]).decode("ascii")}})
            content.append({"type": "text", "text": prompt})
        else:
            for idx, jb in enumerate(send_images):
                content.append({"type": "text", "text": f"frame_{idx + 1}"})
                content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jb).decode("ascii")}})
            content.append({"type": "text", "text": prompt})

        now_ts = time.time()
        with self._route_lock:
            cached_entry = self._resolved_models.get(prov_name)
        raw_candidates: list[str] = []
        if cached_entry is not None and now_ts < cached_entry[1]:
            raw_candidates.append(cached_entry[0])
        raw_candidates.append(str(pcfg["model"]))
        for fb_m in (pcfg.get("fallback_models") or []):
            if str(fb_m or "").strip():
                raw_candidates.append(str(fb_m).strip())
        alt_model = str(pcfg.get("fallback_model") or "").strip()
        if alt_model:
            raw_candidates.append(alt_model)
        models_to_try: list[str] = []
        for cand_m in raw_candidates:
            if cand_m and cand_m not in models_to_try:
                models_to_try.append(cand_m)

        r = None
        for midx, m_id in enumerate(models_to_try):
            drop_reasoning = False
            while True:
                payload: dict[str, Any] = {
                    "model": m_id,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": int(pcfg.get("max_output_tokens", 900)),
                }
                if pcfg.get("reasoning_effort") and not drop_reasoning:
                    payload["reasoning_effort"] = str(pcfg["reasoning_effort"])
                elif not pcfg.get("omit_temperature"):
                    payload["temperature"] = 0
                if "enable_thinking" in pcfg:
                    payload["enable_thinking"] = bool(pcfg["enable_thinking"])
                r = sess.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    timeout=(10, self._timeout_for(pcfg)),
                )
                if r.status_code == 400 and payload.get("reasoning_effort") and "reasoning_effort" in (r.text or ""):
                    drop_reasoning = True
                    continue
                break
            if r.status_code in (400, 404) and (midx + 1) < len(models_to_try) and not QUOTA_EXHAUSTED_RE.search(r.text or ""):
                continue
            r.raise_for_status()
            with self._route_lock:
                self._resolved_models[prov_name] = (m_id, time.time() + 1800.0)
            break
        assert r is not None
        obj = r.json()
        choice0 = (obj.get("choices") or [{}])[0]
        msg_obj = choice0.get("message") or {}
        msg_content = msg_obj.get("content")
        if isinstance(msg_content, list):
            raw_txt = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in msg_content
            )
        else:
            raw_txt = str(msg_content or "")
        if not raw_txt.strip() and msg_obj.get("reasoning_content"):
            raw_txt = str(msg_obj.get("reasoning_content") or "")
        if str(choice0.get("finish_reason") or "") == "content_filter" and not raw_txt.strip():
            raise ContentPolicyBlockedError("OpenAI content_filter")
        u = obj.get("usage") or {}
        in_tok = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        out_tok = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        tot_tok = int(u.get("total_tokens") or (in_tok + out_tok))
        res = parse_json_obj(raw_txt)
        res["_usage"] = {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": max(tot_tok, in_tok + out_tok)}
        return res

    def _cache_key_for(
        self,
        prompt_tag: str,
        images: list[bytes],
        prompt: str,
        reference_images: list[tuple[str, bytes]] | None = None,
    ) -> str:
        kh = hashlib.sha256(f"v3nas|{prompt_tag}|{prompt}".encode("utf-8"))
        for jb in images:
            kh.update(jb)
        if reference_images:
            kh.update(b"|refs|")
            for lbl, rb in reference_images:
                kh.update(str(lbl).encode("utf-8"))
                kh.update(rb)
        return kh.hexdigest()

    def _invoke_provider(
        self,
        prov_name: str,
        pcfg: dict[str, Any],
        cache_key: str,
        tier: str,
        prompt_tag: str,
        images: list[bytes],
        prompt: str,
        reference_images: list[tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        key = self._read_secret(pcfg["secret_file"])
        if not key:
            raise RuntimeError(f"Missing secret for {prov_name}")
        eff_pcfg = self._effective_pcfg(pcfg, tier)
        disp_model = str(pcfg.get("display_model") or pcfg["model"])
        sess = self._session_for(prov_name)
        last_exc: Exception = RuntimeError(f"Unknown failure for {prov_name}")
        with self._route_lock:
            self._inflight[prov_name] = self._inflight.get(prov_name, 0) + 1
        try:
            # 2 transport attempts on transient socket/EOF resets before triggering cooldown
            for attempt in range(2):
                t0 = time.time()
                try:
                    if eff_pcfg["kind"] == "gemini":
                        parsed = self._call_gemini(eff_pcfg, key, sess, images, prompt, reference_images=reference_images)
                    elif eff_pcfg["kind"] == "vertex":
                        parsed = self._call_vertex(eff_pcfg, key, sess, images, prompt, reference_images=reference_images)
                    else:
                        parsed = self._call_openai_compat(prov_name, eff_pcfg, key, sess, images, prompt, reference_images=reference_images)
                    dt_s = round(time.time() - t0, 3)
                    usage = parsed.pop("_usage", None)
                    self.db.record_vlm_result(prov_name, disp_model, True, dt_s, usage=usage)
                    self.db.put_vlm_cache(cache_key, tier, prov_name, disp_model, prompt_tag, parsed, dt_s, usage=usage)
                    parsed["_provider"] = prov_name
                    parsed["_model"] = disp_model
                    return parsed
                except ContentPolicyBlockedError as exc:
                    dt_s = round(time.time() - t0, 3)
                    refusal = normalize_vlm_response(
                        {
                            "vlm_refused": True,
                            "review_status": "pending_review",
                            "refusal_reason": redact_secrets(str(exc)),
                            "suggested_category": "daily_care",
                            "child_box": [0.0, 0.0, 0.0, 0.0],
                            "local_care_fallback": "pending_review",
                        },
                        model_name="vlm_refusal_pending",
                    )
                    self.db.put_vlm_cache(
                        cache_key, tier, prov_name, "vlm_refusal_pending", prompt_tag, refusal, dt_s
                    )
                    refusal["_provider"] = prov_name
                    refusal["_model"] = "vlm_refusal_pending"
                    return refusal
                except requests.HTTPError as exc:
                    last_exc = exc
                    code = exc.response.status_code if exc.response is not None else 0
                    raw_err = exc.response.text if exc.response is not None else str(exc)
                    safe_msg = redact_secrets(f"HTTP {code}: {raw_err}")
                    if _is_hard_quota_exhausted(code, raw_err):
                        self.db.record_vlm_result(
                            prov_name,
                            disp_model,
                            False,
                            0.0,
                            f"[QUOTA_EXHAUSTED] {safe_msg}"[:280],
                            self.quota_cooldown_s,
                        )
                        break
                    if code == 429:
                        self.db.record_vlm_result(
                            prov_name,
                            disp_model,
                            False,
                            0.0,
                            f"[RATE_LIMIT] {safe_msg}"[:280],
                            self.rate_limit_cooldown_s,
                        )
                        break
                    if code in (401, 403, 404, 400):
                        self.db.record_vlm_result(prov_name, disp_model, False, 0.0, safe_msg, self.cooldown_s)
                        break
                    if attempt == 0:
                        time.sleep(1.2)
                        continue
                    self.db.record_vlm_result(prov_name, disp_model, False, 0.0, safe_msg, 300)
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    last_exc = exc
                    safe_msg = redact_secrets(f"{type(exc).__name__}: {exc}")
                    if isinstance(exc, requests.Timeout):
                        self._reset_session(prov_name)
                        self.db.record_vlm_result(prov_name, disp_model, False, 0.0, safe_msg, self.cooldown_s)
                        break
                    if attempt == 0 and isinstance(exc, requests.RequestException):
                        # Reset thread-local session before retry after TLS EOF
                        self._reset_session(prov_name)
                        sess = self._session_for(prov_name)
                        time.sleep(1.2)
                        continue
                    self.db.record_vlm_result(prov_name, disp_model, False, 0.0, safe_msg, 300)
                    break
        finally:
            with self._route_lock:
                self._inflight[prov_name] = max(0, self._inflight.get(prov_name, 1) - 1)
        raise last_exc

    def ask(
        self,
        prompt_tag: str,
        images: list[bytes],
        prompt: str,
        tier: str = "fast",
        reference_images: list[tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("Empty image batch")
        cache_key = self._cache_key_for(prompt_tag, images, prompt, reference_images=reference_images)
        cached = self.db.get_vlm_cache(cache_key)
        if cached is not None:
            return cached

        chain = self._order_chain_for_refs(self._select_provider_chain(tier), bool(reference_images))
        providers = self.vlm_cfg.get("providers", {})
        last_exc: Exception | None = None

        for prov_name in chain:
            pcfg = providers.get(prov_name)
            if not pcfg or not self._provider_available(prov_name, pcfg):
                continue
            try:
                if reference_images:
                    return self._invoke_provider(
                        prov_name, pcfg, cache_key, tier, prompt_tag, images, prompt,
                        reference_images=reference_images,
                    )
                return self._invoke_provider(prov_name, pcfg, cache_key, tier, prompt_tag, images, prompt)
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"All VLM providers unavailable for tier={tier}: {redact_secrets(str(last_exc))}")

    def ask_batch(self, batch: list[tuple[Any, ...]], tier: str = "fast") -> dict[Any, dict[str, Any]]:
        """Work-stealing pull pool: each available provider grabs tasks continuously and auto-exits when quota runs out."""
        results: dict[Any, dict[str, Any]] = {}
        if not batch:
            return results

        def _unpack_task(t: tuple[Any, ...]) -> tuple[Any, str, list[bytes], str, list[tuple[str, bytes]] | None]:
            if len(t) >= 5:
                return t[0], t[1], t[2], t[3], t[4]
            return t[0], t[1], t[2], t[3], None

        ask_overridden = getattr(self.ask, "__func__", None) is not CloudVLMRouter.ask
        if ask_overridden or not self.vlm_cfg.get("providers"):
            conc = max(1, int(self.vlm_cfg.get(f"{tier}_concurrency", 6 if tier == "fast" else 4)))
            try:
                sig_params = inspect.signature(self.ask).parameters
                accepts_refs = "reference_images" in sig_params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
                )
            except (TypeError, ValueError):
                accepts_refs = False

            def _run_compat(task: tuple[Any, ...]):
                item_key, ptag, imgs, prm, ref_imgs = _unpack_task(task)
                try:
                    if ref_imgs and accepts_refs:
                        return item_key, self.ask(ptag, imgs, prm, tier=tier, reference_images=ref_imgs)
                    return item_key, self.ask(ptag, imgs, prm, tier=tier)
                except Exception:
                    return item_key, None

            with ThreadPoolExecutor(max_workers=min(conc, len(batch))) as pool:
                for ik, ans in pool.map(_run_compat, batch):
                    if ans is not None:
                        results[ik] = ans
            return results

        pending: list[dict[str, Any]] = []
        for raw_task in batch:
            item_key, prompt_tag, images, prompt, ref_imgs = _unpack_task(raw_task)
            if not images:
                continue
            ck = self._cache_key_for(prompt_tag, images, prompt, reference_images=ref_imgs)
            cached = self.db.get_vlm_cache(ck)
            if cached is not None:
                results[item_key] = cached
            else:
                pending.append({
                    "key": item_key,
                    "tag": prompt_tag,
                    "images": images,
                    "prompt": prompt,
                    "ref_images": ref_imgs,
                    "cache_key": ck,
                    "tried": set(),
                })

        if not pending:
            return results

        par_tiers = self.vlm_cfg.get("parallel_tiers", {})
        tiers_cfg = self.vlm_cfg.get("tiers", {})
        pool_names = list(par_tiers.get(tier) or tiers_cfg.get(tier) or [])
        if not pool_names and tier == "director":
            pool_names = list(par_tiers.get("pro") or tiers_cfg.get("pro") or [])
        providers = self.vlm_cfg.get("providers", {})
        avail_provs = [p for p in pool_names if p in providers and self._provider_available(p, providers[p])]

        def invoke(name: str, task: dict[str, Any]) -> dict[str, Any]:
            if task.get("ref_images"):
                return self._invoke_provider(
                    name, providers[name], task["cache_key"], tier, task["tag"], task["images"], task["prompt"],
                    reference_images=task["ref_images"],
                )
            return self._invoke_provider(
                name, providers[name], task["cache_key"], tier, task["tag"], task["images"], task["prompt"],
            )

        if avail_provs:
            # Group by logical model family so one model with multiple endpoints never monopolizes slots.
            grouped_batch: dict[str, list[str]] = {}
            for p in avail_provs:
                grouped_batch.setdefault(self._logical_model(providers[p], p), []).append(p)
            model_groups = list(grouped_batch.items())
            with self._route_lock:
                b_rr = self._rr_counter.get(f"batch:{tier}", 0)
                self._rr_counter[f"batch:{tier}"] = b_rr + 1
                b_shift = b_rr % len(model_groups)
                model_groups = model_groups[b_shift:] + model_groups[:b_shift]
                max_per_model = max(1, int(self.vlm_cfg.get("max_slots_per_model", 2)))
                per_model_slots: list[list[str]] = []
                for m_key, m_provs in model_groups:
                    sub_key = f"sub:{tier}:{m_key}"
                    sub_rr = self._rr_counter.get(sub_key, 0)
                    self._rr_counter[sub_key] = sub_rr + 1
                    sub_s = sub_rr % len(m_provs)
                    m_rot = m_provs[sub_s:] + m_provs[:sub_s]
                    m_limits = {p: max(1, int(providers[p].get("worker_slots", 1))) for p in m_rot}
                    m_slots = [p for t in range(max(m_limits.values())) for p in m_rot if t < m_limits[p]]
                    if len(model_groups) > 1:
                        m_slots = m_slots[:max_per_model]
                    per_model_slots.append(m_slots)
            slots = [
                ms[turn]
                for turn in range(max(len(ms) for ms in per_model_slots))
                for ms in per_model_slots
                if turn < len(ms)
            ]
            running = {}
            unavailable: set[str] = set()

            def _can_take_task(prov_name: str, task: dict[str, Any]) -> bool:
                if prov_name in task["tried"]:
                    return False
                if task.get("ref_images") and providers[prov_name].get("single_image_collage"):
                    if any(
                        (not providers[p].get("single_image_collage"))
                        and p not in unavailable
                        and p not in task["tried"]
                        for p in avail_provs
                    ):
                        return False
                return True

            with ThreadPoolExecutor(max_workers=min(len(slots), len(pending))) as pool:
                while pending or running:
                    for name in slots.copy():
                        if not pending:
                            break
                        if name in unavailable:
                            continue
                        idx = next((i for i, task in enumerate(pending) if _can_take_task(name, task)), None)
                        if idx is None:
                            continue
                        if not self._provider_available(name, providers[name]):
                            unavailable.add(name)
                            continue
                        task = pending.pop(idx)
                        task["tried"].add(name)
                        slots.remove(name)
                        running[pool.submit(invoke, name, task)] = (name, task)
                    if not running:
                        break
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        name, task = running.pop(future)
                        slots.append(name)
                        try:
                            results[task["key"]] = future.result()
                        except Exception:
                            pending.append(task)

        for task in pending:
            fb_chain = self._order_chain_for_refs(self._select_provider_chain(tier), bool(task.get("ref_images")))
            for name in fb_chain:
                if name in task["tried"] or name not in providers or not self._provider_available(name, providers[name]):
                    continue
                task["tried"].add(name)
                try:
                    results[task["key"]] = invoke(name, task)
                    break
                except Exception:
                    continue
        return results
