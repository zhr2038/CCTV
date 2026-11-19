"""Four-stage Cinematic Daily Highlight Director & Screenwriter (Highest Thinking Level)."""
from __future__ import annotations

import datetime as dt
import inspect
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .cloud_vlm import is_cloud_allowed
from .db import ArchiveDB, is_unconfirmed_refusal
from .family_faces import (
    FEMALE_ROLES,
    KINSHIP_ROLES,
    MALE_ROLES,
    extract_high_conf_vlm_roles,
    sanitize_caption_kinship,
)
from .media_nas import crop_16_9


ACT_WINDOWS: list[tuple[int, int, str]] = [
    (0, 10, "第一幕 · 晨光初醒"),
    (10, 15, "第二幕 · 小小探索家"),
    (15, 19, "第三幕 · 今日欢笑与高光"),
    (19, 24, "第四幕 · 温柔夜色"),
]

BLOCKED_FACE_VIS = frozenset({
    "edge_clipped",
    "tiny_or_distant",
    "occluded",
    "no_baby",
    "back_only",
})

VALID_MILESTONE_CATS = frozenset({
    "standing",
    "walking",
    "sofa_descent",
    "toy_play",
    "happy_interaction",
    "daily_care",
    "kitchen_visit",
})

PROMPT_HIGHLIGHT_COARSE = """你在为宝宝（约0-14个月）“每日成长纪念短片”做第一道画质与构图筛选。
【判定标准】
- 只要宝宝**头部/五官神态及上半身清晰可见**（即使是半身特写、被抱着下半身自然出画、或小床栏杆仅挡住小腿/脚），属于正常优质镜头：child_clear 填 "yes"，edge_clipped 填 "no"！
- 只有当宝宝**头部/脸被画面边缘截断只剩半边脸**、脸被大人背影完全挡住、缩在远处角落模糊不清、或仅有成人无宝宝时，才淘汰（child_clear 填 partial/no，edge_clipped 填 yes）。
- 描述大人时禁止主观猜测“妈妈/爸爸”，也禁用“男子/女子/家长”等冰冷词，统一用“家人”或省略人物称呼。
只返回单个JSON对象：
{
  "child_clear": "yes|partial|blocked|no",
  "face_clear": "yes|side|no",
  "edge_clipped": "yes|no",
  "awake_active": "yes|sleeping",
  "box_2d": [ymin, xmin, ymax, xmax],
  "moment_worth": 0,
  "brief_zh": "16字以内客观描述"
}
moment_worth(0-5)：5=清晰生动大笑/说话/站立互动；4=正面或侧脸清晰游玩/温馨怀抱；3=构图清楚清醒活动；1-2=全程背影或过暗；0=头部边缘残缺/无宝宝/静睡。"""

PROMPT_DIRECTOR_SHOT_EDIT = """你是一位温暖而专业的家庭纪录片剪辑指导（Director of Photography & Editor），正在逐段精审宝宝（约0-14个月）的一段多帧连续镜头（右上角标注了相对当前窗口起点的秒数 `+X.Xs`）。
请判断是否配入选《每日珍藏成长短剧》，并给出精确动作起止切点与景别：

【一票否决硬标准】若全段满足任一项，usable_for_highlight 必须填 "no"：
1. 宝宝头部/脸部被画面边缘截断只露极少部分（注意：正常的半身/中景胸上出画、被怀抱遮住下半身、或婴儿床木栏略微挡住小腿，**绝不算**残缺，若五官与神态清晰可正常给 usable_for_highlight="yes" 且 child_clear="yes"）；
2. 宝宝被大人背影长时间挡住头部、或全段几乎只见背影/头顶看不清神态；
3. 画面在角落缩成极小一团且模糊，或仅有成人走动；
4. 全段静止不动/纯夜间闭眼熟睡无互动。

【精剪定格任务】若通过（usable_for_highlight="yes"）：
1. 在视频片段跨度内选出一段**动作完整、表情最佳**的 [best_in_rel_s, best_out_rel_s]（必须遵循图中右上角 `+X.Xs` 秒数标尺，长度建议 4.5～8.8 秒；切忌在宝宝刚笑起来或刚要迈步/说话时突然掐断，留出情绪起承余韵）。
2. 选择最佳 `shot_scale`（景别搭配才有电影呼吸感）：
   - "wide": 有家人陪伴互动、抱起、在厨房/客厅穿行，需要完整空间与亲子关系时选 wide（不裁掉大人）；
   - "medium": 宝宝坐玩/扶站/餐椅互动，兼顾环境与人物神态；
   - "closeup": 宝宝笑容、抓握玩具、好奇张望、叫“爸/妈”嘴型特写清晰时选 closeup。
3. 给出 `box_2d` [ymin, xmin, ymax, xmax]（0..1000整数归一化），稳稳包住 [best_in_rel_s, best_out_rel_s] 期间宝宝的主要活动区域。
4. 若换尿布/擦身过程意外露出隐私部位，privacy_exposed 填 "yes" 并框出 privacy_box_2d，否则填 "no" 和 [0,0,0,0]。
5. `cinematic_score` 请严格填写 **1 到 5 整数**（5=神态生动极佳，4=清晰温暖有亮点，3=自然可用，1-2=平淡或瑕疵）。
6. 用温暖、客观、细腻的中文写出 `director_note_zh`（说清宝宝做了什么、开心之处、展现了什么小变化或小脾气，禁止出现“未搀扶/女子/男子/监控”等冰冷词）。
7. **人物称呼铁律**：切勿把抱/照料宝宝的女性默认猜成“妈妈”、也不要把男性默认猜成“爸爸”（家中常有奶奶、爷爷、姥姥、姥爷、小姨等家人照顾宝宝）。只有当附带的家庭成员对照卡或本地人脸识别结果明确高置信指出具体成员（如奶奶/妈妈/爸爸/爷爷/姥姥/姥爷/小姨）时，才在 `director_note_zh` 写出对应称呼并填写 `caregivers_seen`；拿不准或只见背影/低头/局部时，描述里一律用“家人”或省略称呼（例如“在怀里”、“身旁护着”），`caregivers_seen` 填 `[{"role":"unknown","confidence":"low"}]`。

只返回单个JSON对象：
{
  "usable_for_highlight": "yes|no",
  "child_clear": "yes|partial|blocked|no",
  "face_visibility": "clear_front|clear_three_quarter|clear_action_profile|edge_clipped|tiny_or_distant|occluded|no_baby",
  "best_in_rel_s": 0.0,
  "best_out_rel_s": 6.5,
  "shot_scale": "closeup|medium|wide",
  "pan_style": "gentle_push|gentle_pull|steady",
  "box_2d": [ymin, xmin, ymax, xmax],
  "privacy_exposed": "yes|no",
  "privacy_box_2d": [ymin, xmin, ymax, xmax],
  "caregivers_seen": [{"role": "妈妈|爸爸|奶奶|爷爷|姥姥|姥爷|小姨|月嫂|unknown", "confidence": "high|medium|low"}],
  "categories": ["standing"|"walking"|"sofa_descent"|"toy_play"|"happy_interaction"|"daily_care"|"kitchen_visit"|"none"],
  "independence": "independent|assisted|unclear",
  "emotion_tag": "joy|curious|focused|speech|affection|calm",
  "cinematic_score": 4,
  "director_note_zh": "35字以内导演剪辑手记"
}"""

PROMPT_DIRECTOR_SCREENPLAY = """你是一位擅长家庭成长微电影的总导演与编剧，正在用【电影蒙太奇思维】将宝宝（约0-14个月）一整天通过严格质检的高光镜头，创作编排为一部值得家人收藏一辈子的 **1分25秒～2分钟（13～20镜）温暖成长短剧**。

【创作要求】
1. 必须按时间线讲述“今天宝宝是怎么度过的”，涵盖四幕叙事弧线（若某时段素材较少可自然衔接）：
   - 第一幕 · 晨光初醒（早间醒来、开心微笑或清晨探索）
   - 第二幕 · 小小探索家（上午/午后好奇摸索、玩玩具、扶站尝试、厨房旁围观）
   - 第三幕 · 今日欢笑与高光（下午至傍晚互动峰值、大笑、珍贵的“叫爸/妈”童声里程碑）
   - 第四幕 · 温柔夜色（晚间洗护抚触、亲子陪伴、宁静收尾）
2. 镜头调度与节奏变化：
   - 第1镜作为开篇定场，建议留足 6.0～7.5 秒以容纳片头字幕温柔淡出；
   - 善用景别交替（wide 与 medium/closeup 穿插）和真实转场建议：
     * "chapter_dissolve": 仅用于跨入新的一幕（首镜入场用 "chapter_dissolve"）；
     * "soft_cut": 同房间相邻时间延续动作（仅 0.2 秒轻切换）；
     * "warm_dissolve": 不同场景或时间跳跃的温柔叠化（0.55 秒）。
   - 单镜时长 trim_dur_s 控制在 4.5～8.8 秒，整部短片净时长控制在 85～120 秒（从候选镜头中选择 13～20 镜；如有 speech 纯童声镜头，必须保留为情感高潮）。
3. 编剧台词与字幕风格（lyric_zh，10～24个汉字）：
   - 只写画面和声音真实发生过的温暖细节，不虚构未发生的动作；
   - 像绘本或家庭日记一样有温度、有细节感（例：“醒来的第一件事，是对着晨光笑出声”、“踮脚扶住桌沿，又多看了一眼更大的世界”、“这一声清脆的‘爸’，是今天最亮的光”）；
   - **称呼严格尊重事实**：只有当分镜清单的 `people` 字段或手记中明确核实了人物（如 `["奶奶"]`、`["爸爸"]`、`["妈妈"]`）时才可写出具体称呼，切勿把未确认人物默认写成“妈妈/爸爸”，未核实身份时统一用“家人”或省略主语；
   - 严禁出现“男子/女子/未搀扶/监控/镜头捕捉/1.25×”等机械监控术语。

只返回单个JSON对象：
{
  "film_title_zh": "16字以内富有画面感的今日短片片名",
  "logline_zh": "55字以内整天故事简介，让家人一看就知道今天发生了什么",
  "joy_summary_zh": "35字以内总结今天什么事让宝宝最开心",
  "growth_changes_zh": "35字以内总结今天显现的变化、小动作或新技能（严格基于事实）",
  "closing_epilogue_zh": "18字以内片尾温暖寄语（例如：第337天，慢慢长大的每一秒都算数。）",
  "edit_list": [
    {
      "shot_id": "S01",
      "chapter_zh": "第一幕 · 晨光初醒",
      "trim_dur_s": 6.5,
      "shot_scale": "medium",
      "pan_style": "gentle_push",
      "transition": "chapter_dissolve",
      "lyric_zh": "10～24字温暖编剧字幕"
    }
  ]
}"""


def act_for_hour(hour: int | None) -> str:
    if hour is None:
        return ACT_WINDOWS[1][2]
    for lo, hi, title in ACT_WINDOWS:
        if lo <= hour < hi:
            return title
    return ACT_WINDOWS[-1][2]


def _clean_warm_text(text: str, max_len: int = 28) -> str:
    s = str(text or "").strip()
    s = re.sub(r"[（(]\s*\d+(?:\.\d+)?\s*[×xX]\s*[）)]", "", s)
    s = re.sub(r"(?:但|且)?(?:成人)?未搀扶(?:宝宝)?", "", s)
    s = re.sub(r"监控(?:画面|录像|镜头)?", "", s)
    s = s.replace("女子", "家人").replace("男子", "家人")
    s = re.sub(r"\s{2,}", " ", s).strip(" ·，；")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip("，。；、 ") + "…"
    return s


def _box_quality_adjustment(box: list[float] | None, *, cam_role: str = "") -> tuple[float, bool]:
    """Return (score_adjustment, severely_edge_clipped_or_tiny) for Stage-1 YOLO boxes."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0, False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return 0.0, False
    if x2 <= x1 or y2 <= y1:
        return 0.0, False
    bw, bh = x2 - x1, y2 - y1
    area = bw * bh
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    adj = 0.0
    if 0.03 <= area <= 0.34:
        adj += 0.25
    elif area < 0.012:
        adj -= 0.40
    elif area > 0.52:
        adj -= 0.30
    if cam_role == "bedside_crib" and x1 >= 0.68:
        adj -= 0.55
    edge_touch = (x1 <= 0.015) or (x2 >= 0.985) or (y1 <= 0.015) or (y2 >= 0.985)
    severely_edge = bool(edge_touch and (bw < 0.14 or bh < 0.18 or cx < 0.06 or cx > 0.94 or cy < 0.06 or cy > 0.94))
    if severely_edge:
        adj -= 0.45
    elif edge_touch:
        adj -= 0.15
    return adj, severely_edge


def _locate_burst_peaks_in_row(row: dict[str, Any], max_peaks: int = 2) -> list[tuple[float, list[float] | None, float]]:
    """Locate distinct top vocalization/laughter burst timestamps inside a clip (critical for 15m bedside bundles)."""
    cands = json.loads(row.get("candidates_json") or "[]")
    bursts = json.loads(row.get("bursts_json") or "[]")
    if not bursts:
        return []
    scored_b: list[tuple[float, dict[str, Any]]] = []
    for b in bursts:
        sc = (
            float(b.get("y_laughter") or 0.0) * 2.5
            + float(b.get("y_child_babble") or 0.0) * 1.4
            + float(b.get("snr_ratio") or 0.0) * 0.05
        )
        scored_b.append((sc, b))
    scored_b.sort(key=lambda x: -x[0])
    out: list[tuple[float, list[float] | None, float]] = []
    for sc, b in scored_b:
        t_b = 0.5 * (
            float(b.get("burst_start_s") if b.get("burst_start_s") is not None else (b.get("start_s") or 0.0))
            + float(b.get("burst_end_s") if b.get("burst_end_s") is not None else (b.get("end_s") or b.get("burst_start_s") or 0.0))
        )
        if any(abs(t_b - pt[0]) < 120.0 for pt in out):
            continue
        near_pc = min(cands, key=lambda pc: abs(float(pc.get("t") or 0.0) - t_b), default=None)
        if near_pc and abs(float(near_pc.get("t") or 0.0) - t_b) <= 20.0 and near_pc.get("luma", 0) >= 55:
            out.append((t_b, near_pc.get("box"), float(near_pc.get("score") or 0.85)))
        else:
            out.append((t_b, None, min(1.0, 0.65 + sc * 0.1)))
        if len(out) >= max_peaks:
            break
    return out


def _locate_peak_in_row(
    row: dict[str, Any],
    *,
    prefer_laughter: bool = False,
    cam_role: str = "",
) -> tuple[float, list[float] | None, float]:
    cands = json.loads(row.get("candidates_json") or "[]")
    aud_s = json.loads(row.get("audio_summary_json") or "{}")
    dur = max(1.0, float(row.get("duration") or 10.0))

    if prefer_laughter:
        b_peaks = _locate_burst_peaks_in_row(row, max_peaks=1)
        if b_peaks:
            return b_peaks[0]

    good_pcs: list[tuple[float, dict[str, Any]]] = []
    for pc in cands:
        if pc.get("luma", 0) < 65 or pc.get("sharp", 0) < 35:
            continue
        adj, _ = _box_quality_adjustment(pc.get("box"), cam_role=cam_role)
        good_pcs.append((float(pc.get("score") or 0.0) + adj, pc))
    if good_pcs:
        good_pcs.sort(key=lambda x: -x[0])
        top = good_pcs[0][1]
        return float(top.get("t") or 0.0), top.get("box"), good_pcs[0][0]

    segs = aud_s.get("segments") or []
    if segs:
        mid_s = 0.5 * (float(segs[0][0]) + float(segs[0][1]))
        return min(max(0.5, mid_s), dur - 0.5), None, 0.5
    return min(dur * 0.5, max(0.5, dur - 0.5)), None, 0.4


def _hour_and_epoch(
    row: dict[str, Any],
    c_cfg: dict[str, Any],
    t_s: float,
    day: str,
    effective_wall_ts_fn: Callable[[dict[str, Any], dict[str, Any]], str | None],
    parse_wall_epoch_fn: Callable[[str | None, float], float | None],
) -> tuple[str | None, float | None, int | None]:
    eff_wts = effective_wall_ts_fn(row, c_cfg)
    ep = parse_wall_epoch_fn(eff_wts, float(t_s))
    clip_hr: int | None = None
    if eff_wts:
        try:
            clip_hr = (dt.datetime.fromisoformat(eff_wts) + dt.timedelta(seconds=float(t_s))).hour
        except ValueError:
            clip_hr = None
    return eff_wts, ep, clip_hr


def _build_window(center_t: float, clip_dur: float, *, target_span: float = 12.0) -> tuple[float, float]:
    span = min(max(4.0, clip_dur), target_span)
    w_start = max(0.0, center_t - span * 0.42)
    w_end = min(clip_dur, w_start + span)
    if (w_end - w_start) < span and clip_dur >= span:
        w_start = max(0.0, w_end - span)
    return round(w_start, 2), round(w_end, 2)


def stamp_timecode_badges(images: list[bytes], rel_offsets_s: list[float]) -> list[bytes]:
    """Burn crisp top-right relative timestamp labels (+0.0s .. +X.Xs) onto frame copies for HIGH-thinking editors."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    out: list[bytes] = []
    for jb, rel_s in zip(images, rel_offsets_s):
        try:
            im = Image.open(io.BytesIO(jb)).convert("RGB")
            draw = ImageDraw.Draw(im)
            lbl = f"+{max(0.0, float(rel_s)):.1f}s"
            tw = int(draw.textlength(lbl, font=font))
            bw = tw + 16
            bh = 22
            x0 = max(4, im.width - bw - 8)
            y0 = 8
            draw.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=6, fill=(14, 16, 20))
            draw.text((x0 + 8, y0 + 5), lbl, font=font, fill=(255, 240, 200))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            out.append(buf.getvalue())
        except Exception:
            out.append(jb)
    return out


def build_storyboard_sheets(verified_shots: list[dict[str, Any]], max_per_sheet: int = 12) -> list[bytes]:
    """Tile midpoint frames from verified shots into labeled 1280px contact sheets for Master Director review."""
    from PIL import Image, ImageDraw, ImageFont

    if not verified_shots:
        return []
    font = ImageFont.load_default()
    sheets: list[bytes] = []
    for base in range(0, len(verified_shots), max_per_sheet):
        chunk = verified_shots[base : base + max_per_sheet]
        cols = min(4, max(2, int(math.ceil(math.sqrt(len(chunk))))))
        rows = int(math.ceil(len(chunk) / float(cols)))
        tw, th = 310, 174
        gap = 8
        canvas = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), (20, 18, 16))
        for idx, sh in enumerate(chunk):
            r_i, c_i = divmod(idx, cols)
            x0 = gap + c_i * (tw + gap)
            y0 = gap + r_i * (th + gap)
            jb = sh.get("poster_jpg")
            if jb:
                try:
                    tile = Image.open(io.BytesIO(jb)).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
                except Exception:
                    tile = Image.new("RGB", (tw, th), (42, 38, 34))
            else:
                tile = Image.new("RGB", (tw, th), (42, 38, 34))
            draw = ImageDraw.Draw(tile)
            tag = f"{sh['shot_id']} {sh['hh_mm']} {sh['cam_short']}"
            draw.rectangle([4, 4, min(tw - 4, int(draw.textlength(tag, font=font)) + 16), 24], fill=(16, 14, 13))
            draw.text((10, 8), tag, font=font, fill=(255, 240, 200))
            canvas.paste(tile, (x0, y0))
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=85)
        sheets.append(buf.getvalue())
    return sheets


def harvest_daily_candidates(
    day: str,
    rows: list[dict[str, Any]],
    cams: dict[str, Any],
    *,
    speech_items: list[tuple[dict[str, Any], dict[str, Any]]],
    vlm_photos: list[dict[str, Any]],
    kit_reviews: dict[str, dict[str, Any]],
    effective_wall_ts_fn: Callable[[dict[str, Any], dict[str, Any]], str | None],
    parse_wall_epoch_fn: Callable[[str | None, float], float | None],
    resolve_source_path_fn: Callable[[dict[str, Any]], Path | None],
    has_providers: bool,
) -> list[dict[str, Any]]:
    """Gather diverse waking-day highlight candidates across speech, photos, laughs, kitchen & hourly activity."""
    row_by_rel = {str(r["source_rel"]): r for r in rows}
    raw_pool: list[dict[str, Any]] = []
    seen_slots: set[tuple[str, int]] = set()
    is_small_stub = (not has_providers) and (len(rows) <= 20)

    def _add_cand(
        *,
        source_kind: str,
        priority: float,
        r: dict[str, Any],
        src: Path,
        center_t_s: float,
        w_start: float,
        w_end: float,
        box: list[float] | None,
        eff_wts: str | None,
        wall_ep: float | None,
        clip_hr: int | None,
        pre_verified: bool = False,
        must_include: bool = False,
        boost_vocal: bool = False,
        speech_meta: dict[str, Any] | None = None,
        photo_vlm: dict[str, Any] | None = None,
        fallback_note: str = "",
    ) -> None:
        cam = str(r["camera"])
        c_cfg = cams.get(cam, {})
        if not c_cfg.get("milestone_watch", True) and source_kind != "speech":
            return
        slot = (str(r["source_rel"]), int(round(center_t_s / 8.0)))
        if slot in seen_slots:
            return
        seen_slots.add(slot)
        span = max(2.0, w_end - w_start)
        test_offsets = sorted({
            round(w_start + 0.3, 2),
            round(w_start + span * 0.5, 2),
            round(max(w_start + 0.4, w_end - 0.3), 2),
        })
        cloud_ok = is_cloud_allowed(c_cfg, eff_wts, test_offsets)
        if not cloud_ok and not must_include:
            return
        is_waking = 1 if (clip_hr is None or 7 <= clip_hr < 21) else 0
        raw_pool.append({
            "source_kind": source_kind,
            "priority": float(priority),
            "row": r,
            "src": src,
            "cam_id": cam,
            "cam_name": str(c_cfg.get("name", cam)),
            "c_cfg": c_cfg,
            "center_t_s": round(float(center_t_s), 2),
            "w_start": round(float(w_start), 2),
            "w_end": round(float(w_end), 2),
            "box": box,
            "eff_wts": eff_wts or f"{day}T12:00:00",
            "wall_ep": wall_ep,
            "clip_hr": clip_hr,
            "bucket_2h": (clip_hr // 2) if (clip_hr is not None and 7 <= clip_hr < 21) else None,
            "bucket_3h": (clip_hr // 3) if (clip_hr is not None and 7 <= clip_hr < 21) else None,
            "is_waking": is_waking,
            "pre_verified": pre_verified,
            "must_include": must_include,
            "boost_vocal": boost_vocal,
            "cloud_ok": cloud_ok,
            "speech_meta": speech_meta,
            "photo_vlm": photo_vlm,
            "fallback_note": fallback_note,
        })

    # Precompute waking bedside bundle intervals so speech hits on wide ceiling cams also harvest crisp crib close-ups
    bedside_windows: list[tuple[float, float, dict[str, Any], Path, str]] = []
    for rb in rows:
        cb_cfg = cams.get(rb["camera"], {})
        if cb_cfg.get("role") != "bedside_crib":
            continue
        eff_b = effective_wall_ts_fn(rb, cb_cfg)
        ep_b0 = parse_wall_epoch_fn(eff_b, 0.0)
        src_b = resolve_source_path_fn(rb)
        dur_b = float(rb.get("duration") or 0.0)
        if ep_b0 is not None and src_b and dur_b > 10.0 and eff_b:
            bedside_windows.append((ep_b0, ep_b0 + dur_b, rb, src_b, eff_b))

    # 1. Confirmed speech milestones ("爸爸/爸", "妈妈/妈") — emotional climax priority
    for sh, r_match in speech_items[:3]:
        src = Path(sh["source_path"])
        c_cfg = cams.get(r_match["camera"], {})
        dur = max(4.0, float(r_match.get("duration") or 10.0))
        ut_t = float(sh.get("utterance_t_s") or sh.get("start_s") or 0.0)
        s0 = max(0.0, float(sh.get("start_s") or max(0.0, ut_t - 1.5)) - 1.2)
        e0 = min(dur, max(s0 + min(6.5, dur - s0), float(sh.get("end_s") or (ut_t + 1.5)) + 2.5))
        eff_wts, ep, hr = _hour_and_epoch(r_match, c_cfg, s0, day, effective_wall_ts_fn, parse_wall_epoch_fn)
        _add_cand(
            source_kind="speech",
            priority=200.0 + float(sh.get("phonetic_prob") or 0.8),
            r=r_match,
            src=src,
            center_t_s=ut_t,
            w_start=s0,
            w_end=e0,
            box=None,
            eff_wts=eff_wts,
            wall_ep=ep,
            clip_hr=hr,
            pre_verified=True,
            must_include=True,
            boost_vocal=True,
            speech_meta=sh,
            fallback_note=f"清晰发出“{sh.get('matched_word', '')}”的声音",
        )
        if ep is not None and c_cfg.get("role") != "bedside_crib":
            sp_abs_ep = ep + max(0.0, ut_t - s0)
            for b0, b1, rb, src_b, eff_b in bedside_windows:
                if b0 + 3.0 <= sp_abs_ep <= b1 - 4.0:
                    t_b = round(sp_abs_ep - b0, 2)
                    wb0, wb1 = _build_window(t_b, float(rb.get("duration") or 600.0), target_span=11.0)
                    _add_cand(
                        source_kind="speech_bedside_sync",
                        priority=190.0,
                        r=rb,
                        src=src_b,
                        center_t_s=t_b,
                        w_start=wb0,
                        w_end=wb1,
                        box=None,
                        eff_wts=eff_b,
                        wall_ep=sp_abs_ep,
                        clip_hr=hr,
                        pre_verified=True,
                        boost_vocal=True,
                        fallback_note=f"清晰喊出“{sh.get('matched_word', '')}”时的床头近景神态",
                    )
                    break

    # 2. Pre-verified daytime portrait photos (already passed VLM face & composition gate)
    for vp in vlm_photos[:8]:
        r_p = vp["row"]
        src_p = resolve_source_path_fn(r_p)
        if not src_p:
            continue
        c_cfg = cams.get(r_p["camera"], {})
        dur = float(r_p.get("duration") or 10.0)
        t_c = float(vp.get("t") or 0.0)
        w0, w1 = _build_window(t_c, dur, target_span=10.5)
        eff_wts, ep, hr = _hour_and_epoch(r_p, c_cfg, t_c, day, effective_wall_ts_fn, parse_wall_epoch_fn)
        pscore = float((vp.get("vlm") or {}).get("photo_score") or 4.0)
        pbox = (vp.get("vlm") or {}).get("child_box") or vp.get("box")
        _add_cand(
            source_kind="verified_photo",
            priority=120.0 + pscore * 3.0,
            r=r_p,
            src=src_p,
            center_t_s=t_c,
            w_start=w0,
            w_end=w1,
            box=pbox,
            eff_wts=eff_wts,
            wall_ep=ep,
            clip_hr=hr,
            pre_verified=True,
            photo_vlm=vp.get("vlm"),
            fallback_note=str((vp.get("vlm") or {}).get("caption_zh") or ""),
        )

    # 3. Confirmed kitchen baby clips (child_clear == "yes" only)
    kit_list: list[tuple[float, dict[str, Any], Path, float, dict[str, Any]]] = []
    for kr_rev in kit_reviews.values():
        if kr_rev.get("status") != "confirmed":
            continue
        resp = kr_rev.get("response") or {}
        if resp.get("child_clear") != "yes":
            continue
        rm = row_by_rel.get(str(kr_rev.get("source_rel") or ""))
        if not rm:
            continue
        src_k = resolve_source_path_fn(rm)
        if not src_k:
            continue
        t_k = float(kr_rev.get("t_s") or 0.0)
        exc_k = float(resp.get("excitement_score") or 1.0)
        adj_k, sev_k = _box_quality_adjustment(resp.get("child_box"))
        if sev_k:
            continue
        kit_list.append((65.0 + exc_k * 2.0 + adj_k, rm, src_k, t_k, resp))
    kit_list.sort(key=lambda x: -x[0])
    for prio, rm, src_k, t_k, resp in kit_list[:14]:
        c_cfg = cams.get(rm["camera"], {})
        w0, w1 = _build_window(t_k, float(rm.get("duration") or 10.0), target_span=11.0)
        eff_wts, ep, hr = _hour_and_epoch(rm, c_cfg, t_k, day, effective_wall_ts_fn, parse_wall_epoch_fn)
        _add_cand(
            source_kind="kitchen_confirmed",
            priority=prio,
            r=rm,
            src=src_k,
            center_t_s=t_k,
            w_start=w0,
            w_end=w1,
            box=resp.get("child_box"),
            eff_wts=eff_wts,
            wall_ep=ep,
            clip_hr=hr,
            fallback_note=str(resp.get("caption_zh") or "在厨房好奇陪伴与探索"),
        )

    # 4. Acoustic laughter / vocalization peaks & multi-candidate visual activity
    act_rows: list[tuple[int, float, dict[str, Any], float, list[float] | None, str, bool]] = []
    for r in rows:
        cam = r["camera"]
        c_cfg = cams.get(cam, {})
        if not c_cfg.get("milestone_watch", True):
            continue
        cam_role = str(c_cfg.get("role") or "")
        dur = float(r.get("duration") or 0.0)
        if dur < 3.5:
            continue
        pcs = json.loads(r.get("candidates_json") or "[]")
        aud_s = json.loads(r.get("audio_summary_json") or "{}")
        max_laugh = float(aud_s.get("max_laughter") or 0.0)
        max_voc = float(aud_s.get("max_child_voc") or 0.0)

        if max_laugh >= 0.20 or max_voc >= 0.35:
            max_b_cnt = 2 if dur >= 180.0 else 1
            b_peaks = _locate_burst_peaks_in_row(r, max_peaks=max_b_cnt)
            if not b_peaks:
                b_peaks = [_locate_peak_in_row(r, prefer_laughter=True, cam_role=cam_role)]
            for b_idx, (pk_t, pk_box, pk_vis_sc) in enumerate(b_peaks):
                eff_wts, ep, hr = _hour_and_epoch(r, c_cfg, pk_t, day, effective_wall_ts_fn, parse_wall_epoch_fn)
                is_w = 1 if (hr is None or 7 <= hr < 21) else 0
                kind = "laughter_peak" if max_laugh >= 0.20 else "voc_peak"
                score_a = (
                    45.0
                    + max_laugh * 18.0
                    + max_voc * 8.0
                    + pk_vis_sc * 4.0
                    + min(3.0, float(r.get("active_s") or 0.0) / 8.0)
                    + (4.0 if cam_role == "bedside_crib" else 0.0)
                    - b_idx * 4.0
                )
                act_rows.append((is_w, score_a, r, pk_t, pk_box, kind, max_laugh >= 0.35))

        if not pcs:
            continue
        # Evaluate top distinct timestamps within long clips (1st always, plus secondary non-overlapping peak on long bundles)
        picked_t_in_row: list[float] = []
        for pc in pcs[:6]:
            if pc.get("luma", 0) < 65 or pc.get("sharp", 0) < 35:
                continue
            t_c = float(pc.get("t") or 0.0)
            if any(abs(t_c - prev_t) < 25.0 for prev_t in picked_t_in_row):
                continue
            adj, _sev = _box_quality_adjustment(pc.get("box"), cam_role=cam_role)
            eff_wts, ep, hr = _hour_and_epoch(r, c_cfg, t_c, day, effective_wall_ts_fn, parse_wall_epoch_fn)
            is_w = 1 if (hr is None or 7 <= hr < 21) else 0
            base_rk = float(pc.get("score") or 0.0) + min(0.8, float(r.get("active_s") or 0.0) / 15.0) + (
                0.6 if max_laugh >= 0.25 else 0.0
            )
            act_rows.append((is_w, 20.0 + base_rk + adj * 2.0, r, t_c, pc.get("box"), "activity", False))
            picked_t_in_row.append(t_c)
            if len(picked_t_in_row) >= (2 if dur >= 120.0 else 1):
                break

    has_waking_act = any(x[0] == 1 for x in act_rows)
    if has_waking_act:
        act_rows = [x for x in act_rows if x[0] == 1]

    for _is_w, rk, r, t_c, box, kind, boost_v in act_rows:
        src = resolve_source_path_fn(r)
        if not src:
            continue
        c_cfg = cams.get(r["camera"], {})
        w0, w1 = _build_window(t_c, float(r.get("duration") or 10.0), target_span=12.0)
        eff_wts, ep, hr = _hour_and_epoch(r, c_cfg, t_c, day, effective_wall_ts_fn, parse_wall_epoch_fn)
        _add_cand(
            source_kind=kind,
            priority=rk,
            r=r,
            src=src,
            center_t_s=t_c,
            w_start=w0,
            w_end=w1,
            box=box,
            eff_wts=eff_wts,
            wall_ep=ep,
            clip_hr=hr,
            boost_vocal=boost_v,
        )

    # Small deterministic mode when running minimal unit tests without real providers
    if is_small_stub:
        raw_pool.sort(key=lambda c: (-c["is_waking"], -c["priority"]))
        chosen: list[dict[str, Any]] = []
        picked_rels: set[str] = set()
        picked_eps: list[float] = []
        seen_3h: set[int] = set()
        for c in raw_pool:
            b3 = c["bucket_3h"]
            if b3 is None or b3 in seen_3h:
                continue
            ep = c["wall_ep"]
            if ep is not None and any(abs(ep - e0) < 900.0 for e0 in picked_eps):
                continue
            chosen.append(c)
            picked_rels.add(str(c["row"]["source_rel"]))
            if ep is not None:
                picked_eps.append(ep)
            seen_3h.add(b3)
            if len(chosen) >= 8:
                break
        if len(chosen) < 8:
            for c in raw_pool:
                if str(c["row"]["source_rel"]) in picked_rels:
                    continue
                ep = c["wall_ep"]
                if ep is not None and any(abs(ep - e0) < 900.0 for e0 in picked_eps):
                    continue
                chosen.append(c)
                picked_rels.add(str(c["row"]["source_rel"]))
                if ep is not None:
                    picked_eps.append(ep)
                if len(chosen) >= 8:
                    break
        return chosen

    # Full-day wide-net selection (max 64 candidates across all 2h waking buckets, signals & cameras)
    target_pool = 64
    raw_pool.sort(key=lambda c: (-c["is_waking"], -c["priority"]))
    chosen_wide: list[dict[str, Any]] = []
    cam_counts: dict[str, int] = {}
    bucket_counts: dict[int, int] = {}
    selected_eps: list[tuple[str, float]] = []
    selected_keys: set[tuple[str, int]] = set()

    def _can_pick(c: dict[str, Any], same_cam_gap: float, any_cam_gap: float) -> bool:
        key = (str(c["row"]["source_rel"]), int(round(c["center_t_s"] / 8.0)))
        if key in selected_keys:
            return False
        if c["must_include"]:
            return True
        ep = c["wall_ep"]
        if ep is not None:
            for prev_cam, prev_ep in selected_eps:
                gap = same_cam_gap if prev_cam == c["cam_id"] else any_cam_gap
                if abs(ep - prev_ep) < gap:
                    return False
        return True

    def _commit(c: dict[str, Any]) -> None:
        chosen_wide.append(c)
        selected_keys.add((str(c["row"]["source_rel"]), int(round(c["center_t_s"] / 8.0))))
        cam_counts[c["cam_id"]] = cam_counts.get(c["cam_id"], 0) + 1
        if c["bucket_2h"] is not None:
            bucket_counts[c["bucket_2h"]] = bucket_counts.get(c["bucket_2h"], 0) + 1
        if c["wall_ep"] is not None:
            selected_eps.append((c["cam_id"], c["wall_ep"]))

    # Pass 1: Must-include speech hits
    for c in raw_pool:
        if c["must_include"] and _can_pick(c, 30.0, 20.0):
            _commit(c)

    # Pass 2: Pre-verified clear daytime photos + top laughter per 2h bucket + camera coverage
    for b2 in sorted({c["bucket_2h"] for c in raw_pool if c["bucket_2h"] is not None}):
        b_cands = [c for c in raw_pool if c["bucket_2h"] == b2]
        for c in b_cands:
            if len(chosen_wide) >= target_pool:
                break
            if bucket_counts.get(b2, 0) >= 3:
                break
            if _can_pick(c, 180.0, 90.0):
                _commit(c)

    # Pass 3: Balance underrepresented cameras (客厅 / 书房 / 厨房 / 宝宝床头)
    for cid in {c["cam_id"] for c in raw_pool}:
        if cam_counts.get(cid, 0) >= 4:
            continue
        for c in [x for x in raw_pool if x["cam_id"] == cid]:
            if len(chosen_wide) >= target_pool or cam_counts.get(cid, 0) >= 4:
                break
            if _can_pick(c, 150.0, 60.0):
                _commit(c)

    # Pass 4: Fill remaining quota up to 42 sorted by priority (preferring less-crowded buckets/cameras)
    scored_rest = sorted(
        raw_pool,
        key=lambda c: (
            -(c["priority"] - cam_counts.get(c["cam_id"], 0) * 1.2 - bucket_counts.get(c["bucket_2h"] or -1, 0) * 2.5),
        ),
    )
    for c in scored_rest:
        if len(chosen_wide) >= target_pool:
            break
        if _can_pick(c, 120.0, 45.0):
            _commit(c)

    return chosen_wide


def _is_legacy_stub_response(ans: dict[str, Any]) -> bool:
    return (
        "usable_for_highlight" not in ans
        and "face_visibility" not in ans
        and "best_in_rel_s" not in ans
        and ("excitement_score" in ans or "categories" in ans)
    )


def _compute_shot_zoom(shot_scale: str, c_cfg: dict[str, Any], box: list[float] | None) -> float:
    has_box = bool(box and len(box) == 4 and box[2] > box[0] and box[3] > box[1])
    if not has_box or shot_scale == "wide":
        return 1.0
    policy = str(c_cfg.get("zoom_policy") or "gentle_if_wide")
    if policy == "never":
        return 1.0
    cfg_zoom = float(c_cfg.get("zoom_factor", 1.35))
    if shot_scale == "medium":
        return round(min(cfg_zoom, 1.34) if cfg_zoom > 1.05 else 1.22, 2)
    return round(min(1.45, max(1.32, cfg_zoom)), 2)


def _estimate_net_duration_s(shots: list[dict[str, Any]], default_fade_s: float = 0.55) -> float:
    if not shots:
        return 0.0
    tot = sum(float(s["dur_s"]) for s in shots)
    for idx in range(1, len(shots)):
        tr = str(shots[idx].get("transition") or "warm_dissolve")
        ov = 0.20 if tr == "soft_cut" else (0.80 if tr == "chapter_dissolve" else default_fade_s)
        tot -= ov
    return round(max(0.0, tot), 2)


def _build_default_story_meta(
    day: str,
    age_info: dict[str, Any],
    verified: list[dict[str, Any]],
) -> dict[str, str]:
    day_n = age_info.get("day_n")
    age_lbl = age_info.get("label_zh") or ""
    speech_words = [
        str(v["speech_meta"].get("matched_word"))
        for v in verified
        if v.get("speech_meta") and v["speech_meta"].get("matched_word")
    ]
    title = (
        f"第{day_n}天 · {f'第一声“{speech_words[0]}”与' if speech_words else ''}温暖成长记"
        if day_n
        else f"{day} · 今日成长手记"
    )
    cams_lbl = "、".join(dict.fromkeys(v["cam_short"] for v in verified[:4]))
    logline = (
        f"今天是宝宝成长的第 {day_n} 天（{age_lbl}），镜头记录下{cams_lbl}里从清晨到夜色的 {len(verified)} 个清晰温暖瞬间。"
        if day_n
        else f"从清晨到夜色，今天在{cams_lbl}留下了 {len(verified)} 个清晰温暖的成长瞬间。"
    )
    joy_notes = [v["director_note_zh"] for v in verified if v.get("emotion_tag") == "joy" or v.get("boost_vocal")]
    joy_sum = _clean_warm_text(joy_notes[0], 38) if joy_notes else "与家人互动游玩时的舒展神情与明朗笑容"
    skill_notes = [
        v["director_note_zh"]
        for v in verified
        if any(c in ("standing", "walking", "sofa_descent", "kitchen_visit") for c in (v.get("categories") or []))
        or v.get("source_kind") == "speech"
    ]
    growth_sum = (
        _clean_warm_text(f"清晰叫出“{speech_words[0]}”，" + (skill_notes[0] if skill_notes else "活动更自如"), 38)
        if speech_words
        else (_clean_warm_text(skill_notes[0], 38) if skill_notes else "好奇探索周围物品与稳健的日常小进步")
    )
    epilogue = (
        f"第 {day_n} 天，世界慢慢变大，每一步成长都算数。"
        if day_n
        else "把平凡的一天留成珍藏，明天继续慢慢长大。"
    )
    return {
        "film_title_zh": title,
        "logline_zh": logline,
        "joy_summary_zh": joy_sum,
        "growth_changes_zh": growth_sum,
        "closing_epilogue_zh": epilogue,
    }


def _resize_jpegs_max_edge(images: list[bytes], max_edge: int = 768, quality: int = 85) -> list[bytes]:
    out: list[bytes] = []
    for jb in images:
        try:
            with Image.open(io.BytesIO(jb)) as im:
                if max(im.size) <= max_edge:
                    out.append(jb)
                    continue
                rgb = im.convert("RGB")
                rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                rgb.save(buf, "JPEG", quality=quality)
                out.append(buf.getvalue())
        except Exception:
            out.append(jb)
    return out


def _to_rendered_shot(
    v: dict[str, Any],
    *,
    lyric_zh: str,
    chapter_zh: str,
    dur_s: float,
    shot_scale: str,
    pan_style: str,
    transition: str,
) -> dict[str, Any]:
    zoom = _compute_shot_zoom(shot_scale, v["c_cfg"], v["child_box"])
    meta_sub = f"{v['hh_mm']} · {v['cam_name']}"
    if v["source_kind"] == "speech":
        meta_sub += " · 纯童声原声"
    return {
        "shot_id": v["shot_id"],
        "source": str(v["src"]),
        "source_rel": str(v["row"]["source_rel"]),
        "start_s": float(v["abs_in"]),
        "dur_s": round(float(dur_s), 2),
        "nat_w": int(v["row"]["width"]),
        "nat_h": int(v["row"]["height"]),
        "box": v["child_box"] if zoom > 1.02 else None,
        "privacy_box": v["privacy_box"],
        "zoom": zoom,
        "shot_scale": shot_scale,
        "pan_style": pan_style,
        "transition": transition,
        "chapter_zh": chapter_zh,
        "has_audio": bool(v["row"]["has_audio"]),
        "boost_vocal": bool(v["boost_vocal"]),
        "cinema_style": True,
        "title_zh": lyric_zh,
        "sub_zh": meta_sub,
        "wall_time_hms": v["hh_mm_ss"],
        "camera_name": v["cam_name"],
        "source_kind": v["source_kind"],
        "people": list(v.get("verified_people") or []),
    }


def assemble_final_edit_list(
    day: str,
    age_info: dict[str, Any],
    verified: list[dict[str, Any]],
    script_ans: dict[str, Any] | None,
    *,
    target_min_s: float = 84.0,
    target_max_s: float = 126.0,
    max_shots: int = 20,
    active_roles: set[str] | frozenset[str] = frozenset(KINSHIP_ROLES),
    has_enrolled_ref: bool = False,
    strict_unverified: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combine Stage-4 director script with quality/duration/chronological safeguards."""
    if not verified:
        return [], {}
    ver_by_id = {v["shot_id"]: v for v in verified}
    defaults = _build_default_story_meta(day, age_info, verified)
    sa = script_ans if isinstance(script_ans, dict) and not is_unconfirmed_refusal(script_ans) else {}

    film_title_raw = _clean_warm_text(sa.get("film_title_zh") or defaults["film_title_zh"], 32) or defaults["film_title_zh"]
    logline_raw = _clean_warm_text(sa.get("logline_zh") or defaults["logline_zh"], 140) or defaults["logline_zh"]
    joy_summary_raw = _clean_warm_text(sa.get("joy_summary_zh") or defaults["joy_summary_zh"], 90) or defaults["joy_summary_zh"]
    growth_changes_raw = (
        _clean_warm_text(sa.get("growth_changes_zh") or defaults["growth_changes_zh"], 90)
        or defaults["growth_changes_zh"]
    )
    epilogue_raw = (
        _clean_warm_text(sa.get("closing_epilogue_zh") or defaults["closing_epilogue_zh"], 36)
        or defaults["closing_epilogue_zh"]
    )

    # Parse AI edit_list selections
    chosen_specs: dict[str, dict[str, Any]] = {}
    for entry in (sa.get("edit_list") or []):
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("shot_id") or "").strip()
        if sid not in ver_by_id or sid in chosen_specs:
            continue
        chosen_specs[sid] = entry

    # Fallback / test path: start from verified shots filtered cleanly by excitement when surplus exists
    if not chosen_specs:
        filtered = [v for v in verified if v["cinematic_score"] >= 2] or verified
        for v in filtered[:max_shots]:
            chosen_specs[v["shot_id"]] = {
                "shot_id": v["shot_id"],
                "chapter_zh": v["act_hint"],
                "trim_dur_s": v["dur_s"],
                "shot_scale": v["shot_scale"],
                "pan_style": v["pan_style"],
                "lyric_zh": v["director_note_zh"],
            }

    # Ensure every confirmed speech milestone is retained
    for v in verified:
        if v["source_kind"] == "speech" and v["shot_id"] not in chosen_specs:
            chosen_specs[v["shot_id"]] = {
                "shot_id": v["shot_id"],
                "chapter_zh": v["act_hint"],
                "trim_dur_s": max(v["dur_s"], 6.0),
                "shot_scale": v["shot_scale"],
                "pan_style": v["pan_style"],
                "lyric_zh": f"清脆的一声“{v['speech_meta'].get('matched_word', '')}”，被时光轻轻记住"
                if v.get("speech_meta")
                else v["director_note_zh"],
            }

    # Materialize selected shots in chronological order
    selected_ids = sorted(
        chosen_specs.keys(),
        key=lambda sid: (ver_by_id[sid]["sort_key"], sid),
    )[:max_shots]

    def _dur_for(v: dict[str, Any], spec: dict[str, Any], *, is_first: bool = False) -> float:
        verified_dur = float(v["dur_s"])
        req = float(spec.get("trim_dur_s") or verified_dur)
        dur = min(verified_dur, max(min(verified_dur, 4.2), req))
        if is_first and verified_dur >= 6.0:
            dur = max(dur, min(6.6, verified_dur))
        return round(dur, 2)

    def _pair_transition(
        idx: int,
        prev_act_s: str | None,
        prev_v_s: dict[str, Any] | None,
        v_s: dict[str, Any],
        spec_s: dict[str, Any],
    ) -> tuple[str, str, str]:
        raw_ch = str(spec_s.get("chapter_zh") or "").strip()
        eff_act = raw_ch if any(w[2][:4] in raw_ch for w in ACT_WINDOWS) else v_s["act_hint"]
        show_ch = eff_act if (eff_act != prev_act_s) else ""
        tr_s = str(spec_s.get("transition") or "").strip()
        if idx == 0 or show_ch:
            tr_s = "chapter_dissolve"
        elif tr_s not in ("chapter_dissolve", "warm_dissolve", "soft_cut"):
            same_cam_close = bool(
                prev_v_s
                and prev_v_s["cam_id"] == v_s["cam_id"]
                and prev_v_s["wall_ep"] is not None
                and v_s["wall_ep"] is not None
                and abs(v_s["wall_ep"] - prev_v_s["wall_ep"]) <= 900.0
            )
            tr_s = "soft_cut" if same_cam_close else "warm_dissolve"
        return show_ch, eff_act, tr_s

    def _est_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
        ordered = sorted(pairs, key=lambda pair: (pair[0]["sort_key"], pair[0]["shot_id"]))
        sim_shots: list[dict[str, Any]] = []
        p_act: str | None = None
        p_v: dict[str, Any] | None = None
        for i, (vv, sp) in enumerate(ordered):
            show_ch, eff_a, tr_i = _pair_transition(i, p_act, p_v, vv, sp)
            if show_ch:
                p_act = eff_a
            p_v = vv
            sim_shots.append({"dur_s": _dur_for(vv, sp, is_first=(i == 0)), "transition": tr_i})
        return _estimate_net_duration_s(sim_shots)

    selected_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (ver_by_id[sid], chosen_specs[sid]) for sid in selected_ids
    ]

    # Top up to hit >= 1.4 minutes (target_min_s=84s) whenever verified surplus exists across the day
    unused = [v for v in verified if v["shot_id"] not in chosen_specs]
    unused.sort(key=lambda v: (-v["cinematic_score"], -float(v["source_kind"] in ("speech", "laughter_peak")), v["sort_key"]))
    while unused and len(selected_pairs) < max_shots:
        if _est_pairs(selected_pairs) >= target_min_s:
            break
        nxt = unused.pop(0)
        selected_pairs.append((
            nxt,
            {
                "shot_id": nxt["shot_id"],
                "chapter_zh": nxt["act_hint"],
                "trim_dur_s": nxt["dur_s"],
                "shot_scale": nxt["shot_scale"],
                "pan_style": nxt["pan_style"],
                "lyric_zh": nxt["director_note_zh"],
            },
        ))

    # Re-sort chronologically after auto-fill, then trim gently if slightly over target_max_s
    selected_pairs.sort(key=lambda pair: (pair[0]["sort_key"], pair[0]["shot_id"]))
    while len(selected_pairs) > 12:
        if _est_pairs(selected_pairs) <= target_max_s:
            break
        # Drop lowest-priority non-speech, non-first/last shot from most crowded hour
        removable = [
            i for i, (v, _sp) in enumerate(selected_pairs)
            if i not in (0, len(selected_pairs) - 1) and v["source_kind"] != "speech"
        ]
        if not removable:
            break
        drop_i = min(removable, key=lambda idx: selected_pairs[idx][0]["cinematic_score"])
        selected_pairs.pop(drop_i)

    final_shots: list[dict[str, Any]] = []
    storyboard_rows: list[dict[str, Any]] = []
    prev_act: str | None = None
    prev_v: dict[str, Any] | None = None

    for idx, (v, spec) in enumerate(selected_pairs):
        show_chapter, eff_act, tr = _pair_transition(idx, prev_act, prev_v, v, spec)
        if show_chapter:
            prev_act = eff_act

        scale = str(spec.get("shot_scale") or v["shot_scale"])
        if scale not in ("closeup", "medium", "wide"):
            scale = v["shot_scale"]
        pan = str(spec.get("pan_style") or v["pan_style"])
        if pan not in ("gentle_push", "gentle_pull", "steady"):
            pan = "gentle_push"
        dur_s = _dur_for(v, spec, is_first=(idx == 0))
        raw_lyric = _clean_warm_text(spec.get("lyric_zh") or v["director_note_zh"], 24) or _clean_warm_text(v["director_note_zh"], 24)
        v_local = set(v.get("verified_local_roles") or set())
        v_vlm = set(v.get("verified_vlm_roles") or set())
        v_has_ref = bool(v.get("has_enrolled_ref", has_enrolled_ref))
        lyric = sanitize_caption_kinship(
            raw_lyric,
            verified_local_roles=v_local,
            verified_vlm_roles=v_vlm,
            active_roles=active_roles,
            has_enrolled_ref=v_has_ref,
            strict_unverified=strict_unverified,
        )

        rendered = _to_rendered_shot(
            v,
            lyric_zh=lyric,
            chapter_zh=show_chapter,
            dur_s=dur_s,
            shot_scale=scale,
            pan_style=pan,
            transition=tr,
        )
        final_shots.append(rendered)
        storyboard_rows.append({
            "index": idx + 1,
            "shot_id": v["shot_id"],
            "chapter_zh": eff_act,
            "time": v["hh_mm_ss"],
            "camera": v["cam_name"],
            "camera_id": v["cam_id"],
            "shot_scale": scale,
            "pan_style": pan,
            "dur_s": dur_s,
            "transition": tr,
            "lyric_zh": lyric,
            "people": list(v.get("verified_people") or []),
            "source_rel": v["row"]["source_rel"],
            "start_s": v["abs_in"],
            "box": v["child_box"],
            "privacy_box": v["privacy_box"],
            "emotion_tag": v["emotion_tag"],
            "cinematic_score": round(float(v["cinematic_score"]), 2),
            "source_kind": v["source_kind"],
            "boost_vocal": bool(v["boost_vocal"]),
            "categories": list(v["categories"]),
            "verified_local_roles": sorted(v.get("verified_local_roles") or ()),
            "verified_vlm_roles": sorted(v.get("verified_vlm_roles") or ()),
        })
        prev_v = v

    day_local_roles: set[str] = set()
    day_vlm_roles: set[str] = set()
    for v, _sp in selected_pairs:
        day_local_roles.update(v.get("verified_local_roles") or set())
        day_vlm_roles.update(v.get("verified_vlm_roles") or set())
    day_lf = day_local_roles & FEMALE_ROLES
    day_lm = day_local_roles & MALE_ROLES
    if day_lf and "妈妈" not in day_lf:
        day_vlm_roles.discard("妈妈")
    if day_lm and "爸爸" not in day_lm:
        day_vlm_roles.discard("爸爸")
    day_has_ref = has_enrolled_ref or any(bool(v.get("has_enrolled_ref")) for v, _sp in selected_pairs)

    def _guard_global(txt: str) -> str:
        return sanitize_caption_kinship(
            txt,
            verified_local_roles=day_local_roles,
            verified_vlm_roles=day_vlm_roles,
            active_roles=active_roles,
            has_enrolled_ref=day_has_ref,
            strict_unverified=strict_unverified,
        )

    film_title = _guard_global(film_title_raw)
    logline = _guard_global(logline_raw)
    joy_summary = _guard_global(joy_summary_raw)
    growth_changes = _guard_global(growth_changes_raw)
    epilogue = _guard_global(epilogue_raw)

    story_doc = {
        "day": day,
        "film_title_zh": film_title,
        "logline_zh": logline,
        "joy_summary_zh": joy_summary,
        "growth_changes_zh": growth_changes,
        "closing_epilogue_zh": epilogue,
        "director_model": str(sa.get("_model") or "director"),
        "est_duration_s": _estimate_net_duration_s(final_shots),
        "shots": storyboard_rows,
    }
    return final_shots, story_doc


def curate_daily_highlight_and_milestones(
    *,
    day: str,
    tier: str,
    background: bool,
    rows: list[dict[str, Any]],
    cams: dict[str, Any],
    age_info: dict[str, Any],
    vlm: Any,
    vlm_cfg: dict[str, Any],
    extract_frames_fn: Callable[[dict[str, Any], Path, list[float], dict[str, Any]], list[bytes]],
    handle_refusal_fn: Callable[..., str | dict[str, Any] | None],
    effective_wall_ts_fn: Callable[[dict[str, Any], dict[str, Any]], str | None],
    parse_wall_epoch_fn: Callable[[str | None, float], float | None],
    resolve_source_path_fn: Callable[[dict[str, Any]], Path | None],
    resolve_privacy_box_fn: Callable[[dict[str, Any], list[float] | None], list[float] | None],
    progress_fn: Callable[[str, str, str, str, bool], None],
    speech_items: list[tuple[dict[str, Any], dict[str, Any]]],
    vlm_photos: list[dict[str, Any]],
    kit_reviews: dict[str, dict[str, Any]],
    face_service: Any = None,
    extract_hires_frames_fn: Callable[[dict[str, Any], Path, list[float], dict[str, Any]], list[bytes]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Execute 4-stage HIGH-thinking highlight curation; return (highlight_shots, story_doc, milestone_candidates)."""
    has_providers = bool(vlm_cfg.get("providers"))
    candidates = harvest_daily_candidates(
        day,
        rows,
        cams,
        speech_items=speech_items,
        vlm_photos=vlm_photos,
        kit_reviews=kit_reviews,
        effective_wall_ts_fn=effective_wall_ts_fn,
        parse_wall_epoch_fn=parse_wall_epoch_fn,
        resolve_source_path_fn=resolve_source_path_fn,
        has_providers=has_providers,
    )
    if not candidates:
        return [], {}, []

    is_small_stub = (not has_providers) and (len(rows) <= 20)

    # Stage 2: Fast coarse visual filter (only on unverified candidates in multi-provider production)
    shortlisted: list[dict[str, Any]] = []
    if is_small_stub:
        shortlisted = candidates[:24]
    else:
        coarse_cands = [c for c in candidates if not c["pre_verified"] and c["cloud_ok"]][:48]
        pre_ok = [c for c in candidates if c["pre_verified"] or c["must_include"]]
        coarse_kept: list[dict[str, Any]] = []
        if coarse_cands:
            progress_fn(tier, day, "粗筛全天高光", f"多模型并发粗筛 {len(coarse_cands)} 段全天觉醒活跃候选", background)

            def _prep_coarse(item: tuple[int, dict[str, Any]]):
                idx_c, cand = item
                dur = float(cand["row"].get("duration") or 10.0)
                if dur >= 12.0 and cand["center_t_s"] < 2.5 and cand["source_kind"] == "activity":
                    t0 = round(max(1.8, cand["center_t_s"] + 1.2), 2)
                    t1 = round(min(dur - 0.8, max(t0 + 2.0, dur * 0.42)), 2)
                else:
                    t0 = max(0.3, min(dur - 0.8, cand["center_t_s"] - 0.5))
                    t1 = min(max(0.5, dur - 0.3), max(t0 + 1.0, cand["center_t_s"] + 1.4))
                imgs = extract_frames_fn(cand["row"], cand["src"], [t0, t1], cams)
                if len(imgs) < 1:
                    return None
                return idx_c, cand, imgs, t0, t1

            with ThreadPoolExecutor(max_workers=min(4, len(coarse_cands))) as cpool:
                prepped_c = [x for x in cpool.map(_prep_coarse, list(enumerate(coarse_cands))) if x is not None]

            c_tasks = [
                (idx_c, f"hlc|{cand['row']['source_rel']}|{cand['center_t_s']:.1f}", imgs, PROMPT_HIGHLIGHT_COARSE)
                for idx_c, cand, imgs, _t0, _t1 in prepped_c
            ]
            c_ans_map = vlm.ask_batch(c_tasks, tier="fast") if c_tasks else {}
            for idx_c, cand, _imgs, t0, t1 in prepped_c:
                ans = c_ans_map.get(idx_c)
                if not ans:
                    continue
                if is_unconfirmed_refusal(ans):
                    rev = handle_refusal_fn(
                        "highlight_coarse", f"hlc|{cand['row']['source_rel']}|{t0:.1f}", day, cand["row"],
                        float(t0), float(t1), ans, box_fallback=cand.get("box"),
                    )
                    if not isinstance(rev, dict):
                        continue
                    ans = rev
                c_clr = str(ans.get("child_clear") or "")
                f_clr = str(ans.get("face_clear") or "")
                worth = float(ans.get("moment_worth") or 0.0)
                if ans.get("edge_clipped") == "yes":
                    continue
                if f_clr == "no":
                    continue
                if not (c_clr == "yes" or (c_clr == "partial" and worth >= 3.0 and f_clr in ("yes", "side"))):
                    continue
                if ans.get("awake_active") == "sleeping" and worth < 3.0:
                    continue
                if worth < 2.0:
                    continue
                cand["priority"] += worth * 8.0 + (4.0 if f_clr == "yes" else 0.0)
                if ans.get("child_box") and ans["child_box"] != [0.0, 0.0, 0.0, 0.0]:
                    cand["box"] = ans["child_box"]
                if ans.get("brief_zh"):
                    cand["fallback_note"] = sanitize_caption_kinship(
                        str(ans["brief_zh"]),
                        verified_local_roles=set(),
                        has_enrolled_ref=False,
                        strict_unverified=True,
                    )
                coarse_kept.append(cand)

        combined_s2 = pre_ok + sorted(coarse_kept, key=lambda c: -c["priority"])
        # Preserve must_include + verified photos + broad 2h coverage up to 30 shortlisted shots
        chosen_s2: list[dict[str, Any]] = []
        seen_s2_key: set[tuple[str, int]] = set()
        seen_b2: dict[int, int] = {}
        for c in sorted(combined_s2, key=lambda x: (-int(x["must_include"]), -x["priority"])):
            k = (str(c["row"]["source_rel"]), int(round(c["center_t_s"] / 8.0)))
            if k in seen_s2_key:
                continue
            b2 = c["bucket_2h"]
            if len(chosen_s2) < 22 or c["must_include"] or (b2 is not None and seen_b2.get(b2, 0) < 5):
                chosen_s2.append(c)
                seen_s2_key.add(k)
                if b2 is not None:
                    seen_b2[b2] = seen_b2.get(b2, 0) + 1
            if len(chosen_s2) >= 30:
                break
        shortlisted = chosen_s2

    if not shortlisted:
        return [], {}, []

    # Stage 3: Highest-thinking (HIGH) Shot Editor inspecting 6-frame timecode-stamped filmstrip
    s3_tier = "pro" if is_small_stub else "director"
    progress_fn(tier, day, "高思考逐镜精剪", f"Gemini 4 Carbon HIGH + GPT-6.2 High 精审 {len(shortlisted)} 段多帧定剪", background)

    face_active = bool(face_service is not None and getattr(face_service, "enabled", True))
    ref_sheet_bytes: bytes | None = None
    if face_active and bool((face_service.fcfg or {}).get("attach_reference_sheet", True)):
        try:
            ref_sheet_bytes = face_service.get_reference_sheet(day)
        except Exception:
            ref_sheet_bytes = None
    ref_imgs = [("家庭成员识别对照卡", ref_sheet_bytes)] if ref_sheet_bytes else None
    use_hires_for_face = bool(
        face_active
        and extract_hires_frames_fn is not None
        and face_service.has_enrolled_members(day)
    )

    def _prep_s3(item: tuple[int, dict[str, Any]]):
        idx_s, cand = item
        r_s = cand["row"]
        w0, w1 = float(cand["w_start"]), float(cand["w_end"])
        span = max(2.0, w1 - w0)
        if not cand["cloud_ok"]:
            return idx_s, cand, w0, w1, [], [], b"", None
        if is_small_stub:
            offsets = [
                round(w0 + 0.5, 2),
                round(w0 + min(span * 0.5, 3.0), 2),
                round(w0 + min(max(1.0, span - 0.6), 5.8), 2),
            ]
        else:
            n_f = 6 if span >= 6.0 else 4
            lo = w0 + 0.35
            hi = max(lo + 0.5, w1 - 0.35)
            offsets = [round(lo + (hi - lo) * i / float(max(1, n_f - 1)), 2) for i in range(n_f)]
        face_match: dict[str, Any] | None = None
        if use_hires_for_face and extract_hires_frames_fn is not None:
            hires_imgs = extract_hires_frames_fn(r_s, cand["src"], offsets, cams)
            if hires_imgs:
                try:
                    face_match = face_service.identify_images(hires_imgs, day=day, child_box=cand.get("box"))
                except Exception:
                    face_match = None
                raw_imgs = _resize_jpegs_max_edge(hires_imgs, 768)
            else:
                raw_imgs = []
        else:
            raw_imgs = extract_frames_fn(r_s, cand["src"], offsets, cams)
            if face_active and raw_imgs:
                try:
                    face_match = face_service.identify_images(raw_imgs, day=day, child_box=cand.get("box"))
                except Exception:
                    face_match = None
        if len(raw_imgs) < 2 and not cand["must_include"]:
            return None
        if not raw_imgs:
            return idx_s, cand, w0, w1, [], [], b"", face_match
        rel_offsets = [max(0.0, o - w0) for o in offsets[: len(raw_imgs)]]
        stamped = raw_imgs if is_small_stub else stamp_timecode_badges(raw_imgs, rel_offsets)
        poster_jpg = raw_imgs[len(raw_imgs) // 2]
        return idx_s, cand, w0, w1, stamped, rel_offsets, poster_jpg, face_match

    with ThreadPoolExecutor(max_workers=min(4, len(shortlisted))) as s3pool:
        prepped_s3 = [x for x in s3pool.map(_prep_s3, list(enumerate(shortlisted))) if x is not None]

    day_confirmed_local: set[str] = {
        role
        for _idx_p, _c_p, _w0_p, _w1_p, _st_p, _ro_p, _po_p, fm_p in prepped_s3
        if isinstance(fm_p, dict)
        for role in (fm_p.get("verified_roles") or ())
    }

    s3_prompt_tpl = PROMPT_DIRECTOR_SHOT_EDIT
    s3_tasks: list[tuple[Any, ...]] = []
    for idx_s, cand, w0, w1, stamped, _rel_o, _poster, face_match in prepped_s3:
        if not stamped:
            continue
        fam_block = (
            "\n\n" + face_service.build_vlm_family_prompt_block(
                day, face_match, has_ref_sheet=bool(ref_imgs), day_confirmed_roles=day_confirmed_local,
            )
            if face_active
            else ""
        )
        full_prompt = (
            f"{s3_prompt_tpl}\n\n[片段参考] 机位：{cand['cam_name']}；窗口跨度：+0.0s 至 +{max(1.0, w1 - w0):.1f}s。"
            f"{fam_block}"
        )
        if ref_imgs:
            s3_tasks.append((idx_s, f"dl_edit|{cand['row']['source_rel']}|{w0:.1f}", stamped, full_prompt, ref_imgs))
        else:
            s3_tasks.append((idx_s, f"dl_edit|{cand['row']['source_rel']}|{w0:.1f}", stamped, full_prompt))
    s3_answers = vlm.ask_batch(s3_tasks, tier=s3_tier) if s3_tasks else {}

    verified_shots: list[dict[str, Any]] = []
    milestone_candidates: list[dict[str, Any]] = []
    active_roles_day = face_service.active_roles_for_day(day) if face_active else frozenset(KINSHIP_ROLES)
    strict_unverified_day = bool((face_service.fcfg or {}).get("strict_unverified_to_family", True)) if face_active else True

    for idx_s, cand, w0, w1, stamped, _rel_o, poster_jpg, face_match in prepped_s3:
        r = cand["row"]
        cam = cand["cam_id"]
        c_cfg = cand["c_cfg"]
        clip_dur = max(3.6, float(r.get("duration") or (w1 - w0)))
        win_span = max(3.6, w1 - w0)
        ans = s3_answers.get(idx_s) if stamped else None

        if stamped and ans and is_unconfirmed_refusal(ans):
            rev = handle_refusal_fn(
                "milestone", f"ms|{r['source_rel']}|{w0:.1f}", day, r,
                float(w0), float(w1), ans, box_fallback=cand.get("box"),
            )
            if not isinstance(rev, dict):
                continue
            ans = rev

        if ans is None:
            if not cand["must_include"]:
                continue
            # Audio-first preservation for speech hits when camera didn't have cloud-allowed frames
            ans = {
                "usable_for_highlight": "yes",
                "child_clear": "audio_only",
                "face_visibility": "clear_action_profile",
                "best_in_rel_s": 0.0,
                "best_out_rel_s": min(win_span, 6.5),
                "shot_scale": "wide",
                "pan_style": "steady",
                "child_box": [0.0, 0.0, 0.0, 0.0],
                "categories": [],
                "independence": "independent",
                "emotion_tag": "speech",
                "cinematic_score": 5,
                "director_note_zh": cand["fallback_note"] or "珍贵纯童声瞬间",
            }

        is_stub_ans = _is_legacy_stub_response(ans)
        if is_stub_ans:
            if ans.get("child_clear") != "yes":
                continue
            c_score = float(ans.get("excitement_score") or 0.0)
            in_rel = 0.0
            out_rel = min(win_span, 6.5)
            scale = "medium"
            pan = "gentle_push"
            emo = "joy" if c_score >= 4 else "focused"
            note_zh = str(ans.get("caption_zh") or cand["fallback_note"] or "开心瞬间")
        else:
            if ans.get("usable_for_highlight") != "yes" and not cand["must_include"]:
                continue
            fvis = str(ans.get("face_visibility") or "").strip()
            if fvis in BLOCKED_FACE_VIS and not cand["must_include"]:
                continue
            c_clr_s3 = str(ans.get("child_clear") or "")
            if not cand["must_include"]:
                if c_clr_s3 == "yes":
                    pass
                elif c_clr_s3 == "partial" and fvis in ("clear_front", "clear_three_quarter", "clear_action_profile"):
                    pass
                else:
                    continue
            c_score = float(ans.get("cinematic_score") or ans.get("excitement_score") or 0.0)
            if c_score > 10.0:
                c_score = round(c_score / 20.0, 2)
            if c_score < 2.0 and not cand["must_include"]:
                continue
            raw_in = float(ans.get("best_in_rel_s") if ans.get("best_in_rel_s") is not None else 0.0)
            raw_out = float(ans.get("best_out_rel_s") if ans.get("best_out_rel_s") is not None else min(win_span, 6.5))
            if raw_in > win_span + 1.5 and (w0 - 1.5) <= raw_in <= (w1 + 1.5):
                raw_in -= w0
                raw_out -= w0
            in_rel = max(0.0, min(max(0.0, win_span - 4.0), raw_in))
            out_rel = min(win_span, max(in_rel + min(4.2, win_span), min(in_rel + 8.8, raw_out)))
            scale = str(ans.get("shot_scale") or "medium").strip().lower()
            if scale not in ("closeup", "medium", "wide"):
                scale = "medium"
            pan = str(ans.get("pan_style") or "gentle_push").strip().lower()
            if pan not in ("gentle_push", "gentle_pull", "steady"):
                pan = "gentle_push"
            emo = str(ans.get("emotion_tag") or ("speech" if cand["source_kind"] == "speech" else "joy"))
            note_zh = str(ans.get("director_note_zh") or ans.get("caption_zh") or cand["fallback_note"] or "温暖时光")

        abs_in = round(max(0.0, w0 + in_rel), 2)
        abs_out = round(min(clip_dur, max(abs_in + min(3.6, clip_dur - abs_in), w0 + out_rel)), 2)
        dur_s = round(max(3.6, abs_out - abs_in), 2)

        if stamped and not is_cloud_allowed(
            c_cfg, cand["eff_wts"], [abs_in, round((abs_in + abs_out) * 0.5, 2), max(abs_in, abs_out - 0.2)]
        ):
            if not cand["must_include"]:
                continue
            scale = "wide"

        cbox = ans.get("child_box")
        if not (isinstance(cbox, list) and len(cbox) == 4 and cbox != [0.0, 0.0, 0.0, 0.0]):
            cbox = cand.get("box")
        if cand["source_kind"] == "speech" and scale == "wide" and isinstance(cbox, list) and len(cbox) == 4 and cbox != [0.0, 0.0, 0.0, 0.0]:
            scale = "medium"
        pbox = resolve_privacy_box_fn(ans, cbox)

        wts = cand["eff_wts"]
        try:
            wall_dt = dt.datetime.fromisoformat(wts) + dt.timedelta(seconds=abs_in)
            hh_mm_ss = wall_dt.strftime("%H:%M:%S")
            hh_mm = wall_dt.strftime("%H:%M")
            hr_eff = wall_dt.hour
        except ValueError:
            hh_mm_ss, hh_mm, hr_eff = "12:00:00", "12:00", 12

        clip_ep = parse_wall_epoch_fn(wts, float(abs_in))
        sort_key = (clip_ep if clip_ep is not None else float("inf"), str(r.get("source_rel") or ""), float(abs_in))
        if face_active:
            guarded_note, verified_people = face_service.sanitize_text(
                note_zh, day=day, match_res=face_match, vlm_ans=ans, day_confirmed_roles=day_confirmed_local,
            )
            verified_local_roles = set((face_match or {}).get("verified_roles") or set())
            verified_vlm_roles = extract_high_conf_vlm_roles(ans)
            shot_has_ref = bool((face_match or {}).get("has_enrolled_members"))
        else:
            guarded_note = sanitize_caption_kinship(
                note_zh,
                active_roles=active_roles_day,
                strict_unverified=True,
                day_confirmed_roles=day_confirmed_local,
            )
            verified_people = []
            verified_local_roles = set()
            verified_vlm_roles = set()
            shot_has_ref = False

        verified_shots.append({
            "row": r,
            "src": cand["src"],
            "cam_id": cam,
            "cam_name": cand["cam_name"],
            "cam_short": cand["cam_name"].replace(" G350", "").replace("宝宝", ""),
            "c_cfg": c_cfg,
            "source_kind": cand["source_kind"],
            "abs_in": abs_in,
            "dur_s": dur_s,
            "max_clip_dur": min(9.2, clip_dur - abs_in),
            "shot_scale": scale,
            "pan_style": pan,
            "child_box": cbox if (stamped or cand["cloud_ok"]) else None,
            "privacy_box": pbox,
            "categories": [c for c in (ans.get("categories") or []) if c in VALID_MILESTONE_CATS],
            "independence": str(ans.get("independence") or "unclear"),
            "emotion_tag": emo,
            "cinematic_score": max(c_score, 5.0 if cand["source_kind"] == "speech" else c_score),
            "director_note_zh": _clean_warm_text(guarded_note, 40) or "温暖成长瞬间",
            "verified_people": verified_people,
            "verified_local_roles": verified_local_roles,
            "verified_vlm_roles": verified_vlm_roles,
            "has_enrolled_ref": shot_has_ref,
            "hh_mm_ss": hh_mm_ss,
            "hh_mm": hh_mm,
            "act_hint": act_for_hour(hr_eff),
            "wall_ep": clip_ep,
            "sort_key": sort_key,
            "poster_jpg": poster_jpg,
            "boost_vocal": bool(cand["boost_vocal"] or emo in ("joy", "speech")),
            "speech_meta": cand.get("speech_meta"),
            "ans": ans,
        })

        cats_v = [c for c in (ans.get("categories") or []) if c in VALID_MILESTONE_CATS]
        if cats_v and c_score >= 4.0 and cand["source_kind"] != "speech":
            zoom_m = float(c_cfg.get("zoom_factor", 1.35))
            milestone_candidates.append({
                "row": r,
                "src": cand["src"],
                "cam": cam,
                "cname": cand["cam_name"],
                "cat": cats_v[0],
                "s0": abs_in,
                "dur_s": dur_s,
                "box": cbox,
                "pbox": pbox,
                "zoom": zoom_m,
                "t_lbl": hh_mm_ss,
                "exc": c_score,
                "caption_zh": guarded_note,
                "people": verified_people,
                "independence": str(ans.get("independence") or "unclear"),
                "provider": ans.get("_provider"),
                "model": ans.get("_model"),
            })

    if not verified_shots:
        return [], {}, []

    # Sort chronologically and assign stable Storyboard IDs (S01, S02, ...)
    verified_shots.sort(key=lambda v: v["sort_key"])
    for idx_v, v in enumerate(verified_shots, 1):
        v["shot_id"] = f"S{idx_v:02d}"

    # Apply legacy threshold filter if all Stage-3 answers came from minimal unit-test stub without screenplay call
    all_stub = all(_is_legacy_stub_response(v["ans"]) for v in verified_shots)
    if is_small_stub and all_stub:
        if any(v["cinematic_score"] >= 2.0 for v in verified_shots):
            kept = [v for v in verified_shots if v["cinematic_score"] >= 2.0]
        else:
            kept = list(verified_shots)
        shots, story_doc = assemble_final_edit_list(
            day,
            age_info,
            kept,
            None,
            active_roles=active_roles_day,
            has_enrolled_ref=bool(ref_imgs),
            strict_unverified=strict_unverified_day,
        )
        return shots, story_doc, milestone_candidates[:8]

    # Stage 4: Highest-thinking Master Director & Screenwriter over Storyboard Sheets
    progress_fn(tier, day, "总导演编剧统筹", f"总导演创作四幕短剧剧本与配乐转场节拍（{len(verified_shots)} 镜候选）", background)
    sheets = build_storyboard_sheets(verified_shots[:28], max_per_sheet=14)
    script_ans: dict[str, Any] | None = None
    if sheets:
        age_str = (
            f"出生第 {age_info.get('day_n')} 天 · {age_info.get('label_zh')}"
            if age_info.get("day_n")
            else day
        )
        manifest_rows = [
            {
                "shot_id": v["shot_id"],
                "time": v["hh_mm_ss"],
                "act": v["act_hint"],
                "camera": v["cam_name"],
                "avail_dur_s": round(v["dur_s"], 2),
                "suggested_dur_s": v["dur_s"],
                "scale": v["shot_scale"],
                "emotion": v["emotion_tag"],
                "score": round(v["cinematic_score"], 1),
                "has_speech": v["speech_meta"].get("matched_word") if v.get("speech_meta") else "",
                "people": v.get("verified_people") or [],
                "note": v["director_note_zh"],
            }
            for v in verified_shots[:28]
        ]
        roster_tip = f"\n【当日家庭成员身份守则】当日有效成员：{'、'.join(sorted(active_roles_day))}；仅可对 `people` 非空的镜头使用对应具体称呼，其余镜头一律写“家人”或省略称呼。"
        sp_prompt = (
            f"{PROMPT_DIRECTOR_SCREENPLAY}{roster_tip}\n\n"
            f"【今日档案】日期：{day}，宝宝月龄：{age_str}。\n"
            f"【通过质检的可选分镜清单（共 {len(manifest_rows)} 镜，按时间顺序）】\n"
            f"{json.dumps(manifest_rows, ensure_ascii=False)}"
        )
        try:
            accepts_refs = False
            if ref_imgs:
                try:
                    sig_p = inspect.signature(vlm.ask).parameters
                    accepts_refs = "reference_images" in sig_p or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_p.values()
                    )
                except (TypeError, ValueError):
                    accepts_refs = False
            if ref_imgs and accepts_refs:
                script_ans = vlm.ask(f"dir_script|{day}|v3", sheets, sp_prompt, tier="director", reference_images=ref_imgs)
            else:
                script_ans = vlm.ask(f"dir_script|{day}|v3", sheets, sp_prompt, tier="director")
        except Exception:
            script_ans = None

    final_shots, story_doc = assemble_final_edit_list(
        day,
        age_info,
        verified_shots,
        script_ans,
        active_roles=active_roles_day,
        has_enrolled_ref=bool(ref_imgs),
        strict_unverified=strict_unverified_day,
    )
    return final_shots, story_doc, milestone_candidates[:8]
