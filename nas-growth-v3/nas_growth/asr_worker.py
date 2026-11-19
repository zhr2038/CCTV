"""Ephemeral CPU INT8 Whisper worker for isolated child-only '妈妈/妈' & '爸爸/爸' verification.

Audio NEVER leaves the NAS and zero non-matching transcripts are stored or logged.
Run as `python -m nas_growth.asr_worker --model-dir ... --ffmpeg ... --input in.json --output out.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


REDUPLICATED_MA_RE = re.compile(r"^(?:妈妈+|媽媽+|麻麻+|妈咪|媽咪|嘛嘛+|吗吗+|马马+|馬馬+)[呀啊哦呢啦]?$")
SINGLE_MA_RE = re.compile(r"^(?:[阿啊]?[妈媽麻马馬吗嗎嘛嬷])[呀啊哦呢]?$")
REDUPLICATED_BA_RE = re.compile(r"^(?:爸爸+|粑粑+|爸比|吧吧+|叭叭+|八八+|巴巴+)[呀啊哦呢啦]?$")
SINGLE_BA_RE = re.compile(r"^(?:[阿啊]?[爸粑吧叭把八巴拔罢])[呀啊哦呢]?$")

MA_WORDS = ["妈妈", "媽媽", "麻麻", "妈咪", "妈", "媽", "麻", "马", "吗", "嘛", "嬷", "阿嬷"]
BA_WORDS = ["爸爸", "粑粑", "爸", "吧", "叭", "把", "八", "巴"]
# Pure acoustic babble/syllable foils (exclude adult conversational priors like '好'/'到'/'我'/'那')
FOIL_WORDS = ["啊", "呀", "哦", "嗯", "咿", "哇", "哈", "哒", "大", "嘎", "呐", "啦", "哎"]
ALL_WORDS = MA_WORDS + BA_WORDS + FOIL_WORDS


def _clean_zh(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?\"'“”‘’~～…—·_-]+", "", text or "")


def _write_json(path: Path, value: Any) -> None:
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def decode_isolated_burst_16k(ff_bin: str, source_path: str, burst_start_s: float, burst_end_s: float) -> np.ndarray | None:
    """Decode [burst_start-0.10, burst_end+0.12], apply Hann fade + peak normalize, pad with zeros to 1.50s."""
    sr = 16000
    s0 = max(0.0, float(burst_start_s) - 0.10)
    dur_s = max(0.18, (float(burst_end_s) + 0.12) - s0)
    r = subprocess.run(
        [
            ff_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{s0:.3f}", "-t", f"{dur_s:.3f}", "-i", source_path,
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sr),
            "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
        ],
        capture_output=True,
        timeout=45,
    )
    r.check_returncode()
    arr = np.frombuffer(r.stdout[: (len(r.stdout) // 4) * 4], dtype="<f4").copy()
    if arr.size < int(0.18 * sr):
        return None
    n_in = min(int(0.045 * sr), arr.size // 3)
    n_out = min(int(0.060 * sr), arr.size // 3)
    if n_in > 1:
        arr[:n_in] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_in, dtype=np.float32))
    if n_out > 1:
        arr[-n_out:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, n_out, dtype=np.float32))
    pk = float(np.max(np.abs(arr)))
    if pk < 1e-4:
        return None
    arr *= 0.65 / pk
    lead = np.zeros(int(0.25 * sr), dtype=np.float32)
    trail_n = max(int(0.25 * sr), int(1.50 * sr) - lead.size - arr.size)
    return np.concatenate([lead, arr, np.zeros(trail_n, dtype=np.float32)])


def _load_verifier(model_dir: str | Path, ff_bin: str, cpu_threads: int):
    import ctranslate2
    from faster_whisper import WhisperModel
    from tokenizers import Tokenizer

    mdir = Path(model_dir)
    fw = WhisperModel(str(mdir), device="cpu", compute_type="int8", cpu_threads=cpu_threads, local_files_only=True)
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    prompt = [
        tok.token_to_id("<|startoftranscript|>"),
        tok.token_to_id("<|zh|>"),
        tok.token_to_id("<|transcribe|>"),
        tok.token_to_id("<|notimestamps|>"),
    ]
    word_toks = {w: tok.encode(w, add_special_tokens=False).ids for w in ALL_WORDS}
    first_kin_ids = {word_toks[w][0] for w in (MA_WORDS + BA_WORDS) if word_toks[w]}

    def verify_one(b: dict[str, Any]) -> dict[str, Any] | None:
        wave = decode_isolated_burst_16k(ff_bin, b["source_path"], b["burst_start_s"], b["burst_end_s"])
        if wave is None:
            return None

        feat = fw.feature_extractor(wave)
        mel = np.zeros((feat.shape[0], 3000), dtype=np.float32)
        n_copy = min(3000, feat.shape[1])
        mel[:, :n_copy] = feat[:, :n_copy]
        enc = fw.model.encode(ctranslate2.StorageView.from_array(np.ascontiguousarray(mel[None, ...])))

        gen_res = fw.model.generate(enc, [list(prompt)], beam_size=2, max_length=12, suppress_blank=True)[0]
        clean = _clean_zh(tok.decode(gen_res.sequences_ids[0], skip_special_tokens=True))

        is_red_ma = bool(REDUPLICATED_MA_RE.match(clean))
        is_ma_1 = bool(SINGLE_MA_RE.match(clean))
        is_red_ba = bool(REDUPLICATED_BA_RE.match(clean))
        is_ba_1 = bool(SINGLE_BA_RE.match(clean))

        logit_cache: dict[tuple[int, ...], np.ndarray] = {}

        def get_lp(prefix: tuple[int, ...]) -> np.ndarray:
            arr_lp = logit_cache.get(prefix)
            if arr_lp is None:
                res = fw.model.generate(
                    enc, [list(prefix)], beam_size=1, max_length=len(prefix) + 1,
                    suppress_tokens=[], suppress_blank=False, return_logits_vocab=True,
                )[0]
                ll = np.array(res.logits[0][0], dtype=np.float64)
                ll -= np.max(ll)
                arr_lp = ll - math.log(np.sum(np.exp(ll)))
                logit_cache[prefix] = arr_lp
            return arr_lp

        lp0 = get_lp(tuple(prompt))
        top5_ids = set(int(i) for i in np.argsort(-lp0)[:5])
        top_kin_lp = max((float(lp0[tid]) for tid in (top5_ids & first_kin_ids)), default=-99.0)
        if not (is_red_ma or is_ma_1 or is_red_ba or is_ba_1 or top_kin_lp >= -3.5):
            return None

        scores: dict[str, float] = {}
        for w in ALL_WORDS:
            tids = word_toks[w]
            if not tids:
                continue
            cp = tuple(prompt)
            vals = []
            for tid in tids:
                vals.append(float(get_lp(cp)[tid]))
                cp = cp + (tid,)
            scores[w] = float(np.mean(vals))

        ma_lp = max(scores[w] for w in MA_WORDS if w in scores)
        ba_lp = max(scores[w] for w in BA_WORDS if w in scores)
        foil_lp = max(scores[w] for w in FOIL_WORDS if w in scores)
        m3 = max(ma_lp, ba_lp, foil_lp)
        lz = m3 + math.log(math.exp(ma_lp - m3) + math.exp(ba_lp - m3) + math.exp(foil_lp - m3))
        ma_post = math.exp(ma_lp - lz)
        ba_post = math.exp(ba_lp - lz)

        ktype = None
        display_word = ""
        best_lp = -99.0
        best_post = 0.0
        eff_syl = "single_syllable"

        if (ma_lp - foil_lp >= 0.45) and (ma_lp - ba_lp >= 0.38) and (ma_post >= 0.56):
            if is_red_ma or (not is_ba_1 and not is_red_ba) or (ma_lp >= ba_lp + 0.8):
                ktype = "call_mama"
                best_lp = ma_lp
                best_post = ma_post
                is_double = is_red_ma or (
                    b.get("burst_dur_s", 0.0) >= 0.52 and max(scores["妈妈"], scores["媽媽"], scores["麻麻"]) > max(scores["妈"], scores["媽"], scores["嬷"]) + 0.15
                )
                eff_syl = "reduplicated" if is_double else "single_syllable"
                display_word = "妈妈" if is_double else "妈 (单音节)"
        elif (ba_lp - foil_lp >= 0.45) and (ba_lp - ma_lp >= 0.38) and (ba_post >= 0.56):
            if is_red_ba or is_ba_1 or (ba_lp >= ma_lp + 0.8):
                ktype = "call_papa"
                best_lp = ba_lp
                best_post = ba_post
                is_double = is_red_ba or (
                    b.get("burst_dur_s", 0.0) >= 0.52 and max(scores["爸爸"], scores["粑粑"]) > max(scores["爸"], scores["吧"]) + 0.15
                )
                eff_syl = "reduplicated" if is_double else "single_syllable"
                display_word = "爸爸" if is_double else "爸 (单音节)"

        if ktype:
            utt_t = round(0.5 * (b["burst_start_s"] + b["burst_end_s"]), 2)
            return {
                "camera": b["camera"],
                "source_rel": b["source_rel"],
                "source_path": b["source_path"],
                "burst_start_s": b["burst_start_s"],
                "burst_end_s": b["burst_end_s"],
                "start_s": round(max(0.0, b["burst_start_s"] - 1.4), 2),
                "end_s": round(b["burst_end_s"] + 2.8, 2),
                "utterance_t_s": utt_t,
                "keyword_type": ktype,
                "syllable_type": eff_syl,
                "matched_word": display_word,
                "f0_median_hz": b["f0_median_hz"],
                "snr_ratio": b["snr_ratio"],
                "y_child_babble": b["y_child_babble"],
                "phonetic_prob": round(best_post, 3),
                "foil_margin_nats": round(best_lp - foil_lp, 3),
                "kin_margin_nats": round(abs(ma_lp - ba_lp), 3),
                "speaker_verified": "infant_only_verified",
            }
        return None

    return verify_one


def _rank_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits.sort(key=lambda h: (-(1 if h["syllable_type"] == "reduplicated" else 0), -h["phonetic_prob"]))
    deduped: list[dict[str, Any]] = []
    for h in hits:
        if any(d["source_rel"] == h["source_rel"] and abs(d["utterance_t_s"] - h["utterance_t_s"]) < 5.0 for d in deduped):
            continue
        deduped.append(h)
    return deduped


def checkpoint_jobs(model_dir: str | Path, bursts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Identify candidates by model, verifier code, source recording version and acoustic evidence."""
    bursts = [b for b in bursts if float(b.get("snr_ratio", 0.0)) >= 3.2]
    if not bursts:
        return {}
    model = Path(model_dir)
    model_files = []
    for p in sorted(model.iterdir()):
        if p.is_file():
            st = p.stat()
            model_files.append((p.name, st.st_size, st.st_mtime_ns))
    version = [hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), str(model.resolve()), model_files]
    jobs = {}
    for burst in bursts:
        st = Path(burst["source_path"]).stat()
        evidence = {k: v for k, v in burst.items() if k != "source_path"}
        identity = [version, st.st_size, st.st_mtime_ns, evidence]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        jobs[key] = burst
    return jobs


def read_checkpoint(directory: Path, jobs: dict[str, dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    completed = set()
    hits = []
    for key, burst in jobs.items():
        try:
            result = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            continue
        if result is not None:
            if not isinstance(result, dict) or result.get("source_rel") != burst["source_rel"]:
                continue
            if not {"syllable_type", "phonetic_prob", "utterance_t_s"}.issubset(result):
                continue
            hits.append({**result, "source_path": burst["source_path"]})
        completed.add(key)
    return completed, _rank_hits(hits)


def verify_bursts(
    model_dir: str | Path, ff_bin: str, bursts: list[dict[str, Any]], cpu_threads: int = 2,
    out_path: Path | None = None, checkpoint_dir: Path | None = None,
) -> list[dict[str, Any]]:
    completed: set[str] = set()
    hits: list[dict[str, Any]] = []
    if checkpoint_dir is None:
        jobs = {str(i): b for i, b in enumerate(bursts) if float(b.get("snr_ratio", 0.0)) >= 3.2}
    else:
        jobs = checkpoint_jobs(model_dir, bursts)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        completed, hits = read_checkpoint(checkpoint_dir, jobs)
    if len(completed) == len(jobs):
        return hits
    verify_one = _load_verifier(model_dir, ff_bin, cpu_threads)
    failure = None
    for key, burst in jobs.items():
        if key in completed:
            continue
        try:
            hit = verify_one(burst)
        except (OSError, subprocess.SubprocessError) as exc:
            failure = exc
            continue
        if checkpoint_dir is not None:
            # A JSON null is a fully processed non-match. No transcript or audio is persisted.
            result = {k: v for k, v in hit.items() if k != "source_path"} if hit is not None else None
            _write_json(checkpoint_dir / f"{key}.json", result)
        if hit is not None:
            hits.append(hit)
            if out_path is not None:
                _write_json(out_path, _rank_hits(hits.copy()))
    if failure is not None:
        raise failure
    if checkpoint_dir is not None:
        return read_checkpoint(checkpoint_dir, jobs)[1]
    return _rank_hits(hits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)

    bursts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    hits = verify_bursts(args.model_dir, args.ffmpeg, bursts, cpu_threads=args.threads,
                         out_path=Path(args.output), checkpoint_dir=args.checkpoint_dir)
    _write_json(Path(args.output), hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
