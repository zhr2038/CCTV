import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))
from nas_growth import asr_worker, daemon
from test_nas_growth_v3 import _make_test_daemon


def burst(index, rel):
    return {"camera": "cam1", "source_rel": rel, "snr_ratio": 8.0, "y_child_babble": 0.8,
            "burst_start_s": 10 * index + 1.0, "burst_end_s": 10 * index + 1.5}


def result(b):
    return {**b, "matched_word": "妈妈", "start_s": b["burst_start_s"] - 1, "end_s": b["burst_end_s"] + 2,
            "utterance_t_s": b["burst_start_s"] + 0.2, "f0_median_hz": 380, "phonetic_prob": 0.75,
            "syllable_type": "reduplicated"}


@pytest.mark.parametrize("interruption", ["timeout", "nonzero"])
def test_day_resumes_partial_speech_including_negatives_and_low_memory(tmp_path, monkeypatch, interruption):
    dm, data_root, _ = _make_test_daemon(tmp_path, monkeypatch)
    day = "2026-11-17"
    row = dm.db.get_day_rows(day)[0]
    cands = [burst(i, row["source_rel"]) for i in range(3)]
    dm.db.mark_file_done(row["source_rel"], {"duration": 40, "width": 1920, "height": 1080, "t2_rel": "t2.mp4", "bursts": cands})
    monkeypatch.setattr(daemon.med, "concat_mp4_list", lambda *_: True)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"test")
    dm.cfg["asr"]["model_dir"] = str(model_dir)
    dm.cfg["scheduler"]["asr_min_host_mem_available_kib"] = 10
    memory = [50]
    monkeypatch.setattr(daemon, "read_host_mem_available_kib", lambda: memory[0])
    monkeypatch.setattr(daemon, "read_container_headroom_kib", lambda: None)
    monkeypatch.setattr(dm, "_extract_cloud_frames", lambda *_: [])
    monkeypatch.setattr(daemon.med, "promote_t0_keeper", lambda *a: (False, None))
    monkeypatch.setattr(daemon.med, "render_single_clip_with_badge", lambda *a, **kw: True)
    seen = []
    starts = []

    def verify(b):
        seen.append(b["burst_start_s"])
        return None if b["burst_start_s"] == 11.0 else result(b)

    monkeypatch.setattr(asr_worker, "_load_verifier", lambda *a: verify)

    def launch(argv, **kw):
        starts.append(True)
        payload = json.loads(Path(argv[argv.index("--input") + 1]).read_text(encoding="utf-8"))
        ckpt = Path(argv[argv.index("--checkpoint-dir") + 1])
        if len(starts) == 1:
            asr_worker.verify_bursts(model_dir, "ffmpeg", payload[:2], checkpoint_dir=ckpt)
            if interruption == "timeout":
                raise subprocess.TimeoutExpired("asr", 300)
            return subprocess.CompletedProcess(argv, 4)
        asr_worker.verify_bursts(model_dir, "ffmpeg", payload, checkpoint_dir=ckpt)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(daemon.subprocess, "run", launch)
    summary = dm.finalize_day(day, "P1_BACKFILL", final=True)
    assert (summary["status"], summary["asr_status"], summary["speech_count"]) == ("preview", "failed", 1)
    assert seen == [1.0, 11.0]
    records = [p.read_text(encoding="utf-8") for p in (data_root / "state" / "asr" / day).glob("*.json")]
    assert "null" in records and all("source_path" not in item for item in records)

    memory[0] = 0
    paused = dm.finalize_day(day, "P1_BACKFILL", final=True)
    assert (paused["status"], paused["asr_status"], paused["speech_count"]) == ("preview", "deferred_memory", 1)
    assert len(starts) == 1

    memory[0] = 50
    finished = dm.finalize_day(day, "P1_BACKFILL", final=True)
    assert (finished["status"], finished["asr_status"], finished["speech_count"]) == ("done", "complete", 2)
    assert seen == [1.0, 11.0, 21.0]
    memory[0] = 0
    assert dm.finalize_day(day, "P1_BACKFILL", final=True)["asr_status"] == "complete"
    assert len(starts) == 2


def test_candidate_checkpoint_expires_on_source_model_or_evidence_change(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "model.bin"
    weights.write_bytes(b"m")
    src = tmp_path / "source.mp4"
    src.write_bytes(b"a")
    cands = [{**burst(0, "cam/source.mp4"), "source_path": str(src)}]
    ckpt = tmp_path / "state"
    seen = []
    monkeypatch.setattr(asr_worker, "_load_verifier", lambda *args: lambda b: seen.append(b["burst_start_s"]))
    for _ in range(2):
        assert not asr_worker.verify_bursts(model_dir, "ffmpeg", cands, checkpoint_dir=ckpt)
    assert len(seen) == 1
    src.write_bytes(b"bigger")
    asr_worker.verify_bursts(model_dir, "ffmpeg", cands, checkpoint_dir=ckpt)
    weights.write_bytes(b"new model")
    asr_worker.verify_bursts(model_dir, "ffmpeg", cands, checkpoint_dir=ckpt)
    cands[0]["snr_ratio"] += 1
    asr_worker.verify_bursts(model_dir, "ffmpeg", cands, checkpoint_dir=ckpt)
    assert len(seen) == 4
