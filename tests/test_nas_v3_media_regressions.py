from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nas-growth-v3"))
from nas_growth import asr_worker, cloud_vlm, detectors, media_nas


@pytest.fixture
def ffmpeg():
    return shutil.which("ffmpeg") or pytest.skip("FFmpeg is required for media round trips")


class _NoPersons:
    def detect_persons(self, frames, conf_thr):
        return [[] for frame in frames]


class _EventModel:
    window_samples = 15600
    idx = {"child": 0, "babble": 1, "laughter": [], "cry": 2, "sob": 3, "speech": 4, "scream": 5}

    def score_window(self, pcm):
        return np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_t2_can_publish_and_survives_failed_reprocessing(tmp_path, ffmpeg, caplog):
    src, dst = tmp_path / "source.mp4", tmp_path / "t2.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=20",
                    "-t", "3", "-c:v", "libx264", str(src)], check=True)

    def run_t2():
        return media_nas.stream_1fps_t2_and_analyze(ffmpeg, str(src), dst, 160, 90, 3,
                                                    _NoPersons(), 0.28, False, t2_h=90)

    assert run_t2()["n_frames"] == 3
    good = dst.read_bytes()
    assert good
    src.write_bytes(b"truncated")
    assert run_t2() is None
    assert "decoder_rc=" in caplog.text
    assert dst.read_bytes() == good
    assert set(tmp_path.iterdir()) == {src, dst}


def test_t1_retains_previous_file_on_encode_error(tmp_path, ffmpeg):
    dst = tmp_path / "archive.mp4"
    dst.write_bytes(b"previous valid archive")
    assert media_nas.encode_t1_segments(ffmpeg, str(tmp_path / "missing.mp4"), dst, [[0, 3]],
                                       3, False, False) == (False, 0.0)
    assert dst.read_bytes() == b"previous valid archive"
    assert list(tmp_path.iterdir()) == [dst]


def test_t1_retiming_keeps_last_source_frames(tmp_path, ffmpeg):
    src, dst = tmp_path / "source.mp4", tmp_path / "archive.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=blue:160x90:r=20:d=28",
                    "-f", "lavfi", "-i", "color=yellow:160x90:r=20:d=2", "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264", str(src)], check=True)
    assert media_nas.encode_t1_segments(ffmpeg, str(src), dst, [[0, 30]], 30, False, False,
                                       t1_h=90, pts_jump=True) == (True, 30)
    pixels = subprocess.check_output(
        [ffmpeg, "-v", "error", "-i", str(dst), "-vf", "scale=1:1", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    )
    frames = np.frombuffer(pixels, np.uint8).reshape(-1, 3)
    assert len(frames) == 450
    assert frames[0, 2] > 200  # blue at start
    assert frames[-1, :2].min() > 200 and frames[-1, 2] < 40  # yellow at end


def test_t0_verified_copy_published_only_after_checksum(tmp_path, monkeypatch):
    src, dst = tmp_path / "source.mp4", tmp_path / "keeper.mp4"
    src.write_bytes(b"recording")
    dst.write_bytes(b"old keeper")

    def corrupt_copy(command, **kwargs):
        Path(command[-1]).write_bytes(b"copy failed")

    monkeypatch.setattr(media_nas.subprocess, "run", corrupt_copy)
    assert not media_nas.promote_t0_keeper(src, dst)[0]
    assert dst.read_bytes() == b"old keeper"
    assert set(tmp_path.iterdir()) == {src, dst}


def test_audio_decode_error_is_distinct_from_silence(tmp_path, ffmpeg):
    with pytest.raises(RuntimeError, match="audio_decode_failed"):
        detectors.analyze_audio_and_bursts(ffmpeg, str(tmp_path / "missing.mp4"), "src", "cam",
                                          _EventModel(), scan_bursts=False)


def test_final_audio_fraction_is_scored_at_its_source_second(tmp_path, ffmpeg):
    src = tmp_path / "audio.wav"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=400:duration=1.45",
                    "-ar", "16000", str(src)], check=True)
    audio = detectors.analyze_audio_and_bursts(ffmpeg, str(src), "src", "cam", _EventModel(), scan_bursts=False)
    assert audio["active_secs"] == [0, 1]


def test_low_snr_skips_model_loading(tmp_path):
    assert asr_worker.verify_bursts(tmp_path / "no-model", "missing-ffmpeg", [{"snr_ratio": 3.1}]) == []


def test_daily_concat_requires_all_inputs_and_cleans_up_on_timeout(tmp_path, ffmpeg, monkeypatch):
    ffprobe = shutil.which("ffprobe") or pytest.skip("ffprobe is required")
    first, second, output = tmp_path / "clip'1.mp4", tmp_path / "clip2.mp4", tmp_path / "daily.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:r=30:d=0.3",
                    "-c:v", "libx264", str(first)], check=True)
    shutil.copyfile(first, second)
    assert media_nas.concat_mp4_list(ffmpeg, [first, second], output, ffprobe)
    probe = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0",
                                     "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(output)])
    assert int(probe) == 18
    previous = output.read_bytes()
    missing = tmp_path / "missing.mp4"
    assert not media_nas.concat_mp4_list(ffmpeg, [first, missing], output, ffprobe)
    second.write_bytes(b"invalid mp4" * 300)
    assert not media_nas.concat_mp4_list(ffmpeg, [first, second], output, ffprobe)

    def timeout(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, timeout=120)

    monkeypatch.setattr(media_nas.subprocess, "run", timeout)
    assert not media_nas.concat_mp4_list(ffmpeg, [first, second], output, ffprobe)
    assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {first, second, output}


@pytest.mark.parametrize("failure_stage", ["mux", "poster"])
def test_render_keeps_previous_movie_and_poster_on_timeout(tmp_path, ffmpeg, monkeypatch, failure_stage):
    src, output = tmp_path / "src.mp4", tmp_path / "movie.mp4"
    poster = output.with_suffix(".jpg")
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:r=10:d=3",
                    "-c:v", "libx264", str(src)], check=True)
    shots = [{"source": str(src), "start_s": 0, "dur_s": 3, "nat_w": 160, "nat_h": 90, "has_audio": False}]

    def render():
        return media_nas.render_smooth_film_nas(ffmpeg, shots, output, None, False, fps=10, out_w=160, out_h=90)

    assert render()["shots"] == 1
    previous = output.read_bytes(), poster.read_bytes()
    real_run = subprocess.run

    def fail_one_stage(command, **kwargs):
        if (failure_stage == "mux" and "-shortest" in command) or (failure_stage == "poster" and "-q:v" in command):
            Path(command[-1]).write_bytes(b"partial")
            raise subprocess.TimeoutExpired(command, timeout=30)
        return real_run(command, **kwargs)

    monkeypatch.setattr(media_nas.subprocess, "run", fail_one_stage)
    assert render() is None
    assert (output.read_bytes(), poster.read_bytes()) == previous
    assert set(tmp_path.iterdir()) == {src, output, poster}


def test_t2_decode_timeout_budget_covers_heavy_bedside_bundles():
    assert media_nas.t2_decode_timeout_s(60.0) == 360.0
    # 1050s (17.5min) 2880x1620 bundle gets >1200s instead of ~682s
    assert media_nas.t2_decode_timeout_s(1050.0) >= 1207.0


def test_fast_clip_window_and_extract_jpegs_deep_offset(tmp_path, ffmpeg):
    src = tmp_path / "long_bundle.mp4"
    subprocess.run(
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:r=5:d=26",
         "-c:v", "libx264", "-g", "15", str(src)],
        check=True,
    )
    with media_nas.open_fast_clip_window(ffmpeg, str(src), 22.0, 2.5, include_audio=False) as (clip_src, rel_t):
        assert clip_src != str(src)
        assert Path(clip_src).is_file()
        assert abs(rel_t - 5.0) < 0.05

    frames = cloud_vlm.extract_jpegs_ram(ffmpeg, str(src), [22.0, 23.0], max_edge=160)
    assert len(frames) == 2
    assert all(frm.startswith(b"\xff\xd8") for frm in frames)


def test_extract_jpegs_ram_handles_timeout_without_cv2_fallback(tmp_path, monkeypatch):
    import cv2

    src = tmp_path / "dummy.mp4"
    src.write_bytes(b"0" * 2048)

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ffmpeg"], timeout=30)

    monkeypatch.setattr(cloud_vlm.subprocess, "run", raise_timeout)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_kw: pytest.fail("cv2.VideoCapture called after timeout"))
    assert cloud_vlm.extract_jpegs_ram("ffmpeg", str(src), [1.0, 2.0]) == []
