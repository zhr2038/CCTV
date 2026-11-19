from __future__ import annotations

import csv
import hashlib
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .media import resolve_ffmpeg


SAMPLE_RATE = 16_000
MODEL_NAME = "YAMNet TFLite classification v1"

TARGET_CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "laughter": ("Laughter", "Baby laughter", "Giggle", "Chuckle, chortle"),
    "child_vocalization": ("Child speech, kid speaking", "Babbling"),
    "speech": (
        "Speech",
        "Child speech, kid speaking",
        "Conversation",
        "Babbling",
    ),
    "baby_cry": (
        "Baby cry, infant cry",
        "Crying, sobbing",
    ),
    "scream": ("Screaming",),
    "yell": (
        "Yell",
        "Shout",
        "Children shouting",
    ),
    "impact": (
        "Thump, thud",
        "Bang",
        "Slap, smack",
        "Smash, crash",
        "Breaking",
    ),
    "glass_breaking": (
        "Glass",
        "Shatter",
    ),
    "smoke_alarm": ("Smoke detector, smoke alarm",),
    "fire_alarm": ("Fire alarm",),
    "knock": ("Knock",),
    "running_water": (
        "Water",
        "Stream",
        "Water tap, faucet",
    ),
}


@dataclass(frozen=True)
class AudioEvent:
    label: str
    model_label: str
    model_class_index: int
    media_start_seconds: float
    media_end_seconds: float
    confidence: float
    windows: int = 1


@dataclass(frozen=True)
class AudioAnalysis:
    events: tuple[AudioEvent, ...]
    windows_seen: int
    raw_detections: int
    model: str
    model_sha256: str
    minimum_confidence: float
    merge_gap_seconds: float


@dataclass(frozen=True)
class AudioFingerprint:
    mean_scores: tuple[float, ...]
    max_scores: tuple[float, ...]
    windows_seen: int
    media_start_seconds: float
    media_end_seconds: float
    model_sha256: str


def load_class_names(path: str | Path) -> tuple[str, ...]:
    names: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            names[index] = row["display_name"]
    if not names:
        raise ValueError("Audio class map is empty")
    expected = list(range(max(names) + 1))
    if sorted(names) != expected:
        raise ValueError("Audio class map indexes must be contiguous from zero")
    return tuple(names[index] for index in expected)


def resolve_target_indexes(
    configured_labels: Iterable[str],
    class_names: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    by_name = {name: index for index, name in enumerate(class_names)}
    targets: dict[str, tuple[int, ...]] = {}
    for label in configured_labels:
        if label not in TARGET_CLASS_NAMES:
            raise ValueError(f"Unsupported audio event label: {label}")
        missing = [
            name
            for name in TARGET_CLASS_NAMES[label]
            if name not in by_name
        ]
        if missing:
            raise ValueError(
                f"Audio class map is missing classes for {label}: "
                + ", ".join(missing)
            )
        targets[label] = tuple(
            by_name[name] for name in TARGET_CLASS_NAMES[label]
        )
    return targets


def merge_audio_events(
    events: Iterable[AudioEvent],
    gap_seconds: float,
) -> tuple[AudioEvent, ...]:
    merged: list[AudioEvent] = []
    ordered = sorted(
        events,
        key=lambda event: (
            event.label,
            event.media_start_seconds,
            event.media_end_seconds,
        ),
    )
    for event in ordered:
        if (
            merged
            and merged[-1].label == event.label
            and event.media_start_seconds
            <= merged[-1].media_end_seconds + gap_seconds
        ):
            previous = merged[-1]
            best = event if event.confidence > previous.confidence else previous
            merged[-1] = replace(
                previous,
                model_label=best.model_label,
                model_class_index=best.model_class_index,
                media_end_seconds=max(
                    previous.media_end_seconds,
                    event.media_end_seconds,
                ),
                confidence=max(previous.confidence, event.confidence),
                windows=previous.windows + event.windows,
            )
        else:
            merged.append(event)
    return tuple(
        sorted(
            merged,
            key=lambda event: (event.media_start_seconds, event.label),
        )
    )


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class YamnetDetector:
    def __init__(
        self,
        *,
        model_path: str | Path,
        class_map_path: str | Path,
        event_labels: Iterable[str],
        minimum_confidence: float,
        merge_gap_seconds: float,
    ) -> None:
        try:
            import numpy as np
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Audio dependencies are missing; install with "
                "`pip install -e .[audio-events]`"
            ) from exc

        if not 0 < minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in (0, 1]")
        if merge_gap_seconds < 0:
            raise ValueError("merge_gap_seconds cannot be negative")

        self._np = np
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Audio model not found: {self.model_path}; "
                "run `cctv-ai audio-model-install`"
            )
        if not self.class_map_path.is_file():
            raise FileNotFoundError(
                f"Audio class map not found: {self.class_map_path}; "
                "run `cctv-ai audio-model-install`"
            )

        self.class_names = load_class_names(self.class_map_path)
        self.target_indexes = resolve_target_indexes(
            event_labels,
            self.class_names,
        )
        self.minimum_confidence = minimum_confidence
        self.merge_gap_seconds = merge_gap_seconds
        self.model_sha256 = hashlib.sha256(
            self.model_path.read_bytes()
        ).hexdigest()

        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError("Expected one YAMNet input and one output tensor")
        self.input = input_details[0]
        self.output = output_details[0]
        input_shape = tuple(int(value) for value in self.input["shape"])
        output_shape = tuple(int(value) for value in self.output["shape"])
        if len(input_shape) != 1 or input_shape[0] <= 0:
            raise ValueError(f"Unexpected YAMNet input shape: {input_shape}")
        if not output_shape or output_shape[-1] != len(self.class_names):
            raise ValueError(
                "YAMNet output/class-map mismatch: "
                f"{output_shape} versus {len(self.class_names)} classes"
            )
        if self.input["dtype"] != np.float32:
            raise ValueError(
                f"Expected float32 YAMNet input, got {self.input['dtype']}"
            )
        self.window_samples = input_shape[0]
        self.window_seconds = self.window_samples / SAMPLE_RATE

    def _score_window(
        self,
        waveform,
        media_start_seconds: float,
        media_end_seconds: float,
    ) -> list[AudioEvent]:
        scores = self._invoke_scores(waveform)
        detections: list[AudioEvent] = []
        for label, indexes in self.target_indexes.items():
            best_index = max(indexes, key=lambda index: float(scores[index]))
            confidence = float(scores[best_index])
            if confidence >= self.minimum_confidence:
                detections.append(
                    AudioEvent(
                        label=label,
                        model_label=self.class_names[best_index],
                        model_class_index=best_index,
                        media_start_seconds=media_start_seconds,
                        media_end_seconds=media_end_seconds,
                        confidence=confidence,
                    )
                )
        return detections

    def _invoke_scores(self, waveform):
        self.interpreter.set_tensor(self.input["index"], waveform)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self.output["index"]
        ).reshape(-1).astype(self._np.float32, copy=True)

    def fingerprint(
        self,
        path: str | Path,
        *,
        media_start_seconds: float,
        media_end_seconds: float,
        padding_seconds: float = 0.5,
        ffmpeg: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> AudioFingerprint:
        if media_start_seconds < 0:
            raise ValueError("media_start_seconds cannot be negative")
        if media_end_seconds <= media_start_seconds:
            raise ValueError(
                "media_end_seconds must be after media_start_seconds"
            )
        if padding_seconds < 0:
            raise ValueError("padding_seconds cannot be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        decode_start = max(0.0, media_start_seconds - padding_seconds)
        decode_end = media_end_seconds + padding_seconds
        decode_duration = max(
            self.window_seconds,
            decode_end - decode_start,
        )
        executable = resolve_ffmpeg(ffmpeg)
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{decode_start:.6f}",
            "-i",
            str(path),
            "-t",
            f"{decode_duration:.6f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "ffmpeg fingerprint decode exceeded "
                f"{timeout_seconds:g} seconds"
            ) from exc
        if completed.returncode != 0:
            message = completed.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
            raise RuntimeError(
                message
                or "ffmpeg fingerprint decode exited with "
                f"{completed.returncode}"
            )

        waveform = self._np.frombuffer(
            completed.stdout[: len(completed.stdout) // 4 * 4],
            dtype="<f4",
        )
        minimum_samples = SAMPLE_RATE // 4
        score_rows = []
        for offset in range(0, len(waveform), self.window_samples):
            window = waveform[offset : offset + self.window_samples]
            if len(window) < minimum_samples:
                break
            if len(window) < self.window_samples:
                padded = self._np.zeros(
                    self.window_samples,
                    dtype=self._np.float32,
                )
                padded[: len(window)] = window
                window = padded
            else:
                window = window.astype(self._np.float32, copy=False)
            score_rows.append(self._invoke_scores(window))
        if not score_rows:
            raise RuntimeError("Decoded fingerprint segment has no audio")

        matrix = self._np.stack(score_rows)
        return AudioFingerprint(
            mean_scores=tuple(
                float(value) for value in matrix.mean(axis=0)
            ),
            max_scores=tuple(
                float(value) for value in matrix.max(axis=0)
            ),
            windows_seen=len(score_rows),
            media_start_seconds=decode_start,
            media_end_seconds=decode_start + decode_duration,
            model_sha256=self.model_sha256,
        )

    def analyze(
        self,
        path: str | Path,
        *,
        ffmpeg: str | None = None,
        timeout_seconds: float = 180.0,
    ) -> AudioAnalysis:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        executable = resolve_ffmpeg(ffmpeg)
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Unable to open ffmpeg audio pipes")

        # FFmpeg can emit many repeated decode errors for damaged camera
        # containers. Drain stderr while stdout is consumed so the OS pipe
        # cannot fill up and deadlock an otherwise recoverable audio stream.
        stderr_tail = bytearray()
        stderr_limit = 64 * 1024

        def drain_stderr() -> None:
            while True:
                chunk = process.stderr.read(8192)
                if not chunk:
                    return
                stderr_tail.extend(chunk)
                overflow = len(stderr_tail) - stderr_limit
                if overflow > 0:
                    del stderr_tail[:overflow]

        stderr_thread = threading.Thread(
            target=drain_stderr,
            name="cctv-ffmpeg-stderr",
            daemon=True,
        )
        stderr_thread.start()

        timed_out = threading.Event()

        def kill_timed_out_process() -> None:
            timed_out.set()
            try:
                process.kill()
            except OSError:
                pass

        timeout = threading.Timer(
            timeout_seconds,
            kill_timed_out_process,
        )
        timeout.daemon = True
        timeout.start()
        bytes_per_window = self.window_samples * 4
        minimum_partial_bytes = (SAMPLE_RATE // 4) * 4
        windows_seen = 0
        detections: list[AudioEvent] = []
        try:
            while True:
                window_bytes = _read_exact(process.stdout, bytes_per_window)
                if not window_bytes:
                    break
                if len(window_bytes) < minimum_partial_bytes:
                    break
                samples_in_window = len(window_bytes) // 4
                waveform = self._np.frombuffer(
                    window_bytes[: samples_in_window * 4],
                    dtype="<f4",
                )
                if samples_in_window < self.window_samples:
                    padded = self._np.zeros(
                        self.window_samples,
                        dtype=self._np.float32,
                    )
                    padded[:samples_in_window] = waveform
                    waveform = padded
                else:
                    waveform = waveform.astype(
                        self._np.float32,
                        copy=False,
                    )
                media_start = windows_seen * self.window_seconds
                media_end = media_start + (samples_in_window / SAMPLE_RATE)
                detections.extend(
                    self._score_window(waveform, media_start, media_end)
                )
                windows_seen += 1

            return_code = process.wait()
            stderr_thread.join(timeout=2.0)
            error_text = bytes(stderr_tail).decode(
                "utf-8",
                errors="replace",
            )
            if timed_out.is_set():
                raise TimeoutError(
                    "ffmpeg audio decode exceeded "
                    f"{timeout_seconds:g} seconds"
                )
            if return_code != 0:
                raise RuntimeError(
                    error_text.strip()
                    or f"ffmpeg audio decode exited with {return_code}"
                )
        except Exception:
            process.kill()
            process.wait()
            stderr_thread.join(timeout=2.0)
            raise
        finally:
            timeout.cancel()

        events = merge_audio_events(
            detections,
            self.merge_gap_seconds,
        )
        return AudioAnalysis(
            events=events,
            windows_seen=windows_seen,
            raw_detections=len(detections),
            model=MODEL_NAME,
            model_sha256=self.model_sha256,
            minimum_confidence=self.minimum_confidence,
            merge_gap_seconds=self.merge_gap_seconds,
        )
