from __future__ import annotations

import io
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

import numpy as np

from cctv_ai.audio_events import (
    AudioEvent,
    YamnetDetector,
    merge_audio_events,
    resolve_target_indexes,
)


class AudioEventTests(unittest.TestCase):
    def test_fingerprint_decodes_only_event_segment_and_aggregates(self) -> None:
        detector = object.__new__(YamnetDetector)
        detector._np = np
        detector.window_samples = 4_000
        detector.window_seconds = 0.25
        detector.model_sha256 = "model-sha"
        detector._invoke_scores = Mock(
            side_effect=[
                np.asarray([0.2, 0.8], dtype=np.float32),
                np.asarray([0.6, 0.4], dtype=np.float32),
            ]
        )
        waveform = np.arange(8_000, dtype="<f4").tobytes()
        completed = CompletedProcess(
            args=[],
            returncode=0,
            stdout=waveform,
            stderr=b"",
        )

        with (
            patch(
                "cctv_ai.audio_events.resolve_ffmpeg",
                return_value="ffmpeg",
            ),
            patch(
                "cctv_ai.audio_events.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            fingerprint = detector.fingerprint(
                Path("camera/video.mp4"),
                media_start_seconds=10.0,
                media_end_seconds=12.0,
                padding_seconds=0.5,
                timeout_seconds=7.0,
            )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "9.500000")
        self.assertEqual(command[command.index("-t") + 1], "3.000000")
        self.assertEqual(run.call_args.kwargs["timeout"], 7.0)
        self.assertEqual(fingerprint.windows_seen, 2)
        self.assertEqual(fingerprint.model_sha256, "model-sha")
        self.assertAlmostEqual(fingerprint.mean_scores[0], 0.4)
        self.assertAlmostEqual(fingerprint.mean_scores[1], 0.6)
        self.assertAlmostEqual(fingerprint.max_scores[0], 0.6)
        self.assertAlmostEqual(fingerprint.max_scores[1], 0.8)

    def test_audio_decode_has_a_hard_timeout(self) -> None:
        released = threading.Event()

        class BlockingStream:
            def read(self, size: int) -> bytes:
                del size
                released.wait(timeout=1)
                return b""

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = BlockingStream()
                self.stderr = io.BytesIO()
                self.killed = False

            def kill(self) -> None:
                self.killed = True
                released.set()

            def wait(self) -> int:
                return -9 if self.killed else 0

        detector = object.__new__(YamnetDetector)
        detector.window_samples = 15_600
        process = FakeProcess()
        with (
            patch(
                "cctv_ai.audio_events.resolve_ffmpeg",
                return_value="ffmpeg",
            ),
            patch(
                "cctv_ai.audio_events.subprocess.Popen",
                return_value=process,
            ),
        ):
            with self.assertRaisesRegex(
                TimeoutError,
                "exceeded 0.02 seconds",
            ):
                detector.analyze(
                    "video.mp4",
                    timeout_seconds=0.02,
                )
        self.assertTrue(process.killed)

    def test_audio_decode_drains_ffmpeg_stderr_concurrently(self) -> None:
        stderr_started = threading.Event()

        class WaitingStdout:
            def read(self, size: int) -> bytes:
                del size
                if not stderr_started.wait(timeout=0.5):
                    raise RuntimeError("stderr was not drained")
                return b""

        class ErrorStream:
            def __init__(self) -> None:
                self.remaining = b"decode warning\n"

            def read(self, size: int) -> bytes:
                stderr_started.set()
                chunk = self.remaining[:size]
                self.remaining = self.remaining[size:]
                return chunk

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = WaitingStdout()
                self.stderr = ErrorStream()

            def kill(self) -> None:
                return None

            def wait(self) -> int:
                return 1

        detector = object.__new__(YamnetDetector)
        detector.window_samples = 15_600
        with (
            patch(
                "cctv_ai.audio_events.resolve_ffmpeg",
                return_value="ffmpeg",
            ),
            patch(
                "cctv_ai.audio_events.subprocess.Popen",
                return_value=FakeProcess(),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "decode warning"):
                detector.analyze(
                    "video.mp4",
                    timeout_seconds=1.0,
                )

    def test_resolve_target_indexes(self) -> None:
        classes = (
            "Speech",
            "Smoke detector, smoke alarm",
            "Fire alarm",
        )
        targets = resolve_target_indexes(("smoke_alarm", "fire_alarm"), classes)
        self.assertEqual(targets["smoke_alarm"], (1,))
        self.assertEqual(targets["fire_alarm"], (2,))

    def test_merge_adjacent_detections_keeps_best_model_label(self) -> None:
        events = [
            AudioEvent(
                label="impact",
                model_label="Bang",
                model_class_index=460,
                media_start_seconds=0.0,
                media_end_seconds=0.975,
                confidence=0.72,
            ),
            AudioEvent(
                label="impact",
                model_label="Thump, thud",
                model_class_index=454,
                media_start_seconds=1.95,
                media_end_seconds=2.925,
                confidence=0.91,
            ),
        ]
        merged = merge_audio_events(events, gap_seconds=1.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].model_label, "Thump, thud")
        self.assertEqual(merged[0].windows, 2)
        self.assertAlmostEqual(merged[0].media_end_seconds, 2.925)


if __name__ == "__main__":
    unittest.main()
