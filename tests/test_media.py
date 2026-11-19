from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cctv_ai.media import parse_ffprobe, resolve_ffprobe


class MediaTests(unittest.TestCase):
    def test_resolve_explicit_ffprobe_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ffprobe.exe"
            executable.touch()
            self.assertEqual(
                resolve_ffprobe(str(executable)),
                str(executable),
            )

    def test_parse_audio_and_video(self) -> None:
        probe = parse_ffprobe(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "hevc",
                        "width": 2880,
                        "height": 1620,
                        "avg_frame_rate": "5400000/358111",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "opus",
                        "sample_rate": "48000",
                        "channels": 1,
                    },
                ],
                "format": {
                    "duration": "1227.912125",
                    "size": "134217728",
                    "bit_rate": "874445",
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                },
            }
        )
        self.assertEqual(probe.video_codec, "hevc")
        self.assertEqual(probe.audio_codec, "opus")
        self.assertEqual(probe.width, 2880)
        self.assertAlmostEqual(probe.duration_seconds or 0, 1227.912125)


if __name__ == "__main__":
    unittest.main()
