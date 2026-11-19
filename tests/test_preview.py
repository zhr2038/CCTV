from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cctv_ai.preview import generate_event_preview


class PreviewTests(unittest.TestCase):
    def test_source_path_cannot_escape_media_root(self) -> None:
        unsafe_paths = (
            "../secret.mp4",
            "/absolute/secret.mp4",
            "C:/secret.mp4",
            r"C:\secret.mp4",
            r"camera\..\secret.mp4",
        )
        with tempfile.TemporaryDirectory() as directory:
            for relative_path in unsafe_paths:
                with self.subTest(relative_path=relative_path):
                    with self.assertRaises(ValueError):
                        generate_event_preview(
                            {
                                "relative_path": relative_path,
                            },
                            source_root=Path(directory) / "source",
                            output_dir=Path(directory) / "output",
                        )


if __name__ == "__main__":
    unittest.main()
