from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from cctv_ai.database import connect_database
from cctv_ai.pipeline import PipelineOptions, run_pipeline_once
from cctv_ai.web import create_app


def write_config(path: Path, source_root: Path) -> None:
    path.write_text(
        f"""
[project]
timezone = "UTC"
source_root_windows = "{source_root.as_posix()}"
source_root_linux = "{source_root.as_posix()}"
stability_age_minutes = 90
stable_observations = 2

[privacy]
source_read_only = true
face_recognition = false
cloud_upload = false

[audio]
enabled = true
transcription_mode = "on_demand"
store_continuous_transcripts = false
model_path = "missing-model.tflite"
label_map_path = "missing-labels.csv"
minimum_confidence = 0.60
merge_gap_seconds = 1.0
event_labels = ["baby_cry"]

[[cameras]]
id = "cam"
name = "camera"
location = "test"
kind = "aqara_event"
relative_root = "camera"
audio_enabled = true
""".strip(),
        encoding="utf-8",
    )


def build_test_app(
    root: Path,
    *,
    event_count: int = 0,
    allowed_networks: tuple[str, ...] = (),
):
    source = root / "source"
    source.mkdir()
    config = root / "config.toml"
    write_config(config, source)
    database = root / "index.sqlite3"
    options = PipelineOptions(
        database_path=database,
        config_path=config,
        source_root=source,
        camera_ids=("cam",),
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        scan=False,
        probe_limit_per_camera=0,
        audio_limit_per_camera=0,
        watermark_limit_per_camera=0,
        lock_path=root / "cycle.lock",
    )
    run_pipeline_once(options)
    if event_count:
        connection = connect_database(database)
        base_ts = int(
            datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
        )
        for index in range(event_count):
            start_ts = base_ts + index * 120
            media = connection.execute(
                """
                INSERT INTO media_files(
                    camera_id, relative_path, size_bytes, mtime_ns,
                    first_seen_at, last_seen_at, status, capture_start_ts,
                    time_source, requires_watermark_ocr, duration_seconds
                )
                VALUES ('cam', ?, 100, ?, 1, 1, 'probed', ?,
                        'test', 0, 60)
                """,
                (f"camera/{index}.mp4", index + 1, start_ts),
            )
            connection.execute(
                """
                INSERT INTO audio_events(
                    media_file_id, event_label, model_label,
                    model_class_index, media_start_seconds,
                    media_end_seconds, wall_start_ts, wall_end_ts,
                    confidence, windows, model_name, created_at
                )
                VALUES (?, 'baby_cry', 'cry', 1, 0, 1, ?, ?,
                        ?, 1, 'test', 1)
                """,
                (
                    int(media.lastrowid),
                    start_ts,
                    start_ts + 1,
                    0.65 + (index % 5) * 0.05,
                ),
            )
        connection.commit()
        connection.close()
    app = create_app(
        database_path=database,
        config_path=config,
        source_root=source,
        preview_dir=root / "previews",
        visual_dir=root / "visual",
        allowed_networks=allowed_networks,
    )
    return app


class WebTests(unittest.TestCase):
    def test_diary_surface_and_empty_visual_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_test_app(Path(directory))

            with TestClient(app) as client:
                page = client.get("/diary")
                diary_css = client.get("/assets/diary.css")
                diary_js = client.get("/assets/diary.js")
                diary = client.get(
                    "/api/diary?date=2026-01-02&period=day"
                )
                progress = client.get("/api/visual/progress")

            self.assertEqual(page.status_code, 200)
            self.assertIn("然然生活日记", page.text)
            self.assertIn('id="highlight-list"', page.text)
            self.assertIn('id="review-list"', page.text)
            self.assertIn('id="search-index-progress"', page.text)
            self.assertEqual(diary_css.status_code, 200)
            self.assertIn(
                "@media (min-width: 1001px)",
                diary_css.text,
            )
            self.assertIn(
                "overflow-y: auto;",
                diary_css.text,
            )
            self.assertIn(
                "position: sticky;",
                diary_css.text,
            )
            self.assertEqual(diary_js.status_code, 200)
            self.assertIn(
                'event.target.closest("#clip-button")',
                diary_js.text,
            )
            self.assertEqual(diary.status_code, 200)
            self.assertEqual(diary.json()["events"], [])
            self.assertEqual(diary.json()["life_sessions"], [])
            self.assertEqual(diary.json()["review_queue"]["limit"], 3)
            self.assertEqual(
                diary.json()["analysis"]["analysis_file_percent"],
                0.0,
            )
            self.assertEqual(progress.status_code, 200)
            self.assertEqual(progress.json()["keyframes"], 0)
            self.assertIn("search_index", progress.json())
            self.assertEqual(
                progress.json()["search_index"]["indexed_keyframes"],
                0,
            )

    def test_dashboard_and_pipeline_endpoint_expose_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = build_test_app(root)

            with TestClient(app) as client:
                dashboard = client.get("/api/dashboard")
                pipeline = client.get("/api/pipeline")

            self.assertEqual(dashboard.status_code, 200)
            self.assertEqual(
                dashboard.json()["scope"]["indexed_files"],
                0,
            )
            self.assertTrue(dashboard.json()["pipeline"]["configured"])
            self.assertEqual(
                dashboard.json()["pipeline"]["latest_run"]["status"],
                "completed",
            )
            self.assertEqual(pipeline.status_code, 200)
            self.assertEqual(
                pipeline.json()["scope"]["camera_ids"],
                ["cam"],
            )
            self.assertEqual(
                pipeline.headers["cache-control"],
                "no-store",
            )

    def test_dashboard_defaults_to_five_item_smart_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_test_app(Path(directory), event_count=12)

            with TestClient(app) as client:
                smart = client.get(
                    "/api/dashboard?date=2026-01-02"
                )
                all_candidates = client.get(
                    "/api/dashboard"
                    "?date=2026-01-02&review_mode=all"
                )

            self.assertEqual(smart.status_code, 200)
            smart_payload = smart.json()
            self.assertEqual(len(smart_payload["episodes"]), 5)
            self.assertEqual(smart_payload["review_inbox"]["mode"], "smart")
            self.assertEqual(smart_payload["review_inbox"]["archived"], 7)
            self.assertEqual(
                smart_payload["review_inbox"]["lane_counts"],
                {"priority": 2, "learning": 2, "audit": 1},
            )
            self.assertFalse(
                smart_payload["shadow_calibration"][
                    "active_for_selection"
                ]
            )
            self.assertEqual(
                smart_payload["summary"]["total_episodes"],
                12,
            )
            self.assertEqual(all_candidates.status_code, 200)
            self.assertEqual(
                len(all_candidates.json()["episodes"]),
                12,
            )
            self.assertEqual(
                all_candidates.json()["review_inbox"]["archived"],
                0,
            )

    def test_lan_allowlist_rejects_clients_outside_home_subnet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_test_app(
                Path(directory),
                allowed_networks=(
                    "127.0.0.0/8",
                    "192.168.1.0/24",
                ),
            )

            with TestClient(
                app,
                client=("192.168.1.25", 50000),
            ) as allowed_client:
                allowed = allowed_client.get("/api/health")
            with TestClient(
                app,
                client=("10.20.30.40", 50000),
            ) as blocked_client:
                blocked = blocked_client.get("/api/health")

            self.assertEqual(allowed.status_code, 200)
            self.assertEqual(blocked.status_code, 403)
            self.assertEqual(
                blocked.json()["detail"],
                "此设备不在允许的家庭局域网内",
            )
            self.assertEqual(
                blocked.headers["cache-control"],
                "no-store",
            )


if __name__ == "__main__":
    unittest.main()
