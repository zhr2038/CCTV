import datetime as dt
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import backup


class FakeOpenList(BaseHTTPRequestHandler):
    files = {}
    directories = {"/"}
    requests = []
    puts = []
    return_task = False
    fail_put = False

    def reply(self, code=200, data=None):
        body = json.dumps({"code": code, "data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, data))
        path = data.get("path")
        if self.path == "/api/auth/login":
            self.reply(data={"token": "test-token"})
        elif self.path == "/api/fs/mkdir":
            self.directories.add(path)
            self.reply()
        elif self.path == "/api/fs/get":
            if path in self.directories:
                self.reply(data={"is_dir": True, "size": 0})
            elif path in self.files:
                self.reply(data={"is_dir": False, "size": len(self.files[path])})
            else:
                self.reply(500)
        else:
            self.reply(404)

    def do_PUT(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).puts.append(unquote(self.headers["File-Path"]))
        if self.fail_put:
            self.reply(500)
            return
        self.files[unquote(self.headers["File-Path"])] = data
        self.reply(data={"task": {"state": 0}} if self.return_task else None)

    def log_message(self, *args):
        pass


class BackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenList)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        backup.STOP.clear()
        FakeOpenList.files = {}
        FakeOpenList.directories = {"/"}
        FakeOpenList.requests = []
        FakeOpenList.puts = []
        FakeOpenList.return_task = False
        FakeOpenList.fail_put = False
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        secret = Path(self.temp.name) / "secret.json"
        secret.write_text(json.dumps({"username": "test", "password": "test"}))
        self.config = {"source": str(self.root), "state": str(Path(self.temp.name) / "state"),
                       "openlist_url": f"http://127.0.0.1:{self.server.server_port}",
                       "credentials": str(secret), "destination": "/", "retries": 1,
                       "min_age_seconds": 5400, "scan_pause_seconds": 0}

    def file(self, path, content=b"recording", age=6000):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.utime(path, (time.time() - age, time.time() - age))
        return path

    def make(self):
        obj = backup.Backup(self.config)
        self.addCleanup(obj.db.close)
        return obj

    def test_schedule_0330_china(self):
        before = dt.datetime(2026, 11, 3, 3, 29, tzinfo=backup.CHINA)
        self.assertFalse(backup.due_now(before, "03:30", "2026-11-02"))
        self.assertTrue(backup.due_now(before.replace(minute=30), "03:30", "2026-11-02"))
        self.assertFalse(backup.due_now(before.replace(hour=4), "03:30", "2026-11-03"))

    def test_device_number_is_not_a_persistent_fingerprint(self):
        info = SimpleNamespace(st_dev=98, st_ino=42, st_size=123, st_mtime_ns=567, st_ctime_ns=890)
        self.assertTrue(backup.same_file("95:42:123:567:890", info))
        self.assertFalse(backup.same_file("95:42:124:567:890", info))
        self.assertFalse(backup.same_file("95:43:123:567:890", info))

    def test_late_openlist_start_automatically_recovers(self):
        obj = self.make()
        obj.config["openlist_retry_seconds"] = 0
        original = obj.api.login
        with mock.patch.object(obj.api, "login", side_effect=[OSError("DNS not ready"), None]) as login:
            obj.wait_for_openlist()
        self.assertEqual(login.call_count, 2)
        self.assertIsNone(obj.status["last_error"])
        self.assertEqual(obj.api.request_timeout, 60)

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_reboot_preserves_uploaded_legacy_checkpoint(self):
        self.file("camera/a.mp4")
        obj = self.make()
        obj.run()
        row = obj.db.execute("SELECT signature FROM files").fetchone()
        legacy = "95:" + row[0].split(":", 1)[1]
        obj.db.execute("UPDATE files SET signature=?", (legacy,)); obj.db.commit()
        obj.run()
        self.assertEqual(obj.status["run_uploaded"], 0)
        self.assertEqual(obj.totals()["uploaded"]["files"], 1)

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_old_pending_device_number_uploads_without_deferral(self):
        self.file("camera/a.mp4")
        obj = self.make(); obj.scan()
        row = obj.db.execute("SELECT signature FROM files").fetchone()
        obj.db.execute("UPDATE files SET signature=?", ("95:"+row[0].split(":",1)[1],)); obj.db.commit()
        obj.run(resume=True)
        self.assertEqual(obj.status["run_uploaded"], 1)
        self.assertNotIn("recent", obj.totals())

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_missing_sources_do_not_block_remaining_uploads(self):
        paths=[self.file(f"camera/{n}.mp4") for n in range(6)]
        self.file("camera/survives.mp4", age=7000)
        obj=self.make(); obj.scan()
        for path in paths: path.unlink()
        obj.run(resume=True)
        self.assertEqual(obj.status["run_uploaded"], 1)
        self.assertEqual(obj.totals()["missing"]["files"], 6)
        self.assertEqual(obj.status["phase"], "completed_with_missing")
        self.assertEqual(obj.status["run_failures"], 0)

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_unavailable_source_mount_is_not_mislabeled_as_deleted_files(self):
        self.file("camera/a.mp4")
        obj=self.make(); obj.scan()
        obj.config["expected_source_dirs"]=["unavailable-directory"]
        with self.assertRaises(RuntimeError): obj.run(resume=True)
        self.assertNotIn("missing", obj.totals())

    def test_queue_pagination_handles_identical_mtimes_and_releases_wal(self):
        obj=self.make()
        obj.db.executemany("INSERT INTO files(path,signature,size,mtime_ns,seen,status) VALUES (?,?,1,100,'scan','pending')",
                           [(f"camera/{i:04}.mp4", "legacy") for i in range(250)])
        obj.db.commit()
        queue=obj.queued_files("scan"); first=next(queue)
        obj.set_setting("test", "write")
        self.assertEqual(obj.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0], 0)
        names=[first["path"]]+[row["path"] for row in queue]
        self.assertEqual(len(names),250)
        self.assertEqual(len(set(names)),250)

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_vault_cutover_resumes_with_checkpoints_and_capture_date_order(self):
        saved = "aqara_video/baby/20251217/100000.mp4"
        original = self.file(saved, b"already uploaded")
        touched = "xiaomi_camera_videos/kitchen/unchanged.mp4"
        touched_file = self.file(touched, b"untouched content")
        mtime_ns = original.stat().st_mtime_ns
        old = self.make()
        old.run()
        checkpoint = tuple(old.db.execute("SELECT sha256,uploaded_at FROM files WHERE path=?", (saved,)).fetchone())
        gone = "aqara_video/baby/20251217/110000.mp4"
        self.file(gone, b"live-only file")
        old.scan()
        old.db.close()

        vault = Path(self.temp.name) / "vault"
        birthday = "aqara_video/baby/20251217/090000.mp4"
        later = "aqara_video/baby/20251219/090000.mp4"
        before = "aqara_video/baby/20251216/090000.mp4"
        monitor = "xiaomi_camera_videos/B88880620430/00_20251218000000_20251218010000.mp4"
        for relative, content in ((saved, b"already uploaded"), (birthday, b"vault birthday"),
                                  (later, b"two days later"), (before, b"pre-birthday"), (monitor, b"vault monitor")):
            target = vault / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            os.utime(target, ns=(mtime_ns, mtime_ns))
        self.file(birthday, b"wrong live file")
        self.file(monitor, b"wrong live monitor")
        os.utime(touched_file, ns=(mtime_ns - 100, mtime_ns - 100))
        other = "xiaomi_camera_videos/kitchen/other.mp4"
        tesla = "TeslaCam/latest.mp4"
        self.file(other, age=7000)
        self.file(tesla, age=6000)
        self.config.update(vault_root=str(vault), vault_priority_start="2025-12-17")

        new = self.make()
        FakeOpenList.puts.clear()
        new.run(resume=True)
        self.assertEqual(FakeOpenList.puts, ["/" + name for name in (birthday, monitor, later, before, other, tesla)])
        self.assertEqual(FakeOpenList.files["/" + birthday], b"vault birthday")
        self.assertEqual(FakeOpenList.files["/" + monitor], b"vault monitor")
        self.assertEqual(tuple(new.db.execute("SELECT sha256,uploaded_at FROM files WHERE path=?", (saved,)).fetchone()),
                         checkpoint)
        self.assertEqual(new.db.execute("SELECT status FROM files WHERE path=?", (gone,)).fetchone()[0], "missing")
        self.assertEqual(new.db.execute("SELECT status FROM files WHERE path=?", (touched,)).fetchone()[0], "uploaded")
        self.assertEqual(new.status["phase"], "completed_with_missing")

    def test_recent_empty_and_all_top_level_directories(self):
        for name in ["aqara_video", "xiaomi_camera_videos", "TeslaCam", "NORMAL", "USB Copy", "XiaoBaiCam"]:
            self.file(name + "/video.mp4")
        self.file("fresh.mp4", age=10)
        self.file("empty.txt", content=b"")
        self.file("#recycle/ignored.mp4")
        obj = self.make()
        obj.scan()
        self.assertEqual(obj.totals()["pending"]["files"], 6)
        self.assertEqual(obj.totals()["recent"]["files"], 1)
        self.assertEqual(obj.totals()["empty"]["files"], 1)

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_upload_chinese_path_and_incremental_restart(self):
        self.file("宝宝/03 30.mp4", b"actual-content")
        obj = self.make()
        obj.run()
        self.assertEqual(FakeOpenList.files["/宝宝/03 30.mp4"], b"actual-content")
        row = obj.db.execute("SELECT * FROM files").fetchone()
        self.assertEqual(row["status"], "uploaded")
        self.assertEqual(len(row["sha256"]), 64)
        obj.db.close()
        again = self.make()
        again.run()
        self.assertEqual(again.status["run_uploaded"], 0)
        self.assertIsNotNone(again.setting("initial_complete"))

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_changed_source_is_uploaded_again_without_delete(self):
        self.file("a.mp4", b"first")
        obj = self.make()
        obj.run()
        self.file("a.mp4", b"new recording")
        obj.run()
        self.assertEqual(FakeOpenList.files["/a.mp4"], b"new recording")
        self.assertFalse(any("remove" in path for path, _ in FakeOpenList.requests))

    @unittest.skipIf(os.name == "nt", "Linux symlink semantics")
    def test_symlinks_and_path_traversal_are_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "secret").write_text("private")
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        obj = self.make()
        obj.scan()
        self.assertEqual(obj.totals(), {})
        for path in ["../outside/secret", "/outside/secret", "link/secret"]:
            with self.assertRaises((ValueError, OSError)):
                obj.safe_open(path)

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_failure_never_commits_a_success_checkpoint(self):
        self.file("a.mp4")
        FakeOpenList.fail_put = True
        obj = self.make()
        obj.run()
        self.assertEqual(obj.totals()["failed"]["files"], 1)
        self.assertIsNone(obj.setting("initial_complete"))
        FakeOpenList.fail_put = False
        obj.run(resume=True)
        self.assertEqual(obj.totals()["uploaded"]["files"], 1)

    def test_async_ack_is_not_completion(self):
        FakeOpenList.return_task = True
        obj = self.make()
        with self.assertRaises(RuntimeError):
            obj.api.upload(io.BytesIO(b"hello"), "/a", 5, time.time_ns())

    def test_destination_change_cannot_reuse_checkpoints(self):
        self.make()
        self.config["destination"] = "/different"
        with self.assertRaises(RuntimeError):
            self.make()

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_file_changed_after_scan_is_deferred(self):
        path = self.file("a.mp4")
        obj = self.make()
        obj.scan()
        row = obj.db.execute("SELECT * FROM files").fetchone()
        path.write_bytes(b"still recording")
        self.assertFalse(obj.upload_one(row))
        self.assertEqual(obj.totals()["recent"]["files"], 1)
        self.assertEqual(FakeOpenList.files, {})

    def test_interrupted_scan_requires_new_scan_before_resume(self):
        self.file("new.mp4")
        obj = self.make()
        obj.set_setting("last_scan", "old")
        obj.set_setting("run_stage", "scanning")
        obj.run(scan_only=True, resume=True)
        self.assertNotEqual(obj.setting("last_scan"), "old")
        self.assertEqual(obj.totals()["pending"]["files"], 1)

    def test_shutdown_cannot_publish_an_incomplete_inventory(self):
        self.file("new.mp4")
        obj = self.make()
        backup.STOP.set()
        try:
            with self.assertRaises(InterruptedError):
                obj.scan()
            self.assertIsNone(obj.setting("last_scan"))
            self.assertEqual(obj.setting("run_stage"), "scanning")
        finally:
            backup.STOP.clear()

    def test_large_queue_does_not_sort_in_container_tmpfs(self):
        obj = self.make()
        plan = list(obj.db.execute("EXPLAIN QUERY PLAN SELECT * FROM files INDEXED BY upload_queue WHERE status IN ('pending','failed','recent') AND seen=? ORDER BY mtime_ns DESC", ("scan",)))
        self.assertFalse(any("TEMP B-TREE" in str(tuple(row)) for row in plan))

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_many_checkpoints_and_empty_file_restore_metadata(self):
        for index in range(30):
            self.file(f"camera/{index}.mp4", str(index).encode())
        self.file("camera/empty.txt", b"")
        obj = self.make()
        obj.run()
        self.assertEqual(obj.totals()["uploaded"]["files"], 30)
        metadata = json.loads(FakeOpenList.files["/.cctv-backup-meta/empty-files.json"])
        self.assertEqual(metadata["files"][0]["path"], "camera/empty.txt")
        self.assertIsNotNone(obj.setting("initial_complete"))

    @unittest.skipIf(os.name == "nt", "Linux openat semantics")
    def test_initial_snapshot_waits_for_recent_recordings(self):
        self.file("fresh.mp4", age=10)
        obj = self.make()
        obj.run()
        self.assertIsNone(obj.setting("initial_complete"))
        obj.config["min_age_seconds"] = 0
        obj.run(resume=True)
        self.assertEqual(obj.totals()["uploaded"]["files"], 1)
        self.assertIsNotNone(obj.setting("initial_complete"))


if __name__ == "__main__":
    unittest.main()
