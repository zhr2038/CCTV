import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from dashboard import Dashboard


class FakeBackup:
    def __init__(self, path):
        self.state = Path(path)
        self.lock = threading.Lock()
        self.status = {"phase": "uploading", "totals": {"uploaded": {"bytes": 100, "files": 1}}}
        self.config = {"status_port": 0}

    def update(self, **values):
        with self.lock:
            self.status.update(values)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backup = FakeBackup(self.temp.name)
        self.stop = threading.Event()
        self.dashboard = Dashboard(self.backup, self.stop)
        self.server = self.dashboard.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, path, data=None, origin=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Content-Type": "application/json", "X-CCTV-Control": "1"}
        if origin:
            headers["Origin"] = f"http://127.0.0.1:{self.server.server_port}"
        conn.request("GET" if data is None else "POST", path, None if data is None else json.dumps(data), headers)
        response = conn.getresponse()
        body = response.read()
        conn.close()
        return response.status, body

    def test_page_and_live_api(self):
        code, body = self.request("/")
        self.assertEqual(code, 200)
        self.assertIn("首次全量备份", body.decode())
        code, body = self.request("/api/status")
        self.assertEqual(json.loads(body)["phase"], "uploading")

    def test_cross_origin_control_is_rejected(self):
        code, _ = self.request("/api/control", {"action": "pause"}, origin=False)
        self.assertEqual(code, 403)
        self.assertFalse(self.dashboard.paused.is_set())

    def test_pause_persists_and_resume_releases_waiter(self):
        code, _ = self.request("/api/control", {"action": "pause"})
        self.assertEqual(code, 200)
        self.assertTrue(Dashboard(self.backup, self.stop).paused.is_set())
        waiter = threading.Thread(target=self.dashboard.wait_if_paused)
        waiter.start()
        time.sleep(.05)
        self.assertEqual(self.backup.status["phase"], "paused")
        self.request("/api/control", {"action": "resume"})
        waiter.join(timeout=2)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(self.backup.status["phase"], "uploading")

    def test_retry_only_signals_worker_and_invalid_actions_are_rejected(self):
        self.request("/api/control", {"action": "retry"})
        self.assertTrue(self.dashboard.retry_requested.is_set())
        code, _ = self.request("/api/control", {"action": "delete"})
        self.assertEqual(code, 400)


if __name__ == "__main__":
    unittest.main()
