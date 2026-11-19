"""Local-only dashboard and cooperative task controls; no Docker socket."""
import collections
import copy
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import threading
import time


class Dashboard:
    def __init__(self, backup, stop):
        self.backup = backup
        self.stop = stop
        self.lock = threading.Lock()
        self.paused = threading.Event()
        self.retry_requested = threading.Event()
        self.control_path = backup.state / "control.json"
        if self.control_path.exists() and json.loads(self.control_path.read_text(encoding="utf-8")).get("paused"):
            self.paused.set()
        activity = backup.state / "activity.json"
        self.events = collections.deque(json.loads(activity.read_text(encoding="utf-8")) if activity.exists() else [], maxlen=50)
        self.history = collections.deque(maxlen=61)
        self.last_sample = 0
        self.event("备份服务已连接，进度记录已恢复")

    def event(self, message):
        with self.lock:
            timestamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")
            self.events.appendleft({"time": timestamp, "message": message})
            self._save(self.backup.state / "activity.json", list(self.events))

    @staticmethod
    def _save(path, data):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def control(self, action):
        if action not in ("pause", "resume", "retry"):
            raise ValueError("Unsupported action")
        with self.lock:
            if action == "pause":
                self.paused.set()
            else:
                self.paused.clear()
            if action == "retry":
                self.retry_requested.set()
            self._save(self.control_path, {"paused": self.paused.is_set()})
        self.backup.update(pause_requested=self.paused.is_set())
        self.event({"pause": "已请求暂停，将在当前文件完成后暂停", "resume": "已请求继续上传", "retry": "已请求重试未完成任务"}[action])

    def wait_if_paused(self):
        if not self.paused.is_set():
            return
        previous = self.backup.status.get("phase", "starting")
        self.backup.update(phase="paused", pause_requested=True)
        while self.paused.is_set() and not self.stop.wait(0.5):
            pass
        if not self.stop.is_set():
            self.backup.update(phase=previous, pause_requested=False)

    def snapshot(self):
        with self.backup.lock:
            result = copy.deepcopy(self.backup.status)
        uploaded = result.get("totals", {}).get("uploaded", {})
        completed = result.get("uploaded_bytes_total", uploaded.get("bytes", 0))
        with self.lock:
            current = time.time()
            if current - self.last_sample >= 5:
                self.history.append({"time": current, "bytes": completed})
                self.last_sample = current
            while len(self.history) > 1 and self.history[0]["time"] < current - 300:
                self.history.popleft()
            points = list(self.history)
            if len(points) >= 2:
                seconds = points[-1]["time"] - points[0]["time"]
                result["speed_bytes"] = max(0, (points[-1]["bytes"] - points[0]["bytes"]) / max(1, seconds))
            else:
                result["speed_bytes"] = 0
            result["history"] = points
            result["events"] = list(self.events)[:15]
            result["pause_requested"] = self.paused.is_set()
        return result

    def start(self):
        dashboard = self
        assets = Path(__file__).resolve().parent

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(15)

            def reply(self, data, content_type="application/json; charset=utf-8", code=200):
                body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/api/status", "/status.json", "/health"):
                    self.reply(dashboard.snapshot())
                elif self.path in ("/", "/index.html", "/app.js"):
                    name = "app.js" if self.path == "/app.js" else "index.html"
                    self.reply((assets / name).read_bytes(), "text/javascript; charset=utf-8" if name.endswith("js") else "text/html; charset=utf-8")
                else:
                    self.reply({"error": "not found"}, code=404)

            def _is_allowed_host(self, host_only: str) -> bool:
                configured = dashboard.backup.config.get("dashboard_hosts")
                if configured:
                    return host_only in configured
                if host_only in ("127.0.0.1", "localhost", "::1"):
                    return True
                try:
                    return ipaddress.ip_address(host_only).is_private
                except ValueError:
                    return False

            def do_POST(self):
                host = self.headers.get("Host", "")
                host_only = host.split(":")[0]
                if (not self._is_allowed_host(host_only) or self.headers.get("Origin") != "http://" + host
                        or self.headers.get("X-CCTV-Control") != "1"
                        or self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                    self.reply({"error": "forbidden origin"}, code=403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if self.path != "/api/control" or not 0 < length <= 2048:
                        raise ValueError()
                    action = json.loads(self.rfile.read(length)).get("action")
                    dashboard.control(action)
                except (ValueError, TypeError, AttributeError):
                    self.reply({"error": "invalid action"}, code=400)
                    return
                self.reply({"accepted": True, "action": action})

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("0.0.0.0", self.backup.config.get("status_port", 8791)), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.server = server
        return server
