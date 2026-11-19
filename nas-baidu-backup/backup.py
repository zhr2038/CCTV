"""Daily, resumable, upload-only CCTV backup through an existing OpenList.

Python standard library only. Credentials live exclusively in a NAS secret file.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import http.client
import io
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sqlite3
import stat
import threading
import time
from urllib.parse import quote, urlsplit

CHINA = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")
STOP = threading.Event()
LOG = logging.getLogger("cctv-backup")
SYSTEM_NAMES = {"@eaDir", "#recycle", "@Recycle", ".Trash", ".recycle", "$RECYCLE.BIN", "System Volume Information"}
VAULT_PREFIXES = (("aqara_video",), ("xiaomi_camera_videos", "B88880620430"))
BAIDU_VAULT_INVENTORY_VERSION = 1


def now_iso():
    return dt.datetime.now(CHINA).isoformat(timespec="seconds")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class APIError(RuntimeError):
    def __init__(self, code, operation):
        # Do not log upstream messages: they can contain signed URLs/tokens.
        self.code = code
        super().__init__(f"OpenList {operation}: code={code}; see OpenList logs")


class OpenList:
    def __init__(self, url, credentials, timeout=7200):
        self.url = urlsplit(url)
        if self.url.scheme not in ("http", "https") or self.url.path not in ("", "/"):
            raise ValueError("OpenList URL must be an HTTP(S) origin")
        self.credentials = credentials
        self.timeout = timeout
        self.request_timeout = 60
        self.token = ""
        self.directories = {"/"}

    def connection(self, timeout):
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=timeout)

    @staticmethod
    def decode(response, operation):
        data = response.read(2 * 1024 * 1024)
        if response.status != 200:
            raise APIError(response.status, operation)
        result = json.loads(data)
        if result.get("code") != 200:
            raise APIError(result.get("code"), operation)
        return result.get("data")

    def login(self):
        credentials = json.loads(Path(self.credentials).read_text(encoding="utf-8"))
        self.token = self.request("/api/auth/login", credentials, authenticate=False)["token"]

    def request(self, path, data=None, authenticate=True):
        if authenticate and not self.token:
            self.login()
        payload = json.dumps(data or {}).encode()
        for attempt in range(2):
            with contextlib.closing(self.connection(self.request_timeout)) as conn:
                headers = {"Content-Type": "application/json"}
                if authenticate:
                    headers["Authorization"] = self.token
                conn.request("POST", path, payload, headers)
                try:
                    return self.decode(conn.getresponse(), path)
                except APIError as error:
                    if error.code in (401, 403) and authenticate and attempt == 0:
                        self.login()
                    else:
                        raise

    def get(self, path):
        return self.request("/api/fs/get", {"path": path, "password": ""})

    def mkdir(self, path):
        path = str(PurePosixPath(path))
        if path in self.directories:
            return
        self.mkdir(str(PurePosixPath(path).parent))
        try:
            item = self.get(path)
        except APIError as error:
            if error.code not in (404, 500):
                raise
            self.request("/api/fs/mkdir", {"path": path})
            item = self.get(path)
        if not item.get("is_dir"):
            raise RuntimeError("Destination directory is occupied by a file")
        self.directories.add(path)

    def upload(self, stream, destination, size, mtime_ns, bandwidth_mib=0):
        if not self.token:
            self.login()
        digest = hashlib.sha256()
        sent = 0
        started = time.monotonic()
        with contextlib.closing(self.connection(self.timeout)) as conn:
            conn.putrequest("PUT", "/api/fs/put")
            for key, value in {
                "Authorization": self.token,
                "File-Path": quote(destination, safe="/"),
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
                "As-Task": "false",
                "Overwrite": "true",
                "Last-Modified": str(mtime_ns // 1_000_000),
            }.items():
                conn.putheader(key, value)
            conn.endheaders()
            while sent < size:
                if STOP.is_set():
                    raise InterruptedError("Shutdown requested")
                block = stream.read(min(1024 * 1024, size - sent))
                if not block:
                    raise RuntimeError("Source truncated during upload")
                conn.send(block)
                digest.update(block)
                sent += len(block)
                if bandwidth_mib:
                    delay = sent / (bandwidth_mib * 1024 * 1024) - (time.monotonic() - started)
                    if delay > 0 and STOP.wait(delay):
                        raise InterruptedError("Shutdown requested")
            result = self.decode(conn.getresponse(), "upload")
            if result and result.get("task"):
                raise RuntimeError("Asynchronous upload cannot be treated as complete")
        return digest.hexdigest()


def fingerprint(info):
    # st_dev is a boot-local mount number, not a persistent file identity.
    return f"v2:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}"


def same_file(saved, info):
    # Accept existing five-field checkpoints without rewriting the manifest.
    fields = saved.split(":")
    return len(fields) == 5 and fields[1:] == fingerprint(info).split(":")[1:]


def source_files(root, errors, excluded, *, start=(), skip_dirs=()):
    """Never follow directory/file symlinks; bound memory to one directory."""
    def walk(directory):
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if STOP.is_set():
                        return
                    if entry.name in excluded:
                        continue
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if Path(entry.path).relative_to(root).parts in skip_dirs:
                                continue
                            yield from walk(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path), entry.stat(follow_symlinks=False)
                    except OSError:
                        errors.append(str(Path(entry.path).relative_to(root)))
        except OSError:
            errors.append(str(Path(directory).relative_to(root)))
    yield from walk(root.joinpath(*start))


class Backup:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["source"])
        self.vault = Path(config["vault_root"]) if config.get("vault_root") else None
        self.priority_start = (int(dt.date.fromisoformat(config["vault_priority_start"]).strftime("%Y%m%d"))
                               if self.vault else None)
        self.inventory_version = (json.dumps([BAIDU_VAULT_INVENTORY_VERSION, str(self.vault), self.priority_start])
                                  if self.vault else None)
        self.state = Path(config["state"])
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "tmp").mkdir(exist_ok=True)
        self.status_path = self.state / "status.json"
        self.lock = threading.Lock()
        try:
            self.status = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.status = {}
        self.status["phase"] = "starting"
        self.dashboard = None
        self.api = OpenList(config["openlist_url"], config["credentials"], config.get("upload_timeout", 7200))
        self.db = sqlite3.connect(self.state / "manifest.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY, signature TEXT NOT NULL, size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL, seen TEXT NOT NULL, status TEXT NOT NULL,
                sha256 TEXT, uploaded_at TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS pending ON files(status, seen, mtime_ns DESC);
            CREATE INDEX IF NOT EXISTS upload_queue ON files(seen, mtime_ns DESC)
                WHERE status IN ('pending','failed','recent');
            CREATE INDEX IF NOT EXISTS upload_queue_v2 ON files(seen, mtime_ns DESC, path DESC)
                WHERE status IN ('pending','failed','recent');
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        identity = json.dumps({k:config[k] for k in ("source", "openlist_url", "destination", "credentials")}, sort_keys=True)
        existing = self.db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
        if existing and existing[0] != identity:
            self.db.close()
            raise RuntimeError("Backup destination changed; use a separate state directory")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES ('identity', ?)", (identity,))
        columns = {column["name"] for column in self.db.execute("PRAGMA table_info(files)")}
        if "priority" not in columns:
            self.db.execute("ALTER TABLE files ADD COLUMN priority INTEGER NOT NULL DEFAULT 1")
        if "capture_order" not in columns:
            self.db.execute("ALTER TABLE files ADD COLUMN capture_order INTEGER NOT NULL DEFAULT 0")
        self.db.execute("""CREATE INDEX IF NOT EXISTS upload_queue_v3
                         ON files(seen, priority DESC, capture_order DESC, mtime_ns DESC, path DESC)
                         WHERE status IN ('pending','failed','recent')""")
        self.db.commit()

    def update(self, **values):
        with self.lock:
            self.status.update(values)
            self.status["heartbeat"] = now_iso()
            atomic_json(self.status_path, self.status)

    def heartbeat(self):
        while not STOP.wait(30):
            self.update()
            if self.dashboard:
                self.dashboard.snapshot()

    def check_pause(self):
        if self.dashboard:
            self.dashboard.wait_if_paused()

    def event(self, message):
        if self.dashboard:
            self.dashboard.event(message)

    def setting(self, key, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))
        self.db.commit()

    def totals(self):
        return {row["status"]: {"files": row["n"], "bytes": row["b"]} for row in self.db.execute(
            "SELECT status, count(*) n, coalesce(sum(size),0) b FROM files GROUP BY status")}

    def directories(self):
        rows = self.db.execute("""SELECT CASE WHEN instr(path,'/')>0 THEN substr(path,1,instr(path,'/')-1) ELSE '(根目录)' END name,
            count(*) files, sum(size) bytes, sum(status='uploaded') uploaded_files,
            sum(CASE WHEN status='uploaded' THEN size ELSE 0 END) uploaded_bytes,
            sum(status='failed') failed, sum(status='missing') missing FROM files GROUP BY name ORDER BY bytes DESC""")
        return [dict(r) for r in rows]

    def validate_source(self):
        if not self.root.is_dir() or self.root.is_symlink():
            raise RuntimeError("Source directory is not mounted correctly")
        if self.config.get("require_source_mount"):
            if not os.path.ismount(self.root) or not os.statvfs(self.root).f_flag & os.ST_RDONLY:
                raise RuntimeError("Source must be a read-only mount")
        for name in self.config.get("expected_source_dirs", []):
            if not (self.root / name).is_dir():
                raise RuntimeError("Expected source directory is unavailable")
        if self.vault:
            if not self.vault.is_dir() or self.vault.is_symlink():
                raise RuntimeError("Vault directory is unavailable")
            if self.config.get("require_source_mount"):
                if not os.path.ismount(self.vault) or not os.statvfs(self.vault).f_flag & os.ST_RDONLY:
                    raise RuntimeError("Vault must be a read-only mount")
            for prefix in VAULT_PREFIXES:
                for depth in range(1, len(prefix) + 1):
                    directory = self.vault.joinpath(*prefix[:depth])
                    if not directory.is_dir() or directory.is_symlink():
                        raise RuntimeError("Expected vault recording directory is unavailable")

    def uses_vault(self, parts):
        return self.vault is not None and any(parts[:len(prefix)] == prefix for prefix in VAULT_PREFIXES)

    def inventory(self, errors):
        excluded = set(self.config.get("exclude_names", SYSTEM_NAMES))
        if self.vault:
            for prefix in VAULT_PREFIXES:
                for path, info in source_files(self.vault, errors, excluded, start=prefix):
                    yield path.relative_to(self.vault), info, True
        for path, info in source_files(self.root, errors, excluded,
                                       skip_dirs=VAULT_PREFIXES if self.vault else ()):
            yield path.relative_to(self.root), info, False

    def queue_order(self, relative, in_vault):
        if in_vault:
            if relative.parts[0] == "aqara_video":
                match = (re.fullmatch(r"20[0-9]{6}", relative.parts[2])
                         if len(relative.parts) >= 4 else None)
                day = match.group() if match else None
            else:
                match = re.match(r"^\d{2}_(20[0-9]{6})\d{6}_", relative.name)
                day = match.group(1) if match else None
            if day and int(day) >= self.priority_start:
                # Negative date with the descending key puts the earliest day first.
                return 3, -int(day)
            return 2, 0
        return (0, 0) if relative.parts[0] == "TeslaCam" else (1, 0)

    def wait_for_openlist(self):
        attempts = 0
        normal_timeout = self.api.request_timeout
        try:
            self.api.request_timeout = 10
            while not STOP.is_set():
                self.check_pause()
                if STOP.is_set():
                    break
                try:
                    self.api.login()
                    self.api.mkdir(self.config["destination"])
                    self.update(last_error=None, retry_at=None)
                    return
                except (OSError, APIError) as error:
                    if isinstance(error, APIError) and error.code not in (429, 500, 502, 503, 504):
                        raise
                    attempts += 1
                    self.api.token = ""
                    self.update(phase="waiting_openlist", current_file=None,
                                last_error="正在等待 OpenList 启动或网络恢复，每 15 秒重试")
                    if attempts == 1:
                        self.event("等待 OpenList 启动，将自动继续上传")
                    if STOP.wait(self.config.get("openlist_retry_seconds", 15)):
                        break
        finally:
            self.api.request_timeout = normal_timeout
        raise InterruptedError("Shutdown while waiting for OpenList")

    def queued_files(self, scan_id):
        # Never keep a read cursor alive across commits: it pins the WAL for
        # the entire multi-day backup. Bounded keyset pages release it first.
        last = None
        while not STOP.is_set():
            where = "" if last is None else " AND (priority,capture_order,mtime_ns,path) < (?,?,?,?)"
            values = [scan_id] + ([] if last is None else list(last))
            cursor = self.db.execute("SELECT * FROM files INDEXED BY upload_queue_v3 "
                "WHERE status IN ('pending','failed','recent') AND seen=?" + where +
                " ORDER BY priority DESC,capture_order DESC,mtime_ns DESC,path DESC LIMIT 100", values)
            rows = cursor.fetchall()
            cursor.close()
            if not rows:
                return
            last = tuple(rows[-1][key] for key in ("priority", "capture_order", "mtime_ns", "path"))
            yield from rows

    def prepare_queue(self):
        self.update(phase="preparing_queue", current_file=None)
        cutoff = int((time.time() - self.config.get("min_age_seconds", 5400)) * 1_000_000_000)
        self.db.execute("UPDATE files SET status='pending' WHERE status='recent' AND mtime_ns<=?", (cutoff,))
        cursor = self.db.execute("SELECT path FROM files WHERE status='failed' AND error='Upload failed (FileNotFoundError)'")
        rows = cursor.fetchall(); cursor.close()
        for row in rows:
            try:
                with self.safe_open(row["path"]):
                    pass
            except FileNotFoundError:
                self.validate_source()
                self.db.execute("UPDATE files SET status='missing',error='Source no longer exists' WHERE path=?", (row["path"],))
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchall()

    def matches_uploaded_bytes(self, relative, signature, size, mtime_ns, in_vault):
        previous = self.db.execute("""
            SELECT sha256 FROM files
            WHERE path=? AND status='uploaded' AND size=? AND sha256 IS NOT NULL
              AND substr(signature,instr(signature,':')+1)<>?
              AND NOT (? AND mtime_ns=?)
        """, (relative, size, signature.split(":", 1)[1], in_vault, mtime_ns)).fetchone()
        if previous is None:
            return False
        with self.safe_open(relative) as stream:
            if not same_file(signature, os.fstat(stream.fileno())):
                return False
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            return digest == previous["sha256"] and same_file(signature, os.fstat(stream.fileno()))

    def scan(self):
        self.validate_source()
        self.set_setting("run_stage", "scanning")
        scan_id = dt.datetime.now(CHINA).isoformat(timespec="microseconds")
        LOG.info("Inventory scan started")
        self.update(phase="scanning", scan_started=scan_id, scanned_files=0, scan_errors=0)
        self.event("开始扫描首次备份清单")
        errors = []
        count = total = 0
        cutoff = time.time() - self.config.get("min_age_seconds", 5400)
        # Vault clones have new inode/ctime but preserve the recording path, size and mtime.
        # Keep the original upload digest and completion time when that identity matches.
        same_recording = """substr(files.signature,instr(files.signature,':')+1)
                            = substr(excluded.signature,instr(excluded.signature,':')+1)
                            OR (:in_vault AND files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns)"""
        keep_uploaded = f"files.status='uploaded' AND ({same_recording} OR :same_bytes)"
        insert_file = f"""
            INSERT INTO files(path,signature,size,mtime_ns,seen,status,priority,capture_order)
            VALUES (:path,:signature,:size,:mtime,:scan,:status,:priority,:capture_order)
            ON CONFLICT(path) DO UPDATE SET
                status=CASE WHEN {keep_uploaded} THEN 'uploaded' ELSE excluded.status END,
                sha256=CASE WHEN {keep_uploaded} THEN files.sha256 ELSE NULL END,
                uploaded_at=CASE WHEN {keep_uploaded} THEN files.uploaded_at ELSE NULL END,
                error=NULL, signature=excluded.signature,size=excluded.size,
                mtime_ns=excluded.mtime_ns,seen=excluded.seen,
                priority=excluded.priority,capture_order=excluded.capture_order
        """
        for path, info, in_vault in self.inventory(errors):
            relative = path.as_posix()
            signature = fingerprint(info)
            status = "empty" if info.st_size == 0 else ("recent" if info.st_mtime > cutoff else "pending")
            priority, capture_order = self.queue_order(path, in_vault)
            same_bytes = status != "empty" and self.matches_uploaded_bytes(
                relative, signature, info.st_size, info.st_mtime_ns, in_vault)
            self.db.execute(insert_file, {"path": relative, "signature": signature, "size": info.st_size,
                            "mtime": info.st_mtime_ns, "scan": scan_id, "status": status, "in_vault": in_vault,
                            "same_bytes": same_bytes, "priority": priority, "capture_order": capture_order})
            count += 1
            total += info.st_size
            if count % 1000 == 0:
                self.db.commit()
                self.update(scanned_files=count, scanned_bytes=total, scan_errors=len(errors))
                self.check_pause()
                STOP.wait(self.config.get("scan_pause_seconds", 0.05))
        self.db.commit()
        if STOP.is_set():
            self.update(phase="interrupted", scanned_files=count)
            raise InterruptedError("Scan interrupted; a complete rescan is required")
        self.validate_source()
        if self.vault and not errors:
            self.db.execute("""UPDATE files SET status='missing',error='Source no longer exists'
                             WHERE seen<>? AND status IN ('pending','failed','recent')""", (scan_id,))
        atomic_json(self.state / "scan-errors.json", errors)
        # Publish the completed inventory together so a restart cannot resume an old queue.
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO settings VALUES (?, ?)",
                                [("last_scan", scan_id), ("run_stage", "uploading"),
                                 ("inventory_version", self.inventory_version or "")])
        self.update(scanned_files=count, scanned_bytes=total, scan_errors=len(errors), totals=self.totals())
        LOG.info("Inventory scan finished: files=%s bytes=%s errors=%s", count, total, len(errors))
        self.event(f"清单扫描完成：{count:,} 个文件")
        return scan_id

    def safe_open(self, relative):
        # openat with O_NOFOLLOW on every component prevents even a symlink swap.
        parts = PurePosixPath(relative).parts
        if not parts or relative.startswith("/") or ".." in parts:
            raise ValueError("Invalid source path")
        root = self.vault if self.uses_vault(parts) else self.root
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            os.close(directory)
        stream = os.fdopen(descriptor, "rb")
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            stream.close()
            raise ValueError("Source is not a regular file")
        return stream

    def upload_one(self, row):
        relative = row["path"]
        destination = str(PurePosixPath(self.config["destination"]) / relative)
        try:
            stream = self.safe_open(relative)
        except FileNotFoundError:
            # A rotated/deleted recording cannot be backed up. Keep its record,
            # but do not let it stop every subsequent file in the queue.
            self.validate_source()
            self.db.execute("UPDATE files SET status='missing',error='Source no longer exists' WHERE path=?", (relative,))
            self.db.commit()
            return False
        with stream:
            before = os.fstat(stream.fileno())
            if not same_file(row["signature"], before) or before.st_mtime > time.time() - self.config.get("min_age_seconds", 5400):
                self.db.execute("UPDATE files SET status='recent',signature=?,size=?,mtime_ns=?,sha256=NULL WHERE path=?",
                                (fingerprint(before), before.st_size, before.st_mtime_ns, relative))
                self.db.commit()
                return False
            self.api.mkdir(str(PurePosixPath(destination).parent))
            digest = self.api.upload(stream, destination, row["size"], row["mtime_ns"], self.config.get("bandwidth_mib", 0))
            if not same_file(row["signature"], os.fstat(stream.fileno())):
                raise RuntimeError("Source changed during upload; will retry next run")
        # Verify the provider-visible file size before committing the checkpoint.
        item = self.api.get(destination)
        if item.get("is_dir") or item.get("size") != row["size"]:
            raise RuntimeError("Remote file size verification failed")
        self.db.execute("UPDATE files SET status='uploaded',sha256=?,uploaded_at=?,error=NULL WHERE path=?", (digest, now_iso(), relative))
        self.db.commit()
        return True

    def run(self, limit=None, scan_only=False, resume=False):
        self.set_setting("run_active", "1")
        scan_id = self.setting("last_scan") if resume and self.setting("run_stage") == "uploading" else None
        if self.vault and self.setting("inventory_version") != self.inventory_version:
            scan_id = None
        if not scan_id:
            scan_id = self.scan()
        if scan_only:
            self.update(phase="scan_complete", totals=self.totals())
            self.set_setting("run_active", "0")
            return
        self.validate_source()
        self.wait_for_openlist()
        self.prepare_queue()
        totals = self.totals()
        baseline = totals.get("uploaded", {"files": 0, "bytes": 0})
        directories = self.directories()
        self.update(phase="uploading", run_started=now_iso(), run_uploaded=0, run_bytes=0, run_failures=0, last_error=None,
                    totals=totals, directories=directories, uploaded_files_total=baseline["files"], uploaded_bytes_total=baseline["bytes"],
                    scanned_files=sum(x["files"] for x in totals.values()), scanned_bytes=sum(x["bytes"] for x in totals.values()))
        self.event("已连接百度网盘，继续上传未完成的文件")
        uploaded = size = failures = consecutive = 0
        last_totals = time.monotonic()
        # The inventory owns the upload order for both initial backup and later runs.
        rows = self.queued_files(scan_id)
        processed = 0
        for row in rows:
            self.check_pause()
            if STOP.is_set() or (limit is not None and uploaded >= limit):
                break
            self.update(current_file=row["path"], current_file_bytes=row["size"])
            for attempt in range(self.config.get("retries", 3)):
                try:
                    if self.upload_one(row):
                        uploaded += 1
                        size += row["size"]
                        name = row["path"].split("/", 1)[0] if "/" in row["path"] else "(根目录)"
                        for directory in directories:
                            if directory["name"] == name:
                                directory["uploaded_files"] += 1
                                directory["uploaded_bytes"] += row["size"]
                                if row["status"] == "failed":
                                    directory["failed"] = max(0, directory["failed"] - 1)
                    else:
                        state = self.db.execute("SELECT status FROM files WHERE path=?", (row["path"],)).fetchone()[0]
                        if state == "missing":
                            name = row["path"].split("/", 1)[0] if "/" in row["path"] else "(根目录)"
                            for directory in directories:
                                if directory["name"] == name:
                                    directory["missing"] += 1
                                    if row["status"] == "failed":
                                        directory["failed"] = max(0, directory["failed"] - 1)
                    consecutive = 0
                    break
                except InterruptedError:
                    raise
                except Exception as error:
                    # Keep secrets out of persistent logs and status pages.
                    code = error.code if isinstance(error, APIError) else type(error).__name__
                    message = f"Upload failed ({code})"
                    if attempt + 1 < self.config.get("retries", 3):
                        LOG.warning("Retry file=%r attempt=%s reason=%s", row["path"], attempt + 1, code)
                        self.api.token = ""
                        if STOP.wait(min(60, 5 * 2**attempt)):
                            raise InterruptedError("Shutdown requested")
                        continue
                    self.db.execute("UPDATE files SET status='failed',error=? WHERE path=?", (message, row["path"]))
                    self.db.commit()
                    failures += 1
                    consecutive += 1
                    LOG.error("%s file=%r", message, row["path"])
                    self.update(last_error=message)
                    self.event(f"文件上传失败，已保留重试：{row['path']}（{code}）")
                    name = row["path"].split("/", 1)[0] if "/" in row["path"] else "(根目录)"
                    for directory in directories:
                        if directory["name"] == name and row["status"] != "failed":
                            directory["failed"] += 1
            self.update(run_uploaded=uploaded, run_bytes=size, run_failures=failures, uploaded_files_total=baseline["files"]+uploaded,
                        uploaded_bytes_total=baseline["bytes"]+size, directories=[dict(x) for x in directories],
                        missing_files=sum(x["missing"] for x in directories))
            processed += 1
            if processed % 100 == 0:
                self.db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchall()
            if self.dashboard and self.dashboard.retry_requested.is_set():
                break
            if time.monotonic() - last_totals > 300:
                self.update(totals=self.totals())
                LOG.info("Upload progress: files=%s bytes=%s failures=%s", uploaded, size, failures)
                last_totals = time.monotonic()
            if consecutive >= self.config.get("max_consecutive_failures", 5):
                raise RuntimeError("Consecutive upload failures; remaining files stay queued")
        # Baidu rejects zero-byte files. Preserve their paths as restore metadata.
        empty = [dict(r) for r in self.db.execute("SELECT path,mtime_ns FROM files WHERE status='empty' AND seen=?", (scan_id,))]
        if empty and not STOP.is_set():
            metadata_path = str(PurePosixPath(self.config["destination"]) / ".cctv-backup-meta" / "empty-files.json")
            self.api.mkdir(str(PurePosixPath(metadata_path).parent))
            content = json.dumps({"format": 1, "note": "Recreate these paths as zero-byte files when restoring", "files": empty}, ensure_ascii=False).encode()
            self.api.upload(io.BytesIO(content), metadata_path, len(content), time.time_ns())
            if self.api.get(metadata_path).get("size") != len(content):
                raise RuntimeError("Empty-file restore metadata verification failed")
        remaining = self.db.execute("SELECT count(*) FROM files WHERE seen=? AND status IN ('pending','failed','recent')", (scan_id,)).fetchone()[0]
        scan_errors = len(json.loads((self.state / "scan-errors.json").read_text(encoding="utf-8")))
        complete = not STOP.is_set() and not remaining and not scan_errors
        totals = self.totals()
        missing = totals.get("missing", {}).get("files", 0)
        phase = "interrupted" if STOP.is_set() else (("completed_with_missing" if missing else "completed") if complete else "incomplete")
        self.update(phase=phase, run_finished=now_iso(), current_file=None, totals=totals, scan_errors=scan_errors, missing_files=missing)
        if complete:
            self.set_setting("initial_complete", now_iso())
            self.set_setting("missing_files", str(missing))
            self.event(f"可读取文件已上传完成；{missing} 个源文件已不存在" if missing else "首次清单已全部上传完成；每日增量仍未启用")
        elif scan_errors:
            self.set_setting("run_stage", "scanning")
        self.set_setting("run_active", "1" if STOP.is_set() else "0")
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchall()
        LOG.info("Run finished: uploaded=%s bytes=%s failures=%s", uploaded, size, failures)


def due_now(now, daily_time, last_date):
    hour, minute = map(int, daily_time.split(":"))
    return now.strftime("%Y-%m-%d") != last_date and (now.hour, now.minute) >= (hour, minute)


def next_run(daily_time, last_date):
    now = dt.datetime.now(CHINA)
    hour, minute = map(int, daily_time.split(":"))
    result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if result <= now or last_date == now.strftime("%Y-%m-%d"):
        result += dt.timedelta(days=1)
    return result.isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["daemon", "once", "scan", "status", "health"])
    parser.add_argument("--config", default=os.environ.get("BACKUP_CONFIG", "/app/config.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    status_path = Path(config["state"]) / "status.json"
    if args.command in ("status", "health"):
        result = json.loads(status_path.read_text(encoding="utf-8"))
        if args.command == "health":
            age = (dt.datetime.now(CHINA) - dt.datetime.fromisoformat(result["heartbeat"])).total_seconds()
            raise SystemExit(0 if age < 180 else 1)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    import fcntl
    Path(config["state"]).mkdir(parents=True, exist_ok=True)
    lock = open(Path(config["state"]) / "backup.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A backup process already owns this state directory")
    os.nice(15)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    backup = Backup(config)
    backup.update(daily_time=config["daily_time"], daily_enabled=config.get("daily_enabled", False),
                  timezone="Asia/Shanghai", next_run=None, initial_complete=backup.setting("initial_complete"),
                  phase=("completed_with_missing" if int(backup.setting("missing_files", "0")) else "completed") if backup.setting("initial_complete") else "starting")
    threading.Thread(target=backup.heartbeat, daemon=True).start()
    if args.command != "daemon":
        backup.run(limit=args.limit, scan_only=args.command == "scan")
        return
    from dashboard import Dashboard
    backup.dashboard = Dashboard(backup, STOP)
    backup.dashboard.start()
    if not backup.setting("initial_complete"):
        backup.set_setting("retry_after", "0")
    while not STOP.is_set():
        backup.check_pause()
        if backup.dashboard.retry_requested.is_set():
            backup.dashboard.retry_requested.clear()
            backup.set_setting("retry_after", "0")
        last = backup.setting("last_run_date", "")
        active = backup.setting("run_active") == "1"
        current = dt.datetime.now(CHINA)
        initial_incomplete = not backup.setting("initial_complete")
        retry_due = time.time() >= float(backup.setting("retry_after", "0"))
        initial_due = config.get("run_on_first_start", True) and initial_incomplete and retry_due
        daily_due = config.get("daily_enabled", False) and due_now(current, config["daily_time"], last)
        if active or initial_due or daily_due:
            backup.set_setting("last_run_date", current.strftime("%Y-%m-%d"))
            try:
                backup.run(resume=active or (initial_incomplete and bool(backup.setting("last_scan"))))
            except InterruptedError:
                backup.update(phase="interrupted", totals=backup.totals())
            except Exception as error:
                LOG.error("Run stopped: %s", type(error).__name__)
                backup.update(phase="failed", last_error=f"Run stopped ({type(error).__name__}); inspect container/OpenList logs", totals=backup.totals())
                backup.set_setting("run_active", "0")
                backup.event(f"上传暂时停止：{type(error).__name__}，未完成文件将在稍后重试")
            backup.set_setting("retry_after", str(time.time() + config.get("retry_run_seconds", 1800)))
        backup.update(next_run=next_run(config["daily_time"], backup.setting("last_run_date", "")) if config.get("daily_enabled", False) else None,
                      daily_time=config["daily_time"], daily_enabled=config.get("daily_enabled", False),
                      initial_complete=backup.setting("initial_complete"), timezone="Asia/Shanghai")
        STOP.wait(15)
    backup.update(phase="stopped")


if __name__ == "__main__":
    main()
