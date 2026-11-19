"""Upload one real, small recording and compare a downloaded SHA-256 hash.

Executed once before starting the daemon; its success checkpoint is reused.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import backup


def main():
    config = json.loads(Path("/app/config.json").read_text())
    obj = backup.Backup(config)
    if not os.statvfs(obj.root).f_flag & os.ST_RDONLY:
        raise RuntimeError("Recording mount must be read-only")
    selected = None
    errors = []
    for path, info in backup.source_files(obj.root / "aqara_video", errors, backup.SYSTEM_NAMES):
        if path.suffix.lower() == ".mp4" and 1024 <= info.st_size <= 8 * 1024 * 1024 and info.st_mtime < time.time() - 5400:
            selected = path, info
            break
    if not selected:
        raise RuntimeError("No suitable small recording for smoke test")
    path, info = selected
    relative = path.relative_to(obj.root).as_posix()
    obj.db.execute("INSERT OR IGNORE INTO files(path,signature,size,mtime_ns,seen,status) VALUES (?,?,?,?,?,'pending')",
                   (relative, backup.fingerprint(info), info.st_size, info.st_mtime_ns, "smoke"))
    obj.db.commit()
    row = obj.db.execute("SELECT * FROM files WHERE path=?", (relative,)).fetchone()
    if row["status"] != "uploaded" and not obj.upload_one(row):
        raise RuntimeError("Sample changed before test upload")
    item = obj.api.get("/" + relative)
    headers = {"User-Agent": "pan.baidu.com", **(item.get("header") or {})}
    request = urllib.request.Request(item["raw_url"], headers=headers)
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:
        while block := response.read(1024 * 1024):
            digest.update(block)
    expected = obj.db.execute("SELECT sha256 FROM files WHERE path=?", (relative,)).fetchone()[0]
    if digest.hexdigest() != expected:
        raise RuntimeError("Roundtrip hash mismatch")
    print(json.dumps({"roundtrip_sha256_match": True, "bytes": info.st_size, "source_mount_readonly": True}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Smoke failed:", type(error).__name__)
        raise SystemExit(1)
