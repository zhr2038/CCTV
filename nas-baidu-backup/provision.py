"""Run as NAS root once; reuse the existing authorized Baidu mount safely."""
import json
import os
from pathlib import Path
import secrets
import sqlite3

from backup import OpenList


def main():
    base = Path(__file__).resolve().parent
    con = sqlite3.connect(
        "file:" + os.environ.get("OPENLIST_DB", "/volume1/docker/openlist-organizer/data/data.db") + "?mode=ro",
        uri=True,
    )
    con.row_factory = sqlite3.Row
    storage = con.execute("SELECT * FROM x_storages WHERE driver='BaiduNetdisk' AND disabled=0").fetchone()
    if not storage or storage["status"] != "work":
        raise SystemExit("No healthy Baidu storage found")
    mount = storage["mount_path"]
    admin_token = con.execute("SELECT value FROM x_setting_items WHERE key='token'").fetchone()[0]
    openlist_url = os.environ.get("OPENLIST_URL", "http://127.0.0.1:5244").rstrip("/")
    client = OpenList(openlist_url, "unused")
    client.token = admin_token
    target = mount.rstrip("/") + "/CCTV"
    client.mkdir(target)
    username = "cctv-backup"
    secret_dir = base / "secrets"
    secret_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(secret_dir, 0o700)
    secret_path = secret_dir / "openlist.json"
    existing = con.execute("SELECT id,base_path,permission FROM x_users WHERE username=?", (username,)).fetchone()
    if existing:
        if not secret_path.is_file() or existing["base_path"] != target or existing["permission"] != 8:
            raise SystemExit("Account already exists with a different setup; inspect before changing it")
    else:
        credentials = {"username": username, "password": secrets.token_urlsafe(40)}
        fd = os.open(secret_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(credentials, stream)
        client.request("/api/admin/user/create", {**credentials, "base_path": target, "role": 0,
                                                "permission": 8, "disabled": False, "allow_ldap": False})
    os.chmod(secret_path, 0o600)
    check = OpenList(openlist_url, str(secret_path))
    check.login()
    result = check.get("/")
    if not result["is_dir"]:
        raise SystemExit("Backup account root verification failed")
    (base / "state").mkdir(mode=0o700, exist_ok=True)
    os.chmod(base / "state", 0o700)
    print(json.dumps({"account": username, "destination": target, "permission": "mkdir/upload; no delete/rename/move", "ready": True}, ensure_ascii=True))


if __name__ == "__main__":
    main()
