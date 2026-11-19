"""Entrypoint for NAS Unattended Growth Archive V3."""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import signal
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import lifecycle as lc
from .daemon import GrowthDaemon
from .db import ArchiveDB
from .web_server import create_server


def resolve_config_path(raw: str | Path) -> Path:
    p = Path(raw)
    if str(raw) == "/app/config.toml" and Path("/app/conf/config.toml").is_file():
        return Path("/app/conf/config.toml")
    return p


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def _lock_processor(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NAS Unattended Growth Archive V3")
    ap.add_argument("--config", default="/app/config.toml")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_srv = sub.add_parser("serve", help="Run HTTP Web Gallery + P0/P1 unattended daemon")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=18795)

    p_day = sub.add_parser("process-day", help="Run bounded synchronous test on one day")
    p_day.add_argument("--day", required=True)
    p_day.add_argument("--limit", type=int, default=20)
    p_day.add_argument("--tier", default="P0_LIVE")

    p_film = sub.add_parser("compile-film", help="Compile & atomically publish a monthly or annual film")
    p_film.add_argument("--kind", choices=["monthly", "annual"], default="monthly")
    p_film.add_argument("--months-old", type=int, default=0)
    p_film.add_argument("--period-start", default="")
    p_film.add_argument("--period-end", default="")
    p_film.add_argument("--max-shots", type=int, default=0)
    p_film.add_argument("--out-w", type=int, default=1920)
    p_film.add_argument("--out-h", type=int, default=1080)
    p_film.add_argument("--allow-partial", action="store_true")
    p_film.add_argument("--force", action="store_true")

    p_drain = sub.add_parser("drain-queue", help="Drain queued files using scheduler memory/heavy caps")
    p_drain.add_argument("--days", nargs="*", default=[])
    p_drain.add_argument("--workers", type=int, default=0)
    p_drain.add_argument("--max-heavy", type=int, default=0)
    p_drain.add_argument("--max-seconds", type=float, default=3600.0)

    sub.add_parser("health", help="Print health JSON")
    p_ctl = sub.add_parser("ctl", help="Pause or resume daemon")
    p_ctl.add_argument("action", choices=["pause", "resume"])

    args = ap.parse_args(argv)
    cfg_path = resolve_config_path(args.config)
    args._resolved_config = str(cfg_path)
    cfg = load_config(cfg_path)
    data_root = Path(cfg["project"]["data_root"])
    if args.cmd == "health":
        db = ArchiveDB(data_root / "state" / "growth_v3.sqlite3")
        snap = db.get_dashboard_snapshot()
        print(json.dumps({"ok": True, **snap.get("state", {})}, ensure_ascii=False))
        return 0

    if args.cmd == "ctl":
        p = data_root / "state" / "PAUSE"
        p.parent.mkdir(parents=True, exist_ok=True)
        if args.action == "pause":
            p.write_text("paused\n", encoding="utf-8")
        else:
            p.unlink(missing_ok=True)
        print(json.dumps({"paused": p.is_file()}))
        return 0

    # Both entry points write the same database and replace published media.
    state_dir = data_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "processor.lock").open("a+b") as lock:
        try:
            _lock_processor(lock)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            print("该归档目录已有 V3 处理进程。", file=sys.stderr)
            return 2
        return _run_processor(args, cfg, data_root, ArchiveDB(state_dir / "growth_v3.sqlite3"))


def _run_processor(args: argparse.Namespace, cfg: dict[str, Any], data_root: Path, db: ArchiveDB) -> int:
    if args.cmd == "drain-queue":
        if getattr(args, "workers", 0) > 0:
            cfg.setdefault("scheduler", {})["workers"] = int(args.workers)
        if getattr(args, "max_heavy", 0) > 0:
            cfg.setdefault("scheduler", {})["max_heavy_workers"] = int(args.max_heavy)
    cfg_path_val = getattr(args, "_resolved_config", None) or getattr(args, "config", None)
    daemon = GrowthDaemon(cfg, db, ff_bin=getattr(args, "ffmpeg", "ffmpeg"), fp_bin=getattr(args, "ffprobe", "ffprobe"), config_path=cfg_path_val)
    if args.cmd == "process-day":
        cams = daemon.refresh_camera_health()["cameras"]
        daemon.discover_day_files(args.day)
        daemon._run_day_batch(args.day, args.tier, args.limit, cams)
        st = daemon.db.count_day_stats(args.day)
        summary = daemon.finalize_day(args.day, args.tier, final=(st["queued"] == 0))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "drain-queue":
        res = daemon.run_queue_until_idle(
            discover_days=list(args.days) if args.days else None,
            max_seconds=float(args.max_seconds),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("errors", 0) == 0 else 1

    if args.cmd == "compile-film":
        tz = ZoneInfo(cfg.get("project", {}).get("timezone", "Asia/Shanghai"))
        bday = dt.date.fromisoformat(str(cfg.get("project", {}).get("birthday", "2025-12-17")))
        today_d = dt.datetime.now(tz).date()
        compile_day = int(cfg.get("lifecycle", {}).get("monthly_compile_day", 18))
        age = lc.compute_age_info(bday.isoformat(), today_d)
        months_old = int(args.months_old) if args.months_old > 0 else max(1, int(age["months"] or 1))
        spec = lc.build_film_spec(bday, months_old, args.kind, compile_day=compile_day)
        if args.period_start:
            spec["period_start"] = args.period_start
        if args.period_end:
            spec["period_end"] = args.period_end
        if not args.allow_partial:
            c_stats = db.count_day_stats(spec["period_end"])
            if c_stats["queued"] > 0:
                print(json.dumps({"error": "cutoff_day_still_queued", "queued": c_stats["queued"], "spec": spec}, ensure_ascii=False, indent=2))
                return 2
        res = daemon.compile_film(
            spec,
            max_shots=(int(args.max_shots) if args.max_shots > 0 else None),
            force=bool(args.force),
            out_w=int(args.out_w),
            out_h=int(args.out_h),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("status") == "published" else 1

    # serve: daemon thread + blocking HTTP server
    t = threading.Thread(target=daemon.run_forever, name="growth-daemon", daemon=True)
    t.start()
    httpd = create_server(args.host, args.port, data_root, db, daemon, cfg)

    def _sig_handler(*_a) -> None:
        daemon.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)
    print(f"[NAS Growth V3] Web & Unattended Daemon listening on {args.host}:{args.port} (VAAPI={daemon.use_vaapi})", flush=True)
    try:
        httpd.serve_forever()
    finally:
        daemon.stop()
        httpd.server_close()
        t.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
