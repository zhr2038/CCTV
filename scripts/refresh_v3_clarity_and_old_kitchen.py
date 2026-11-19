"""Re-render V3 deliverables without 2.0x digital blur (native 1:1 1080p floor) + weave old Xiaomi kitchen highlights."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import getpass
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tarfile
import tomllib
from zoneinfo import ZoneInfo

PROJECT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from cctv_ai import growth_v2 as v2  # noqa: E402
from cctv_ai import growth_v3_media as v3m  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")


def _load_birthday() -> dt.date:
    for rel in ("nas-growth-v3/config.toml", "config/growth_archive.toml"):
        cfg_p = PROJECT / rel
        if cfg_p.is_file():
            bday = tomllib.loads(cfg_p.read_text(encoding="utf-8")).get("project", {}).get("birthday")
            if bday:
                return dt.date.fromisoformat(str(bday))
    return dt.date(2025, 1, 1)


def _load_env_nas() -> None:
    env_file = PROJECT / ".env.nas"
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _resolve_ssh_creds() -> tuple[str, str, str, str]:
    _load_env_nas()
    host = os.environ.get("CCTV_NAS_HOST") or input("NAS LAN IP / Host: ").strip()
    user = os.environ.get("CCTV_NAS_USER") or input("NAS SSH Username: ").strip()
    group = os.environ.get("CCTV_NAS_GROUP") or "admin"
    pwd = os.environ.get("CCTV_NAS_PASSWORD") or getpass.getpass(f"SSH password for {user}@{host}: ")
    return host, user, group, pwd


def sync_v2_to_nas(v2_dir: pathlib.Path) -> None:
    import paramiko

    host, user, group, pwd = _resolve_ssh_creds()
    buf = io.BytesIO()
    include_items = [
        "deliverables.json",
        "t0_manifest.json",
        "photos",
        "highlights",
        "t0_keepers",
        "monthly_cards",
        "monthly_growth_1to11m.mp4",
        "monthly_growth_1to11m.jpg",
        "growth_smooth_progression.mp4",
        "growth_smooth_progression.jpg",
        "smooth_living_room_2x.mp4",
        "smooth_living_room_2x.jpg",
        "smooth_bedside_135x.mp4",
        "smooth_bedside_135x.jpg",
        "smooth_kitchen_1x.mp4",
        "smooth_kitchen_1x.jpg",
    ]
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in include_items:
            p = v2_dir / name
            if p.exists():
                tf.add(p, arcname=f"v2/{name}")
    tar_bytes = buf.getvalue()
    print(f"[Sync NAS] Streaming {len(tar_bytes) / (1024 * 1024):.1f} MB refreshed v2 bundle to NAS...", flush=True)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=pwd, timeout=20)
    try:
        sftp = ssh.open_sftp()
        try:
            sftp.putfo(io.BytesIO(tar_bytes), "/docker/cctv-growth-v3/.v2_sync_tmp.tar.gz")
        finally:
            sftp.close()
        cmd = (
            "sudo -S -p '' bash -c '"
            "set -e; "
            "trap \"rm -f /volume1/docker/cctv-growth-v3/.v2_sync_tmp.tar.gz\" EXIT; "
            "tar -xzf /volume1/docker/cctv-growth-v3/.v2_sync_tmp.tar.gz -C /volume2/Growth_Archive_V3; "
            f"chown -R {user}:{group} /volume2/Growth_Archive_V3/v2"
            "'"
        )
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=180)
        stdin.write(pwd + "\n")
        stdin.flush()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(f"NAS sync failed rc={rc}: {err[-500:]}")
        if out.strip():
            print(out.strip())
    finally:
        ssh.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh V3 deliverables at crisp native 1080p zoom + old kitchen footage")
    parser.add_argument("--pilot-root", default=str(PROJECT / "output" / "growth-archive-pilot-20261117"))
    parser.add_argument("--old-kitchen-json", default="")
    parser.add_argument("--old-kitchen-camera", default=os.environ.get("CCTV_OLD_KITCHEN_CAMERA", "cam_kitchen_legacy"))
    parser.add_argument("--sync-nas", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()
    birthday = _load_birthday()

    out_root = pathlib.Path(args.pilot_root)
    v2_dir = out_root / "v2"
    hl_1108 = out_root / "day-20261108" / "highlight_20261108.mp4"
    if hl_1108.is_file() and hl_1108.stat().st_size != 43652591:
        raise RuntimeError(f"Unexpected pre-check size on {hl_1108}")

    if args.sync_only:
        sync_v2_to_nas(v2_dir)
        return 0

    vlm_p = v2_dir / "vlm_verified.json"
    bak_p = v2_dir / "work" / "vlm_verified.pre_clarity.json"
    bak_p.parent.mkdir(parents=True, exist_ok=True)
    if not bak_p.is_file():
        shutil.copy2(vlm_p, bak_p)

    vlm_data = json.loads(bak_p.read_text(encoding="utf-8"))
    if args.old_kitchen_json and pathlib.Path(args.old_kitchen_json).is_file():
        old_cam = args.old_kitchen_camera
        old_items = json.loads(pathlib.Path(args.old_kitchen_json).read_text(encoding="utf-8"))
        ms_list = [m for m in vlm_data.get("milestones", []) if m.get("camera") != old_cam]
        k_smooth = [it for it in (vlm_data.get("smooth_4angles", {}).get("kitchen_4k") or []) if it.get("camera") != old_cam]
        added_smooth = []
        for idx, oi in enumerate(old_items):
            src_win = oi["source_win"]
            t0 = float(oi.get("t0", 12.0))
            s0 = max(0.0, t0 - 2.5)
            day_s, t_lbl = _parse_old_kitchen_wall_time(src_win, s0)
            day_n = (dt.date.fromisoformat(day_s) - birthday).days + 1
            vlm_obj = dict(oi.get("vlm") or {})
            cats = list(vlm_obj.get("categories") or [])
            if "kitchen_visit" not in cats:
                cats.insert(0, "kitchen_visit")
            vlm_obj["categories"] = cats
            vlm_obj["excitement_score"] = max(4, int(vlm_obj.get("excitement_score", 4) or 4))
            vlm_obj["stage_zh"] = (vlm_obj.get("caption_zh") or "旧厨房温馨时刻")[:18]
            ms_list.append({
                "camera": old_cam,
                "source": src_win,
                "start_s": s0,
                "end_s": min(59.0, s0 + 6.5),
                "hint": "old_kitchen_verified",
                "priority": 9.5 - idx * 0.1,
                "time_str": t_lbl,
                "vlm": vlm_obj,
            })
            added_smooth.append({
                "day": day_s,
                "day_n": day_n,
                "camera": old_cam,
                "source": src_win,
                "start_s": s0,
                "vlm": vlm_obj,
            })
        added_smooth.sort(key=lambda x: x["day"])
        vlm_data["milestones"] = ms_list
        vlm_data.setdefault("smooth_4angles", {})["kitchen_4k"] = added_smooth[:6] + k_smooth
        print(f"[Refresh V3] Merged {len(old_items)} confirmed {old_cam} (old kitchen) clips!", flush=True)

    vlm_p.write_text(json.dumps(vlm_data, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = v2.load_v2_config(PROJECT / "config" / "growth_archive.toml")
    deliv = v3m.build_v3_deliverables(cfg, out_root)
    print(
        "[Refresh V3] Rebuilt deliverables:",
        "photos=", len(deliv.get("photos", [])),
        "highlights=", len(deliv.get("highlights", [])),
        "smooth_films=", {k: (v.get("name"), v.get("zoom_x"), v.get("shots")) for k, v in deliv.get("smooth_films", {}).items()},
        flush=True,
    )

    if hl_1108.is_file() and hl_1108.stat().st_size != 43652591:
        raise RuntimeError(f"CRITICAL: {hl_1108} size altered!")

    if args.sync_nas:
        sync_v2_to_nas(v2_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
