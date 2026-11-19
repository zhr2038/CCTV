"""One-command Windows-dev -> NAS 24x7 deploy script for cctv-growth-v3.

Streams clean source tarball via SSH (bypasses SFTP chroot), preserves existing
/volume1/docker/cctv-growth-v3/secrets/*, syncs openlist.json if missing,
rebuilds the image layer-cached (~2s), recreates the container, and verifies
health, MemAvailable >= 4.0 GiB, HTTP 200, and SHA-256 parity.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import io
import json
import os
import pathlib
import tarfile
import time
import urllib.request

import paramiko


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCAL_PKG_DIR = REPO_ROOT / "nas-growth-v3"
MIN_MEM_KIB = 4194304


def _load_env_nas() -> None:
    env_file = REPO_ROOT / ".env.nas"
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _resolve_nas_target(args: argparse.Namespace) -> tuple[str, str, str, str, str]:
    _load_env_nas()
    host = args.host or os.environ.get("CCTV_NAS_HOST") or ""
    user = args.user or os.environ.get("CCTV_NAS_USER") or ""
    group = args.group or os.environ.get("CCTV_NAS_GROUP") or "admin"
    deploy_dir = args.deploy_dir or os.environ.get("CCTV_NAS_DEPLOY_DIR") or "/volume1/docker/cctv-growth-v3"
    if not host:
        host = input("NAS LAN IP / Host: ").strip()
    if not user:
        user = input("NAS SSH Username: ").strip()
    if not host or not user:
        raise RuntimeError("Missing NAS host or SSH user. Configure .env.nas or pass --host/--user.")
    pwd = os.environ.get("CCTV_NAS_PASSWORD") or getpass.getpass(f"SSH/sudo password for {user}@{host} (non-echo): ")
    return host, user, group, deploy_dir, pwd


def _build_source_tar(with_models: bool = False) -> tuple[bytes, dict[str, str]]:
    buf = io.BytesIO()
    hashes: dict[str, str] = {}
    top_files = ["config.toml", "compose.yaml", "Dockerfile", "README.md", ".dockerignore"]

    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for fn in top_files:
            p = LOCAL_PKG_DIR / fn
            if p.is_file():
                tf.add(p, arcname=fn)
                if fn == "config.toml":
                    hashes[fn] = hashlib.sha256(p.read_bytes()).hexdigest()
        for py in sorted((LOCAL_PKG_DIR / "nas_growth").glob("*.py")):
            arc = f"nas_growth/{py.name}"
            tf.add(py, arcname=arc)
            hashes[arc] = hashlib.sha256(py.read_bytes()).hexdigest()
        face_models_dir = LOCAL_PKG_DIR / "models" / "face"
        if not with_models and face_models_dir.is_dir():
            for mf in sorted(face_models_dir.glob("*.onnx")):
                if mf.is_file():
                    arc = mf.relative_to(LOCAL_PKG_DIR).as_posix()
                    tf.add(mf, arcname=arc)
                    hashes[arc] = hashlib.sha256(mf.read_bytes()).hexdigest()
        if with_models and (LOCAL_PKG_DIR / "models").is_dir():
            for mf in sorted((LOCAL_PKG_DIR / "models").rglob("*")):
                if mf.is_file():
                    tf.add(mf, arcname=mf.relative_to(LOCAL_PKG_DIR).as_posix())
    return buf.getvalue(), hashes


def deploy(args: argparse.Namespace) -> int:
    host, user, group, deploy_dir, pwd = _resolve_nas_target(args)
    tar_bytes, local_hashes = _build_source_tar(with_models=args.with_models)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=pwd, timeout=15)

    try:
        up_in, up_out, up_err = ssh.exec_command("cat > /tmp/.cctv_growth_v3_src.tgz", timeout=180)
        up_in.write(tar_bytes)
        up_in.flush()
        up_in.channel.shutdown_write()
        if up_out.channel.recv_exit_status() != 0:
            print("UPLOAD STDERR:", up_err.read().decode("utf-8", errors="replace")[-800:])
            return 1

        rebuild_cmd = "docker compose up -d --build" if not args.no_rebuild else "docker compose up -d"
        remote_script = f"""sudo -S -p '' bash -c '
set -e
mkdir -p {deploy_dir} {deploy_dir}/conf
tar -xzf /tmp/.cctv_growth_v3_src.tgz -C {deploy_dir}
rm -f /tmp/.cctv_growth_v3_src.tgz
cp {deploy_dir}/config.toml {deploy_dir}/conf/config.toml
if [ -f /volume1/docker/cctv-baidu-backup/secrets/openlist.json ] && [ ! -f {deploy_dir}/secrets/openlist.json ]; then
  mkdir -p {deploy_dir}/secrets
  cp /volume1/docker/cctv-baidu-backup/secrets/openlist.json {deploy_dir}/secrets/openlist.json
  chmod 0600 {deploy_dir}/secrets/openlist.json
fi
chown -R {user}:{group} {deploy_dir}/config.toml {deploy_dir}/conf {deploy_dir}/compose.yaml {deploy_dir}/Dockerfile {deploy_dir}/README.md {deploy_dir}/nas_growth
if [ -d {deploy_dir}/secrets ]; then chown -R {user}:{group} {deploy_dir}/secrets; chmod 0700 {deploy_dir}/secrets; fi
if [ -d {deploy_dir}/models/face ]; then chown -R {user}:{group} {deploy_dir}/models/face; chmod -R a+rX {deploy_dir}/models/face; fi
chmod 0644 {deploy_dir}/conf/config.toml
cd {deploy_dir}
{rebuild_cmd}
'"""
        stdin, stdout, stderr = ssh.exec_command(remote_script, timeout=180)
        stdin.write(pwd + "\n")
        stdin.flush()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        if out.strip():
            print(out.strip())
        if rc != 0:
            print("DEPLOY STDERR:", err[-1500:])
            return rc

        # Wait up to 40s for container HEALTHCHECK -> healthy
        health = "unknown"
        for _ in range(20):
            stdin, stdout, _ = ssh.exec_command(
                "sudo -S -p '' bash -c \"docker inspect cctv-growth-v3 --format '{{.RestartCount}}|{{.State.OOMKilled}}|{{.State.Health.Status}}'\"",
                timeout=15,
            )
            stdin.write(pwd + "\n")
            stdin.flush()
            st_line = stdout.read().decode("utf-8", errors="replace").strip()
            parts = st_line.split("|")
            if len(parts) == 3 and parts[2] == "healthy":
                health = st_line
                break
            health = st_line
            time.sleep(2)

        if health.split("|")[1:] != ["false", "healthy"]:
            print(f"Deployment health check failed: {health}")
            return 4

        verify_inner = f"""set -e
echo "CONTAINER_STATE={health}"
awk '/MemAvailable:/{{print "HOST_MEM_KIB=" $2}}' /proc/meminfo
docker inspect cctv-growth-v3 --format 'MOUNTS={{{{range .Mounts}}}}{{{{.Destination}}}}:{{{{.Mode}}}} {{{{end}}}}'
docker exec -i cctv-growth-v3 python - <<'PY'
import hashlib, json, pathlib
files = {json.dumps(list(local_hashes.keys()))}
print("SHA256_JSON=" + json.dumps({{f: hashlib.sha256((pathlib.Path("/app") / f).read_bytes()).hexdigest() for f in files}}))
PY
docker ps --format '{{{{.Names}}}}|{{{{.Status}}}}' | grep -E 'cctv-growth-v3|cctv-baidu-backup|quant-' || true
"""
        v_b64 = base64.b64encode(verify_inner.encode("utf-8")).decode("ascii")
        stdin, stdout, stderr = ssh.exec_command(
            f"sudo -S -p '' bash -c \"echo {v_b64} | base64 -d | bash\"",
            timeout=30,
        )
        stdin.write(pwd + "\n")
        stdin.flush()
        vout = stdout.read().decode("utf-8", errors="replace").strip()
        verr = stderr.read().decode("utf-8", errors="replace").strip()
        print(vout)
        if verr:
            print("VERIFY_STDERR:", verr[:400])
        remote_hashes: dict[str, str] = {}
        for line in vout.splitlines():
            if line.startswith("SHA256_JSON="):
                remote_hashes = json.loads(line.split("=", 1)[1])
        for k, expected in local_hashes.items():
            if remote_hashes.get(k) != expected:
                print(f"WARNING: SHA-256 mismatch check on {k}")
                return 2
        for line in vout.splitlines():
            if line.startswith("HOST_MEM_KIB="):
                mem_kib = int(line.split("=", 1)[1])
                if mem_kib < MIN_MEM_KIB:
                    print(f"WARNING: Host MemAvailable {mem_kib} KiB < {MIN_MEM_KIB} KiB!")
                    return 3
    finally:
        ssh.close()

    with urllib.request.urlopen(f"http://{host}:18795/api/status", timeout=8) as r:
        st_json = json.loads(r.read().decode("utf-8"))
        print(
            "HTTP_200_OK:",
            "done=", st_json.get("totals", {}).get("done_files"),
            "/", st_json.get("totals", {}).get("total_files"),
            "tokens=", st_json.get("token_totals", {}).get("total_tokens"),
            "guard=", (st_json.get("lifecycle") or {}).get("guard_summary_zh"),
        )
    return 0


def hot_reload_config(args: argparse.Namespace) -> int:
    host, user, group, deploy_dir, pwd = _resolve_nas_target(args)
    cfg_bytes = (LOCAL_PKG_DIR / "config.toml").read_bytes()
    cfg_b64 = base64.b64encode(cfg_bytes).decode("ascii")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=pwd, timeout=15)
    try:
        remote_cmd = f"""sudo -S -p '' bash -c '
set -e
mkdir -p {deploy_dir}/conf
echo "{cfg_b64}" | base64 -d > {deploy_dir}/conf/config.toml.tmp
mv -f {deploy_dir}/conf/config.toml.tmp {deploy_dir}/conf/config.toml
cp -f {deploy_dir}/conf/config.toml {deploy_dir}/config.toml
chown {user}:{group} {deploy_dir}/conf/config.toml {deploy_dir}/config.toml
chmod 0644 {deploy_dir}/conf/config.toml
'"""
        sin, sout, serr = ssh.exec_command(remote_cmd, timeout=30)
        sin.write(pwd + "\n")
        sin.flush()
        rc = sout.channel.recv_exit_status()
        if rc != 0:
            print("HOT_RELOAD_STDERR:", serr.read().decode("utf-8", errors="replace"))
            return rc
    finally:
        ssh.close()

    req = urllib.request.Request(
        f"http://{host}:18795/api/ctl/reload_config",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as r:
        res = json.loads(r.read().decode("utf-8"))
        print("HOT_RELOAD_OK:", json.dumps(res, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy nas-growth-v3 from local dev to NAS 24x7 container")
    parser.add_argument("--host", default="", help="NAS host/IP (default: CCTV_NAS_HOST from .env.nas or env)")
    parser.add_argument("--user", default="", help="NAS SSH username (default: CCTV_NAS_USER)")
    parser.add_argument("--group", default="", help="NAS file owner group (default: CCTV_NAS_GROUP or admin)")
    parser.add_argument("--deploy-dir", default="", help="Remote deploy directory (default: /volume1/docker/cctv-growth-v3)")
    parser.add_argument("--with-models", action="store_true", help="Also upload nas-growth-v3/models (normally cached)")
    parser.add_argument("--no-rebuild", action="store_true", help="Skip docker compose build")
    parser.add_argument("--config-only", action="store_true", help="Hot-reload config.toml via /app/conf without restarting container")
    args = parser.parse_args()
    if args.config_only:
        return hot_reload_config(args)
    return deploy(args)


if __name__ == "__main__":
    raise SystemExit(main())
