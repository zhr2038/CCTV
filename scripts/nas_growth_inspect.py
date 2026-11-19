"""Read-only NAS capability inspection. Password stays in RAM/non-echo input."""
from pathlib import Path
import getpass, json, os
import paramiko

target = os.environ.get('CCTV_NAS_HOST') or input('NAS LAN IP / Host: ').strip()
user = os.environ.get('CCTV_NAS_USER') or input('NAS SSH username: ').strip()
password = getpass.getpass(f'NAS SSH password for {user}@{target} (non-echo): ')
client = paramiko.SSHClient(); client.load_system_host_keys(); client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(target, username=user, password=password, timeout=12, look_for_keys=False, allow_agent=False)
commands={
 'platform':'uname -a',
 'resources':"python3 -c \"import json,os,platform;print(json.dumps({'arch':platform.machine(),'cpus':os.cpu_count(),'python':platform.python_version()}));print(open('/proc/meminfo').read().splitlines()[:3]);print(open('/proc/loadavg').read())\"",
 'storage':'df -B1 /volume2/CCTV',
 'tools':'command -v docker; command -v python3; command -v ffmpeg; command -v kimi',
 'ports':"ss -ltn | awk 'NR==1 || /:1879[0-9] /'",
 'path':"python3 -c \"import os;print(os.path.realpath('/volume2/CCTV'));print(os.path.isdir('/volume2/docker'));print(os.path.isdir('/volume2/Apps'))\"",
 'docker':'sudo -S -p \'\' docker ps --format \'{{.Names}}|{{.Image}}|{{.Status}}\'',
 'docker_info':'sudo -S -p \'\' docker info --format \'{{.OSType}}|{{.Architecture}}|{{.NCPU}}|{{.MemTotal}}|{{.DockerRootDir}}\''}
result={}
try:
 for name,command in commands.items():
  stdin,stdout,stderr=client.exec_command(command,timeout=25)
  if command.startswith('sudo '):stdin.write(password+'\n');stdin.flush();stdin.channel.shutdown_write()
  result[name]={'exit':stdout.channel.recv_exit_status(),'stdout':stdout.read().decode(errors='replace')[:8000],
    'stderr':stderr.read().decode(errors='replace')[:1000]}
  print(json.dumps({name:result[name]},ensure_ascii=True),flush=True)
 Path('output/nas_growth_capabilities_20261104.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
finally:client.close()
