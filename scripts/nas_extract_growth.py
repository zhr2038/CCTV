"""Verify and unpack only regular deployment files beneath the named service."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,shutil,os

ROOT=Path('/volume2/cctv-growth-service')
items=[('state-transfer.tar.gz','state','7956e8399342b304c9f49af24098073cb2d3d809f741b10009f6a3fe4a8b8749'),
       ('code-transfer.tar.gz','.','bd2c06c1b900be142daae36f03f8a747de5d734ada5aba01516daf205947bf0a')]
for filename,folder,expected in items:
 archive=ROOT/filename;digest=hashlib.sha256(archive.read_bytes()).hexdigest()
 if digest!=expected:raise ValueError('Transfer archive checksum mismatch')
 target=(ROOT/folder).resolve()
 if not target.is_relative_to(ROOT):raise ValueError('Target escaped service directory')
 with tarfile.open(archive,'r:gz') as source:
  for member in source:
   relative=PurePosixPath(member.name)
   if relative.is_absolute() or '..' in relative.parts or not member.isfile():raise ValueError('Unexpected archive member')
   path=(target/str(relative)).resolve()
   if not path.is_relative_to(target):raise ValueError('Archive member escaped')
   path.parent.mkdir(parents=True,exist_ok=True)
   with source.extractfile(member) as incoming,path.open('wb') as outgoing:shutil.copyfileobj(incoming,outgoing)
   path.chmod(0o640)
 print(json.dumps({'archive':filename,'sha256':digest,'verified':True}),flush=True)
(ROOT/'vendor').mkdir(exist_ok=True)
binary=ROOT/'code/kimi-linux-2.4.0'
if hashlib.sha256(binary.read_bytes()).hexdigest()!='8221a22f3417656a247cfe26240e73a6bcfba2456d6b2927e7209fe995f1da68':raise ValueError('CLI checksum mismatch')
destination=ROOT/'vendor/kimi'
if not destination.exists():os.link(binary,destination)
destination.chmod(0o755)
(ROOT/'state/home').mkdir(exist_ok=True)
jobs=ROOT/'state/jobs';(jobs/'kimi').mkdir(parents=True,exist_ok=True);(jobs/'deepseek').mkdir(parents=True,exist_ok=True)
kimi=json.loads((ROOT/'state/history/kimi-batch/progress.json').read_text())
(jobs/'kimi/progress.json').write_text(json.dumps(kimi),encoding='utf-8')
(jobs/'deepseek/progress.json').write_text(json.dumps({'status':'diagnostic_pending','source_deletions':0}),encoding='utf-8')
shutil.copy2(ROOT/'offline-migration-manifest.json',ROOT/'state/transfer-manifest.json')
print('Deployment files extracted; no source media tree modified',flush=True)
