"""One user-requested, frozen short labeling batch; no schedule or live scan."""
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from cctv_ai.growth_workflow import save_json, sha_file
from cctv_ai.pipeline import PipelineLock

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/growth-first-batch-20261103'
OUTPUT=ROOT/'output/growth-kimi-current-batch-20261103'
FIRST=ROOT/'output/growth-kimi-interactive-20261103'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect('file:'+(SOURCE/'queue.sqlite3').as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    files=[dict(r) for r in db.execute('SELECT * FROM files ORDER BY relative')];db.close()
    if len(files)>11 or sum(f['duration'] for f in files)>360:raise ValueError('Frozen short-batch bound exceeded')
    selected=[{'id':f['id'],'relative':f['relative'],'sha256':f['sha256'],
        'duration':f['duration'],'frames':f['frames'],'output':str(FIRST if i==0 else OUTPUT/f['id'])}
        for i,f in enumerate(files)]
    quota=json.loads((FIRST/'subscription_usage_snapshot.json').read_text(encoding='utf-8'))
    reset_at=datetime.fromisoformat(quota['quota']['usages']['limit5h']['resetAt'].replace('Z','+00:00'))
    plan={'scope':'one explicit user-requested current interactive task; existing cached originals only',
        'model':'kimi-code/k3.1-pro','files':selected,'files_total':len(selected),
        'recorded_seconds':sum(f['duration'] for f in selected),'main_frames':sum(f['frames'] for f in selected),
        'window_seconds':6,'overlap_seconds':1,'source_deletions':0,
        'not_an_annual_background_service':True,'quota_window_end':reset_at.isoformat()}
    plan_path=OUTPUT/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Existing frozen batch differs')
    save_json(plan_path,plan)
    state={'status':'running','started_at':now(),'files_total':len(selected),'files_completed':0,
        'current_file':None,'current_child_pid':None,'source_deletions':0}
    with PipelineLock(OUTPUT/'worker.lock',label='current Kimi labeling batch'):
        try:
            for index,f in enumerate(files):
                if datetime.now(ZoneInfo('Asia/Shanghai'))>=reset_at:
                    state['status']='current_quota_window_ended';break
                original=SOURCE/f['cache']
                if sha_file(original)!=f['sha256']:raise ValueError('Cached original changed')
                folder=Path(selected[index]['output']);state['current_file']=f['relative'];save_json(OUTPUT/'progress.json',state)
                progress=folder/'progress.json'
                if index==0:
                    # Adopt the already submitted job without issuing any duplicate prompt.
                    while True:
                        first=json.loads(progress.read_text(encoding='utf-8'))
                        if first['status']=='finished':break
                        if first['status']!='running':raise RuntimeError('First active file requires inspection; no automatic retry')
                        if datetime.now(ZoneInfo('Asia/Shanghai'))>=reset_at:raise RuntimeError('Quota window ended while awaiting first file')
                        time.sleep(5)
                else:
                    if progress.exists():
                        prior=json.loads(progress.read_text(encoding='utf-8'))
                        if prior['status']=='finished':state['files_completed']+=1;continue
                        raise RuntimeError('Prior child needs inspection; no automatic retry')
                    folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'worker.log').open('w',encoding='utf-8') as log:
                        command=[sys.executable,str(ROOT/'scripts/kimi_growth_interactive.py'),
                            '--source-batch',str(SOURCE),'--file-id',f['id'],'--output',str(folder)]
                        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                        state['current_child_pid']=child.pid;save_json(OUTPUT/'progress.json',state)
                        result=child.wait()
                    state['current_child_pid']=None
                    if result!=0:raise RuntimeError('Child stopped; inspect its preserved session before any retry')
                    if json.loads(progress.read_text(encoding='utf-8'))['status']!='finished':raise RuntimeError('Child did not finish')
                state['files_completed']+=1;save_json(OUTPUT/'progress.json',state)
                print(json.dumps({'file':f['relative'],'completed':state['files_completed'],'total':len(selected)}),flush=True)
            else:state['status']='finished'
        except BaseException as exc:
            state.update(status='needs_inspection',error_kind=type(exc).__name__,safe_detail=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else 'Details withheld')
        finally:
            state['updated_at']=now();save_json(OUTPUT/'progress.json',state)
    print(json.dumps(state),flush=True)
    if state['status']!='finished':raise SystemExit(1)


if __name__=='__main__':main()
