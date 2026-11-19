"""Publish this one frozen Kimi batch while it runs; no cloud requests."""
from datetime import datetime
import json
from pathlib import Path
import time

from publish_kimi_growth_review import publish, ROOT
from cctv_ai.growth_library import build_library
from cctv_ai.growth_workflow import save_json
from cctv_ai.pipeline import PipelineLock

MASTER=ROOT/'output/growth-kimi-current-batch-20261103'
REVIEW=ROOT/'output/growth-kimi-review-20261103'


def main():
    plan=json.loads((MASTER/'plan.json').read_text(encoding='utf-8'))
    deadline=datetime.fromisoformat(plan['quota_window_end'])
    previous=None
    with PipelineLock(REVIEW/'publisher.lock',label='publish current Kimi batch'):
        while True:
            state=json.loads((MASTER/'progress.json').read_text(encoding='utf-8'))
            signature=[]
            for item in plan['files']:
                for path in sorted(Path(item['output']).glob('window-*.json')):
                    stat=path.stat();signature.append((str(path),stat.st_mtime_ns,stat.st_size))
            marker=(state['status'],state['files_completed'],tuple(signature))
            if marker!=previous:
                result=publish()
                catalog=ROOT/'output/growth_review_catalog.json'
                if catalog.exists():build_library(ROOT/'output/growth-first-batch-20261103',catalog)
                previous=marker
                save_json(REVIEW/'live_publication.json',{'status':'following_current_batch',
                    'published_windows':result['windows_completed'],'total_windows':result['windows_total'],
                    'master_status':state['status'],'cloud_requests':0})
                print(json.dumps({'published_windows':result['windows_completed'],'total_windows':result['windows_total'],'master_status':state['status']}),flush=True)
            if state['status']!='running' or datetime.now(deadline.tzinfo)>=deadline:
                save_json(REVIEW/'live_publication.json',{'status':'finished_following',
                    'master_status':state['status'],'cloud_requests':0})
                break
            time.sleep(10)


if __name__=='__main__':main()
