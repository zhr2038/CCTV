"""Publish existing Kimi labels to an independent, local-only review queue.

Reads only local cached originals and worker artifacts. No cloud calls, NAS
access, worker progress writes, credentials, or server startup are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
from zoneinfo import ZoneInfo

import av
from cctv_ai.growth_workflow import (NEUTRAL_PROMPT, connect, dumps, image_bytes,
    meta, report, save_json, sha_file, validate_tags)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/growth-first-batch-20261103'
MASTER = ROOT / 'output/growth-kimi-current-batch-20261103/plan.json'
OUTPUT = ROOT / 'output/growth-kimi-review-20261103'
MODEL = 'kimi-code/k3.1-pro'
PROMPT = NEUTRAL_PROMPT + '\n这些是当前完整原片中一个连续短窗的全部主视频解码帧，主图最大边512，最后3张补图最大边1024。只返回JSON，不调用工具、不读其他文件、不执行命令、不写文件。'
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def time_windows(ledger, duration):
    specs = []; first = 0
    while first < len(ledger):
        begin = ledger[first]['media_seconds']; stop = begin + 6
        last = first + 1
        while last < len(ledger) and ledger[last]['media_seconds'] < stop:last += 1
        specs.append({'ordinal':len(specs), 'first_frame':first, 'last_frame':last-1,
                      'start':begin, 'end':min(stop,duration)})
        if last == len(ledger):break
        following = first+1
        while following < last and ledger[following]['media_seconds'] < stop-1:following += 1
        first = following
    return specs


def prepare_original(row, ledger, specs):
    source = (SOURCE / row['cache']).resolve()
    if not source.is_relative_to(SOURCE) or sha_file(source) != row['sha256']:
        raise ValueError('Local source copy checksum mismatch')
    destination = OUTPUT / 'originals' / (row['id']+'.mp4')
    if not destination.exists():
        temporary = destination.with_suffix('.copying')
        shutil.copyfile(source, temporary)
        if sha_file(temporary) != row['sha256']:raise ValueError('Copied original checksum mismatch')
        temporary.replace(destination)
    if sha_file(destination) != row['sha256']:raise ValueError('Review original checksum mismatch')
    digest_path = OUTPUT / 'frame_audits' / (row['id']+'.json')
    ledger_sha = hashlib.sha256(dumps(ledger).encode()).hexdigest()
    poster_indices = {(s['first_frame']+s['last_frame'])//2:s['ordinal'] for s in specs}
    if digest_path.exists():
        audit = load(digest_path)
        if audit['source_sha256'] != row['sha256'] or audit['ledger_sha256'] != ledger_sha:
            raise ValueError('Previously verified frame audit differs')
        return audit['frame512_sha256']
    digests = []
    with av.open(str(destination)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(ledger) or frame.pts != ledger[index]['pts']:
                raise ValueError('Decoded frame order differs from source ledger')
            jpeg = image_bytes(frame.to_image(), 512)
            digests.append(hashlib.sha256(jpeg).hexdigest())
            if index in poster_indices:
                (OUTPUT/'posters'/(row['id']+f'-{poster_indices[index]:05d}.jpg')).write_bytes(jpeg)
    if len(digests) != len(ledger):raise ValueError('Decoded frame count differs from source ledger')
    save_json(digest_path, {'source_sha256':row['sha256'], 'ledger_sha256':ledger_sha,
        'frame512_sha256':digests, 'basis':'independent decode of hash-verified complete local original'})
    save_json(OUTPUT/'ledgers'/(row['id']+'.json'), ledger)
    return digests


def completed_label(record, row, spec, ledger, digests):
    if record.get('source_sha256') != row['sha256'] or record.get('model') != MODEL:
        raise ValueError('Kimi label source/model differs')
    if record.get('recipe_prompt_sha256') != PROMPT_SHA:
        raise ValueError('Kimi label prompt differs')
    for key in ['ordinal','first_frame','last_frame','start','end']:
        if abs(record.get(key,-100)-spec[key]) > 0.000001:
            raise ValueError('Kimi label time/frame boundary differs')
    audit = record.get('submitted_frame_ledger',[])
    indices = list(range(spec['first_frame'],spec['last_frame']+1))
    if record.get('input_frames') != len(indices) or [f.get('source_frame') for f in audit] != indices:
        raise ValueError('Kimi submitted-frame audit has a coverage hole or duplicate')
    for index, item in zip(indices,audit):
        if item.get('sha256') != digests[index] or abs(item.get('media_seconds',-100)-ledger[index]['media_seconds']) > 0.000001:
            raise ValueError('Kimi input frame differs from independently decoded source')
    return validate_tags(dumps(record['tags']))


def publish():
    plan = load(MASTER)
    if plan.get('model') != MODEL or plan.get('window_seconds') != 6 or plan.get('overlap_seconds') != 1:
        raise ValueError('Unexpected master model or window recipe')
    if len(plan['files']) > 11 or sum(f['duration'] for f in plan['files']) > 360:
        raise ValueError('Frozen short-batch bound exceeded')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name in ['originals','posters','frame_audits','ledgers']:(OUTPUT/name).mkdir(exist_ok=True)
    source_db = sqlite3.connect('file:'+(SOURCE/'queue.sqlite3').as_posix()+'?mode=ro',uri=True)
    source_db.row_factory = sqlite3.Row
    source_manifest = meta(source_db,'manifest')
    rows = {r['id']:dict(r) for r in source_db.execute('SELECT * FROM files')}
    source_db.close()
    recipe = {'workflow':{'model':MODEL,'image_max_edge':512,'detail_image_max_edge':1024,
        'prompt_style':'neutral_observer_v2_kimi','window_seconds':6,'overlap_seconds':1},
        'prompt':PROMPT,'prompt_sha256':PROMPT_SHA,'audio':{'source':'existing local YAMNet evidence; Kimi receives no audio'}}
    recipe_hash = hashlib.sha256(dumps(recipe).encode()).hexdigest()
    plan_sha = hashlib.sha256(dumps(plan).encode()).hexdigest()
    db = connect(OUTPUT)
    previous = meta(db,'publisher_master_plan_sha256')
    if previous and previous != plan_sha:raise ValueError('Frozen master plan changed')
    old_recipe = meta(db,'recipe_hash')
    if old_recipe and old_recipe != recipe_hash:raise ValueError('Published recipe changed; preserve this review queue')
    manifest = dict(source_manifest)
    cameras = sorted({rows[f['id']]['camera'] for f in plan['files']})
    manifest.update(recipe_hash=recipe_hash, selection_mode='explicit_whole_files',
        caption='Kimi当前批次 · 6秒连续标签', camera_ids=cameras,
        excluded_camera_ids=sorted(set(source_manifest['camera_ids']+source_manifest.get('excluded_camera_ids',[]))-set(cameras)),
        selected_sources=[{'relative_path':f['relative'],'source_sha256':f['sha256']} for f in plan['files']],
        scope_rule='11 explicit complete local originals; no annual or intervening-time coverage implied',
        scope_note='Kimi vision labels only; no human truth, no training authorization, no source deletion',
        window_seconds=6, overlap_seconds=1, source_read_only=True, source_deletions=0,
        scan_issues=[],full_family_or_annual_recall_verified=False)
    for key,value in [('manifest',manifest),('recipe',recipe),('recipe_hash',recipe_hash),
                      ('publisher_master_plan_sha256',plan_sha)]:meta(db,key,value)
    issues = []; expected_windows = 0
    try:
        for selected in plan['files']:
            row = rows[selected['id']]
            if (row['relative'],row['sha256'],row['frames'],row['duration']) != (
                selected['relative'],selected['sha256'],selected['frames'],selected['duration']):
                raise ValueError('Master file differs from verified source metadata')
            ledger = load(SOURCE/'ledgers'/(row['id']+'.json'))
            specs = time_windows(ledger,row['duration']);expected_windows += len(specs)
            digests = prepare_original(row,ledger,specs)
            work = Path(selected['output']).resolve()
            if not work.is_relative_to(ROOT/'output'):raise ValueError('Worker output is outside local experiment tree')
            worker_plan = work/'plan.json'
            if worker_plan.exists():
                wp=load(worker_plan)
                if wp['source_sha256'] != row['sha256'] or wp['windows'] != specs:
                    raise ValueError('Worker plan differs from expected source/window boundaries')
            copied = dict(row);copied['cache']='originals/'+row['id']+'.mp4';copied['status']='pending';copied['error']=None
            fields=list(copied)
            db.execute('INSERT INTO files('+','.join(fields)+') VALUES('+','.join('?' for _ in fields)+') ON CONFLICT(id) DO UPDATE SET '+','.join(k+'=excluded.'+k for k in fields if k!='id'),[copied[k] for k in fields])
            statuses=[]
            for spec in specs:
                record_path=work/f"window-{spec['ordinal']:03d}.json"
                record=load(record_path) if record_path.exists() else {}
                status='pending';tags=None;error=None;usage=None;seconds=None
                if record.get('status')=='completed':
                    try:
                        tags=completed_label(record,row,spec,ledger,digests);status='completed'
                        usage=record.get('session_usage');seconds=record.get('wait_seconds_this_run')
                    except (ValueError,KeyError,TypeError) as exc:
                        status='error';error=str(exc);issues.append({'file_id':row['id'],'window':spec['ordinal'],'error':error})
                elif record.get('status') in {'error','needs_inspection'}:
                    status='error';error='Worker requires inspection; no completed label accepted'
                statuses.append(status)
                identity=row['id']+f"-{spec['ordinal']:05d}"
                digest=hashlib.sha256(dumps({'source_sha256':row['sha256'],'recipe_hash':recipe_hash,'window':spec}).encode()).hexdigest()
                values=(identity,row['id'],spec['ordinal'],spec['first_frame'],spec['last_frame'],spec['start'],spec['end'],spec['last_frame']-spec['first_frame']+1,digest,status,dumps(tags) if tags else None,dumps(usage) if usage is not None else None,seconds,error,'posters/'+identity+'.jpg')
                db.execute('''INSERT INTO windows(id,file_id,ordinal,first_frame,last_frame,start,end,frame_count,input_digest,status,tags,usage,seconds,error,poster)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,tags=excluded.tags,usage=excluded.usage,seconds=excluded.seconds,error=excluded.error''',values)
            file_status='completed' if statuses and all(s=='completed' for s in statuses) else ('error' if 'error' in statuses else 'pending')
            db.execute('UPDATE files SET status=? WHERE id=?',(file_status,row['id']))
            db.commit()
    finally:
        db.close()
    report(OUTPUT)
    progress=load(OUTPUT/'progress.json')
    if progress['windows_total'] != expected_windows:raise ValueError('Published queue contains an unexpected window')
    validation={'published_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'master_plan_sha256':plan_sha,'recipe_hash':recipe_hash,'local_cached_source_only':True,
        'copied_originals_all_sha256_verified':True,'independent_decoded_frame_hashes_checked':True,
        'files_total':progress['files_total'],'files_by_status':progress['files_by_status'],
        'windows_total':expected_windows,'windows_completed':progress['windows_completed'],
        'main_source_frames_in_completed_windows':progress['unique_frames_in_completed_requests'],
        'manual_review_rows_preserved':True,'ai_labels_are_not_human_truth':True,
        'source_deletions':0,'cloud_requests_made':0,'worker_progress_modified':False,
        'issues':issues}
    save_json(OUTPUT/'publication_validation.json',validation)
    return validation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch-seconds',type=int,default=0)
    args=parser.parse_args()
    if not 0 <= args.watch_seconds <= 180:raise ValueError('Watch bound is 0..180 seconds')
    deadline=time.monotonic()+args.watch_seconds
    while True:
        result=publish()
        print(dumps({k:result[k] for k in ['files_total','files_by_status','windows_total','windows_completed','main_source_frames_in_completed_windows']}),flush=True)
        if time.monotonic()>=deadline:break
        time.sleep(min(10,max(0,deadline-time.monotonic())))


if __name__=='__main__':main()
