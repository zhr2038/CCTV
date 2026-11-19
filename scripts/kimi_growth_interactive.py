"""Current-session, bounded Kimi Code labeling of one complete cached original.

Uses the already logged-in official loopback service. No NAS writes, deletion,
scheduled service, autonomous annual batch, or automatic retry is provided.
"""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import sqlite3
import time
from zoneinfo import ZoneInfo

import av
from PIL import Image
import requests
from cctv_ai.growth_workflow import NEUTRAL_PROMPT, ENUMS, image_bytes, save_json, sha_file, validate_tags

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'kimi-code/k3.1-pro'


def time_windows(ledger, duration, seconds=6, overlap=1):
    if not 0 <= overlap < seconds:raise ValueError('Invalid overlap')
    specs = []; first = 0
    while first < len(ledger):
        begin = ledger[first]['media_seconds']
        stop = begin + seconds
        last = first + 1
        while last < len(ledger) and ledger[last]['media_seconds'] < stop:last += 1
        specs.append((first,last,begin,min(stop,duration)))
        if last == len(ledger):break
        following = first+1
        while following < last and ledger[following]['media_seconds'] < stop-overlap:following += 1
        first = following
    return specs


class LocalService:
    def __init__(self):
        home = Path.home()/'.kimi-code'
        entries = [json.loads(p.read_text(encoding='utf-8')) for p in (home/'server/instances').glob('*.json')]
        instance = next(x for x in entries if x['host'] == '127.0.0.1')
        self.base = 'http://127.0.0.1:'+str(int(instance['port']))
        self.http = requests.Session();self.http.trust_env=False
        self.http.headers['Authorization'] = 'Bearer '+(home/'server.token').read_text().strip()

    def call(self, method, path, **kwargs):
        try:
            response = self.http.request(method,self.base+path,timeout=45,allow_redirects=False,**kwargs)
            if response.status_code != 200:raise RuntimeError('Official local service HTTP '+str(response.status_code))
            data = response.json()
            if data.get('code') != 0:raise RuntimeError('Official local service rejected request; message withheld')
            return data['data']
        except requests.RequestException:
            raise RuntimeError('Official local service connection failed; no automatic retry') from None

    def close(self):
        self.http.headers.pop('Authorization',None);self.http.close()


def parse_labels(answer):
    text=answer.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text);normalizations=[]
    # A structural alias, with original model text retained. Never turn an
    # uncertain judgment into a definite event or a human-reviewed label.
    for field in ENUMS:
        if value.get(field)=='uncertain':
            value[field]='unclear';normalizations.append({'field':field,'from':'uncertain','to':'unclear'})
    return validate_tags(json.dumps(value,ensure_ascii=False)),normalizations


def wait_for_labels(service,sid,target,record,runtime,output):
    start=time.monotonic()
    while time.monotonic()-start<480:
        session=service.call('GET',f'/api/v1/sessions/{sid}')
        pending=session.get('pending_interaction')
        if pending not in (None,False,'none') and pending != {}:
            raise RuntimeError('Unexpected interaction/tool request; no permission automatically granted')
        if not session.get('busy') and not session.get('main_turn_active'):
            if session.get('last_turn_reason')=='failed':raise RuntimeError('Kimi session failed or hit a usage limit; no automatic retry')
            messages=service.call('GET',f'/api/v1/sessions/{sid}/messages',params={'page_size':30})
            answers=[part['text'] for m in messages['items'] if m.get('role')=='assistant'
                for part in m.get('content',[]) if part.get('type')=='text' and isinstance(part.get('text'),str)]
            tags=None;normalizations=[];original_answer=None
            for answer in answers:
                try:
                    tags,normalizations=parse_labels(answer);original_answer=answer;break
                except (ValueError,KeyError,TypeError):continue
            if tags is None:raise RuntimeError('No valid final label JSON; do not count complete')
            record.update(status='completed',tags=tags,wait_seconds_this_run=round(time.monotonic()-start,3),
                original_final_text=original_answer,structural_normalizations=normalizations,
                session_usage=session.get('usage'),usage_caution='aggregate subscription API can return zeros; not proof of free usage')
            save_json(target,record);runtime['completed_windows']+=1
            save_json(output/'progress.json',runtime)
            print(json.dumps({'window':record['ordinal'],'status':'completed','frames':record['input_frames'],'tags':tags},ensure_ascii=True),flush=True)
            return
        time.sleep(2)
    raise RuntimeError('Kimi still running after bounded wait; preserve session without duplicate submission')


def run(args):
    source_batch = Path(args.source_batch).resolve()
    output = Path(args.output).resolve()
    if not source_batch.is_relative_to(ROOT/'output') or not output.is_relative_to(ROOT/'output'):
        raise ValueError('Use local existing experiment folders, outside NAS')
    output.mkdir(parents=True,exist_ok=True)
    db = sqlite3.connect('file:'+(source_batch/'queue.sqlite3').as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    row = db.execute('SELECT * FROM files WHERE id=?',(args.file_id,)).fetchone();db.close()
    if not row or row['status'] != 'completed':raise ValueError('Choose one already verified complete original')
    row = dict(row);original = source_batch/row['cache']
    if sha_file(original) != row['sha256']:raise ValueError('Whole original checksum mismatch')
    ledger = json.loads((source_batch/'ledgers'/(row['id']+'.json')).read_text(encoding='utf-8'))
    specs = time_windows(ledger,row['duration'])
    assert {i for first,last,_,_ in specs for i in range(first,last)} == set(range(len(ledger)))
    plan = {'model':MODEL,'source_batch':str(source_batch),'source_relative':row['relative'],
        'file_id':row['id'],'source_sha256':row['sha256'],'source_frames':len(ledger),
        'source_recorded_seconds':row['duration'],'main_image_max_edge':512,'supplementary_image_max_edge':1024,
        'window_seconds':6,'overlap_seconds':1,'windows':[
            {'ordinal':i,'first_frame':f,'last_frame':l-1,'start':s,'end':e} for i,(f,l,s,e) in enumerate(specs)],
        'scope':'one complete cached original in the current interactive user task; not a background annual batch',
        'audio_submitted':False,'source_deletions':0,'all_recording_tracks_preserved_in_complete_original':True,
        'frame_input_scope':'first/main video stream, reduced resolution',
        'important_event_recall_verified':False,'model_labels_are_not_human_training_truth':True}
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Immutable plan differs; preserve it and choose a different output')
    save_json(plan_path,plan)
    proxy = output/'frames512';detail = output/'details1024'
    proxy.mkdir(exist_ok=True);detail.mkdir(exist_ok=True)
    detail_indices = {x for f,l,_,_ in specs for x in (f,(f+l-1)//2,l-1)}
    with av.open(str(original)) as media:
        count=0
        for i,frame in enumerate(media.decode(video=0)):
            if frame.pts != ledger[i]['pts']:raise ValueError('Source frame order changed')
            im=frame.to_image();(proxy/f'{i:06d}.jpg').write_bytes(image_bytes(im,512))
            if i in detail_indices:(detail/f'{i:06d}.jpg').write_bytes(image_bytes(im,1024))
            count=i+1
        if count != len(ledger):raise ValueError('Source decode count changed')
    service=LocalService()
    runtime={'status':'running','file_id':row['id'],'completed_windows':0,'planned_windows':len(specs),
             'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'current_session':None}
    save_json(output/'progress.json',runtime)
    try:
        for ordinal,(first,last,begin,end) in enumerate(specs):
            target=output/f'window-{ordinal:03d}.json'
            if target.exists():
                existing=json.loads(target.read_text(encoding='utf-8'))
                if existing.get('status')=='completed':
                    runtime['completed_windows']+=1;continue
                if existing.get('status')=='submitted' and existing.get('session_id') and existing.get('prompt_id'):
                    runtime['current_session']=existing['session_id'];runtime['current_window']=ordinal
                    save_json(output/'progress.json',runtime)
                    print(json.dumps({'window':ordinal,'status':'resuming_existing_session_no_resubmission'}),flush=True)
                    wait_for_labels(service,existing['session_id'],target,existing,runtime,output)
                    continue
                raise RuntimeError('Prior window is unfinished; inspect it rather than automatically resubmit')
            content=[];audit=[]
            for i in range(first,last):
                path=proxy/f'{i:06d}.jpg'
                content += [{'type':'text','text':f'source_frame={i},source_media_seconds={ledger[i]["media_seconds"]:.6f}'},
                    {'type':'image','source':{'kind':'path','path':str(path)},'name':path.name}]
                audit.append({'source_frame':i,'media_seconds':ledger[i]['media_seconds'],'sha256':sha_file(path)})
            for i in (first,(first+last-1)//2,last-1):
                path=detail/f'{i:06d}.jpg'
                content += [{'type':'text','text':f'supplementary_detail_frame={i}'},
                    {'type':'image','source':{'kind':'path','path':str(path)},'name':'detail-'+path.name}]
            text=NEUTRAL_PROMPT+'\n这些是当前完整原片中一个连续短窗的全部主视频解码帧，主图最大边512，最后3张补图最大边1024。只返回JSON，不调用工具、不读其他文件、不执行命令、不写文件。'
            content.append({'type':'text','text':text})
            record={'status':'prepared','ordinal':ordinal,'first_frame':first,'last_frame':last-1,
                'start':begin,'end':end,'source_sha256':row['sha256'],'model':MODEL,
                'input_frames':last-first,'submitted_frame_ledger':audit,'human_gold':False,
                'recipe_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'audio_submitted':False}
            session=service.call('POST','/api/v1/sessions',json={'title':'Growth film current interactive labeling '+str(ordinal),
                'metadata':{'cwd':str(output)}});sid=session['id']
            record['session_id']=sid;save_json(target,record)
            service.call('POST',f'/api/v1/sessions/{sid}/profile',json={'agent_config':{
                'model':MODEL,'thinking':'high','tools':[],'mcp_servers':[],'plan_mode':False,
                'swarm_mode':False,'tower_mode':False,'permission_mode':'manual'}})
            status=service.call('GET',f'/api/v1/sessions/{sid}/status')
            if status.get('model') != MODEL:raise RuntimeError('Pinned K3.1 Pro profile was not applied')
            record['status']='submitted';save_json(target,record)
            reply=service.call('POST',f'/api/v1/sessions/{sid}/prompts',json={'content':content})
            record['prompt_id']=reply['prompt_id'];save_json(target,record)
            runtime['current_session']=sid;runtime['current_window']=ordinal;save_json(output/'progress.json',runtime)
            print(json.dumps({'window':ordinal,'status':'submitted','frames':last-first,'start':begin,'end':end}),flush=True)
            wait_for_labels(service,sid,target,record,runtime,output)
        runtime.update(status='finished',unique_main_frames_covered=len(ledger),
            all_main_stream_frames_submitted=True,important_event_recall_verified=False)
    except BaseException as exc:
        runtime.update(status='needs_inspection',error_kind=type(exc).__name__)
        print(json.dumps({'status':'needs_inspection','error_kind':type(exc).__name__,
            'safe_detail':str(exc) if isinstance(exc,(ValueError,RuntimeError)) else 'Details withheld'}),flush=True)
    finally:
        service.close();save_json(output/'progress.json',runtime)
    if runtime['status']!='finished':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-batch',required=True)
    parser.add_argument('--file-id',required=True);parser.add_argument('--output',required=True)
    run(parser.parse_args())
