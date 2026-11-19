"""Make browser-compatible review media, preserving whole originals separately."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import time

from .growth_workflow import command, save_json, sha_file


def probe(path, count=False):
    args=['ffprobe','-v','error']
    if count:args+=['-count_frames']
    return json.loads(command(args+['-show_entries','format=start_time,duration:stream=codec_type,codec_name,nb_read_frames','-of','json',path],timeout=240))


def prepare(batch,row):
    source=batch/row['cache']
    if sha_file(source)!=row['sha256']:raise RuntimeError('Playback original checksum mismatch')
    info=probe(source)
    videos=[s for s in info['streams'] if s['codec_type']=='video']
    audio=[s for s in info['streams'] if s['codec_type']=='audio']
    if videos[0]['codec_name']=='h264' and (not audio or audio[0]['codec_name']=='aac'):
        return {'source_sha256':row['sha256'],'playback':row['cache'],'source_to_playback_offset_seconds':0,
                'kind':'original_browser_compatible','frame_count_verified_by_main_pipeline':True}
    ledger=json.loads((batch/'ledgers'/(row['id']+'.json')).read_text(encoding='utf-8'))
    folder=batch/'playback';folder.mkdir(exist_ok=True)
    target=folder/(row['id']+'.mp4')
    if target.exists():raise FileExistsError('Unregistered playback artifact; preserve it and inspect before regenerating')
    temporary=folder/(row['id']+'.partial.mp4')
    args=['ffmpeg','-y','-nostdin','-v','error','-copyts','-start_at_zero','-i',source,
          '-map','0:v:0','-map','0:a:0?','-vf','scale=1280:720:force_original_aspect_ratio=decrease,setsar=1',
          '-c:v','h264_nvenc','-preset','p4','-cq','21','-pix_fmt','yuv420p','-fps_mode','passthrough',
          '-c:a','aac','-ar','48000','-ac','2','-b:a','160k','-video_track_timescale','1000000',
          '-movflags','+faststart',temporary]
    command(args,timeout=900)
    output=probe(temporary,True)
    video=next(s for s in output['streams'] if s['codec_type']=='video')
    if int(video['nb_read_frames'])!=len(ledger):raise RuntimeError('Review encode changed the source video frame count')
    command(['ffmpeg','-nostdin','-v','error','-xerror','-i',temporary,'-map','0:v:0','-map','0:a:0?','-f','null','-'],timeout=240)
    first_frames=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-read_intervals','%+1',
        '-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',temporary]))['frames']
    shift=float(first_frames[0]['best_effort_timestamp_time'])-ledger[0]['media_seconds']
    temporary.replace(target)
    return {'source_sha256':row['sha256'],'playback':target.relative_to(batch).as_posix(),
            'source_to_playback_offset_seconds':shift,'kind':'derived_h264_aac_review_copy',
            'observed_first_frame_shift_verified':True,
            'video_source_frames':len(ledger),'video_output_frames':int(video['nb_read_frames']),
            'audio_video_full_decode_passed':True,'playback_sha256':sha_file(target),
            'original_retained':True,'audio_video_relative_timing_preserved':'ffmpeg copyts/start_at_zero, one shared origin shift',
            'not_a_lossless_original':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args();batch=Path(args.batch).resolve();output=batch/'playback_map.json'
    data=json.loads(output.read_text(encoding='utf-8')) if output.exists() else {'files':{}}
    start=time.monotonic();save_json(batch/'playback_runtime.json',{'status':'running'})
    try:
        while True:
            db=sqlite3.connect('file:'+(batch/'queue.sqlite3').as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
            rows=[dict(r) for r in db.execute('SELECT * FROM files')];db.close()
            for row in rows:
                if row['id'] in data['files']:
                    if data['files'][row['id']]['source_sha256']!=row['sha256']:raise RuntimeError('Stale playback source')
                    continue
                if not row['cache'] or not (batch/'ledgers'/(row['id']+'.json')).exists():continue
                result=prepare(batch,row);data['files'][row['id']]=result;save_json(output,data)
                print(json.dumps({'file':row['relative'],'playback_kind':result['kind'],'frames':result.get('video_output_frames')}),flush=True)
            if len(data['files'])==len(rows):break
            if time.monotonic()-start>=args.wait_seconds:break
            time.sleep(10)
        save_json(batch/'playback_runtime.json',{'status':'finished' if len(data['files'])==len(rows) else 'pending_source_preparation','files':len(data['files']),'expected':len(rows)})
    except BaseException:
        save_json(batch/'playback_runtime.json',{'status':'error','files':len(data['files'])});raise


if __name__=='__main__':main()
