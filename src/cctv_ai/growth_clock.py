"""Read image-clock anchors from frozen bedside originals, without interpolation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time

from .config import load_config
from .growth_workflow import frame_windows, save_json, sha_file
from .watermark import parse_watermark_text


def map_file(batch, row, cfg, engine):
    import av
    import numpy as np
    batch=Path(batch)
    original=batch/row['cache']
    if sha_file(original)!=row['sha256']:
        raise RuntimeError('Clock source checksum mismatch')
    project=load_config(cfg['workflow']['camera_config'])
    camera=next(c for c in project.cameras if c.id==row['camera'])
    ledger=json.loads((batch/'ledgers'/(row['id']+'.json')).read_text(encoding='utf-8'))
    ranges=list(frame_windows(len(ledger),cfg['workflow']['max_frames'],cfg['workflow']['overlap_frames']))
    targets=sorted({index for first,last in ranges for index in (first,(first+last-1)//2,last-1)})
    target_set=set(targets)
    observations={}
    crop_root=batch/'clock_crops'/row['id']
    crop_root.mkdir(parents=True,exist_ok=True)
    x,y,width,height=camera.watermark_crop
    with av.open(str(original)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index not in target_set:
                continue
            if frame.pts!=ledger[index]['pts']:
                raise RuntimeError('Clock frame order differs from source ledger')
            crop=frame.to_image().convert('RGB').crop((x,y,x+width,y+height))
            evidence=crop_root/f'{index:07d}.jpg'
            crop.save(evidence,quality=93)
            result=engine(np.asarray(crop),use_det=False,use_cls=False,use_rec=True)
            text=result.txts[0] if result.txts else ''
            confidence=float(result.scores[0]) if result.scores else 0
            observation={'frame_index':index,'media_seconds':ledger[index]['media_seconds'],
                         'text':text,'confidence':confidence,'crop':evidence.relative_to(batch).as_posix(),'wall_time':None}
            if confidence>=0.94:
                try:observation['wall_time']=parse_watermark_text(text,project.timezone).isoformat()
                except ValueError:pass
            observations[index]=observation
    windows={}
    for ordinal,(first,last) in enumerate(ranges):
        anchors=[observations[i] for i in (first,(first+last-1)//2,last-1)]
        valid=[a for a in anchors if a['wall_time']]
        residuals=[]
        for a,b in zip(valid,valid[1:]):
            wall_delta=(datetime.fromisoformat(b['wall_time'])-datetime.fromisoformat(a['wall_time'])).total_seconds()
            residuals.append(wall_delta-(b['media_seconds']-a['media_seconds']))
        windows[row['id']+f'-{ordinal:05d}']={'anchors':anchors,'recognized':len(valid),
            'jump_observed_between_anchors':any(abs(v)>2 for v in residuals),
            'wall_minus_media_delta_seconds':residuals,'full_window_clock_continuity_verified':False}
    return {'source_sha256':row['sha256'],'source_relative_path':row['relative'],
            'frame_scope':'first, middle and last frame of every planned model window',
            'full_timestamp_timeline_verified':False,'model':'RapidOCR PP-OCRv6_rec_small local',
            'observations':list(observations.values()),'windows':windows,
            'recognized':sum(bool(o['wall_time']) for o in observations.values()),'requested':len(targets)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args()
    batch=Path(args.batch).resolve()
    model_root=Path(__import__('rapidocr').__file__).resolve().parent/'models'
    paths={name:model_root/file for name,file in [('Rec','PP-OCRv6_rec_small.onnx'),('Det','PP-OCRv6_det_small.onnx'),('Cls','ch_ppocr_mobile_v2.0_cls_mobile.onnx')]}
    if not all(p.is_file() for p in paths.values()):
        raise FileNotFoundError('Existing local OCR models required; no implicit downloads')
    from rapidocr import RapidOCR
    params={f'{key}.model_path':str(path) for key,path in paths.items()}
    params.update({'EngineConfig.onnxruntime.intra_op_num_threads':2,'EngineConfig.onnxruntime.inter_op_num_threads':1,'Global.log_level':'warning'})
    engine=RapidOCR(params=params)
    result_path=batch/'clock_map.json'
    output=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'files':{}}
    start=time.monotonic()
    while True:
        db=sqlite3.connect('file:'+(batch/'queue.sqlite3').as_posix()+'?mode=ro',uri=True)
        db.row_factory=sqlite3.Row
        cfg=json.loads(db.execute("SELECT value FROM meta WHERE key='recipe'").fetchone()[0])
        rows=[dict(r) for r in db.execute('SELECT * FROM files WHERE watermark_required=1')]
        db.close()
        for row in rows:
            if row['id'] in output['files']:
                if output['files'][row['id']]['source_sha256']!=row['sha256']:
                    raise RuntimeError('Existing clock map belongs to a different original')
                continue
            if not row['cache'] or not (batch/'ledgers'/(row['id']+'.json')).is_file():
                continue
            mapped=map_file(batch,row,cfg,engine)
            output['files'][row['id']]=mapped
            save_json(result_path,output)
            print(json.dumps({'file':row['relative'],'clock_anchors_recognized':mapped['recognized'],'requested':mapped['requested'],'full_timeline_verified':False}),flush=True)
        if len(output['files'])==len(rows):
            break
        if time.monotonic()-start>=args.wait_seconds:
            print(json.dumps({'status':'pending_source_preparation','mapped_files':len(output['files']),'expected_files':len(rows)}),flush=True)
            break
        time.sleep(10)


if __name__=='__main__':
    main()
