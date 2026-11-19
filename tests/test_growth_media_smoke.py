"""Exercise the source->coverage->review->film loop on synthetic media only."""
from argparse import Namespace
from contextlib import contextmanager, redirect_stdout, closing
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from cctv_ai.growth_workflow import plan, plan_files, run, settings, sha_file, import_review
from cctv_ai.growth_render import render_reviewed, merge_keeps, subtract_exclusions


class GrowthMediaSmoke(unittest.TestCase):
    def test_explicit_not_for_film_wins_only_for_same_source(self):
        keep = {'source_sha256':'source','media_start_seconds':0,'media_end_seconds':12}
        excluded = [{'source_sha256':'source','media_start_seconds':5,'media_end_seconds':7},
                    {'source_sha256':'other','media_start_seconds':0,'media_end_seconds':20}]
        parts = subtract_exclusions([keep],excluded)
        self.assertEqual([(p['media_start_seconds'],p['media_end_seconds']) for p in parts],[(0,5),(7,12)])

    def test_merge_overlapping_keeps_preserves_provenance(self):
        base = {'source_sha256':'source','capture_hint':'2026-01-01','source_relative_path':'a.mp4'}
        merged = merge_keeps([{**base,'media_start_seconds':0,'media_end_seconds':6,'reviewed_window_ids':['a']},
                              {**base,'media_start_seconds':5,'media_end_seconds':12,'reviewed_window_ids':['b']},
                              {**base,'media_start_seconds':15,'media_end_seconds':18,'reviewed_window_ids':['c']}])
        self.assertEqual([(x['media_start_seconds'],x['media_end_seconds']) for x in merged], [(0,12),(15,18)])
        self.assertEqual(merged[0]['reviewed_window_ids'], ['a','b'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'media tools required')
    def test_resume_review_and_real_render_without_touching_source(self):
        try:
            import av
        except ImportError:
            self.skipTest('PyAV required')
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = root/'source'
            camera = 'lumi1.54ef4482e358'
            original = source/'aqara_video'/camera/'20250101'/'010203.mp4'
            original.parent.mkdir(parents=True)
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=10:duration=2',
                            '-f','lavfi','-i','sine=frequency=500:sample_rate=16000:duration=2',
                            '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(original)], check=True)
            os.utime(original, (time.time()-7200,time.time()-7200))
            before = (sha_file(original), original.stat().st_size, original.stat().st_mtime_ns)
            cfg = settings(str(Path(__file__).resolve().parents[1]/'config/growth.toml'))
            cfg = copy.deepcopy(cfg)
            cfg['workflow'].update(max_frames=12,overlap_frames=2)
            cfg['audio']['enabled'] = False
            cfg['recipe_hash'] = 'synthetic-media-smoke-only'
            batch = root/'batch'
            selection_file=root/'selection.json'
            selection_file.write_text(json.dumps({'files':[{'camera_id':camera,'relative_path':original.relative_to(source).as_posix()}]}),encoding='utf-8')
            explicit=root/'explicit'
            plan_files(Namespace(batch=str(explicit),root=str(source),selection=str(selection_file)),cfg)
            explicit_progress=json.loads((explicit/'progress.json').read_text(encoding='utf-8'))
            self.assertEqual(explicit_progress['files_total'],1)
            self.assertEqual(explicit_progress['manifest']['selection_mode'],'explicit_whole_files')
            selection_file.write_text(json.dumps({'files':[{'camera_id':camera,'relative_path':'../escape.mp4'}]}),encoding='utf-8')
            with self.assertRaises(ValueError):
                plan_files(Namespace(batch=str(root/'unsafe'),root=str(source),selection=str(selection_file)),cfg)
            self.assertFalse((root/'unsafe').exists())
            plan(Namespace(batch=str(batch),root=str(source),since='2024-12-31T17:00:00+00:00',until='2024-12-31T18:00:00+00:00',camera=[camera]), cfg)
            calls = []
            tags = {'child_visibility':'not_seen','adult_interaction':'no','preview_fit':'not_suitable',
                    'private_care':'no','observed_actions':[],'uncertain':[],'reason':'Synthetic calibration image'}
            class Response:
                def raise_for_status(self): pass
                def json(self): return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(tags)}}],'usage':{}}
            class Session:
                def post(self,url,**kwargs):
                    calls.append(kwargs['json'])
                    return Response()
            @contextmanager
            def backend(*args):
                yield Session(), 'http://synthetic-test-only'
            args = Namespace(batch=str(batch),limit_files=5,max_cache_gb=1)
            with patch('cctv_ai.growth_workflow.model_server', backend):
                self.assertEqual(run(args,cfg),0)
                first_count = len(calls)
                self.assertEqual(run(args,cfg),0)
            self.assertEqual(first_count,2)
            self.assertEqual(len(calls),first_count)
            result = json.loads((batch/'review_data.json').read_text(encoding='utf-8'))
            self.assertEqual(result['summary']['unique_frames_in_completed_requests'],20)
            self.assertTrue(result['summary']['full_scope_processing_complete'])
            self.assertFalse(result['summary']['important_event_recall_verified'])
            review_file = root/'synthetic-review.json'
            review_file.write_text(json.dumps({'recipe_hash':cfg['recipe_hash'],'reviews':[
                {'window_id':w['id'],'source_sha256':before[0],'verdict':'keep','note':'synthetic test'}
                for w in result['windows']]}), encoding='utf-8')
            import_review(Namespace(batch=str(batch),file=str(review_file)))
            with closing(sqlite3.connect(batch/'queue.sqlite3')) as db, db:
                db.execute("UPDATE windows SET status='error' WHERE ordinal=0")
                db.execute("UPDATE files SET status='error'")
            with patch('cctv_ai.growth_workflow.model_server', backend):
                self.assertEqual(run(args,cfg),0)
            self.assertEqual(len(calls),first_count+1)
            with closing(sqlite3.connect(batch/'queue.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM reviews').fetchone()[0],2)
            output = root/'film'
            render_reviewed([str(batch)],str(output))
            edit = json.loads((output/'edit.json').read_text(encoding='utf-8'))
            self.assertEqual(len(edit['edits']),1)
            self.assertEqual(len(edit['edits'][0]['reviewed_window_ids']),2)
            self.assertTrue(edit['full_movie_decode_passed'])
            self.assertGreater(edit['duration_seconds'],1.9)
            self.assertEqual(sha_file(output/edit['edits'][0]['protected_full_original']),before[0])
            self.assertEqual((sha_file(original),original.stat().st_size,original.stat().st_mtime_ns),before)


if __name__ == '__main__':
    unittest.main()
