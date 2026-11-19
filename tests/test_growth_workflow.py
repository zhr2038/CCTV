from pathlib import Path
import json
import sqlite3
import tempfile
import unittest

from cctv_ai.growth_workflow import frame_windows, validate_tags, safe_output, connect, coverage, import_review, meta


class GrowthWorkflowTests(unittest.TestCase):
    def test_window_boundaries_do_not_drop_short_tail_or_overlap_only(self):
        for count in (1, 16, 127, 128, 129, 240, 241, 1007):
            ranges = list(frame_windows(count, 128, 16))
            frames = {i for first, last in ranges for i in range(first, last)}
            self.assertEqual(frames, set(range(count)))
            self.assertEqual(ranges[-1][1], count)
            self.assertTrue(all(last > first for first, last in ranges))

    def test_output_cannot_be_recording_directory_or_ancestor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for unsafe in (root, root / "cameras", root / "cameras" / "output"):
                with self.assertRaises(ValueError):
                    safe_output(unsafe, root / "cameras")
            safe_output(root / "work", root / "cameras")

    def test_incomplete_model_json_is_not_a_success(self):
        with self.assertRaises(ValueError):
            validate_tags('{"child_visibility":"clear"}')
        value = {"child_visibility": "unclear", "adult_interaction": "unclear",
                 "preview_fit": "unclear", "private_care": "unclear", "reason": "看不清",
                 "observed_actions": [], "uncertain": ["遮挡"]}
        self.assertEqual(validate_tags('```json\n'+json.dumps(value)+'\n```'), value)

    def test_failed_window_keeps_coverage_hole(self):
        with tempfile.TemporaryDirectory() as folder:
            db = connect(Path(folder))
            db.execute("INSERT INTO files(id,camera,name,relative,capture_hint,time_source,watermark_required,size,mtime_ns) VALUES('f','cam','cam','source.mp4','date','hint',0,1,1)")
            for identity, first, last, status in (("one", 0, 127, "completed"), ("two", 112, 239, "error"), ("three", 224, 250, "completed")):
                db.execute("INSERT INTO windows(id,file_id,ordinal,first_frame,last_frame,start,end,frame_count,input_digest,status) VALUES(?,'f',?,?,?,?,?,?,?,?)", (identity, {"one":0,"two":1,"three":2}[identity], first,last,0,1,last-first+1,'digest',status))
            self.assertEqual(coverage(db, 'f', 251), (155, False))
            db.execute("UPDATE windows SET status='completed' WHERE id='two'")
            self.assertEqual(coverage(db, 'f', 251), (251, True))
            db.close()

    def test_review_import_is_atomic_on_source_mismatch(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            db = connect(batch)
            meta(db, 'manifest', {'recipe_hash': 'recipe'})
            db.execute("INSERT INTO files(id,camera,name,relative,capture_hint,time_source,watermark_required,size,mtime_ns,sha256) VALUES('f','cam','cam','source.mp4','date','hint',0,1,1,'actual')")
            db.execute("INSERT INTO windows(id,file_id,ordinal,first_frame,last_frame,start,end,frame_count,input_digest,status) VALUES('one','f',0,0,1,0,1,2,'digest','completed')")
            db.commit()
            review = {'recipe_hash':'recipe','reviews':[
                {'window_id':'one','source_sha256':'actual','verdict':'keep'},
                {'window_id':'one','source_sha256':'wrong','verdict':'not_for_film'}]}
            path = batch / 'review.json'
            path.write_text(json.dumps(review), encoding='utf-8')
            with self.assertRaises(ValueError):
                import_review(Namespace(batch=str(batch),file=str(path)))
            self.assertEqual(db.execute('SELECT count(*) FROM reviews').fetchone()[0], 0)
            db.close()


if __name__ == '__main__':
    unittest.main()
