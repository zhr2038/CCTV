import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('kimi_growth_interactive',
    Path(__file__).resolve().parents[1]/'scripts/kimi_growth_interactive.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class KimiTimeWindowsTests(unittest.TestCase):
    def test_uncertain_alias_stays_uncertain_and_is_audited(self):
        import json
        value={'child_visibility':'clear','adult_interaction':'yes','preview_fit':'good',
            'private_care':'uncertain','observed_actions':['interaction'],'reason':'occlusion','uncertain':['occluded']}
        tags,normalizations=module.parse_labels(json.dumps(value))
        self.assertEqual(tags['private_care'],'unclear')
        self.assertEqual(normalizations,[{'field':'private_care','from':'uncertain','to':'unclear'}])

    def test_six_second_windows_cover_every_irregular_source_frame(self):
        times=[0.0,0.1,0.13,0.9,4.9,5.0,5.9,6.0,8.5,10.0,10.01,11.99,12.1]
        ledger=[{'media_seconds':t} for t in times]
        windows=module.time_windows(ledger,12.2)
        self.assertEqual({i for first,last,_,_ in windows for i in range(first,last)},set(range(len(times))))
        self.assertTrue(all(last>first and end-begin<=6 for first,last,begin,end in windows))
        self.assertEqual(windows[0][0],0)
        self.assertEqual(windows[-1][1],len(times))

    def test_none_status_string_does_not_block_or_resubmit_live_session(self):
        import tempfile
        class Service:
            calls=[]
            def call(self,method,path,**kwargs):
                self.calls.append((method,path))
                if path.endswith('/messages'):
                    import json
                    tags={'child_visibility':'unclear','adult_interaction':'unclear',
                        'preview_fit':'unclear','private_care':'unclear','observed_actions':[],
                        'reason':'occluded','uncertain':['occlusion']}
                    return {'items':[{'role':'assistant','content':[{'type':'text','text':json.dumps(tags)}]}]}
                return {'pending_interaction':'none','busy':False,'main_turn_active':False,'last_turn_reason':'completed','usage':{}}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);runtime={'completed_windows':0};service=Service()
            module.wait_for_labels(service,'existing',p/'window.json',{'ordinal':0,'input_frames':120},runtime,p)
            self.assertEqual(runtime['completed_windows'],1)
            self.assertTrue(all(method=='GET' for method,_ in service.calls))


if __name__=='__main__':unittest.main()
