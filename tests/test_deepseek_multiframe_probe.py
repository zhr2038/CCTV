import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('deepseek_multiframe_probe',
    Path(__file__).resolve().parents[1]/'scripts/deepseek_multiframe_probe.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeResponse:
    status_code=500
    headers={'Content-Type':'application/json','x-request-id':'test-request'}
    def __init__(self,body):self.body=body;self.closed=False
    def iter_content(self,chunk_size):yield self.body
    def close(self):self.closed=True


class ProbeTests(unittest.TestCase):
    def test_selected_count_and_layout_keep_every_synthetic_image(self):
        for count in probe.COUNTS:
            body,descriptor=probe.build_request(count)
            obj=json.loads(body)
            self.assertEqual(sum(x['type']=='image_url' for x in obj['messages'][0]['content']),count)
            self.assertEqual(descriptor['unique_images'],count)
            self.assertEqual(obj['model'],'deepseek-v4.1-flash-1017')
            self.assertEqual(descriptor['request_body_bytes'],len(body))
            self.assertFalse(obj['enable_thinking'])
            self.assertNotIn('thinking_budget',obj)
            self.assertNotIn('temperature',obj)
        for layout in ['images-first','text-first','interleaved']:
            body,descriptor=probe.build_request(2,layout=layout)
            self.assertEqual(descriptor['count'],2)
        _,repeated=probe.build_request(8,pattern='repeated')
        self.assertEqual(repeated['unique_images'],1)

    def test_error_body_redacts_key_and_media_and_sends_once(self):
        key='sk-sp-offline-unit-key'
        uri='data:image/png;base64,'+'A'*600
        response=FakeResponse(json.dumps({'error':{'message':'Inference engine abort UNKNOWN '+key+' '+uri}}).encode())
        class Session:
            calls=0
            def mount(self,*args):pass
            def post(self,url,**kwargs):
                self.calls+=1
                self.url=url;self.kwargs=kwargs
                return response
            def close(self):pass
        session=Session()
        body,descriptor=probe.build_request(2)
        result=probe.send_once(body,descriptor,key,lambda:session)
        text=json.dumps(result)
        self.assertNotIn(key,text)
        self.assertNotIn(uri,text)
        self.assertNotIn('A'*128,text)
        self.assertIn('Inference engine abort UNKNOWN',text)
        self.assertEqual(result['http_status'],500)
        self.assertEqual(session.calls,1)
        self.assertEqual(session.url,probe.ENDPOINT)
        self.assertFalse(session.kwargs['allow_redirects'])
        self.assertTrue(response.closed)
        self.assertIsNone(result['actual_token_plan_credits'])

    def test_stream_empty_choices_usage_and_provider_error(self):
        class Stream:
            def __init__(self,items):self.items=items
            def iter_lines(self):
                for item in self.items:yield b'data: '+json.dumps(item).encode()
                yield b'data: [DONE]'
        result=probe.parse_stream(Stream([
            {'model':probe.MODEL,'choices':[{'delta':{'content':'{}'}}]},
            {'choices':[{'finish_reason':'stop','delta':{}}]},
            {'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':2}}]))
        self.assertTrue(result['api_complete'])
        self.assertEqual(result['usage']['prompt_tokens'],10)
        error=probe.parse_stream(Stream([{'error':{'message':'sk-sp-test data:image/png;base64,AAAA'}}]))
        self.assertFalse(error['api_complete'])
        self.assertNotIn('sk-sp-test',error['stream_error_body'])
        self.assertNotIn('data:image',error['stream_error_body'])


if __name__=='__main__':unittest.main()
