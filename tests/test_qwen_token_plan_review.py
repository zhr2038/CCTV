"""Credential protection and incomplete-stream guards, without cloud requests."""
import importlib.util
import json
from pathlib import Path
import unittest

import requests

spec = importlib.util.spec_from_file_location('qwen_token_plan_review',
    Path(__file__).resolve().parents[1] / 'scripts/qwen_token_plan_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeResponse:
    def __init__(self, events, status=200):
        self.events = events
        self.status_code = status
        self.closed = False

    def iter_lines(self):
        for event in self.events:
            yield b'data: ' + json.dumps(event).encode()
        yield b'data: [DONE]'

    def close(self):
        self.closed = True


class QwenInteractiveTests(unittest.TestCase):
    def test_provider_error_redacts_credential_and_embedded_media(self):
        class Response:
            status_code=400
            def json(self):return {'error':{'message':'Invalid key sk-sp-sentinel-only image data:image/jpeg;base64,AAABBBCCC'}}
            def close(self):pass
        class Session:
            headers={'Authorization':'Bearer sk-sp-sentinel-only'}
            def request(self,*args,**kwargs):return Response()
        client=module.InteractiveClient('sk-sp-test-only');client.session.close();client.session=Session()
        with self.assertRaises(module.ReviewError) as caught:client.request('POST','/chat/completions',{})
        self.assertNotIn('sentinel-only',str(caught.exception))
        self.assertNotIn('AAABBBCCC',str(caught.exception))

    def test_usage_chunk_with_no_choices_preserved_and_incomplete_rejected(self):
        response = FakeResponse([
            {'model':'qwen3.8-flash', 'choices':[{'delta':{'content':'{}'}, 'finish_reason':None}]},
            {'choices':[{'delta':{}, 'finish_reason':'stop'}]},
            {'choices':[], 'usage':{'prompt_tokens':200, 'completion_tokens':2}}])
        result = module.collect_stream(response)
        self.assertEqual(result['usage']['prompt_tokens'], 200)
        self.assertEqual(result['choices'][0]['message']['content'], '{}')
        self.assertTrue(response.closed)
        broken = FakeResponse([{'choices':[{'delta':{'content':'partial'}}]}])
        with self.assertRaises(module.ReviewError):module.collect_stream(broken)
        self.assertTrue(broken.closed)

    def test_missing_usage_is_unknown_cost_and_cache_not_double_counted(self):
        self.assertIsNone(module.usage_equivalent({})['public_payg_cny_without_discounts'])
        cost = module.usage_equivalent({'prompt_tokens':10000,'completion_tokens':300,
            'prompt_tokens_details':{'cached_tokens':1000}})
        self.assertAlmostEqual(cost['public_payg_cny_without_discounts'], 0.00881)
        self.assertIsNone(cost['actual_token_plan_credits'])

    def test_exception_cannot_echo_key_and_no_redirect_or_payg_fallback(self):
        class Session:
            def request(self, method, url, **kwargs):
                self.url, self.kwargs = url, kwargs
                raise requests.ConnectionError('DO-NOT-PRINT-THIS-CREDENTIAL')
        client = module.InteractiveClient('sk-sp-test-only')
        client.session.close()
        session = Session()
        client.session = session
        with self.assertRaises(module.ReviewError) as caught:
            client.request('POST','/chat/completions',{'stream':True})
        self.assertNotIn('DO-NOT-PRINT',str(caught.exception))
        self.assertEqual(session.url,module.ENDPOINT+'/chat/completions')
        self.assertFalse(session.kwargs['allow_redirects'])


if __name__ == '__main__':unittest.main()
