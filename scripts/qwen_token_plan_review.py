"""Bounded, interactive Qwen review inside the user's current agent session.

No credentials are persisted, no model/endpoint fallback exists, and no background
queue is connected. Each review command selects one existing short test clip.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import getpass
import hashlib
import io
import json
import re
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw
from cctv_ai.growth_workflow import frame_windows, image_bytes, save_json, sha_file

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'output/annual-film-validation-20261103'
COMPARISON = ROOT / 'output/model-comparison-20261103'
ENDPOINT = 'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1'
MODEL = 'qwen3.8-flash'
MODEL_RATES = {'qwen3.8-flash': (0.8, 2.7), 'qwen3.8-max': (12.0, 36.0),
               'deepseek-v4.1-flash-1017': (None, None)}


class ReviewError(RuntimeError):
    pass


def usage_equivalent(usage, model=MODEL):
    """Public PAYG equivalent, explicitly not a Token Plan charge or Credits."""
    if 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
        return {'public_payg_cny_without_discounts':None, 'actual_token_plan_credits':None,
                'actual_rmb_charge':None, 'basis':'provider usage unavailable; never infer free usage'}
    prompt = int(usage['prompt_tokens'])
    completion = int(usage['completion_tokens'])
    cached = int((usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0))
    if not 0 <= cached <= prompt:
        raise ReviewError('Invalid cached token count')
    input_rate, output_rate = MODEL_RATES[model]
    if input_rate is None:
        return {'public_payg_cny_without_discounts':None,
                'public_payg_cny_reference_range':[(prompt+completion*4)/1e6,(prompt*2+completion*8)/1e6],
                'actual_token_plan_credits':None,'actual_rmb_charge':None,
                'basis':'public Beijing/global idle/busy reference rates only; actual route, Credits and cache discounts unknown'}
    return {'public_payg_cny_without_discounts': (prompt * input_rate + completion * output_rate) / 1e6,
            'actual_token_plan_credits': None, 'actual_rmb_charge': None,
            'basis': 'public list-price equivalent only; actual Credits require platform usage records'}


def validate_answer(text):
    if text.strip().startswith('```'):
        text = text.strip().split('\n', 1)[1].rsplit('```', 1)[0]
    obj = json.loads(text)
    if not isinstance(obj.get('child_visible'), bool):
        raise ReviewError('Invalid child_visible field')
    for key in ['bottle_mouth_contact', 'adult_lift_or_lower', 'toy_interaction']:
        if obj.get(key) not in {'yes', 'no', 'unclear'}:
            raise ReviewError('Invalid event field')
    if not isinstance(obj.get('uncertain_details'), list):
        raise ReviewError('Invalid uncertainty field')
    return obj


class InteractiveClient:
    def __init__(self, key, model=MODEL, thinking_budget=0):
        if not key.startswith('sk-sp-') or key.strip() != key:
            raise ReviewError('Expected a Token Plan key, supplied only through non-echo input')
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers['Authorization'] = 'Bearer ' + key
        self.calls = 0
        if model not in MODEL_RATES:raise ReviewError('Unsupported explicit experiment model')
        self.model = model
        self.thinking_budget = thinking_budget
        if model.startswith('deepseek-') and thinking_budget:
            raise ReviewError('This DeepSeek proxy does not support Qwen thinking_budget')

    def request(self, method, path, body=None):
        # Do not log exception text, HTTP bodies, request payloads, or headers:
        # auth errors can echo credentials, and payloads contain private images.
        try:
            response = self.session.request(method, ENDPOINT + path, json=body,
                timeout=(90, 240), allow_redirects=False, stream=bool(body and body.get('stream')))
        except requests.RequestException as exc:
            kinds = [type(exc).__name__]
            for value in exc.args:
                if isinstance(value, BaseException):
                    kinds.append(type(value).__name__)
                    for nested in value.args:
                        if isinstance(nested, BaseException):kinds.append(type(nested).__name__)
            functions = [frame.name for frame in traceback.extract_tb(exc.__traceback__)]
            for value in exc.args:
                if isinstance(value, BaseException):
                    functions += [frame.name for frame in traceback.extract_tb(value.__traceback__)]
                    for nested in value.args:
                        if isinstance(nested, BaseException):
                            functions += [frame.name for frame in traceback.extract_tb(nested.__traceback__)]
            raise ReviewError('Transport failure: ' + '/'.join(kinds) + '; stack functions=' + ','.join(functions[-12:])) from None
        if response.status_code != 200:
            detail = ''
            try:
                error = response.json().get('error') or {}
                detail = str(error.get('message') or error.get('code') or '')
                credential = self.session.headers.get('Authorization','').removeprefix('Bearer ')
                if credential:detail = detail.replace(credential,'[credential withheld]')
                detail = re.sub(r'sk-[A-Za-z0-9_.-]+','[credential withheld]',detail)
                detail = re.sub(r'data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+','[media withheld]',detail)
            except (ValueError,AttributeError):pass
            response.close()
            raise ReviewError('Provider HTTP ' + str(response.status_code) + '; sanitized detail: ' + detail[:600])
        if body and body.get('stream'):
            return collect_stream(response)
        try:
            return response.json()
        except ValueError:
            raise ReviewError('Provider returned invalid JSON') from None

    def complete(self, content, max_tokens=768):
        if self.calls >= 7:
            raise ReviewError('Interactive session request bound reached; no background continuation')
        self.calls += 1
        started = time.perf_counter()
        payload = {'model':self.model, 'messages':[{'role':'user', 'content':content}],
            'temperature':0, 'max_tokens':max_tokens+self.thinking_budget,
            'enable_thinking':bool(self.thinking_budget),
            'stream':True, 'stream_options':{'include_usage':True}}
        if self.thinking_budget:payload['thinking_budget'] = self.thinking_budget
        body = self.request('POST', '/chat/completions', payload)
        choice = body['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ReviewError('Model output was truncated or non-final; do not mark complete')
        # Hosted snapshot aliases may be returned. Record the exact returned ID;
        # do not silently try a different model when the requested one fails.
        usage = body.get('usage') or {}
        return {'requested_model':self.model, 'returned_model':body.get('model'),
                'thinking_requested':bool(self.thinking_budget), 'thinking_budget':self.thinking_budget,
                'response_id':body.get('id'), 'text':choice['message']['content'],
                'usage':usage, 'cost_equivalent':usage_equivalent(usage, self.model),
                'seconds':round(time.perf_counter()-started, 3)}

    def close(self):
        self.session.headers.pop('Authorization', None)
        self.session.close()


def synthetic_many(client, count):
    if not 1 <= count <= 70:raise ReviewError('Synthetic diagnostic image bound exceeded')
    picture=Image.new('RGB',(128,128),'white');ImageDraw.Draw(picture).ellipse((25,25,100,100),fill='red')
    uri='data:image/jpeg;base64,'+base64.b64encode(image_bytes(picture)).decode()
    content=[{'type':'image_url','image_url':{'url':uri}} for _ in range(count)]
    content.append({'type':'text','text':'Return only JSON with supplied_image_count (integer). Count the supplied images.'})
    return client.complete(content,128)


def smoke(client, output):
    target = output / 'synthetic_smoke.json'
    if target.exists():
        raise ReviewError('Existing synthetic result is preserved; choose a new output for a fresh session')
    # A local synthetic image contains no family information.
    picture = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(picture)
    for xy, color in [((15,30,75,90),'red'), ((95,100,155,160),'blue'), ((175,170,235,230),'green')]:
        draw.ellipse(xy, fill=color)
    picture.save(output / 'synthetic_smoke.png')
    content = [{'type':'image_url', 'image_url':{'url':'data:image/jpeg;base64,' + base64.b64encode(image_bytes(picture)).decode()}},
               {'type':'text', 'text':'Inspect this image. Return only JSON with circle_count (integer) and colors (array of English color names).'}]
    record = client.complete(content, 128)
    obj = json.loads(record['text'].strip().removeprefix('```json').removesuffix('```').strip())
    record['synthetic_expected'] = {'circle_count':3, 'colors':['red','blue','green']}
    record['synthetic_passed'] = obj.get('circle_count') == 3 and set(obj.get('colors', [])) == {'red','blue','green'}
    save_json(target, record)
    if not record['synthetic_passed']:
        raise ReviewError('Synthetic vision check did not pass')
    return {k:record[k] for k in ['returned_model','usage','seconds','synthetic_passed']}


def collect_stream(response):
    text = []; usage = {}; finish = None; model = None; identity = None
    try:
        for raw in response.iter_lines():
            if not raw or raw.startswith(b':') or not raw.startswith(b'data:'):
                continue
            raw = raw[5:].strip()
            if raw == b'[DONE]':break
            event = json.loads(raw)
            if event.get('error'):raise ReviewError('Provider stream error; body withheld')
            model = event.get('model') or model
            identity = event.get('id') or identity
            if event.get('usage'):usage = event['usage']
            for choice in event.get('choices',[]):
                if choice.get('finish_reason'):finish = choice['finish_reason']
                content = (choice.get('delta') or {}).get('content')
                if content:text.append(content)
    except requests.RequestException:
        raise ReviewError('Transport failure while reading stream; output incomplete') from None
    finally:
        response.close()
    if finish is None:
        raise ReviewError('Stream ended before final completion; output incomplete')
    return {'model':model, 'id':identity, 'usage':usage,
            'choices':[{'finish_reason':finish, 'message':{'content':''.join(text)}}]}


def review(client, sample_id, output, retry_failed=False):
    samples = json.loads((LAB / 'samples.json').read_text(encoding='utf-8'))['samples']
    sample = next((s for s in samples if s['id'] == sample_id), None)
    if not sample or sample['frames'] > 150 or sample['duration_seconds'] > 8:
        raise ReviewError('Select one existing short challenge clip, at most 150 frames and 8 seconds')
    # Verify the same complete short clip and frame JPEGs as the local comparison.
    if sha_file(LAB / sample['clip']) != sample['clip_sha256']:
        raise ReviewError('Source test clip checksum mismatch')
    frames = json.loads((LAB / sample['frame_manifest']).read_text(encoding='utf-8'))
    if [f['frame_index'] for f in frames] != list(range(sample['frames'])):
        raise ReviewError('Frame ledger is incomplete')
    conditions = json.loads((COMPARISON / 'comparison_conditions.json').read_text(encoding='utf-8'))
    baseline = [json.loads(line) for line in (COMPARISON / 'qwen38/results.jsonl').read_text(encoding='utf-8').splitlines() if line]
    covered = set()
    rows = []
    for ordinal, (first, last) in enumerate(frame_windows(len(frames), 64, 8)):
        target = output / (sample_id + f'-{ordinal:02d}.json')
        if target.exists():
            old = json.loads(target.read_text(encoding='utf-8'))
            if old.get('status') != 'completed':
                if not retry_failed:
                    raise ReviewError('Previous request is unfinished; inspect it before retrying')
                archive = target.with_name(target.stem+'-prior-failed.json')
                if archive.exists():archive = target.with_name(target.stem+'-prior-failed-2.json')
                if archive.exists():raise ReviewError('Two explicit failed-request retries already recorded')
                target.replace(archive)
            else:
                if old.get('requested_model') != client.model or old.get('thinking_budget',0) != client.thinking_budget:
                    raise ReviewError('Existing result belongs to a different model or reasoning mode')
                rows.append(old)
                covered.update(range(first, last))
                continue
        content = []
        window = frames[first:last]
        for frame in window:
            path = LAB / frame['file']
            if sha_file(path) != frame['sha256']:
                raise ReviewError('Source frame checksum mismatch')
            with Image.open(path) as im:
                jpeg = image_bytes(im, 512)
            content.extend([{'type':'text', 'text':f"source_frame={frame['frame_index']},clip_seconds={frame['clip_seconds']:.6f}"},
                {'type':'image_url', 'image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()}}])
        for frame in (window[0], window[len(window)//2], window[-1]):
            with Image.open(LAB / frame['file']) as im:
                jpeg = image_bytes(im, 1024)
            content.append({'type':'image_url', 'image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()}})
        content.append({'type':'text', 'text':conditions['prompt']})
        digest = hashlib.sha256(json.dumps(content, ensure_ascii=True, separators=(',',':')).encode()).hexdigest()
        reference = next(r for r in baseline if r['sample_id'] == sample_id and r['window_index'] == ordinal)
        if digest != reference['input_content_sha256']:
            raise ReviewError('Payload differs from the local-model comparison')
        row = {'sample_id':sample_id, 'window_index':ordinal, 'status':'prepared',
            'source_clip_sha256':sample['clip_sha256'], 'input_content_sha256':digest,
            'identical_content_to_local_comparison':True, 'first_frame':first,
            'last_frame':last-1, 'main_frames':last-first, 'available_clip_frames':len(frames),
            'main_max_edge':512, 'detail_max_edge':1024, 'audio_submitted':False,
            'source_deletion_authorized':False, 'training_label_authorized':False,
            'independent_human_gold':False}
        row['submitted_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        save_json(target, row)
        # Persist the attempt before sending; interruption never silently retries.
        row['status'] = 'submitted'
        save_json(target, row)
        try:
            result = client.complete(content)
            row.update(result, parsed=validate_answer(result['text']), status='completed')
        except Exception as exc:
            row['status'] = 'error'
            row['error_kind'] = type(exc).__name__
            row['safe_error_detail'] = str(exc) if isinstance(exc, ReviewError) else 'Details withheld'
            save_json(target, row)
            raise
        save_json(target, row)
        rows.append(row)
        covered.update(range(first, last))
        print(json.dumps({'sample':sample_id, 'window':ordinal, 'frames':last-first,
            'usage':row['usage'], 'seconds':row['seconds'], 'parsed':row['parsed']}, ensure_ascii=True), flush=True)
    assert covered == set(range(len(frames)))
    summary = {'sample_id':sample_id, 'status':'completed', 'unique_clip_frames':len(covered),
        'all_source_clip_frames_submitted':True, 'windows':len(rows),
        'prompt_tokens':sum(r['usage'].get('prompt_tokens',0) for r in rows),
        'completion_tokens':sum(r['usage'].get('completion_tokens',0) for r in rows),
        'public_payg_equivalent_cny':sum(r['cost_equivalent']['public_payg_cny_without_discounts'] for r in rows)
            if all(r['cost_equivalent']['public_payg_cny_without_discounts'] is not None for r in rows) else None,
        'actual_token_plan_credits':None, 'actual_rmb_charge':None,
        'important_event_recall_verified':False, 'nas_source_deletions':0}
    save_json(output / (sample_id + '-summary.json'), summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=tuple(MODEL_RATES), default=MODEL)
    parser.add_argument('--thinking-budget', type=int, default=0, choices=(0,500,1024),
                        help='Explicit bounded reasoning experiment; zero matches the non-thinking baseline')
    parser.add_argument('--resume', action='store_true', help='Reuse successful synthetic check; never repeats a completed review request')
    parser.add_argument('--retry-failed', action='store_true', help='One explicit retry, retaining the prior failure record')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise ReviewError('Keep this bounded experiment beneath project/output')
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'endpoint':ENDPOINT, 'requested_model':args.model, 'billing_route':'Token Plan',
        'scope':'current interactive agent session; synthetic smoke plus individually selected short clips',
        'key_persisted':False, 'background_batch_enabled':False,
        'user_authorized_small_personal_plan_video_test':True,
        'actual_token_plan_credits':None, 'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    metadata['transport'] = {'streamed':True, 'connect_and_upload_timeout_seconds':90,
                             'read_timeout_seconds':240, 'automatic_retries':0}
    metadata['thinking_budget'] = args.thinking_budget
    session_path = output / 'session.json'
    if session_path.exists():
        prior = json.loads(session_path.read_text(encoding='utf-8'))
        if not args.resume:raise ReviewError('Existing session; use explicit resume')
        history = output / 'session-prior.json'
        if not history.exists():save_json(history, prior)
    save_json(session_path, metadata)
    key = getpass.getpass('Token Plan API key (non-echo; RAM only): ')
    client = InteractiveClient(key, args.model, args.thinking_budget)
    del key
    try:
        models = client.request('GET', '/models')
        ids = [m['id'] for m in models.get('data',[]) if isinstance(m,dict) and 'id' in m]
        save_json(output / 'model_catalog_check.json', {'requested_model':client.model,
            'exact_model_available':client.model in ids, 'advertised_models':ids})
        if client.model not in ids:
            raise ReviewError('Requested Flash model not advertised; no model fallback')
        if args.resume:
            prior_smoke = json.loads((output/'synthetic_smoke.json').read_text(encoding='utf-8'))
            if not prior_smoke.get('synthetic_passed') or prior_smoke.get('requested_model') != client.model:
                raise ReviewError('No successful pinned synthetic test to reuse')
            print('Prior successful synthetic check retained; no repeated call.', flush=True)
        else:
            print(json.dumps({'synthetic':smoke(client, output)}, ensure_ascii=True), flush=True)
        print('READY: enter one sample ID per command, or quit. No automatic queue.', flush=True)
        while True:
            command = input('review> ').strip()
            if command == 'quit':
                break
            if not command:
                continue
            if command.startswith('synthetic-many '):
                count=int(command.split()[1]);result=synthetic_many(client,count)
                save_json(output/f'synthetic-many-{count}.json',result)
                print(json.dumps(result,ensure_ascii=True),flush=True);continue
            print(json.dumps(review(client, command, output, args.retry_failed), ensure_ascii=True), flush=True)
        metadata['status'] = 'finished'
    except Exception as exc:
        metadata.update(status='error', error_kind=type(exc).__name__)
        # Only our fixed safe error messages reach stdout.
        print(json.dumps({'status':'error', 'kind':type(exc).__name__,
            'safe_detail':str(exc) if isinstance(exc, ReviewError) else 'Details withheld to protect credentials and media'}), flush=True)
    finally:
        metadata['completion_requests_sent'] = client.calls
        client.close()
        metadata['credentials_released_with_session'] = True
        save_json(output / 'session.json', metadata)
    if metadata.get('status') != 'finished':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
