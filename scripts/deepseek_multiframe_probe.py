"""Single synthetic-image diagnostic for the existing Token Plan endpoint.

Defaults to dry-run. --send authorizes exactly one POST, never a retry or model
fallback. No family media or credential files are read. Error bodies are redacted.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw
import requests
from requests.adapters import HTTPAdapter

ROOT = Path(os.environ.get('CCTV_PROBE_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
ENDPOINT = 'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1/chat/completions'
MODEL = 'deepseek-v4.1-flash-1017'
COUNTS = (1, 2, 8, 32, 64, 67)
MAX_REQUEST_BYTES = 8 * 1024 * 1024  # conservative local guard, not a claimed provider limit
MAX_RESPONSE_BYTES = 64 * 1024


def sanitized(text, secret=''):
    text = str(text)
    if secret:
        text = text.replace(secret, '[REDACTED_KEY]')
    text = re.sub(r'sk-[A-Za-z0-9_.-]+', '[REDACTED_KEY]', text)
    text = re.sub(r'(?i)Bearer[ \t]+[^\s"\'<>]+', 'Bearer [REDACTED]', text)
    # Over-redact the rest of a scalar/line containing a data URI. A diagnostic
    # does not need echoed request media, including whitespace-wrapped base64.
    text = re.sub(r'(?i)data:[^,\s"\']{1,200},[^"\']*', '[REDACTED_DATA_URI]', text)
    text = re.sub(r'[A-Za-z0-9+/=_-]{128,}', '[REDACTED_LONG_DATA]', text)
    return text


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_images(count, width, height, pattern='distinct', codec='jpeg'):
    if count not in COUNTS:
        raise ValueError('Select one supported image count')
    if not 64 <= width <= 1024 or not 64 <= height <= 1024:
        raise ValueError('Synthetic dimensions must be 64..1024')
    uris, hashes, sizes = [], [], []
    for index in range(count):
        identity = index if pattern == 'distinct' else 0
        picture = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(picture)
        color = ((identity * 71 + 80) % 230, (identity * 43 + 30) % 230, (identity * 97 + 90) % 230)
        draw.rectangle((width//8, height//8, width*7//8, height*7//8), fill=color)
        draw.text((width//8+8, height//8+8), f'SYNTHETIC FRAME {identity+1:03d}', fill='black')
        # The last small colored marker makes distinct images byte-distinct even
        # at low resolution. This is a transport diagnostic, not an OCR benchmark.
        draw.rectangle((width-16, height-16, width-1, height-1), fill=(identity,0,0))
        memory = io.BytesIO()
        picture.save(memory, format='JPEG' if codec == 'jpeg' else 'PNG', **({'quality':90} if codec == 'jpeg' else {}))
        data = memory.getvalue()
        mime = 'image/jpeg' if codec == 'jpeg' else 'image/png'
        uris.append('data:'+mime+';base64,'+base64.b64encode(data).decode('ascii'))
        hashes.append(hashlib.sha256(data).hexdigest())
        sizes.append(len(data))
    return uris, hashes, sizes


def build_request(count, width=544, height=544, layout='images-first',
                  pattern='distinct', codec='jpeg', detail='omit', thinking='false',
                  max_tokens=128, stream=False):
    uris, hashes, sizes = make_images(count, width, height, pattern, codec)
    images = []
    for uri in uris:
        image = {'url':uri}
        if detail != 'omit':image['detail'] = detail
        images.append({'type':'image_url', 'image_url':image})
    instruction = {'type':'text', 'text':'Inspect the supplied synthetic images in order. Return only a JSON object with supplied_image_count (integer).'}
    if layout == 'images-first':
        content = images+[instruction]
    elif layout == 'text-first':
        content = [instruction]+images
    elif layout == 'interleaved':
        content=[]
        for index, image in enumerate(images):
            content.extend([{'type':'text','text':f'frame_index={index}'},image])
        content.append(instruction)
    else:
        raise ValueError('Unsupported explicit layout')
    payload = {'model':MODEL, 'messages':[{'role':'user','content':content}],
               'max_tokens':max_tokens, 'stream':stream}
    if thinking == 'false':payload['enable_thinking'] = False
    if stream:payload['stream_options'] = {'include_usage':True}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',',':')).encode('utf-8')
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError('Request exceeds conservative 8 MiB local guard; no request sent')
    descriptor = {'model':MODEL,'count':count,'dimensions':[width,height],
        'layout':layout,'pattern':pattern,'codec':codec,'detail':detail,
        'enable_thinking':'omitted' if thinking == 'omit' else False,
        'max_tokens':max_tokens,'stream':stream,'request_body_bytes':len(encoded),
        'request_body_sha256':hashlib.sha256(encoded).hexdigest(),
        'image_sha256':hashes,'unique_images':len(set(hashes)),
        'image_file_bytes':sizes,'content_block_types':[part['type'] for part in content],
        'user_messages':1,'no_audio_or_video_parts':True,'no_family_media':True,
        'local_request_guard_bytes':MAX_REQUEST_BYTES,'provider_limit_claimed':False}
    return encoded, descriptor


def bounded_body(response):
    data = bytearray()
    truncated = False
    for block in response.iter_content(chunk_size=4096):
        if len(data)+len(block) > MAX_RESPONSE_BYTES:
            data.extend(block[:MAX_RESPONSE_BYTES-len(data)])
            truncated = True
            break
        data.extend(block)
    return bytes(data), truncated


def parse_stream(response, secret=''):
    answer=[]; finish=None; usage=None; model=None; response_id=None; read_bytes=0
    for raw in response.iter_lines():
        read_bytes += len(raw)
        if read_bytes > MAX_RESPONSE_BYTES:
            raise ValueError('Diagnostic response limit exceeded; no further data read')
        if not raw or raw.startswith(b':') or not raw.startswith(b'data:'):
            continue
        raw = raw[5:].strip()
        if raw == b'[DONE]':break
        try:
            chunk=json.loads(raw)
        except ValueError:
            return {'api_complete':False,'invalid_stream_event':sanitized(raw.decode('utf-8','replace'),secret)[:4096],
                    'usage':usage,'returned_model':model,'response_id':response_id}
        if chunk.get('error'):
            return {'api_complete':False,'stream_error_body':sanitized(json.dumps(chunk,ensure_ascii=False),secret)[:4096],
                    'usage':usage,'returned_model':model,'response_id':response_id}
        model = chunk.get('model') or model
        response_id = chunk.get('id') or response_id
        if chunk.get('usage'):usage=chunk['usage']
        for choice in chunk.get('choices',[]):
            if choice.get('finish_reason'):finish=choice['finish_reason']
            text=(choice.get('delta') or {}).get('content')
            if text:answer.append(text)
    return {'api_complete':finish=='stop','finish_reason':finish,
            'text':sanitized(''.join(answer),secret),'usage':usage,
            'returned_model':model,'response_id':response_id}


def send_once(body, descriptor, key, session_factory=requests.Session):
    if not key.startswith('sk-sp-') or key != key.strip():
        raise ValueError('Expected a Token Plan key supplied via environment or non-echo input')
    session=session_factory()
    session.trust_env=False
    # requests has no retry by default; pin it explicitly for this diagnostic.
    session.mount('https://',HTTPAdapter(max_retries=0))
    response=None
    started=time.perf_counter()
    try:
        response=session.post(ENDPOINT,data=body,headers={
            'Authorization':'Bearer '+key,'Content-Type':'application/json'},
            timeout=(90,180),allow_redirects=False,stream=True)
        result={'http_status':response.status_code,
            'response_content_type':sanitized(response.headers.get('Content-Type',''),key),
            'request_id':sanitized(response.headers.get('x-request-id',''),key),
            'returned_model':None,'api_complete':False,'usage':None,
            'actual_token_plan_credits':None,'actual_rmb_charge':None,
            'automatic_retries':0,'model_fallback':False}
        if response.status_code != 200:
            data,truncated=bounded_body(response)
            result.update(status='http_error',error_body=sanitized(data.decode('utf-8','replace'),key)[:4096],
                          error_body_read_truncated=truncated)
        elif descriptor['stream']:
            result.update(parse_stream(response,key))
            result['status']='completed' if result['api_complete'] else 'incomplete'
        else:
            data,truncated=bounded_body(response)
            try:
                obj=json.loads(data) if not truncated else None
            except ValueError:
                obj=None
            if not isinstance(obj,dict) or not obj.get('error') and not obj.get('choices'):
                result.update(status='invalid_response',error_body=sanitized(data.decode('utf-8','replace'),key)[:4096],
                              error_body_read_truncated=truncated)
            elif obj.get('error'):
                result.update(status='provider_error',error_body=sanitized(json.dumps(obj,ensure_ascii=False),key)[:4096])
            else:
                choice=obj['choices'][0]
                result.update(returned_model=obj.get('model'),response_id=obj.get('id'),
                    finish_reason=choice.get('finish_reason'),usage=obj.get('usage'),
                    text=sanitized(choice['message'].get('content',''),key),
                    api_complete=choice.get('finish_reason')=='stop')
                result['status']='completed' if result['api_complete'] else 'incomplete'
        if result.get('text'):
            try:
                parsed=json.loads(result['text'].strip().removeprefix('```json').removesuffix('```').strip())
                result['synthetic_count_answer_matches']=parsed.get('supplied_image_count')==descriptor['count']
            except (ValueError,AttributeError):result['synthetic_count_answer_matches']=None
        result['seconds']=round(time.perf_counter()-started,3)
        return result
    except requests.RequestException as exc:
        return {'status':'transport_error','error_type':type(exc).__name__,
                'http_status':getattr(response,'status_code',None),'api_complete':False,
                'usage':None,'actual_token_plan_credits':None,'actual_rmb_charge':None,
                'seconds':round(time.perf_counter()-started,3),'automatic_retries':0,'model_fallback':False}
    finally:
        if response is not None:response.close()
        session.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,choices=COUNTS,required=True)
    parser.add_argument('--layout',choices=('images-first','text-first','interleaved'),default='images-first')
    parser.add_argument('--pattern',choices=('distinct','repeated'),default='distinct')
    parser.add_argument('--width',type=int,default=544)
    parser.add_argument('--height',type=int,default=544)
    parser.add_argument('--codec',choices=('jpeg','png'),default='jpeg')
    parser.add_argument('--detail',choices=('omit','low','high','original','auto'),default='omit')
    parser.add_argument('--thinking',choices=('false','omit'),default='false')
    parser.add_argument('--max-tokens',type=int,choices=(128,256,512),default=128)
    parser.add_argument('--stream',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--send',action='store_true',help='Exactly one synthetic POST; omitted means no credential access and no network')
    args=parser.parse_args(argv)
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'output'):
        parser.error('Output must be beneath E:/CCTV/output')
    target=output/'result.json'
    if target.exists():parser.error('Existing result preserved; choose a new output folder')
    output.mkdir(parents=True,exist_ok=True)
    body,descriptor=build_request(args.count,args.width,args.height,args.layout,args.pattern,
        args.codec,args.detail,args.thinking,args.max_tokens,args.stream)
    write(output/'request_descriptor.json',descriptor)
    result={'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'endpoint':ENDPOINT,'model':MODEL,'status':'prepared' if args.send else 'dry_run',
        'request':descriptor,'family_media_accessed':False,'credential_persisted':False,
        'api_requests_sent':0,'automatic_retries':0,'model_fallback':False,
        'quality_acceptance':False,'actual_token_plan_credits':None,'actual_rmb_charge':None}
    write(target,result)
    if args.send:
        key=os.environ.get('CCTV_QWEN_API_KEY') or getpass.getpass('Token Plan API key (non-echo; RAM only): ')
        if not key.startswith('sk-sp-') or key.strip()!=key:
            raise SystemExit('Invalid Token Plan credential format; no request sent')
        result.update(status='submitted',api_requests_sent=1)
        write(target,result)
        try:
            result['response']=send_once(body,descriptor,key)
            result['status']=result['response']['status']
        except Exception as exc:
            result.update(status='diagnostic_parse_error',error_type=type(exc).__name__,
                          safe_error=sanitized(str(exc),key)[:600])
        finally:
            del key
            result['credential_reference_released']=True
            write(target,result)
    print(json.dumps({'status':result['status'],'count':args.count,
        'request_bytes':descriptor['request_body_bytes'],'output':str(target),
        'http_status':(result.get('response') or {}).get('http_status')},ensure_ascii=True))
    return 0 if result['status'] in ('completed','dry_run') else 1


if __name__=='__main__':raise SystemExit(main())
