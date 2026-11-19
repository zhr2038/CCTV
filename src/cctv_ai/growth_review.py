"""Offline, non-destructive review of a growth-workflow batch."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def classify(window, file):
    tags = window.get("tags")
    if window["status"] != "completed" or not tags:
        return "error"
    if window.get('review_flags'):
        return 'unclear'
    if tags["private_care"] == "yes":
        return "private"
    if tags["preview_fit"] in {"good", "usable"} and tags["child_visibility"] in {"clear", "partial"}:
        return "candidate"
    if tags["child_visibility"] == "unclear" or tags["preview_fit"] == "unclear" or tags["private_care"] == "unclear":
        return "unclear"
    for event in file["audio"].get("analysis", {}).get("events", []):
        if event["label"] in {"laughter", "child_vocalization", "baby_cry"} and event["media_start_seconds"] < window["end"] and event["media_end_seconds"] > window["start"]:
            return "sound"
    # Deterministic 20% of otherwise unselected windows, independent of scores.
    if int(hashlib.sha256(window["id"].encode()).hexdigest()[:8], 16) % 5 == 0:
        return "audit"
    return "other"


def build_review(batch: Path, payload):
    by_file = {f["id"]: f for f in payload["files"]}
    for window in payload["windows"]:
        window["category"] = classify(window, by_file[window["file_id"]])
    # Candidate groups keep every contributing source range, not just one thumbnail.
    groups = []
    windows_by_file = defaultdict(list)
    for window in payload["windows"]:
        windows_by_file[window["file_id"]].append(window)
    for file in payload["files"]:
        current = None
        for window in windows_by_file[file["id"]]:
            if window["category"] not in {"candidate", "unclear", "sound", "private"}:
                current = None
                continue
            if current is None or window["start"] > current["media_end_seconds"]:
                current = {"id": window["id"], "source_relative_path": file["relative"],
                           "source_sha256": file["sha256"], "local_full_original": file["cache"],
                           "capture_hint": file["capture_hint"], "time_source": file["time_source"],
                           "watermark_required": bool(file["watermark_required"]),
                           "media_start_seconds": window["start"], "media_end_seconds": window["end"],
                           "contributing_window_ids": [], "categories": [], "human_review": "pending",
                           "NAS_cleanup_authorized": False}
                groups.append(current)
            current["media_end_seconds"] = max(current["media_end_seconds"], window["end"])
            current["contributing_window_ids"].append(window["id"])
            if window["category"] not in current["categories"]:
                current["categories"].append(window["category"])
    (batch / "candidate_groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    review_groups = []
    for file in payload["files"]:
        current = None
        for window in windows_by_file[file["id"]]:
            if current is not None and window["category"] == current["category"] and window["start"] <= current["end"]:
                current["end"] = max(current["end"], window["end"])
                current["last_frame"] = max(current["last_frame"], window["last_frame"])
                current["frame_count"] = current["last_frame"] - current["first_frame"] + 1
                current["member_ids"].append(window["id"])
                if current.get('clock_evidence') and window.get('clock_evidence'):
                    before=current['clock_evidence'];after=window['clock_evidence']
                    before['anchors']=before['anchors'][:1]+after['anchors'][-1:]
                    before['jump_observed_between_anchors']=before['jump_observed_between_anchors'] or after['jump_observed_between_anchors']
                if window["tags"]:
                    for field in ("observed_actions", "uncertain"):
                        current["tags"][field] = list(dict.fromkeys(current["tags"][field] + window["tags"][field]))
            else:
                current = json.loads(json.dumps(window))
                current["member_ids"] = [window["id"]]
                review_groups.append(current)
    payload["group_windows"] = review_groups
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>成长影片 · 月度选片</title>
<style>
:root{color-scheme:light;--ink:#1e3529;--paper:#f6f5ef;--line:#d7dfd3;--muted:#606c60;--green:#285740}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1160px;margin:auto;padding:32px 26px 70px}h1{font-size:32px;font-weight:500;margin:8px 0}h2{font-size:22px;font-weight:500}p{margin:8px 0}.muted,small{color:var(--muted);font-size:13px}a{color:var(--green)}button,select,textarea{font:inherit}button,select{padding:6px 12px;background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:3px;cursor:pointer}button:hover,button[aria-pressed=true]{background:#e1eadc;border-color:#7d9b7d}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid var(--green);outline-offset:3px}.banner{padding:12px 16px;border-left:3px solid #93a68d;background:#edf1e7;margin:18px 0}.metrics{display:flex;flex-wrap:wrap;gap:28px;padding:18px 0;border-block:1px solid var(--line);margin:18px 0}.metrics strong{font-size:24px;font-weight:500;display:block}.metrics span{font-size:13px;color:var(--muted)}video{width:100%;aspect-ratio:16/9;background:#14291f;max-height:510px;display:block}.player{background:#edf0e7;padding:14px}.player p{max-width:100%}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}.list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.entry{border-bottom:1px solid var(--line);padding-bottom:20px}.entry img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#1d3023}.actions{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.badge{font-size:12px;background:#e6ecdf;padding:3px 8px}.entry h3{font-size:16px;font-weight:500;margin:8px 0}.entry p{margin:6px 0}.entry textarea{width:100%;resize:vertical;background:#fffdf7;border:1px solid var(--line);padding:6px;margin-top:8px}.entry details{font-size:12px;color:var(--muted);overflow-wrap:anywhere}summary{cursor:pointer}#issues{white-space:pre-wrap;overflow-wrap:anywhere}footer{margin-top:30px;border-top:1px solid var(--line);padding-top:16px}code{font:12px Consolas,monospace;overflow-wrap:anywhere}.empty{padding:30px 0;color:var(--muted)}@media(max-width:700px){main{padding:20px 14px}.list{grid-template-columns:1fr}.metrics{gap:18px}h1{font-size:27px}}
</style></head><body><main>
<div class="muted">家庭成长档案 · 本地审阅</div><h1>从日常里，留住值得重看的片段</h1>
<p id="scope"></p><p class="muted" id="cameras"></p>
<div class="banner">这是候选发现和原声审阅队列。原片全部保留。输入帧覆盖完整也不能证明重要事件全部识别；声音类别可能来自电视或成人，尚未确认孩子身份或语句。任何“暂不入片”都不批准删除。</div>
<div class="metrics" id="metrics"></div>
<div class="player"><video id="player" controls preload="none" playsinline></video><p id="playing">从下方选择一个片段，播放完整原片中的对应时间窗。</p><p class="muted">播放器会在时间窗末尾暂停；“打开完整原片”可以查看前后文。没有自动播放。</p></div>
<div class="toolbar"><select id="filter" aria-label="筛选片段"><option value="all">全部时间窗</option><option value="candidate">入片候选</option><option value="unclear">看不清，需复核</option><option value="sound">声音线索</option><option value="audit">随机未命中抽查</option><option value="other">其他未命中</option><option value="private">隐私照护，暂不入片</option><option value="error">处理未完成</option><option value="keep">已选保留</option></select><button id="export" type="button">导出我的选片记录</button><span class="muted" id="reviewCount"></span></div>
<p class="muted" id="shown"></p><div class="list" id="list"></div><div class="toolbar"><button id="previous" type="button">上一页</button><button id="next" type="button">下一页</button></div>
<h2>范围与异常</h2><div id="issues"></div><p class="muted">相机未录下的时段不等于“没有事情发生”。本批次按明确的文件时间范围登记，不代表全天、整月或全年已完成。床头包的日期仅作线索，精确时间仍需水印映射。</p>
<div class="toolbar"><a href="progress.json">覆盖进度</a><a href="review_data.json">完整处理明细</a><a href="candidate_groups.json">合并候选及原片引用</a></div>
<footer class="muted">本页无外部网络请求。选片记录暂存于浏览器，并可导出JSON；若浏览器禁止本地存储，导出仍可用。标记只用于剪辑与复核，不会训练模型或操作NAS。</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), summary=data.summary, manifest=summary.manifest;
const files=Object.fromEntries(data.files.map(x=>[x.id,x])), byId=Object.fromEntries(data.windows.map(x=>[x.id,x]));
const labels={candidate:'入片候选',unclear:'需复核',sound:'声音线索',audit:'随机未命中抽查',other:'其他未命中',private:'疑似照护，需复核',error:'未完成'};
const storageKey='growth-review-'+manifest.recipe_hash+'-'+manifest.since+'-'+manifest.camera_ids.join(',');
document.querySelector('#filter option[value="private"]').textContent='疑似隐私照护，需复核';
const view=document.createElement('select');view.setAttribute('aria-label','显示方式');view.add(new Option('合并重叠窗','groups'));view.add(new Option('逐窗检查','windows'));document.getElementById('filter').parentElement.prepend(view);
let reviews=Object.fromEntries(data.reviews.map(x=>[x.window_id,{verdict:x.verdict,note:x.note}]));
try{reviews={...reviews,...JSON.parse(localStorage.getItem(storageKey)||'{}')}}catch(e){}
const player=document.getElementById('player');let endAt=null,activeFileId=null;
let pageIndex=0;const pageSize=40;
const time=x=>{const minutes=Math.floor(x/60);return String(minutes).padStart(2,'0')+':'+(x%60).toFixed(2).padStart(5,'0')};
function node(tag,text,klass){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(klass)el.className=klass;return el}
function store(){try{localStorage.setItem(storageKey,JSON.stringify(reviews))}catch(e){}document.getElementById('reviewCount').textContent='已标记 '+Object.keys(reviews).filter(id=>byId[id]).length+' 个时间窗'}
document.getElementById('scope').textContent=manifest.since+' 至 '+manifest.until;
if(manifest.selection_mode==='explicit_whole_files')document.getElementById('scope').textContent=(manifest.caption||'跨场景验证')+' · 仅处理明确选定的 '+summary.files_total+' 个完整原片，不代表日期范围内全部录像。';
document.getElementById('cameras').textContent='已纳入相机：'+manifest.camera_ids.join('、')+'；本批未纳入：'+(manifest.excluded_camera_ids.join('、')||'无');
const metrics=[['文件',String(summary.files_by_status.completed||0)+' / '+summary.files_total],['时间窗',summary.windows_completed+' / '+summary.windows_total],['已入模的独立源帧',String(summary.unique_frames_in_completed_requests)],['原片删除','0']];
for(const [label,value] of metrics){const item=node('div');item.append(node('strong',value),node('span',label));document.getElementById('metrics').append(item)}
document.getElementById('issues').textContent=JSON.stringify({files_by_status:summary.files_by_status,scan_issues:manifest.scan_issues,file_errors:data.files.filter(f=>f.error).map(f=>({source:f.relative,error:f.error})),full_scope_processing_complete:summary.full_scope_processing_complete,important_event_recall_verified:false},null,2);
function play(window,whole=false){const file=files[window.file_id],media=file.review_playback,shift=media?.source_to_playback_offset_seconds||0;activeFileId=file.id;endAt=whole?null:window.end+shift;player.src=encodeURI(media?.playback||file.cache);player.addEventListener('loadedmetadata',()=>{player.currentTime=whole?0:Math.max(0,window.start+shift)},{once:true});player.load();document.getElementById('playing').textContent=file.name+' · '+file.capture_hint+(whole?' · 完整原片回看':' · 源媒体 '+time(window.start)+'—'+time(window.end))+(media?.kind==='derived_h264_aac_review_copy'?' · 播放兼容副本，完整原片另存':'');player.scrollIntoView({behavior:'smooth',block:'center'})}
const fullList=node('details');fullList.style.margin='16px 0';fullList.append(node('summary','完整原片巡检（'+data.files.length+'份，不按AI标签筛选）'));for(const file of data.files){const line=node('p'),button=node('button',file.name+' · '+file.capture_hint.slice(0,19).replace('T',' ')+' · '+(file.duration?time(file.duration):'待处理'));button.type='button';button.disabled=!file.cache;button.onclick=()=>play({file_id:file.id,start:0,end:file.duration||0},true);line.append(button);fullList.append(line)}document.querySelector('.player').after(fullList);
const manual=node('p'),markNow=node('button','保留当前附近片段'),markStatus=node('span','','muted');markNow.type='button';manual.append(markNow,markStatus);document.querySelector('.player').append(manual);markNow.onclick=()=>{if(!activeFileId||player.readyState<1){markStatus.textContent=' 请先打开原片。';return}const sourceTime=player.currentTime-(files[activeFileId].review_playback?.source_to_playback_offset_seconds||0),chosen=data.windows.filter(w=>w.file_id===activeFileId&&w.start<=sourceTime+8&&w.end>=sourceTime-8);if(!chosen.length){markStatus.textContent=' 这个位置尚未生成时间窗，完整原片继续保留。';return}for(const w of chosen)reviews[w.id]={verdict:'keep',note:reviews[w.id]?.note||'手动保留源媒体 '+time(sourceTime)+' 附近，包含前后文。'};store();render();markStatus.textContent=' 已保留这个位置附近 '+chosen.length+' 个时间窗；不受AI标签限制。'};
player.addEventListener('timeupdate',()=>{if(endAt!==null&&player.currentTime>=endAt&&!player.paused)player.pause()});
function memberIds(w){return w.member_ids||[w.id]}
function verdictOf(w){const values=memberIds(w).map(id=>reviews[id]?.verdict);return values.every(x=>x===values[0])?values[0]:undefined}
function render(){const filter=document.getElementById('filter').value,list=document.getElementById('list');list.replaceChildren();const items=view.value==='groups'?data.group_windows:data.windows;const matching=items.filter(w=>filter==='all'||(filter==='keep'?verdictOf(w)==='keep':w.category===filter));pageIndex=Math.max(0,Math.min(pageIndex,Math.ceil(matching.length/pageSize)-1));const windows=matching.slice(pageIndex*pageSize,(pageIndex+1)*pageSize);document.getElementById('shown').textContent='共 '+matching.length+' 段，当前第 '+(pageIndex+1)+' / '+Math.max(1,Math.ceil(matching.length/pageSize))+' 页；同文件的重叠窗已合并，可切换逐窗检查，所有源片段和帧记录仍保留。';document.getElementById('previous').disabled=pageIndex===0;document.getElementById('next').disabled=(pageIndex+1)*pageSize>=matching.length;
for(const window of windows){const file=files[window.file_id],entry=node('article',undefined,'entry');
entry.id='clip-'+window.id;
if(window.poster){const image=node('img');image.src=encodeURI(window.poster);image.alt=file.name+' · 片段中间帧';image.loading='lazy';entry.append(image)}
entry.append(node('span',labels[window.category],'badge'),node('h3',file.name+' · '+time(window.start)+'—'+time(window.end)),node('p',file.capture_hint,'muted'));
const tags=window.tags;if(tags){entry.append(node('p',tags.observed_actions.join('；')||'未确认具体动作'),node('p',tags.reason,'muted'));if(tags.uncertain.length)entry.append(node('p','看不准：'+tags.uncertain.join('；'),'muted'))}else entry.append(node('p',window.error||'处理未完成','muted'));
if(window.review_flags?.length)entry.append(node('p','这个短窗识别到孩子，相邻窗却未识别到：可能是短暂出现，也可能是误判，需要回看前后文；原片继续保留。','muted'));
const clocks=window.clock_evidence?.anchors?.filter(a=>a.wall_time)||[];if(clocks.length)entry.append(node('p','画面时钟锚点：'+clocks.map(a=>a.wall_time.slice(0,19).replace('T',' ')).join(' / ')+'；非逐秒完整映射。','muted'));if(window.clock_evidence?.jump_observed_between_anchors)entry.append(node('p','检测到画面时钟跳变，请按原片媒体偏移回看。','muted'));
for(const review of (data.cloud_reviews||[]).filter(r=>memberIds(window).includes(r.window_id))){entry.append(node('p','Kimi K3.1复核（3张画面，非全段）：'+review.result.reason,'muted'));if(review.result.uncertain?.length)entry.append(node('p','云端仍看不准：'+review.result.uncertain.join('；'),'muted'))}
const events=file.audio.analysis?.events?.filter(e=>e.media_start_seconds<window.end&&e.media_end_seconds>window.start)||[];
if(events.length)entry.append(node('p','声音类别线索：'+[...new Set(events.map(e=>e.model_label))].join('、')+'（未确认声源）','muted'));
const actions=node('div',undefined,'actions'),playButton=node('button','定位此片段');playButton.type='button';playButton.onclick=()=>play(window);actions.append(playButton);
if(file.cache){const original=node('a','打开完整原片');original.href=encodeURI(file.cache);original.target='_blank';original.rel='noopener';actions.append(original)}entry.append(actions);
const decisions=node('div',undefined,'actions');for(const [verdict,text] of [['keep','保留'],['not_for_film','暂不入片'],['unclear','看不准']]){const button=node('button',text);button.type='button';button.setAttribute('aria-pressed',verdictOf(window)===verdict);button.onclick=()=>{for(const id of memberIds(window))reviews[id]={...reviews[id],verdict};store();render()};decisions.append(button)}entry.append(decisions);
const note=node('textarea');note.rows=1;note.placeholder='补充动作、日期或你想留下的原因';note.value=reviews[window.id]?.note||'';note.setAttribute('aria-label','片段备注');note.onchange=()=>{for(const id of memberIds(window))reviews[id]={verdict:reviews[id]?.verdict||'unclear',note:note.value};store()};entry.append(note);
const details=node('details');details.append(node('summary','查看原片与输入覆盖'),node('code',file.relative+' | SHA-256 '+file.sha256+' | 源帧 '+window.first_frame+'—'+window.last_frame+'，共'+window.frame_count+'帧；主图最大边'+(summary.input_main_image_max_edge||512)+'，另加3张'+(summary.input_detail_image_max_edge?'最大边'+summary.input_detail_image_max_edge:'原尺寸')+'补图；声音单独分析。'));entry.append(details);list.append(entry)}if(!windows.length)list.append(node('p','这个分类目前没有片段。','empty'));store()}
document.getElementById('filter').onchange=()=>{pageIndex=0;render()};
view.onchange=()=>{pageIndex=0;render()};
document.getElementById('previous').onclick=()=>{pageIndex--;render()};document.getElementById('next').onclick=()=>{pageIndex++;render()};
document.getElementById('export').onclick=()=>{const payload={version:1,recipe_hash:manifest.recipe_hash,scope:{since:manifest.since,until:manifest.until,camera_ids:manifest.camera_ids},source_deletion_authorized:false,training_authorized:false,reviews:Object.entries(reviews).filter(([id])=>byId[id]).map(([id,r])=>({window_id:id,source_sha256:files[byId[id].file_id].sha256,verdict:r.verdict,note:r.note||''}))};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json;charset=utf-8'}));const anchor=node('a');anchor.href=url;anchor.download='growth-review.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();
const linked=new URLSearchParams(location.hash.slice(1)).get('window');
if(linked&&byId[linked]){view.value='windows';document.getElementById('filter').value='all';pageIndex=Math.floor(data.windows.findIndex(w=>w.id===linked)/pageSize);render();play(byId[linked]);}
</script></body></html>'''
    (batch / "index.html").write_text(page.replace("__DATA__", data), encoding="utf-8")
