"""A local entry point to multiple review batches. No review decisions are moved."""
from pathlib import Path
import json


def build_library(default_batch, catalog_path):
    root=Path(default_batch).resolve()
    catalog=json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    batches=[]
    for alias,folder in catalog.items():
        path=Path(folder)
        progress=path/'progress.json'
        if progress.exists():
            data=json.loads(progress.read_text(encoding='utf-8'))
            caption=data['manifest'].get('caption') or ('首批客厅原声选片' if alias=='first' else alias)
            batches.append({'alias':alias,'caption':caption,'type':'batch','progress':data})
        elif (path/'movie_validation.json').exists():
            batches.append({'alias':alias,'caption':'2026年1—11月原声样片','type':'pilot',
                            'movie':json.loads((path/'movie_validation.json').read_text(encoding='utf-8'))})
    payload=json.dumps(batches,ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>成长影片 · 素材总览</title><link rel="icon" href="data:,"><style>
*{box-sizing:border-box}body{margin:0;background:#f6f5ef;color:#20392c;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1050px;margin:auto;padding:40px 24px 70px}h1{font-size:38px;line-height:1.3;font-weight:500;margin:10px 0}h2{font-size:22px;font-weight:500;margin:0 0 8px}p{margin:8px 0}.muted,small{font-size:13px;color:#647064}a{color:#2a5a40;text-underline-offset:3px}.note{border-left:3px solid #99ae91;background:#edf1e6;padding:12px 18px;margin:22px 0}.batch{padding:26px 0;border-bottom:1px solid #d8dfd2;display:grid;grid-template-columns:1fr auto;gap:20px}.status{font-size:14px;min-width:170px}.status strong{font-size:26px;font-weight:500}input{width:100%;padding:13px 15px;background:#fffdf7;border:1px solid #b8c6b3;border-radius:3px;font:inherit}button{font:inherit;background:#e6edde;border:1px solid #a3b49a;padding:6px 13px;cursor:pointer;color:#254e37}.hit{padding:15px 0;border-bottom:1px solid #d8dfd2}.hit p{font-size:14px}code{font:12px Consolas,monospace;overflow-wrap:anywhere}footer{border-top:1px solid #d8dfd2;margin-top:35px;padding-top:18px;font-size:13px;color:#647064}@media(max-width:650px){h1{font-size:29px}.batch{grid-template-columns:1fr}.status{min-width:0}}
</style></head><body><main><div class="muted">给未来的他 · 家庭成长档案</div><h1>把平常的日子，慢慢留成年片</h1><p>看原声、留出处，逐批积累值得重看的片段。</p><div class="note">完整原片继续保留，NAS删除为0。这里汇集的是样片与明确范围的验证批次，尚未完成全年关键事件召回验收。模型标签只作为回看线索。</div><div id="batches"></div><h2 style="margin-top:32px">找一个动作或声音线索</h2><p class="muted">例如：抱、站、玩具、笑声、说话、baby、laughter。搜索只检查已处理的标签；未命中不代表原片中没有。</p><input id="query" type="search" aria-label="查找已处理素材" placeholder="输入想回看的动作或声音"><p class="muted" id="searchStatus">正在加载已处理的线索……</p><div id="results"></div><footer>本页只读取本机已生成的记录，不上传媒体。选片仍在各批次页面中保存；原来的首批地址和浏览器选片存储保持可用。扩量前继续核验漏检、声音与时间信息。</footer></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const batches=JSON.parse(document.getElementById('payload').textContent),holder=document.getElementById('batches');let searchRows=[];
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
for(const batch of batches){const row=el('section',undefined,'batch'),left=el('div'),link=el('a',batch.caption);link.href='/batches/'+batch.alias+'/index.html';left.append(el('h2'));left.firstChild.append(link);const status=el('div',undefined,'status');if(batch.type==='pilot'){left.append(el('p','2分27秒 · 11个月 · 原声流程样片','muted'));status.append(el('strong','查看样片'))}else{const p=batch.progress;left.append(el('p',p.manifest.selection_mode==='explicit_whole_files'?'只纳入明确选定的 '+p.files_total+' 份完整原片；非连续日期覆盖。':p.manifest.since+' 至 '+p.manifest.until,'muted'));left.append(el('p','模型分析、原声回看、原片追溯和选片导出。','muted'));status.append(el('strong',(p.files_by_status.completed||0)+' / '+p.files_total),el('div','原始文件已处理'),el('small',p.windows_completed+'个时间窗 · '+p.unique_frames_in_completed_requests+'个独立主视频源帧'))}row.append(left,status);holder.append(row)}
async function load(){for(const batch of batches.filter(x=>x.type==='batch')){try{const response=await fetch('/batches/'+batch.alias+'/review_data.json',{cache:'no-store'});if(!response.ok)throw Error('read');const data=await response.json(),files=Object.fromEntries(data.files.map(f=>[f.id,f]));for(const w of data.windows){if(!w.tags)continue;const f=files[w.file_id],sound=f.audio?.analysis?.events?.filter(e=>e.media_start_seconds<w.end&&e.media_end_seconds>w.start).map(e=>e.model_label)||[];const aliases=sound.some(x=>/laugh|giggle|chuckle/i.test(x))?'笑声':sound.some(x=>/speech|babbl|conversation/i.test(x))?'说话 咿呀':'';const text=[...w.tags.observed_actions,w.tags.reason,...w.tags.uncertain,...sound,aliases].join('；');searchRows.push({batch:batch.alias,id:w.id,name:f.name,date:f.capture_hint,start:w.start,end:w.end,text,source:f.relative})}}catch(e){document.getElementById('searchStatus').textContent='有一批记录暂不可读，其余已加载内容仍可查找。'}}document.getElementById('searchStatus').textContent='已加载 '+searchRows.length+' 个时间窗；处理中的批次可刷新页面查看新结果。';search()}
function search(){const q=document.getElementById('query').value.trim().toLowerCase(),target=document.getElementById('results');target.replaceChildren();if(!q)return;const matches=searchRows.filter(r=>r.text.toLowerCase().includes(q));document.getElementById('searchStatus').textContent='找到 '+matches.length+' 个时间窗，显示前40个；标签未经人工确认。';for(const r of matches.slice(0,40)){const row=el('article',undefined,'hit'),a=el('a',r.name+' · '+r.date.slice(0,10)+' · 源媒体 '+r.start.toFixed(2)+'—'+r.end.toFixed(2)+'秒');a.href='/batches/'+r.batch+'/index.html#window='+encodeURIComponent(r.id);row.append(a,el('p',r.text),el('code',r.source));target.append(row)}}document.getElementById('query').oninput=search;load();
</script></body></html>'''
    html=html.replace('完整原片继续保留，NAS删除为0。这里汇集的是样片与明确范围的验证批次，尚未完成全年关键事件召回验收。',
                      '本流程不删除NAS原片；已处理样本的完整副本保留。这里汇集的是样片与明确范围的验证批次，全年档案完整性及关键事件召回尚未验收。')
    (root/'library.html').write_text(html.replace('__PAYLOAD__',payload),encoding='utf-8')
    return str(root/'library.html')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--catalog',required=True)
    args=parser.parse_args()
    print(build_library(args.batch,args.catalog))
