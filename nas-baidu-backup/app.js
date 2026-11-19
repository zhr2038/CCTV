(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = n => Number(n || 0).toLocaleString('zh-CN');
  function bytes(n, digits = 2) {
    n = Math.max(0, Number(n || 0));
    const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
    const unit = Math.min(5, Math.floor(Math.log10(Math.max(1, n)) / 3));
    return `${(n / 1000 ** unit).toFixed(unit ? digits : 0)} ${units[unit]}`;
  }
  function duration(seconds) {
    if (seconds > 86400) return `约 ${Math.ceil(seconds / 86400)} 天`;
    if (seconds > 3600) return `约 ${Math.ceil(seconds / 3600)} 小时`;
    return `约 ${Math.max(1, Math.ceil(seconds / 60))} 分钟`;
  }
  const phases = {starting:'正在准备', preparing_queue:'正在恢复上传队列', waiting_openlist:'等待网盘服务', scanning:'正在扫描', uploading:'正在上传', paused:'已暂停', failed:'等待重试',
    completed_with_missing:'上传完成，有源文件缺失',
    incomplete:'等待补传', completed:'首次备份完成', completed_with_errors:'等待重试', interrupted:'等待恢复', stopped:'已停止', scan_complete:'扫描完成'};
  let busy = false;
  let requestActive = false;
  let toastTimer;
  function toast(message) {
    $('toast').textContent = message;
    $('toast').classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('show'), 3500);
  }
  function render(data) {
    const totals = data.totals || {};
    const totalBytes = data.scanned_bytes || Object.values(totals).reduce((n, r) => n + r.bytes, 0);
    const totalFiles = data.scanned_files || Object.values(totals).reduce((n, r) => n + r.files, 0);
    const uploadedBytes = data.uploaded_bytes_total ?? totals.uploaded?.bytes ?? 0;
    const uploadedFiles = data.uploaded_files_total ?? totals.uploaded?.files ?? 0;
    const directories = data.directories || [];
    const failures = directories.length ? directories.reduce((n, r) => n + r.failed, 0) : (totals.failed?.files || data.run_failures || 0);
    const missing = data.missing_files ?? totals.missing?.files ?? 0;
    const percent = totalBytes ? Math.min(100, uploadedBytes / totalBytes * 100) : 0;
    const scanning = data.phase === 'scanning';
    $('percent').textContent = scanning ? '—' : percent.toFixed(2);
    $('progress-fill').style.width = `${percent}%`;
    $('progress').setAttribute('aria-valuenow', percent.toFixed(2));
    $('count-summary').textContent = scanning ? `已扫描 ${fmt(totalFiles)} 个文件` : `${fmt(uploadedFiles)} / ${fmt(totalFiles)} 个文件`;
    $('bytes-summary').textContent = `${bytes(uploadedBytes)} / ${scanning ? '正在统计总容量' : bytes(totalBytes)}`;
    $('uploaded-bytes').textContent = bytes(uploadedBytes);
    $('speed').textContent = `${bytes(data.speed_bytes || 0)}/s`;
    $('failed-count').textContent = fmt(failures);
    $('scanned-count').textContent = `${fmt(totalFiles)} 个`;
    $('recent-count').textContent = `${fmt(totals.recent?.files)} 个`;
    $('missing-count').textContent = `${fmt(missing)} 个`;
    $('empty-count').textContent = `${fmt(totals.empty?.files)} 个`;
    const phase = data.pause_requested && data.phase !== 'paused' ? '即将暂停' : (phases[data.phase] || data.phase || '正在准备');
    $('phase').textContent = phase;
    $('phase-badge').querySelector('.dot').classList.toggle('pulse', ['uploading','scanning'].includes(data.phase));
    const history = data.history || [];
    const period = history.length > 1 ? history.at(-1).time - history[0].time : 0;
    $('eta').textContent = data.phase === 'completed' ? '首次清单上传完成' : data.phase === 'paused' ? '继续后恢复上传' :
      scanning ? '扫描完成后显示总进度' : period >= 30 && data.speed_bytes > 0 ? `预计剩余 ${duration((totalBytes-uploadedBytes)/data.speed_bytes)}` : '速度稳定后估算剩余时间';
    $('chart-note').textContent = data.speed_bytes ? `${bytes(data.speed_bytes)}/s` : '等待文件上传完成';
    const speeds = history.slice(1).map((p, i) => Math.max(0, (p.bytes - history[i].bytes) / Math.max(1, p.time-history[i].time)));
    if (speeds.length > 1) {
      const max = Math.max(1, ...speeds);
      const points = speeds.map((v, i) => `${i/(speeds.length-1)*600},${80-v/max*65}`);
      $('chart-line').setAttribute('d', `M${points.join(' L')}`);
      $('chart-area').setAttribute('d', `M0,80 L${points.join(' L')} L600,80 Z`);
    }
    $('current-file').textContent = data.current_file || (scanning ? '正在建立文件清单' : data.phase === 'completed' ? '首次清单已上传完成' : '等待下一个文件');
    $('current-meta').textContent = data.current_file ? `${bytes(data.current_file_bytes)} · ${data.phase === 'paused' ? '已暂停' : '上传中'}` : phase;
    $('file-spinner').style.display = ['uploading','scanning'].includes(data.phase) && !data.pause_requested ? '' : 'none';
    $('schedule-tag').textContent = data.daily_enabled ? '已开启' : '尚未开启';
    $('pause').disabled = busy || data.pause_requested || !['uploading','scanning','starting','waiting_openlist','failed','incomplete'].includes(data.phase);
    $('resume').disabled = busy || !(data.pause_requested || data.phase === 'paused');
    $('retry').disabled = busy || !(failures || ['failed','incomplete','interrupted'].includes(data.phase));
    $('control-status').textContent = data.phase === 'paused' ? '进度已保存，可随时继续。' : data.pause_requested ? '已请求暂停，正在完成当前文件。' : '';
    $('directory-note').textContent = directories.length ? `${directories.length} 个目录` : '全量清单统计中';
    if (directories.length) {
      const rows = directories.map(r => {
        const tr = document.createElement('tr');
        const name = document.createElement('td'); name.textContent = r.name; name.title = r.name;
        const capacity = document.createElement('td'); capacity.textContent = `${bytes(r.uploaded_bytes,1)} / ${bytes(r.bytes,1)}`;
        const count = document.createElement('td'); count.textContent = `${fmt(r.uploaded_files)} / ${fmt(r.files)}`;
        const track = document.createElement('div'); track.className='folder-progress';
        const fill = document.createElement('i'); fill.style.width=`${r.bytes ? Math.min(100,r.uploaded_bytes/r.bytes*100) : 0}%`; track.append(fill); count.append(track);
        const status = document.createElement('td'); const label = document.createElement('span'); label.className=`state-small${r.failed?' error':''}`;
        label.textContent = r.failed ? `${fmt(r.failed)} 个失败` : r.missing ? `${fmt(r.missing)} 个源文件缺失` : r.uploaded_files === r.files ? '已完成' : data.phase === 'paused' ? '已暂停' : data.current_file?.startsWith(`${r.name}/`) ? '上传中' : '等待';
        status.append(label); tr.append(name,capacity,count,status); return tr;
      });
      $('directory-rows').replaceChildren(...rows);
    }
    if (data.events?.length) {
      $('events').replaceChildren(...data.events.slice(0,8).map(e => {
        const row=document.createElement('div'); row.className='event';
        const time=document.createElement('time'); time.dateTime=e.time; time.textContent=e.time.slice(11,19);
        const text=document.createElement('span');text.textContent=e.message;row.append(time,text);return row;
      }));
    }
    const hasError = data.phase === 'failed' || failures > 0 || missing > 0 || data.scan_errors > 0 || data.phase === 'waiting_openlist';
    $('error-banner').style.display = hasError ? 'block' : 'none';
    $('error-banner').textContent = data.scan_errors > 0 ? `有 ${fmt(data.scan_errors)} 个路径读取失败，请检查任务记录。` :
      failures ? `${fmt(failures)} 个文件上传失败，已保留重试。其他文件会继续上传。` :
      missing ? `${fmt(missing)} 个文件在原始录像目录中已不存在，已保留缺失记录；其余文件继续上传。` :
      data.phase === 'waiting_openlist' ? '正在等待 OpenList 启动或网络恢复，每 15 秒自动重试。' : `任务暂时停止，未完成文件保留在队列中。${data.last_error || ''}`;
    const age = data.heartbeat ? Date.now() - Date.parse(data.heartbeat) : Infinity;
    $('connection').textContent = age < 180000 ? 'NAS 已连接' : '心跳延迟';
    $('connection-dot').style.background = age < 180000 ? 'var(--accent)' : 'var(--warn)';
    $('updated').textContent = `更新于 ${data.heartbeat?.slice(11,19) || '—'} · 每 3 秒刷新`;
  }
  async function refresh() {
    if(requestActive) return;
    requestActive=true;
    try {
      const response=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(8000)});
      if(!response.ok) throw new Error();
      render(await response.json());
    } catch {
      $('connection').textContent='连接中断'; $('connection-dot').style.background='var(--warn)';
      $('updated').textContent='暂时无法读取 NAS 状态，正在重连…';
    } finally {requestActive=false;}
  }
  ['pause','resume','retry'].forEach(action => $(action).addEventListener('click',async()=>{
    if(busy) return;busy=true;
    ['pause','resume','retry'].forEach(id=>$(id).disabled=true);
    try{
      const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-CCTV-Control':'1'},body:JSON.stringify({action}),signal:AbortSignal.timeout(8000)});
      if(!response.ok) throw new Error();
      toast({pause:'已请求暂停，当前文件完成后生效',resume:'已继续上传',retry:'已请求重试未完成任务'}[action]);
    }catch{toast('操作未成功，请稍后重试。');}
    finally{busy=false;await refresh();}
  }));
  refresh();setInterval(refresh,3000);
})();
