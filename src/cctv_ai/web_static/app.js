const state = {
  data: null,
  selectedEpisodeId: null,
  selectedCamera: "",
  previewUrls: new Map(),
  timezone: "Asia/Shanghai",
};

const elements = {
  cameraList: document.querySelector("#camera-list"),
  allCameras: document.querySelector("#all-cameras"),
  lastScan: document.querySelector("#last-scan"),
  archiveCount: document.querySelector("#archive-count"),
  pageTitle: document.querySelector("#page-title"),
  modeFilter: document.querySelector("#mode-filter"),
  dateFilter: document.querySelector("#date-filter"),
  labelFilter: document.querySelector("#label-filter"),
  confidenceFilter: document.querySelector("#confidence-filter"),
  rawCount: document.querySelector("#raw-count"),
  pendingCount: document.querySelector("#pending-count"),
  completedCount: document.querySelector("#completed-count"),
  autoArchiveCount: document.querySelector("#auto-archive-count"),
  pendingLabel: document.querySelector("#pending-label"),
  completedLabel: document.querySelector("#completed-label"),
  archiveLabel: document.querySelector("#archive-label"),
  pipelineStrip: document.querySelector("#pipeline-strip"),
  pipelineBeacon: document.querySelector("#pipeline-beacon"),
  pipelineStatus: document.querySelector("#pipeline-status"),
  pipelineScope: document.querySelector("#pipeline-scope"),
  pipelineProgressCopy: document.querySelector("#pipeline-progress-copy"),
  pipelineTrack: document.querySelector("#pipeline-track"),
  pipelineProbeQueue: document.querySelector("#pipeline-probe-queue"),
  pipelineAudioQueue: document.querySelector("#pipeline-audio-queue"),
  pipelineWatermarkQueue: document.querySelector("#pipeline-watermark-queue"),
  pipelineErrorGroup: document.querySelector("#pipeline-error-group"),
  pipelineErrorCount: document.querySelector("#pipeline-error-count"),
  timelineScope: document.querySelector("#timeline-scope"),
  timelineTitle: document.querySelector("#timeline-title"),
  inboxNote: document.querySelector("#inbox-note"),
  timeline: document.querySelector("#timeline"),
  inspector: document.querySelector(".inspector"),
  visualStage: document.querySelector(".visual-stage"),
  previewVideo: document.querySelector("#preview-video"),
  previewPlaceholder: document.querySelector("#preview-placeholder"),
  previewButton: document.querySelector("#preview-button"),
  selectedCamera: document.querySelector("#selected-camera"),
  selectedTime: document.querySelector("#selected-time"),
  selectedStatus: document.querySelector("#selected-status"),
  inspectorTitle: document.querySelector("#inspector-title"),
  selectedMeta: document.querySelector("#selected-meta"),
  confidenceTrack: document.querySelector("#confidence-track"),
  selectedConfidence: document.querySelector("#selected-confidence"),
  reviewNote: document.querySelector("#review-note"),
  sourceCount: document.querySelector("#source-count"),
  sourceList: document.querySelector("#source-list"),
  toast: document.querySelector("#toast"),
};

const labelNames = {
  baby_cry: "哭声候选",
  speech: "说话候选",
  scream: "尖叫候选",
  yell: "喊叫候选",
  impact: "撞击候选",
  glass_breaking: "玻璃破碎",
  smoke_alarm: "烟雾报警",
  fire_alarm: "火警",
  knock: "敲击候选",
  running_water: "水声候选",
};

const reviewNames = {
  pending: "待复核",
  confirmed: "已确认",
  false_positive: "误报",
  uncertain: "不确定",
  mixed: "部分复核",
};

const numberFormatter = new Intl.NumberFormat("zh-CN");

function formatTime(timestamp, withSeconds = false) {
  if (!timestamp) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: state.timezone,
    hour: "2-digit",
    minute: "2-digit",
    second: withSeconds ? "2-digit" : undefined,
    hour12: false,
  }).format(new Date(timestamp * 1000));
}

function formatDateTime(value) {
  if (!value) return "尚无";
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: state.timezone,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(value));
}

function formatDuration(seconds) {
  const rounded = Math.max(1, Math.round(seconds));
  if (rounded < 60) return `${rounded} 秒`;
  return `${Math.floor(rounded / 60)} 分 ${rounded % 60} 秒`;
}

function fileName(path) {
  return String(path).split("/").pop();
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(
    () => elements.toast.classList.remove("visible"),
    2400,
  );
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the HTTP fallback.
    }
    throw new Error(message);
  }
  return response.json();
}

function renderCameras() {
  elements.cameraList.replaceChildren();
  for (const camera of state.data.cameras) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "camera-item";
    if (camera.id === state.selectedCamera) button.classList.add("active");
    button.dataset.cameraId = camera.id;

    const dot = document.createElement("i");
    dot.className = camera.health_status;
    const name = document.createElement("span");
    name.textContent = camera.name;
    const time = document.createElement("time");
    time.textContent = camera.last_capture_iso
      ? formatDateTime(camera.last_capture_iso)
      : "无录像";
    button.append(dot, name, time);
    button.addEventListener("click", () => {
      state.selectedCamera = camera.id;
      loadDashboard();
    });
    elements.cameraList.append(button);
  }
}

function populateSelect(select, values, selected, labeler) {
  const current = selected;
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labeler(value);
    option.selected = value === current;
    select.append(option);
  }
}

function renderFilters() {
  populateSelect(
    elements.dateFilter,
    state.data.available_dates,
    state.data.selected_date,
    (date) => date,
  );
  populateSelect(
    elements.labelFilter,
    state.data.labels.map((item) => item.label),
    elements.labelFilter.value || "baby_cry",
    (label) => labelNames[label] || label,
  );
  elements.pageTitle.textContent =
    labelNames[elements.labelFilter.value] || elements.labelFilter.value;
}

function renderSummary() {
  const summary = state.data.summary;
  const inbox = state.data.review_inbox;
  const smartMode = inbox.mode === "smart";
  elements.rawCount.textContent = summary.raw_events;
  elements.pendingCount.textContent = inbox.pending;
  elements.completedCount.textContent = inbox.completed;
  elements.autoArchiveCount.textContent = inbox.archived;
  elements.pendingLabel.textContent = smartMode ? "今日还需审核" : "全部待复核";
  elements.completedLabel.textContent = smartMode ? "今日已完成" : "已有结论";
  elements.archiveLabel.textContent = smartMode ? "免人工归档" : "自动归档";
  elements.timelineTitle.textContent = smartMode
    ? "今日智能审核"
    : "全部候选";
  elements.timelineScope.textContent = smartMode
    ? `${state.data.selected_date || "无日期"} · ${inbox.selected}/${inbox.quota} 条固定名单`
    : `${state.data.selected_date || "无日期"} · ${inbox.total_episodes} 段`;
  if (smartMode && inbox.pending === 0 && inbox.selected > 0) {
    elements.inboxNote.textContent =
      "今天完成了，名单不会自动补充；其余候选无需逐条处理。";
  } else if (smartMode) {
    elements.inboxNote.textContent = inbox.strategy === "shadow_active_learning"
      ? `只审核这 ${inbox.selected} 条高信息样本；影子评分不代替人工结论。`
      : `只审核这 ${inbox.selected} 条；其余 ${inbox.archived} 段自动归档并保留。`;
  } else {
    elements.inboxNote.textContent =
      "完整索引仅供抽查，无需把这里的待复核逐条清空。";
  }
  elements.lastScan.textContent = formatDateTime(state.data.scope.last_scan_iso);
  elements.archiveCount.textContent =
    `${numberFormatter.format(state.data.scope.indexed_files)} 个已索引文件`;
}

function pipelineStatus(pipeline) {
  const worker = pipeline.worker;
  if (worker?.alive) {
    return { label: "自动处理运行中", tone: "running" };
  }
  if (worker?.status === "running") {
    return { label: "自动处理未响应", tone: "error" };
  }
  if (worker?.status === "error") {
    return { label: "自动处理已停止", tone: "error" };
  }
  const latest = pipeline.latest_run;
  if (latest?.status === "failed") {
    return { label: "最近批次失败", tone: "error" };
  }
  if (latest?.status === "partial") {
    return { label: "最近批次部分完成", tone: "warning" };
  }
  if (latest?.status === "running") {
    return { label: "批次处理中", tone: "running" };
  }
  if (latest?.status === "completed") {
    return { label: "最近批次已完成", tone: "complete" };
  }
  return { label: "自动处理尚未启动", tone: "idle" };
}

function renderPipeline() {
  const pipeline = state.data?.pipeline;
  if (!pipeline?.configured) {
    elements.pipelineStatus.textContent = "自动处理尚未配置";
    elements.pipelineScope.textContent = "等待设定处理范围";
    elements.pipelineProgressCopy.textContent = "—";
    elements.pipelineTrack.value = 0;
    elements.pipelineBeacon.className = "pipeline-beacon idle";
    elements.pipelineStrip.dataset.tone = "idle";
    elements.pipelineStrip.removeAttribute("title");
    return;
  }

  const progress = pipeline.progress;
  const status = pipelineStatus(pipeline);
  const percent = Math.max(
    0,
    Math.min(100, Number(progress.progress_percent) || 0),
  );
  const cameraNames = (pipeline.scope.camera_ids || []).map((cameraId) => {
    return (
      state.data.cameras.find((camera) => camera.id === cameraId)?.name ||
      cameraId
    );
  });
  const since = pipeline.scope.since_ts
    ? new Intl.DateTimeFormat("zh-CN", {
        timeZone: state.timezone,
        month: "2-digit",
        day: "2-digit",
      }).format(new Date(pipeline.scope.since_ts * 1000))
    : "全部时间";
  const errors =
    progress.error_files +
    progress.audio_error +
    progress.watermark_error;

  elements.pipelineStatus.textContent = status.label;
  elements.pipelineBeacon.className = `pipeline-beacon ${status.tone}`;
  elements.pipelineStrip.dataset.tone = status.tone;
  elements.pipelineScope.textContent =
    `${since} 起 · ${cameraNames.join(" + ") || "所选摄像头"}`;
  elements.pipelineProgressCopy.textContent =
    `${numberFormatter.format(progress.audio_completed)} / ` +
    `${numberFormatter.format(progress.total_files)} · ${percent.toFixed(1)}%`;
  elements.pipelineTrack.value = percent;
  elements.pipelineProbeQueue.textContent = numberFormatter.format(
    progress.waiting_files + progress.ready_files,
  );
  elements.pipelineAudioQueue.textContent = numberFormatter.format(
    progress.audio_queued + progress.audio_running,
  );
  elements.pipelineWatermarkQueue.textContent = numberFormatter.format(
    progress.watermark_queued + progress.watermark_running,
  );
  elements.pipelineErrorCount.textContent = numberFormatter.format(errors);
  elements.pipelineErrorGroup.classList.toggle("has-errors", errors > 0);

  const errorSummary = (pipeline.recent_errors || [])
    .slice(0, 3)
    .map((item) => `${item.stage}: ${item.relative_path}`)
    .join("\n");
  if (errorSummary) {
    elements.pipelineStrip.title = errorSummary;
  } else {
    elements.pipelineStrip.removeAttribute("title");
  }
}

function episodeSubtitle(episode) {
  const cameras = episode.cameras.map((camera) => camera.name).join("、");
  return `${cameras} · ${episode.event_count} 个模型片段`;
}

function renderTimeline() {
  elements.timeline.replaceChildren();
  if (!state.data.episodes.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "当前筛选条件下没有候选事件";
    elements.timeline.append(empty);
    selectEpisode(null);
    return;
  }

  state.data.episodes.forEach((episode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "episode-row";
    button.dataset.episodeId = episode.episode_id;
    button.setAttribute(
      "aria-label",
      `第 ${episode.review_rank} 条，${episode.review_reason}`,
    );
    if (episode.episode_id === state.selectedEpisodeId) {
      button.classList.add("selected");
    }
    if (!["pending", "mixed"].includes(episode.review_status)) {
      button.classList.add("reviewed");
    }

    const time = document.createElement("div");
    time.className = "episode-time";
    time.innerHTML =
      `<strong>${formatTime(episode.start_ts)}</strong>` +
      `<span>${formatDuration(episode.duration_seconds)}</span>`;

    const copy = document.createElement("div");
    copy.className = "episode-copy";
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent =
      episode.duration_seconds > 20 ? "连续声音段" : "短声音段";
    const score = document.createElement("span");
    score.textContent = episode.max_confidence.toFixed(2);
    header.append(title, score);
    const subtitle = document.createElement("p");
    subtitle.textContent = episodeSubtitle(episode);
    const footer = document.createElement("footer");
    const review = document.createElement("span");
    review.innerHTML =
      `<i class="review-dot ${episode.review_status}"></i>` +
      `${reviewNames[episode.review_status] || episode.review_status}`;
    const source = document.createElement("span");
    source.textContent = `${episode.cameras.length} 个视角`;
    const reason = document.createElement("span");
    reason.className = `review-reason ${episode.review_lane}`;
    reason.textContent = episode.review_reason;
    footer.append(review, reason, source);
    copy.append(header, subtitle, footer);
    button.append(time, copy);
    button.addEventListener("click", () => {
      selectEpisode(episode.episode_id, { reveal: true });
    });
    elements.timeline.append(button);
  });
}

function currentEpisode() {
  return state.data?.episodes.find(
    (episode) => episode.episode_id === state.selectedEpisodeId,
  );
}

function resetPreview() {
  elements.previewVideo.pause();
  elements.previewVideo.removeAttribute("src");
  elements.previewVideo.load();
  elements.previewVideo.hidden = true;
  elements.previewPlaceholder.hidden = false;
  elements.visualStage.classList.remove("has-video");
}

function revealEpisodeOnMobile(episode) {
  if (!window.matchMedia("(max-width: 820px)").matches) return;
  showToast(`已打开 ${formatTime(episode.start_ts)} 的片段`);
  window.requestAnimationFrame(() => {
    const top =
      elements.inspector.getBoundingClientRect().top + window.pageYOffset;
    // Numeric scrollTo is supported by older iOS Safari versions that do not
    // reliably handle the options object or smooth scrolling.
    window.scrollTo(0, Math.max(0, Math.round(top)));
  });
}

function selectEpisode(episodeId, { reveal = false } = {}) {
  state.selectedEpisodeId = episodeId;
  document.querySelectorAll(".episode-row").forEach((row) => {
    row.classList.toggle("selected", row.dataset.episodeId === episodeId);
  });
  const episode = currentEpisode();
  if (!episode) {
    elements.previewButton.disabled = true;
    document.querySelectorAll(".review-actions button").forEach(
      (button) => { button.disabled = true; },
    );
    return;
  }
  elements.inspector.scrollTop = 0;

  const representative = episode.representative_event;
  elements.selectedCamera.textContent =
    episode.cameras.map((camera) => camera.name).join(" + ");
  elements.selectedTime.textContent = formatTime(episode.start_ts, true);
  elements.selectedStatus.textContent =
    reviewNames[episode.review_status] || episode.review_status;
  elements.selectedStatus.className = `review-status ${episode.review_status}`;
  elements.inspectorTitle.textContent =
    `${formatTime(episode.start_ts, true)} 的${labelNames[episode.event_label] || episode.event_label}`;
  elements.selectedMeta.textContent =
    `${formatDuration(episode.duration_seconds)} · ` +
    `${episode.event_count} 个候选片段 · ${episode.cameras.length} 个视角` +
    (
      state.data.review_inbox.mode === "smart"
        ? ` · 入选原因：${episode.review_reason}`
        : ""
    ) +
    (
      Number.isFinite(episode.shadow_score)
        ? ` · 影子评分 ${episode.shadow_score.toFixed(2)}（仅抽样）`
        : ""
    );
  elements.selectedConfidence.textContent =
    episode.max_confidence.toFixed(3);
  elements.confidenceTrack.value = episode.max_confidence;
  elements.reviewNote.value = episode.review_note || "";
  elements.previewButton.disabled = false;
  elements.previewButton.textContent = "生成短预览";

  document.querySelectorAll(".review-actions button").forEach((button) => {
    button.disabled = false;
    button.classList.toggle(
      "active",
      button.dataset.verdict === episode.review_status,
    );
  });

  elements.sourceCount.textContent = `${episode.events.length} 个片段`;
  elements.sourceList.replaceChildren();
  for (const event of [...episode.events].reverse()) {
    const row = document.createElement("div");
    row.className = "source-row";
    const time = document.createElement("strong");
    time.textContent = formatTime(event.start_ts, true);
    const copy = document.createElement("div");
    const camera = document.createElement("strong");
    camera.textContent = event.camera_name;
    const path = document.createElement("p");
    path.textContent =
      `${fileName(event.relative_path)} @ ${event.media_start_seconds.toFixed(1)}s`;
    copy.append(camera, path);
    const score = document.createElement("span");
    score.textContent = event.confidence.toFixed(2);
    row.append(time, copy, score);
    elements.sourceList.append(row);
  }

  const previewUrl = state.previewUrls.get(representative.event_id);
  if (previewUrl) {
    showPreview(previewUrl);
    elements.previewButton.textContent = "预览已就绪";
  } else {
    resetPreview();
  }
  if (reveal) {
    revealEpisodeOnMobile(episode);
  }
}

function showPreview(url) {
  elements.previewPlaceholder.hidden = true;
  elements.previewVideo.hidden = false;
  elements.visualStage.classList.add("has-video");
  if (elements.previewVideo.getAttribute("src") !== url) {
    elements.previewVideo.src = url;
    elements.previewVideo.load();
  }
}

async function generatePreview() {
  const episode = currentEpisode();
  if (!episode) return;
  const eventId = episode.representative_event.event_id;
  elements.previewButton.disabled = true;
  elements.previewButton.textContent = "正在生成…";
  try {
    const result = await request(`/api/events/${eventId}/preview`, {
      method: "POST",
    });
    state.previewUrls.set(eventId, result.url);
    showPreview(result.url);
    elements.previewButton.textContent = result.cached ? "预览已就绪" : "预览已生成";
    showToast(result.cached ? "已读取本地预览" : "短预览已在本机生成");
  } catch (error) {
    elements.previewButton.textContent = "重新生成";
    showToast(error.message);
  } finally {
    elements.previewButton.disabled = false;
  }
}

async function reviewEpisode(verdict) {
  const episode = currentEpisode();
  if (!episode) return;
  document.querySelectorAll(".review-actions button").forEach(
    (button) => { button.disabled = true; },
  );
  try {
    await request(`/api/episodes/${episode.episode_id}/review`, {
      method: "POST",
      body: JSON.stringify({
        event_label: episode.event_label,
        event_ids: episode.event_ids,
        verdict,
        note: elements.reviewNote.value.trim(),
      }),
    });
    showToast("人工结论已写入本地索引");
    await loadDashboard(episode.episode_id);
  } catch (error) {
    showToast(error.message);
    document.querySelectorAll(".review-actions button").forEach(
      (button) => { button.disabled = false; },
    );
  }
}

async function loadDashboard(preferredEpisodeId = null) {
  elements.timeline.innerHTML =
    '<div class="timeline-skeleton"></div>' +
    '<div class="timeline-skeleton short"></div>' +
    '<div class="timeline-skeleton"></div>';
  const parameters = new URLSearchParams({
    label: elements.labelFilter.value || "baby_cry",
    minimum_confidence: elements.confidenceFilter.value || "0.60",
    review_mode: elements.modeFilter.value || "smart",
    daily_quota: "5",
  });
  if (elements.dateFilter.value) {
    parameters.set("date", elements.dateFilter.value);
  }
  if (state.selectedCamera) {
    parameters.set("camera", state.selectedCamera);
  }

  try {
    state.data = await request(`/api/dashboard?${parameters}`);
    state.timezone = state.data.timezone;
    elements.modeFilter.value = state.data.review_inbox.mode;
    renderCameras();
    renderFilters();
    renderSummary();
    renderPipeline();
    const candidate =
      preferredEpisodeId &&
      state.data.episodes.some(
        (episode) => episode.episode_id === preferredEpisodeId,
      )
        ? preferredEpisodeId
        : state.data.episodes[0]?.episode_id || null;
    state.selectedEpisodeId = candidate;
    renderTimeline();
    selectEpisode(candidate);
  } catch (error) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.textContent = error.message;
    elements.timeline.replaceChildren(emptyState);
    showToast(error.message);
  }
}

async function refreshPipeline() {
  if (!state.data || document.hidden) return;
  try {
    state.data.pipeline = await request("/api/pipeline");
    renderPipeline();
  } catch (error) {
    elements.pipelineStatus.textContent = "处理状态暂不可用";
    elements.pipelineBeacon.className = "pipeline-beacon error";
    elements.pipelineStrip.dataset.tone = "error";
  }
}

elements.previewButton.addEventListener("click", generatePreview);
elements.modeFilter.addEventListener("change", () => {
  state.selectedEpisodeId = null;
  loadDashboard();
});
elements.dateFilter.addEventListener("change", () => loadDashboard());
elements.labelFilter.addEventListener("change", () => {
  state.selectedEpisodeId = null;
  loadDashboard();
});
elements.confidenceFilter.addEventListener("change", () => loadDashboard());
elements.allCameras.addEventListener("click", () => {
  state.selectedCamera = "";
  loadDashboard();
});
document.querySelectorAll(".review-actions button").forEach((button) => {
  button.disabled = true;
  button.addEventListener("click", () => reviewEpisode(button.dataset.verdict));
});

document.addEventListener("keydown", (event) => {
  if (!state.data?.episodes.length) return;
  if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
  if (["TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  event.preventDefault();
  const current = state.data.episodes.findIndex(
    (episode) => episode.episode_id === state.selectedEpisodeId,
  );
  const offset = event.key === "ArrowDown" ? 1 : -1;
  const next = Math.min(
    state.data.episodes.length - 1,
    Math.max(0, current + offset),
  );
  selectEpisode(state.data.episodes[next].episode_id);
  document
    .querySelector(`[data-episode-id="${state.selectedEpisodeId}"]`)
    ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
});

loadDashboard();
window.setInterval(refreshPipeline, 10000);
