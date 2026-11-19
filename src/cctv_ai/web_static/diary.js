const state = {
  data: null,
  period: "day",
  activityFilter: "",
  selectedEventId: null,
  detail: null,
  timezone: "Asia/Shanghai",
  previewUrls: new Map(),
  searchGeneration: 0,
  detailGeneration: 0,
};

const elements = Object.fromEntries(
  [
    "date-input",
    "scope-copy",
    "analysis-beacon",
    "analysis-status",
    "analysis-progress",
    "analysis-progress-copy",
    "analysis-detail",
    "search-index-progress",
    "search-index-progress-copy",
    "life-headline",
    "insight-list",
    "highlight-list",
    "review-copy",
    "review-list",
    "search-form",
    "search-input",
    "search-note",
    "search-results",
    "event-list",
    "timeline-note",
    "clear-filter",
    "evidence-panel",
    "evidence-stage",
    "evidence-image",
    "evidence-video",
    "evidence-empty",
    "evidence-location",
    "evidence-time",
    "evidence-kind",
    "evidence-heading",
    "evidence-meta",
    "clip-button",
    "evidence-confidence",
    "evidence-confidence-copy",
    "evidence-count",
    "evidence-thumbs",
    "toy-list",
    "coverage-list",
    "notice-list",
    "toast",
  ].map((id) => [id.replaceAll("-", "_"), document.getElementById(id)]),
);

const activityLabels = {
  feeding: "喝奶 / 进食",
  play: "玩耍",
  crawling: "爬行姿态",
  sitting: "坐姿",
  standing: "站立",
  lying_awake: "清醒静卧",
  inferred_sleep: "推定睡眠",
};

const precisionLabels = {
  exact: "精确时间",
  mapped: "水印时间",
  estimated: "时间估计",
};

const subtypeLabels = {
  feeding_bottle: "奶瓶喂养",
  breastfeeding: "母乳喂养",
  solid_food: "辅食",
  playing: "玩玩具",
  reading: "亲子阅读",
};

const numberFormatter = new Intl.NumberFormat("zh-CN");

function request(url, options = {}) {
  return fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  }).then(async (response) => {
    if (response.ok) return response.json();
    let message = `请求失败 (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the HTTP fallback.
    }
    throw new Error(message);
  });
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(
    () => elements.toast.classList.remove("visible"),
    2600,
  );
}

function formatTime(timestamp, withDate = false) {
  if (!Number.isFinite(timestamp)) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: state.timezone,
    month: withDate ? "2-digit" : undefined,
    day: withDate ? "2-digit" : undefined,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(timestamp * 1000));
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value} 秒`;
  if (value < 3600) {
    const minutes = Math.floor(value / 60);
    const remainder = value % 60;
    return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分钟`;
  }
  const hours = Math.floor(value / 3600);
  const minutes = Math.round((value % 3600) / 60);
  return minutes ? `${hours} 小时 ${minutes} 分` : `${hours} 小时`;
}

function formatApproxMinutes(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return "1 分钟内";
  return `${Math.max(1, Math.round(value / 60))} 分钟`;
}

function formatFeedingEpisode(episode) {
  const start = formatTime(episode.start_ts);
  const last = Number(episode.last_sighting_ts);
  const hasSecondSighting = Number(episode.sighting_count) > 1
    && Number.isFinite(last)
    && last - Number(episode.start_ts) >= 5 * 60;
  const time = hasSecondSighting
    ? `${start}–${formatTime(last)}`
    : start;
  return `${time}${episode.certainty === "probable" ? "（可能）" : ""}`;
}

function summaryFor(activityType) {
  return state.data?.summary.find(
    (item) => item.activity_type === activityType,
  ) || {
    count: 0,
    duration_seconds: 0,
    average_confidence: 0,
    evidence_events: 0,
  };
}

function bedsideRecordingCoverage() {
  const bedside = state.data?.coverage?.find(
    (item) => item.coverage_mode === "continuous_recording",
  );
  return Number(bedside?.recording_coverage_percent) || 0;
}

function renderSummary() {
  const family = state.data?.family_reference || {};
  for (const activityType of [
    "feeding",
    "play",
    "crawling",
    "sitting",
    "inferred_sleep",
  ]) {
    const summary = summaryFor(activityType);
    const safe = activityType.replaceAll("_", "-");
    const value = document.getElementById(`stat-${safe}`);
    const detail = document.getElementById(`stat-${safe}-detail`);
    const sessionCount = summary.session_count ?? summary.count;
    const sessionDuration =
      summary.session_duration_seconds ?? summary.duration_seconds;
    const fragmentCount = summary.event_count ?? summary.count;
    if (activityType === "feeding") {
      const confirmed = family.feeding?.confirmed_sessions ?? sessionCount;
      const probable = family.feeding?.probable_sessions ?? 0;
      const observed = family.feeding?.visible_sessions ?? confirmed + probable;
      const sightings = family.feeding?.visible_sightings ?? fragmentCount;
      const routine = family.feeding?.routine_estimate || {};
      const estimated = Number(routine.estimated_sessions);
      const hasEstimate = Number.isFinite(estimated) && estimated > 0;
      value.textContent = hasEstimate
        ? `约 ${estimated} 次（醒着时段）`
        : observed
          ? `监控看到 ${observed} 顿`
        : "未发现";
      detail.textContent = observed
        ? `监控看到 ${observed} 顿（${confirmed} 顿明确${probable ? ` + ${probable} 顿可能` : ""}）· ${sightings} 个时点 · 时长未知`
        : "未发现不等于没有发生；镜头外、遮挡或未触发录像都可能漏拍";
    } else if (activityType === "inferred_sleep") {
      const sleep = family.sleep || {};
      const coverage = sleep.recording_coverage_percent ?? bedsideRecordingCoverage();
      const publishDuration = Boolean(sleep.duration_publishable);
      const sleepSeconds = sleep.estimated_seconds ?? sessionDuration;
      const sleepSessions = sleep.sessions ?? sessionCount;
      value.textContent = publishDuration
        ? `约 ${formatDuration(sleepSeconds)}`
        : sleepSessions
          ? "有睡眠候选"
          : "暂无明确窗口";
      const boundaryValue = sleep.max_boundary_error_seconds;
      const boundary = Number(boundaryValue);
      const hasBoundary = boundaryValue != null && Number.isFinite(boundary);
      detail.textContent = sleepSessions
        ? `${sleepSessions} 个状态窗口 · 床头录像覆盖 ${Number(coverage).toFixed(0)}%${hasBoundary ? ` · 最大边界误差约 ${formatApproxMinutes(boundary)}` : ""}`
        : `床头录像覆盖 ${coverage.toFixed(0)}%；没有候选不作睡眠结论`;
    } else {
      value.textContent = sessionCount ? `${sessionCount} 个片段` : "未发现";
      detail.textContent = sessionCount
        ? `${fragmentCount} 段监控目击 · 动作录像不能统计全天时长`
        : "动作触发摄像头可能漏拍；未发现不等于 0 次";
    }
  }
}

function eventStartHour(timestamp) {
  const part = new Intl.DateTimeFormat("zh-CN", {
    timeZone: state.timezone,
    hour: "2-digit",
    hourCycle: "h23",
  })
    .formatToParts(new Date(timestamp * 1000))
    .find((item) => item.type === "hour");
  return Number(part?.value || 0);
}

function timingNote(events) {
  if (!events.length) return "尚无通过自动门槛的时段";
  if (events.length <= 4) {
    return `记录于 ${events.map((item) => formatTime(item.start_ts)).join("、")}`;
  }
  const counts = new Map();
  events.forEach((item) => {
    const hour = eventStartHour(item.start_ts);
    counts.set(hour, (counts.get(hour) || 0) + 1);
  });
  const hours = [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0] - right[0])
    .slice(0, 3)
    .map(([hour]) => `${String(hour).padStart(2, "0")} 时`);
  return `高频起始时段 ${hours.join("、")}`;
}

function topByDuration(events, property) {
  const totals = new Map();
  events.forEach((item) => {
    const key = item[property];
    if (!key) return;
    totals.set(key, (totals.get(key) || 0) + item.duration_seconds);
  });
  return [...totals.entries()].sort(
    (left, right) => right[1] - left[1],
  )[0]?.[0] || null;
}

function appendInsight(label, value, note) {
  const item = document.createElement("div");
  item.className = "insight-item";
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.textContent = value;
  const noteNode = document.createElement("small");
  noteNode.textContent = note;
  item.append(labelNode, valueNode, noteNode);
  elements.insight_list.append(item);
}

function renderInsights() {
  elements.insight_list.replaceChildren();
  const brief = state.data.life_brief;
  if (!brief) return;
  elements.life_headline.textContent = brief.headline;
  const family = state.data.family_reference || {};
  const feeding = family.feeding || brief.feeding;
  const sleep = family.sleep || brief.sleep;
  const sightings = feeding.sightings || brief.feeding.sightings || [];
  const episodes = feeding.episodes || [];
  const feedingTimes = episodes.length
    ? episodes.map(formatFeedingEpisode)
    : sightings.length
      ? sightings.map((item) =>
          `${formatTime(item.start_ts)}${item.certainty === "probable" ? "（可能）" : ""}`,
        )
      : brief.feeding.times.map((value) => formatTime(value));
  const confirmedFeeds = feeding.confirmed_sessions ?? brief.feeding.sessions;
  const probableFeeds = feeding.probable_sessions ?? 0;
  const observedFeeds = feeding.visible_sessions ?? confirmedFeeds + probableFeeds;
  const visibleSightings = feeding.visible_sightings ?? sightings.length;
  const routine = feeding.routine_estimate || {};

  appendInsight(
    "监控看到的喂养",
    observedFeeds
      ? `${observedFeeds} 顿`
      : "未发现",
    observedFeeds
      ? `${confirmedFeeds} 顿明确${probableFeeds ? ` + ${probableFeeds} 顿可能` : ""}；${visibleSightings} 个时点合并为 ${feedingTimes.join("、")}`
      : "镜头外、遮挡或动作录像未触发时可能漏拍；未发现不等于没有发生",
  );

  const estimatedFeeds = Number(routine.estimated_sessions);
  const missedFeeds = Number(routine.likely_unobserved_sessions);
  const targetInterval = Number(routine.target_interval_seconds);
  const likelySlots = Array.isArray(routine.likely_unobserved_slots)
    ? routine.likely_unobserved_slots
    : [];
  const routineStart = Number(routine.routine_window_start_ts);
  const routineEnd = Number(routine.routine_window_end_ts);
  const hasSleepWindow = routine.routine_window_source === "sleep_boundaries"
    && Number.isFinite(routineStart)
    && Number.isFinite(routineEnd);
  const routineWindowCopy = hasSleepWindow
    ? `仅在 ${formatTime(routineStart)} 晨醒至 ${formatTime(routineEnd)} 入睡之间计算；睡着后不补次数。`
    : "只对当前醒着且已有数据的时段计算；睡着后不补次数。";
  const missingCopy = Math.max(0, missedFeeds) > 0
    ? `监控看到 ${observedFeeds} 顿，推测 ${Math.max(0, missedFeeds)} 顿可能未拍到${likelySlots.length ? `，待核对约 ${likelySlots.map((value) => formatTime(value)).join("、")}` : ""}`
    : `监控看到的 ${observedFeeds} 顿已经覆盖本次作息估算，没有补出夜间喂奶`;
  appendInsight(
    "按家庭作息估算",
    Number.isFinite(estimatedFeeds) && estimatedFeeds > 0
      ? `约 ${estimatedFeeds} 顿`
      : "样本不足",
    Number.isFinite(estimatedFeeds) && estimatedFeeds > 0
      ? `${routineWindowCopy}按约每 ${formatDuration(targetInterval)}一次；${missingCopy}`
      : "规律估算不等于已经发生，必须与家庭记录或原片核对",
  );

  const averageFeedingGapValue = routine.average_observed_interval_seconds;
  const averageFeedingGap = Number(averageFeedingGapValue);
  const hasFeedingGap = averageFeedingGapValue != null
    && Number.isFinite(averageFeedingGap);
  appendInsight(
    "监控内顿间隔",
    hasFeedingGap
      ? `平均约 ${formatDuration(averageFeedingGap)}`
      : "样本不足",
    hasFeedingGap
      ? `${routine.cadence_fit === "strong" ? "与约 3 小时一次的作息高度吻合" : "只比较监控看到的相邻顿"}；同一顿内的续喂目击已先合并`
      : "至少需要两顿可见喂养；镜头外喂养不会出现在这里",
  );

  appendInsight(
    "玩耍",
    brief.play.sessions ? `${brief.play.sessions} 个片段` : "未发现",
    brief.play.sessions
      ? `${brief.play.top_location || "位置未知"}出现较多 · 动作录像不能统计全天时长`
      : "尚无通过自动门槛的监控目击；不能解释为没有玩耍",
  );

  appendInsight(
    "爬行与坐姿",
    `爬 ${brief.mobility.crawling_sessions} 个片段 · 坐 ${brief.mobility.sitting_sessions} 个片段`,
    "只记录摄像头看见的姿态片段，不把固定扩窗当成真实活动时长",
  );

  const favoriteToy = brief.play.top_toy;
  appendInsight(
    "最常互动的玩具类型（近似）",
    favoriteToy?.name || "尚未识别",
    favoriteToy
      ? `${favoriteToy.interaction_count} 个可见互动片段；不代表心理偏好，也不统计真实玩耍时长`
      : "未细分玩具不会被误报成“最爱”",
  );

  const sleepCoverage = sleep.recording_coverage_percent ?? bedsideRecordingCoverage();
  const boundaryValue = sleep.max_boundary_error_seconds;
  const boundary = Number(boundaryValue);
  const hasBoundary = boundaryValue != null && Number.isFinite(boundary);
  appendInsight(
    "睡眠总览",
    sleep.duration_publishable
      ? `约 ${formatDuration(sleep.estimated_seconds)}`
      : sleep.sessions
        ? "有睡眠候选"
        : "暂无明确窗口",
    sleep.sessions
      ? `${sleep.sessions} 个状态窗口 · ${sleep.verified_sleep_points ?? "—"} 个睡着锚点 / ${sleep.verified_awake_points ?? "—"} 个清醒锚点 / ${sleep.unclear_points ?? "—"} 个不确定 · 覆盖 ${Number(sleepCoverage).toFixed(0)}%${hasBoundary ? ` · 最大边界误差约 ${formatApproxMinutes(boundary)}` : ""}`
      : `床头录像覆盖 ${Number(sleepCoverage).toFixed(0)}%；没有候选不作睡眠结论`,
  );

  appendInsight(
    "入睡与晨醒参考",
    sleep.evening_sleep_start_ts
      ? `晚间约 ${formatTime(sleep.evening_sleep_start_ts)} 入睡`
      : "晚间边界不足",
    sleep.morning_wake_ts
      ? `晨醒边界约 ${formatTime(sleep.morning_wake_ts)}；录像缺口 ${formatDuration(sleep.recording_gap_seconds || 0)}`
      : `未找到可靠晨醒边界；录像缺口 ${formatDuration(sleep.recording_gap_seconds || 0)}`,
  );
}

function renderFocus() {
  elements.highlight_list.replaceChildren();
  const highlights = state.data.highlights || [];
  if (!highlights.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy compact";
    empty.textContent = "当前范围还没有可回看的重点会话。";
    elements.highlight_list.append(empty);
  }
  highlights.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "highlight-item";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = item.image_url;
    image.alt = item.activity_name;
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = item.subtype_name || item.activity_name;
    const note = document.createElement("small");
    note.textContent = `${formatTime(item.start_ts)} · ${item.reason}`;
    copy.append(title, note);
    button.append(image, copy);
    button.addEventListener("click", () =>
      selectEvent(item.representative_event_id, true),
    );
    elements.highlight_list.append(button);
  });

  elements.review_list.replaceChildren();
  const queue = state.data.review_queue || { items: [], automatic_sessions: 0 };
  elements.review_copy.textContent = queue.items.length
    ? `${queue.items.length} 条待看 · 其余 ${queue.automatic_sessions} 个会话无需逐条复核`
    : `无需人工复核 · ${queue.automatic_sessions} 个会话自动保留`;
  if (!queue.items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy compact";
    empty.textContent = "今天没有会显著改变统计的模糊候选。";
    elements.review_list.append(empty);
    return;
  }
  queue.items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "review-item";
    const time = document.createElement("time");
    time.textContent = formatTime(item.start_ts);
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = item.subtype_name || item.activity_name;
    const note = document.createElement("small");
    note.textContent = item.reason;
    copy.append(title, note);
    const confidence = document.createElement("small");
    confidence.textContent = `${Math.round(item.confidence * 100)}%`;
    button.append(time, copy, confidence);
    button.addEventListener("click", () =>
      selectEvent(item.representative_event_id, true),
    );
    elements.review_list.append(button);
  });
}

function visibleEvents() {
  if (!state.data) return [];
  const sessions = state.data.life_sessions || state.data.events;
  if (!state.activityFilter) return sessions;
  return sessions.filter(
    (event) => event.activity_type === state.activityFilter,
  );
}

function renderEvents() {
  const events = visibleEvents();
  elements.event_list.replaceChildren();
  elements.clear_filter.hidden = !state.activityFilter;
  document.querySelectorAll(".stat-line button").forEach((button) => {
    button.classList.toggle(
      "active",
      button.dataset.activity === state.activityFilter,
    );
  });
  elements.timeline_note.textContent = state.activityFilter
    ? `${activityLabels[state.activityFilter] || state.activityFilter} · ${events.length} 个生活会话`
    : `共 ${events.length} 个生活会话；相邻和跨视角证据已自动合并。`;
  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    const progress = state.data?.analysis?.analysis_file_percent || 0;
    empty.textContent = progress < 100
      ? `当前已分析 ${progress.toFixed(1)}%，暂未找到高置信度活动。`
      : "当前范围没有达到高置信度门槛的活动。";
    elements.event_list.append(empty);
    return;
  }
  for (const event of events) {
    const eventId = event.representative_event_id ?? event.id;
    const row = document.createElement("button");
    row.type = "button";
    row.className = "event-row";
    row.dataset.eventId = eventId;
    row.classList.toggle("selected", eventId === state.selectedEventId);
    const time = document.createElement("time");
    time.textContent = formatTime(event.start_ts);
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = `${event.activity_name}代表画面`;
    if (event.image_url) image.src = event.image_url;
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = event.subtype_name || event.activity_name;
    const meta = document.createElement("small");
    const fragmentCount = event.source_event_count ?? 1;
    if (event.activity_type === "feeding") {
      const certainty = event.certainty === "probable" ? "可能喂养" : "明确喂养";
      meta.textContent = `${event.location} · ${certainty} · ${event.evidence_points || fragmentCount} 个原片证据点 · 时长未知`;
    } else if (event.activity_type === "inferred_sleep") {
      const visibleSeconds = event.visible_seconds ?? event.duration_seconds;
      const sleep = state.data.family_reference?.sleep || {};
      const coverage = sleep.recording_coverage_percent ?? bedsideRecordingCoverage();
      const boundary = Math.max(
        Number(event.start_boundary_error_seconds) || 0,
        Number(event.end_boundary_error_seconds) || 0,
      );
      meta.textContent = sleep.duration_publishable
        ? `${event.location} · 状态推定约 ${formatDuration(visibleSeconds)}${boundary ? ` · 边界误差约 ${formatApproxMinutes(boundary)}` : ""} · ${event.evidence_points || fragmentCount} 个锚点`
        : `${event.location} · 有睡眠候选 · 当天录像覆盖 ${coverage.toFixed(0)}%，暂不发布时长 · ${fragmentCount} 段合并`;
    } else {
      meta.textContent = `${event.location} · 监控目击 · ${fragmentCount} 段证据 · 非全天时长`;
    }
    const precision = document.createElement("span");
    precision.className = "precision";
    precision.textContent =
      precisionLabels[event.time_precision] || event.time_precision;
    meta.append(precision);
    copy.append(title, meta);
    row.append(time, image, copy);
    row.addEventListener("click", () => selectEvent(eventId, true));
    elements.event_list.append(row);
  }
}

function resetEvidenceMedia() {
  elements.evidence_video.pause();
  elements.evidence_video.removeAttribute("src");
  elements.evidence_video.load();
  elements.evidence_video.hidden = true;
  elements.evidence_image.hidden = true;
}

function showEvidenceImage(url, alt) {
  resetEvidenceMedia();
  elements.evidence_image.src = url;
  elements.evidence_image.alt = alt;
  elements.evidence_image.hidden = false;
  elements.evidence_empty.hidden = true;
}

function showEvidenceVideo(url) {
  elements.evidence_image.hidden = true;
  elements.evidence_video.src = url;
  elements.evidence_video.hidden = false;
  elements.evidence_empty.hidden = true;
  elements.evidence_video.load();
  elements.evidence_video.play().catch(() => {});
}

function renderEvidence() {
  const event = state.detail;
  if (!event) {
    resetEvidenceMedia();
    elements.evidence_empty.hidden = false;
    elements.clip_button.disabled = true;
    return;
  }
  const first = event.evidence[0];
  if (first) {
    showEvidenceImage(first.image_url, `${event.activity_name}证据`);
  }
  elements.evidence_location.textContent = event.location;
  elements.evidence_time.textContent = formatTime(event.start_ts, true);
  elements.evidence_kind.textContent =
    event.activity_type === "inferred_sleep"
      ? "AI 状态推定"
      : event.activity_type === "feeding"
        ? event.details?.certainty === "probable"
          ? "可能喂养"
          : "明确喂养"
        : "可见活动";
  elements.evidence_heading.textContent =
    event.subtype_name || event.activity_name;
  const sleepBoundary = Math.max(
    Number(event.details?.start_boundary_error_seconds) || 0,
    Number(event.details?.end_boundary_error_seconds) || 0,
  );
  const sleepReference = state.data?.family_reference?.sleep || {};
  const durationCopy = event.activity_type === "feeding"
    ? `${event.details?.certainty === "probable" ? "可能喂养" : "明确喂养"} 1 次 · 时长未知`
    : event.activity_type === "inferred_sleep"
      ? sleepReference.duration_publishable
        ? `状态窗口约 ${formatDuration(event.duration_seconds)}${sleepBoundary ? ` · 边界误差约 ${formatApproxMinutes(sleepBoundary)}` : ""}`
        : `睡眠候选 · 当天录像覆盖 ${bedsideRecordingCoverage().toFixed(0)}%，暂不发布时长`
      : "监控目击片段 · 不代表全天活动时长";
  elements.evidence_meta.textContent =
    `${durationCopy} · ` +
    `${precisionLabels[event.time_precision] || event.time_precision} · ` +
    `${event.review_status === "machine" ? "尚未人工纠正" : "已有人工结论"}`;
  elements.evidence_confidence.value = event.confidence;
  elements.evidence_confidence_copy.textContent =
    `${Math.round(event.confidence * 100)}%`;
  elements.clip_button.disabled = !first;
  elements.evidence_count.textContent = `${event.evidence.length} 张`;
  elements.evidence_thumbs.replaceChildren();
  event.evidence.forEach((evidence, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.classList.toggle("active", index === 0);
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = evidence.image_url;
    image.alt = `第 ${index + 1} 张支持证据`;
    button.append(image);
    button.addEventListener("click", () => {
      elements.evidence_thumbs
        .querySelectorAll("button")
        .forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      showEvidenceImage(evidence.image_url, image.alt);
    });
    elements.evidence_thumbs.append(button);
  });
}

async function selectEvent(eventId, reveal = false) {
  const generation = ++state.detailGeneration;
  state.selectedEventId = eventId;
  document.querySelectorAll(".event-row").forEach((row) => {
    row.classList.toggle(
      "selected",
      Number(row.dataset.eventId) === eventId,
    );
  });
  try {
    const detail = await request(`/api/activities/${eventId}`);
    if (
      generation !== state.detailGeneration
      || state.selectedEventId !== eventId
    ) return;
    state.detail = detail;
    renderEvidence();
    if (reveal) {
      elements.evidence_panel.scrollIntoView({
        block: "start",
        behavior: "smooth",
      });
    }
  } catch (error) {
    showToast(error.message);
  }
}

function renderToys() {
  elements.toy_list.replaceChildren();
  if (!state.data.toys.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "尚无达到高置信度门槛的玩具互动。";
    elements.toy_list.append(empty);
    return;
  }
  const specific = state.data.toys.filter((toy) => toy.is_specific);
  const unresolved = state.data.toys.filter((toy) => !toy.is_specific);
  [...specific, ...unresolved].forEach((toy, index) => {
    const row = document.createElement("div");
    row.className = "toy-row";
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = toy.is_specific
      ? String(specific.indexOf(toy) + 1).padStart(2, "0")
      : "—";
    const copy = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = toy.name;
    const note = document.createElement("small");
    note.textContent = toy.is_specific
      ? `${toy.interaction_count} 次可见互动`
      : `${toy.interaction_count} 次画面尚未细分`;
    copy.append(name, note);
    const duration = document.createElement("span");
    duration.textContent = "仅按片段";
    row.append(rank, copy, duration);
    elements.toy_list.append(row);
  });
}

function renderCoverage() {
  elements.coverage_list.replaceChildren();
  for (const item of state.data.coverage) {
    const percent = item.coverage_mode === "continuous_recording"
      ? item.recording_coverage_percent
      : item.analysis_file_percent;
    const row = document.createElement("div");
    row.className = "coverage-row";
    const copy = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = item.name;
    const detail = document.createElement("small");
    detail.textContent = item.coverage_mode === "continuous_recording"
      ? `${formatDuration(item.observed_seconds)} 有录像`
      : `${numberFormatter.format(item.files)} 个动作片段`;
    copy.append(name, detail);
    const meter = document.createElement("span");
    meter.className = "coverage-meter";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(100, percent || 0))}%`;
    meter.append(fill);
    const status = document.createElement("span");
    const stableFiles = Number(item.stable_files ?? item.files);
    const sourceNotes = [];
    if (item.unstable_files) {
      sourceNotes.push(`${item.unstable_files} 个写入中`);
    }
    if (item.source_error_files) {
      sourceNotes.push(`${item.source_error_files} 个损坏`);
    }
    status.textContent =
      `${item.analyzed_files}/${stableFiles} 个稳定文件已分析 · ` +
      `${item.precise_time_percent.toFixed(0)}% 精确时间` +
      (sourceNotes.length ? ` · ${sourceNotes.join(" · ")}` : "");
    row.append(copy, meter, status);
    elements.coverage_list.append(row);
  }
}

function renderNotices() {
  elements.notice_list.replaceChildren();
  for (const notice of state.data.notices) {
    const paragraph = document.createElement("p");
    paragraph.textContent = notice;
    elements.notice_list.append(paragraph);
  }
}

function renderScope() {
  const periodNames = { day: "当天", week: "本周", month: "本月" };
  const progress = state.data.analysis.analysis_file_percent;
  const sessions = state.data.life_sessions || state.data.events;
  elements.scope_copy.textContent =
    `${periodNames[state.period]} · ${sessions.length} 个可见生活会话 · ` +
    `${progress.toFixed(1)}% 文件已完成视觉分析`;
}

async function loadDiary(preferredEventId = null) {
  state.detailGeneration += 1;
  state.selectedEventId = null;
  state.detail = null;
  renderEvidence();
  const parameters = new URLSearchParams({
    period: state.period,
  });
  if (elements.date_input.value) {
    parameters.set("date", elements.date_input.value);
  }
  elements.event_list.innerHTML =
    '<div class="loading-line"></div>' +
    '<div class="loading-line short"></div>' +
    '<div class="loading-line"></div>';
  try {
    state.data = await request(`/api/diary?${parameters}`);
    state.timezone = state.data.timezone;
    elements.date_input.value = state.data.selected_date;
    renderScope();
    renderSummary();
    renderInsights();
    renderFocus();
    renderEvents();
    renderToys();
    renderCoverage();
    renderNotices();
    const events = visibleEvents();
    const candidate = events.some(
      (event) =>
        (event.representative_event_id ?? event.id) === preferredEventId,
    )
      ? preferredEventId
      : events[0]?.representative_event_id ?? events[0]?.id ?? null;
    if (candidate) {
      await selectEvent(candidate);
    } else {
      state.selectedEventId = null;
      state.detail = null;
      renderEvidence();
    }
  } catch (error) {
    elements.event_list.innerHTML = `<p class="empty-copy">${error.message}</p>`;
    showToast(error.message);
  }
}

async function refreshProgress() {
  try {
    const progress = await request("/api/visual/progress");
    const percent = Math.max(
      0,
      Math.min(100, Number(progress.progress_percent) || 0),
    );
    elements.analysis_progress.value = percent;
    elements.analysis_progress_copy.textContent = `${percent.toFixed(1)}%`;
    const searchIndex = progress.search_index || {};
    const searchPercent = Math.max(
      0,
      Math.min(100, Number(searchIndex.coverage_percent) || 0),
    );
    elements.search_index_progress.value = searchPercent;
    elements.search_index_progress_copy.textContent =
      `${searchPercent.toFixed(1)}%`;
    const latest = progress.latest_run;
    const complete =
      percent >= 100 && Number(progress.files_pending_schedule || 0) === 0;
    const stableTotal = Number(
      progress.files_stable ?? progress.files_total ?? 0,
    );
    const unstable = Number(progress.files_unstable || 0);
    const sourceErrors = Number(progress.files_source_error || 0);
    elements.analysis_beacon.classList.toggle("complete", complete);
    elements.analysis_status.textContent = complete
      ? "近一个月稳定文件分析已完成"
      : latest?.status === "failed"
        ? "视觉分析需要检查"
        : "近一个月视觉分析进行中";
    const sourceNotes = [];
    if (unstable) {
      sourceNotes.push(
        `${numberFormatter.format(unstable)} 个仍在写入，待下轮确认`,
      );
    }
    if (sourceErrors) {
      sourceNotes.push(
        `${numberFormatter.format(sourceErrors)} 个源文件损坏`,
      );
    }
    elements.analysis_detail.textContent =
      `${numberFormatter.format(progress.files_fully_analyzed)}/` +
      `${numberFormatter.format(stableTotal)} 个稳定文件完成 · ` +
      `${numberFormatter.format(progress.keyframes)} 张关键帧 · ` +
      `${numberFormatter.format(progress.observations)} 条语义观察 · ` +
      `${numberFormatter.format(progress.activity_events)} 段活动 · ` +
      `Qwen3 搜索 ${numberFormatter.format(searchIndex.indexed_keyframes || 0)}/` +
      `${numberFormatter.format(searchIndex.total_keyframes || progress.keyframes)} 张` +
      (searchIndex.active
        ? "，已启用"
        : `，到 ${Number(searchIndex.activation_percent || 95).toFixed(0)}% 自动启用`) +
      (sourceNotes.length ? ` · ${sourceNotes.join(" · ")}` : "");
  } catch {
    elements.analysis_status.textContent = "视觉进度暂不可用";
    elements.search_index_progress_copy.textContent = "—";
  }
}

function renderSearchResults(payload, query) {
  const statusLabels = {
    family_confirmed: "家人已确认",
    machine_event: "已有事件记录",
    qwen_verified: "Qwen3 已确认",
    possible: "可能相关",
  };
  elements.search_results.replaceChildren();
  payload.results.forEach((result) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "search-result";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = result.image_url;
    image.alt = result.activity_name || query;
    const time = document.createElement("span");
    time.textContent =
      `${formatTime(result.wall_ts, true)} · ${result.camera_name}`;
    const score = document.createElement("small");
    const status = statusLabels[result.verification_status] || "可能相关";
    score.textContent = result.match_source === "structured_event"
      ? `${status} · ${result.activity_name}`
      : `${status} · 相似度 ${Number(result.score).toFixed(3)}`;
    item.append(image, time, score);
    item.addEventListener("click", () => {
      if (result.activity_event_id) {
        selectEvent(Number(result.activity_event_id), true);
        return;
      }
      state.selectedEventId = null;
      elements.clip_button.disabled = true;
      showEvidenceImage(result.image_url, query);
      elements.evidence_location.textContent = result.camera_name;
      elements.evidence_time.textContent = formatTime(result.wall_ts, true);
      elements.evidence_kind.textContent = status;
      elements.evidence_heading.textContent = query;
      elements.evidence_meta.textContent =
        `${result.relative_path.split("/").pop()} @ ` +
        `${result.media_offset_seconds.toFixed(1)}s`;
      if (window.matchMedia("(max-width: 1000px)").matches) {
        elements.evidence_panel.scrollIntoView({ block: "start" });
      }
    });
    elements.search_results.append(item);
  });
  elements.search_results.hidden = false;
  const counts = payload.status_counts || {};
  const known = (counts.family_confirmed || 0) + (counts.machine_event || 0);
  const verified = counts.qwen_verified || 0;
  const possible = counts.possible || 0;
  elements.search_note.textContent =
    `全库 ${numberFormatter.format(payload.indexed_keyframes)} 张索引画面 · ` +
    `${known} 条已有事件 · ${verified} 条深度确认 · ${possible} 条可能相关。` +
    payload.notice;
}

async function search(event) {
  event.preventDefault();
  const query = elements.search_input.value.trim();
  if (!query) return;
  const generation = ++state.searchGeneration;
  const button = elements.search_form.querySelector("button");
  button.disabled = true;
  button.textContent = "搜索中…";
  elements.search_note.textContent =
    "先查全部录像中的已有事件和相似画面，不会等待深度模型后才显示。";
  let initialPayload = null;
  const parameters = new URLSearchParams({
    q: query,
    limit: "30",
    deep_verify: "false",
  });
  try {
    initialPayload = await request(`/api/visual/search?${parameters}`);
    if (generation !== state.searchGeneration) return;
    renderSearchResults(initialPayload, query);
  } catch (error) {
    elements.search_note.textContent = error.message;
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "搜索全部录像";
  }
  if (!initialPayload?.deep_verification_available) return;
  parameters.set("deep_verify", "true");
  elements.search_note.textContent += " 正在后台深度核验前排结果…";
  request(`/api/visual/search?${parameters}`)
    .then((payload) => {
      if (generation !== state.searchGeneration) return;
      renderSearchResults(payload, query);
    })
    .catch((error) => {
      if (generation !== state.searchGeneration) return;
      elements.search_note.textContent += ` 深度核验暂不可用：${error.message}`;
    });
}

async function playClip() {
  if (!state.selectedEventId) {
    showToast("请先选择一个有原片证据的活动");
    return;
  }
  elements.clip_button.disabled = true;
  elements.clip_button.textContent = "生成中…";
  elements.evidence_stage.setAttribute("aria-busy", "true");
  showToast("正在生成原片片段…");
  try {
    let url = state.previewUrls.get(state.selectedEventId);
    if (!url) {
      const payload = await request(
        `/api/activities/${state.selectedEventId}/preview`,
        { method: "POST" },
      );
      url = payload.url;
      state.previewUrls.set(state.selectedEventId, url);
    }
    showEvidenceVideo(url);
    elements.clip_button.textContent = "原片片段已就绪";
    elements.evidence_panel.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    elements.clip_button.textContent = "重试播放";
    showToast(error.message);
  } finally {
    elements.clip_button.disabled = false;
    elements.evidence_stage.removeAttribute("aria-busy");
  }
}

async function review(verdict) {
  if (!state.selectedEventId) return;
  try {
    await request(`/api/activities/${state.selectedEventId}/review`, {
      method: "POST",
      body: JSON.stringify({ verdict, note: "" }),
    });
    showToast(verdict === "rejected" ? "已排除这条识别" : "人工结论已保存");
    await loadDiary(state.selectedEventId);
  } catch (error) {
    showToast(error.message);
  }
}

elements.search_form.addEventListener("submit", search);
// Listen in the capture phase so Safari cannot lose the playback action when
// the sticky evidence panel swaps an image for a video element.
document.addEventListener("click", (event) => {
  const target = event.target instanceof Element
    ? event.target.closest("#clip-button")
    : null;
  if (!target) return;
  event.preventDefault();
  playClip();
}, true);
elements.date_input.addEventListener("change", () => loadDiary());
elements.clear_filter.addEventListener("click", () => {
  state.activityFilter = "";
  state.selectedEventId = null;
  renderEvents();
});

document.querySelectorAll(".period-switch button").forEach((button) => {
  button.addEventListener("click", () => {
    state.period = button.dataset.period;
    document.querySelectorAll(".period-switch button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    state.selectedEventId = null;
    loadDiary();
  });
});

document.querySelectorAll(".stat-line button").forEach((button) => {
  button.addEventListener("click", () => {
    state.activityFilter =
      state.activityFilter === button.dataset.activity
        ? ""
        : button.dataset.activity;
    state.selectedEventId = null;
    renderEvents();
    document.querySelector(".event-column").scrollIntoView({
      block: "start",
      behavior: "smooth",
    });
  });
});

document.querySelectorAll(".event-feedback button").forEach((button) => {
  button.addEventListener("click", () => review(button.dataset.verdict));
});

loadDiary();
refreshProgress();
window.setInterval(refreshProgress, 10000);
