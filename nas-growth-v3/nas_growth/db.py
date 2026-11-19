"""SQLite WAL state database for NAS Unattended Growth Archive V3."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


T2_RETRY_DELAYS = (60, 300)
FILM_RETRY_DELAYS_CAP = 1800

REFUSAL_MODELS = frozenset({"local_care_fallback", "vlm_refusal_pending"})
LEGACY_REFUSAL_CAPTIONS = (
    "温馨亲子日常照护时刻（本地留存）",
    "已自动启用本地隐私柔焦保护留存",
    "本地托底留存",
    "云端自动避让",
)


def _extract_meta_dict(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta")
    if isinstance(meta, dict):
        return meta
    raw = record.get("meta_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def is_unconfirmed_refusal(obj: dict[str, Any] | None, model_name: str = "") -> bool:
    """Return True when a VLM payload, cache row, or milestone represents an unconfirmed policy refusal."""
    m_name = str(model_name or "").strip()
    if not isinstance(obj, dict):
        return m_name in REFUSAL_MODELS
    meta = _extract_meta_dict(obj)
    rev_st = str(obj.get("review_status") or meta.get("review_status") or "").strip()
    if rev_st == "approved":
        return False
    if rev_st == "pending_review":
        return True
    eff_model = str(obj.get("_model") or obj.get("model") or meta.get("model") or m_name).strip()
    eff_prov = str(obj.get("_provider") or obj.get("provider") or meta.get("provider") or "").strip()
    if eff_model in REFUSAL_MODELS or eff_prov in REFUSAL_MODELS:
        return True
    if bool(obj.get("vlm_refused") or meta.get("vlm_refused")):
        return True
    lcf = obj.get("local_care_fallback") if "local_care_fallback" in obj else meta.get("local_care_fallback")
    if lcf in ("yes", "pending_review", True):
        return True
    cap = str(obj.get("caption_zh") or "")
    if cap and any(marker in cap for marker in LEGACY_REFUSAL_CAPTIONS):
        return True
    return False


def normalize_vlm_response(obj: dict[str, Any], model_name: str = "") -> dict[str, Any]:
    """Normalize live and legacy VLM refusal payloads into non-confirmed pending_review shape."""
    if not isinstance(obj, dict):
        return {}
    if not is_unconfirmed_refusal(obj, model_name):
        return dict(obj)
    out = dict(obj)
    cats = out.get("categories") if isinstance(out.get("categories"), list) else []
    suggested = str(out.get("suggested_category") or (cats[0] if cats else "") or "daily_care")
    if suggested == "none":
        suggested = "daily_care"
    cbox = out.get("child_box")
    if not (isinstance(cbox, list) and len(cbox) == 4):
        cbox = [0.0, 0.0, 0.0, 0.0]
    out.update({
        "child_clear": "pending_review",
        "face_or_expression_clear": "no",
        "categories": [],
        "suggested_category": suggested,
        "independence": "unclear",
        "excitement_score": 0,
        "photo_score": 0,
        "child_box": cbox,
        "privacy_exposed": "yes",
        "local_care_fallback": "pending_review",
        "vlm_refused": True,
        "review_status": "pending_review",
        "caption_zh": "云端模型安全策略拒答，已保留原视频区间待人工复核",
    })
    return out


def is_confirmed_milestone_record(record: dict[str, Any] | None) -> bool:
    """Return True only if a milestone/highlight record is confirmed (approved and not an unconfirmed refusal)."""
    if not isinstance(record, dict):
        return False
    meta = _extract_meta_dict(record)
    rev_st = str(record.get("review_status") or meta.get("review_status") or "").strip()
    if rev_st in ("pending_review", "rejected"):
        return False
    return not is_unconfirmed_refusal(record)


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA busy_timeout=15000;

CREATE TABLE IF NOT EXISTS files (
    source_rel TEXT PRIMARY KEY,
    source_root TEXT NOT NULL DEFAULT 'cctv',
    day TEXT NOT NULL,
    camera TEXT NOT NULL,
    wall_ts TEXT,
    size INTEGER NOT NULL DEFAULT 0,
    mtime REAL NOT NULL DEFAULT 0,
    duration REAL NOT NULL DEFAULT 0,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    has_audio INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',
    active_s INTEGER NOT NULL DEFAULT 0,
    person_s INTEGER NOT NULL DEFAULT 0,
    static_suppressed_s INTEGER NOT NULL DEFAULT 0,
    night_sleep_s INTEGER NOT NULL DEFAULT 0,
    t1_mode TEXT NOT NULL DEFAULT 'motion_or_person',
    t1_rel TEXT,
    t1_bytes INTEGER NOT NULL DEFAULT 0,
    t2_rel TEXT,
    t2_bytes INTEGER NOT NULL DEFAULT 0,
    max_box_score REAL NOT NULL DEFAULT 0.0,
    audio_summary_json TEXT NOT NULL DEFAULT '{}',
    candidates_json TEXT NOT NULL DEFAULT '[]',
    bursts_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_at REAL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_day_status ON files(day, status);
CREATE INDEX IF NOT EXISTS idx_files_day_cam ON files(day, camera);
CREATE INDEX IF NOT EXISTS idx_files_root_day ON files(source_root, day, status);
CREATE INDEX IF NOT EXISTS idx_files_queued_pick ON files(
    day,
    CASE WHEN CAST(substr(COALESCE(wall_ts,'12:00:00'),12,2) AS INT) BETWEEN 7 AND 20 THEN 0 ELSE 1 END,
    CASE WHEN size > 60000000 THEN 1 ELSE 0 END,
    wall_ts, source_rel
) WHERE status='queued';

CREATE TABLE IF NOT EXISTS days (
    day TEXT PRIMARY KEY,
    priority_tier TEXT NOT NULL DEFAULT 'P1',
    total_files INTEGER NOT NULL DEFAULT 0,
    done_files INTEGER NOT NULL DEFAULT 0,
    error_files INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    orig_bytes INTEGER NOT NULL DEFAULT 0,
    t1_bytes INTEGER NOT NULL DEFAULT 0,
    t2_bytes INTEGER NOT NULL DEFAULT 0,
    highlight_rel TEXT,
    photos_count INTEGER NOT NULL DEFAULT 0,
    milestones_count INTEGER NOT NULL DEFAULT 0,
    speech_count INTEGER NOT NULL DEFAULT 0,
    summary_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vlm_cache (
    cache_key TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tag TEXT NOT NULL,
    response_json TEXT NOT NULL,
    latency_s REAL NOT NULL DEFAULT 0.0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vlm_stats (
    provider TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    calls_ok INTEGER NOT NULL DEFAULT 0,
    calls_err INTEGER NOT NULL DEFAULT 0,
    consecutive_err INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL NOT NULL DEFAULT 0.0,
    last_error TEXT,
    total_latency_s REAL NOT NULL DEFAULT 0.0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS milestones (
    id TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    camera TEXT NOT NULL,
    category TEXT NOT NULL,
    source_rel TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    caption_zh TEXT NOT NULL,
    speaker_note TEXT NOT NULL DEFAULT '',
    independence TEXT NOT NULL DEFAULT 'unclear',
    score REAL NOT NULL DEFAULT 0.0,
    video_rel TEXT,
    poster_rel TEXT,
    t0_rel TEXT,
    t0_sha256 TEXT,
    review_status TEXT NOT NULL DEFAULT 'approved',
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_milestones_day ON milestones(day, category);

CREATE TABLE IF NOT EXISTS vlm_reviews (
    review_id TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    camera TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_rel TEXT NOT NULL,
    start_s REAL NOT NULL DEFAULT 0.0,
    end_s REAL NOT NULL DEFAULT 0.0,
    suggested_category TEXT NOT NULL DEFAULT 'daily_care',
    status TEXT NOT NULL DEFAULT 'pending_review',
    reason TEXT,
    reviewer_note TEXT NOT NULL DEFAULT '',
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vlm_reviews_status ON vlm_reviews(status, day);

CREATE TABLE IF NOT EXISTS kitchen_candidate_reviews (
    cand_key TEXT PRIMARY KEY,
    day TEXT NOT NULL,
    camera TEXT NOT NULL,
    source_rel TEXT NOT NULL,
    t_s REAL NOT NULL,
    source_size INTEGER NOT NULL DEFAULT 0,
    source_mtime REAL NOT NULL DEFAULT 0,
    frame_sha256 TEXT,
    status TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    response_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kit_rev_day ON kitchen_candidate_reviews(day, status);
CREATE INDEX IF NOT EXISTS idx_kit_rev_sha ON kitchen_candidate_reviews(day, frame_sha256);

CREATE TABLE IF NOT EXISTS film_productions (
    film_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    age_months INTEGER NOT NULL,
    title_zh TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    review_status TEXT NOT NULL DEFAULT 'pending_review',
    reviewer_note TEXT NOT NULL DEFAULT '',
    excluded_shots_json TEXT NOT NULL DEFAULT '[]',
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL,
    last_error TEXT,
    video_rel TEXT,
    poster_rel TEXT,
    card_rel TEXT,
    manifest_rel TEXT,
    duration_s REAL NOT NULL DEFAULT 0.0,
    shots_count INTEGER NOT NULL DEFAULT 0,
    bytes INTEGER NOT NULL DEFAULT 0,
    shots_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    reviewed_at REAL
);

CREATE TABLE IF NOT EXISTS daemon_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cloud_sync_files (
    rel_path TEXT PRIMARY KEY,
    size INTEGER NOT NULL DEFAULT 0,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT,
    synced_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS family_members (
    member_id TEXT PRIMARY KEY,
    label_zh TEXT NOT NULL,
    gender TEXT NOT NULL DEFAULT 'unknown',
    role_group TEXT NOT NULL DEFAULT 'family',
    active_from TEXT NOT NULL DEFAULT '',
    active_until TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    sort_order INTEGER NOT NULL DEFAULT 100,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS family_face_samples (
    sample_id TEXT PRIMARY KEY,
    member_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
    source_note TEXT NOT NULL DEFAULT '',
    thumb_rel TEXT NOT NULL,
    embedding_json TEXT NOT NULL DEFAULT '[]',
    det_score REAL NOT NULL DEFAULT 0.0,
    has_embedding INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_samples_member ON family_face_samples(member_id, created_at);

CREATE TABLE IF NOT EXISTS family_face_ignores (
    face_hash TEXT PRIMARY KEY,
    reason TEXT NOT NULL DEFAULT 'ignored',
    created_at REAL NOT NULL
);
"""


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class ArchiveDB:
    """Thread-safe SQLite repository with WAL persistence."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.executescript(SCHEMA_SQL)
                con.execute("BEGIN IMMEDIATE")
                file_cols = {r["name"] for r in con.execute("PRAGMA table_info(files)")}
                if "retry_count" not in file_cols:
                    con.execute("ALTER TABLE files ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
                if "retry_at" not in file_cols:
                    con.execute("ALTER TABLE files ADD COLUMN retry_at REAL")
                    con.execute(
                        "UPDATE files SET retry_at=updated_at+? WHERE status='error' AND error='t2_decode_failed'",
                        (T2_RETRY_DELAYS[0],),
                    )
                con.execute(
                    "CREATE INDEX IF NOT EXISTS idx_files_due_retry ON files(retry_at) "
                    "WHERE status='error' AND retry_at IS NOT NULL"
                )
                ms_cols = {r["name"] for r in con.execute("PRAGMA table_info(milestones)")}
                if "review_status" not in ms_cols:
                    con.execute("ALTER TABLE milestones ADD COLUMN review_status TEXT NOT NULL DEFAULT 'approved'")
                con.execute("CREATE INDEX IF NOT EXISTS idx_milestones_review ON milestones(review_status, day)")
                for tbl in ("vlm_stats", "vlm_cache"):
                    cols = {r["name"] for r in con.execute(f"PRAGMA table_info({tbl})").fetchall()}
                    for col in ("input_tokens", "output_tokens", "total_tokens"):
                        if col not in cols:
                            con.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
                con.execute(
                    "UPDATE vlm_cache SET input_tokens=520, output_tokens=220, total_tokens=740 "
                    "WHERE total_tokens=0 AND prompt_tag LIKE 'ph|%' "
                    "AND model NOT IN ('local_care_fallback', 'vlm_refusal_pending')"
                )
                con.execute(
                    "UPDATE vlm_cache SET input_tokens=1150, output_tokens=420, total_tokens=1570 "
                    "WHERE total_tokens=0 AND prompt_tag NOT LIKE 'ph|%' "
                    "AND model NOT IN ('local_care_fallback', 'vlm_refusal_pending')"
                )
                # Reconcile cached days.error_files against live files rows on startup
                con.execute(
                    "UPDATE days SET error_files = ("
                    "SELECT COUNT(*) FROM files WHERE files.day = days.day AND files.status = 'error'"
                    ")"
                )
                # Migrate any historical fallback-confirmed milestones to pending_review while respecting manual approvals
                ts_now = time.time()
                approved_revs = {
                    str(r["review_id"])
                    for r in con.execute("SELECT review_id FROM vlm_reviews WHERE status='approved'").fetchall()
                }
                for mr in con.execute(
                    "SELECT id, day, camera, category, source_rel, start_s, end_s, caption_zh, meta_json "
                    "FROM milestones WHERE review_status='approved'"
                ).fetchall():
                    row_d = dict(mr)
                    meta_obj = _extract_meta_dict(row_d)
                    rev_id = f"legacy_ms|{row_d['id']}"
                    if meta_obj.get("review_status") == "approved" or rev_id in approved_revs:
                        continue
                    if not is_unconfirmed_refusal(row_d):
                        continue
                    con.execute("UPDATE milestones SET review_status='pending_review' WHERE id=?", (row_d["id"],))
                    meta_obj["milestone_id"] = row_d["id"]
                    con.execute(
                        """INSERT INTO vlm_reviews(review_id, day, camera, kind, source_rel, start_s, end_s,
                           suggested_category, status, reason, meta_json, created_at, updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(review_id) DO NOTHING""",
                        (
                            rev_id, row_d["day"], row_d["camera"], "milestone", row_d["source_rel"],
                            float(row_d["start_s"]), float(row_d["end_s"]),
                            str(row_d["category"] or "daily_care"), "pending_review",
                            "legacy_local_care_fallback", json.dumps(meta_obj, ensure_ascii=False),
                            ts_now, ts_now,
                        ),
                    )
                zero_rows = con.execute(
                    "SELECT provider, calls_ok FROM vlm_stats WHERE calls_ok > 0 AND total_tokens = 0"
                ).fetchall()
                for zr in zero_rows:
                    prov = zr["provider"]
                    calls_ok = int(zr["calls_ok"] or 0)
                    agg = con.execute(
                        "SELECT COALESCE(SUM(input_tokens),0) AS i, COALESCE(SUM(output_tokens),0) AS o, COALESCE(SUM(total_tokens),0) AS t "
                        "FROM vlm_cache WHERE provider=?",
                        (prov,),
                    ).fetchone()
                    in_t, out_t, tot_t = int(agg["i"] or 0), int(agg["o"] or 0), int(agg["t"] or 0)
                    if tot_t < calls_ok * 500:
                        in_t, out_t, tot_t = calls_ok * 650, calls_ok * 250, calls_ok * 900
                    con.execute(
                        "UPDATE vlm_stats SET input_tokens=?, output_tokens=?, total_tokens=? WHERE provider=?",
                        (in_t, out_t, tot_t, prov),
                    )
                con.commit()
            finally:
                con.close()

    @contextmanager
    def session(self):
        with self._lock:
            con = self._connect()
            try:
                yield con
                con.commit()
            finally:
                con.close()

    def set_state(self, key: str, value: Any) -> None:
        raw = json.dumps(value, ensure_ascii=False)
        ts = time.time()
        with self.session() as con:
            con.execute(
                "INSERT INTO daemon_state(key, value_json, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                (key, raw, ts),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self.session() as con:
            row = con.execute("SELECT value_json FROM daemon_state WHERE key=?", (key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value_json"])
            except Exception:
                return default

    def _sync_days_error_counts(self, con: sqlite3.Connection, days: set[str] | list[str]) -> None:
        for d in sorted({str(x) for x in days if x}):
            con.execute(
                "UPDATE days SET error_files = ("
                "SELECT COUNT(*) FROM files WHERE files.day = days.day AND files.status = 'error'"
                ") WHERE day=?",
                (d,),
            )

    def upsert_discovered_files(self, records: list[dict[str, Any]]) -> int:
        """Return the number of new, changed or recovered sources queued by this scan."""
        if not records:
            return 0
        ts = time.time()
        queued = 0
        dirty_days: set[str] = set()
        with self.session() as con:
            for r in records:
                existing = con.execute(
                    "SELECT size, mtime, status FROM files WHERE source_rel=?", (r["source_rel"],)
                ).fetchone()
                if existing is None:
                    con.execute(
                        "INSERT INTO files(source_rel, source_root, day, camera, wall_ts, size, mtime, status, updated_at) "
                        "VALUES(?,?,?,?,?,?,?,'queued',?)",
                        (r["source_rel"], r.get("source_root", "cctv"), r["day"], r["camera"],
                         r.get("wall_ts"), int(r["size"]), float(r["mtime"]), ts),
                    )
                    queued += 1
                    dirty_days.add(str(r["day"]))
                elif existing["status"] != "running" and (
                    existing["status"] == "missing" or int(existing["size"]) != int(r["size"])
                    or abs(float(existing["mtime"]) - float(r["mtime"])) > 2.0
                ):
                    changed = (
                        int(existing["size"]) != int(r["size"])
                        or abs(float(existing["mtime"]) - float(r["mtime"])) > 2.0
                    )
                    con.execute(
                        "UPDATE files SET size=?, mtime=?, status='queued', source_root=?, error=NULL, "
                        "retry_count=0, retry_at=NULL, updated_at=? WHERE source_rel=?",
                        (int(r["size"]), float(r["mtime"]), r.get("source_root", "cctv"), ts, r["source_rel"]),
                    )
                    if changed:
                        con.execute("DELETE FROM kitchen_candidate_reviews WHERE source_rel=?", (r["source_rel"],))
                    queued += 1
                    dirty_days.add(str(r["day"]))
            if dirty_days:
                self._sync_days_error_counts(con, dirty_days)
        return queued

    def mark_file_done(self, source_rel: str, data: dict[str, Any]) -> None:
        ts = time.time()
        t2_failure = data.get("status") == "error" and data.get("error") == "t2_decode_failed"
        with self.session() as con:
            con.execute(
                """UPDATE files SET
                   status=?, duration=?, width=?, height=?, has_audio=?,
                   active_s=?, person_s=?, static_suppressed_s=?, night_sleep_s=?,
                   t1_mode=?, t1_rel=?, t1_bytes=?, t2_rel=?, t2_bytes=?, max_box_score=?,
                   audio_summary_json=?, candidates_json=?, bursts_json=?, error=?, updated_at=?,
                   retry_at=CASE WHEN ? THEN CASE retry_count WHEN 0 THEN ? WHEN 1 THEN ? END END
                   WHERE source_rel=?""",
                (
                    data.get("status", "done"),
                    float(data.get("duration", 0.0)),
                    int(data.get("width", 0)),
                    int(data.get("height", 0)),
                    int(bool(data.get("has_audio", False))),
                    int(data.get("active_s", 0)),
                    int(data.get("person_s", 0)),
                    int(data.get("static_suppressed_s", 0)),
                    int(data.get("night_sleep_s", 0)),
                    data.get("t1_mode", "motion_or_person"),
                    data.get("t1_rel"),
                    int(data.get("t1_bytes", 0)),
                    data.get("t2_rel"),
                    int(data.get("t2_bytes", 0)),
                    float(data.get("max_box_score", 0.0)),
                    json.dumps(data.get("audio_summary", {}), ensure_ascii=False),
                    json.dumps(data.get("candidates", []), ensure_ascii=False),
                    json.dumps(data.get("bursts", []), ensure_ascii=False),
                    data.get("error"),
                    ts,
                    t2_failure, ts + T2_RETRY_DELAYS[0], ts + T2_RETRY_DELAYS[1],
                    source_rel,
                ),
            )
            con.execute(
                "UPDATE days SET error_files = ("
                "SELECT COUNT(*) FROM files WHERE files.day = days.day AND files.status = 'error'"
                ") WHERE day = (SELECT day FROM files WHERE source_rel = ?)",
                (source_rel,),
            )

    def requeue_due_t2_failures(self, min_day: str, max_day: str) -> int:
        """Claim each due retry once, keeping its budget across daemon restarts."""
        with self.session() as con:
            cur = con.execute(
                "UPDATE files SET status='queued', retry_count=retry_count+1, retry_at=NULL "
                "WHERE status='error' AND retry_at <= ? AND day BETWEEN ? AND ? "
                "AND error='t2_decode_failed' AND retry_count < ?",
                (time.time(), min_day, max_day, len(T2_RETRY_DELAYS)),
            )
            if cur.rowcount:
                con.execute(
                    "UPDATE days SET error_files = ("
                    "SELECT COUNT(*) FROM files WHERE files.day = days.day AND files.status = 'error'"
                    ") WHERE day BETWEEN ? AND ?",
                    (min_day, max_day),
                )
            return cur.rowcount

    def first_live_day(self, p1_start: str, yesterday: str) -> str:
        """Preserve the start of continuous P0 coverage, including upgrades with already indexed live days."""
        key = "first_live_scan_day"
        with self.session() as con:
            con.execute("BEGIN IMMEDIATE")
            saved = con.execute("SELECT value_json FROM daemon_state WHERE key=?", (key,)).fetchone()
            if saved:
                return json.loads(saved["value_json"])
            prior = con.execute(
                "SELECT MIN(day) FROM files WHERE day > ? AND day <= ?", (p1_start, yesterday)
            ).fetchone()[0]
            first = prior or yesterday
            con.execute(
                "INSERT INTO daemon_state(key,value_json,updated_at) VALUES(?,?,?)",
                (key, json.dumps(first), time.time()),
            )
            return first

    def get_queued_files_for_day(self, day: str, limit: int = 50, daytime_first: bool = True, *, light_only: bool = False) -> list[dict[str, Any]]:
        order = (
            "ORDER BY CASE WHEN CAST(substr(COALESCE(wall_ts,'12:00:00'), 12, 2) AS INT) BETWEEN 7 AND 20 THEN 0 ELSE 1 END, "
            "CASE WHEN size > 60000000 THEN 1 ELSE 0 END, "
            "wall_ts ASC, source_rel ASC"
            if daytime_first
            else "ORDER BY wall_ts ASC, source_rel ASC"
        )
        size_filter = "AND size <= 60000000" if light_only else ""
        with self.session() as con:
            rows = con.execute(
                f"SELECT * FROM files WHERE day=? AND status='queued' {size_filter} {order} LIMIT ?",
                (day, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_day_rows(self, day: str) -> list[dict[str, Any]]:
        with self.session() as con:
            rows = con.execute(
                "SELECT * FROM files WHERE day=? ORDER BY wall_ts ASC, source_rel ASC",
                (day,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_day_stats(self, day: str) -> dict[str, Any]:
        with self.session() as con:
            r = con.execute(
                """SELECT
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                   SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errs,
                   SUM(CASE WHEN status='done' THEN size ELSE 0 END) AS orig_bytes,
                   SUM(CASE WHEN status='done' THEN t1_bytes ELSE 0 END) AS t1_bytes,
                   SUM(CASE WHEN status='done' THEN t2_bytes ELSE 0 END) AS t2_bytes,
                   SUM(CASE WHEN status='done' AND active_s > 0 THEN 1 ELSE 0 END) AS active_files
                   FROM files WHERE day=?""",
                (day,),
            ).fetchone()
            return {k: (r[k] or 0) for k in r.keys()} if r else {"total": 0, "done": 0, "queued": 0, "errs": 0}

    def upsert_day_summary(self, day: str, tier: str, status: str, summary: dict[str, Any]) -> None:
        st = self.count_day_stats(day)
        ts = time.time()
        with self.session() as con:
            con.execute(
                """INSERT INTO days(day, priority_tier, total_files, done_files, error_files, status,
                   orig_bytes, t1_bytes, t2_bytes, highlight_rel, photos_count, milestones_count, speech_count,
                   summary_json, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(day) DO UPDATE SET
                   priority_tier=excluded.priority_tier,
                   total_files=excluded.total_files,
                   done_files=excluded.done_files,
                   error_files=excluded.error_files,
                   status=excluded.status,
                   orig_bytes=excluded.orig_bytes,
                   t1_bytes=excluded.t1_bytes,
                   t2_bytes=excluded.t2_bytes,
                   highlight_rel=excluded.highlight_rel,
                   photos_count=excluded.photos_count,
                   milestones_count=excluded.milestones_count,
                   speech_count=excluded.speech_count,
                   summary_json=excluded.summary_json,
                   updated_at=excluded.updated_at""",
                (
                    day, tier, st["total"], st["done"], st["errs"], status,
                    int(st["orig_bytes"]), int(st["t1_bytes"]), int(st["t2_bytes"]),
                    summary.get("highlight_rel"),
                    int(len(summary.get("photos", []))),
                    int(summary.get("milestones_count", 0)),
                    int(summary.get("speech_count", 0)),
                    json.dumps(summary, ensure_ascii=False),
                    ts,
                ),
            )

    def get_next_incomplete_historical_day(self, p0_days: set[str], min_day: str, max_day: str) -> str | None:
        placeholders = ",".join("?" for _ in p0_days) if p0_days else "'__none__'"
        params = list(p0_days) + [min_day, max_day]
        with self.session() as con:
            row = con.execute(
                f"""SELECT day FROM files
                    WHERE status='queued' AND day NOT IN ({placeholders}) AND day >= ? AND day <= ?
                    GROUP BY day ORDER BY day DESC LIMIT 1""",
                params,
            ).fetchone()
            return row["day"] if row else None

    def get_incomplete_days(self, min_day: str, max_day: str) -> list[str]:
        with self.session() as con:
            rows = con.execute(
                "SELECT day FROM files WHERE status='queued' AND day >= ? AND day <= ? GROUP BY day ORDER BY day DESC",
                (min_day, max_day),
            ).fetchall()
            return [str(r["day"]) for r in rows]

    def get_day_queue_weight_counts(self, days: list[str], heavy_bytes: int = 60_000_000) -> dict[str, dict[str, int]]:
        if not days:
            return {}
        placeholders = ",".join("?" for _ in days)
        params: list[Any] = [int(heavy_bytes), int(heavy_bytes), *days]
        with self.session() as con:
            rows = con.execute(
                f"""SELECT day,
                           SUM(CASE WHEN size <= ? THEN 1 ELSE 0 END) AS light,
                           SUM(CASE WHEN size > ? THEN 1 ELSE 0 END) AS heavy
                    FROM files
                    WHERE status='queued' AND day IN ({placeholders})
                    GROUP BY day""",
                params,
            ).fetchall()
            out = {str(d): {"light": 0, "heavy": 0} for d in days}
            for r in rows:
                out[str(r["day"])] = {"light": int(r["light"] or 0), "heavy": int(r["heavy"] or 0)}
            return out

    def get_vlm_cache(self, cache_key: str) -> dict[str, Any] | None:
        with self.session() as con:
            row = con.execute("SELECT response_json, provider, model FROM vlm_cache WHERE cache_key=?", (cache_key,)).fetchone()
            if row is None:
                return None
            obj = json.loads(row["response_json"])
            obj.setdefault("_provider", row["provider"])
            obj.setdefault("_model", row["model"])
            return normalize_vlm_response(obj, model_name=str(row["model"] or ""))

    def put_vlm_cache(self, cache_key: str, tier: str, provider: str, model: str, tag: str, resp: dict[str, Any], latency_s: float, usage: dict[str, int] | None = None) -> None:
        clean = {k: v for k, v in resp.items() if not k.startswith("_")}
        u = usage or {}
        in_t, out_t = int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
        tot_t = int(u.get("total_tokens", 0) or (in_t + out_t))
        with self.session() as con:
            con.execute(
                "INSERT OR REPLACE INTO vlm_cache(cache_key, tier, provider, model, prompt_tag, response_json, latency_s, input_tokens, output_tokens, total_tokens, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (cache_key, tier, provider, model, tag, json.dumps(clean, ensure_ascii=False), float(latency_s), in_t, out_t, tot_t, time.time()),
            )

    def get_vlm_cooldown(self, provider: str) -> float:
        with self.session() as con:
            row = con.execute("SELECT cooldown_until FROM vlm_stats WHERE provider=?", (provider,)).fetchone()
            return float(row["cooldown_until"]) if row else 0.0

    def prune_retired_vlm_stats(self, active_providers: set[str]) -> int:
        if not active_providers:
            return 0
        provs = sorted(str(p) for p in active_providers if p)
        if not provs:
            return 0
        placeholders = ",".join("?" for _ in provs)
        with self.session() as con:
            cur = con.execute(f"DELETE FROM vlm_stats WHERE provider NOT IN ({placeholders})", provs)
            return int(cur.rowcount or 0)

    def record_vlm_result(self, provider: str, model: str, ok: bool, latency_s: float, err_msg: str | None = None, cooldown_s: int = 900, usage: dict[str, int] | None = None) -> None:
        ts = time.time()
        u = usage or {}
        in_t, out_t = (int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)) if ok else (0, 0)
        tot_t = int(u.get("total_tokens", 0) or (in_t + out_t)) if ok else 0
        with self.session() as con:
            con.execute(
                """INSERT INTO vlm_stats(provider, model, calls_ok, calls_err, consecutive_err, cooldown_until,
                           last_error, total_latency_s, input_tokens, output_tokens, total_tokens, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(provider) DO UPDATE SET
                       model=excluded.model, calls_ok=calls_ok+excluded.calls_ok, calls_err=calls_err+excluded.calls_err,
                       consecutive_err=CASE WHEN excluded.calls_ok THEN 0 ELSE consecutive_err+1 END,
                       cooldown_until=CASE WHEN cooldown_until > excluded.updated_at
                                           THEN MAX(cooldown_until, excluded.cooldown_until) ELSE excluded.cooldown_until END,
                       last_error=CASE WHEN excluded.calls_err AND excluded.cooldown_until >= cooldown_until
                                       THEN excluded.last_error ELSE last_error END,
                       total_latency_s=total_latency_s+excluded.total_latency_s, input_tokens=input_tokens+excluded.input_tokens,
                       output_tokens=output_tokens+excluded.output_tokens, total_tokens=total_tokens+excluded.total_tokens,
                       updated_at=excluded.updated_at""",
                (provider, model, int(ok), int(not ok), int(not ok), 0.0 if ok else ts + cooldown_s,
                 err_msg, latency_s if ok else 0.0, in_t, out_t, tot_t, ts),
            )

    def upsert_milestone(self, m: dict[str, Any]) -> None:
        rev_status = str(m.get("review_status") or "approved")
        if rev_status != "approved":
            pass
        elif is_unconfirmed_refusal(m):
            rev_status = "pending_review"
        meta = dict(m.get("meta") or {})
        if rev_status == "approved" and meta.get("review_status") != "approved" and is_unconfirmed_refusal({"meta": meta, "caption_zh": m.get("caption_zh")}):
            rev_status = "pending_review"
        with self.session() as con:
            con.execute(
                """INSERT OR REPLACE INTO milestones
                   (id, day, camera, category, source_rel, start_s, end_s, caption_zh, speaker_note,
                    independence, score, video_rel, poster_rel, t0_rel, t0_sha256, review_status, meta_json, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m["id"], m["day"], m["camera"], m["category"], m["source_rel"],
                    float(m["start_s"]), float(m["end_s"]), m["caption_zh"],
                    m.get("speaker_note", ""), m.get("independence", "unclear"),
                    float(m.get("score", 0.0)), m.get("video_rel"), m.get("poster_rel"),
                    m.get("t0_rel"), m.get("t0_sha256"),
                    rev_status,
                    json.dumps(meta, ensure_ascii=False),
                    float(m.get("created_at", time.time())),
                ),
            )

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        with self.session() as con:
            totals = con.execute(
                """SELECT
                   COUNT(*) AS total_files,
                   SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done_files,
                   SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued_files,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS error_files,
                   SUM(CASE WHEN status='done' THEN size ELSE 0 END) AS orig_bytes,
                   SUM(CASE WHEN status='done' THEN t1_bytes ELSE 0 END) AS t1_bytes,
                   SUM(CASE WHEN status='done' THEN t2_bytes ELSE 0 END) AS t2_bytes
                   FROM files"""
            ).fetchone()
            vlm = [dict(r) for r in con.execute("SELECT * FROM vlm_stats ORDER BY calls_ok DESC, provider ASC").fetchall()]
            days_list = [dict(r) for r in con.execute("SELECT day, priority_tier, total_files, done_files, error_files, status, orig_bytes, t1_bytes, t2_bytes, highlight_rel, photos_count, milestones_count, speech_count, updated_at FROM days ORDER BY day DESC LIMIT 400").fetchall()]
            ms_rows = [
                dict(r) for r in con.execute(
                    "SELECT * FROM milestones WHERE review_status='approved' ORDER BY day DESC, score DESC, created_at DESC LIMIT 120"
                ).fetchall()
                if is_confirmed_milestone_record(dict(r))
            ]
            st_rows = {r["key"]: json.loads(r["value_json"]) for r in con.execute("SELECT key, value_json FROM daemon_state").fetchall()}
            ms_agg = con.execute(
                "SELECT COUNT(*) AS cnt, SUM(CASE WHEN category='speech' THEN 1 ELSE 0 END) AS sp_cnt FROM milestones WHERE review_status='approved'"
            ).fetchone()
            rev_agg = con.execute(
                "SELECT COUNT(*) AS cnt FROM vlm_reviews WHERE status='pending_review'"
            ).fetchone()
            tok_in = sum(int(v.get("input_tokens") or 0) for v in vlm)
            tok_out = sum(int(v.get("output_tokens") or 0) for v in vlm)
            tok_tot = sum(int(v.get("total_tokens") or 0) for v in vlm)
            return {
                "totals": {k: (totals[k] or 0) for k in totals.keys()} if totals else {},
                "vlm_stats": vlm,
                "token_totals": {"input_tokens": tok_in, "output_tokens": tok_out, "total_tokens": tok_tot},
                "days": days_list,
                "milestones": ms_rows,
                "archive_counts": {
                    "days_count": len(days_list),
                    "highlight_days": sum(1 for d in days_list if d.get("highlight_rel")),
                    "photos_count": sum(int(d.get("photos_count") or 0) for d in days_list),
                    "milestones_count": int((ms_agg["cnt"] if ms_agg else 0) or 0),
                    "speech_count": int((ms_agg["sp_cnt"] if ms_agg else 0) or 0),
                    "pending_review_count": int((rev_agg["cnt"] if rev_agg else 0) or 0),
                },
                "state": st_rows,
            }

    def get_day_milestones(self, day: str, *, include_pending: bool = False) -> list[dict[str, Any]]:
        with self.session() as con:
            sql = "SELECT * FROM milestones WHERE day=?"
            if not include_pending:
                sql += " AND review_status='approved'"
            sql += " ORDER BY score DESC, created_at DESC"
            rows = [dict(r) for r in con.execute(sql, (day,)).fetchall()]
            if include_pending:
                return rows
            return [r for r in rows if is_confirmed_milestone_record(r)]

    def query_milestones(
        self,
        category: str = "",
        camera: str = "",
        q: str = "",
        limit: int = 200,
        *,
        include_pending: bool = False,
        min_day: str = "",
        max_day: str = "",
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_pending:
            clauses.append("review_status='approved'")
        if min_day:
            clauses.append("day >= ?")
            params.append(min_day)
        if max_day:
            clauses.append("day <= ?")
            params.append(max_day)
        if category and category != "all":
            clauses.append("category=?")
            params.append(category)
        if camera and camera != "all":
            clauses.append("camera=?")
            params.append(camera)
        if q:
            like = f"%{q.strip()}%"
            clauses.append("(caption_zh LIKE ? OR day LIKE ? OR camera LIKE ? OR speaker_note LIKE ?)")
            params.extend([like, like, like, like])
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(500, int(limit))))
        with self.session() as con:
            rows = [
                dict(r) for r in con.execute(
                    f"SELECT * FROM milestones {where_sql} ORDER BY day DESC, score DESC, created_at DESC LIMIT ?",
                    params,
                ).fetchall()
            ]
            if include_pending:
                return rows
            return [r for r in rows if is_confirmed_milestone_record(r)]

    def upsert_vlm_review(self, item: dict[str, Any], *, force: bool = False) -> None:
        ts = time.time()
        meta = dict(item.get("meta") or {})
        with self.session() as con:
            existing = con.execute("SELECT status, reviewer_note FROM vlm_reviews WHERE review_id=?", (item["review_id"],)).fetchone()
            if existing is not None and not force and existing["status"] in ("approved", "rejected"):
                return
            status = str(item.get("status") or "pending_review")
            con.execute(
                """INSERT INTO vlm_reviews
                   (review_id, day, camera, kind, source_rel, start_s, end_s, suggested_category,
                    status, reason, reviewer_note, meta_json, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(review_id) DO UPDATE SET
                       day=excluded.day,
                       camera=excluded.camera,
                       kind=excluded.kind,
                       source_rel=excluded.source_rel,
                       start_s=excluded.start_s,
                       end_s=excluded.end_s,
                       suggested_category=excluded.suggested_category,
                       status=excluded.status,
                       reason=excluded.reason,
                       meta_json=excluded.meta_json,
                       updated_at=excluded.updated_at""",
                (
                    item["review_id"], item["day"], item["camera"], item["kind"], item["source_rel"],
                    float(item.get("start_s", 0.0)), float(item.get("end_s", 0.0)),
                    str(item.get("suggested_category") or "daily_care"),
                    status, item.get("reason") or "", item.get("reviewer_note") or "",
                    json.dumps(meta, ensure_ascii=False),
                    float(item.get("created_at", ts)), ts,
                ),
            )

    def get_vlm_review(self, review_id: str) -> dict[str, Any] | None:
        with self.session() as con:
            row = con.execute("SELECT * FROM vlm_reviews WHERE review_id=?", (review_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["meta"] = _extract_meta_dict(d)
            return d

    def list_vlm_reviews(self, status: str = "", day: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            clauses.append("status=?")
            params.append(status)
        if day:
            clauses.append("day=?")
            params.append(day)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(500, int(limit))))
        with self.session() as con:
            rows = con.execute(
                f"SELECT * FROM vlm_reviews {where_sql} ORDER BY day DESC, updated_at DESC LIMIT ?",
                params,
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["meta"] = _extract_meta_dict(d)
                out.append(d)
            return out

    def decide_vlm_review(self, review_id: str, status: str, category: str | None = None, note: str = "") -> dict[str, Any] | None:
        if status not in ("approved", "rejected", "pending_review"):
            raise ValueError(f"Unsupported review status: {status}")
        ts = time.time()
        with self.session() as con:
            row = con.execute("SELECT * FROM vlm_reviews WHERE review_id=?", (review_id,)).fetchone()
            if row is None:
                return None
            eff_cat = str(category or row["suggested_category"] or "daily_care")
            con.execute(
                "UPDATE vlm_reviews SET status=?, suggested_category=?, reviewer_note=?, updated_at=? WHERE review_id=?",
                (status, eff_cat, str(note or row["reviewer_note"] or ""), ts, review_id),
            )
            meta = _extract_meta_dict(dict(row))
            ms_id = str(meta.get("milestone_id") or "")
            if not ms_id and review_id.startswith("legacy_ms|"):
                ms_id = review_id[len("legacy_ms|") :]
            if ms_id:
                ms_row = con.execute("SELECT id, meta_json, caption_zh FROM milestones WHERE id=?", (ms_id,)).fetchone()
                if ms_row is not None:
                    ms_meta = _extract_meta_dict(dict(ms_row))
                    ms_meta["review_status"] = status
                    ms_meta["reviewed_at"] = ts
                    new_cap = note or ms_row["caption_zh"]
                    if status == "approved" and any(m in str(new_cap) for m in LEGACY_REFUSAL_CAPTIONS):
                        new_cap = "人工复核确认 · 温馨亲子日常照护"
                    con.execute(
                        "UPDATE milestones SET review_status=?, category=?, caption_zh=?, meta_json=? WHERE id=?",
                        (status, eff_cat, new_cap, json.dumps(ms_meta, ensure_ascii=False), ms_id),
                    )
            updated = con.execute("SELECT * FROM vlm_reviews WHERE review_id=?", (review_id,)).fetchone()
            d = dict(updated)
            d["meta"] = _extract_meta_dict(d)
            return d

    def get_kitchen_reviews_for_day(self, day: str) -> dict[str, dict[str, Any]]:
        with self.session() as con:
            rows = con.execute(
                "SELECT * FROM kitchen_candidate_reviews WHERE day=? ORDER BY updated_at ASC",
                (day,),
            ).fetchall()
            out: dict[str, dict[str, Any]] = {}
            for r in rows:
                d = dict(r)
                try:
                    d["response"] = json.loads(d.get("response_json") or "{}")
                except ValueError:
                    d["response"] = {}
                out[str(d["cand_key"])] = d
            return out

    def get_kitchen_review_by_frame_sha(self, day: str, frame_sha256: str) -> dict[str, Any] | None:
        if not frame_sha256:
            return None
        with self.session() as con:
            row = con.execute(
                "SELECT * FROM kitchen_candidate_reviews WHERE day=? AND frame_sha256=? ORDER BY updated_at DESC LIMIT 1",
                (day, frame_sha256),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            try:
                d["response"] = json.loads(d.get("response_json") or "{}")
            except ValueError:
                d["response"] = {}
            return d

    def upsert_kitchen_review(self, rec: dict[str, Any]) -> None:
        ts = time.time()
        with self.session() as con:
            con.execute(
                """INSERT INTO kitchen_candidate_reviews
                   (cand_key, day, camera, source_rel, t_s, source_size, source_mtime,
                    frame_sha256, status, provider, model, response_json, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(cand_key) DO UPDATE SET
                       source_size=excluded.source_size,
                       source_mtime=excluded.source_mtime,
                       frame_sha256=COALESCE(excluded.frame_sha256, frame_sha256),
                       status=excluded.status,
                       provider=excluded.provider,
                       model=excluded.model,
                       response_json=excluded.response_json,
                       updated_at=excluded.updated_at""",
                (
                    rec["cand_key"], rec["day"], rec["camera"], rec["source_rel"],
                    float(rec.get("t_s", 0.0)), int(rec.get("source_size", 0)), float(rec.get("source_mtime", 0.0)),
                    rec.get("frame_sha256"), rec["status"], rec.get("provider") or "", rec.get("model") or "",
                    json.dumps(rec.get("response") or {}, ensure_ascii=False), ts,
                ),
            )

    @staticmethod
    def _decode_film_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        d = dict(row)
        for jkey, outkey in (("shots_json", "shots"), ("excluded_shots_json", "excluded_shots")):
            try:
                val = json.loads(d.get(jkey) or "[]")
                d[outkey] = val if isinstance(val, list) else []
            except ValueError:
                d[outkey] = []
        return d

    def enqueue_film_production(
        self,
        spec: dict[str, Any],
        *,
        force: bool = False,
        preserve_rendering: bool = False,
    ) -> dict[str, Any]:
        ts = time.time()
        film_id = str(spec["film_id"])
        with self.session() as con:
            existing = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            if existing is not None:
                if preserve_rendering and str(existing["status"] or "") == "rendering":
                    return self._decode_film_row(existing)
                if not force:
                    return self._decode_film_row(existing)
            if existing is None:
                con.execute(
                    """INSERT INTO film_productions
                       (film_id, kind, age_months, title_zh, period_start, period_end, status,
                        review_status, excluded_shots_json, created_at, updated_at)
                       VALUES(?,?,?,?,?,?,'queued','pending_review',?,?,?)""",
                    (
                        film_id, spec["kind"], int(spec["age_months"]), spec["title_zh"],
                        spec["period_start"], spec["period_end"],
                        json.dumps(spec.get("excluded_shots") or [], ensure_ascii=False),
                        ts, ts,
                    ),
                )
            else:
                con.execute(
                    """UPDATE film_productions SET
                       kind=?, age_months=?, title_zh=?, period_start=?, period_end=?,
                       status='queued', retry_at=NULL, last_error=NULL, updated_at=?
                       WHERE film_id=?""",
                    (spec["kind"], int(spec["age_months"]), spec["title_zh"],
                     spec["period_start"], spec["period_end"], ts, film_id),
                )
            row = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            assert row is not None
            return self._decode_film_row(row)

    def get_film_production(self, film_id: str) -> dict[str, Any] | None:
        with self.session() as con:
            row = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            return self._decode_film_row(row) if row is not None else None

    def list_film_productions(self) -> list[dict[str, Any]]:
        with self.session() as con:
            rows = con.execute(
                """SELECT * FROM film_productions
                   ORDER BY CASE kind WHEN 'growth_chronicle' THEN 0 WHEN 'annual' THEN 1 WHEN 'seasonal' THEN 2 WHEN 'monthly' THEN 3 WHEN 'camera_chronicle' THEN 4 ELSE 5 END,
                            age_months DESC, period_end DESC"""
            ).fetchall()
            return [self._decode_film_row(r) for r in rows]

    def next_ready_film_production(self, now: float | None = None) -> dict[str, Any] | None:
        ts = time.time() if now is None else float(now)
        with self.session() as con:
            row = con.execute(
                """SELECT * FROM film_productions
                   WHERE status='queued' OR (status='retry_wait' AND COALESCE(retry_at, 0) <= ?)
                   ORDER BY CASE kind WHEN 'growth_chronicle' THEN 0 WHEN 'monthly' THEN 1 WHEN 'camera_chronicle' THEN 2 WHEN 'seasonal' THEN 3 WHEN 'annual' THEN 4 ELSE 5 END,
                            age_months ASC, created_at ASC LIMIT 1""",
                (ts,),
            ).fetchone()
            return self._decode_film_row(row) if row is not None else None

    def mark_film_rendering(self, film_id: str) -> None:
        with self.session() as con:
            con.execute(
                "UPDATE film_productions SET status='rendering', updated_at=? WHERE film_id=?",
                (time.time(), film_id),
            )

    def reset_stale_rendering_films(self) -> int:
        """Re-queue any film_productions row left in 'rendering' after an unclean restart."""
        ts = time.time()
        with self.session() as con:
            cur = con.execute(
                "UPDATE film_productions SET status='queued', updated_at=? WHERE status='rendering'",
                (ts,),
            )
            return int(cur.rowcount or 0)

    def mark_film_published(self, film_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        ts = time.time()
        with self.session() as con:
            con.execute(
                """UPDATE film_productions SET
                   status='published',
                   review_status=?,
                   attempts=0,
                   retry_at=NULL,
                   last_error=NULL,
                   video_rel=?,
                   poster_rel=?,
                   card_rel=?,
                   manifest_rel=?,
                   duration_s=?,
                   shots_count=?,
                   bytes=?,
                   shots_json=?,
                   updated_at=?
                   WHERE film_id=?""",
                (
                    str(data.get("review_status") or "pending_review"),
                    data.get("video_rel"),
                    data.get("poster_rel"),
                    data.get("card_rel"),
                    data.get("manifest_rel"),
                    float(data.get("duration_s", 0.0)),
                    int(data.get("shots_count", 0)),
                    int(data.get("bytes", 0)),
                    json.dumps(data.get("shots") or [], ensure_ascii=False),
                    ts,
                    film_id,
                ),
            )
            row = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            return self._decode_film_row(row) if row is not None else None

    def mark_film_failed(self, film_id: str, error: str, *, now: float | None = None) -> dict[str, Any] | None:
        ts = time.time() if now is None else float(now)
        with self.session() as con:
            row = con.execute("SELECT attempts FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            attempts = (int(row["attempts"] or 0) if row is not None else 0) + 1
            delay = min(FILM_RETRY_DELAYS_CAP, 60 * (2 ** min(5, attempts - 1)))
            con.execute(
                """UPDATE film_productions SET
                   status='retry_wait',
                   attempts=?,
                   retry_at=?,
                   last_error=?,
                   updated_at=?
                   WHERE film_id=?""",
                (attempts, ts + delay, str(error)[:400], ts, film_id),
            )
            updated = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            return self._decode_film_row(updated) if updated is not None else None

    def review_film_production(
        self,
        film_id: str,
        action: str,
        note: str = "",
        exclude_shots: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        ts = time.time()
        with self.session() as con:
            row = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            if row is None:
                return None
            cur = self._decode_film_row(row)
            merged_ex = list(cur.get("excluded_shots") or [])
            if exclude_shots:
                for ex in exclude_shots:
                    if ex not in merged_ex:
                        merged_ex.append(ex)
            if action == "approve":
                if cur.get("status") != "published" or not cur.get("video_rel"):
                    raise ValueError("Cannot approve unpublished film")
                con.execute(
                    """UPDATE film_productions SET
                       review_status='approved', reviewer_note=?, reviewed_at=?, updated_at=?
                       WHERE film_id=?""",
                    (str(note or cur.get("reviewer_note") or ""), ts, ts, film_id),
                )
            elif action == "reject":
                con.execute(
                    """UPDATE film_productions SET
                       review_status='rejected', reviewer_note=?, excluded_shots_json=?, reviewed_at=?, updated_at=?
                       WHERE film_id=?""",
                    (str(note or ""), json.dumps(merged_ex, ensure_ascii=False), ts, ts, film_id),
                )
            elif action in ("retry", "regenerate"):
                con.execute(
                    """UPDATE film_productions SET
                       status='queued', review_status='pending_review', attempts=0, retry_at=NULL,
                       last_error=NULL, reviewer_note=?, excluded_shots_json=?, updated_at=?
                       WHERE film_id=?""",
                    (str(note or ""), json.dumps(merged_ex, ensure_ascii=False), ts, film_id),
                )
            else:
                raise ValueError(f"Unsupported film review action: {action}")
            updated = con.execute("SELECT * FROM film_productions WHERE film_id=?", (film_id,)).fetchone()
            return self._decode_film_row(updated) if updated is not None else None

    def get_historical_queue_counts(self, min_day: str, max_day: str) -> dict[str, int]:
        with self.session() as con:
            r = con.execute(
                """SELECT
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                   SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS error
                   FROM files WHERE day >= ? AND day <= ?""",
                (min_day, max_day),
            ).fetchone()
            return {k: int(r[k] or 0) for k in ("total", "done", "queued", "error")} if r else {"total": 0, "done": 0, "queued": 0, "error": 0}

    def get_verified_days_before(self, cutoff_day: str) -> list[str]:
        with self.session() as con:
            rows = con.execute(
                """SELECT day FROM files
                   WHERE day < ?
                   GROUP BY day
                   HAVING COUNT(*) > 0 AND SUM(CASE WHEN status != 'done' THEN 1 ELSE 0 END) = 0
                   ORDER BY day ASC""",
                (cutoff_day,),
            ).fetchall()
            return [str(r["day"]) for r in rows]

    def get_cloud_sync_map(self) -> dict[str, dict[str, Any]]:
        with self.session() as con:
            rows = con.execute("SELECT rel_path, size, mtime_ns, sha256, synced_at FROM cloud_sync_files").fetchall()
            return {str(r["rel_path"]): dict(r) for r in rows}

    def mark_cloud_synced(self, rel_path: str, size: int, mtime_ns: int, sha256: str) -> None:
        with self.session() as con:
            con.execute(
                "INSERT OR REPLACE INTO cloud_sync_files(rel_path, size, mtime_ns, sha256, synced_at) VALUES(?,?,?,?,?)",
                (rel_path, int(size), int(mtime_ns), sha256, time.time()),
            )

    def get_live_monitor_data(self) -> dict[str, Any]:
        """Return rich real-time progress, throughput, per-day/per-camera queues, and recent files & VLM evaluations."""
        snap = self.get_dashboard_snapshot()
        now = time.time()
        days_map = {d["day"]: d for d in snap["days"]}
        with self.session() as con:
            r15 = con.execute(
                "SELECT COUNT(*) AS c, COALESCE(SUM(size),0) AS b FROM files WHERE status='done' AND updated_at >= ?",
                (now - 900,),
            ).fetchone()
            r60 = con.execute(
                "SELECT COUNT(*) AS c, COALESCE(SUM(size),0) AS b FROM files WHERE status='done' AND updated_at >= ?",
                (now - 3600,),
            ).fetchone()
            c15, b15 = int(r15["c"] or 0), int(r15["b"] or 0)
            c60, b60 = int(r60["c"] or 0), int(r60["b"] or 0)
            files_per_hour = c60 if c60 >= 8 else c15 * 4
            gb_per_hour = round((b60 if c60 >= 8 else b15 * 4) / 1e9, 2)
            queued_total = int(snap["totals"].get("queued_files", 0))
            eta_hours = round(queued_total / float(files_per_hour), 1) if files_per_hour > 0 and queued_total > 0 else None

            day_rows = con.execute(
                """SELECT day,
                          COUNT(*) AS total,
                          SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                          SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued,
                          SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errs,
                          COALESCE(SUM(CASE WHEN status='done' THEN size ELSE 0 END), 0) AS orig_bytes,
                          COALESCE(SUM(CASE WHEN status='done' THEN t1_bytes ELSE 0 END), 0) AS t1_bytes,
                          COALESCE(SUM(CASE WHEN status='done' THEN t2_bytes ELSE 0 END), 0) AS t2_bytes,
                          COALESCE(SUM(CASE WHEN status='done' THEN active_s ELSE 0 END), 0) AS active_s,
                          MAX(CASE WHEN status='done' THEN wall_ts ELSE NULL END) AS latest_done_ts
                   FROM files GROUP BY day ORDER BY day DESC LIMIT 35"""
            ).fetchall()
            day_queues: list[dict[str, Any]] = []
            for idx, dr in enumerate(day_rows):
                d_str = dr["day"]
                dm = days_map.get(d_str, {})
                day_queues.append({
                    "day": d_str,
                    "total": int(dr["total"] or 0),
                    "done": int(dr["done"] or 0),
                    "queued": int(dr["queued"] or 0),
                    "errs": int(dr["errs"] or 0),
                    "orig_bytes": int(dr["orig_bytes"] or 0),
                    "t1_bytes": int(dr["t1_bytes"] or 0),
                    "t2_bytes": int(dr["t2_bytes"] or 0),
                    "active_s": int(dr["active_s"] or 0),
                    "latest_done_ts": dr["latest_done_ts"],
                    "priority_tier": dm.get("priority_tier") or ("P0_LIVE" if idx < 2 else "P1"),
                    "summary_status": dm.get("status", "processing"),
                    "highlight_rel": dm.get("highlight_rel"),
                    "photos_count": int(dm.get("photos_count", 0) or 0),
                    "milestones_count": int(dm.get("milestones_count", 0) or 0),
                    "speech_count": int(dm.get("speech_count", 0) or 0),
                })

            cam_rows = con.execute(
                """SELECT camera,
                          COUNT(*) AS total,
                          SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done,
                          SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued,
                          SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errs,
                          COALESCE(SUM(CASE WHEN status='done' THEN active_s ELSE 0 END), 0) AS active_s,
                          COALESCE(SUM(CASE WHEN status='done' THEN night_sleep_s ELSE 0 END), 0) AS night_sleep_s,
                          COALESCE(SUM(CASE WHEN status='done' THEN static_suppressed_s ELSE 0 END), 0) AS static_suppressed_s,
                          COALESCE(SUM(CASE WHEN status='done' THEN size ELSE 0 END), 0) AS orig_bytes,
                          COALESCE(SUM(CASE WHEN status='done' THEN t1_bytes ELSE 0 END), 0) AS t1_bytes
                   FROM files GROUP BY camera ORDER BY total DESC"""
            ).fetchall()
            camera_queues = [
                {
                    "camera": r["camera"],
                    "total": int(r["total"] or 0),
                    "done": int(r["done"] or 0),
                    "queued": int(r["queued"] or 0),
                    "errs": int(r["errs"] or 0),
                    "active_s": int(r["active_s"] or 0),
                    "night_sleep_s": int(r["night_sleep_s"] or 0),
                    "static_suppressed_s": int(r["static_suppressed_s"] or 0),
                    "orig_bytes": int(r["orig_bytes"] or 0),
                    "t1_bytes": int(r["t1_bytes"] or 0),
                }
                for r in cam_rows
            ]

            recent_rows = con.execute(
                """SELECT source_rel, day, camera, wall_ts, status, ROUND(duration, 1) AS duration,
                          active_s, person_s, size, t1_bytes, t2_bytes, ROUND(max_box_score, 3) AS max_box_score,
                          error, updated_at
                   FROM files WHERE status IN ('done', 'error') ORDER BY updated_at DESC LIMIT 20"""
            ).fetchall()
            recent_files = [dict(r) for r in recent_rows]

            vlm_rows = con.execute(
                """SELECT cache_key, tier, provider, model, prompt_tag, response_json,
                          ROUND(latency_s, 2) AS latency_s, input_tokens, output_tokens, total_tokens, created_at
                   FROM vlm_cache ORDER BY created_at DESC LIMIT 12"""
            ).fetchall()
            recent_vlm: list[dict[str, Any]] = []
            for vr in vlm_rows:
                try:
                    rj = normalize_vlm_response(json.loads(vr["response_json"] or "{}"), model_name=str(vr["model"] or ""))
                except Exception:
                    rj = {}
                ck = str(vr["prompt_tag"] or "")
                parts = ck.split("|")
                recent_vlm.append({
                    "tier": vr["tier"],
                    "model": vr["model"],
                    "latency_s": float(vr["latency_s"] or 0.0),
                    "total_tokens": int(vr["total_tokens"] or 0),
                    "created_at": float(vr["created_at"] or 0.0),
                    "kind": parts[0] if len(parts) > 1 else vr["tier"],
                    "source_hint": parts[1] if len(parts) > 1 else ck[:40],
                    "child_clear": rj.get("child_clear", ""),
                    "review_status": rj.get("review_status", ""),
                    "score": rj.get("photo_score") or rj.get("excitement_score") or rj.get("clarity_score") or rj.get("moment_worth") or rj.get("cinematic_score"),
                    "categories": rj.get("categories") or [],
                    "caption_zh": rj.get("caption_zh") or rj.get("director_note_zh") or rj.get("film_title_zh") or rj.get("brief_zh") or rj.get("stage_zh") or "",
                })

        return {
            **snap,
            "throughput": {
                "done_15m": c15,
                "done_60m": c60,
                "files_per_hour": files_per_hour,
                "gb_per_hour": gb_per_hour,
                "eta_hours": eta_hours,
            },
            "day_queues": day_queues,
            "camera_queues": camera_queues,
            "recent_files": recent_files,
            "recent_vlm": recent_vlm,
        }

    @staticmethod
    def _decode_member_row(row: sqlite3.Row, samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        out = {
            "member_id": str(row["member_id"]),
            "label_zh": str(row["label_zh"]),
            "gender": str(row["gender"] or "unknown"),
            "role_group": str(row["role_group"] or "family"),
            "active_from": str(row["active_from"] or ""),
            "active_until": str(row["active_until"] or ""),
            "notes": str(row["notes"] or ""),
            "enabled": bool(row["enabled"]),
            "sort_order": int(row["sort_order"] or 100),
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
        }
        if samples is not None:
            out["samples"] = samples
            out["sample_count"] = len(samples)
            out["embedding_count"] = sum(1 for s in samples if s.get("has_embedding"))
        return out

    @staticmethod
    def _decode_sample_row(row: sqlite3.Row, *, include_embedding: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "member_id": str(row["member_id"]),
            "source": str(row["source"] or "upload"),
            "source_note": str(row["source_note"] or ""),
            "thumb_rel": str(row["thumb_rel"] or ""),
            "det_score": float(row["det_score"] or 0.0),
            "has_embedding": bool(row["has_embedding"]),
            "created_at": float(row["created_at"] or 0.0),
        }
        if include_embedding:
            try:
                emb = json.loads(row["embedding_json"] or "[]")
                out["embedding"] = [float(v) for v in emb] if isinstance(emb, list) else []
            except ValueError:
                out["embedding"] = []
        return out

    def seed_default_family_members(self, defaults: list[dict[str, Any]]) -> None:
        if not defaults:
            return
        ts = time.time()
        with self.session() as con:
            for idx, m in enumerate(defaults, start=1):
                mid = str(m.get("member_id") or "").strip()
                lbl = str(m.get("label_zh") or "").strip()
                if not mid or not lbl:
                    continue
                con.execute(
                    """INSERT OR IGNORE INTO family_members
                       (member_id, label_zh, gender, role_group, active_from, active_until,
                        notes, enabled, sort_order, created_at, updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mid,
                        lbl,
                        str(m.get("gender") or "unknown"),
                        str(m.get("role_group") or "family"),
                        str(m.get("active_from") or ""),
                        str(m.get("active_until") or ""),
                        str(m.get("notes") or ""),
                        int(bool(m.get("enabled", True))),
                        int(m.get("sort_order", idx * 10)),
                        ts,
                        ts,
                    ),
                )

    def list_family_members(self, include_samples: bool = True) -> list[dict[str, Any]]:
        with self.session() as con:
            rows = con.execute(
                "SELECT * FROM family_members ORDER BY sort_order ASC, created_at ASC, member_id ASC"
            ).fetchall()
            samples_by_member: dict[str, list[dict[str, Any]]] = {}
            if include_samples:
                srows = con.execute(
                    "SELECT * FROM family_face_samples ORDER BY created_at DESC"
                ).fetchall()
                for sr in srows:
                    dec = self._decode_sample_row(sr, include_embedding=False)
                    samples_by_member.setdefault(dec["member_id"], []).append(dec)
            return [
                self._decode_member_row(
                    r, samples=samples_by_member.get(str(r["member_id"]), []) if include_samples else None
                )
                for r in rows
            ]

    def get_family_member(self, member_id: str) -> dict[str, Any] | None:
        mid = str(member_id or "").strip()
        if not mid:
            return None
        with self.session() as con:
            row = con.execute("SELECT * FROM family_members WHERE member_id=?", (mid,)).fetchone()
            if row is None:
                return None
            srows = con.execute(
                "SELECT * FROM family_face_samples WHERE member_id=? ORDER BY created_at DESC", (mid,)
            ).fetchall()
            samples = [self._decode_sample_row(sr, include_embedding=False) for sr in srows]
            return self._decode_member_row(row, samples=samples)

    def upsert_family_member(self, member: dict[str, Any]) -> dict[str, Any]:
        mid = str(member.get("member_id") or "").strip()
        lbl = str(member.get("label_zh") or "").strip()
        if not mid or not lbl:
            raise ValueError("member_id and label_zh are required")
        ts = time.time()
        with self.session() as con:
            existing = con.execute("SELECT * FROM family_members WHERE member_id=?", (mid,)).fetchone()
            created_at = float(existing["created_at"]) if existing is not None else ts
            sort_order = int(
                member["sort_order"]
                if "sort_order" in member and member["sort_order"] is not None
                else (existing["sort_order"] if existing is not None else 100)
            )
            con.execute(
                """INSERT INTO family_members
                   (member_id, label_zh, gender, role_group, active_from, active_until,
                    notes, enabled, sort_order, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(member_id) DO UPDATE SET
                       label_zh=excluded.label_zh,
                       gender=excluded.gender,
                       role_group=excluded.role_group,
                       active_from=excluded.active_from,
                       active_until=excluded.active_until,
                       notes=excluded.notes,
                       enabled=excluded.enabled,
                       sort_order=excluded.sort_order,
                       updated_at=excluded.updated_at""",
                (
                    mid,
                    lbl,
                    str(member.get("gender") if "gender" in member else (existing["gender"] if existing else "unknown")),
                    str(member.get("role_group") if "role_group" in member else (existing["role_group"] if existing else "family")),
                    str(member.get("active_from") if "active_from" in member else (existing["active_from"] if existing else "")),
                    str(member.get("active_until") if "active_until" in member else (existing["active_until"] if existing else "")),
                    str(member.get("notes") if "notes" in member else (existing["notes"] if existing else "")),
                    int(bool(member.get("enabled") if "enabled" in member else (existing["enabled"] if existing else True))),
                    sort_order,
                    created_at,
                    ts,
                ),
            )
            row = con.execute("SELECT * FROM family_members WHERE member_id=?", (mid,)).fetchone()
            srows = con.execute(
                "SELECT * FROM family_face_samples WHERE member_id=? ORDER BY created_at DESC", (mid,)
            ).fetchall()
            return self._decode_member_row(row, samples=[self._decode_sample_row(sr) for sr in srows])

    def delete_family_member(self, member_id: str) -> list[dict[str, Any]]:
        mid = str(member_id or "").strip()
        if not mid:
            return []
        with self.session() as con:
            srows = con.execute("SELECT * FROM family_face_samples WHERE member_id=?", (mid,)).fetchall()
            removed = [self._decode_sample_row(sr) for sr in srows]
            con.execute("DELETE FROM family_face_samples WHERE member_id=?", (mid,))
            con.execute("DELETE FROM family_members WHERE member_id=?", (mid,))
            return removed

    def add_family_face_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        sid = str(sample.get("sample_id") or "").strip()
        mid = str(sample.get("member_id") or "").strip()
        if not sid or not mid:
            raise ValueError("sample_id and member_id are required")
        emb = sample.get("embedding") or []
        has_emb = 1 if (isinstance(emb, list) and len(emb) >= 64) else 0
        ts = float(sample.get("created_at") or time.time())
        with self.session() as con:
            con.execute(
                """INSERT OR REPLACE INTO family_face_samples
                   (sample_id, member_id, source, source_note, thumb_rel, embedding_json, det_score, has_embedding, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    sid,
                    mid,
                    str(sample.get("source") or "upload"),
                    str(sample.get("source_note") or ""),
                    str(sample.get("thumb_rel") or ""),
                    json.dumps(emb if has_emb else [], separators=(",", ":")),
                    float(sample.get("det_score") or 0.0),
                    has_emb,
                    ts,
                ),
            )
            row = con.execute("SELECT * FROM family_face_samples WHERE sample_id=?", (sid,)).fetchone()
            return self._decode_sample_row(row, include_embedding=False)

    def delete_family_face_sample(self, sample_id: str) -> dict[str, Any] | None:
        sid = str(sample_id or "").strip()
        if not sid:
            return None
        with self.session() as con:
            row = con.execute("SELECT * FROM family_face_samples WHERE sample_id=?", (sid,)).fetchone()
            if row is None:
                return None
            out = self._decode_sample_row(row, include_embedding=False)
            con.execute("DELETE FROM family_face_samples WHERE sample_id=?", (sid,))
            return out

    def get_family_face_gallery(self, day: str | None = None) -> list[dict[str, Any]]:
        with self.session() as con:
            rows = con.execute(
                "SELECT * FROM family_members ORDER BY sort_order ASC, created_at ASC, member_id ASC"
            ).fetchall()
            srows = con.execute(
                "SELECT * FROM family_face_samples ORDER BY created_at DESC"
            ).fetchall()
        samples_by_member: dict[str, list[dict[str, Any]]] = {}
        for sr in srows:
            dec = self._decode_sample_row(sr, include_embedding=True)
            samples_by_member.setdefault(dec["member_id"], []).append(dec)
        out: list[dict[str, Any]] = []
        day_s = str(day or "").strip()
        for r in rows:
            m = self._decode_member_row(r, samples=samples_by_member.get(str(r["member_id"]), []))
            if not m["enabled"]:
                continue
            if day_s:
                if m["active_from"] and day_s < m["active_from"]:
                    continue
                if m["active_until"] and day_s > m["active_until"]:
                    continue
            out.append(m)
        return out

    def add_ignored_face(self, face_hash: str, reason: str = "ignored") -> None:
        fh = str(face_hash or "").strip()
        if not fh:
            return
        with self.session() as con:
            con.execute(
                "INSERT OR REPLACE INTO family_face_ignores(face_hash, reason, created_at) VALUES(?,?,?)",
                (fh, str(reason or "ignored"), time.time()),
            )

    def get_ignored_face_hashes(self) -> set[str]:
        with self.session() as con:
            rows = con.execute("SELECT face_hash FROM family_face_ignores").fetchall()
            return {str(r["face_hash"]) for r in rows}
