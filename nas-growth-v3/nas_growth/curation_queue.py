"""Durable day-level publication requests, thread-safe with manual re-curate trigger support."""
from __future__ import annotations

import threading
import time
from typing import Any

from .db import ArchiveDB


class CurationQueue:
    def __init__(self, db: ArchiveDB) -> None:
        self.db = db
        self._lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = dict(db.get_state("curation_queue", {}) or {})

    def _save(self, jobs: dict[str, dict[str, Any]]) -> None:
        self.db.set_state("curation_queue", jobs)
        self.jobs = jobs

    def request(self, day: str, tier: str, final: bool, done_files: int) -> None:
        with self._lock:
            old = self.jobs.get(day, {})
            job = {**old, "tier": tier, "final": final or old.get("final", False),
                   "done_files": done_files}
            if job != old:
                self._save({**self.jobs, day: job})

    def force_request(self, day: str, tier: str, final: bool, done_files: int) -> dict[str, Any]:
        """Enqueue or immediately reset a day curation job with manual force priority."""
        with self._lock:
            old = self.jobs.get(day, {})
            job = {
                **old,
                "tier": tier,
                "final": final or old.get("final", False),
                "done_files": done_files,
                "force_ts": round(time.time(), 3),
            }
            for k in ("retry_at", "error", "attempts"):
                job.pop(k, None)
            self._save({**self.jobs, day: job})
            return job.copy()

    def next_ready(self) -> tuple[str, dict[str, Any]] | None:
        with self._lock:
            now = time.time()
            # Forced jobs first, then newest ISO dates first.
            ordered = sorted(
                self.jobs.items(),
                key=lambda kv: (float(kv[1].get("force_ts") or 0.0), kv[0]),
                reverse=True,
            )
            for day, job in ordered:
                if job.get("retry_at", 0) <= now:
                    return day, job.copy()
            return None

    def clear_backoff(self, day: str) -> None:
        with self._lock:
            if day not in self.jobs:
                return
            cur = self.jobs[day]
            if "retry_at" not in cur and "error" not in cur:
                return
            updated = cur.copy()
            updated.pop("retry_at", None)
            updated.pop("error", None)
            self._save({**self.jobs, day: updated})

    def finish(self, day: str, submitted: dict[str, Any], error: str = "") -> None:
        with self._lock:
            jobs = self.jobs.copy()
            current = jobs[day] = jobs[day].copy()
            if error:
                attempts = current.get("attempts", 0) + 1
                current.update(attempts=attempts, error=error[:400],
                               retry_at=time.time() + min(1800, 60 * 2 ** min(5, attempts - 1)))
            elif current == submitted:
                del jobs[day]
            else:
                # More footage or a new force request arrived while rendering.
                for key in ("attempts", "error", "retry_at"):
                    current.pop(key, None)
            self._save(jobs)
