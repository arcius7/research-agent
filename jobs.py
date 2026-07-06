"""
Background job queue.

A single worker thread runs heavy tasks (embedding, LLM, audio) ONE AT A TIME.
Serializing heavy work is the anti-overheat design: the embedding model, the
LLM, and `say` never run in parallel fighting for CPU/GPU on a 16 GB M1.

Endpoints submit a job and return immediately with a job_id; the UI polls
`/api/job?id=…` for status + result, so nothing blocks and the machine stays cool.

Set WORKERS=2 in the environment only if your machine has thermal headroom.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from typing import Any, Callable, Optional

_MAX_WORKERS = max(1, int(os.environ.get("WORKERS", "1")))
_MAX_KEPT    = 200          # finished jobs retained for status polling

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_q: "queue.Queue[tuple]" = queue.Queue()
_started = False


def _update(job_id: str, **kw) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(**kw)


def _evict_locked() -> None:
    """Drop the oldest finished jobs beyond the cap. Call under _jobs_lock."""
    if len(_jobs) <= _MAX_KEPT:
        return
    done = sorted(
        (j for j in _jobs.values() if j["status"] in ("done", "error")),
        key=lambda j: j["created"],
    )
    for j in done[: len(_jobs) - _MAX_KEPT]:
        _jobs.pop(j["id"], None)


def submit(kind: str, fn: Callable, *args, **kwargs) -> str:
    """Queue a job. Returns its id immediately."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id, "kind": kind, "status": "queued",
            "result": None, "error": None,
            "created": time.time(), "started": None, "finished": None,
        }
        _evict_locked()
    _q.put((job_id, fn, args, kwargs))
    return job_id


def status(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            return None
        out = dict(j)
        if j["status"] == "queued":
            # live position: queued jobs submitted before this one, plus any
            # currently running job
            ahead = sum(
                1 for o in _jobs.values()
                if o["status"] == "queued" and o["created"] < j["created"]
            )
            running = sum(1 for o in _jobs.values() if o["status"] == "running")
            out["queue_position"] = ahead + running + 1
        return out


def _worker() -> None:
    while True:
        job_id, fn, args, kwargs = _q.get()
        _update(job_id, status="running", started=time.time())
        try:
            result = fn(*args, **kwargs)
            _update(job_id, status="done", result=result, finished=time.time())
        except Exception as e:                    # noqa: BLE001 — surfaced via status
            _update(job_id, status="error", error=str(e), finished=time.time())
        finally:
            _q.task_done()


def start_workers() -> None:
    """Spin up the worker thread(s). Idempotent."""
    global _started
    if _started:
        return
    _started = True
    for _ in range(_MAX_WORKERS):
        threading.Thread(target=_worker, daemon=True).start()
