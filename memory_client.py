"""
Research memory — thin, resilient client over agentmemory's REST API (:3111).

agentmemory (https://github.com/rohitg00/agentmemory) is a hybrid BM25 + vector
+ graph memory server. Here we use it as the research agent's long-term memory:
which papers were ingested, what questions were asked, which references were
downloaded.

Every call is best-effort: if the agentmemory server isn't running, the agent
keeps working and these functions quietly no-op. Start the server with:
    npx @agentmemory/agentmemory
"""

from __future__ import annotations

import os
from typing import Optional

import requests

AGENTMEMORY_URL = os.environ.get("AGENTMEMORY_URL", "http://localhost:3111")
PROJECT         = os.environ.get("AGENTMEMORY_PROJECT", "research-agent")
_TIMEOUT        = 8


def is_up() -> bool:
    try:
        r = requests.get(f"{AGENTMEMORY_URL}/agentmemory/health", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


def remember(text: str, kind: str = "note", meta: Optional[dict] = None) -> bool:
    """
    Persist a research memory (paper ingested, query asked, reference saved).
    Returns True on success, False if the server is unreachable.
    """
    payload = {
        "project": PROJECT,
        "content": text,
        "type":    kind,
        "metadata": meta or {},
    }
    for route in ("/agentmemory/save", "/agentmemory/remember"):
        try:
            r = requests.post(f"{AGENTMEMORY_URL}{route}", json=payload, timeout=_TIMEOUT)
            if r.ok:
                return True
        except requests.RequestException:
            continue
    return False


def recall(query: str, limit: int = 5) -> list[dict]:
    """
    Hybrid search over past research memories. Returns [] if the server is down.
    """
    try:
        r = requests.post(
            f"{AGENTMEMORY_URL}/agentmemory/smart-search",
            json={"project": PROJECT, "query": query, "limit": limit},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        # agentmemory returns {results: [...]} or a bare list depending on version
        return data.get("results", data) if isinstance(data, dict) else data
    except requests.RequestException:
        return []


# ── Convenience wrappers tied to research events ──────────────────────────────

def log_ingest(paper: str, profile: dict) -> bool:
    return remember(
        f"Ingested paper '{paper}': {profile.get('n_pages')} pages, "
        f"{profile.get('work_minutes')}-min sessions, voice #{profile.get('speaker_id')}.",
        kind="paper",
        meta={"paper": paper, **profile},
    )


def log_query(question: str, answer: str) -> bool:
    return remember(
        f"Q: {question}\nA: {answer[:500]}",
        kind="query",
        meta={"question": question},
    )


def log_download(title: str, source_paper: str, path: str) -> bool:
    return remember(
        f"Downloaded reference '{title}' (cited in {source_paper}).",
        kind="reference",
        meta={"title": title, "source_paper": source_paper, "path": path},
    )
