"""
SearXNG client — the agent's in-built search engine.

Runs against a self-hosted SearXNG (Docker, JSON API on :8080). Used to find
and download the references cited in a paper, and to answer "find me papers
about X" queries.

Start the engine with:  docker compose -f searxng/container/docker-compose.yml up -d
(JSON output must be enabled — see setup.sh / README.)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

SEARXNG_BASE = os.environ.get("SEARXNG_BASE", "http://localhost:8080")

# Engines good for academic references
SCIENCE_CATEGORIES = "science"


def is_up() -> bool:
    """True if the SearXNG container is reachable."""
    try:
        requests.get(f"{SEARXNG_BASE}/healthz", timeout=3)
        return True
    except requests.RequestException:
        try:
            requests.get(SEARXNG_BASE, timeout=3)
            return True
        except requests.RequestException:
            return False


def search(query: str, categories: str = SCIENCE_CATEGORIES,
           max_results: int = 10) -> list[dict]:
    """
    Run a SearXNG search and return result dicts:
      {title, url, content, engine, ...}

    Raises a clear error if JSON output isn't enabled or the engine is down.
    """
    try:
        resp = requests.get(
            f"{SEARXNG_BASE}/search",
            params={"q": query, "format": "json", "categories": categories},
            headers={"Accept": "application/json"},
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"SearXNG unreachable at {SEARXNG_BASE}. "
            f"Start it with docker compose. ({e})"
        ) from e

    if resp.status_code == 403 or "json" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(
            "SearXNG did not return JSON. Enable it in searxng settings.yml:\n"
            "  search:\n    formats:\n      - html\n      - json"
        )
    resp.raise_for_status()
    return resp.json().get("results", [])[:max_results]


# ── PDF resolution ─────────────────────────────────────────────────────────────

_ARXIV_ABS = re.compile(r"arxiv\.org/abs/([\w.\-/]+)", re.I)
_ARXIV_ID  = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", re.I)


def to_pdf_url(url: str) -> Optional[str]:
    """Map a landing-page URL to a direct PDF URL when we recognise the host."""
    if not url:
        return None
    if url.lower().endswith(".pdf"):
        return url
    m = _ARXIV_ABS.search(url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    # biorxiv / medrxiv: append .full.pdf
    if "biorxiv.org/content/" in url or "medrxiv.org/content/" in url:
        return url.rstrip("/") + ".full.pdf"
    return None


def find_pdf(title: str, max_results: int = 8) -> Optional[dict]:
    """
    Search for a paper by title and return the best downloadable hit:
      {title, page_url, pdf_url}  — or None if nothing usable was found.
    """
    results = search(title, max_results=max_results)
    # Prefer results we can turn into a direct PDF
    for r in results:
        pdf = to_pdf_url(r.get("url", ""))
        if pdf:
            return {"title": r.get("title"), "page_url": r.get("url"), "pdf_url": pdf}
    # Fall back to the top hit's landing page (no direct PDF found)
    if results:
        top = results[0]
        return {"title": top.get("title"), "page_url": top.get("url"), "pdf_url": None}
    return None


def download_pdf(pdf_url: str, dest_dir: str, filename: Optional[str] = None) -> str:
    """Download a PDF to dest_dir. Returns the saved path."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        slug = re.sub(r"[^\w\-]+", "_", Path(pdf_url).stem)[:80] or f"paper_{int(time.time())}"
        filename = f"{slug}.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    dest = os.path.join(dest_dir, filename)
    resp = requests.get(pdf_url, timeout=120, stream=True,
                        headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest
