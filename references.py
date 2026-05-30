"""
Reference extraction + reference search/download.

1. extract_references(pdf)  — pull the bibliography out of a paper. Grabs the
   References/Bibliography section text, then asks Gemma (via Ollama) to turn it
   into a clean structured list [{title, authors, year}].

2. find_and_download(ref)   — use SearXNG to locate each reference online and
   download the PDF into ./downloaded_refs/, giving you a fresh reading list.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import requests

import searxng_client
import memory_client

_HERE        = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(_HERE, "downloaded_refs")

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL", "gemma4:e4b")

_REF_HEADINGS = re.compile(
    r"\n\s*(references|bibliography|works cited|literature cited)\s*\n",
    re.I,
)


# ── 1. extract ────────────────────────────────────────────────────────────────

def _raw_reference_text(pdf_path: str) -> str:
    """Return the text of the References section (or the last ~25% as fallback)."""
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    full   = "\n".join(p.extract_text() or "" for p in reader.pages)

    m = _REF_HEADINGS.search(full)
    if m:
        return full[m.end():][:12000]      # cap to keep the LLM prompt sane
    # Fallback: assume references live in the final quarter of the document
    return full[int(len(full) * 0.75):][:12000]


_EXTRACT_PROMPT = """\
Below is the reference/bibliography section of an academic paper. Extract each
cited work as JSON. Return ONLY a JSON array, no prose, of objects with keys:
"title", "authors" (string), "year" (string). Skip anything that isn't a real
citation.

Reference section:
---
{refs}
---

JSON array:"""


def extract_references(pdf_path: str) -> list[dict]:
    """Extract a structured reference list from a paper using Gemma."""
    raw = _raw_reference_text(pdf_path)
    if not raw.strip():
        return []

    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": _EXTRACT_PROMPT.format(refs=raw),
            "stream": False,
            "format": "json",          # ask Ollama to constrain to JSON
        },
        timeout=300,
    )
    resp.raise_for_status()
    text = resp.json()["response"]

    refs = _coerce_json_list(text)
    # Normalise + drop entries with no title
    out = []
    for r in refs:
        title = (r.get("title") or "").strip()
        if len(title) > 5:
            out.append({
                "title":   title,
                "authors": (r.get("authors") or "").strip(),
                "year":    str(r.get("year") or "").strip(),
            })
    return out


def _coerce_json_list(text: str) -> list[dict]:
    """Best-effort parse of an LLM response into a list of dicts."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Sometimes wrapped like {"references": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        return []


# ── 2. search + download ──────────────────────────────────────────────────────

def find_and_download(ref: dict, source_paper: str = "") -> dict:
    """
    Locate one reference online via SearXNG and download its PDF.

    Returns a status dict:
      {title, found, pdf_url, page_url, path, error}
    """
    title = ref.get("title", "")
    query = title
    if ref.get("year"):
        query = f"{title} {ref['year']}"

    result = {"title": title, "found": False, "pdf_url": None,
              "page_url": None, "path": None, "error": None}

    try:
        hit = searxng_client.find_pdf(query)
    except RuntimeError as e:
        result["error"] = str(e)
        return result

    if not hit:
        result["error"] = "no search results"
        return result

    result.update(found=True, pdf_url=hit["pdf_url"], page_url=hit["page_url"])

    if hit["pdf_url"]:
        try:
            path = searxng_client.download_pdf(hit["pdf_url"], DOWNLOAD_DIR)
            result["path"] = path
            memory_client.log_download(title, source_paper, path)
        except requests.RequestException as e:
            result["error"] = f"download failed: {e}"
    return result


def download_matching(refs: list[dict], query: str, source_paper: str = "",
                      max_downloads: int = 5) -> list[dict]:
    """
    From an extracted reference list, download those whose title best matches a
    free-text query (e.g. "transformer attention"). Simple keyword scoring keeps
    it fast and offline; the LLM already produced clean titles.
    """
    terms   = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    scored  = []
    for ref in refs:
        title = ref.get("title", "").lower()
        score = sum(t in title for t in terms)
        if score:
            scored.append((score, ref))

    scored.sort(key=lambda s: s[0], reverse=True)
    return [find_and_download(ref, source_paper) for _, ref in scored[:max_downloads]]
