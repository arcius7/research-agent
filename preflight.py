"""
Startup preflight — verifies every external dependency the agent uses and
fixes what it can, so failures show up in the log at boot instead of as
mysterious errors when you click a tab.

Checks (runs in a background thread; boot stays instant):

  1. Ollama    — reachable? are LLM_MODEL and EMBED_MODEL actually pulled?
  2. SearXNG   — reachable? if not and Docker is up, `docker compose up -d`
                 is run automatically (references / find-papers need it).
  3. agentmemory — optional; noted if absent, never an error.

Everything is logged through the central logger (console + logs/agent.log).
Disable the SearXNG autostart with SEARXNG_AUTOSTART=0.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import requests

from logging_setup import get_logger

log = get_logger("preflight")

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── 1. Ollama ─────────────────────────────────────────────────────────────────

def check_ollama() -> None:
    base  = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
    llm   = os.environ.get("LLM_MODEL", "gemma4:e4b")
    embed = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        log.error("Ollama NOT reachable at %s — start it (open the Ollama app "
                  "or run `ollama serve`). Embedding and Ask will fail until then.", base)
        return

    names = [m["name"] for m in r.json().get("models", [])]
    log.info("Ollama OK — %d models installed", len(names))

    def _have(model: str) -> bool:
        return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)

    if not _have(llm):
        log.warning("LLM model '%s' is NOT pulled. Fix:  ollama pull %s   "
                    "(or set LLM_MODEL to one of: %s)", llm, llm, ", ".join(names) or "none")
    if not _have(embed):
        log.warning("Embedding model '%s' is NOT pulled. Fix:  ollama pull %s",
                    embed, embed)


# ── 2. SearXNG (+ Docker autostart) ──────────────────────────────────────────

def _searxng_up(timeout: float = 3) -> bool:
    base = os.environ.get("SEARXNG_BASE", "http://localhost:8080")
    try:
        requests.get(base, timeout=timeout)
        return True
    except requests.RequestException:
        return False


def check_searxng() -> None:
    if _searxng_up():
        log.info("SearXNG OK at %s", os.environ.get("SEARXNG_BASE", "http://localhost:8080"))
        return

    if os.environ.get("SEARXNG_AUTOSTART", "1") == "0":
        log.warning("SearXNG down (autostart disabled) — references/find tabs won't work")
        return

    # Is the Docker daemon up?
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        log.warning("SearXNG down and Docker daemon not running — start Docker "
                    "Desktop, then run:  docker compose up -d   "
                    "(references / find-papers disabled until then)")
        return

    log.info("SearXNG down — starting it via `docker compose up -d` …")
    try:
        subprocess.run(["docker", "compose", "up", "-d"],
                       cwd=_HERE, capture_output=True, timeout=180, check=True)
    except subprocess.SubprocessError as e:
        out = getattr(e, "stderr", b"") or b""
        log.error("docker compose up failed: %s %s", e, out.decode(errors="replace")[:300])
        return

    # The container takes a few seconds to accept connections.
    for _ in range(15):
        if _searxng_up():
            log.info("SearXNG started OK (Docker) — references/find tabs ready")
            return
        time.sleep(2)
    log.warning("SearXNG container started but not answering yet — give it a "
                "moment, then retry the References tab")


# ── 3. agentmemory (optional) ─────────────────────────────────────────────────

def check_agentmemory() -> None:
    url = os.environ.get("AGENTMEMORY_URL", "http://localhost:3111")
    try:
        requests.get(f"{url}/agentmemory/health", timeout=3)
        log.info("agentmemory OK at %s", url)
    except requests.RequestException:
        log.info("agentmemory not running (optional) — research memory disabled")


# ── runner ────────────────────────────────────────────────────────────────────

def run_async() -> None:
    """Run all checks in a daemon thread so server boot stays instant."""
    def _run():
        log.info("preflight: checking dependencies…")
        for check in (check_ollama, check_searxng, check_agentmemory):
            try:
                check()
            except Exception as e:                     # noqa: BLE001 — never kill boot
                log.error("preflight check %s crashed: %s", check.__name__, e)
        log.info("preflight: done")
    threading.Thread(target=_run, daemon=True).start()
