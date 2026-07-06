#!/usr/bin/env python3
"""
Research Agent Backend — REST API + static frontend server.

Serves index.html (the upload / viewer / page-index / references UI) and exposes:

  Timer      /api/state /api/start /api/pause /api/reset /api/skip /api/settings ...
  Papers     POST /api/upload   GET /api/papers   GET /api/pdf?name=
  RAG        POST /api/ingest   POST /api/query
  PageIndex  GET  /api/tree?name=
  References GET  /api/references?name=   POST /api/references/search
  Web search POST /api/search

Heavy modules (agent.py → torch/vits, pageindex, references) are imported lazily
inside each handler so the server boots instantly and a missing optional
dependency only breaks the one endpoint that needs it.
"""

import json
import os
import time
import traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from timer_state import state, state_lock, advance_mode, start_ticker
from logging_setup import get_logger

log = get_logger(__name__)

_HERE         = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR    = os.path.join(_HERE, "uploads")
AUDIO_DIR     = os.path.join(_HERE, "audio")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))


# ── Background tasks (run on the serialized job worker) ───────────────────────

def _embed_task(path: str, name: str) -> dict:
    """Heavy: embed the paper into the vector store."""
    import agent
    result = agent.ingest([path])
    r0 = result["results"][0]
    try:
        import memory_client
        memory_client.log_ingest(name, r0.get("profile") or {})
    except Exception:
        pass
    return {"ingested": r0["ingested"], "total_docs": r0["total_docs"]}


def _refs_task(path: str, name: str) -> dict:
    """Heavy: LLM reference extraction (result is cached to disk)."""
    import references
    refs = references.extract_references(path)
    return {"paper": name, "count": len(refs), "references": refs}


def _audio_task(paper: str) -> dict:
    """Heavy: LLM post-mortem → spoken audio file."""
    import agent, tts
    text = agent.learn(paper=paper or None)
    res  = tts.synthesize(text, paper=paper or "")
    return {"text": text,
            "audio_url": f"/api/audio?name={res['filename']}",
            "voice": res["voice"]}


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    # — helpers —
    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _raw_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _fail(self, exc: Exception):
        # Full trace to the log (file + console) only — never to the client.
        log.error("%s %s FAILED: %s", self.command, self.path, exc, exc_info=True)
        self._json({"error": str(exc)}, 500)

    # Capture the status code of every response so we can log it.
    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def _log_req(self, t0: float):
        ms = (time.time() - t0) * 1000
        status = getattr(self, "_status", "?")
        # /api/state polls every second — log those at DEBUG so they don't drown
        # the interesting lines, everything else at INFO.
        lvl = log.debug if self.path.startswith("/api/state") else log.info
        lvl("%-4s %-24s → %s  (%.0f ms)", self.command, self.path, status, ms)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.end_headers()

    # — GET —
    def do_GET(self):
        t0 = time.time()
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        try:
            if path == "/api/state":
                with state_lock:
                    self._json(state)

            elif path == "/api/papers":
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                papers = sorted(f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf"))
                self._json({"papers": papers})

            elif path == "/api/models":
                import agent
                m = agent.list_models()          # {"local": [...], "cloud": [...]}
                self._json({"models": m["local"] + m["cloud"],   # back-compat flat list
                            "local": m["local"], "cloud": m["cloud"],
                            "current": agent.get_model()})

            elif path == "/api/job":
                import jobs
                j = jobs.status(qs.get("id", [""])[0])
                self._json(j if j else {"error": "unknown job"}, 200 if j else 404)

            elif path == "/api/pdf":
                self._serve_pdf(qs.get("name", [""])[0])

            elif path == "/api/audio":
                self._serve_audio(qs.get("name", [""])[0])

            elif path == "/api/tree":
                self._tree(qs.get("name", [""])[0])

            elif path == "/api/references":
                self._references(qs.get("name", [""])[0])

            elif path in ("/", "/index.html"):
                self._serve_file(os.path.join(_HERE, "index.html"), "text/html")

            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:               # noqa: BLE001 — return JSON, never a raw traceback
            self._fail(e)
        finally:
            self._log_req(t0)

    # — POST —
    def do_POST(self):
        t0   = time.time()
        path = urlparse(self.path).path
        try:
            match path:
                case "/api/start":
                    with state_lock: state["running"] = True
                    self._json({"ok": True})
                case "/api/pause":
                    with state_lock: state["running"] = False
                    self._json({"ok": True})
                case "/api/reset":
                    with state_lock:
                        state["running"] = False; state["elapsed"] = 0
                    self._json({"ok": True})
                case "/api/skip":
                    with state_lock: advance_mode()
                    self._json({"ok": True})
                case "/api/settings":
                    self._settings(self._body())
                case "/api/upload":
                    self._upload()
                case "/api/ingest":
                    self._ingest(self._body())
                case "/api/model":
                    import agent
                    name = self._body().get("model", "")
                    agent.set_model(name)
                    self._json({"ok": True, "model": agent.get_model()})
                case "/api/query":
                    self._query(self._body())
                case "/api/query_stream":
                    self._query_stream(self._body())
                case "/api/learn":
                    self._learn(self._body())
                case "/api/audio_job":
                    import jobs
                    paper = self._current_paper()
                    self._json({"job_id": jobs.submit("audio", _audio_task, paper)})
                case "/api/speak":
                    self._speak(self._body())
                case "/api/references/search":
                    self._ref_search(self._body())
                case "/api/search":
                    self._web_search(self._body())
                case _:
                    self._json({"error": "Not found"}, 404)
        except Exception as e:               # noqa: BLE001 — surface to the UI
            self._fail(e)
        finally:
            self._log_req(t0)

    # ── endpoint impls ────────────────────────────────────────────────────────

    def _serve_file(self, path, ctype, inline=False):
        try:
            with open(path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            # `inline` tells the browser to DISPLAY the file (in the viewer
            # iframe / audio player) instead of downloading it. Without this,
            # some browsers pop a "save as .pdf" dialog for application/pdf.
            if inline:
                self.send_header("Content-Disposition", "inline")
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def _serve_pdf(self, name):
        safe = os.path.basename(name or "")
        path = os.path.join(UPLOAD_DIR, safe)
        if safe and os.path.exists(path):
            log.info("serve PDF inline: %s (%.1f KB) — Content-Disposition: inline",
                     safe, os.path.getsize(path) / 1024)
            self._serve_file(path, "application/pdf", inline=True)
        else:
            log.warning("PDF not found: %s", safe)
            self.send_response(404); self.end_headers()

    def _settings(self, data):
        with state_lock:
            for key in ("work_duration", "short_break", "long_break", "long_break_interval"):
                if key in data:
                    state["settings"][key] = int(data[key])
            state["elapsed"] = 0
            state["running"] = False
        self._json({"ok": True})

    def _upload(self):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_MB * 1024 * 1024:
            log.warning("upload rejected: %d bytes > %d MB cap", length, MAX_UPLOAD_MB)
            return self._json({"error": f"file too large (max {MAX_UPLOAD_MB} MB)"}, 413)
        raw_name = self.headers.get("X-Filename", "")
        name = os.path.basename(raw_name or f"paper_{int(time.time())}.pdf")
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        data = self._raw_body()
        path = os.path.join(UPLOAD_DIR, name)
        with open(path, "wb") as f:
            f.write(data)
        log.info("upload: '%s' → saved as %s (%.1f KB)", raw_name, name, len(data) / 1024)
        with state_lock:
            state["current_paper"] = name

        # Instant: derive the profile + resize the Pomodoro timer right away.
        import agent
        profile = agent.quick_profile(path)
        log.info("profile: %s → %d pages, %d-min session, chunk %d, voice #%s",
                 name, profile.get("n_pages"), profile.get("work_minutes"),
                 profile.get("chunk_size"), profile.get("speaker_id"))

        # Background: embed the paper (heavy) on the serialized worker.
        import jobs
        job_id = jobs.submit("embed", _embed_task, path, name)

        self._json({"ok": True, "name": name, "bytes": len(data),
                    "profile": profile, "job_id": job_id})

    def _ingest(self, data):
        name = os.path.basename(data.get("name", ""))
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.exists(path):
            return self._json({"error": f"unknown paper: {name}"}, 404)

        import agent
        result  = agent.ingest([path])
        profile = result["results"][0].get("profile") or {}

        try:
            import memory_client
            memory_client.log_ingest(name, profile)
        except Exception:
            pass

        with state_lock:
            state["current_paper"] = name
            snapshot = {**state}
        self._json({"ok": True, "profile": profile, "timer": snapshot,
                    "ingested": result["results"][0]["ingested"]})

    def _current_paper(self) -> str:
        with state_lock:
            return state.get("current_paper") or ""

    def _query(self, data):
        question = data.get("question", "").strip()
        if not question:
            return self._json({"error": "empty question"}, 400)
        import agent
        answer = agent.query(question, paper=self._current_paper() or None)
        try:
            import memory_client
            memory_client.log_query(question, answer)
        except Exception:
            pass
        self._json({"answer": answer})

    def _query_stream(self, data):
        """Stream the RAG answer token-by-token over Server-Sent Events."""
        question = data.get("question", "").strip()
        if not question:
            return self._json({"error": "empty question"}, 400)
        import agent
        paper = self._current_paper() or None
        log.info("query (stream): paper=%s model=%s q=%r",
                 paper, agent.get_model(), question[:80])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        full = []
        try:
            for tok in agent.stream_answer(question, paper):
                full.append(tok)
                self.wfile.write(f"data: {json.dumps({'t': tok})}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()
        except Exception as e:                       # noqa: BLE001
            self.wfile.write(f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n".encode())
            self.wfile.flush()

        try:
            import memory_client
            memory_client.log_query(question, "".join(full))
        except Exception:
            pass

    def _learn(self, data):
        import agent
        text = agent.learn(paper=self._current_paper() or None)
        self._json({"text": text})

    def _speak(self, data):
        """Generate a spoken audio file with macOS `say`. Returns a JSON url the
        UI plays + offers as a download. The voice is chosen per paper."""
        text = (data.get("text", "") or "").strip()[:3000]
        if not text:
            return self._json({"error": "empty text"}, 400)
        paper = os.path.basename(data.get("paper", "") or "")
        import tts
        result = tts.synthesize(text, paper=paper)
        self._json({
            "audio_url": f"/api/audio?name={result['filename']}",
            "voice":     result["voice"],
        })

    def _serve_audio(self, name):
        safe = os.path.basename(name or "")
        path = os.path.join(AUDIO_DIR, safe)
        if safe and os.path.exists(path):
            ctype = "audio/mpeg" if safe.lower().endswith(".mp3") else "audio/mp4"
            self._serve_file(path, ctype, inline=True)
        else:
            self.send_response(404); self.end_headers()

    def _tree(self, name):
        name = os.path.basename(name or "")
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.exists(path):
            return self._json({"error": f"unknown paper: {name}"}, 404)
        try:
            import pageindex_tree
        except ModuleNotFoundError:
            return self._json({"error": "Page Index is an optional feature. "
                               "Install it with:  pip install -r requirements-pageindex.txt"}, 200)
        payload = pageindex_tree.build_tree(path)
        self._json(payload)

    def _references(self, name):
        """LLM-heavy → runs on the serialized job worker (anti-overheat), except
        when the result is already cached, which returns instantly."""
        name = os.path.basename(name or "")
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.exists(path):
            return self._json({"error": f"unknown paper: {name}"}, 404)
        import references, jobs
        cache = os.path.join(references.CACHE_DIR, os.path.splitext(name)[0] + ".json")
        if os.path.exists(cache):
            refs = references.extract_references(path)      # cache hit — instant
            return self._json({"paper": name, "count": len(refs), "references": refs})
        self._json({"job_id": jobs.submit("references", _refs_task, path, name)})

    def _ref_search(self, data):
        name  = os.path.basename(data.get("name", ""))
        query = data.get("query", "").strip()
        path  = os.path.join(UPLOAD_DIR, name)
        if not os.path.exists(path):
            return self._json({"error": f"unknown paper: {name}"}, 404)
        import references
        refs    = references.extract_references(path)
        results = references.download_matching(refs, query, source_paper=name)
        self._json({"query": query, "downloaded": results})

    def _web_search(self, data):
        query = data.get("query", "").strip()
        if not query:
            return self._json({"error": "empty query"}, 400)
        import searxng_client
        results = searxng_client.search(query, max_results=int(data.get("k", 10)))
        self._json({"query": query, "results": results})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    start_ticker()

    import jobs
    jobs.start_workers()                 # serialized background worker(s)

    port   = int(os.environ.get("PORT", 8765))
    host   = os.environ.get("HOST", "127.0.0.1")   # localhost only by default —
    server = ThreadingHTTPServer((host, port), Handler)  # set HOST=0.0.0.0 to expose on LAN
    server.daemon_threads = True          # request threads never block Ctrl+C
    from logging_setup import LOG_FILE
    log.info("Research Agent up on http://%s:%d  (logging to %s)", host, port, LOG_FILE)
    print(f"Research Agent running at http://localhost:{port}")
    print(f"Logs → {LOG_FILE}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("server stopping")
        server.shutdown()
        server.server_close()
        print("\nServer stopped.")
        os._exit(0)                       # exit immediately, even mid-request


if __name__ == "__main__":
    main()
