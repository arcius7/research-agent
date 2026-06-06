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
import threading
import traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_HERE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(_HERE, "uploads")
AUDIO_DIR   = os.path.join(_HERE, "audio")

# ── Timer state ───────────────────────────────────────────────────────────────

state: dict = {
    "mode": "work",
    "running": False,
    "elapsed": 0,
    "session_count": 0,
    "tasks": [],
    "current_paper": None,
    "settings": {
        "work_duration":       25 * 60,
        "short_break":          5 * 60,
        "long_break":          15 * 60,
        "long_break_interval": 4,
    },
}

state_lock = threading.Lock()

_MODE_KEY = {
    "work":        "work_duration",
    "short_break": "short_break",
    "long_break":  "long_break",
}


def _advance_mode() -> None:
    """Transition to the next mode. Must be called under state_lock."""
    if state["mode"] == "work":
        state["session_count"] += 1
        interval = state["settings"]["long_break_interval"]
        state["mode"] = (
            "long_break" if state["session_count"] % interval == 0
            else "short_break"
        )
    else:
        state["mode"] = "work"
    state["elapsed"] = 0
    state["running"] = False


def _tick() -> None:
    while True:
        time.sleep(1)
        with state_lock:
            if not state["running"]:
                continue
            state["elapsed"] += 1
            if state["elapsed"] >= state["settings"][_MODE_KEY[state["mode"]]]:
                _advance_mode()


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
        self._json({"error": str(exc), "trace": traceback.format_exc()}, 500)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.end_headers()

    # — GET —
    def do_GET(self):
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
                self._json({"models": agent.list_models(), "current": agent.get_model()})

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

    # — POST —
    def do_POST(self):
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
                    with state_lock: _advance_mode()
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

    # ── endpoint impls ────────────────────────────────────────────────────────

    def _serve_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def _serve_pdf(self, name):
        safe = os.path.basename(name or "")
        path = os.path.join(UPLOAD_DIR, safe)
        if safe and os.path.exists(path):
            self._serve_file(path, "application/pdf")
        else:
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
        name = os.path.basename(self.headers.get("X-Filename", "") or f"paper_{int(time.time())}.pdf")
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        data = self._raw_body()
        with open(os.path.join(UPLOAD_DIR, name), "wb") as f:
            f.write(data)
        with state_lock:
            state["current_paper"] = name
        self._json({"ok": True, "name": name, "bytes": len(data)})

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
            self._serve_file(path, "audio/mp4")
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
                               "Install it with:  pip install litellm pymupdf PyPDF2"}, 200)
        payload = pageindex_tree.build_tree(path)
        self._json(payload)

    def _references(self, name):
        name = os.path.basename(name or "")
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.exists(path):
            return self._json({"error": f"unknown paper: {name}"}, 404)
        import references
        refs = references.extract_references(path)
        self._json({"paper": name, "count": len(refs), "references": refs})

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
    threading.Thread(target=_tick, daemon=True).start()

    port   = int(os.environ.get("PORT", 8765))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Research Agent running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
