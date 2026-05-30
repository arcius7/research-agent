#!/usr/bin/env python3
"""
Pomodoro Timer Backend — REST API + static file server.
State is importable by agent.py so LangGraph nodes share it directly.
"""

import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import os

# ── Timer state ───────────────────────────────────────────────────────────────

state: dict = {
    "mode": "work",           # "work" | "short_break" | "long_break"
    "running": False,
    "elapsed": 0,             # seconds into current session
    "session_count": 0,       # completed pomodoros
    "tasks": [],
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
    """Background thread: advances elapsed and auto-transitions on completion."""
    while True:
        time.sleep(1)
        with state_lock:
            if not state["running"]:
                continue
            state["elapsed"] += 1
            duration = state["settings"][_MODE_KEY[state["mode"]]]
            if state["elapsed"] >= duration:
                _advance_mode()


# ── HTTP handler ──────────────────────────────────────────────────────────────

class PomodoroHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/state":
            with state_lock:
                self.send_json(state)

        elif path in ("/", "/index.html"):
            try:
                with open("index.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()

        else:
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        match path:
            case "/api/start":
                with state_lock:
                    state["running"] = True
                self.send_json({"ok": True})

            case "/api/pause":
                with state_lock:
                    state["running"] = False
                self.send_json({"ok": True})

            case "/api/reset":
                with state_lock:
                    state["running"] = False
                    state["elapsed"] = 0
                self.send_json({"ok": True})

            case "/api/skip":
                with state_lock:
                    _advance_mode()
                self.send_json({"ok": True})

            case "/api/set_mode":
                mode = data.get("mode", "work")
                if mode in _MODE_KEY:
                    with state_lock:
                        state["mode"] = mode
                        state["elapsed"] = 0
                        state["running"] = False
                self.send_json({"ok": True})

            case "/api/settings":
                with state_lock:
                    for key in ("work_duration", "short_break", "long_break", "long_break_interval"):
                        if key in data:
                            state["settings"][key] = int(data[key])
                    state["elapsed"] = 0
                    state["running"] = False
                self.send_json({"ok": True})

            case "/api/tasks/add":
                task = data.get("task", "").strip()
                if task:
                    with state_lock:
                        state["tasks"].append({
                            "id": int(time.time() * 1000),
                            "text": task,
                            "done": False,
                        })
                self.send_json({"ok": True})

            case "/api/tasks/toggle":
                task_id = data.get("id")
                with state_lock:
                    for t in state["tasks"]:
                        if t["id"] == task_id:
                            t["done"] = not t["done"]
                            break
                self.send_json({"ok": True})

            case "/api/tasks/delete":
                task_id = data.get("id")
                with state_lock:
                    state["tasks"] = [t for t in state["tasks"] if t["id"] != task_id]
                self.send_json({"ok": True})

            case "/api/reset_sessions":
                with state_lock:
                    state.update(mode="work", elapsed=0, running=False, session_count=0)
                self.send_json({"ok": True})

            case _:
                self.send_json({"error": "Not found"}, 404)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    threading.Thread(target=_tick, daemon=True).start()

    port = 8765
    server = HTTPServer(("0.0.0.0", port), PomodoroHandler)
    print(f"Pomodoro Timer running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
