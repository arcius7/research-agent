"""
Shared Pomodoro timer state — single source of truth.

Both server.py (HTTP endpoints) and agent.py (adaptive paper profile) import
this module. Keeping the state here, in a module neither of them owns, is what
guarantees they see the SAME dict: when server.py runs as `__main__`, an
`import server` from agent.py would load a second copy of server.py with a
second, disconnected state dict — the timer the UI polls would never receive
the paper-size adjustments. A dedicated module is imported exactly once.
"""

from __future__ import annotations

import threading
import time

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

MODE_KEY = {
    "work":        "work_duration",
    "short_break": "short_break",
    "long_break":  "long_break",
}


def advance_mode() -> None:
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


def _tick_loop() -> None:
    while True:
        time.sleep(1)
        with state_lock:
            if not state["running"]:
                continue
            state["elapsed"] += 1
            if state["elapsed"] >= state["settings"][MODE_KEY[state["mode"]]]:
                advance_mode()


_ticker_started = False


def start_ticker() -> None:
    """Start the 1 Hz timer thread. Idempotent."""
    global _ticker_started
    if _ticker_started:
        return
    _ticker_started = True
    threading.Thread(target=_tick_loop, daemon=True).start()
