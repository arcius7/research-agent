"""
Central logging for the whole agent.

Every module calls `get_logger(__name__)` and logs through it. `setup_logging()`
(called once from server.py, and defensively on first import) wires up:

  • a console handler  — what you see in the terminal
  • a rotating file handler at logs/agent.log — the full history (5 × 2 MB)

Control it with env vars:
  LOG_LEVEL=DEBUG        more detail (default INFO)
  LOG_FILE=/path.log     override the log file location
  LOG_CONSOLE=0          silence the console, file only
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

_HERE     = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(_HERE, "logs")
LOG_FILE  = os.environ.get("LOG_FILE", os.path.join(LOG_DIR, "agent.log"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-16s │ %(message)s"
_DATEFMT = "%H:%M:%S"

_configured = False


def setup_logging() -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    # Rotating file — the durable record.
    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console — live view in the terminal.
    if os.environ.get("LOG_CONSOLE", "1") != "0":
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Quiet noisy third parties unless we asked for DEBUG.
    if LOG_LEVEL != "DEBUG":
        for noisy in ("urllib3", "httpx", "httpcore", "LiteLLM", "litellm"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a module logger, ensuring logging is configured first."""
    setup_logging()
    # Trim the __main__/module path to a short, readable tag.
    short = name.rsplit(".", 1)[-1] if name and name != "__main__" else "server"
    return logging.getLogger(short)
