"""
macOS `say` text-to-speech — generates a real, downloadable .m4a audio file
with a selectable voice. Powers the spoken post-mortem of a paper.

Each paper maps deterministically to one voice (its "speaker id"), so every
paper gets a consistent, distinct narrator. No model or checkpoint needed.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

_HERE     = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(_HERE, "audio")

# Natural, intelligible English voices — novelty voices (Boing, Bells, Bad News…)
# are deliberately excluded. Intersected with whatever `say` has installed.
_CURATED = [
    "Samantha", "Alex", "Daniel", "Karen", "Moira", "Tessa", "Fiona",
    "Veena", "Rishi", "Victoria", "Allison", "Ava", "Susan", "Tom",
    "Aman", "Evan", "Nicky", "Serena", "Oliver",
]

_voices_cache: Optional[list[str]] = None


def available_voices() -> list[str]:
    """Curated natural voices that are actually installed, in a stable order."""
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        out = ""
    installed = set()
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s{2,}([a-z]{2}_[A-Z]{2})", line)
        if m and m.group(2).startswith("en"):
            installed.add(m.group(1).strip())
    _voices_cache = [v for v in _CURATED if v in installed]
    return _voices_cache


def voice_for(speaker_id: Optional[int] = None, paper: str = "") -> str:
    """Deterministic voice pick. Same paper → same voice every time.
    Returns "" when no curated voice is installed (→ system default)."""
    voices = available_voices()
    if not voices:
        return ""
    if speaker_id is None:
        speaker_id = int(hashlib.md5(paper.encode()).hexdigest()[:8], 16)
    return voices[speaker_id % len(voices)]


def synthesize(text: str, speaker_id: Optional[int] = None, paper: str = "") -> dict:
    """
    Generate an .m4a audio file from text via macOS `say`.
    Returns {path, filename, voice}.
    """
    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    voice    = voice_for(speaker_id, paper)
    filename = f"postmortem_{int(time.time())}.m4a"
    path     = os.path.join(AUDIO_DIR, filename)

    cmd = ["say", "-o", path]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    subprocess.run(cmd, check=True, timeout=300)

    return {"path": path, "filename": filename, "voice": voice or "default"}
