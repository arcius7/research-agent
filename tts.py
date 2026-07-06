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

from logging_setup import get_logger
log = get_logger(__name__)

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


try:
    import lameenc                       # bundled LAME — no external binary needed
    _HAS_LAME = True
except ImportError:
    _HAS_LAME = False


def _wav_to_mp3(wav_path: str, mp3_path: str, bitrate: int = 96) -> None:
    """Encode a 16-bit PCM WAV to MP3 using lameenc + stdlib wave."""
    import wave
    w = wave.open(wav_path, "rb")
    try:
        pcm = w.readframes(w.getnframes())
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate)
        enc.set_in_sample_rate(w.getframerate())
        enc.set_channels(w.getnchannels())
        enc.set_quality(5)               # 2=best/slow … 7=fast
        data = enc.encode(pcm) + enc.flush()
    finally:
        w.close()
    with open(mp3_path, "wb") as f:
        f.write(data)


_KEEP_NEWEST = 20   # generated audio files retained; older ones are pruned


def _prune_audio() -> None:
    """Keep only the newest N generated files so audio/ doesn't grow forever."""
    try:
        files = sorted(Path(AUDIO_DIR).glob("postmortem_*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[_KEEP_NEWEST:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


def synthesize(text: str, speaker_id: Optional[int] = None, paper: str = "") -> dict:
    """
    Generate a downloadable audio file from text via macOS `say`.
    Produces .mp3 when lameenc is installed, else falls back to .m4a.
    Returns {path, filename, voice, format}.
    """
    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    _prune_audio()
    voice = voice_for(speaker_id, paper)
    stamp = int(time.time())
    log.info("TTS: %d chars, voice=%s, format=%s",
             len(text), voice or "default", "mp3" if _HAS_LAME else "m4a")

    if _HAS_LAME:
        wav = os.path.join(AUDIO_DIR, f"_tmp_{stamp}.wav")
        cmd = ["say", "-o", wav, "--file-format=WAVE", "--data-format=LEI16"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, check=True, timeout=300)

        filename = f"postmortem_{stamp}.mp3"
        path     = os.path.join(AUDIO_DIR, filename)
        _wav_to_mp3(wav, path)
        try:
            os.remove(wav)
        except OSError:
            pass
        return {"path": path, "filename": filename, "voice": voice or "default", "format": "mp3"}

    # Fallback: AAC .m4a (also downloadable/playable)
    filename = f"postmortem_{stamp}.m4a"
    path     = os.path.join(AUDIO_DIR, filename)
    cmd = ["say", "-o", path]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    subprocess.run(cmd, check=True, timeout=300)
    return {"path": path, "filename": filename, "voice": voice or "default", "format": "m4a"}
