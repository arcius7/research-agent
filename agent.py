"""
Research RAG Agent  —  LangGraph multi-node pipeline
=====================================================

Core idea
─────────
When a paper is ingested the agent analyses its size and derives three things
automatically:

  1. Chunk size   — finer chunks for long papers (better retrieval precision),
                    coarser chunks for short papers (better context coverage).

  2. Pomodoro     — work-session duration set to ≈ 2 min/page so the timer
     duration       scales with how long the paper actually takes to read.

  3. VITS voice   — a speaker ID is picked deterministically from the paper's
                    filename, giving each paper a consistent, distinct voice.

Nodes
─────
  ingest     PDF / Word / Excel / JSON → adaptive chunks → nomic-embed-text
             → turbovec 4-bit quantized store → updates Pomodoro timer →
             VITS announces the session profile

  retrieve   ANN similarity search → fills context

  llm        Ollama gemma4:e4b RAG answer over retrieved context

  pomodoro   Pomodoro timer (25/5/15 min default, overridden by paper size)

  vits       VITS2 TTS — uses the speaker assigned to the current paper

Graph routing
─────────────
  START → dispatch
    "ingest"        → ingest → vits → END
    "query"         → retrieve → llm → END
    "search"        → retrieve → END
    timer actions   → pomodoro → vits → END
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, TypedDict

import requests
from langgraph.graph import StateGraph, START, END

# ── VITS2 (submodule: vits2/) ─────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
VITS_ROOT = os.path.join(_HERE, "vits2")
if VITS_ROOT not in sys.path:
    sys.path.insert(0, VITS_ROOT)

from model.models import SynthesizerTrn            # noqa: E402
from text import tokenizer                          # noqa: E402
from utils.hparams import get_hparams_from_file    # noqa: E402
from utils.task import load_checkpoint, load_vocab  # noqa: E402

# ── turbovec ──────────────────────────────────────────────────────────────────
from turbovec.langchain import TurboQuantVectorStore  # noqa: E402

# ── shared pomodoro state ─────────────────────────────────────────────────────
from server import state as _timer, state_lock, _advance_mode, _MODE_KEY  # noqa: E402

import torch  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Static config
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE  = "http://localhost:11434"
LLM_MODEL    = "gemma4:e4b"        # ollama pull gemma4:e4b
EMBED_MODEL  = "nomic-embed-text"  # ollama pull nomic-embed-text

TURBOVEC_DIR = os.path.join(_HERE, ".turbovec_store")

# VITS — use "vctk_base" for multi-speaker (109 voices), "ljs_base" for single
VITS_DATASET    = "vctk_base"
VITS_CONFIG     = os.path.join(VITS_ROOT, f"datasets/{VITS_DATASET}/config.yaml")
VITS_VOCAB      = os.path.join(VITS_ROOT, f"datasets/{VITS_DATASET}/vocab.txt")
VITS_CHECKPOINT = os.path.join(VITS_ROOT, f"datasets/{VITS_DATASET}/logs/G_1000.pth")

# VCTK speakers 0-20 are clear, well-trained English voices
_VCTK_SPEAKERS = list(range(21))

_device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper profile  —  the heart of the adaptive logic
# ═══════════════════════════════════════════════════════════════════════════════

def paper_profile(filename: str, total_chars: int, n_pages: int) -> dict:
    """
    Derive chunk_size, overlap, Pomodoro work duration, and VITS speaker ID
    from a paper's size metrics.

    Returns
    -------
    dict with keys:
      chunk_size    int   characters per embedding chunk
      overlap       int   overlap between consecutive chunks
      work_minutes  int   Pomodoro work-session length in minutes
      work_seconds  int   same in seconds (used to set the timer)
      speaker_id    int   VCTK speaker index (0-108)
      total_chars   int
      n_pages       int
    """
    # ── chunk size: finer for longer papers ───────────────────────────────────
    # Short paper → large chunks (more context per embedding)
    # Long paper  → small chunks (sharper retrieval, more chunks to cover detail)
    if total_chars < 8_000:        # ~4 pages
        chunk_size, overlap = 1400, 200
    elif total_chars < 25_000:     # ~15 pages
        chunk_size, overlap = 950, 130
    elif total_chars < 60_000:     # ~35 pages
        chunk_size, overlap = 650, 95
    elif total_chars < 120_000:    # ~70 pages
        chunk_size, overlap = 450, 65
    else:                           # thesis / book chapter
        chunk_size, overlap = 300, 45

    # ── Pomodoro: 2 min/page, clamped to [15, 50] min ────────────────────────
    work_minutes = max(15, min(50, n_pages * 2))

    # ── VITS speaker: deterministic hash of filename → consistent per paper ───
    h = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
    speaker_id = _VCTK_SPEAKERS[h % len(_VCTK_SPEAKERS)]

    return {
        "chunk_size":   chunk_size,
        "overlap":      overlap,
        "work_minutes": work_minutes,
        "work_seconds": work_minutes * 60,
        "speaker_id":   speaker_id,
        "total_chars":  total_chars,
        "n_pages":      n_pages,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Graph state
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # timer
    action:        str
    timer:         dict
    announce:      Optional[str]   # text → VITS TTS
    audio_path:    Optional[str]

    # RAG
    files:         list            # paths for "ingest"
    query:         Optional[str]   # question for "query" / "search"
    context:       list            # retrieved chunks
    answer:        Optional[str]   # LLM response
    results:       list            # raw search hits

    # paper profile (set by ingest_node, carried through the run)
    speaker_id:    Optional[int]   # VITS speaker for this paper
    paper_profile: Optional[dict]  # full profile dict


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ollama_embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _ollama_generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


class _OllamaEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _ollama_embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return _ollama_embed([text])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Node: pomodoro
# ═══════════════════════════════════════════════════════════════════════════════

_TRANSITION_TEXT = {
    "short_break": "Nice work! Take a short break.",
    "long_break":  "Great effort! Enjoy a long break.",
    "work":        "Break over. Time to focus.",
}


def pomodoro_node(state: AgentState) -> AgentState:
    action = state.get("action", "tick")
    announce: Optional[str] = None

    with state_lock:
        if action == "start":
            _timer["running"] = True
        elif action == "pause":
            _timer["running"] = False
        elif action == "reset":
            _timer["running"] = False
            _timer["elapsed"] = 0
        elif action == "skip":
            _advance_mode()
            announce = _TRANSITION_TEXT[_timer["mode"]]
        elif action == "tick" and _timer["running"]:
            _timer["elapsed"] += 1
            duration = _timer["settings"][_MODE_KEY[_timer["mode"]]]
            if _timer["elapsed"] >= duration:
                _advance_mode()
                announce = _TRANSITION_TEXT[_timer["mode"]]
        snapshot = {**_timer, "tasks": list(_timer["tasks"])}

    return {**state, "timer": snapshot, "announce": announce}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: vits
# ═══════════════════════════════════════════════════════════════════════════════

_net_g:      Optional[SynthesizerTrn] = None
_vits_hps   = None
_vits_vocab = None
_model_lock = threading.Lock()


def _load_vits() -> None:
    global _net_g, _vits_hps, _vits_vocab
    _vits_hps   = get_hparams_from_file(VITS_CONFIG)
    _vits_vocab = load_vocab(VITS_VOCAB)
    filter_len  = (
        _vits_hps.data.n_mels if _vits_hps.data.use_mel
        else _vits_hps.data.n_fft // 2 + 1
    )
    seg_size   = _vits_hps.train.segment_size // _vits_hps.data.hop_length
    n_speakers = getattr(_vits_hps.data, "n_speakers", 0)
    net = SynthesizerTrn(
        len(_vits_vocab), filter_len, seg_size,
        n_speakers=n_speakers, **_vits_hps.model,
    ).to(_device)
    net.eval()
    load_checkpoint(VITS_CHECKPOINT, net, None)
    _net_g = net


def _get_vits() -> tuple:
    if _net_g is None:
        with _model_lock:
            if _net_g is None:
                _load_vits()
    return _net_g, _vits_hps, _vits_vocab


def synthesize(text: str, out_path: str,
               speaker_id: Optional[int] = None,
               noise_scale: float = 0.667,
               noise_scale_w: float = 0.8,
               length_scale: float = 1.0) -> str:
    """text → .wav at out_path via VITS2. speaker_id selects the VCTK voice."""
    net, hps, vocab = _get_vits()
    tokens = tokenizer(
        text, vocab, hps.data.text_cleaners,
        language=hps.data.language, cleaned_text=False,
    )
    x     = torch.LongTensor(tokens).unsqueeze(0).to(_device)
    x_len = torch.LongTensor([len(tokens)]).to(_device)
    sid   = torch.LongTensor([speaker_id]).to(_device) if speaker_id is not None else None
    with torch.no_grad():
        out = net.infer(x, x_len, sid=sid,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=length_scale)
    import soundfile as sf
    sf.write(out_path, out[0][0, 0].cpu().float().numpy(), hps.data.sample_rate)
    return out_path


def vits_node(state: AgentState) -> AgentState:
    text = state.get("announce")
    if not text:
        return state
    # Use the speaker assigned to this paper; fall back to None (ljs single-speaker)
    speaker_id = state.get("speaker_id")
    out_path   = f"/tmp/pomo_{int(time.time())}.wav"
    synthesize(text, out_path, speaker_id=speaker_id)
    return {**state, "audio_path": out_path}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: ingest
# ═══════════════════════════════════════════════════════════════════════════════

# ── file extractors ───────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> list[tuple[str, dict]]:
    import pypdf
    fname  = Path(path).name
    reader = pypdf.PdfReader(path)
    pages  = [
        (page.extract_text(), {"source": fname, "page": i})
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]
    return pages


def _extract_docx(path: str) -> list[tuple[str, dict]]:
    import docx
    fname = Path(path).name
    doc   = docx.Document(path)
    return [(p.text, {"source": fname}) for p in doc.paragraphs if p.text.strip()]


def _extract_excel(path: str) -> list[tuple[str, dict]]:
    import openpyxl
    fname = Path(path).name
    wb    = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out   = []
    for sheet in wb.worksheets:
        rows = [
            " | ".join(str(c) for c in row if c is not None)
            for row in sheet.iter_rows(values_only=True)
        ]
        rows = [r for r in rows if r.strip()]
        if rows:
            out.append(("\n".join(rows), {"source": fname, "sheet": sheet.title}))
    return out


def _extract_json(path: str) -> list[tuple[str, dict]]:
    fname = Path(path).name
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    return [(json.dumps(item, ensure_ascii=False), {"source": fname}) for item in items]


_EXTRACTORS = {
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".doc":  _extract_docx,
    ".xlsx": _extract_excel,
    ".xls":  _extract_excel,
    ".json": _extract_json,
}

# ── adaptive chunker ──────────────────────────────────────────────────────────

def _chunk(text: str, meta: dict, chunk_size: int, overlap: int) -> list[tuple[str, dict]]:
    """Split text into overlapping chunks sized for this paper."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in separators:
                pos = text.rfind(sep, start + overlap, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, {**meta, "chunk": len(chunks)}))
        start = end - overlap
    return chunks

# ── vector store ──────────────────────────────────────────────────────────────

_store: Optional[TurboQuantVectorStore] = None
_store_lock = threading.Lock()


def _get_store() -> TurboQuantVectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                emb  = _OllamaEmbeddings()
                path = Path(TURBOVEC_DIR)
                _store = (
                    TurboQuantVectorStore.load(path, emb)
                    if (path / "index.tvim").exists()
                    else TurboQuantVectorStore(embedding=emb, bit_width=4)
                )
    return _store

# ── ingest node ───────────────────────────────────────────────────────────────

def ingest_node(state: AgentState) -> AgentState:
    store    = _get_store()
    files    = state.get("files", [])
    ingested = []
    profile  = None

    for path in files:
        ext     = Path(path).suffix.lower()
        extract = _EXTRACTORS.get(ext)
        if extract is None:
            raise ValueError(f"Unsupported file type: {ext}  ({path})")

        raw         = extract(path)
        total_chars = sum(len(t) for t, _ in raw)
        n_pages     = (
            len(raw) if ext == ".pdf"            # one entry per page
            else max(1, total_chars // 2_500)    # estimate for other formats
        )
        fname   = Path(path).name
        profile = paper_profile(fname, total_chars, n_pages)

        # ── chunk with sizes tuned to this paper ──────────────────────────────
        texts, metas = [], []
        for text, meta in raw:
            for chunk, cmeta in _chunk(text, meta,
                                       profile["chunk_size"], profile["overlap"]):
                texts.append(chunk)
                metas.append(cmeta)

        if texts:
            store.add_texts(texts, metadatas=metas)
            ingested.append(fname)

        # ── set Pomodoro work duration from paper size ─────────────────────────
        with state_lock:
            _timer["settings"]["work_duration"] = profile["work_seconds"]
            _timer["elapsed"]  = 0
            _timer["running"]  = False
            _timer["mode"]     = "work"

    store.dump(TURBOVEC_DIR)

    # ── announce the profile via VITS ─────────────────────────────────────────
    announce = None
    if profile and ingested:
        announce = (
            f"Paper loaded. {profile['work_minutes']} minute work sessions. "
            f"Voice {profile['speaker_id']} selected. "
            f"{len(ingested)} document{'s' if len(ingested) > 1 else ''} ready."
        )

    with state_lock:
        snapshot = {**_timer, "tasks": list(_timer["tasks"])}

    return {
        **state,
        "timer":         snapshot,
        "paper_profile": profile,
        "speaker_id":    profile["speaker_id"] if profile else state.get("speaker_id"),
        "announce":      announce,
        "results":       [{"ingested": ingested,
                           "total_docs": len(store),
                           "profile": profile}],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: retrieve
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_node(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    store = _get_store()
    hits  = store.similarity_search_with_score(query, k=5)
    return {
        **state,
        "context": [doc.page_content for doc, _ in hits],
        "results": [
            {"text": doc.page_content, "score": round(score, 4), "meta": doc.metadata}
            for doc, score in hits
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: llm  (Ollama RAG)
# ═══════════════════════════════════════════════════════════════════════════════

_RAG_PROMPT = """\
You are a research assistant that answers questions about academic papers.

Retrieved context from papers:
---
{context}
---

Question: {question}

Instructions:
- Answer using only the provided context.
- Be precise; cite the source document name when relevant.
- If the context is insufficient, say so explicitly.

Answer:"""


def llm_node(state: AgentState) -> AgentState:
    question = state.get("query") or ""
    context  = state.get("context") or []
    if not context:
        return {**state, "answer": "No relevant context found in the document store."}
    prompt = _RAG_PROMPT.format(
        context="\n\n---\n\n".join(context),
        question=question,
    )
    return {**state, "answer": _ollama_generate(prompt)}


# ═══════════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════════

_TIMER_ACTIONS = {"start", "pause", "reset", "skip", "tick"}


def _dispatch(state: AgentState) -> str:
    action = state.get("action", "tick")
    if action in _TIMER_ACTIONS:        return "pomodoro"
    if action == "ingest":              return "ingest"
    if action in ("query", "search"):   return "retrieve"
    return END


def _after_pomodoro(state: AgentState) -> str:
    return "vits" if state.get("announce") else END


def _after_ingest(state: AgentState) -> str:
    # Always speak the profile announcement after ingesting
    return "vits" if state.get("announce") else END


def _after_retrieve(state: AgentState) -> str:
    return "llm" if state.get("action") == "query" else END


# ═══════════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("pomodoro", pomodoro_node)
    g.add_node("vits",     vits_node)
    g.add_node("ingest",   ingest_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("llm",      llm_node)

    g.add_conditional_edges(START, _dispatch, {
        "pomodoro": "pomodoro",
        "ingest":   "ingest",
        "retrieve": "retrieve",
        END:        END,
    })
    g.add_conditional_edges("pomodoro", _after_pomodoro, {"vits": "vits", END: END})
    g.add_conditional_edges("ingest",   _after_ingest,   {"vits": "vits", END: END})
    g.add_conditional_edges("retrieve", _after_retrieve,  {"llm":  "llm",  END: END})

    g.add_edge("vits", END)
    g.add_edge("llm",  END)

    return g.compile()


agent = build_graph()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════════

def _base(**kw) -> AgentState:
    return {
        "action": "tick", "timer": {}, "announce": None, "audio_path": None,
        "files": [], "query": None, "context": [], "answer": None, "results": [],
        "speaker_id": None, "paper_profile": None,
        **kw,
    }


def ingest(files: list[str]) -> dict:
    """Ingest files. Returns result including the derived paper profile."""
    return agent.invoke(_base(action="ingest", files=files))


def query(question: str) -> str:
    """Ask a question; returns LLM answer grounded in ingested papers."""
    return agent.invoke(_base(action="query", query=question))["answer"]


def search(question: str) -> list[dict]:
    """Raw semantic search — returns top-5 chunks with scores, no LLM."""
    return agent.invoke(_base(action="search", query=question))["results"]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys as _sys

    cmd  = _sys.argv[1] if len(_sys.argv) > 1 else "help"
    args = _sys.argv[2:]

    if cmd == "ingest":
        r = ingest(args)
        p = r["results"][0]["profile"]
        print(f"Ingested:         {r['results'][0]['ingested']}")
        print(f"Total docs:       {r['results'][0]['total_docs']}")
        print(f"Chunk size:       {p['chunk_size']} chars (overlap {p['overlap']})")
        print(f"Pomodoro session: {p['work_minutes']} min")
        print(f"VITS speaker:     #{p['speaker_id']}")

    elif cmd == "query":
        print(query(" ".join(args)))

    elif cmd == "search":
        for h in search(" ".join(args)):
            print(f"[{h['score']:.3f}] ({h['meta'].get('source','?')})  {h['text'][:120]}…")

    elif cmd == "start":
        r = agent.invoke(_base(action="start"))
        t = r["timer"]
        print(f"Timer started — mode: {t['mode']}, duration: {t['settings']['work_duration']//60} min")

    else:
        print(__doc__)
