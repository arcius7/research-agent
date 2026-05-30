"""
Research RAG Agent  —  LangGraph multi-node pipeline
=====================================================

Nodes
─────
  pomodoro   timer management (shared state with server.py)
  vits       speaks mode-transition announcements via VITS2 TTS
  ingest     PDF / Word / Excel / JSON → chunks → Ollama embeddings → turbovec
  retrieve   similarity search in turbovec → fills context
  llm        Ollama gemma4:e4b RAG answer over retrieved context

Graph routing
─────────────
  START → dispatch
    "ingest"        → ingest   → END
    "query"         → retrieve → llm → END
    "search"        → retrieve → END     (raw vector results, no LLM)
    timer actions   → pomodoro → vits? → END
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
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

# ── turbovec (pip install turbovec[langchain]) ────────────────────────────────
from turbovec.langchain import TurboQuantVectorStore  # noqa: E402

# ── shared pomodoro state (same objects server.py uses) ──────────────────────
from server import state as _timer, state_lock, _advance_mode, _MODE_KEY  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Config  — edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE   = "http://localhost:11434"
LLM_MODEL     = "gemma4:e4b"        # ollama pull gemma4:e4b
EMBED_MODEL   = "nomic-embed-text"  # ollama pull nomic-embed-text

TURBOVEC_DIR  = os.path.join(_HERE, ".turbovec_store")
CHUNK_SIZE    = 900     # characters per chunk
CHUNK_OVERLAP = 120     # overlap between chunks

# VITS voice
# Use "ljs_base" (single speaker) or "vctk_base" (109 speakers → set VITS_SPEAKER_ID)
VITS_DATASET    = "ljs_base"
VITS_CONFIG     = os.path.join(VITS_ROOT, f"datasets/{VITS_DATASET}/config.yaml")
VITS_VOCAB      = os.path.join(VITS_ROOT, f"datasets/{VITS_DATASET}/vocab.txt")
VITS_CHECKPOINT = os.path.join(VITS_ROOT, f"datasets/{VITS_DATASET}/logs/G_1000.pth")
VITS_SPEAKER_ID: Optional[int] = None  # None = ljs single-speaker; 0-108 = VCTK

import torch  # noqa: E402
_device = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph state
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    action:     str             # "start"|"pause"|"reset"|"skip"|"tick"|"ingest"|"query"|"search"
    timer:      dict
    announce:   Optional[str]   # text → VITS TTS on timer transition
    audio_path: Optional[str]   # .wav written by vits_node
    files:      list            # file paths for "ingest"
    query:      Optional[str]   # question for "query" / "search"
    context:    list            # retrieved chunks (filled by retrieve_node)
    answer:     Optional[str]   # LLM response (filled by llm_node)
    results:    list            # raw search hits [{text, score, meta}, ...]


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Batch embed via Ollama /api/embed  →  768-dim float32 vectors."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _ollama_generate(prompt: str) -> str:
    """Single-shot generation via Ollama (non-streaming)."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


class _OllamaEmbeddings:
    """Thin wrapper satisfying the embed_documents / embed_query interface
    that TurboQuantVectorStore expects."""

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
               speaker_id: Optional[int] = VITS_SPEAKER_ID,
               noise_scale: float = 0.667,
               noise_scale_w: float = 0.8,
               length_scale: float = 1.0) -> str:
    """text → .wav at out_path. Returns out_path."""
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
    out_path = f"/tmp/pomo_{int(time.time())}.wav"
    synthesize(text, out_path)
    return {**state, "audio_path": out_path}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: ingest  (PDF / Word / Excel / JSON → chunks → turbovec)
# ═══════════════════════════════════════════════════════════════════════════════

# ── file extractors ───────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> list[tuple[str, dict]]:
    import pypdf
    fname  = Path(path).name
    reader = pypdf.PdfReader(path)
    return [
        (page.extract_text(), {"source": fname, "page": i})
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]


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

# ── chunker ───────────────────────────────────────────────────────────────────

def _chunk(text: str, meta: dict) -> list[tuple[str, dict]]:
    """Split text into overlapping chunks, snapping to paragraph/sentence seams."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in separators:
                pos = text.rfind(sep, start + CHUNK_OVERLAP, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, {**meta, "chunk": len(chunks)}))
        start = end - CHUNK_OVERLAP
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


def ingest_node(state: AgentState) -> AgentState:
    store    = _get_store()
    ingested = []

    for path in state.get("files", []):
        ext     = Path(path).suffix.lower()
        extract = _EXTRACTORS.get(ext)
        if extract is None:
            raise ValueError(f"Unsupported file type: {ext}  ({path})")

        texts, metas = [], []
        for text, meta in extract(path):
            for chunk, cmeta in _chunk(text, meta):
                texts.append(chunk)
                metas.append(cmeta)

        if texts:
            store.add_texts(texts, metadatas=metas)
            ingested.append(Path(path).name)

    store.dump(TURBOVEC_DIR)
    return {**state, "results": [{"ingested": ingested, "total_docs": len(store)}]}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: retrieve  (turbovec similarity search → context)
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
# Node: llm  (Ollama gemma4:e4b  RAG)
# ═══════════════════════════════════════════════════════════════════════════════

_RAG_PROMPT = """\
You are a research assistant that answers questions about academic papers.

Retrieved context:
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
    if action in _TIMER_ACTIONS:  return "pomodoro"
    if action == "ingest":        return "ingest"
    if action in ("query", "search"): return "retrieve"
    return END


def _after_pomodoro(state: AgentState) -> str:
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
    g.add_conditional_edges("retrieve", _after_retrieve, {"llm":  "llm",  END: END})

    g.add_edge("vits",     END)
    g.add_edge("ingest",   END)
    g.add_edge("llm",      END)

    return g.compile()


agent = build_graph()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════════

def _base(**kw) -> AgentState:
    return {
        "action": "tick", "timer": {}, "announce": None, "audio_path": None,
        "files": [], "query": None, "context": [], "answer": None, "results": [],
        **kw,
    }


def ingest(files: list[str]) -> dict:
    """Ingest a list of file paths into the vector store."""
    return agent.invoke(_base(action="ingest", files=files))


def query(question: str) -> str:
    """Ask a question; returns the LLM answer grounded in ingested papers."""
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
        print(f"Ingested: {r['results'][0]['ingested']}")
        print(f"Total docs in store: {r['results'][0]['total_docs']}")

    elif cmd == "query":
        print(query(" ".join(args)))

    elif cmd == "search":
        for hit in search(" ".join(args)):
            print(f"[{hit['score']:.3f}] ({hit['meta'].get('source','?')})  {hit['text'][:120]}...")

    elif cmd == "start":
        r = agent.invoke(_base(action="start"))
        print("Timer started. Mode:", r["timer"]["mode"])

    else:
        print(__doc__)
