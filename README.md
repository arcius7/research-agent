# Research RAG Agent

A multi-node [LangGraph](https://github.com/langchain-ai/langgraph) agent for ingesting and querying academic papers locally — no cloud APIs required.

## Architecture

```
START → dispatch
  "ingest"       → ingest   → END
  "query"        → retrieve → llm → END
  "search"       → retrieve → END   (raw vector results, no LLM)
  timer actions  → pomodoro → vits? → END
```

### Nodes

| Node | Role |
|---|---|
| **ingest** | PDF / Word / Excel / JSON → chunks (900 chars, 120 overlap) → `nomic-embed-text` embeddings → TurboVec 4-bit quantized index |
| **retrieve** | Semantic ANN search over the stored embeddings → fills context |
| **llm** | RAG prompt + context → Ollama `gemma4:e4b` → answer |
| **pomodoro** | 25/5/15-min Pomodoro timer; shared state with `server.py` HTTP API |
| **vits** | VITS2 TTS announces timer mode transitions as speech |

### Dependencies (as git submodules)

- [vits2](https://github.com/daniilrobnikov/vits2) — neural TTS
- [turbovec](https://github.com/arcius7/turbovec) — quantized vector store (Rust, Python bindings)

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-user>/research_agent.git
cd research_agent
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models

```bash
# LLM for answering questions
ollama pull gemma4:e4b

# Embedding model for semantic search
ollama pull nomic-embed-text
```

### 4. (Optional) VITS2 voice

Download a trained checkpoint into `vits2/datasets/ljs_base/logs/G_1000.pth`  
or switch `VITS_DATASET` in `agent.py` to `"vctk_base"` for 109 speaker voices.

---

## Usage

### Run the Pomodoro server (optional, for the HTTP API)

```bash
python server.py
# → http://localhost:8765
```

### Ingest papers

```bash
# Single file
python agent.py ingest paper.pdf

# Multiple files / formats
python agent.py ingest paper1.pdf notes.docx data.xlsx refs.json
```

### Query your papers

```bash
# Full RAG (retrieve + LLM answer)
python agent.py query "What attention mechanism does this paper propose?"

# Raw semantic search (no LLM, returns scored chunks)
python agent.py search "transformer self-attention"
```

### Python API

```python
from agent import ingest, query, search

# Ingest documents
ingest(["paper1.pdf", "paper2.pdf", "notes.docx"])

# Ask a question — grounded in your papers via gemma4:e4b
answer = query("What datasets were used for evaluation?")
print(answer)

# Raw vector search (returns list of {text, score, meta})
hits = search("contrastive learning representation")
for h in hits:
    print(f"[{h['score']:.3f}] {h['meta']['source']}  →  {h['text'][:100]}")
```

### Timer control

```python
from agent import agent, _base

agent.invoke(_base(action="start"))   # start Pomodoro
agent.invoke(_base(action="pause"))   # pause
agent.invoke(_base(action="skip"))    # skip to next mode → VITS speaks
```

---

## Project Structure

```
research_agent/
├── agent.py              # LangGraph agent (5 nodes)
├── server.py             # Pomodoro HTTP server (REST API)
├── requirements.txt
├── .gitignore
├── vits2/                # VITS2 TTS (git submodule)
├── turbovec/             # TurboVec vector DB (git submodule)
└── .turbovec_store/      # persisted embeddings — gitignored, rebuilt by ingest
```

---

## Supported File Types

| Extension | Parser |
|---|---|
| `.pdf` | `pypdf` — page-by-page extraction |
| `.docx` / `.doc` | `python-docx` — paragraph extraction |
| `.xlsx` / `.xls` | `openpyxl` — sheet rows as pipe-delimited text |
| `.json` | built-in — list items or full object as chunks |

---

## Configuration (`agent.py`)

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `"gemma4:e4b"` | Ollama model for answering |
| `EMBED_MODEL` | `"nomic-embed-text"` | Ollama model for embeddings |
| `VITS_DATASET` | `"ljs_base"` | `"ljs_base"` (1 voice) or `"vctk_base"` (109 voices) |
| `VITS_SPEAKER_ID` | `None` | `None` = ljs; `0-108` = VCTK speaker |
| `CHUNK_SIZE` | `900` | Characters per text chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between consecutive chunks |
