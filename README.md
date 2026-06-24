# Research Agent

A local-first agent for reading, understanding, and expanding on academic papers.
Upload a PDF and the agent embeds it, builds a navigable **page index**, extracts
its **references** and downloads the ones you care about, answers questions about
it with a local LLM, and paces your reading with an adaptive **Pomodoro** timer
that even **speaks** to you.

Everything runs on your machine by default — no cloud APIs needed. But if you
want faster or smarter answers, drop an API key in `.env` and cloud models
(OpenAI, Anthropic, Google, Groq, Together) appear in the model picker.

![flow](https://img.shields.io/badge/local--first-cloud_optional-3fb950)

---

## What it does

| Capability | How |
|---|---|
| 📄 **Upload & view** papers | Drag-drop PDF → embedded viewer with page jumps |
| 🌲 **Page Index** | [PageIndex](https://github.com/VectifyAI/PageIndex) builds a reasoning tree (table-of-contents) — click a section to jump there |
| 🔎 **Embeddings & RAG** | Chunks → [nomic-embed-text] → [turbovec](https://github.com/arcius7/turbovec) 4-bit quantized store → answered by **gemma4:e4b** |
| 📚 **References** | Gemma extracts the bibliography; [SearXNG](https://github.com/searxng/searxng) finds & **downloads** the ones matching your query |
| 🌐 **Find new papers** | Built-in web search via your self-hosted SearXNG |
| 🍅 **Adaptive Pomodoro** | Work-session length scales with paper size (≈2 min/page) |
| 🔊 **Voice** | macOS `say` TTS with deterministic per-paper voice; downloadable .mp3 post-mortem |
| ☁️ **Cloud LLMs** | Optional — add an API key in `.env` and use GPT-4o, Claude, Gemini, Groq, or Together models |
| 🧠 **Research memory** | [agentmemory](https://github.com/rohitg00/agentmemory) remembers papers read, questions asked, refs downloaded (optional) |

---

## Architecture

```
                         ┌─────────────── Frontend (index.html) ───────────────┐
                         │  PDF viewer │ Page Index │ References │ Ask │ Find    │
                         └───────────────────────┬─────────────────────────────┘
                                                 │  REST
                         ┌───────────────────────▼─────────────────────────────┐
                         │                   server.py                          │
                         │  upload · pdf · ingest · tree · references · query   │
                         └──┬───────┬──────────┬───────────┬──────────┬─────────┘
              ┌─────────────┘       │          │           │          └──────────────┐
              ▼                     ▼          ▼           ▼                         ▼
        agent.py             pageindex_   references.py  searxng_           memory_client.py
        (LangGraph)          tree.py      ┌──────────┐   client.py          ┌──────────────┐
   ┌────────────────┐    ┌────────────┐   │ Gemma     │  ┌────────────┐     │ agentmemory  │
   │ ingest→turbovec│    │ PageIndex  │   │ extract → │  │  SearXNG   │     │  REST :3111  │
   │ retrieve→gemma │    │ → Ollama   │   │ SearXNG ↓ │  │  Docker    │     └──────────────┘
   │ pomodoro→vits  │    └────────────┘   └──────────┘  │  :8080 JSON│
   └────────────────┘                                   └────────────┘
                              Ollama (gemma4:e4b + nomic-embed-text) :11434
```

### Submodules (all under this folder)

| Path | Repo | Role |
|---|---|---|
| `vits2/` | daniilrobnikov/vits2 | TTS voice |
| `turbovec/` | arcius7/turbovec | quantized vector store |
| `PageIndex/` | VectifyAI/PageIndex | reasoning tree index |
| `agentmemory/` | rohitg00/agentmemory | research memory server |
| `searxng/` | searxng/searxng | metasearch engine source |

---

## Setup

```bash
git clone --recurse-submodules <your-repo-url> research_agent
cd research_agent
./setup.sh          # submodules + pip + ollama pulls + docker compose up
```

Or manually:

```bash
git submodule update --init --recursive
pip install -r requirements.txt
ollama pull gemma4:e4b && ollama pull nomic-embed-text
docker compose up -d                      # SearXNG on :8080
npx @agentmemory/agentmemory              # optional, memory on :3111
cp .env.example .env                      # optional, for cloud LLMs
python server.py                          # → http://localhost:8765
```

### Prerequisites

- **[Ollama](https://ollama.com)** — local LLM + embeddings (required even when using cloud LLMs, because embeddings always run locally)
- **Docker** — runs SearXNG for reference search (required for the References/Find tabs)
- **Node** — only if you want agentmemory's research memory (optional)

---

## Use it

1. **`python server.py`** and open **http://localhost:8765**
2. **Drop a PDF** — it auto-embeds and shows an adaptive session profile
   (pages, work-minutes, chunk size, voice).
3. **Page Index** tab → *Build page index* → click sections to jump in the PDF.
4. **References** tab → *Extract references*, then type a topic
   (e.g. `attention`) → *Search & download* pulls matching cited papers into
   `downloaded_refs/`.
5. **Ask** tab → ask anything; retrieval + gemma4:e4b answer grounded in the paper.
6. **Find Papers** tab → web search via SearXNG for brand-new papers.

The Pomodoro timer (top-right) is sized to the paper and speaks at each
transition.

### CLI (no browser)

```bash
python agent.py ingest paper.pdf
python agent.py query "What is the main contribution?"
python agent.py search "contrastive learning"
```

---

## Configuration

### Cloud LLM setup (optional)

To use paid cloud models instead of (or alongside) local Ollama:

```bash
cp .env.example .env
```

Then uncomment and fill in the API key(s) you want:

| Provider | Env variable | Models unlocked |
|---|---|---|
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3-mini |
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-sonnet-4-6, claude-haiku-4-5-20251001 |
| **Google Gemini** | `GEMINI_API_KEY` | gemini-2.5-flash, gemini-2.5-pro |
| **Groq** | `GROQ_API_KEY` | llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768 |
| **Together** | `TOGETHER_API_KEY` | Llama-3.3-70B-Instruct-Turbo, Mixtral-8x7B-Instruct-v0.1 |

Cloud models appear in the **model picker** dropdown in the Ask tab. Select one and your questions route through that provider. Switch back to any Ollama model at any time.

> **Note:** Embeddings always run locally via Ollama (`nomic-embed-text`) — only the answer-generation LLM is routed to the cloud. This keeps your paper data local.

### Other settings

| File | Variable | Default |
|---|---|---|
| `.env` | `LLM_MODEL` / `EMBED_MODEL` | `gemma4:e4b` / `nomic-embed-text` |
| `.env` | `OLLAMA_BASE` | `http://localhost:11434` |
| `.env` | `PORT` | `8765` |
| `pageindex_tree.py` | `PAGEINDEX_MODEL` | `ollama_chat/gemma4:e4b` |
| `searxng_client.py` | `SEARXNG_BASE` | `http://localhost:8080` |
| `memory_client.py` | `AGENTMEMORY_URL` | `http://localhost:3111` |
| `searxng-config/settings.yml` | `search.formats` | `[html, json]` — **JSON required** |

---

## Notes

- **SearXNG JSON API** is off by default; `searxng-config/settings.yml` enables it.
  If reference search returns a JSON error, that file isn't mounted — re-run
  `docker compose up -d`.
- **agentmemory is optional and resilient** — if the Node server isn't running,
  the agent works fine; memory calls quietly no-op. It's a coding-agent memory
  engine repurposed as research memory.
- **PageIndex** calls Gemma per section, so building a tree for a long paper
  takes a minute; results are cached in `.pageindex_trees/`.
