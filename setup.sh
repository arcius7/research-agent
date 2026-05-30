#!/usr/bin/env bash
# Research Agent — one-shot setup.
set -e

echo "▶ 1/5  Submodules (vits2, turbovec, PageIndex, agentmemory, searxng)…"
git submodule update --init --recursive

echo "▶ 2/5  Python dependencies…"
pip install -r requirements.txt

echo "▶ 3/5  Ollama models (gemma4:e4b + nomic-embed-text)…"
if command -v ollama >/dev/null 2>&1; then
  ollama pull gemma4:e4b
  ollama pull nomic-embed-text
else
  echo "   ⚠ ollama not found — install from https://ollama.com then:"
  echo "     ollama pull gemma4:e4b && ollama pull nomic-embed-text"
fi

echo "▶ 4/5  SearXNG (Docker, reference search engine)…"
if docker info >/dev/null 2>&1; then
  docker compose up -d
  echo "   SearXNG → http://localhost:8080"
else
  echo "   ⚠ Docker daemon not running — start Docker Desktop, then: docker compose up -d"
fi

echo "▶ 5/5  agentmemory (optional research memory, port 3111)…"
echo "   Start it in a separate terminal when you want memory:"
echo "     npx @agentmemory/agentmemory"

echo
echo "✅ Done. Launch the app:"
echo "     python server.py     → http://localhost:8765"
