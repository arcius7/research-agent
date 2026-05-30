"""
PageIndex wrapper — builds a reasoning-based tree (table-of-contents) of a PDF.

Powers the "Page Index" panel in the frontend: a hierarchical, navigable
structure of the paper with per-section summaries and page ranges.

PageIndex calls its LLM through LiteLLM, so we point it at the local Ollama
server (no OpenAI key needed) using the `ollama_chat/<model>` route.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the in-repo submodule; fall back to a sibling clone in ~/Downloads
PAGEINDEX_ROOT = os.path.join(_HERE, "PageIndex")
if not os.path.isdir(PAGEINDEX_ROOT):
    PAGEINDEX_ROOT = os.path.expanduser("~/Downloads/PageIndex")
if PAGEINDEX_ROOT not in sys.path:
    sys.path.insert(0, PAGEINDEX_ROOT)

# Route PageIndex's LiteLLM calls to local Ollama
os.environ.setdefault("OLLAMA_API_BASE", os.environ.get("OLLAMA_BASE", "http://localhost:11434"))

# LiteLLM model string for Ollama chat models
PAGEINDEX_MODEL = os.environ.get("PAGEINDEX_MODEL", "ollama_chat/gemma4:e4b")

# Where generated trees are cached (one JSON per paper)
TREE_DIR = os.path.join(_HERE, ".pageindex_trees")


def _opt():
    from pageindex.utils import ConfigLoader
    return ConfigLoader().load({
        "model": PAGEINDEX_MODEL,
        "if_add_node_id": "yes",
        "if_add_node_summary": "yes",
        "if_add_doc_description": "yes",
    })


def build_tree(pdf_path: str, force: bool = False) -> dict:
    """
    Generate (or load cached) the PageIndex tree for a PDF.

    Returns a dict:  {doc_name, tree: [...nodes...]} where each node has
    title, node_id, start_index, end_index, summary, and nested `nodes`.
    """
    from pageindex import page_index_main

    Path(TREE_DIR).mkdir(parents=True, exist_ok=True)
    cache = os.path.join(TREE_DIR, Path(pdf_path).stem + ".json")

    if os.path.exists(cache) and not force:
        with open(cache) as f:
            return json.load(f)

    result = page_index_main(pdf_path, _opt())
    # page_index_main returns the tree structure (list of nodes or dict)
    tree = result if isinstance(result, (list, dict)) else getattr(result, "structure", result)
    payload = {"doc_name": Path(pdf_path).name, "tree": tree}

    with open(cache, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def tree_to_markdown(payload: dict) -> str:
    """Flatten a tree payload into an indented Markdown outline for the LLM/UI."""
    lines: list[str] = []

    def walk(nodes, depth=0):
        if isinstance(nodes, dict):
            nodes = nodes.get("nodes", [nodes])
        for n in nodes or []:
            title = n.get("title", "Untitled")
            pages = ""
            if "start_index" in n and "end_index" in n:
                pages = f"  (pp. {n['start_index']}–{n['end_index']})"
            lines.append("  " * depth + f"- {title}{pages}")
            if n.get("nodes"):
                walk(n["nodes"], depth + 1)

    walk(payload.get("tree", []))
    return "\n".join(lines)
