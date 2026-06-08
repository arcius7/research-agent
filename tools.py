"""
LangChain tool surface for the research agent.

These wrap the deterministic pipeline as idiomatic, named LangChain tools. They
can be handed to a LangChain agent later, but we deliberately do NOT run an
agent loop: the workflow is fixed (upload → embed → ask / audio), so invoking
these directly keeps it fast and cool (no extra LLM reasoning rounds).

    from tools import TOOLS, answer_question
    answer_question.invoke({"question": "What is the main contribution?"})
"""

from __future__ import annotations

from langchain_core.tools import tool

import agent


@tool
def answer_question(question: str, paper: str = "") -> str:
    """Answer a question grounded in the embedded paper(s) via RAG.
    `paper` restricts retrieval to one paper (filename); empty = all papers."""
    return agent.query(question, paper=paper or None)


@tool
def summarize_paper(paper: str = "") -> str:
    """Produce a spoken-prose post-mortem summary of the paper
    (problem, methods, findings, contributions, limitations)."""
    return agent.learn(paper=paper or None)


@tool
def search_chunks(query: str, paper: str = "") -> list:
    """Semantic search over embedded chunks (no LLM). Returns top matches with
    scores and source metadata."""
    return agent.search(query, paper=paper or None)


@tool
def ingest_paper(path: str) -> dict:
    """Embed a paper file (PDF/Word/Excel/JSON) into the vector store."""
    return agent.ingest([path])["results"][0]


# The named-tool registry — ready to plug into a LangChain agent if ever needed.
TOOLS = [answer_question, summarize_paper, search_chunks, ingest_paper]
