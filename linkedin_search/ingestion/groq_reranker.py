"""
ingestion/groq_reranker.py
--------------------------
Uses Claude (Anthropic) for two tasks:
  1. Query expansion  — enriches the user's query before FAISS search
  2. LLM reranking    — scores retrieved candidates and re-orders them
"""

import json
import logging
import os
from typing import Any

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    api_key = settings.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file or environment.")
    return anthropic.Anthropic(api_key=api_key)


# ── Query expansion ───────────────────────────────────────────────────────────

EXPANSION_PROMPT = """\
You are a job search query optimizer.

Given a user's natural language job search query, return 3 semantically related \
alternative phrasings that would help retrieve relevant LinkedIn posts.

User query: {query}

Return ONLY a JSON array of 3 strings. No explanation, no markdown.
Example: ["ML engineer remote", "machine learning engineer hiring", "AI engineer job opening"]
"""


def expand_query(query: str, client: anthropic.Anthropic | None = None) -> list[str]:
    client = client or _get_client()
    try:
        resp = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
        )
        raw = resp.content[0].text.strip()
        variants = json.loads(raw)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            logger.info(f"Expanded '{query}' → {variants}")
            return [query] + variants[:3]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}. Using original query.")
    return [query]


# ── LLM reranker ──────────────────────────────────────────────────────────────

RERANK_PROMPT = """\
You are a job search relevance scorer.

User query: "{query}"

Below are {n} LinkedIn posts retrieved as candidates. Score each post from 0–10 \
on how relevant it is to the query (10 = highly relevant, 0 = irrelevant).

Posts:
{posts_block}

Return ONLY a JSON array of {n} integers (scores), in the same order as the posts.
Example for 3 posts: [8, 3, 6]
"""


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 10,
    client: anthropic.Anthropic | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    client = client or _get_client()

    posts_block = "\n\n".join(
        f"[{i+1}] {c['text'][:400]}"
        for i, c in enumerate(candidates)
    )

    try:
        resp = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": RERANK_PROMPT.format(
                    query=query,
                    n=len(candidates),
                    posts_block=posts_block,
                )
            }],
        )
        raw = resp.content[0].text.strip()
        scores = json.loads(raw)

        if isinstance(scores, list) and len(scores) == len(candidates):
            for cand, score in zip(candidates, scores):
                cand["llm_score"] = int(score)
            reranked = sorted(candidates, key=lambda c: c.get("llm_score", 0), reverse=True)
            logger.info(f"Reranked {len(candidates)} candidates for query '{query}'")
            return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning FAISS order.")

    return candidates[:top_k]


# ── Combined search helper ────────────────────────────────────────────────────

def groq_enhanced_search(
    query: str,
    store,
    embedder,
    k: int = 10,
    filters: dict | None = None,
    expand: bool = True,
    rerank_results: bool = True,
) -> list[dict]:
    queries = expand_query(query) if expand else [query]

    seen_ids: set[str] = set()
    all_candidates: list[dict] = []

    for q in queries:
        qvec = embedder.embed_query(q)
        hits = store.search(qvec, k=k * 2, filters=filters)
        for hit in hits:
            if hit["post_id"] not in seen_ids:
                seen_ids.add(hit["post_id"])
                all_candidates.append(hit)

    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    all_candidates = all_candidates[:k * 3]

    if rerank_results and all_candidates:
        return rerank(query, all_candidates, top_k=k)

    return all_candidates[:k]
