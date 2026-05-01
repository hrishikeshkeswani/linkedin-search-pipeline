"""
api/main.py
-----------
FastAPI search service for the LinkedIn RAG pipeline.

Endpoints:
  GET  /health           — liveness + index readiness
  POST /search           — full RAG search (retrieve + optional synthesis)
  GET  /search           — same as POST via query params (for quick browser testing)
  GET  /stats            — FAISS index stats
  POST /index/evict      — trigger TTL eviction (admin)

Run locally:
  uvicorn api.main:app --reload --port 8000
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    IndexStatsResponse,
    PostResult,
    SearchRequest,
    SearchResponse,
)
from config.settings import settings
from indexer.faiss_store import FAISSStore
from ingestion.embedder import PostEmbedder
from ingestion.groq_reranker import groq_enhanced_search
from pipeline.rag_chain import LinkedInRAGChain

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
)
logger = logging.getLogger(__name__)


def _search_event(query: str, n_results: int, latency_ms: float, expanded: bool, reranked: bool) -> None:
    logger.info(json.dumps({
        "query": query,
        "n_results": n_results,
        "latency_ms": round(latency_ms, 2),
        "expanded_queries_used": expanded,
        "reranked": reranked,
    }))

# ── App state (loaded once at startup) ────────────────────────────────────────

_store: FAISSStore | None = None
_embedder: PostEmbedder | None = None
_rag: LinkedInRAGChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _embedder, _rag
    logger.info("Loading FAISS index and embedding model…")
    _store = FAISSStore()
    _embedder = PostEmbedder()
    _rag = LinkedInRAGChain(store=_store, embedder=_embedder)
    logger.info(f"Index ready: {_store.stats()['total_vectors']} vectors")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="LinkedIn Search API",
    description="RAG-based job search over LinkedIn posts (FAISS + Groq + LangChain)",
    version="1.0.0",
    lifespan=lifespan,
)

_allowed_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_ready():
    if _store is None or _store.stats()["total_vectors"] == 0:
        raise HTTPException(status_code=503, detail="Index not ready. Run the pipeline first.")


def _to_post_result(r: dict) -> PostResult:
    return PostResult(
        post_id=r.get("post_id", ""),
        url=r.get("url"),
        author=r.get("author"),
        author_title=r.get("author_title"),
        text=r.get("text", ""),
        likes=r.get("likes", 0),
        posted_at=r.get("posted_at"),
        roles=r.get("roles", []),
        skills=r.get("skills", []),
        is_hiring=r.get("is_hiring", False),
        score=r.get("score", 0.0),
        llm_score=r.get("llm_score"),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    s = _store.stats() if _store else {}
    return HealthResponse(
        status="ok",
        index_ready=s.get("total_vectors", 0) > 0,
        total_vectors=s.get("total_vectors", 0),
        last_refreshed=s.get("last_refreshed"),
    )


@app.get("/stats", response_model=IndexStatsResponse, tags=["ops"])
def stats():
    _require_ready()
    return IndexStatsResponse(**_store.stats())


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search_post(req: SearchRequest):
    _require_ready()
    t0 = time.perf_counter()

    # Hiring Posts = people's feed posts (LinkedIn, HN, Reddit)
    # Job Posts    = structured job board listings (JobSpy, RemoteOK, Adzuna, Indeed)
    HIRING_SOURCES = {"linkedin", "hackernews", "reddit"}
    JOB_SOURCES    = {"jobspy", "remoteok", "adzuna", "indeed", "zip_recruiter"}
    filters = req.filters or {}
    if req.post_type == "hiring":
        filters = {**filters, "sources": HIRING_SOURCES}
    elif req.post_type == "jobs":
        filters = {**filters, "sources": JOB_SOURCES}

    if req.synthesize:
        out = _rag.invoke(req.query, filters=filters or None)
        raw = out["results"]
        answer = out["answer"]
    else:
        raw = groq_enhanced_search(
            query=req.query,
            store=_store,
            embedder=_embedder,
            k=req.k,
            filters=filters or None,
            expand=req.expand_query,
            rerank_results=req.rerank,
        )
        answer = None

    # Sort by date if requested
    if req.sort_by == "date":
        raw = sorted(raw, key=lambda r: r.get("posted_at") or "", reverse=True)

    results = [_to_post_result(r) for r in raw]

    latency_ms = (time.perf_counter() - t0) * 1000
    _search_event(req.query, len(results), latency_ms, req.expand_query, req.rerank)

    return SearchResponse(
        query=req.query,
        answer=answer,
        results=results,
        n_results=len(results),
        expanded_queries_used=req.expand_query,
        reranked=req.rerank,
    )


@app.get("/search", response_model=SearchResponse, tags=["search"])
def search_get(
    q: str = Query(..., min_length=3, description="Search query"),
    k: int = Query(default=10, ge=1, le=50),
    expand: bool = Query(default=True),
    rerank: bool = Query(default=True),
    synthesize: bool = Query(default=False),
):
    req = SearchRequest(
        query=q,
        k=k,
        expand_query=expand,
        rerank=rerank,
        synthesize=synthesize,
    )
    return search_post(req)


@app.post("/index/evict", tags=["ops"])
def evict_stale(ttl_days: int = Query(default=None, description="Override TTL days")):
    _require_ready()
    evicted = _store.evict_stale(ttl_days=ttl_days)
    if evicted > 0:
        _store.save()
    return {"evicted": evicted, "remaining": _store.stats()["total_vectors"]}
