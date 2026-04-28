"""
api/schemas.py
--------------
Pydantic request/response models for the FastAPI search service.
"""

from typing import Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Natural language job search query")
    k: int = Field(default=25, ge=1, le=100, description="Number of results to return")
    expand_query: bool = Field(default=True, description="Use Groq to expand query into variants")
    rerank: bool = Field(default=True, description="Use Groq LLM to rerank FAISS results")
    synthesize: bool = Field(default=True, description="Return LLM-synthesized answer alongside raw results")
    filters: dict[str, Any] | None = Field(default=None, description="Optional filters: {is_hiring, skills, roles}")
    sort_by: str = Field(default="relevance", pattern="^(relevance|date)$")
    post_type: str = Field(default="all", pattern="^(all|hiring|jobs)$")


class PostResult(BaseModel):
    post_id: str
    url: str | None = None
    author: str | None = None
    author_title: str | None = None
    text: str
    likes: int = 0
    posted_at: str | None = None
    roles: list[str] = []
    skills: list[str] = []
    is_hiring: bool = False
    score: float
    llm_score: int | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str | None = None        # present when synthesize=True
    results: list[PostResult]
    n_results: int
    expanded_queries_used: bool
    reranked: bool


class IndexStatsResponse(BaseModel):
    total_vectors: int
    total_metadata: int
    embed_dim: int
    index_path: str
    meta_path: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    total_vectors: int
