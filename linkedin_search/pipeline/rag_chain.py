"""
pipeline/rag_chain.py
---------------------
LangChain RAG chain over the FAISS vector store.

Flow:
  user query
    → query expansion (Groq)
    → FAISS retrieval (multi-query)
    → LLM reranking (Groq)
    → answer synthesis (Groq)
    → structured response
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from config.settings import settings
from indexer.faiss_store import FAISSStore
from ingestion.embedder import PostEmbedder
from ingestion.groq_reranker import groq_enhanced_search

logger = logging.getLogger(__name__)


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a LinkedIn job search assistant. You have been given a set of relevant LinkedIn posts \
retrieved for the user's query. Your job is to synthesize these into a concise, helpful answer.

Guidelines:
- Highlight the most relevant opportunities, companies, or insights from the posts
- Mention specific roles, skills, or companies when present
- If posts mention hiring, call that out clearly
- Be direct and actionable — the user is job searching
- Do not fabricate information not present in the posts
"""),
    ("human", """\
User query: {query}

Retrieved LinkedIn posts ({n_results} results):
{context}

Provide a concise synthesis of these results for the user's job search query.
"""),
])


# ── Context formatter ──────────────────────────────────────────────────────────

def _format_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        hiring_flag = " [HIRING]" if r.get("is_hiring") else ""
        author_line = f"{r.get('author', 'Unknown')}"
        if r.get("author_title"):
            author_line += f" — {r['author_title']}"
        skills = ", ".join(r.get("skills", []))
        roles = ", ".join(r.get("roles", []))

        block = f"[{i}]{hiring_flag} {author_line}\n{r.get('text', '')[:500]}"
        if skills:
            block += f"\nSkills: {skills}"
        if roles:
            block += f"\nRoles: {roles}"
        if r.get("url"):
            block += f"\nURL: {r['url']}"
        parts.append(block)
    return "\n\n---\n\n".join(parts)


def _results_to_documents(results: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=r.get("text", ""),
            metadata={k: v for k, v in r.items() if k != "text"},
        )
        for r in results
    ]


# ── RAG Chain ──────────────────────────────────────────────────────────────────

class LinkedInRAGChain:
    """
    End-to-end RAG chain:
      query → FAISS retrieval (with query expansion + reranking) → LLM synthesis
    """

    def __init__(
        self,
        store: FAISSStore,
        embedder: PostEmbedder,
        k: int = 10,
        expand_queries: bool = True,
        rerank: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.k = k
        self.expand_queries = expand_queries
        self.rerank = rerank

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.3,
        )

        self._chain = self._build_chain()

    def _retrieve(self, query: str, filters: dict | None = None) -> list[dict]:
        return groq_enhanced_search(
            query=query,
            store=self.store,
            embedder=self.embedder,
            k=self.k,
            filters=filters,
            expand=self.expand_queries,
            rerank_results=self.rerank,
        )

    def _build_chain(self):
        def retrieve_and_format(inputs: dict) -> dict:
            query = inputs["query"]
            filters = inputs.get("filters")
            results = self._retrieve(query, filters=filters)
            return {
                "query": query,
                "context": _format_context(results),
                "n_results": len(results),
                "raw_results": results,
            }

        synthesis_chain = (
            RunnableLambda(lambda x: {
                "query": x["query"],
                "context": x["context"],
                "n_results": x["n_results"],
            })
            | SYNTHESIS_PROMPT
            | self.llm
            | StrOutputParser()
        )

        def full_pipeline(inputs: dict) -> dict:
            enriched = retrieve_and_format(inputs)
            answer = synthesis_chain.invoke(enriched)
            return {
                "query": inputs["query"],
                "answer": answer,
                "results": enriched["raw_results"],
                "n_results": enriched["n_results"],
            }

        return RunnableLambda(full_pipeline)

    def invoke(self, query: str, filters: dict | None = None) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            {
                "query": str,
                "answer": str,          # LLM-synthesized response
                "results": list[dict],  # raw retrieved posts
                "n_results": int,
            }
        """
        return self._chain.invoke({"query": query, "filters": filters})

    def retrieve_only(self, query: str, filters: dict | None = None) -> list[dict]:
        """Return raw FAISS + reranked results without LLM synthesis."""
        return self._retrieve(query, filters=filters)

    def as_langchain_retriever(self):
        """Expose as a LangChain-compatible retriever (returns Documents)."""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        store = self.store
        embedder = self.embedder
        k = self.k
        expand = self.expand_queries
        rerank = self.rerank

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                results = groq_enhanced_search(
                    query=query,
                    store=store,
                    embedder=embedder,
                    k=k,
                    expand=expand,
                    rerank_results=rerank,
                )
                return _results_to_documents(results)

        return _Retriever()
