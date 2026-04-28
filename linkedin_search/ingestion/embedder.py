"""
ingestion/embedder.py
---------------------
Embeds cleaned LinkedIn posts using BAAI/bge-large-en-v1.5 (sentence-transformers).

Why BGE over OpenAI embeddings?
  - Free, runs locally, no API cost per post
  - bge-large-en-v1.5 scores competitively on MTEB retrieval benchmarks
  - Groq is used for LLM tasks (query expansion, reranking), not embeddings

Usage:
    embedder = PostEmbedder()
    vectors, ids = embedder.embed_posts(posts)
"""

import logging
import numpy as np
from pathlib import Path
from typing import NamedTuple

from sentence_transformers import SentenceTransformer

from config.settings import settings
from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)


# ── Embedding template (BGE instruction prefix) ───────────────────────────────

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# For indexing, BGE works best WITHOUT the prefix on the document side
DOC_PREFIX = ""


class EmbeddingResult(NamedTuple):
    vectors: np.ndarray      # shape: (n_posts, embed_dim)
    post_ids: list[str]
    texts: list[str]


# ── Embedder ──────────────────────────────────────────────────────────────────

class PostEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.EMBED_MODEL
        self.device = device or settings.EMBED_DEVICE
        self.batch_size = settings.EMBED_BATCH_SIZE

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.embed_dim}")

    def _post_to_text(self, post: LinkedInPost) -> str:
        """
        Construct the text to embed. We concatenate the post body with
        author headline and extracted roles/skills for richer retrieval signal.
        """
        parts = [post.text]
        if post.author_title:
            parts.append(post.author_title)
        # Include any enriched fields stored as private attrs
        roles = getattr(post, "_roles", [])
        skills = getattr(post, "_skills", [])
        if roles:
            parts.append("Roles: " + ", ".join(roles))
        if skills:
            parts.append("Skills: " + ", ".join(skills))
        return " | ".join(parts)

    def embed_posts(self, posts: list[LinkedInPost]) -> EmbeddingResult:
        """Embed a list of posts. Returns (vectors, post_ids, texts)."""
        if not posts:
            return EmbeddingResult(
                np.empty((0, self.embed_dim), dtype=np.float32), [], []
            )

        texts = [DOC_PREFIX + self._post_to_text(p) for p in posts]
        post_ids = [p.post_id for p in posts]

        logger.info(f"Embedding {len(texts)} posts in batches of {self.batch_size}…")
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine via inner product after normalisation
            convert_to_numpy=True,
        ).astype(np.float32)

        logger.info(f"Embedding complete. Shape: {vectors.shape}")
        return EmbeddingResult(vectors=vectors, post_ids=post_ids, texts=texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query (with BGE instruction prefix)."""
        vec = self.model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vec.reshape(1, -1)


# ── Incremental embedding (for Airflow upserts) ───────────────────────────────

class IncrementalEmbedder(PostEmbedder):
    """
    Wraps PostEmbedder to track which post_ids have already been embedded,
    so Airflow's daily DAG only embeds new posts.
    """

    def __init__(self, seen_ids_path: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.seen_ids_path = seen_ids_path or Path("data/seen_ids.txt")
        self._seen: set[str] = self._load_seen()

    def _load_seen(self) -> set[str]:
        if self.seen_ids_path.exists():
            with open(self.seen_ids_path) as f:
                ids = {line.strip() for line in f if line.strip()}
            logger.info(f"Loaded {len(ids)} seen post IDs from {self.seen_ids_path}")
            return ids
        return set()

    def _save_seen(self) -> None:
        self.seen_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.seen_ids_path, "w") as f:
            f.write("\n".join(self._seen))

    def embed_new_posts(self, posts: list[LinkedInPost]) -> EmbeddingResult:
        """Filter to unseen posts, embed, and persist seen IDs."""
        new_posts = [p for p in posts if p.post_id not in self._seen]
        logger.info(f"{len(new_posts)} new posts out of {len(posts)} total.")

        if not new_posts:
            return EmbeddingResult(
                np.empty((0, self.embed_dim), dtype=np.float32), [], []
            )

        result = self.embed_posts(new_posts)
        self._seen.update(result.post_ids)
        self._save_seen()
        return result
