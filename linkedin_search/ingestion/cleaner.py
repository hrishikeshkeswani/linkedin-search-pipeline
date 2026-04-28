"""
ingestion/cleaner.py
--------------------
Cleans and normalises raw LinkedIn posts before embedding.

Steps:
  1. Deduplication (exact + near-duplicate via fingerprint)
  2. Text normalisation (unicode, whitespace, emoji strip)
  3. Quality filters (min length, language, spam heuristics)
  4. Field enrichment (role/skill extraction via regex)
"""

import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from typing import Iterable

from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CHARS = 80
MAX_CHARS = 5_000

JOB_ROLES = [
    "ml engineer", "machine learning engineer", "data scientist", "data engineer",
    "backend engineer", "software engineer", "python developer", "ai engineer",
    "llm engineer", "mlops engineer", "nlp engineer", "research engineer",
    "full stack engineer", "frontend engineer", "devops engineer",
]

TECH_SKILLS = [
    "python", "pytorch", "tensorflow", "langchain", "faiss", "kubernetes",
    "docker", "fastapi", "spark", "airflow", "ray", "triton", "onnx",
    "hugging face", "transformers", "llama", "gpt", "claude", "rag",
    "vector database", "pinecone", "weaviate", "milvus", "chroma",
    "postgresql", "redis", "kafka", "gcp", "aws", "azure",
]

SPAM_PATTERNS = re.compile(
    r"(follow me|click the link|🔥{3,}|dm me for|limited seats|"
    r"comment YES|tag someone|repost this|free webinar)",
    re.IGNORECASE,
)

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.]+")
HASHTAG_RE = re.compile(r"#(\w+)")  # keep word, strip hash


# ── Text normalisation ────────────────────────────────────────────────────────

def normalise_text(text: str, strip_emoji: bool = True) -> str:
    """Clean raw LinkedIn text into indexable form."""
    # Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # Strip URLs (keep context)
    text = URL_RE.sub(" ", text)

    # Strip mentions
    text = MENTION_RE.sub(" ", text)

    # Flatten hashtags to plain words
    text = HASHTAG_RE.sub(r"\1", text)

    # Optionally strip emoji
    if strip_emoji:
        text = EMOJI_RE.sub(" ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Fingerprint for near-dedup ─────────────────────────────────────────────────

def _shingle_fingerprint(text: str, k: int = 5) -> frozenset[str]:
    """Character k-gram set for Jaccard near-dedup."""
    text = text.lower()
    return frozenset(text[i:i+k] for i in range(len(text) - k + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Field enrichment ──────────────────────────────────────────────────────────

def extract_roles(text: str) -> list[str]:
    tl = text.lower()
    return [r for r in JOB_ROLES if r in tl]


def extract_skills(text: str) -> list[str]:
    tl = text.lower()
    return [s for s in TECH_SKILLS if s in tl]


def is_hiring_post(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in ["we're hiring", "we are hiring", "open role", "open position",
                                    "job opening", "apply now", "join our team", "looking for"])


# ── Quality filter ────────────────────────────────────────────────────────────

def passes_quality(text: str) -> tuple[bool, str]:
    """Return (passes, reason) for a cleaned text string."""
    if len(text) < MIN_CHARS:
        return False, f"too_short ({len(text)} chars)"
    if len(text) > MAX_CHARS:
        return False, f"too_long ({len(text)} chars)"
    if SPAM_PATTERNS.search(text):
        return False, "spam_pattern"
    # Must have some alphabetic content
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False, f"low_alpha ({alpha_ratio:.2f})"
    return True, "ok"


# ── Main cleaner ──────────────────────────────────────────────────────────────

class PostCleaner:
    def __init__(
        self,
        dedup_exact: bool = True,
        dedup_near: bool = True,
        near_dedup_threshold: float = 0.85,
    ):
        self.dedup_exact = dedup_exact
        self.dedup_near = dedup_near
        self.near_dedup_threshold = near_dedup_threshold

        self._seen_ids: set[str] = set()
        self._fingerprints: list[tuple[frozenset, str]] = []  # (shingle_set, post_id)

        self.stats: dict[str, int] = defaultdict(int)

    def _is_near_duplicate(self, fp: frozenset, post_id: str) -> bool:
        for existing_fp, existing_id in self._fingerprints:
            if _jaccard(fp, existing_fp) >= self.near_dedup_threshold:
                logger.debug(f"Near-dup: {post_id} ~ {existing_id}")
                return True
        return False

    def clean_post(self, post: LinkedInPost) -> LinkedInPost | None:
        """Returns cleaned post or None if filtered out."""
        self.stats["total"] += 1

        # Exact dedup by post_id
        if self.dedup_exact:
            if post.post_id in self._seen_ids:
                self.stats["exact_dup"] += 1
                return None
            self._seen_ids.add(post.post_id)

        # Normalise text
        clean_text = normalise_text(post.text)

        # Quality filter
        ok, reason = passes_quality(clean_text)
        if not ok:
            self.stats[f"filtered_{reason.split('(')[0].strip()}"] += 1
            logger.debug(f"Filtered {post.post_id}: {reason}")
            return None

        # Near-dedup
        if self.dedup_near:
            fp = _shingle_fingerprint(clean_text)
            if self._is_near_duplicate(fp, post.post_id):
                self.stats["near_dup"] += 1
                return None
            self._fingerprints.append((fp, post.post_id))

        # Enrich
        roles = extract_roles(clean_text)
        skills = extract_skills(clean_text)
        hiring = is_hiring_post(clean_text)

        self.stats["kept"] += 1
        return post.model_copy(update={
            "text": clean_text,
            # Store enrichment in a metadata field (we add dynamically)
            **{
                "_roles": roles,
                "_skills": skills,
                "_is_hiring": hiring,
            }
        })

    def clean_batch(self, posts: Iterable[LinkedInPost]) -> list[LinkedInPost]:
        cleaned = []
        for post in posts:
            result = self.clean_post(post)
            if result is not None:
                cleaned.append(result)
        logger.info(
            f"Cleaning complete. Stats: {dict(self.stats)}"
        )
        return cleaned

    def report(self) -> dict:
        return dict(self.stats)
