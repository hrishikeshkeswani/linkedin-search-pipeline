"""
Central config — loaded from environment variables with sensible defaults.
Copy .env.example → .env and fill in your values.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── Groq ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"     # used for query expansion / reranking

    # ── Embeddings ────────────────────────────────────────────────────
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"  # local sentence-transformers
    EMBED_BATCH_SIZE: int = 64
    EMBED_DEVICE: str = "cpu"                    # "cuda" if GPU available

    # ── FAISS ─────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: Path = Path("data/faiss_index")
    FAISS_METADATA_PATH: Path = Path("data/metadata.jsonl")
    FAISS_NLIST: int = 128                       # IVF cells (tune for corpus size)
    FAISS_NPROBE: int = 16                       # cells searched at query time

    # ── Scraper ───────────────────────────────────────────────────────
    SCRAPER_HEADLESS: bool = True
    SCRAPER_RATE_LIMIT_SECS: float = 3.0        # polite delay between requests
    SCRAPER_MAX_POSTS: int = 500                 # per run cap
    SCRAPER_OUTPUT_DIR: Path = Path("data/raw")

    # ── LinkedIn (optional session cookie auth) ────────────────────────
    LINKEDIN_SESSION_COOKIE: str = ""           # li_at cookie value

    # ── Airflow / TTL ─────────────────────────────────────────────────
    VECTOR_TTL_DAYS: int = 30                   # evict vectors older than this
    AIRFLOW_DAG_SCHEDULE: str = "@daily"

    # ── GCS (optional remote persistence) ─────────────────────────────
    GCS_BUCKET: str = ""
    GCS_INDEX_PREFIX: str = "faiss/"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
