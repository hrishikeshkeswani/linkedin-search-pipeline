"""
airflow/dags/linkedin_ingestion_dag.py
---------------------------------------
Two DAGs:

1. linkedin_ingest_daily
   Schedule: @daily
   Tasks: scrape → clean → embed → upsert_to_faiss → upload_to_gcs

2. linkedin_ttl_refresh_weekly
   Schedule: @weekly
   Tasks: evict_stale_vectors → rebuild_index → upload_to_gcs

Prerequisites:
  - Airflow variables: LINKEDIN_QUERIES (JSON list), GROQ_API_KEY
  - Airflow connection: google_cloud_default (for GCS, optional)
  - pip install apache-airflow sentence-transformers faiss-cpu playwright groq
"""

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── Default args ──────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "linkedin_search",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_queries() -> list[str]:
    """Pull search queries from Airflow Variable (JSON list)."""
    raw = Variable.get("LINKEDIN_QUERIES", default_var='["python engineer", "ML engineer hiring"]')
    return json.loads(raw)


def _get_raw_dir():
    from pathlib import Path
    return Path(Variable.get("SCRAPER_OUTPUT_DIR", default_var="data/raw"))


# ── Task functions ────────────────────────────────────────────────────────────

def task_scrape(**context):
    """Scrape LinkedIn posts for all configured queries."""
    import asyncio
    import os
    from scraper.linkedin_scraper import LinkedInScraper, MockScraper
    from config.settings import settings

    queries = _get_queries()
    use_mock = Variable.get("USE_MOCK_SCRAPER", default_var="false").lower() == "true"

    if use_mock:
        log.info("Using mock scraper (USE_MOCK_SCRAPER=true)")
        scraper = MockScraper()
        from datetime import timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = _get_raw_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        all_posts = []
        for q in queries:
            all_posts.extend(scraper.generate(n=100, query=q))
        out_path = out_dir / f"mock_{ts}.jsonl"
        with open(out_path, "w") as f:
            for p in all_posts:
                f.write(p.model_dump_json() + "\n")
        context["ti"].xcom_push(key="raw_path", value=str(out_path))
        return

    scraper = LinkedInScraper(output_dir=_get_raw_dir())
    posts = asyncio.run(scraper.run(queries))
    # raw_path is set inside LinkedInScraper.run; we find it via glob
    raw_files = sorted(_get_raw_dir().glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    context["ti"].xcom_push(key="raw_path", value=str(raw_files[-1]))
    log.info(f"Scraped {len(posts)} posts.")


def task_clean(**context):
    """Load raw JSONL, clean posts, write cleaned JSONL."""
    from ingestion.cleaner import PostCleaner
    from scraper.linkedin_scraper import LinkedInPost
    from pathlib import Path

    raw_path = context["ti"].xcom_pull(key="raw_path", task_ids="scrape")
    clean_path = Path(raw_path).with_suffix(".cleaned.jsonl")

    posts = []
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                posts.append(LinkedInPost.model_validate_json(line))

    log.info(f"Loaded {len(posts)} raw posts from {raw_path}")
    cleaner = PostCleaner()
    cleaned = cleaner.clean_batch(posts)
    log.info(f"Cleaning stats: {cleaner.report()}")

    with open(clean_path, "w") as f:
        for p in cleaned:
            f.write(p.model_dump_json() + "\n")

    context["ti"].xcom_push(key="clean_path", value=str(clean_path))
    context["ti"].xcom_push(key="n_clean", value=len(cleaned))
    log.info(f"Wrote {len(cleaned)} cleaned posts → {clean_path}")


def task_embed(**context):
    """Embed cleaned posts, write vectors to .npy sidecar."""
    import numpy as np
    from ingestion.embedder import IncrementalEmbedder
    from scraper.linkedin_scraper import LinkedInPost
    from pathlib import Path

    clean_path = context["ti"].xcom_pull(key="clean_path", task_ids="clean")
    vec_path = Path(clean_path).with_suffix(".npy")
    ids_path = Path(clean_path).with_suffix(".ids.json")

    posts = []
    with open(clean_path) as f:
        for line in f:
            if line.strip():
                posts.append(LinkedInPost.model_validate_json(line))

    embedder = IncrementalEmbedder()
    result = embedder.embed_new_posts(posts)

    if result.vectors.shape[0] == 0:
        log.info("No new posts to embed.")
        context["ti"].xcom_push(key="vec_path", value="")
        return

    np.save(str(vec_path), result.vectors)
    with open(ids_path, "w") as f:
        json.dump(result.post_ids, f)

    context["ti"].xcom_push(key="vec_path", value=str(vec_path))
    context["ti"].xcom_push(key="ids_path", value=str(ids_path))
    context["ti"].xcom_push(key="n_vectors", value=result.vectors.shape[0])
    log.info(f"Embedded {result.vectors.shape[0]} vectors → {vec_path}")


def task_upsert_faiss(**context):
    """Add new embeddings to the persistent FAISS index."""
    import numpy as np
    from indexer.faiss_store import FAISSStore
    from scraper.linkedin_scraper import LinkedInPost
    from pathlib import Path

    vec_path = context["ti"].xcom_pull(key="vec_path", task_ids="embed")
    if not vec_path:
        log.info("No new vectors to upsert.")
        return

    ids_path = context["ti"].xcom_pull(key="ids_path", task_ids="embed")
    clean_path = context["ti"].xcom_pull(key="clean_path", task_ids="clean")

    vectors = np.load(vec_path)
    with open(ids_path) as f:
        new_ids = set(json.load(f))

    posts = []
    with open(clean_path) as f:
        for line in f:
            if line.strip():
                p = LinkedInPost.model_validate_json(line)
                if p.post_id in new_ids:
                    posts.append(p)

    store = FAISSStore()
    n_added = store.add(vectors, posts)
    store.save()

    log.info(f"Upserted {n_added} vectors. Index stats: {store.stats()}")
    context["ti"].xcom_push(key="index_stats", value=json.dumps(store.stats()))


def task_upload_gcs(**context):
    """Upload index + metadata to GCS (skipped if GCS_BUCKET not set)."""
    from config.settings import settings

    if not settings.GCS_BUCKET:
        log.info("GCS_BUCKET not configured — skipping upload.")
        return

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(settings.GCS_BUCKET)

        files = [
            (settings.FAISS_INDEX_PATH / "index.bin", f"{settings.GCS_INDEX_PREFIX}index.bin"),
            (settings.FAISS_METADATA_PATH, f"{settings.GCS_INDEX_PREFIX}metadata.jsonl"),
        ]
        for local, remote in files:
            if local.exists():
                bucket.blob(remote).upload_from_filename(str(local))
                log.info(f"Uploaded {local} → gs://{settings.GCS_BUCKET}/{remote}")

    except ImportError:
        log.warning("google-cloud-storage not installed — skipping GCS upload.")


def task_evict_stale(**context):
    """TTL eviction: remove vectors older than VECTOR_TTL_DAYS."""
    from indexer.faiss_store import FAISSStore
    from config.settings import settings

    store = FAISSStore()
    n_evicted = store.evict_stale(ttl_days=settings.VECTOR_TTL_DAYS)
    store.save()
    log.info(f"Evicted {n_evicted} stale vectors. Remaining: {store.stats()['total_vectors']}")
    context["ti"].xcom_push(key="n_evicted", value=n_evicted)


# ── DAG 1: Daily ingestion ─────────────────────────────────────────────────────

with DAG(
    dag_id="linkedin_ingest_daily",
    description="Scrape → clean → embed → index LinkedIn posts daily",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    default_args=DEFAULT_ARGS,
    catchup=False,
    tags=["linkedin", "ingestion", "rag"],
) as ingest_dag:

    scrape = PythonOperator(task_id="scrape", python_callable=task_scrape)
    clean = PythonOperator(task_id="clean", python_callable=task_clean)
    embed = PythonOperator(task_id="embed", python_callable=task_embed)
    upsert = PythonOperator(task_id="upsert_faiss", python_callable=task_upsert_faiss)
    upload = PythonOperator(task_id="upload_gcs", python_callable=task_upload_gcs)

    scrape >> clean >> embed >> upsert >> upload


# ── DAG 2: Weekly TTL refresh ──────────────────────────────────────────────────

with DAG(
    dag_id="linkedin_ttl_refresh_weekly",
    description="Evict stale FAISS vectors and upload refreshed index weekly",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    default_args=DEFAULT_ARGS,
    catchup=False,
    tags=["linkedin", "ttl", "rag"],
) as ttl_dag:

    evict = PythonOperator(task_id="evict_stale", python_callable=task_evict_stale)
    upload_ttl = PythonOperator(task_id="upload_gcs", python_callable=task_upload_gcs)

    evict >> upload_ttl
