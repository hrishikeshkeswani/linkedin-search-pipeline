# LinkedIn Search Optimizer — Data Ingestion Pipeline

A RAG-based LinkedIn job search engine. This module handles the full data pipeline:
scraping → cleaning → embedding → FAISS indexing.

## Quick start (mock data, no LinkedIn account needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Copy and configure environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 3. Run the pipeline with mock data
python run_pipeline.py --mock --queries "python engineer" "ML jobs" --max 200

# 4. Test search
python run_pipeline.py --mock --queries "python engineer" --max 50 --search "senior ML engineer remote"
```

## Project structure

```
linkedin_search/
├── config/
│   └── settings.py          # Pydantic settings (loads from .env)
├── scraper/
│   └── linkedin_scraper.py  # Playwright scraper + MockScraper for dev
├── ingestion/
│   ├── cleaner.py           # Text normalisation, dedup, quality filters
│   ├── embedder.py          # BGE-large-en embeddings (sentence-transformers)
│   └── groq_reranker.py     # Groq query expansion + LLM reranking
├── indexer/
│   └── faiss_store.py       # FAISS IVFFlat index + metadata + TTL eviction
├── airflow/
│   └── dags/
│       └── linkedin_ingestion_dag.py  # Daily ingest + weekly TTL refresh DAGs
├── run_pipeline.py          # Local runner (no Airflow needed)
├── requirements.txt
└── .env.example
```

## With real LinkedIn data

1. Open LinkedIn in Chrome, log in
2. Open DevTools → Application → Cookies → `.linkedin.com`
3. Copy the `li_at` cookie value into your `.env` as `LINKEDIN_SESSION_COOKIE`
4. Run: `python run_pipeline.py --queries "python engineer" "ML engineer hiring" --max 500`

## Airflow deployment (GCP)

```bash
# Set Airflow variables
airflow variables set LINKEDIN_QUERIES '["python engineer", "ML engineer hiring", "data scientist jobs"]'
airflow variables set USE_MOCK_SCRAPER false

# Place DAG in your Airflow DAGs folder
cp airflow/dags/linkedin_ingestion_dag.py $AIRFLOW_HOME/dags/

# Trigger manually to test
airflow dags trigger linkedin_ingest_daily
```

## Next steps (Phase 2)

- `fastapi_service/` — FastAPI search endpoint with `/search`, `/health`, `/reindex`
- `eval/` — Recall@10 evaluation against labeled query-job pairs
- `k8s/` — Kubernetes manifests + Dockerfile
