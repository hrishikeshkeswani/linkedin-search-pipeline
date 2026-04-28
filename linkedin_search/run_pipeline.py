"""
run_pipeline.py
---------------
Local runner — executes the full ingestion pipeline without Airflow.
Useful for initial data load and development.

Usage:
    # With mock data (no LinkedIn account needed):
    python run_pipeline.py --mock --queries "python engineer" "ML jobs" --n 200

    # With real scraping:
    python run_pipeline.py --queries "python engineer" "ML engineer hiring" --max 500
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.settings import settings
from scraper.linkedin_scraper import LinkedInScraper, MockScraper, LinkedInPost
from scraper.remoteok_scraper import fetch_remoteok
from scraper.adzuna_scraper import fetch_adzuna
from scraper.jobspy_scraper import fetch_jobspy
from scraper.hn_scraper import fetch_hn_hiring
from scraper.reddit_scraper import fetch_reddit_hiring
from ingestion.cleaner import PostCleaner
from ingestion.embedder import IncrementalEmbedder
from indexer.faiss_store import FAISSStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline")


def run(
    queries: list[str],
    max_per_query: int = 100,
    use_mock: bool = False,
    skip_scrape: bool = False,
    raw_file: str | None = None,
    source: str = "linkedin",   # linkedin | remoteok | adzuna | mock
) -> dict:
    """
    Full pipeline: scrape → clean → embed → index.
    Returns a summary dict.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = settings.SCRAPER_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Scrape ────────────────────────────────────────────────
    if skip_scrape and raw_file:
        log.info(f"Skipping scrape — loading from {raw_file}")
        raw_path = Path(raw_file)
    elif use_mock or source == "mock":
        log.info("Generating mock posts…")
        scraper = MockScraper()
        posts: list[LinkedInPost] = []
        for q in queries:
            posts.extend(scraper.generate(n=max_per_query, query=q))
        raw_path = output_dir / f"mock_{ts}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(p.model_dump_json() + "\n")
        log.info(f"Mock posts written → {raw_path}")
    elif source == "remoteok":
        log.info("Fetching from RemoteOK…")
        posts = fetch_remoteok(queries, max_total=max_per_query * len(queries))
        raw_path = output_dir / f"remoteok_{ts}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(p.model_dump_json() + "\n")
        log.info(f"RemoteOK posts written → {raw_path}")
    elif source == "reddit":
        log.info("Fetching from Reddit hiring posts...")
        posts = fetch_reddit_hiring(queries, max_total=max_per_query * len(queries))
        raw_path = output_dir / f"reddit_{ts}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(p.model_dump_json() + "\n")
        log.info(f"Reddit posts written -> {raw_path}")
    elif source == "hn":
        log.info("Fetching from Hacker News Who's Hiring threads...")
        posts = fetch_hn_hiring(queries, max_total=max_per_query * len(queries))
        raw_path = output_dir / f"hn_{ts}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(p.model_dump_json() + "\n")
        log.info(f"HN posts written -> {raw_path}")
    elif source == "jobspy":
        log.info("Fetching from JobSpy (LinkedIn + Indeed + Glassdoor)...")
        posts = fetch_jobspy(queries, max_per_query=max_per_query)
        raw_path = output_dir / f"jobspy_{ts}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(p.model_dump_json() + "\n")
        log.info(f"JobSpy posts written -> {raw_path}")
    elif source == "adzuna":
        log.info("Fetching from Adzuna…")
        posts = fetch_adzuna(queries, max_per_query=max_per_query)
        raw_path = output_dir / f"adzuna_{ts}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for p in posts:
                f.write(p.model_dump_json() + "\n")
        log.info(f"Adzuna posts written → {raw_path}")
    else:
        log.info("Starting live LinkedIn scrape…")
        raw_path = output_dir / f"raw_{ts}.jsonl"
        posts = asyncio.run(
            LinkedInScraper(output_dir).run(queries, max_per_query, str(raw_path.name))
        )

    # ── Step 2: Load raw posts ────────────────────────────────────────
    raw_posts: list[LinkedInPost] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_posts.append(LinkedInPost.model_validate_json(line))
    log.info(f"Loaded {len(raw_posts)} raw posts from {raw_path}")

    # ── Step 3: Clean ─────────────────────────────────────────────────
    log.info("Cleaning posts…")
    cleaner = PostCleaner()
    cleaned = cleaner.clean_batch(raw_posts)
    log.info(f"After cleaning: {len(cleaned)} posts. Stats: {cleaner.report()}")

    clean_path = raw_path.with_suffix(".cleaned.jsonl")
    with open(clean_path, "w", encoding="utf-8") as f:
        for p in cleaned:
            f.write(p.model_dump_json() + "\n")

    if not cleaned:
        log.warning("No posts survived cleaning. Check filters or data quality.")
        return {"error": "no_posts_after_cleaning"}

    # ── Step 4: Embed ─────────────────────────────────────────────────
    log.info("Embedding posts…")
    embedder = IncrementalEmbedder()
    result = embedder.embed_new_posts(cleaned)

    if result.vectors.shape[0] == 0:
        log.info("All posts already embedded (incremental dedup). Nothing to add to index.")
        return {"status": "already_indexed", "n_posts": len(cleaned)}

    vec_path = clean_path.with_suffix(".npy")
    np.save(str(vec_path), result.vectors)
    log.info(f"Saved {result.vectors.shape} vectors → {vec_path}")

    # ── Step 5: Index ─────────────────────────────────────────────────
    log.info("Building / updating FAISS index…")
    store = FAISSStore(embed_dim=result.vectors.shape[1])

    # Match posts to embedded IDs
    embedded_id_set = set(result.post_ids)
    posts_to_index = [p for p in cleaned if p.post_id in embedded_id_set]

    n_added = store.add(result.vectors, posts_to_index)
    store.save()

    summary = {
        "status": "ok",
        "n_raw": len(raw_posts),
        "n_cleaned": len(cleaned),
        "n_embedded": result.vectors.shape[0],
        "n_indexed": n_added,
        "index_stats": store.stats(),
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
    }
    log.info(f"Pipeline complete: {json.dumps(summary, indent=2, default=str)}")
    return summary


# ── Quick search test ─────────────────────────────────────────────────────────

def test_search(query: str, k: int = 5) -> None:
    """Run a quick search against the built index."""
    from ingestion.embedder import PostEmbedder
    from indexer.faiss_store import FAISSStore

    embedder = PostEmbedder()
    store = FAISSStore(embed_dim=embedder.embed_dim)

    if store.index is None:
        print("Index not found. Run the pipeline first.")
        return

    qvec = embedder.embed_query(query)
    results = store.search(qvec, k=k)

    print(f"\n🔍 Query: '{query}' — Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score: {r['score']:.4f}")
        print(f"       Author: {r['author']} ({r['author_title']})")
        print(f"       Text:   {r['text'][:160]}…")
        print(f"       URL:    {r['url']}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinkedIn search ingestion pipeline")
    parser.add_argument("--queries", nargs="+", default=["python engineer", "ML engineer hiring"])
    parser.add_argument("--max", type=int, default=100, help="Max posts per query")
    parser.add_argument("--mock", action="store_true", help="Use mock data (same as --source mock)")
    parser.add_argument("--source", default="linkedin", choices=["linkedin", "remoteok", "adzuna", "jobspy", "hn", "reddit", "mock"],
                        help="Data source (default: linkedin)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping, use --raw-file")
    parser.add_argument("--raw-file", default=None, help="Existing raw JSONL to process")
    parser.add_argument("--search", default=None, help="Run a test search after indexing")
    args = parser.parse_args()

    result = run(
        queries=args.queries,
        max_per_query=args.max,
        use_mock=args.mock,
        skip_scrape=args.skip_scrape,
        raw_file=args.raw_file,
        source=args.source,
    )
    print("\nPipeline result:", json.dumps(result, indent=2, default=str))

    if args.search:
        test_search(args.search)
