"""
eval/recall_at_k.py
-------------------
Evaluates Recall@K on labeled query-post pairs.

Recall@K = (# relevant posts in top-K) / (# total relevant posts for query)

Usage:
    python -m eval.recall_at_k --labels eval/labels.json --k 10
    python -m eval.recall_at_k --labels eval/labels.json --k 10 --no-expand --no-rerank

Label file format (eval/labels.json):
    [
      {
        "query": "senior ML engineer remote jobs",
        "relevant_post_ids": ["post_001", "post_042", "post_107"]
      },
      ...
    ]

Outputs:
    - Per-query Recall@K
    - Mean Recall@K across all queries
    - Optional: save detailed results to eval/results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

# Allow running as: python -m eval.recall_at_k from linkedin_search/
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from indexer.faiss_store import FAISSStore
from ingestion.embedder import PostEmbedder
from ingestion.groq_reranker import groq_enhanced_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────────

class QueryLabel(NamedTuple):
    query: str
    relevant_ids: set[str]


class EvalResult(NamedTuple):
    query: str
    recall_at_k: float
    retrieved_ids: list[str]
    relevant_ids: list[str]
    hits: list[str]         # intersection of retrieved & relevant
    k: int


# ── Core metric ────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ top-K retrieved| / |relevant|
    Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = relevant_ids.intersection(top_k)
    return len(hits) / len(relevant_ids)


# ── Evaluation runner ──────────────────────────────────────────────────────────

def run_eval(
    labels: list[QueryLabel],
    store: FAISSStore,
    embedder: PostEmbedder,
    k: int = 10,
    expand: bool = True,
    rerank: bool = True,
) -> list[EvalResult]:
    results = []
    for label in labels:
        logger.info(f"Evaluating: '{label.query}'")
        hits_raw = groq_enhanced_search(
            query=label.query,
            store=store,
            embedder=embedder,
            k=k,
            expand=expand,
            rerank_results=rerank,
        )
        retrieved_ids = [r["post_id"] for r in hits_raw]
        r_at_k = recall_at_k(retrieved_ids, label.relevant_ids, k)
        hits = list(label.relevant_ids.intersection(retrieved_ids[:k]))

        results.append(EvalResult(
            query=label.query,
            recall_at_k=r_at_k,
            retrieved_ids=retrieved_ids,
            relevant_ids=list(label.relevant_ids),
            hits=hits,
            k=k,
        ))
        logger.info(f"  Recall@{k} = {r_at_k:.3f}  ({len(hits)}/{len(label.relevant_ids)} relevant found)")

    return results


def mean_recall(results: list[EvalResult]) -> float:
    if not results:
        return 0.0
    return sum(r.recall_at_k for r in results) / len(results)


# ── Label generation helpers ───────────────────────────────────────────────────

def generate_mock_labels(store: FAISSStore, n_queries: int = 20) -> list[dict]:
    """
    Bootstrap a label file from the index using exact keyword match as a proxy.
    Useful for getting started without manual annotation.
    """
    sample_queries = [
        "machine learning engineer remote",
        "data scientist hiring Python",
        "MLOps engineer Kubernetes",
        "NLP engineer transformer models",
        "AI research scientist deep learning",
        "backend engineer Python FastAPI",
        "data engineer Spark Airflow",
        "software engineer ML platform",
        "computer vision engineer PyTorch",
        "LLM engineer fine-tuning",
        "senior data scientist hiring now",
        "ML infrastructure engineer",
        "full stack engineer AI startup",
        "platform engineer GCP Kubernetes",
        "research engineer Anthropic OpenAI",
        "applied scientist recommendations",
        "data science manager remote",
        "reinforcement learning engineer",
        "generative AI engineer open role",
        "ML engineer new grad entry level",
    ]

    labels = []
    meta_records = list(store._meta.values())

    for q in sample_queries[:n_queries]:
        q_lower = q.lower()
        relevant = [
            rec["post_id"]
            for rec in meta_records
            if any(word in rec.get("text", "").lower() for word in q_lower.split())
        ][:5]   # cap at 5 relevant per query for bootstrap labels
        if relevant:
            labels.append({"query": q, "relevant_post_ids": relevant})

    return labels


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Recall@K for the LinkedIn search pipeline")
    parser.add_argument("--labels", type=Path, default=Path("eval/labels.json"),
                        help="Path to labeled query-post JSON file")
    parser.add_argument("--k", type=int, default=10, help="K for Recall@K (default: 10)")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    parser.add_argument("--no-rerank", action="store_true", help="Disable LLM reranking")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save detailed results JSON to this path")
    parser.add_argument("--generate-labels", action="store_true",
                        help="Auto-generate bootstrap labels from index and save to --labels path")
    args = parser.parse_args()

    store = FAISSStore()
    if store.stats()["total_vectors"] == 0:
        logger.error("FAISS index is empty. Run the pipeline first.")
        sys.exit(1)

    embedder = PostEmbedder()

    # Generate labels if requested
    if args.generate_labels:
        logger.info("Generating bootstrap labels from index…")
        mock = generate_mock_labels(store)
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        args.labels.write_text(json.dumps(mock, indent=2))
        logger.info(f"Saved {len(mock)} bootstrap labels to {args.labels}")

    # Load labels
    if not args.labels.exists():
        logger.error(f"Labels file not found: {args.labels}. Use --generate-labels to bootstrap.")
        sys.exit(1)

    raw_labels = json.loads(args.labels.read_text())
    labels = [
        QueryLabel(query=l["query"], relevant_ids=set(l["relevant_post_ids"]))
        for l in raw_labels
    ]
    logger.info(f"Loaded {len(labels)} labeled queries.")

    # Run evaluation
    results = run_eval(
        labels=labels,
        store=store,
        embedder=embedder,
        k=args.k,
        expand=not args.no_expand,
        rerank=not args.no_rerank,
    )

    # Report
    mean_r = mean_recall(results)
    print(f"\n{'='*50}")
    print(f"Recall@{args.k} Results")
    print(f"{'='*50}")
    for r in results:
        bar = "█" * int(r.recall_at_k * 20)
        print(f"  {r.recall_at_k:.3f}  {bar:<20}  {r.query[:60]}")
    print(f"{'─'*50}")
    print(f"  Mean Recall@{args.k}: {mean_r:.4f}")
    print(f"  Queries evaluated: {len(results)}")
    print(f"  Query expansion: {'on' if not args.no_expand else 'off'}")
    print(f"  LLM reranking:   {'on' if not args.no_rerank else 'off'}")
    print(f"{'='*50}\n")

    # Save detailed output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "mean_recall_at_k": mean_r,
            "k": args.k,
            "expand": not args.no_expand,
            "rerank": not args.no_rerank,
            "per_query": [
                {
                    "query": r.query,
                    "recall_at_k": r.recall_at_k,
                    "hits": r.hits,
                    "relevant_ids": r.relevant_ids,
                    "retrieved_ids": r.retrieved_ids,
                }
                for r in results
            ],
        }
        args.output.write_text(json.dumps(out, indent=2))
        logger.info(f"Detailed results saved to {args.output}")

    return mean_r


if __name__ == "__main__":
    main()
