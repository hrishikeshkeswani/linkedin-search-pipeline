"""
scraper/remoteok_scraper.py
---------------------------
Fetches remote tech jobs from RemoteOK's public JSON API.
No API key or authentication required.

API docs: https://remoteok.com/api
"""

import hashlib
import logging
import time
from datetime import datetime, timezone

import urllib.request
import json

from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)

REMOTEOK_API = "https://remoteok.com/api"
TAGS_OF_INTEREST = [
    "machine-learning", "python", "data-science", "ai", "deep-learning",
    "nlp", "mlops", "backend", "devops", "cloud", "golang", "rust",
    "react", "fullstack", "engineer", "developer",
]


def _post_id(job: dict) -> str:
    key = f"remoteok:{job.get('id', '')}:{job.get('position', '')[:100]}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _job_to_post(job: dict, query: str) -> LinkedInPost:
    company = job.get("company", "Unknown")
    position = job.get("position", "")
    location = job.get("location", "Remote")
    tags = job.get("tags", [])
    description = job.get("description", "") or ""
    salary = job.get("salary", "")

    # Build a natural-language post body similar to LinkedIn hiring posts
    text_parts = [f"{company} is hiring a {position}."]
    if location:
        text_parts.append(f"Location: {location}.")
    if salary:
        text_parts.append(f"Salary: {salary}.")
    if tags:
        text_parts.append(f"Skills: {', '.join(tags[:10])}.")
    if description:
        # strip HTML tags crudely
        import re
        clean = re.sub(r"<[^>]+>", " ", description)
        clean = re.sub(r"\s+", " ", clean).strip()
        text_parts.append(clean[:800])

    text = " ".join(text_parts)
    date = job.get("date", datetime.now(timezone.utc).isoformat())

    return LinkedInPost(
        post_id=_post_id(job),
        url=job.get("url", f"https://remoteok.com/remote-jobs/{job.get('id','')}"),
        text=text,
        author=company,
        author_title=f"Hiring: {position}",
        likes=int(job.get("likes", 0)),
        posted_at=date,
        query=query,
        source="remoteok",
    )


def fetch_remoteok(queries: list[str], max_total: int = 500) -> list[LinkedInPost]:
    """
    Fetch jobs from RemoteOK and filter by query keywords.
    Returns LinkedInPost objects so the rest of the pipeline works unchanged.
    """
    logger.info("Fetching RemoteOK jobs…")

    req = urllib.request.Request(REMOTEOK_API, headers={"User-Agent": "linkedin-search-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    # First item is a metadata notice, skip it
    jobs = [j for j in data if isinstance(j, dict) and "position" in j]
    logger.info(f"RemoteOK returned {len(jobs)} total jobs")

    posts: list[LinkedInPost] = []
    seen: set[str] = set()

    for query in queries:
        keywords = query.lower().split()
        matched = 0
        for job in jobs:
            text_blob = (
                job.get("position", "") + " " +
                job.get("description", "") + " " +
                " ".join(job.get("tags", []))
            ).lower()
            if any(kw in text_blob for kw in keywords):
                pid = _post_id(job)
                if pid not in seen:
                    seen.add(pid)
                    posts.append(_job_to_post(job, query))
                    matched += 1
        logger.info(f"Query '{query}' → {matched} RemoteOK matches")

    posts = posts[:max_total]
    logger.info(f"Total RemoteOK posts collected: {len(posts)}")
    return posts
