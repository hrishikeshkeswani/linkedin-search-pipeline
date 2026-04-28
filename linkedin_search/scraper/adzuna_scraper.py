"""
scraper/adzuna_scraper.py
-------------------------
Fetches job listings from the Adzuna API.
Free tier: 250 requests/day — sign up at https://developer.adzuna.com

Set in .env:
    ADZUNA_APP_ID=your_app_id
    ADZUNA_API_KEY=your_api_key
    ADZUNA_COUNTRY=us   # us, gb, ca, au, de, fr, in, etc.
"""

import hashlib
import logging
import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)

ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"


def _post_id(job: dict) -> str:
    key = f"adzuna:{job.get('id', '')}:{job.get('title', '')[:100]}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _job_to_post(job: dict, query: str) -> LinkedInPost:
    company = job.get("company", {}).get("display_name", "Unknown Company")
    title = job.get("title", "")
    location = job.get("location", {}).get("display_name", "")
    description = job.get("description", "")
    salary_min = job.get("salary_min")
    salary_max = job.get("salary_max")
    category = job.get("category", {}).get("label", "")

    text_parts = [f"{company} is hiring a {title}."]
    if location:
        text_parts.append(f"Location: {location}.")
    if category:
        text_parts.append(f"Category: {category}.")
    if salary_min and salary_max:
        text_parts.append(f"Salary: ${int(salary_min):,}–${int(salary_max):,}.")
    if description:
        text_parts.append(description[:800])

    return LinkedInPost(
        post_id=_post_id(job),
        url=job.get("redirect_url", ""),
        text=" ".join(text_parts),
        author=company,
        author_title=f"Hiring: {title}",
        likes=0,
        posted_at=job.get("created", datetime.now(timezone.utc).isoformat()),
        query=query,
        source="adzuna",
    )


def fetch_adzuna(
    queries: list[str],
    max_per_query: int = 50,
    country: str | None = None,
) -> list[LinkedInPost]:
    """
    Fetch jobs from Adzuna API for each query.
    Requires ADZUNA_APP_ID and ADZUNA_API_KEY in environment.
    """
    app_id = os.environ.get("ADZUNA_APP_ID", "")
    api_key = os.environ.get("ADZUNA_API_KEY", "")
    country = country or os.environ.get("ADZUNA_COUNTRY", "us")

    if not app_id or not api_key:
        raise ValueError(
            "ADZUNA_APP_ID and ADZUNA_API_KEY must be set in .env\n"
            "Sign up free at https://developer.adzuna.com"
        )

    posts: list[LinkedInPost] = []
    seen: set[str] = set()

    for query in queries:
        page = 1
        collected = 0
        while collected < max_per_query:
            results_per_page = min(50, max_per_query - collected)
            params = urllib.parse.urlencode({
                "app_id": app_id,
                "app_key": api_key,
                "results_per_page": results_per_page,
                "what": query,
                "content-type": "application/json",
            })
            url = ADZUNA_BASE.format(country=country, page=page) + "?" + params

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "linkedin-search-pipeline/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                logger.warning(f"Adzuna request failed for '{query}' page {page}: {e}")
                break

            jobs = data.get("results", [])
            if not jobs:
                break

            for job in jobs:
                pid = _post_id(job)
                if pid not in seen:
                    seen.add(pid)
                    posts.append(_job_to_post(job, query))
                    collected += 1

            logger.info(f"Adzuna '{query}' page {page} → {len(jobs)} jobs")
            page += 1

            if len(jobs) < results_per_page:
                break

    logger.info(f"Total Adzuna posts: {len(posts)}")
    return posts
