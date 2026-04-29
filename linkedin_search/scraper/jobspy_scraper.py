"""
scraper/jobspy_scraper.py
-------------------------
Fetches job listings via python-jobspy (LinkedIn, Indeed, Glassdoor, ZipRecruiter).
No cookies, no browser automation, no account needed.

Usage:
    python run_pipeline.py --source jobspy --queries "machine learning engineer" "data scientist"
"""

import hashlib
import logging
from datetime import datetime, timezone

from jobspy import scrape_jobs

from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)


def _post_id(job_id: str, title: str) -> str:
    key = f"jobspy:{job_id}:{title[:100]}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _job_to_post(row, query: str) -> LinkedInPost:
    title = str(row.get("title", "")) or ""
    company = str(row.get("company", "Unknown")) or "Unknown"
    location = str(row.get("location", "")) or ""
    description = str(row.get("description", "")) or ""
    job_url = str(row.get("job_url", "")) or ""
    site = str(row.get("site", "linkedin")) or "linkedin"
    date_posted = row.get("date_posted")
    min_salary = row.get("min_amount")
    max_salary = row.get("max_amount")
    job_type = str(row.get("job_type", "")) or ""
    is_remote = row.get("is_remote", False)

    # Build a natural-language post body
    parts = [f"{company} is hiring a {title}."]
    if location:
        parts.append(f"Location: {location}.")
    if is_remote:
        parts.append("This is a remote role.")
    if job_type:
        parts.append(f"Job type: {job_type}.")
    try:
        if min_salary and max_salary and min_salary == min_salary and max_salary == max_salary:
            parts.append(f"Salary: ${int(min_salary):,} - ${int(max_salary):,}.")
    except (ValueError, TypeError):
        pass
    if description:
        # First 800 chars of the description
        clean_desc = description[:800].strip()
        if clean_desc:
            parts.append(clean_desc)

    text = " ".join(parts)

    posted_at = ""
    if date_posted:
        try:
            if hasattr(date_posted, "isoformat"):
                posted_at = date_posted.isoformat()
            else:
                posted_at = str(date_posted)
        except Exception:
            pass

    job_id = str(row.get("id", "")) or job_url

    return LinkedInPost(
        post_id=_post_id(job_id, title),
        url=job_url,
        text=text,
        author=company,
        author_title=f"Hiring: {title}",
        likes=0,
        posted_at=posted_at,
        query=query,
        source="jobspy",
    )


def fetch_jobspy(
    queries: list[str],
    max_per_query: int = 50,
    sites: list[str] | None = None,
    country: str = "USA",
) -> list[LinkedInPost]:
    """
    Scrape jobs via JobSpy for each query.
    sites: list of ["linkedin", "indeed", "glassdoor", "zip_recruiter"] — defaults to all.
    """
    sites = sites or ["linkedin", "indeed", "zip_recruiter"]

    posts: list[LinkedInPost] = []
    seen: set[str] = set()

    for query in queries:
        logger.info(f"JobSpy scraping: '{query}' from {sites}")
        try:
            df = scrape_jobs(
                site_name=sites,
                search_term=query,
                location=country,
                results_wanted=max_per_query,
                hours_old=72,           # only posts from last 3 days
                country_indeed=country,
            )
        except Exception as e:
            logger.warning(f"JobSpy failed for '{query}': {e}")
            continue

        if df is None or df.empty:
            logger.info(f"No results for '{query}'")
            continue

        count = 0
        for _, row in df.iterrows():
            post = _job_to_post(row.to_dict(), query)
            if post.post_id not in seen:
                seen.add(post.post_id)
                posts.append(post)
                count += 1

        logger.info(f"JobSpy '{query}' -> {count} jobs from {len(df)} results")

    logger.info(f"Total JobSpy posts: {len(posts)}")
    return posts
