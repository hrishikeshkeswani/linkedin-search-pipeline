"""
scraper/hn_scraper.py
---------------------
Scrapes Hacker News "Who's Hiring" monthly threads.
These are feed-style posts where people write about open roles —
exactly like LinkedIn hiring posts but from HN.

No API key, no auth, completely free.
API docs: https://github.com/HackerNews/API
"""

import hashlib
import html
import logging
import re
import urllib.request
import json
from datetime import datetime, timezone

from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)

HN_API = "https://hacker-news.firebaseio.com/v0"
ALGOLIA_API = "https://hn.algolia.com/api/v1"


def _clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<p>|</p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _post_id(item_id: int) -> str:
    return hashlib.md5(f"hn:{item_id}".encode()).hexdigest()[:16]


def _fetch_json(url: str) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "linkedin-search-pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Fetch failed {url}: {e}")
        return None


def get_whos_hiring_threads(n: int = 3) -> list[dict]:
    """Find the most recent N 'Who is hiring?' threads on HN."""
    data = _fetch_json(
        f"{ALGOLIA_API}/search_by_date?query=Ask+HN+Who+is+hiring&tags=story,ask_hn&hitsPerPage=20"
    )
    if not data:
        return []

    threads = []
    for hit in data.get("hits", []):
        title = hit.get("title", "")
        if "who is hiring" in title.lower() and "ask hn" in title.lower():
            threads.append({
                "id": hit["objectID"],
                "title": title,
                "date": hit.get("created_at", ""),
            })
        if len(threads) >= n:
            break

    logger.info(f"Found {len(threads)} Who's Hiring threads: {[t['title'] for t in threads]}")
    return threads


def fetch_hn_hiring(
    queries: list[str],
    max_total: int = 500,
    n_threads: int = 3,
) -> list[LinkedInPost]:
    """
    Fetch comments from HN Who's Hiring threads matching the given queries.
    Each comment is a person/company writing about an open role.
    """
    threads = get_whos_hiring_threads(n_threads)
    if not threads:
        logger.warning("No HN hiring threads found.")
        return []

    posts: list[LinkedInPost] = []
    seen: set[str] = set()
    keywords = [q.lower().split() for q in queries]

    for thread in threads:
        thread_id = thread["id"]
        logger.info(f"Fetching comments from: {thread['title']}")

        # Use Algolia to get all comments from this thread
        page = 0
        while len(posts) < max_total:
            data = _fetch_json(
                f"{ALGOLIA_API}/search?tags=comment,story_{thread_id}&hitsPerPage=100&page={page}"
            )
            if not data or not data.get("hits"):
                break

            for hit in data["hits"]:
                text_raw = hit.get("comment_text") or hit.get("story_text") or ""
                if not text_raw or len(text_raw) < 80:
                    continue

                text = _clean_html(text_raw)

                # Filter by query keywords
                text_lower = text.lower()
                if not any(
                    any(kw in text_lower for kw in kws)
                    for kws in keywords
                ):
                    continue

                item_id = int(hit.get("objectID", 0))
                pid = _post_id(item_id)
                if pid in seen:
                    continue
                seen.add(pid)

                author = hit.get("author", "Anonymous")
                created = hit.get("created_at", datetime.now(timezone.utc).isoformat())
                url = f"https://news.ycombinator.com/item?id={item_id}"

                # Extract company/role from first line (common HN format:
                # "CompanyName | Role | Location | ...")
                first_line = text.split("\n")[0].strip()
                author_title = first_line[:120] if "|" in first_line else ""

                posts.append(LinkedInPost(
                    post_id=pid,
                    url=url,
                    text=text[:2000],
                    author=author,
                    author_title=author_title,
                    likes=int(hit.get("points") or 0),
                    posted_at=created,
                    query=queries[0],
                    source="hackernews",
                ))

                if len(posts) >= max_total:
                    break

            if page >= data.get("nbPages", 1) - 1:
                break
            page += 1

        logger.info(f"Thread '{thread['title']}' -> {len(posts)} matching posts so far")

    logger.info(f"Total HN hiring posts: {len(posts)}")
    return posts
