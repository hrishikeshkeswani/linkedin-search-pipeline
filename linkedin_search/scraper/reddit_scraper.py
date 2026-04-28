"""
scraper/reddit_scraper.py
-------------------------
Scrapes hiring posts from ML/data science subreddits.
People write these exactly like LinkedIn hiring posts.

Subreddits targeted:
  r/MachineLearning   — "ML News" flair has company hiring posts
  r/datascience       — weekly hiring threads
  r/cscareerquestions — hiring/job posts
  r/forhire           — direct hiring posts [HIRING] tagged
  r/LLMDevs           — LLM/AI hiring
  r/artificial        — AI hiring posts

No API key needed — uses Reddit's public JSON API.
"""

import hashlib
import logging
import re
import time
import urllib.request
import json
from datetime import datetime, timezone

from scraper.linkedin_scraper import LinkedInPost

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "linkedin-search-pipeline/1.0 (job search research tool)"
}

SUBREDDITS = [
    "MachineLearning",
    "datascience",
    "forhire",
    "cscareerquestions",
    "LLMDevs",
    "artificial",
    "LanguageTechnology",
]


def _fetch(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Reddit fetch failed {url}: {e}")
        return None


def _post_id(reddit_id: str) -> str:
    return hashlib.md5(f"reddit:{reddit_id}".encode()).hexdigest()[:16]


def _is_hiring_post(title: str, text: str) -> bool:
    combined = (title + " " + text).lower()
    hiring_signals = [
        "[hiring]", "we're hiring", "we are hiring", "now hiring",
        "looking for", "open position", "job opening", "open role",
        "join our team", "join us", "we need a", "seeking a",
        "hiring for", "open to hiring", "accepting applications",
    ]
    return any(s in combined for s in hiring_signals)


def _to_post(post: dict, query: str) -> LinkedInPost:
    data = post.get("data", {})
    title = data.get("title", "")
    selftext = data.get("selftext", "") or ""
    author = data.get("author", "")
    subreddit = data.get("subreddit", "")
    score = data.get("score", 0)
    created = data.get("created_utc", 0)
    permalink = data.get("permalink", "")

    text = f"{title}\n\n{selftext}".strip()
    posted_at = datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else ""
    url = f"https://www.reddit.com{permalink}" if permalink else ""

    return LinkedInPost(
        post_id=_post_id(data.get("id", title)),
        url=url,
        text=text[:2000],
        author=f"u/{author}",
        author_title=f"r/{subreddit}",
        likes=score,
        posted_at=posted_at,
        query=query,
        source="reddit",
    )


def _search_subreddit(subreddit: str, query: str, limit: int = 25) -> list[dict]:
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={urllib.parse.quote(query + ' hiring')}&restrict_sr=1&sort=new&limit={limit}&t=month"
    )
    data = _fetch(url)
    if not data:
        return []
    return data.get("data", {}).get("children", [])


def _fetch_hiring_flair(subreddit: str, limit: int = 50) -> list[dict]:
    """Fetch posts with hiring-related flairs or [HIRING] tags."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    data = _fetch(url)
    if not data:
        return []
    posts = data.get("data", {}).get("children", [])
    return [
        p for p in posts
        if _is_hiring_post(
            p.get("data", {}).get("title", ""),
            p.get("data", {}).get("selftext", "") or ""
        )
    ]


def fetch_reddit_hiring(
    queries: list[str],
    max_total: int = 500,
) -> list[LinkedInPost]:
    import urllib.parse

    posts: list[LinkedInPost] = []
    seen: set[str] = set()

    for subreddit in SUBREDDITS:
        # 1. Search by query + "hiring"
        for query in queries:
            results = _search_subreddit(subreddit, query, limit=25)
            for p in results:
                post = _to_post(p, query)
                data = p.get("data", {})
                title = data.get("title", "")
                text = data.get("selftext", "") or ""
                if post.post_id not in seen and (
                    _is_hiring_post(title, text) or len(text) > 100
                ):
                    seen.add(post.post_id)
                    posts.append(post)
            time.sleep(1)  # be polite to Reddit

        # 2. Browse new posts in each subreddit for hiring signals
        hiring_posts = _fetch_hiring_flair(subreddit, limit=50)
        for p in hiring_posts:
            post = _to_post(p, queries[0])
            if post.post_id not in seen:
                seen.add(post.post_id)
                posts.append(post)
        time.sleep(1)

        logger.info(f"r/{subreddit} -> {len(posts)} total posts so far")
        if len(posts) >= max_total:
            break

    logger.info(f"Total Reddit hiring posts: {len(posts)}")
    return posts[:max_total]
