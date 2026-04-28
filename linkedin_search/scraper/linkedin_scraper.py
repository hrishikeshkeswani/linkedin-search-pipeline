"""
LinkedIn post scraper (Playwright-based).

Targets LinkedIn search results for job-related posts.
Uses a li_at session cookie for auth — no username/password stored.

Usage:
    python -m scraper.linkedin_scraper --queries "python engineer" "ML jobs" --max 200
"""

import asyncio
import json
import hashlib
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from playwright.async_api import async_playwright, Page, Browser
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Data model ────────────────────────────────────────────────────────────────

class LinkedInPost(BaseModel):
    post_id: str
    url: str
    text: str
    author: str = ""
    author_title: str = ""
    likes: int = 0
    comments: int = 0
    posted_at: str = ""          # ISO-8601 string
    scraped_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    source: str = "linkedin"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post_id(url: str, text: str) -> str:
    """Stable ID: hash of URL + first 200 chars of text."""
    key = f"{url}:{text[:200]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _parse_count(raw: str) -> int:
    """'1,234 likes' → 1234, '2.1K' → 2100."""
    raw = raw.strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*([km]?)", raw)
    if not m:
        return 0
    n = float(m.group(1))
    suffix = m.group(2)
    if suffix == "k":
        n *= 1_000
    elif suffix == "m":
        n *= 1_000_000
    return int(n)


# ── Core scraper ──────────────────────────────────────────────────────────────

class LinkedInScraper:
    SEARCH_URL = (
        "https://www.linkedin.com/search/results/content/"
        "?keywords={query}&sortBy=date_posted"
    )

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.SCRAPER_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()

    # ── Auth ──────────────────────────────────────────────────────────

    async def _inject_cookie(self, page: Page) -> None:
        """Inject li_at session cookie so we're logged in."""
        if not settings.LINKEDIN_SESSION_COOKIE:
            logger.warning(
                "No LINKEDIN_SESSION_COOKIE set — scraper will see public content only. "
                "Set li_at cookie in .env for authenticated access."
            )
            return
        await page.context.add_cookies([
            {
                "name": "li_at",
                "value": settings.LINKEDIN_SESSION_COOKIE,
                "domain": ".linkedin.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
            }
        ])
        logger.info("Session cookie injected.")

    # ── Page parsing ──────────────────────────────────────────────────

    async def _parse_feed_page(self, page: Page, query: str) -> list[LinkedInPost]:
        """
        Extract posts using LinkedIn's current SDUI DOM structure.
        Uses JS evaluation to parse all posts in one pass.
        """
        posts: list[LinkedInPost] = []

        # Wait for post text to appear
        try:
            await page.wait_for_selector(
                '[data-testid="expandable-text-box"], [data-testid="main-feed-activity-card"]',
                timeout=15_000,
            )
        except Exception:
            logger.warning("Feed container not found — possibly not logged in or rate-limited.")
            return posts

        # Parse all posts via JS in one pass
        raw_posts = await page.evaluate("""() => {
            const results = [];
            const textBoxes = document.querySelectorAll('[data-testid="expandable-text-box"]');

            textBoxes.forEach(box => {
                try {
                    const text = box.innerText.trim();
                    if (!text || text.length < 30) return;

                    // Walk up to find the post container (has the control menu button)
                    let container = box;
                    let author = '', profileUrl = '', headline = '';
                    for (let i = 0; i < 20; i++) {
                        container = container.parentElement;
                        if (!container) break;

                        // Author from control menu aria-label
                        if (!author) {
                            const btn = container.querySelector('[aria-label^="Open control menu for post by"]');
                            if (btn) {
                                const label = btn.getAttribute('aria-label') || '';
                                author = label.replace('Open control menu for post by ', '').trim();
                            }
                        }

                        // Profile URL
                        if (!profileUrl) {
                            const links = container.querySelectorAll('a[href*="/in/"]');
                            if (links.length > 0) {
                                profileUrl = links[0].href.split('?')[0];
                            }
                        }

                        // Headline — text node near profile link that isn't the author name
                        if (author && profileUrl && !headline) {
                            const spans = container.querySelectorAll('span, p');
                            for (const s of spans) {
                                const t = s.innerText.trim();
                                if (t && t !== author && t.length > 10 && t.length < 200 &&
                                    !t.includes('Follow') && !t.includes('Connect') &&
                                    !t.startsWith('http')) {
                                    headline = t;
                                    break;
                                }
                            }
                        }

                        if (author && profileUrl) break;
                    }

                    // Reactions count
                    let likes = 0;
                    const reactionEl = document.querySelector(
                        '[aria-label*="reaction"], [aria-label*="like"]'
                    );
                    if (reactionEl) {
                        const m = (reactionEl.getAttribute('aria-label') || '').match(/[\\d,]+/);
                        if (m) likes = parseInt(m[0].replace(',', ''));
                    }

                    results.push({ text, author, profileUrl, headline, likes });
                } catch(e) {}
            });

            return results;
        }""")

        logger.info(f"Found {len(raw_posts)} posts via JS for query '{query}'")

        for r in raw_posts:
            text = r.get("text", "")
            author = r.get("author", "")
            url = r.get("profileUrl", "")
            post_id = _post_id(url, text)
            if post_id in self._seen:
                continue
            self._seen.add(post_id)
            posts.append(LinkedInPost(
                post_id=post_id,
                url=url,
                text=text,
                author=author,
                author_title=r.get("headline", ""),
                likes=r.get("likes", 0),
                query=query,
                source="linkedin",
            ))

        return posts

    # ── Scroll + paginate ─────────────────────────────────────────────

    async def _scroll_and_collect(
        self, page: Page, query: str, max_posts: int
    ) -> AsyncIterator[LinkedInPost]:
        """Scroll feed, yielding posts as they appear."""
        collected = 0
        scroll_attempts = 0
        no_new_streak = 0          # stop if scroll yields nothing new N times
        max_scrolls = max_posts * 3 + 10

        while collected < max_posts and scroll_attempts < max_scrolls:
            prev_count = collected
            batch = await self._parse_feed_page(page, query)
            for post in batch:
                yield post
                collected += 1
                if collected >= max_posts:
                    return

            # Stop if scrolling produces no new posts 3 times in a row
            if collected == prev_count:
                no_new_streak += 1
                if no_new_streak >= 3:
                    logger.info("No new posts after 3 scrolls — stopping.")
                    return
            else:
                no_new_streak = 0

            # Scroll to bottom of page to trigger lazy loading
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(settings.SCRAPER_RATE_LIMIT_SECS)

            # Click "Show more results" if present
            for selector in [
                "button.scaffold-finite-scroll__load-button",
                "button[aria-label*='Load more']",
                "button[aria-label*='Show more']",
            ]:
                btn = await page.query_selector(selector)
                if btn:
                    await btn.click()
                    await asyncio.sleep(settings.SCRAPER_RATE_LIMIT_SECS)
                    break

            scroll_attempts += 1

    # ── Public API ────────────────────────────────────────────────────

    async def scrape_query(
        self,
        browser: Browser,
        query: str,
        max_posts: int | None = None,
    ) -> list[LinkedInPost]:
        max_posts = max_posts or settings.SCRAPER_MAX_POSTS
        page = await browser.new_page()

        try:
            # Navigate to linkedin.com first to establish domain context,
            # then inject cookie and reload — prevents ERR_TOO_MANY_REDIRECTS
            await page.goto("https://www.linkedin.com", wait_until="commit", timeout=60_000)
            await self._inject_cookie(page)
            await asyncio.sleep(2)

            url = self.SEARCH_URL.format(query=query.replace(" ", "%20"))
            logger.info(f"Navigating: {url}")
            await page.goto(url, wait_until="commit", timeout=60_000)
            await asyncio.sleep(5)  # let JS render fully

            posts: list[LinkedInPost] = []
            async for post in self._scroll_and_collect(page, query, max_posts):
                posts.append(post)

            logger.info(f"Scraped {len(posts)} posts for '{query}'")
            return posts

        finally:
            await page.close()

    async def run(
        self,
        queries: list[str],
        max_per_query: int | None = None,
        output_file: str | None = None,
    ) -> list[LinkedInPost]:
        all_posts: list[LinkedInPost] = []

        async with async_playwright() as pw:
            # Use real installed Chrome (not Playwright's Chromium) so LinkedIn
            # doesn't flag it as an unsupported/bot browser.
            browser = await pw.chromium.launch(
                channel="chrome",
                headless=settings.SCRAPER_HEADLESS,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                for query in queries:
                    posts = await self.scrape_query(browser, query, max_per_query)
                    all_posts.extend(posts)
                    await asyncio.sleep(settings.SCRAPER_RATE_LIMIT_SECS * 2)
            finally:
                await browser.close()

        # Persist raw output
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = output_file or f"raw_posts_{ts}.jsonl"
        out_path = self.output_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            for post in all_posts:
                f.write(post.model_dump_json() + "\n")

        logger.info(f"Saved {len(all_posts)} posts → {out_path}")
        return all_posts


# ── Fallback: mock scraper for testing without LinkedIn access ─────────────────

class MockScraper:
    """
    Generates realistic synthetic posts for local dev/testing.
    Run with: python -m scraper.linkedin_scraper --mock
    """

    TEMPLATES = [
        "Excited to share that I just landed a {role} role at {company}! "
        "The interview process focused heavily on {skill}. Happy to answer questions.",
        "3 things I wish I knew before my {role} interview at {company}: "
        "1) {skill} matters more than you think 2) System design is key 3) Always negotiate.",
        "Hot take: {skill} is the most underrated skill for {role} candidates in {year}. "
        "Here's why companies are prioritising it now…",
        "Just finished my first month as a {role} at {company}. "
        "Tech stack: {skill}. Culture: excellent. Hiring: YES — DM me.",
        "We're hiring {role}s at {company}! "
        "Must-have: {skill}. Remote-friendly. Competitive comp. Link in comments.",
    ]

    ROLES = ["ML Engineer", "Data Scientist", "Backend Engineer", "Python Developer",
             "AI Engineer", "MLOps Engineer", "LLM Engineer", "Data Engineer"]
    COMPANIES = ["Google", "Meta", "Anthropic", "OpenAI", "Stripe", "Databricks",
                 "Snowflake", "Hugging Face", "Scale AI", "Cohere"]
    SKILLS = ["Python", "LangChain", "PyTorch", "Kubernetes", "FAISS", "RAG pipelines",
              "distributed systems", "transformer fine-tuning", "vector databases", "FastAPI"]

    def generate(self, n: int = 100, query: str = "ML jobs") -> list[LinkedInPost]:
        import random
        random.seed(42)
        posts = []
        for i in range(n):
            tmpl = random.choice(self.TEMPLATES)
            text = tmpl.format(
                role=random.choice(self.ROLES),
                company=random.choice(self.COMPANIES),
                skill=random.choice(self.SKILLS),
                year=2024,
            )
            post_id = _post_id(f"mock_{i}", text)
            posts.append(LinkedInPost(
                post_id=post_id,
                url=f"https://www.linkedin.com/posts/mock-{i}",
                text=text,
                author=f"User {i}",
                author_title=random.choice(self.ROLES) + " @ " + random.choice(self.COMPANIES),
                likes=random.randint(10, 5000),
                posted_at=datetime.now(timezone.utc).isoformat(),
                query=query,
            ))
        return posts


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LinkedIn post scraper")
    parser.add_argument("--queries", nargs="+", default=["python engineer jobs", "ML engineer hiring"],
                        help="Search queries to scrape")
    parser.add_argument("--max", type=int, default=100, help="Max posts per query")
    parser.add_argument("--mock", action="store_true", help="Use mock data (no LinkedIn access needed)")
    parser.add_argument("--output", default=None, help="Output filename (default: timestamped)")
    args = parser.parse_args()

    if args.mock:
        scraper = MockScraper()
        output_dir = settings.SCRAPER_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        all_posts = []
        for q in args.queries:
            all_posts.extend(scraper.generate(n=args.max, query=q))
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = output_dir / (args.output or f"mock_posts_{ts}.jsonl")
        with open(out, "w") as f:
            for p in all_posts:
                f.write(p.model_dump_json() + "\n")
        print(f"Generated {len(all_posts)} mock posts → {out}")
    else:
        asyncio.run(
            LinkedInScraper().run(args.queries, max_per_query=args.max, output_file=args.output)
        )
