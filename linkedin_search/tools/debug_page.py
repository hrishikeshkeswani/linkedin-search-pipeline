"""
tools/debug_page.py
-------------------
Opens LinkedIn search in a visible browser, takes a screenshot,
and prints the page HTML so we can find the correct CSS selectors.

Usage:
    python tools/debug_page.py
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from playwright.async_api import async_playwright
from config.settings import settings

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome", headless=False)
        page = await browser.new_page()

        # Inject cookie
        await page.goto("https://www.linkedin.com", wait_until="domcontentloaded")
        await page.context.add_cookies([{
            "name": "li_at",
            "value": settings.LINKEDIN_SESSION_COOKIE,
            "domain": ".linkedin.com",
            "path": "/",
            "httpOnly": True,
            "secure": True,
        }])

        url = "https://www.linkedin.com/search/results/content/?keywords=machine%20learning%20engineer&sortBy=date_posted"
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(5)  # let JS fully render

        # Screenshot
        shot_path = Path("tools/debug_screenshot.png")
        await page.screenshot(path=str(shot_path), full_page=False)
        print(f"Screenshot saved -> {shot_path}")

        await asyncio.sleep(8)  # wait for full JS render

        # Dump full page HTML to file and search for post patterns
        content = await page.content()
        out = Path("tools/page_dump.html")
        out.write_text(content, encoding="utf-8")
        print(f"Full HTML saved -> tools/page_dump.html ({len(content)} chars)")

        # Find all unique data-* attributes on the page
        attrs = await page.evaluate("""() => {
            const all = document.querySelectorAll('*');
            const found = new Set();
            all.forEach(el => {
                [...el.attributes].forEach(a => {
                    if (a.name.startsWith('data-')) found.add(a.name + '=' + a.value.slice(0,60));
                });
            });
            return [...found].slice(0, 80);
        }""")
        print("\ndata-* attributes on page:")
        for a in attrs:
            print(f"  {a}")

        await browser.close()

asyncio.run(main())
