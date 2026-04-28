"""
tools/get_cookie.py
-------------------
Opens a visible browser, lets you log into LinkedIn manually,
then extracts the li_at cookie and saves it to .env automatically.

Usage:
    python tools/get_cookie.py
"""

import asyncio
import re
from pathlib import Path
from playwright.async_api import async_playwright

ENV_PATH = Path(__file__).parent.parent / ".env"


async def main():
    print("\n=== LinkedIn Cookie Extractor ===")
    print("A browser will open. Log into LinkedIn, then come back here.\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome", headless=False, slow_mo=50)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://www.linkedin.com/login")

        print("Waiting for you to log in... (watching for feed or jobs page)")

        # Wait until LinkedIn redirects away from /login — means user is logged in
        await page.wait_for_url(
            lambda url: "linkedin.com/feed" in url or "linkedin.com/jobs" in url or "linkedin.com/mynetwork" in url,
            timeout=120_000,   # 2 minutes to log in
        )

        print("Logged in! Extracting cookie...")
        await asyncio.sleep(2)  # let cookies settle

        cookies = await context.cookies("https://www.linkedin.com")
        li_at = next((c["value"] for c in cookies if c["name"] == "li_at"), None)

        await browser.close()

    if not li_at:
        print("ERROR: Could not find li_at cookie. Make sure you're fully logged in.")
        return

    print(f"Found li_at cookie ({len(li_at)} chars)")

    # Update .env file
    env_text = ENV_PATH.read_text(encoding="utf-8")
    if "LINKEDIN_SESSION_COOKIE=" in env_text:
        env_text = re.sub(
            r"LINKEDIN_SESSION_COOKIE=.*",
            f"LINKEDIN_SESSION_COOKIE={li_at}",
            env_text,
        )
    else:
        env_text += f"\nLINKEDIN_SESSION_COOKIE={li_at}\n"

    ENV_PATH.write_text(env_text, encoding="utf-8")
    print(f"Saved to {ENV_PATH}")
    print("\nDone! Now run:")
    print('  python run_pipeline.py --queries "machine learning engineer" "data scientist hiring"')


if __name__ == "__main__":
    asyncio.run(main())
