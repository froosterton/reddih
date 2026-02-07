"""
Reddit Monitor for r/RobloxTrading

Continuously watches for new posts with images, scans them for
Roblox limited items, and sends Discord alerts for hits.

Usage:
    python monitor.py                               (run continuously)
    python monitor.py --test                         (also sends skip notices to Discord)
    python monitor.py --once                         (check once and exit, don't loop)
    python monitor.py --scan-last 3 --test --once    (scan the last 3 posts and exit)
"""

import os
import sys
import time
import praw
import requests
from main import (
    fetch_item_database,
    build_lookup_tables,
    process_image,
    screen_text_post,
    send_discord_text_lead,
    DISCORD_WEBHOOK_URL,
)

# ──────────────────────────────────────────────
# REDDIT CONFIGURATION (env vars; fallback for local run)
# ──────────────────────────────────────────────
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "bcHPImj3ngGQlY6A2OULag")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "YyEJHOM7C5RircOXPFM8kZfa3WCoYQ")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "VisionScanner/1.0 by smg110")

SUBREDDIT_NAMES = ["RobloxTrading", "crosstradingroblox", "RobloxLimiteds"]
POLL_INTERVAL = 45          # seconds between checks
MAX_POSTS_PER_CHECK = 15    # how many "new" posts to look at per subreddit each cycle
ROLIMONS_REFRESH_MINS = 30  # re-fetch Rolimons data every N minutes

# Image file extensions and domains we care about
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
IMAGE_DOMAINS = ("i.redd.it", "i.imgur.com", "preview.redd.it")

# Flairs that strongly indicate trade/sell intent (exact match, lowercased)
TRADE_FLAIRS = {
    "trade ad", "trade ads",
    "trading help",
    "w/l", "wfl",
}

# If a title contains ANY of these, skip the post immediately — not what we want
EXCLUDE_KEYWORDS = [
    "scammer", "scam alert", "scam", "scammed",
    "beware", "warning", "banned", "report",
    "giveaway", "giving away", "free",
    "meme", "funny", "lol", "lmao",
    "rant", "vent",
]

# Keywords that suggest a text post might be from a potential seller / returning player
# These posts go through Gemini screening for confirmation (no false positives)
TEXT_LEAD_KEYWORDS = [
    "haven't played", "havent played",
    "haven't been on", "havent been on",
    "old account", "my old", "years ago",
    "came back", "got back", "just got back", "returning",
    "is this rare", "are these rare", "is this worth",
    "are my items worth", "what are my items",
    "how do i sell", "where do i sell", "where can i sell",
    "sell my items", "sell my account", "sell limiteds",
    "worth any money", "worth anything",
    "items worth", "account worth",
    "sell limited", "sell expensive",
    "how much are my", "how much is my",
    "cash out", "cashout",
    "quit roblox", "quitting roblox", "leaving roblox",
]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def is_excluded(post) -> bool:
    """Check if a post is obvious noise (scam report, meme, giveaway, etc.)."""
    title_lower = post.title.strip().lower()
    flair = (post.link_flair_text or "").strip().lower()

    for exclude in EXCLUDE_KEYWORDS:
        if exclude in title_lower or exclude in flair:
            return True
    return False

def is_potential_text_lead(post) -> bool:
    """Check if a text-only post might be from a potential seller or returning player.

    This is a keyword pre-filter. Posts that pass this go to Gemini for strict screening.
    """
    title_lower = post.title.strip().lower()
    body_lower = (post.selftext or "").strip().lower()
    flair = (post.link_flair_text or "").strip().lower()

    # Hard exclude first
    for exclude in EXCLUDE_KEYWORDS:
        if exclude in title_lower or exclude in flair:
            return False

    # Check title + body for text lead keywords
    combined = title_lower + " " + body_lower
    for keyword in TEXT_LEAD_KEYWORDS:
        if keyword in combined:
            return True

    # Also check trade flairs for text posts (e.g. "Trading Help" flair with no image)
    if flair in TRADE_FLAIRS:
        return True

    return False


def get_image_urls_from_post(post) -> list[str]:
    """Extract ALL image URLs from a Reddit post.

    Checks every possible source in order:
      1. Gallery (multiple images via media_metadata)
      2. Direct image link (post.url ends with image extension or is on known domain)
      3. Preview images (post.preview.images[].source.url)
      4. Non-gallery media_metadata (single embedded image)
      5. Body text scan (regex for i.redd.it / imgur / preview.redd.it links)
    """
    urls = []
    url = post.url

    # 1. Reddit gallery (multiple images)
    if hasattr(post, "is_gallery") and post.is_gallery:
        try:
            media = post.media_metadata
            for item in media.values():
                if item.get("status") == "valid" and "s" in item:
                    img_url = item["s"].get("u", "")
                    if img_url:
                        urls.append(img_url.replace("&amp;", "&"))
        except Exception:
            pass
        if urls:
            return urls

    # 2. Direct image link (post.url is the image itself)
    if any(url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        return [url]
    if any(domain in url for domain in IMAGE_DOMAINS):
        return [url]

    # 3. Preview images (Reddit generates these for image posts and link posts)
    try:
        preview = getattr(post, "preview", None)
        if preview:
            images = preview.get("images", [])
            for img in images:
                source = img.get("source")
                if source and source.get("url"):
                    img_url = source["url"].replace("&amp;", "&")
                    urls.append(img_url)
            if urls:
                return urls
    except Exception:
        pass

    # 4. Non-gallery media_metadata (single embedded image)
    try:
        media = getattr(post, "media_metadata", None)
        if media:
            for item in media.values():
                if isinstance(item, dict) and item.get("status") == "valid":
                    img_url = ""
                    if "s" in item:
                        img_url = item["s"].get("u", "") or item["s"].get("url", "")
                    if img_url:
                        urls.append(img_url.replace("&amp;", "&"))
            if urls:
                return urls
    except Exception:
        pass

    # 5. Scan selftext body for image URLs (i.redd.it, imgur, preview.redd.it)
    body = getattr(post, "selftext", "") or ""
    if body:
        import re
        img_pattern = re.findall(
            r'https?://(?:i\.redd\.it|i\.imgur\.com|preview\.redd\.it)/[^\s\)\]>"]+',
            body,
        )
        for found_url in img_pattern:
            clean = found_url.replace("&amp;", "&")
            if clean not in urls:
                urls.append(clean)
        if urls:
            return urls

    return []


def send_startup_notice(subreddit_names: list[str]) -> None:
    """Send a Discord embed to confirm the monitor is running."""
    subs = ", ".join(f"**r/{s}**" for s in subreddit_names)
    embed = {
        "title": "Monitor Started",
        "color": 0x00CC00,
        "description": (
            f"Now watching {subs} for new posts.\n"
            f"Polling every **{POLL_INTERVAL}s**."
        ),
    }
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            timeout=10,
        )
    except Exception:
        pass


# ──────────────────────────────────────────────
# MAIN MONITOR LOOP
# ──────────────────────────────────────────────

def _process_post(post, name_lookup, acronym_lookup, seen_post_ids, testing, sub_name):
    """Process a single Reddit post. Returns (result, reason) where result is 'hit', 'skip', or 'lead'."""
    seen_post_ids.add(post.id)
    post_link = f"https://reddit.com{post.permalink}"
    image_urls = get_image_urls_from_post(post)

    # ── Any post with images (Gemini pre-screen is the real filter) ──
    if image_urls:
        print(f"  Image post — {len(image_urls)} image(s). Sending to Gemini...")
        for idx, img_url in enumerate(image_urls):
            print(f"  Image {idx + 1}/{len(image_urls)}: {img_url}")
            try:
                found = process_image(
                    img_url, name_lookup, acronym_lookup,
                    testing=testing,
                    post_title=post.title,
                    post_url=post_link,
                )
                if found:
                    return ("hit", "")
            except Exception as e:
                print(f"  Error: {e}")
        return ("skip", "scanned image(s) but no Rolimons item at or above 100k value")

    # ── Text-only post: potential seller / returning player ──
    if is_potential_text_lead(post):
        print(f"  Potential text lead. Screening...")
        body = (post.selftext or "").strip()
        is_lead, reason, matched_items = screen_text_post(
            post.title, body, name_lookup, acronym_lookup,
        )
        if is_lead:
            print(f"  LEAD confirmed: {reason}")
            send_discord_text_lead(post.title, post_link, body, reason, matched_items)
            return ("lead", "")
        else:
            print(f"  Not a lead: {reason}")
            return ("skip", reason)

    return ("skip", "")


def run_monitor(testing: bool = False, once: bool = False, scan_last: int = 0):
    # ── Connect to Reddit (read-only, no password needed) ──
    print("Connecting to Reddit...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    subs_str = ", ".join(f"r/{s}" for s in SUBREDDIT_NAMES)
    print(f"  Connected. Monitoring: {subs_str}")

    # ── Load Rolimons database ──
    items_db = fetch_item_database()
    name_lookup, acronym_lookup = build_lookup_tables(items_db)
    last_rolimons_refresh = time.time()

    # ── Track seen posts ──
    seen_post_ids = set()

    # --scan-last N: process the last N posts from each subreddit
    if scan_last > 0:
        total_hits = 0
        total_skips = 0
        total_posts = 0

        for sub_name in SUBREDDIT_NAMES:
            subreddit = reddit.subreddit(sub_name)
            print(f"\n{'='*50}")
            print(f"  Scanning last {scan_last} post(s) from r/{sub_name}")
            print(f"{'='*50}")

            for post in subreddit.new(limit=scan_last):
                if post.id in seen_post_ids:
                    continue
                total_posts += 1
                flair = (post.link_flair_text or "none").strip()
                print(f"\n  Post:  \"{post.title}\"")
                print(f"  Flair: {flair}")
                print(f"  Link:  https://reddit.com{post.permalink}")

                # Quick filter — skip noise; image posts always go through
                scan_images = get_image_urls_from_post(post)
                if is_excluded(post):
                    print(f"  Excluded (noise). Skipping.")
                    seen_post_ids.add(post.id)
                    total_skips += 1
                    continue
                if not scan_images and not is_potential_text_lead(post):
                    print(f"  Text-only, no lead keywords. Skipping.")
                    seen_post_ids.add(post.id)
                    total_skips += 1
                    continue

                result, reason = _process_post(post, name_lookup, acronym_lookup, seen_post_ids, testing, sub_name)
                if result in ("hit", "lead"):
                    total_hits += 1
                else:
                    total_skips += 1
                    if reason:
                        print(f"  No alert: {reason}")

        print(f"\n{'='*50}")
        print(f"Scan complete. {total_hits} hit(s), {total_skips} skip(s) out of {total_posts} post(s).")
        print(f"{'='*50}")
        if once:
            return
    else:
        # Normal startup: seed with existing posts so we don't re-process
        print("Seeding with existing posts...")
        for sub_name in SUBREDDIT_NAMES:
            subreddit = reddit.subreddit(sub_name)
            count = 0
            for post in subreddit.new(limit=MAX_POSTS_PER_CHECK):
                seen_post_ids.add(post.id)
                count += 1
            print(f"  r/{sub_name}: seeded {count} post(s)")
        print(f"  Total: {len(seen_post_ids)} post(s). Will only process NEW posts from now on.")

    send_startup_notice(SUBREDDIT_NAMES)

    print(f"\nMonitor is live. Polling every {POLL_INTERVAL}s. Press Ctrl+C to stop.\n")

    # ── Poll loop ──
    while True:
        try:
            # Refresh Rolimons data periodically
            if time.time() - last_rolimons_refresh > ROLIMONS_REFRESH_MINS * 60:
                print("Refreshing Rolimons data...")
                try:
                    items_db = fetch_item_database()
                    name_lookup, acronym_lookup = build_lookup_tables(items_db)
                    last_rolimons_refresh = time.time()
                except Exception as e:
                    print(f"  Warning: Rolimons refresh failed ({e}), using cached data.")

            # Fetch latest posts from all subreddits
            new_count = 0
            hit_count = 0

            for sub_name in SUBREDDIT_NAMES:
                subreddit = reddit.subreddit(sub_name)

                for post in subreddit.new(limit=MAX_POSTS_PER_CHECK):
                    if post.id in seen_post_ids:
                        continue

                    new_count += 1
                    post_link = f"https://reddit.com{post.permalink}"
                    flair = (post.link_flair_text or "none").strip()

                    # Quick filter — skip obvious noise; let everything else through
                    image_urls = get_image_urls_from_post(post)
                    if is_excluded(post):
                        seen_post_ids.add(post.id)
                        continue
                    # Text-only posts still need keyword match; image posts always go through
                    if not image_urls and not is_potential_text_lead(post):
                        seen_post_ids.add(post.id)
                        continue

                    print(f"\n[r/{sub_name}] \"{post.title}\" [{flair}]")
                    print(f"  Link: {post_link}")

                    result, reason = _process_post(post, name_lookup, acronym_lookup, seen_post_ids, testing, sub_name)
                    if result in ("hit", "lead"):
                        hit_count += 1
                    elif reason:
                        print(f"  No alert: {reason}")

            if new_count > 0:
                print(f"\n[{time.strftime('%H:%M:%S')}] Checked {new_count} new post(s) across {len(SUBREDDIT_NAMES)} subs, {hit_count} hit(s).")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No new posts.", end="\r")

        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user.")
            break
        except Exception as e:
            print(f"\nError during poll: {e}")
            print("Retrying in 30s...")
            time.sleep(30)
            continue

        if once:
            print("\n--once flag set. Exiting after single check.")
            break

        # Wait before next poll
        time.sleep(POLL_INTERVAL)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    testing = "--test" in sys.argv
    once = "--once" in sys.argv

    # Parse --scan-last N
    scan_last = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--scan-last" and i + 1 < len(sys.argv):
            try:
                scan_last = int(sys.argv[i + 1])
            except ValueError:
                pass

    print("=" * 50)
    print("  Roblox Trading Reddit Monitor")
    print("=" * 50)

    run_monitor(testing=testing, once=once, scan_last=scan_last)
