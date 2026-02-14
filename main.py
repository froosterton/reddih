import os
import sys
import json
import re
import requests
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# CONFIGURATION (env only — set in .env or host)
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
ROLIMONS_API_URL = os.environ.get("ROLIMONS_API_URL", "https://www.rolimons.com/itemapi/itemdetails")
MIN_VALUE_THRESHOLD = int(os.environ.get("MIN_VALUE_THRESHOLD", "100000"))


# ──────────────────────────────────────────────
# ROLIMONS ITEM DATABASE
# ──────────────────────────────────────────────

def fetch_item_database():
    """Fetch all Roblox limited items from the Rolimons API."""
    print("Fetching Rolimons item database...")
    resp = requests.get(
        ROLIMONS_API_URL,
        headers={"User-Agent": "VisionScanner/1.0"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if not data.get("success"):
        raise Exception("Rolimons API returned an error")

    print(f"  Loaded {data['item_count']} items.")
    return data["items"]


def build_lookup_tables(items_db: dict) -> tuple:
    """Pre-build fast lookup tables from the Rolimons database."""
    name_lookup = {}
    acronym_lookup = {}

    for item_id, item_data in items_db.items():
        norm = normalize_name(item_data[0])
        name_lookup[norm] = (item_id, item_data)

        acr = item_data[1].strip().lower()
        if acr:
            acronym_lookup[acr] = (item_id, item_data)

    return name_lookup, acronym_lookup


def normalize_name(name: str) -> str:
    """Normalize an item name for flexible comparison."""
    s = name.lower().strip()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("'s", "s").replace("'s", "s")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ──────────────────────────────────────────────
# GEMINI VISION — HELPERS
# ──────────────────────────────────────────────

def _download_image(image_url: str) -> types.Part:
    """Download an image URL and return a Gemini Part."""
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    mime_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    return types.Part.from_bytes(data=resp.content, mime_type=mime_type)


def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


# ──────────────────────────────────────────────
# STEP 1: PRE-SCREEN — does this image reference
#         a Roblox limited item at all?
# ──────────────────────────────────────────────

def prescreen_image(image_part: types.Part) -> bool:
    """Ask Gemini a simple yes/no: does this image reference a Roblox limited item?"""
    client = _get_gemini_client()

    prompt = (
        "Look at this image carefully.\n"
        "Is this image referencing a Roblox limited item? "
        "Roblox limited items are special virtual accessories/gear that can be traded "
        "between players (hats, faces, gear, etc.).\n\n"
        "Signs that an image references a limited item:\n"
        "- A Roblox trade window showing items\n"
        "- An inventory showing items with RAP/value numbers\n"
        "- Text mentioning specific Roblox limited item names or acronyms\n"
        "- A Roblox avatar wearing recognizable limited items\n"
        "- A Rolimons page or similar value-checking site\n\n"
        "Answer with ONLY the word: yes or no"
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image_part],
    )

    answer = response.text.strip().lower()
    return answer.startswith("yes")


# ──────────────────────────────────────────────
# STEP 2: EXTRACT — identify every item in the image
# ──────────────────────────────────────────────

def extract_items_from_image(image_part: types.Part) -> str:
    """Use Gemini Vision to extract Roblox item details from an image."""
    client = _get_gemini_client()

    prompt = (
        "This image is from a Reddit post about Roblox limited items.\n"
        "Your job is to identify EVERY Roblox limited item name mentioned or shown "
        "anywhere in this image.\n\n"
        "The image could be ANY of these formats:\n"
        "- A Roblox trade window showing items on both sides\n"
        "- An inventory or catalog screenshot\n"
        "- A Rolimons value change notification (item name as title, old/new values)\n"
        "- A Rolimons item page or chart\n"
        "- A text post or meme mentioning item names\n"
        "- An avatar wearing limited items\n"
        "- A screenshot of any Roblox-related site or app\n\n"
        "For each item, extract:\n"
        '- "name": the full item name exactly as displayed in the image\n'
        '- "value": the highest numerical value shown for that item '
        "(could be labeled as value, RAP, new value, price, etc). "
        "Use 0 if no value is visible.\n\n"
        "Return ONLY a valid JSON array of objects.\n"
        "Examples:\n"
        '  [{"name": "Domino Crown", "value": 24000000}]\n'
        '  [{"name": "Hooded Firelord", "value": 4200000}]\n'
        '  [{"name": "Bighead", "value": 5000}, {"name": "Goldrow", "value": 316}]\n\n'
        "Important:\n"
        "- Read the EXACT item names from the image text, do not guess.\n"
        "- If a value is shown with commas (like 4,200,000), return it as a number (4200000).\n"
        "- Look EVERYWHERE in the image for item names — titles, labels, text, etc.\n"
        "- Even if only ONE item is shown, return it in the array.\n"
        "- If you truly cannot find any Roblox item names, return: []"
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image_part],
    )

    return response.text


# ──────────────────────────────────────────────
# PARSING AND MATCHING
# ──────────────────────────────────────────────

def parse_gemini_response(raw_text: str) -> list:
    """Parse the JSON array of item objects from Gemini's response."""
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        print(f"  Warning: Could not parse Gemini response as JSON.")
        print(f"  Raw response: {raw_text[:500]}")
        return []

    if not isinstance(result, list):
        return []

    items = []
    for entry in result:
        if isinstance(entry, str):
            items.append({"name": entry.strip(), "value": 0})
        elif isinstance(entry, dict) and "name" in entry:
            val = entry.get("value", 0)
            if isinstance(val, str):
                val = int(re.sub(r"[^0-9]", "", val) or "0")
            items.append({"name": entry["name"].strip(), "value": int(val or 0)})

    return items


def match_single_item(
    detected_name: str,
    name_lookup: dict,
    acronym_lookup: dict,
) -> tuple | None:
    """Try to match a single detected name against the database.

    Matching order:
      1. Exact normalised name match
      2. Exact acronym match
      3. Prefix match — if Gemini returned a truncated name (e.g. "Dominus Formidulos..."
         from a cut-off label), match it against the start of Rolimons names.
         Only matches if the prefix is 3+ words to avoid false positives.

    Returns (item_id, item_data) or None.
    """
    det_lower = detected_name.strip().lower()
    det_norm = normalize_name(detected_name)

    # 1. Exact normalised name
    if det_norm in name_lookup:
        return name_lookup[det_norm]

    # 2. Exact acronym
    if det_lower in acronym_lookup:
        return acronym_lookup[det_lower]

    # 3. Prefix match (handles truncated names like "Dominus Formidulos...")
    #    Require at least 2 words so single common words don't false-match
    if len(det_norm.split()) >= 2 and len(det_norm) >= 8:
        best_match = None
        best_len = 0
        for db_norm, (item_id, item_data) in name_lookup.items():
            if db_norm.startswith(det_norm) and len(db_norm) > best_len:
                best_match = (item_id, item_data)
                best_len = len(db_norm)
        if best_match:
            return best_match

    return None


def match_items_rolimons_only(
    detected_items: list,
    name_lookup: dict,
    acronym_lookup: dict,
) -> list:
    """Match detected items against the Rolimons database.

    ONLY items that match a Rolimons entry are returned.
    If nothing matches, the image is considered irrelevant.
    """
    results = []
    seen_ids = set()

    for det in detected_items:
        det_name = det["name"]

        match = match_single_item(det_name, name_lookup, acronym_lookup)

        if match and match[0] not in seen_ids:
            item_id, item_data = match
            rolimons_value = item_data[3] if item_data[3] != -1 else item_data[2]
            if rolimons_value < MIN_VALUE_THRESHOLD:
                continue  # skip items below the value threshold

            results.append({
                "id": item_id,
                "name": item_data[0],
                "acronym": item_data[1],
                "value": rolimons_value,
                "detected_as": det_name,
            })
            seen_ids.add(item_id)

    results.sort(key=lambda x: x["value"], reverse=True)
    return results


# ──────────────────────────────────────────────
# ROBLOX THUMBNAIL
# ──────────────────────────────────────────────

def get_item_thumbnail(item_id: str | None) -> str:
    """Fetch the Roblox CDN thumbnail URL for an item."""
    if not item_id:
        return ""
    url = (
        f"https://thumbnails.roblox.com/v1/assets"
        f"?assetIds={item_id}"
        f"&returnPolicy=PlaceHolder&size=420x420&format=Png&isCircular=false"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("data") and len(data["data"]) > 0:
            return data["data"][0].get("imageUrl", "")
    except Exception:
        pass
    return ""


# ──────────────────────────────────────────────
# DISCORD EMBED
# ──────────────────────────────────────────────

def send_discord_embed(
    item: dict,
    source_url: str = None,
    post_title: str = None,
    post_url: str = None,
) -> None:
    """Send a Discord embed for a detected high-value Roblox limited item."""
    thumbnail_url = get_item_thumbnail(item["id"])
    value_str = f"R$ {item['value']:,}"

    embed = {
        "title": "Possible Hit on Reddit",
        "color": 0xFF4500,
        "fields": [
            {"name": "Item Name", "value": item["name"], "inline": True},
            {"name": "Value (Rolimons)", "value": value_str, "inline": True},
        ],
    }

    if item["acronym"]:
        embed["fields"].append(
            {"name": "Acronym", "value": item["acronym"], "inline": True}
        )

    if item["detected_as"] != item["name"]:
        embed["fields"].append(
            {"name": "Detected As", "value": item["detected_as"], "inline": True}
        )

    if thumbnail_url:
        embed["thumbnail"] = {"url": thumbnail_url}

    # Reddit post info
    if post_title and post_url:
        embed["fields"].append(
            {"name": "Reddit Post", "value": f"[{post_title}]({post_url})", "inline": False}
        )
    elif source_url:
        embed["fields"].append(
            {"name": "Source", "value": f"[View Image]({source_url})", "inline": False}
        )

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"embeds": [embed]},
        timeout=10,
    )

    if resp.status_code == 204:
        print("    Discord embed sent successfully!")
    else:
        print(f"    Discord error ({resp.status_code}): {resp.text}")


def send_discord_text_lead(
    post_title: str,
    post_url: str,
    post_body: str,
    verdict: str,
    matched_items: list[dict] = None,
) -> None:
    """Send a Discord embed for a text-only post that looks like a potential seller."""
    body_preview = post_body[:400] + "..." if len(post_body) > 400 else post_body

    embed = {
        "title": "Potential Seller / Lead on Reddit",
        "color": 0x5865F2,
        "fields": [
            {"name": "Post", "value": f"[{post_title}]({post_url})", "inline": False},
            {"name": "Preview", "value": body_preview or "(no text)", "inline": False},
            {"name": "Why", "value": verdict, "inline": False},
        ],
    }

    # If specific items were matched, show the highest value one
    if matched_items:
        best = matched_items[0]
        embed["fields"].insert(1, {
            "name": "Top Item Mentioned",
            "value": f"{best['name']} — R$ {best['value']:,}",
            "inline": True,
        })
        thumbnail_url = get_item_thumbnail(best["id"])
        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"embeds": [embed]},
        timeout=10,
    )

    if resp.status_code == 204:
        print("    Discord text-lead embed sent!")
    else:
        print(f"    Discord error ({resp.status_code}): {resp.text}")


# ──────────────────────────────────────────────
# GEMINI TEXT POST SCREENING
# ──────────────────────────────────────────────

def find_mentioned_items(
    text: str,
    name_lookup: dict,
    acronym_lookup: dict,
) -> tuple[list[dict], list[dict]]:
    """Scan text for exact Rolimons item names or acronyms.

    Returns (above_threshold, below_threshold) — two lists of matched items.
    """
    text_lower = text.lower()
    text_norm = normalize_name(text)
    above = []
    below = []
    seen_ids = set()

    # Check every item name (normalized) against the normalized text
    for norm_name, (item_id, item_data) in name_lookup.items():
        if item_id in seen_ids:
            continue
        # Only match names with 2+ words to avoid false positives on short words
        if len(norm_name.split()) < 2:
            continue
        if norm_name in text_norm:
            value = item_data[3] if item_data[3] != -1 else item_data[2]
            item = {
                "id": item_id,
                "name": item_data[0],
                "acronym": item_data[1],
                "value": value,
            }
            if value >= MIN_VALUE_THRESHOLD:
                above.append(item)
            elif value > 0:
                below.append(item)
            seen_ids.add(item_id)

    # Common short words/slang that collide with Rolimons acronyms — never match these
    ACRONYM_BLACKLIST = {
        "mm",   # middleman
        "dc",   # disconnect
        "w",    # win
        "l",    # loss
        "f",    # fair
        "op",   # original poster / overpowered
        "pc",   # price check
        "nvm",  # nevermind
        "pm",   # private message
        "dm",   # direct message
        "rn",   # right now
        "gg",   # good game
        "bb",   # baby / bye bye
        "gl",   # good luck
        "ty",   # thank you
        "np",   # no problem
        "lf",   # looking for
        "ft",   # for trade
        "nft",  # not for trade
        "id",   # identification
        "da",   # the
        "pf",   # profile
        "fb",   # facebook
        "sc",   # snapchat
        "rt",   # retweet
        "ep",   # episode
        "hb",   # hurry back
        "sb",   # somebody
        "cs",   # customer service
        "ci",   # see i / confidential informant
        "aa",   # alcoholics anonymous
        "bt",   # bluetooth
        "dh",   # dear husband
        "rs",   # runescape
        "gw",   # guild wars
        "ac",   # animal crossing
        "iv",   # four
        "es",   # spanish
        "ss",   # screenshot
        "bm",   # bad manners
        "se",   # special edition
        "tv",   # television
    }

    # Check acronyms — must be 3+ chars, or a non-blacklisted standalone word
    words_in_text = set(text_lower.split())
    for acr, (item_id, item_data) in acronym_lookup.items():
        if item_id in seen_ids:
            continue
        if len(acr) < 2:
            continue
        if acr in ACRONYM_BLACKLIST:
            continue
        # For short acronyms (2-3 chars), require them to be uppercase in original text
        # to reduce false positives on common words
        if len(acr) <= 3:
            # Check if the acronym appears as an uppercase word in the original text
            original_words = set(text.split())
            if acr.upper() not in original_words:
                continue
        if acr in words_in_text:
            value = item_data[3] if item_data[3] != -1 else item_data[2]
            item = {
                "id": item_id,
                "name": item_data[0],
                "acronym": item_data[1],
                "value": value,
            }
            if value >= MIN_VALUE_THRESHOLD:
                above.append(item)
            elif value > 0:
                below.append(item)
            seen_ids.add(item_id)

    above.sort(key=lambda x: x["value"], reverse=True)
    below.sort(key=lambda x: x["value"], reverse=True)
    return above, below


def screen_text_post(
    title: str,
    body: str,
    name_lookup: dict,
    acronym_lookup: dict,
) -> tuple[bool, str, list[dict]]:
    """Screen a text post for potential limited item sellers.

    Two-pass approach:
      1. Code-level scan: check title+body for exact Rolimons names/acronyms.
         If a high-value item is found, it's an automatic lead.
      2. Gemini screening: only for posts that don't mention specific items
         but sound like a returning player asking about their account value.

    Returns (is_lead, reason, matched_items).
    """
    combined = f"{title} {body}".strip()

    # ── Pass 1: exact item name/acronym scan ──
    above, below = find_mentioned_items(combined, name_lookup, acronym_lookup)

    if above:
        names = ", ".join(m["name"] for m in above)
        return True, f"Mentions Rolimons-listed item(s): {names}", above

    if below:
        names = ", ".join(f"{m['name']} (R$ {m['value']:,})" for m in below)
        return False, f"Item(s) found but below R$ {MIN_VALUE_THRESHOLD:,} threshold: {names}", []

    # ── Pass 2: Gemini screening for generic "returning player" / "sell my account" posts ──
    client = _get_gemini_client()
    combined_text = f"Title: {title}\n\nBody: {body}" if body else f"Title: {title}"

    prompt = (
        "You are a strict filter for a Roblox LIMITED ITEM trading monitor.\n\n"
        "Roblox 'limited items' are special collectible avatar accessories "
        "(hats, faces, gear, hair) with finite supply that are traded between players.\n\n"
        "Read this Reddit post and determine if the author is:\n"
        "1. A returning player who owns Roblox limited items and wants to know their value\n"
        "2. Asking how to sell their Roblox limited items or limited-holding account\n"
        "3. Asking what their Roblox account with limiteds is worth\n"
        "4. Trading or offering specific Roblox limited items\n\n"
        "The person must clearly indicate they OWN limited items (or an old account "
        "that likely has limiteds) and want to sell, value, or trade them.\n\n"
        "Answer NO if:\n"
        "- They are selling Adopt Me, Blox Fruits, Murder Mystery, Royale High, "
        "or any other in-game items (NOT limiteds)\n"
        "- They are selling Robux or gift cards\n"
        "- They are buying, not selling\n"
        "- The post is a meme, joke, scam report, or rant\n"
        "- It's unclear or vague what they are selling\n\n"
        "Be EXTREMELY strict. When in doubt, say NO.\n\n"
        f"--- POST ---\n{combined_text}\n--- END ---\n\n"
        "Respond in EXACTLY this format (three lines only):\n"
        "VERDICT: yes or no\n"
        "REASON: one sentence explaining why\n"
        "ITEMS: comma-separated list of the specific Roblox limited item names "
        "mentioned in the post (use the full official item name if you know it, "
        "otherwise use exactly what the post says). Write 'none' if no specific "
        "item names are mentioned."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = response.text.strip()
    except Exception as e:
        print(f"    Gemini text screening error: {e}")
        return False, "", []

    lines = text.strip().split("\n")
    verdict = False
    reason = ""
    gemini_items_raw = ""

    for line in lines:
        line_lower = line.strip().lower()
        if line_lower.startswith("verdict:"):
            verdict = "yes" in line_lower.split(":", 1)[1]
        elif line_lower.startswith("reason:"):
            reason = line.strip().split(":", 1)[1].strip()
        elif line_lower.startswith("items:"):
            gemini_items_raw = line.strip().split(":", 1)[1].strip()

    if not verdict:
        return False, reason, []

    # ── Cross-reference Gemini-identified items against Rolimons ──
    # If Gemini flagged the post AND identified specific items, check their values.
    # Filter out the lead if all identified items are at or below the threshold.
    gemini_item_names = []
    if gemini_items_raw and gemini_items_raw.lower() != "none":
        gemini_item_names = [n.strip() for n in gemini_items_raw.split(",") if n.strip()]

    if gemini_item_names:
        print(f"    Gemini identified item(s): {', '.join(gemini_item_names)}")
        matched_above = []
        matched_below = []

        for item_name in gemini_item_names:
            match = match_single_item(item_name, name_lookup, acronym_lookup)
            if match:
                item_id, item_data = match
                value = item_data[3] if item_data[3] != -1 else item_data[2]
                item_info = {
                    "id": item_id,
                    "name": item_data[0],
                    "acronym": item_data[1],
                    "value": value,
                }
                if value > MIN_VALUE_THRESHOLD:
                    matched_above.append(item_info)
                    print(f"      {item_data[0]} — R$ {value:,} (ABOVE threshold)")
                else:
                    matched_below.append(item_info)
                    print(f"      {item_data[0]} — R$ {value:,} (at or below threshold)")
            else:
                print(f"      '{item_name}' — no Rolimons match found")

        # If we matched at least one item but ALL are at or below threshold, skip
        if (matched_above or matched_below) and not matched_above:
            names = ", ".join(f"{m['name']} (R$ {m['value']:,})" for m in matched_below)
            skip_reason = f"Item(s) identified by AI but at or below R$ {MIN_VALUE_THRESHOLD:,} threshold: {names}"
            print(f"    Filtered out: {skip_reason}")
            return False, skip_reason, []

        # If we have items above threshold, return them as a confirmed lead
        if matched_above:
            matched_above.sort(key=lambda x: x["value"], reverse=True)
            names = ", ".join(m["name"] for m in matched_above)
            return True, f"{reason} (Confirmed item(s): {names})", matched_above

    # No specific items identified or none matched Rolimons — trust Gemini's verdict
    # (e.g. "returning player" posts where no item names are mentioned)
    return verdict, reason, []


def send_discord_skip_notice(image_url: str, reason: str) -> None:
    """Send a simple Discord message when an image is skipped (for testing)."""
    embed = {
        "title": "Image Skipped",
        "color": 0x808080,
        "fields": [
            {"name": "Reason", "value": reason, "inline": False},
            {"name": "Source", "value": f"[View Image]({image_url})", "inline": False},
        ],
    }

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"embeds": [embed]},
        timeout=10,
    )

    if resp.status_code == 204:
        print("    Skip notice sent to Discord.")
    else:
        print(f"    Discord error ({resp.status_code}): {resp.text}")


# ──────────────────────────────────────────────
# PROCESS A SINGLE IMAGE
# ──────────────────────────────────────────────

def process_image(
    image_url: str,
    name_lookup: dict,
    acronym_lookup: dict,
    testing: bool = False,
    post_title: str = None,
    post_url: str = None,
):
    """Full pipeline for a single image URL.

    Returns True if a hit was found and sent, False otherwise.
    """
    print(f"\n{'='*50}")
    print(f"Processing: {image_url}")
    print(f"{'='*50}")

    # ── Download image once (reused for both calls) ──
    try:
        print("  Downloading image...")
        image_part = _download_image(image_url)
    except Exception as e:
        print(f"  Error downloading image: {e}")
        return False

    # ── Pre-screen: does this image reference a limited? ──
    print("  Pre-screening with Gemini...")
    is_relevant = prescreen_image(image_part)

    if not is_relevant:
        print("  Result: NOT a limited item image. Skipping.")
        if testing:
            send_discord_skip_notice(image_url, "Gemini determined this image does not reference a Roblox limited item.")
        return False

    print("  Result: Image likely references a limited item. Extracting details...")

    # ── Extract items ──
    try:
        raw_response = extract_items_from_image(image_part)
    except Exception as e:
        print(f"  Error extracting items: {e}")
        return False

    print(f"  Gemini output: {raw_response.strip()[:300]}")

    # ── Parse ──
    detected_items = parse_gemini_response(raw_response)

    if not detected_items:
        print("  No items detected by Gemini.")
        if testing:
            send_discord_skip_notice(image_url, "Gemini could not extract any item names from this image.")
        return False

    print(f"  Detected {len(detected_items)} item(s):")
    for d in detected_items:
        print(f"    - {d['name']}  (displayed: {d['value']:,})")

    # ── Match against Rolimons (ONLY Rolimons matches count) ──
    matches = match_items_rolimons_only(detected_items, name_lookup, acronym_lookup)

    if not matches:
        print(f"  No items above R$ {MIN_VALUE_THRESHOLD:,} threshold. Skipping.")
        return False

    # ── Report ──
    print(f"  Matched {len(matches)} Rolimons item(s):")
    for m in matches:
        tag = f" [{m['acronym']}]" if m["acronym"] else ""
        det = f"  (detected as: {m['detected_as']})" if m["detected_as"] != m["name"] else ""
        print(f"    - {m['name']}{tag}  =  R$ {m['value']:,}{det}")

    best = matches[0]
    print(f"\n  Highest value: {best['name']} at R$ {best['value']:,}")
    print("  Sending embed to Discord...")
    send_discord_embed(best, source_url=image_url, post_title=post_title, post_url=post_url)
    return True


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_url> [image_url2] [image_url3] ...")
        print("  Scans one or more image URLs for Roblox limited items.")
        print("  Only images containing Rolimons-listed limiteds trigger a Discord alert.")
        print()
        print("  Options:")
        print("    --test   Send skip notices to Discord too (for testing)")
        print()
        print('  Example: python main.py "https://i.redd.it/img1.png" "https://i.redd.it/img2.png" --test')
        sys.exit(1)

    # Parse args
    testing = "--test" in sys.argv
    image_urls = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if not image_urls:
        print("No image URLs provided.")
        sys.exit(1)

    # ── Fetch Rolimons database (once for all images) ──
    try:
        items_db = fetch_item_database()
    except Exception as e:
        print(f"Error fetching Rolimons data: {e}")
        sys.exit(1)

    name_lookup, acronym_lookup = build_lookup_tables(items_db)

    # ── Process each image ──
    hits = 0
    skips = 0

    for url in image_urls:
        found = process_image(url, name_lookup, acronym_lookup, testing=testing)
        if found:
            hits += 1
        else:
            skips += 1

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Done. {hits} hit(s), {skips} skip(s) out of {len(image_urls)} image(s).")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
