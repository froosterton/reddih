import os
import sys
import json
import re
import requests
from google import genai
from google.genai import types

# ──────────────────────────────────────────────
# CONFIGURATION (env vars — do not commit secrets)
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
ROLIMONS_API_URL = "https://www.rolimons.com/itemapi/itemdetails"
MIN_VALUE_THRESHOLD = 100_000  # ignore items worth less than this


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

    if text.startswith("
