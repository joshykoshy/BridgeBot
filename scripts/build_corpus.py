"""Build the cached knowledge base: ~60 English Wikipedia articles + religiondatabase.org.

Why a cached corpus: v1 scraped one page at every start-up, so the bot knew little and
broke without internet. Here we fetch once, save each source as markdown with front-matter
metadata (source_url, title, retrieved_at, country, religion, topic), and commit the files.
The demo then works offline and every answer can cite a stable URL.

How: the MediaWiki `extracts` API (plain text, section markers like `== Etiquette ==`) is
converted to markdown headings so the heading-aware chunker can use them. Boilerplate
sections (References, See also, ...) are dropped and each article is capped in length so a
few very long articles (e.g. Christmas) don't dominate the index.

Run: python scripts/build_corpus.py [--force]
Licence: Wikipedia text is CC BY-SA 4.0; attribution is in the README and in each file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "BridgeBot/2.0 (student portfolio project; cultural etiquette RAG demo)"}
MAX_WORDS = 6000
DROP_SECTIONS = {"references", "see also", "external links", "further reading", "notes", "bibliography",
                 "sources", "citations", "footnotes", "gallery", "explanatory notes", "works cited"}

# (title, country, religion, topic). Empty string = not specific.
ARTICLES: list[tuple[str, str, str, str]] = [
    # Etiquette by region
    ("Etiquette in Japan", "Japan", "", "etiquette"),
    ("Etiquette in Asia", "", "", "etiquette"),
    ("Etiquette in the Middle East", "", "Islam", "etiquette"),
    ("Etiquette in Latin America", "", "", "etiquette"),
    ("Etiquette in Africa", "", "", "etiquette"),
    ("Etiquette in Europe", "", "", "etiquette"),
    ("Etiquette in Canada and the United States", "United States", "", "etiquette"),
    ("Etiquette in Australia and New Zealand", "Australia", "", "etiquette"),
    ("Etiquette in Indonesia", "Indonesia", "", "etiquette"),
    ("Etiquette in South Korea", "South Korea", "", "etiquette"),
    ("Etiquette in India", "India", "", "etiquette"),
    ("Chinese social relations", "China", "", "etiquette"),
    ("Etiquette", "", "", "etiquette"),
    ("Table manners", "", "", "dining"),
    ("Face (sociological concept)", "China", "", "communication"),
    ("Guanxi", "China", "", "business"),
    ("High-context and low-context cultures", "", "", "communication"),
    ("Japanese honorifics", "Japan", "", "communication"),
    ("Korean honorifics", "South Korea", "", "communication"),
    # Greetings and gestures
    ("Bowing", "", "", "greetings"),
    ("Bowing in Japan", "Japan", "", "greetings"),
    ("Handshake", "", "", "greetings"),
    ("Namaste", "India", "Hinduism", "greetings"),
    ("Wai (gesture)", "Thailand", "Buddhism", "greetings"),
    ("Cheek kissing", "", "", "greetings"),
    ("Salah", "", "Islam", "observance"),
    # Gifts, business, tipping, dress
    ("Gift", "", "", "gifts"),
    ("Red envelope", "China", "", "gifts"),
    ("Omiyage", "Japan", "", "gifts"),
    ("Business card", "", "", "business"),
    ("Gratuity", "", "", "tipping"),
    ("Dress code", "", "", "dress"),
    ("Hijab", "", "Islam", "dress"),
    ("Modesty", "", "", "dress"),
    ("Japanese tea ceremony", "Japan", "", "etiquette"),
    ("Chopsticks", "China", "", "dining"),
    # Places of worship
    ("Mosque", "", "Islam", "places of worship"),
    ("Hindu temple", "India", "Hinduism", "places of worship"),
    ("Gurdwara", "India", "Sikhism", "places of worship"),
    ("Synagogue", "", "Judaism", "places of worship"),
    ("Langar (Sikhism)", "India", "Sikhism", "food"),
    # Religious periods and festivals
    ("Ramadan", "", "Islam", "observance"),
    ("Iftar", "", "Islam", "observance"),
    ("Eid al-Fitr", "", "Islam", "observance"),
    ("Eid al-Adha", "", "Islam", "observance"),
    ("Diwali", "India", "Hinduism", "observance"),
    ("Holi", "India", "Hinduism", "observance"),
    ("Navaratri", "India", "Hinduism", "observance"),
    ("Chinese New Year", "China", "", "observance"),
    ("Mid-Autumn Festival", "China", "", "observance"),
    ("Chuseok", "South Korea", "", "observance"),
    ("Seollal", "South Korea", "", "observance"),
    ("Hanukkah", "", "Judaism", "observance"),
    ("Shabbat", "", "Judaism", "observance"),
    ("Passover", "", "Judaism", "observance"),
    ("Yom Kippur", "", "Judaism", "observance"),
    ("Christmas", "", "Christianity", "observance"),
    ("Lent", "", "Christianity", "observance"),
    ("Easter", "", "Christianity", "observance"),
    ("Rio Carnival", "Brazil", "", "observance"),
    ("Day of the Dead", "Mexico", "", "observance"),
    ("Vesak", "", "Buddhism", "observance"),
    ("Songkran", "Thailand", "Buddhism", "observance"),
    # Diet
    ("Halal", "", "Islam", "food"),
    ("Kashrut", "", "Judaism", "food"),
    ("Diet in Hinduism", "India", "Hinduism", "food"),
    ("Buddhist vegetarianism", "", "Buddhism", "food"),
    ("Islamic dietary laws", "", "Islam", "food"),
]


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def fetch_extract(title: str) -> tuple[str, str, str] | None:
    """Return (resolved_title, url, plain_text) or None if the page doesn't exist."""
    params = {"action": "query", "prop": "extracts|info", "inprop": "url", "explaintext": 1,
              "exsectionformat": "wiki", "redirects": 1, "titles": title, "format": "json", "formatversion": 2}
    for attempt in range(5):
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 429:
            break
        wait = int(resp.headers.get("Retry-After", 0) or 0) or 5 * 2 ** attempt
        print(f"  rate limited, waiting {wait}s")
        time.sleep(wait)
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return page["title"], page["fullurl"], page["extract"]


def wiki_to_markdown(title: str, text: str) -> str:
    """'== Heading ==' -> '## Heading'; drop boilerplate sections; cap length."""
    out = [f"# {title}", ""]
    skipping = False
    words = 0
    for line in text.splitlines():
        match = re.match(r"^(={2,6})\s*(.+?)\s*\1\s*$", line)
        if match:
            level = len(match.group(1))
            name = match.group(2)
            if level == 2:
                skipping = name.lower() in DROP_SECTIONS
            if not skipping:
                out += ["", "#" * level + " " + name, ""]
            continue
        if skipping or not line.strip():
            if not skipping:
                out.append("")
            continue
        words += len(line.split())
        if words > MAX_WORDS:
            break
        out.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def front_matter(meta: dict) -> str:
    lines = ["---"] + [f"{k}: {str(v).replace(chr(10), ' ')}" for k, v in meta.items()] + ["---", ""]
    return "\n".join(lines)


def fetch_religiondatabase() -> str | None:
    """v1's original source, kept for continuity (page text only)."""
    from bs4 import BeautifulSoup
    try:
        resp = requests.get("https://religiondatabase.org/browse", headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as err:
        print("  religiondatabase.org failed:", err)
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = "\n".join(ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip())
    return text if len(text.split()) > 50 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    ok, missing = 0, []
    seen = {re.search(r"^title: (.*)$", p.read_text(encoding="utf-8"), re.M).group(1)
            for p in CORPUS_DIR.glob("*.md")}
    for title, country, religion, topic in ARTICLES:
        path = CORPUS_DIR / f"{slug(title)}.md"
        if path.exists() and not args.force:
            ok += 1
            continue
        result = fetch_extract(title)
        if result is None:
            missing.append(title)
            print("  missing:", title)
            continue
        resolved, url, text = result
        if resolved in seen:  # a redirect to an article we already have
            missing.append(f"{title} (duplicate of {resolved})")
            continue
        seen.add(resolved)
        meta = {"title": resolved, "source_url": url, "retrieved_at": today, "country": country,
                "religion": religion, "topic": topic, "license": "CC BY-SA 4.0 (Wikipedia)"}
        path.write_text(front_matter(meta) + wiki_to_markdown(resolved, text), encoding="utf-8")
        ok += 1
        print(f"  ok: {resolved}")
        time.sleep(1.5)  # be polite to the API
    rdb = CORPUS_DIR / "religiondatabase-browse.md"
    if args.force or not rdb.exists():
        text = fetch_religiondatabase()
        if text:
            meta = {"title": "Database of Religious History (browse)", "source_url": "https://religiondatabase.org/browse",
                    "retrieved_at": today, "country": "", "religion": "", "topic": "religion",
                    "license": "see source site"}
            rdb.write_text(front_matter(meta) + "# Database of Religious History\n\n" + text + "\n", encoding="utf-8")
            ok += 1
    print(f"done: {ok} sources in {CORPUS_DIR}, missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
