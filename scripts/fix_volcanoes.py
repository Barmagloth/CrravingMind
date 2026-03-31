#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix empty/bad discourse files."""

import re, json, html as html_mod, urllib.request
from pathlib import Path

headers = {"User-Agent": "Mozilla/5.0"}

def fetch(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")

def clean_text(text):
    text = html_mod.unescape(text)
    # Fix Windows-1252 smart quotes that sneak in as replacement chars
    text = re.sub(r'[\x80-\x9f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

DISCOURSE = Path("R:/Projects/CravingMind/data/sources/discourse")

# ── Fix 45_usgs_volcanoes.txt with PLoS volcanology ──────────────────────────
url = (
    "https://api.plos.org/search?q=subject:volcanology"
    "&rows=5&fl=abstract,title,author_display,publication_date&fq=doc_type:full"
)
raw = fetch(url)
data = json.loads(raw)
docs = data["response"]["docs"]
pieces = []
for doc in docs:
    title = doc.get("title", "")
    abstract_list = doc.get("abstract", [])
    abstract = " ".join(abstract_list) if isinstance(abstract_list, list) else str(abstract_list)
    authors = ", ".join(doc.get("author_display", [])[:3])
    date = doc.get("publication_date", "")[:10]
    if abstract and len(abstract) > 100:
        pieces.append(f"{title}\n{authors} ({date})\n\n{abstract}")
text = "\n\n---\n\n".join(pieces)
text = clean_text(text)
text = " ".join(text.split()[:1500])
out = DISCOURSE / "45_plos_volcanology.txt"
out.write_text(text, encoding="utf-8")
print(f"SAVED {out.name}: {len(text)} chars")

# Remove old empty file and rename
old = DISCOURSE / "45_usgs_volcanoes.txt"
if old.exists():
    old.unlink()
    print(f"Removed empty {old.name}")

# ── Fix 44_usgs_earthquakes.txt — decode HTML entities ───────────────────────
f44 = DISCOURSE / "44_usgs_earthquakes.txt"
text = f44.read_text(encoding="utf-8", errors="replace")
text = clean_text(text)
f44.write_text(text, encoding="utf-8")
print(f"Fixed {f44.name}: {len(text)} chars")

# ── Fix SEP files — decode HTML entities ─────────────────────────────────────
for fname in ["41_sep_consciousness.txt", "42_sep_behaviorism.txt", "43_sep_social_norms.txt"]:
    f = DISCOURSE / fname
    text = f.read_text(encoding="utf-8", errors="replace")
    text = clean_text(text)
    f.write_text(text, encoding="utf-8")
    print(f"Fixed {fname}: {len(text)} chars")

print("Done.")
