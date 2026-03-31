#!/usr/bin/env python3
"""Fetch diverse science texts and multi-language code files."""

import re
import time
import urllib.request
import urllib.error
import json
from pathlib import Path

DISCOURSE_DIR = Path("R:/Projects/CravingMind/data/sources/discourse")
CODE_DIR = Path("R:/Projects/CravingMind/data/sources/code")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; +https://example.com)"
}

def fetch_url(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

def strip_html(html):
    # Remove script/style blocks
    html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def words(text):
    return text.split()

def trim_words(text, n=1500):
    w = words(text)
    return ' '.join(w[:n])

def trim_lines(text, n=200):
    lines = text.splitlines()
    return '\n'.join(lines[:n])

def save(path, content):
    path.write_text(content, encoding="utf-8")
    print(f"  SAVED {path.name} ({len(content)} chars)")

# ── Discourse: Stanford Encyclopedia of Philosophy ──────────────────────────

sep_texts = [
    ("41_sep_consciousness.txt",  "https://plato.stanford.edu/entries/consciousness/"),
    ("42_sep_behaviorism.txt",    "https://plato.stanford.edu/entries/behaviorism/"),
    ("43_sep_social_norms.txt",   "https://plato.stanford.edu/entries/social-norms/"),
]

def extract_sep_content(html):
    """Extract main article body from SEP page."""
    # SEP wraps content in <div id="main-text"> or <div id="aueditable">
    for pattern in [
        r'<div[^>]*id=["\']main-text["\'][^>]*>(.*?)</div>\s*<div[^>]*id=["\']bibliography',
        r'<div[^>]*id=["\']aueditable["\'][^>]*>(.*?)</div>\s*(?=<div)',
    ]:
        m = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if m:
            return strip_html(m.group(1))
    # Fallback: strip all HTML
    return strip_html(html)

print("=== DISCOURSE: Stanford Encyclopedia of Philosophy ===")
for fname, url in sep_texts:
    out = DISCOURSE_DIR / fname
    if out.exists():
        print(f"  SKIP {fname} (exists)")
        continue
    try:
        print(f"  Fetching {url}")
        html = fetch_url(url)
        content = extract_sep_content(html)
        # Skip nav by finding first real paragraph (after "1.")
        idx = content.find("1.")
        if idx > 200:
            content = content[idx:]
        text = trim_words(content, 1500)
        save(out, text)
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR {fname}: {e}")

# ── Discourse: USGS geology pages ───────────────────────────────────────────

usgs_texts = [
    ("44_usgs_earthquakes.txt", "https://pubs.usgs.gov/gip/earthq1/"),
    ("45_usgs_volcanoes.txt",   "https://pubs.usgs.gov/gip/volc/"),
]

print("\n=== DISCOURSE: USGS ===")
for fname, url in usgs_texts:
    out = DISCOURSE_DIR / fname
    if out.exists():
        print(f"  SKIP {fname} (exists)")
        continue
    try:
        print(f"  Fetching {url}")
        html = fetch_url(url)
        text = strip_html(html)
        # Skip first 300 chars (likely navigation)
        text = text[300:]
        text = trim_words(text, 1500)
        save(out, text)
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR {fname}: {e}")

# ── Discourse: PLoS ONE abstracts ────────────────────────────────────────────

plos_subjects = [
    ("46_plos_paleontology.txt", "paleontology"),
    ("47_plos_anthropology.txt", "anthropology"),
    ("48_plos_psychology.txt",   "psychology"),
    ("49_plos_sociology.txt",    "sociology"),
    ("50_plos_ecology.txt",      "ecology"),
]

print("\n=== DISCOURSE: PLoS ONE ===")
for fname, subject in plos_subjects:
    out = DISCOURSE_DIR / fname
    if out.exists():
        print(f"  SKIP {fname} (exists)")
        continue
    try:
        url = (
            f"https://api.plos.org/search?q=subject:{subject}"
            f"&rows=5&fl=abstract,title,author_display,publication_date"
            f"&fq=doc_type:full"
        )
        print(f"  Fetching PLoS subject:{subject}")
        raw = fetch_url(url)
        data = json.loads(raw)
        docs = data.get("response", {}).get("docs", [])
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
        text = trim_words(text, 1500)
        save(out, text)
        time.sleep(1)
    except Exception as e:
        print(f"  ERROR {fname}: {e}")

# ── Code: GitHub raw files ───────────────────────────────────────────────────

GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"

diverse_code = [
    # Rust
    ("41_rust_option.txt",        "rust-lang/rust",                     "library/core/src/option.rs"),
    ("42_rust_vec.txt",           "rust-lang/rust",                     "library/alloc/src/vec/mod.rs"),
    # Go
    ("43_go_sort.txt",            "golang/go",                          "src/sort/sort.go"),
    ("44_go_http.txt",            "golang/go",                          "src/net/http/server.go"),
    # TypeScript
    ("45_ts_event.txt",           "microsoft/vscode",                   "src/vs/base/common/event.ts"),
    # C#
    ("46_csharp_list.txt",        "dotnet/runtime",                     "src/libraries/System.Private.CoreLib/src/System/Collections/Generic/List.cs"),
    ("47_csharp_linq.txt",        "dotnet/runtime",                     "src/libraries/System.Linq/src/System/Linq/Where.cs"),
    # Java
    ("48_java_hashmap.txt",       "openjdk/jdk",                        "src/java.base/share/classes/java/util/HashMap.java"),
    ("49_java_stream.txt",        "openjdk/jdk",                        "src/java.base/share/classes/java/util/stream/Stream.java"),
    # Ruby
    ("50_ruby_array.txt",         "ruby/ruby",                          "array.c"),
    # Haskell
    ("51_haskell_prelude.txt",    "ghc/ghc",                            "libraries/base/Prelude.hs"),
    # Kotlin
    ("52_kotlin_flow.txt",        "Kotlin/kotlinx.coroutines",          "kotlinx-coroutines-core/common/src/flow/Flow.kt"),
    # Swift
    ("53_swift_array.txt",        "apple/swift",                        "stdlib/public/core/Array.swift"),
    # Scala
    ("54_scala_list.txt",         "scala/scala",                        "src/library/scala/collection/immutable/List.scala"),
    # Shell/Bash
    ("55_bash_git_prompt.txt",    "git/git",                            "contrib/completion/git-prompt.sh"),
    # Lua
    ("56_lua_table.txt",          "lua/lua",                            "ltable.c"),
    # CSS
    ("57_css_normalize.txt",      "necolas/normalize.css",              "normalize.css"),
    # YAML (GitHub Actions)
    ("58_yaml_ci.txt",            "actions/starter-workflows",          "ci/python-app.yml"),
    # Makefile
    ("59_makefile_linux.txt",     "torvalds/linux",                     "Makefile"),
    # Assembly
    ("60_asm_hello.txt",          "cirosantilli/x86-assembly-cheat",    "hello.asm"),
    # COBOL
    ("61_cobol_hello.txt",        "OpenMainframeProject/cobol-programming-course",
                                  "COBOL Programming Course #2 - Advanced Topics/Labs/cbl/CBL0001.cbl"),
    # SQL
    ("62_sql_northwind.txt",      "microsoft/sql-server-samples",
                                  "samples/databases/northwind-pubs/instnwnd.sql"),
    # TypeScript compiler
    ("63_ts_compiler.txt",        "microsoft/TypeScript",               "src/compiler/checker.ts"),
]

BRANCH_CANDIDATES = ["main", "master"]

def fetch_github(repo, path):
    """Try main then master branch."""
    for branch in BRANCH_CANDIDATES:
        url = GITHUB_RAW.format(repo=repo, branch=branch, path=path)
        try:
            content = fetch_url(url)
            if content.startswith("404:"):
                continue
            return content, url
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    return None, None

print("\n=== CODE: GitHub files ===")
skipped = []
for fname, repo, path in diverse_code:
    out = CODE_DIR / fname
    if out.exists():
        print(f"  SKIP {fname} (exists)")
        continue
    try:
        print(f"  Fetching {repo}/{path}")
        content, url = fetch_github(repo, path)
        if content is None:
            print(f"  NOT FOUND {fname}")
            skipped.append(fname)
            continue
        text = trim_lines(content, 200)
        save(out, text)
        time.sleep(0.5)
    except Exception as e:
        print(f"  ERROR {fname}: {e}")
        skipped.append(fname)

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n=== SUMMARY ===")
code_files   = list(CODE_DIR.glob("*.txt"))
disc_files   = list(DISCOURSE_DIR.glob("*.txt"))
needle_files = list(Path("R:/Projects/CravingMind/data/sources/needle").glob("*.txt"))

print(f"Code:      {len(code_files)} files")
print(f"Discourse: {len(disc_files)} files")
print(f"Needle:    {len(needle_files)} files")
print(f"Total:     {len(code_files)+len(disc_files)+len(needle_files)} files")

# Wikipedia vs non-Wikipedia in discourse
wiki = [f for f in disc_files if "_wp_" in f.name or f.name.startswith("0") or "_wiki" in f.name.lower()]
# Rough check by filename prefix pattern
wv = [f for f in disc_files if "_wv_" in f.name]
print(f"\nDiscourse files by source:")
for f in sorted(disc_files, key=lambda x: x.name):
    print(f"  {f.name}")

if skipped:
    print(f"\nSkipped (not found): {skipped}")
