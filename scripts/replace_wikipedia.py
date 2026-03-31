"""Replace 26 Wikipedia files with texts from Gutenberg, lib.ru, RFC, man pages."""
import os, re, time, requests
from pathlib import Path

BASE = Path("data/sources")
H = {"User-Agent": "CravingMindBenchmark/1.0 (research)"}
SLEEP = 1.5


def clean(text):
    return re.sub(r'\n{3,}', '\n\n', text.strip())


def save(cat, name, text, min_words=100, max_words=1500):
    text = clean(text)
    words = text.split()
    if len(words) < min_words:
        print(f"  SKIP {cat}/{name} — too short ({len(words)} words)")
        return False
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    (BASE / cat).mkdir(parents=True, exist_ok=True)
    (BASE / cat / name).write_text(text, encoding="utf-8")
    print(f"  OK   {cat}/{name} — {len(text.split())} words")
    return True


def delete(cat, name):
    p = BASE / cat / name
    if p.exists():
        p.unlink()
        print(f"  DEL  {cat}/{name}")
    else:
        print(f"  MISS {cat}/{name} (already gone)")


def fetch(url, timeout=30):
    time.sleep(SLEEP)
    r = requests.get(url, headers=H, timeout=timeout)
    r.raise_for_status()
    return r.text


def gutenberg(book_id, start_phrase, max_words=1200):
    """Download from Project Gutenberg and extract excerpt starting at start_phrase."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        text = fetch(url)
    except Exception:
        # fallback mirror
        url2 = f"https://gutenberg.org/files/{book_id}/{book_id}-0.txt"
        try:
            text = fetch(url2)
        except Exception as e:
            print(f"  ERR  gutenberg:{book_id} — {e}")
            return None

    # find start phrase
    idx = text.find(start_phrase)
    if idx == -1:
        # try case-insensitive
        lower = text.lower()
        idx = lower.find(start_phrase.lower())
    if idx == -1:
        print(f"  ERR  gutenberg:{book_id} — start phrase not found: {start_phrase[:40]!r}")
        # just take first 1200 words after header
        lines = text.split('\n')
        body_start = 0
        for i, line in enumerate(lines):
            if '*** START OF' in line or '***START OF' in line:
                body_start = i + 1
                break
        excerpt = '\n'.join(lines[body_start:body_start + 200])
        words = excerpt.split()
        return ' '.join(words[:max_words]) if len(words) >= 100 else None

    excerpt = text[idx:idx + max_words * 8]  # rough char estimate
    words = excerpt.split()
    return ' '.join(words[:max_words])


def rfc(number, section_hint=None, max_words=1200):
    """Download RFC text from rfc-editor.org."""
    url = f"https://www.rfc-editor.org/rfc/rfc{number}.txt"
    try:
        text = fetch(url)
    except Exception as e:
        print(f"  ERR  rfc{number} — {e}")
        return None

    # Skip the header boilerplate — find first real content after page 1 header
    lines = text.split('\n')
    body_lines = []
    in_body = False
    skip_next_blank = False
    for line in lines:
        # Skip page headers/footers (lines with RFC number at start/end)
        if re.match(r'^RFC \d+', line) or re.match(r'^\s*\[Page \d+\]', line):
            skip_next_blank = True
            continue
        if skip_next_blank and line.strip() == '':
            skip_next_blank = False
            continue
        if not in_body:
            # Start after "Abstract" or "1." or "Status of this Memo"
            if re.match(r'^(Abstract|1\.|Status of|Introduction)', line.strip()):
                in_body = True
        if in_body:
            body_lines.append(line)
        if len(body_lines) > 400:
            break

    body = '\n'.join(body_lines)
    words = body.split()
    if len(words) < 100:
        # fallback: just take words from middle of doc
        all_words = text.split()
        start = len(all_words) // 10
        return ' '.join(all_words[start:start + max_words])
    return ' '.join(words[:max_words])


def libru(path, encoding='utf-8', max_words=1200):
    """Download from lib.ru."""
    url = f"https://lib.ru/{path}"
    try:
        text = fetch(url)
    except Exception as e:
        print(f"  ERR  lib.ru/{path} — {e}")
        return None

    # lib.ru often has HTML — strip tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    words = text.split()
    if len(words) < 100:
        print(f"  ERR  lib.ru/{path} — too short after HTML strip")
        return None
    # Skip first ~50 words (usually header/meta)
    start = min(50, len(words) // 10)
    return ' '.join(words[start:start + max_words])


def manpage(cmd, url=None, max_words=1200):
    """Download man page from man7.org."""
    if url is None:
        url = f"https://man7.org/linux/man-pages/man1/{cmd}.1.html"
    try:
        text = fetch(url)
    except Exception as e:
        print(f"  ERR  man/{cmd} — {e}")
        return None

    # Strip HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Find NAME section
    idx = text.find('NAME')
    if idx == -1:
        idx = text.find('SYNOPSIS')
    if idx == -1:
        idx = 0

    excerpt = text[idx:idx + max_words * 8]
    words = excerpt.split()
    if len(words) < 100:
        words = text.split()
        start = len(words) // 5
        return ' '.join(words[start:start + max_words])
    return ' '.join(words[:max_words])


# ─── DISCOURSE replacements ────────────────────────────────────────────────

print("\n=== DISCOURSE ===")

# Delete old Wikipedia files
for f in [
    "26_wiki_photosynthesis.txt", "27_wiki_jazz.txt", "28_wiki_roman_roads.txt",
    "29_wiki_game_theory.txt", "30_wiki_coral_reef.txt", "31_wiki_crispr.txt",
    "32_wiki_silk_road.txt", "33_wiki_impressionism.txt", "34_wiki_blockchain.txt",
    "35_wiki_stoicism.txt", "36_wiki_plate_tectonics.txt", "37_wiki_dark_matter.txt",
    "38_wiki_cognitive_bias.txt", "39_wiki_fermentation.txt", "40_wiki_beekeeping.txt",
]:
    delete("discourse", f)

print("\n-- Gutenberg fiction --")
t = gutenberg(1661, "To Sherlock Holmes")
if t: save("discourse", "26_gutenberg_sherlock.txt", t)

t = gutenberg(345, "3 May. Bistritz")
if t: save("discourse", "27_gutenberg_dracula.txt", t)

t = gutenberg(11, "Alice was beginning")
if t: save("discourse", "28_gutenberg_alice.txt", t)

t = gutenberg(35, "The Time Traveller")
if t: save("discourse", "29_gutenberg_time_machine.txt", t)

t = gutenberg(36, "No one would have believed")
if t: save("discourse", "30_gutenberg_war_worlds.txt", t)

print("\n-- lib.ru English texts --")
t = libru("LONDON/london07.txt")
if t: save("discourse", "31_libru_london_before_adam.txt", t)

t = libru("LONDON/bbtard.txt")
if t: save("discourse", "32_libru_london_batard.txt", t)

t = libru("LONDON/godfather.txt")
if t: save("discourse", "33_libru_london_godfather.txt", t)

print("\n-- RFC specs --")
t = rfc(2616)
if t: save("discourse", "34_rfc2616_http11.txt", t)

t = rfc(7231)
if t: save("discourse", "35_rfc7231_http_methods.txt", t)

t = rfc(6749)
if t: save("discourse", "36_rfc6749_oauth.txt", t)

print("\n-- Man pages --")
t = manpage("find")
if t: save("discourse", "37_man_find.txt", t)

t = manpage("ssh", url="https://man7.org/linux/man-pages/man1/ssh.1.html")
if t: save("discourse", "38_man_ssh.txt", t)

t = manpage("tar", url="https://man7.org/linux/man-pages/man1/tar.1.html")
if t: save("discourse", "39_man_tar.txt", t)

t = manpage("curl")
if t: save("discourse", "40_man_curl.txt", t)


# ─── NEEDLE replacements ──────────────────────────────────────────────────

print("\n=== NEEDLE ===")

# Delete old Wikipedia files
for f in [
    "21_diabetes.txt", "22_malaria.txt", "25_tuberculosis.txt",
    "26_titanium.txt", "27_lithium.txt", "28_graphene.txt",
    "30_silicon.txt", "31_boeing_747.txt", "32_iss.txt",
    "33_lhc.txt", "34_hubble.txt", "35_panama_canal.txt",
    "36_vesuvius.txt", "37_yellowstone.txt", "38_krakatoa.txt",
    "40_great_barrier.txt",
]:
    delete("needle", f)

# Also delete the ones that may have been replaced already
for f in ["23_alzheimers.txt", "24_parkinsons.txt", "29_carbon_fiber.txt", "39_tohoku_eq.txt"]:
    delete("needle", f)

print("\n-- RFC needle --")
t = rfc(5246)
if t: save("needle", "21_rfc5246_tls12.txt", t)

t = rfc(2068)
if t: save("needle", "22_rfc2068_http10.txt", t)

t = rfc(2818)
if t: save("needle", "36_rfc2818_http_tls.txt", t)

t = rfc(4648)
if t: save("needle", "37_rfc4648_base_encodings.txt", t)

t = rfc(7519)
if t: save("needle", "38_rfc7519_jwt.txt", t)

t = rfc(3986)
if t: save("needle", "39_rfc3986_uri.txt", t)

t = rfc(2822)
if t: save("needle", "40_rfc2822_email.txt", t)

t = rfc(791)
if t: save("needle", "29_rfc791_ip.txt", t)

print("\n-- Gutenberg needle --")
t = gutenberg(1228, "When on board")
if t: save("needle", "31_gutenberg_origin_species.txt", t)

t = gutenberg(3300, "The annual labour")
if t: save("needle", "32_gutenberg_wealth_nations.txt", t)

# Newton's Principia — just grab first real content
t = gutenberg(28233, "DEFINITIONS")
if t: save("needle", "33_gutenberg_principia.txt", t)

t = gutenberg(132, "Sun Tzu said")
if t: save("needle", "34_gutenberg_art_of_war.txt", t)

t = gutenberg(1497, "PERSONS OF THE DIALOGUE")
if t: save("needle", "35_gutenberg_republic.txt", t)

t = gutenberg(21076, "BOOK I")
if t: save("needle", "23_gutenberg_euclid.txt", t)

t = gutenberg(147, "INTRODUCTION")
if t: save("needle", "26_gutenberg_common_sense.txt", t)

t = gutenberg(1232, "THE PRINCE")
if t: save("needle", "30_gutenberg_prince.txt", t)

print("\n-- Man pages needle --")
t = manpage("git", url="https://man7.org/linux/man-pages/man1/git.1.html")
if t: save("needle", "24_man_git.txt", t)

t = manpage("ip", url="https://man7.org/linux/man-pages/man8/ip.8.html")
if t: save("needle", "25_man_ip.txt", t)

t = manpage("systemctl", url="https://man7.org/linux/man-pages/man1/systemctl.1.html")
if t: save("needle", "28_man_systemctl.txt", t)

print("\n-- lib.ru needle --")
# Bernard Shaw — try a few files
t = libru("SHAW/pyg.txt")
if not t:
    t = libru("SHAW/candida.txt")
if t: save("needle", "27_libru_shaw.txt", t)


# ─── Summary ─────────────────────────────────────────────────────────────

print("\n=== SUMMARY ===")
for cat in ["discourse", "needle"]:
    files = sorted((BASE / cat).glob("*.txt"))
    wiki = [f for f in files if "wiki" in f.name]
    print(f"{cat}: {len(files)} total, {len(wiki)} Wikipedia")
    if wiki:
        for f in wiki:
            print(f"  still wiki: {f.name}")
