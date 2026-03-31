#!/usr/bin/env python3
"""Download 120 benchmark texts from diverse real sources."""
import os, re, json, time, requests, xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path("data/sources")
H = {"User-Agent": "CravingMindBenchmark/1.0"}

def save(cat, name, text, min_words=100):
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    if len(text.split()) < min_words:
        print(f"  SKIP {cat}/{name} — too short ({len(text.split())} words)")
        return False
    # Trim to ~1500 words for discourse/needle
    if cat != "code":
        words = text.split()
        if len(words) > 1500:
            text = " ".join(words[:1500])
            last_dot = text.rfind(".")
            if last_dot > len(text)*0.7: text = text[:last_dot+1]
    else:
        lines = text.split("\n")
        if len(lines) > 200: text = "\n".join(lines[:200])
    (BASE/cat).mkdir(parents=True, exist_ok=True)
    (BASE/cat/name).write_text(text, encoding="utf-8")
    wc = len(text.split())
    print(f"  OK {cat}/{name} — {wc} words")
    return True

def gutenberg(gid, start_phrase):
    for pat in [f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt", f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"]:
        try:
            r = requests.get(pat, headers=H, timeout=30)
            if r.status_code == 200:
                t = r.text
                i = t.find(start_phrase)
                if i == -1: i = t.lower().find(start_phrase.lower())
                if i == -1:
                    i = t.rfind("***")
                    i = i+3 if i != -1 else 0
                return t[i:i+8000]
        except: pass
    return None

def wiki(title):
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={"action":"query","titles":title,"prop":"extracts","explaintext":True,"format":"json"}, headers=H, timeout=30)
        pages = r.json()["query"]["pages"]
        for p in pages.values(): return p.get("extract","")
    except: return None

def wikivoyage(title):
    try:
        r = requests.get("https://en.wikivoyage.org/w/api.php", params={"action":"query","titles":title,"prop":"extracts","explaintext":True,"format":"json"}, headers=H, timeout=30)
        pages = r.json()["query"]["pages"]
        for p in pages.values(): return p.get("extract","")
    except: return None

def rfc(num, start_pat=None):
    try:
        r = requests.get(f"https://www.rfc-editor.org/rfc/rfc{num}.txt", headers=H, timeout=30)
        if r.status_code != 200: return None
        t = r.text
        if start_pat:
            m = re.search(start_pat, t, re.IGNORECASE)
            if m: return t[m.start():m.start()+8000]
        # Skip TOC
        i = t.find("1.  Introduction")
        if i == -1: i = t.find("1. Introduction")
        if i == -1: i = min(2000, len(t)//4)
        return t[i:i+8000]
    except: return None

def arxiv(query):
    try:
        r = requests.get(f"http://export.arxiv.org/api/query?search_query={query}&max_results=1&sortBy=relevance", headers=H, timeout=30)
        root = ET.fromstring(r.text)
        ns = {"a":"http://www.w3.org/2005/Atom"}
        e = root.findall("a:entry", ns)
        if not e: return None
        title = e[0].find("a:title",ns).text.strip()
        summary = e[0].find("a:summary",ns).text.strip()
        authors = ", ".join([a.find("a:name",ns).text for a in e[0].findall("a:author",ns)[:5]])
        return f"{title}\n\n{authors}\n\nAbstract: {summary}"
    except: return None

def github(repo, path):
    for branch in ["main","master"]:
        try:
            r = requests.get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}", headers=H, timeout=30)
            if r.status_code == 200: return r.text
        except: pass
    return None

def libru(path):
    try:
        r = requests.get(f"https://lib.ru/{path}", headers=H, timeout=30)
        if r.status_code == 200:
            t = r.text
            # Strip HTML if present
            t = re.sub(r'<[^>]+>', '', t)
            # Find start of actual text (skip navigation)
            for marker in ["Day had broken", "CHAPTER", "Chapter", "It was", "The ", "In the"]:
                i = t.find(marker)
                if i > 0 and i < 2000: return t[i:i+8000]
            return t[500:8500]  # fallback
    except: pass
    return None

def manpage(cmd):
    try:
        r = requests.get(f"https://man7.org/linux/man-pages/man1/{cmd}.1.html", headers=H, timeout=30)
        if r.status_code != 200:
            r = requests.get(f"https://man7.org/linux/man-pages/man1/{cmd}.1p.html", headers=H, timeout=30)
        if r.status_code == 200:
            t = re.sub(r'<[^>]+>', '', r.text)
            t = re.sub(r'\s+', ' ', t).strip()
            # Find NAME section
            i = t.find("NAME")
            if i > 0: return t[i:i+8000]
            return t[:8000]
    except: pass
    return None

# ============================================================
print("=== CLEANING ===")
for cat in ["discourse","needle","code"]:
    d = BASE/cat
    if d.exists():
        for f in d.glob("*.txt"): f.unlink()
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
print("\n=== DISCOURSE (40 texts) ===")

# Gutenberg — literary prose (5)
print("\n--- Gutenberg ---")
guten = [
    ("01_austen_pride.txt", 1342, "It is a truth universally acknowledged"),
    ("02_shelley_frankenstein.txt", 84, "You will rejoice to hear"),
    ("03_dickens_two_cities.txt", 98, "It was the best of times"),
    ("04_melville_moby_dick.txt", 2701, "Call me Ishmael"),
    ("05_twain_tom_sawyer.txt", 74, "TOM!"),
]
for name, gid, start in guten:
    t = gutenberg(gid, start)
    if t: save("discourse", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# lib.ru — English originals (5)
print("\n--- lib.ru ---")
libru_texts = [
    ("06_london_fire.txt", "LONDON/fire.txt"),
    ("07_london_call_wild.txt", "LONDON/callw10.txt"),
    ("08_wilde_dorian.txt", "WILDE/dorian_engl.txt"),
    ("09_orwell_animal.txt", "ORWELL/animalfarm_engl.txt"),
    ("10_hemingway_old_man.txt", "HEMINGWAY/oldman_engl.txt"),
]
for name, path in libru_texts:
    t = libru(path)
    if t: save("discourse", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# arXiv — scientific (5)
print("\n--- arXiv ---")
arxiv_q = [
    ("11_arxiv_transformers.txt", "abs:attention+mechanism+transformer"),
    ("12_arxiv_gan.txt", "abs:generative+adversarial+network"),
    ("13_arxiv_rl_atari.txt", "abs:deep+reinforcement+learning+atari"),
    ("14_arxiv_alphafold.txt", "abs:protein+structure+prediction+alphafold"),
    ("15_arxiv_diffusion.txt", "abs:denoising+diffusion+probabilistic"),
]
for name, q in arxiv_q:
    t = arxiv(q)
    if t: save("discourse", name, t, min_words=50)
    else: print(f"  FAIL {name}")
    time.sleep(3)

# RFC intros — technical spec (3)
print("\n--- RFC ---")
rfc_disc = [
    ("16_rfc9110_http.txt", 9110, None),
    ("17_rfc793_tcp.txt", 793, None),
    ("18_rfc1035_dns.txt", 1035, None),
]
for name, num, pat in rfc_disc:
    t = rfc(num, pat)
    if t: save("discourse", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# Wikivoyage — travel guides (4)
print("\n--- Wikivoyage ---")
wv = [("19_wv_tokyo.txt","Tokyo"),("20_wv_paris.txt","Paris"),("21_wv_cairo.txt","Cairo"),("22_wv_reykjavik.txt","Reykjavik")]
for name, title in wv:
    t = wikivoyage(title)
    if t: save("discourse", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# Man pages — Linux documentation (3)
print("\n--- Man pages ---")
mans = [("23_man_grep.txt","grep"),("24_man_awk.txt","awk"),("25_man_sed.txt","sed")]
for name, cmd in mans:
    t = manpage(cmd)
    if t: save("discourse", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# Wikipedia — diverse topics, encyclopedia format (15)
print("\n--- Wikipedia (diverse topics) ---")
wiki_disc = [
    ("26_wiki_photosynthesis.txt","Photosynthesis"),
    ("27_wiki_jazz.txt","Jazz"),
    ("28_wiki_roman_roads.txt","Roman_roads"),
    ("29_wiki_game_theory.txt","Game_theory"),
    ("30_wiki_coral_reef.txt","Coral_reef"),
    ("31_wiki_crispr.txt","CRISPR_gene_editing"),
    ("32_wiki_silk_road.txt","Silk_Road"),
    ("33_wiki_impressionism.txt","Impressionism"),
    ("34_wiki_blockchain.txt","Blockchain"),
    ("35_wiki_stoicism.txt","Stoicism"),
    ("36_wiki_plate_tectonics.txt","Plate_tectonics"),
    ("37_wiki_dark_matter.txt","Dark_matter"),
    ("38_wiki_cognitive_bias.txt","Cognitive_bias"),
    ("39_wiki_fermentation.txt","Fermentation"),
    ("40_wiki_beekeeping.txt","Beekeeping"),
]
for name, title in wiki_disc:
    t = wiki(title)
    if t: save("discourse", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# ============================================================
print("\n=== NEEDLE (40 texts) ===")

# RFC specific sections — protocol details with numbers (5)
print("\n--- RFC sections ---")
rfc_needle = [
    ("01_rfc9110_methods.txt", 9110, r"9\.\s+Methods"),
    ("02_rfc793_segment.txt", 793, r"3\.\s+FUNCTIONAL"),
    ("03_rfc1035_rr.txt", 1035, r"3\.2\.\s+RR"),
    ("04_rfc8446_handshake.txt", 8446, r"4\.\s+Handshake"),
    ("05_rfc5321_commands.txt", 5321, r"4\.\s+The SMTP"),
]
for name, num, pat in rfc_needle:
    t = rfc(num, pat)
    if t: save("needle", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(1)

# Wikipedia — fact-dense (biographies, countries, battles, diseases, companies) (35)
print("\n--- Wikipedia fact-dense ---")
wiki_needle = [
    ("06_tesla_bio.txt","Nikola_Tesla"),
    ("07_curie_bio.txt","Marie_Curie"),
    ("08_turing_bio.txt","Alan_Turing"),
    ("09_lovelace_bio.txt","Ada_Lovelace"),
    ("10_einstein_bio.txt","Albert_Einstein"),
    ("11_japan.txt","Japan"),
    ("12_brazil.txt","Brazil"),
    ("13_australia.txt","Australia"),
    ("14_south_korea.txt","South_Korea"),
    ("15_european_union.txt","European_Union"),
    ("16_stalingrad.txt","Battle_of_Stalingrad"),
    ("17_normandy.txt","Normandy_landings"),
    ("18_midway.txt","Battle_of_Midway"),
    ("19_leningrad.txt","Siege_of_Leningrad"),
    ("20_kursk.txt","Battle_of_Kursk"),
    ("21_diabetes.txt","Type_2_diabetes"),
    ("22_malaria.txt","Malaria"),
    ("23_alzheimers.txt","Alzheimer%27s_disease"),
    ("24_parkinsons.txt","Parkinson%27s_disease"),
    ("25_tuberculosis.txt","Tuberculosis"),
    ("26_titanium.txt","Titanium"),
    ("27_lithium.txt","Lithium"),
    ("28_graphene.txt","Graphene"),
    ("29_carbon_fiber.txt","Carbon-fiber-reinforced_polymers"),
    ("30_silicon.txt","Silicon"),
    ("31_boeing_747.txt","Boeing_747"),
    ("32_iss.txt","International_Space_Station"),
    ("33_lhc.txt","Large_Hadron_Collider"),
    ("34_hubble.txt","Hubble_Space_Telescope"),
    ("35_panama_canal.txt","Panama_Canal"),
    ("36_vesuvius.txt","Mount_Vesuvius"),
    ("37_yellowstone.txt","Yellowstone_Caldera"),
    ("38_krakatoa.txt","1883_eruption_of_Krakatoa"),
    ("39_tohoku_eq.txt","2011_T%C5%8Dhoku_earthquake_and_tsunami"),
    ("40_great_barrier.txt","Great_Barrier_Reef"),
]
for name, title in wiki_needle:
    t = wiki(title)
    if t: save("needle", name, t)
    else: print(f"  FAIL {name}")
    time.sleep(0.5)

# ============================================================
print("\n=== CODE (40 texts) ===")

# GitHub TheAlgorithms/Python (20)
print("\n--- GitHub TheAlgorithms ---")
algo = [
    ("01_quick_sort.txt","TheAlgorithms/Python","sorts/quick_sort.py"),
    ("02_merge_sort.txt","TheAlgorithms/Python","sorts/merge_sort.py"),
    ("03_heap_sort.txt","TheAlgorithms/Python","sorts/heap_sort.py"),
    ("04_binary_search.txt","TheAlgorithms/Python","searches/binary_search.py"),
    ("05_linear_search.txt","TheAlgorithms/Python","searches/linear_search.py"),
    ("06_bst.txt","TheAlgorithms/Python","data_structures/binary_search_tree.py"),
    ("07_linked_list.txt","TheAlgorithms/Python","data_structures/linked_list/singly_linked_list.py"),
    ("08_trie.txt","TheAlgorithms/Python","data_structures/trie/trie.py"),
    ("09_heap.txt","TheAlgorithms/Python","data_structures/heap/heap.py"),
    ("10_hash_table.txt","TheAlgorithms/Python","data_structures/hashing/hash_table.py"),
    ("11_dijkstra.txt","TheAlgorithms/Python","graphs/dijkstra_algorithm.py"),
    ("12_bfs.txt","TheAlgorithms/Python","graphs/breadth_first_search.py"),
    ("13_a_star.txt","TheAlgorithms/Python","graphs/a_star.py"),
    ("14_fibonacci.txt","TheAlgorithms/Python","dynamic_programming/fibonacci.py"),
    ("15_knapsack.txt","TheAlgorithms/Python","dynamic_programming/knapsack.py"),
    ("16_prime.txt","TheAlgorithms/Python","maths/prime_check.py"),
    ("17_matrix.txt","TheAlgorithms/Python","matrix/matrix_operation.py"),
    ("18_gcd.txt","TheAlgorithms/Python","maths/greatest_common_divisor.py"),
    ("19_tower_hanoi.txt","TheAlgorithms/Python","other/tower_of_hanoi.py"),
    ("20_lru_cache.txt","TheAlgorithms/Python","other/lru_cache.py"),
]
for name, repo, path in algo:
    t = github(repo, path)
    if t: save("code", name, t, min_words=20)
    else: print(f"  FAIL {name}")
    time.sleep(0.5)

# GitHub other repos (10)
print("\n--- GitHub other repos ---")
other_gh = [
    ("21_flask_app.txt","pallets/flask","src/flask/app.py"),
    ("22_requests_api.txt","psf/requests","src/requests/api.py"),
    ("23_fastapi_routing.txt","fastapi/fastapi","fastapi/routing.py"),
    ("24_httpie_sessions.txt","httpie/cli","httpie/sessions.py"),
    ("25_black_mode.txt","psf/black","src/black/mode.py"),
    ("26_rich_console.txt","Textualize/rich","rich/console.py"),
    ("27_click_core.txt","pallets/click","src/click/core.py"),
    ("28_pydantic_main.txt","pydantic/pydantic","pydantic/main.py"),
    ("29_sqlmodel_main.txt","fastapi/sqlmodel","sqlmodel/main.py"),
    ("30_typer_main.txt","fastapi/typer","typer/main.py"),
]
for name, repo, path in other_gh:
    t = github(repo, path)
    if t: save("code", name, t, min_words=20)
    else: print(f"  FAIL {name}")
    time.sleep(0.5)

# GitHub configs/devops/infra (10)
print("\n--- GitHub configs/infra ---")
infra = [
    ("31_docker_compose.txt","docker/compose","cmd/compose/compose.go"),
    ("32_terraform_main.txt","hashicorp/terraform","internal/command/apply.go"),
    ("33_prometheus_rules.txt","prometheus/prometheus","rules/manager.go"),
    ("34_grafana_api.txt","grafana/grafana","pkg/api/api.go"),
    ("35_kubernetes_pod.txt","kubernetes/kubernetes","pkg/apis/core/types.go"),
    ("36_nginx_conf.txt","nginx/nginx","src/core/nginx.c"),
    ("37_redis_server.txt","redis/redis","src/server.c"),
    ("38_postgres_select.txt","postgres/postgres","src/backend/parser/gram.c"),
    ("39_linux_sched.txt","torvalds/linux","kernel/sched/core.c"),
    ("40_git_commit.txt","git/git","builtin/commit.c"),
]
for name, repo, path in infra:
    t = github(repo, path)
    if t: save("code", name, t, min_words=20)
    else: print(f"  FAIL {name}")
    time.sleep(0.5)

# ============================================================
print("\n=== SUMMARY ===")
for cat in ["discourse","needle","code"]:
    d = BASE/cat
    files = list(d.glob("*.txt"))
    total_words = sum(len(f.read_text(encoding="utf-8").split()) for f in files)
    print(f"  {cat}: {len(files)} files, {total_words} words")
print("\nDone!")
