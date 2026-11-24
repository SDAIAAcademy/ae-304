# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 1.2 · BM25 with and without Arabic normalisation
#
# Day 1 · Session 2 (Lexical search and Arabic text) · 25 min.
#
# Students write "اخر موعد", the regulation says "آخر موعد"; they write "جامعه", the circular says
# "جامعة". You index the Public pages of the corpus twice with SQLite full-text search (FTS5), once on raw
# text and once after your own normalisation and light stemming, and compare recall by language and on
# the ten dialect queries.
#
# **Reference state.** The corpus (`data/corpus/`) and test set v1. Nothing from Lab 1.1: the metrics come
# from `nuha`.
#
# **Tasks**
# - [ ] 1 · FTS5 on raw text: the baseline again (4')
# - [ ] 2 · Write the normaliser (7')
# - [ ] 3 · Light stemming and a second index (6')
# - [ ] 4 · Compare by language and on dialect queries (5')
# - [ ] 5 · Read three dialect failures (3')
# - [ ] ✱ Stretch · Give the title its own column and weight

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ae-304.git"
    if not Path("/content/ae-304").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ae-304"], check=True)
    os.chdir("/content/ae-304/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("nuha.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import json
import re
import sqlite3
import time
import unicodedata

import pandas as pd

import nuha as U

PAGES = U.public_pages()
QUERIES, QRELS = U.testset("v1")
print(len(PAGES), "Public pages |", len(QUERIES), "queries")

# %% [markdown]
# ## 1 · FTS5 on raw text: the baseline again (4')
#
# FTS5 keeps an inverted index (word to pages) and ranks with `bm25()`: rare words weigh more, repeated
# words saturate, long pages are penalised. Its default tokeniser splits on letters and digits and
# lower-cases Latin text. It does nothing for Arabic spelling.

# %%
def build(rows, analyse):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE idx USING fts5(id UNINDEXED, body)")
    con.executemany("INSERT INTO idx VALUES (?, ?)", [(r["page_id"], " ".join(analyse(r["text"]))) for r in rows])
    return con


def search(con, analyse, query, k=10):
    words = [w for w in dict.fromkeys(analyse(query)) if w]
    if not words:
        return []
    match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
    return [i for (i,) in con.execute("SELECT id FROM idx WHERE idx MATCH ? ORDER BY bm25(idx) LIMIT ?", (match, k))]


def raw_words(text):
    return re.findall(r"\w+", text.lower())


t = time.perf_counter()
RAW = build(PAGES, raw_words)
print(f"raw index built in {time.perf_counter() - t:.1f} s")
run_raw = {q["query_id"]: search(RAW, raw_words, q["text"], 20) for q in QUERIES}
print(U.evaluate(run_raw, QUERIES, QRELS))

# %% [markdown]
# ## 2 · Write the normaliser (7')
#
# ✏️ **Exercise** · `normalise(text)`: fold presentation forms with Unicode NFKC, remove the diacritics
# and the tatweel (U+064B to U+0652, U+0670, U+0640), write أ إ آ ٱ as ا, ى as ي, ة as ه, ؤ as و, ئ as ي,
# and lower-case. Normalise the pages and the query the same way: one side alone makes it worse.

# %%
def normalise(text: str) -> str:
    # BEGIN SOLUTION
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("[\u064b-\u0652\u0670\u0640]", "", text)
    text = re.sub("[\u0623\u0625\u0622\u0671]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text.lower()
    # END SOLUTION


# %%
# Check, with a safety net
try:
    sample = normalise("الإختبارُ النهائيّ للجامعة الأُولى")
except NotImplementedError:
    normalise = U.normalise
    sample = normalise("الإختبارُ النهائيّ للجامعة الأُولى")
print(sample)
U.check(sample == U.normalise("الإختبارُ النهائيّ للجامعة الأُولى"), "normalise() matches the reference")

# %% [markdown]
# ## 3 · Light stemming and a second index (6')
#
# Arabic attaches the article and conjunctions to the word: "بالاختبارات" holds ب + ال + اختبار + ات.
# Light stemming removes one common prefix and one common suffix and keeps at least three letters. It is
# crude on purpose: it never needs a dictionary, and it errs on the side of keeping the word.
#
# ✏️ **Exercise** · `analyse(text)`: normalise, split into words, and apply `U.light_stem` to each.
# Build the second index with it.

# %%
def analyse(text: str) -> list[str]:
    # BEGIN SOLUTION
    return [U.light_stem(w) for w in re.findall(r"\w+", normalise(text))]
    # END SOLUTION


try:
    print(analyse("وبالاختبارات النهائية للطالبات"))
except NotImplementedError:
    analyse = U.analyse
    print(analyse("وبالاختبارات النهائية للطالبات"))
t = time.perf_counter()
NORM = build(PAGES, analyse)
print(f"normalised index built in {time.perf_counter() - t:.1f} s")
run_norm = {q["query_id"]: search(NORM, analyse, q["text"], 20) for q in QUERIES}
U.check(len(run_norm) == 60, "60 queries searched")

# %% [markdown]
# ## 4 · Compare by language and on dialect queries (5')

# %%
def table(run, name):
    by_lang = U.evaluate(run, QUERIES, QRELS)
    by_type = U.evaluate(run, QUERIES, QRELS, by="type")
    return {"method": name, "recall@10 ar": by_lang.loc["ar", "recall@10"], "recall@10 en": by_lang.loc["en", "recall@10"],
            "nDCG@10 all": by_lang.loc["all", "nDCG@10"], "recall@10 dialect": by_type.loc["dialect", "recall@10"]}


RESULTS = pd.DataFrame([table(run_raw, "BM25 raw"), table(run_norm, "BM25 normalised + light stem")])
print(RESULTS.to_string(index=False))
lat = []
for q in QUERIES:
    t = time.perf_counter()
    search(NORM, analyse, q["text"], 10)
    lat.append((time.perf_counter() - t) * 1000)
print(f"normalised BM25 latency: p50 {sorted(lat)[30]:.2f} ms, p95 {U.p95(lat):.2f} ms")
U.check(RESULTS.iloc[1]["recall@10 ar"] > RESULTS.iloc[0]["recall@10 ar"], "normalisation raises Arabic recall")
(U.lab_dir("D1_2") / "results.json").write_text(RESULTS.to_json(orient="records"))

# %% [markdown]
# ## 5 · Read three dialect failures (3')
#
# Normalisation fixes spelling. It cannot turn "ابي انسحب من ماده" into "الانسحاب من المقرر": different
# words for the same meaning. Read three dialect queries that still fail.

# %%
for q in [q for q in QUERIES if q["type"] == "dialect"]:
    r, g = run_norm[q["query_id"]], QRELS[q["query_id"]]
    if U.recall_at(r, g) == 0:
        top = U.page(r[0])["title"] if r else "nothing"
        print(f"{q['query_id']} {q['text']}  ->  {top}")

# %% [markdown]
# 🔍 For each failure, which word in the query has no match in the answering page? Session 3 asks
# whether a vector search, which compares meanings, finds these.

# %% [markdown]
# **If you finish early · Rewrite with the corpus's words.** A query word that occurs in no Public page
# can match nothing, however well it is spelled. Count, for every analysed word, how many Public pages
# hold it (`DF`). The dialect words with a count of 0 are the gap.
#
# ✏️ **Exercise** · Fill `REWRITES`: for each unseen dialect word of the failing queries, the words the
# answering article itself uses (read its title with `U.page`). Keep a rewrite only if every word it adds
# occurs in the index. `expand` keeps the student's words and adds the rewrite; nothing is replaced.

# %%
from collections import Counter

DF = Counter()
for p in PAGES:
    DF.update(set(analyse(p["text"])))
DIALECT_Q = [q for q in QUERIES if q["type"] == "dialect"]
for q in DIALECT_Q:
    print(q["query_id"], "unseen:", [w for w in analyse(q["text"]) if DF[w] == 0])

try:
    # BEGIN SOLUTION
    REWRITES = {"اغيب": "الغياب المواظبة", "انحرم": "الحرمان", "انسحب": "الانسحاب", "احول": "التحويل",
                "عدت": "إعادة دراسة", "اوقف": "تأجيل", "اعترض": "التظلم"}
    # END SOLUTION
except NotImplementedError:
    REWRITES = {}


def expand(text: str) -> str:
    return " ".join([text] + [REWRITES[w] for w in text.split() if w in REWRITES])


if REWRITES:
    U.check(all(DF[w] > 0 for r in REWRITES.values() for w in analyse(r)), "every rewrite adds only words the index holds")
    before = [U.recall_at(run_norm[q["query_id"]], QRELS[q["query_id"]]) for q in DIALECT_Q]
    after = [U.recall_at(search(NORM, analyse, expand(q["text"]), 20), QRELS[q["query_id"]]) for q in DIALECT_Q]
    print(f"dialect recall@10: {sum(before) / len(before):.3f} as typed, {sum(after) / len(after):.3f} with the rewrites")
    U.check(sum(after) > sum(before), "rewriting into the corpus's words raises dialect recall")
else:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# 🔍 A rewrite into a word the index never saw changes nothing, and it fails silently. Where would the
# rewrites come from in production: search logs, a model that proposes them, a person? Whoever proposes
# them, the index's own vocabulary decides which to keep.

# %% [markdown]
# ## ✱ Stretch · Give the title its own column and weight
#
# FTS5's `bm25()` takes one weight per column. Index the title and the body as two columns and weight
# the title 3 to 1. Does nDCG move?

# %%
try:
    # BEGIN SOLUTION
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE idx USING fts5(id UNINDEXED, title, body)")
    con.executemany("INSERT INTO idx VALUES (?, ?, ?)", [(p["page_id"], " ".join(analyse(p["title"])), " ".join(analyse(p["text"]))) for p in PAGES])


    def search_w(query, k=20):
        match = " OR ".join('"' + w + '"' for w in dict.fromkeys(analyse(query)) if w)
        return [i for (i,) in con.execute("SELECT id FROM idx WHERE idx MATCH ? ORDER BY bm25(idx, 0.0, 3.0, 1.0) LIMIT ?", (match, k))]


    print(U.evaluate({q["query_id"]: search_w(q["text"]) for q in QUERIES}, QUERIES, QRELS).loc["all"].to_dict())
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. You normalise the pages but not the queries. What happens to "إختبار" in a query?
#    <details><summary>Answer</summary>It keeps its hamza and no longer matches the index, which holds "اختبار": recall falls below the raw index.</details>
# 2. Why does normalisation leave English results unchanged here?
#    <details><summary>Answer</summary>English is already lower-cased by the tokeniser and has no letter variants; only the plural "s" is stripped.</details>
# 3. Which dialect query still fails, and why will no normaliser fix it?
#    <details><summary>Answer</summary>Queries such as "ابي انسحب من ماده" use other words for the same meaning; that needs other words, not other spellings: a rewrite into the words the corpus uses, or a model of meaning.</details>
#
# **Carry forward.** `artifacts/D1_2/results.json`: two rows of the results table.
