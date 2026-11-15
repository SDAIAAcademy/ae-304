# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 2.3 · Effective dates, versions and duplicates
#
# Day 2 · Session 3 (Dates, versions and duplicates) · 25 min.
#
# Fifteen articles of the study regulation were amended by later circulars, most of them only in Arabic,
# some copied again for the colleges, and one approved but not yet in force. You read the effective dates
# and the amended article from the text, give every chunk a validity interval, drop near-duplicate
# copies, and turn the 15 changed questions into staleness tests that must pass.
#
# **Reference state.** `data/versions/chunks.jsonl`: the 60 regulation pages, the amending circulars and
# their copies, as the parser gave them. `changed_questions.jsonl`: 15 questions, the words of the current
# answer and of the stale one. "Today" in the case study is 1 November 2026.
#
# **Tasks**
# - [ ] 1 · Fifteen questions, answered from stale text (3')
# - [ ] 2 · Read effective dates and amended articles from the text (7')
# - [ ] 3 · A validity interval for every version (5')
# - [ ] 4 · Near-duplicates by shingles (5')
# - [ ] 5 · Staleness tests (5')
# - [ ] ✱ Stretch · A question about the past

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

import re

import numpy as np
import pandas as pd

import nuha as U

CHUNKS = U.jsonl("versions/chunks.jsonl")
QS = U.jsonl("versions/changed_questions.jsonl")
POS = {p["page_id"]: i for i, p in enumerate(U.pages())}
print(len(CHUNKS), "chunks |", len(QS), "changed questions | today:", U.TODAY)


def search(chunks, queries, k=5):
    """Hybrid search over a chunk set: BM25 and e5 for each query (one, or the question in both
    languages), all rankings fused by RRF."""
    queries = [queries] if isinstance(queries, str) else queries
    con = U.fts_index(chunks)
    allowed = np.zeros(len(POS), dtype=bool)
    allowed[[POS[c["chunk_id"]] for c in chunks]] = True
    runs = []
    for q in queries:
        runs.append([i for i, _ in U.fts_query(con, q, 20)])
        qv = U.embed_query(q, "e5")
        if qv is not None:
            runs.append([i for i, _ in U.dense_query(qv, "e5", 20, allowed)])
    by_id = {c["chunk_id"]: c for c in chunks}
    return [by_id[i] for i in U.rrf(runs)[:k]]


def current(text, q):
    return q["expect"] in text or q["expect_ar"] in text


# %% [markdown]
# ## 1 · Fifteen questions, answered from stale text (3')

# %%
before = sum(current(search(CHUNKS, q["q"])[0]["text"], q) for q in QS)
print(f"top result states the current rule for {before} of {len(QS)} questions")
print(search(CHUNKS, QS[0]["q"])[0]["text"][:300])

# %% [markdown]
# ## 2 · Read effective dates and amended articles from the text (7')
#
# The regulation says "effective from 2022-09-01" (يُعمل بهذه اللائحة اعتبارًا من). A circular says which
# article it amends (تعديل المادة 17, amend Article 17) and from when (اعتبارًا من, effective from).
#
# ✏️ **Exercise** · `metadata(chunk)` returns `{"article": int, "effective_from": "YYYY-MM-DD",
# "kind": "regulation" or "amendment"}` read from the text in either language.

# %%
def metadata(chunk: dict) -> dict:
    # BEGIN SOLUTION
    t = chunk["text"]
    eff = re.search(r"(?:effective from|اعتبارًا من)\s+(\d{4}-\d{2}-\d{2})", t)
    amend = re.search(r"(?:amend Article|amendment of Article|تعديل المادة)\s+(\d+)", t)
    if amend:
        return {"article": int(amend.group(1)), "effective_from": eff.group(1), "kind": "amendment"}
    art = re.search(r"(?:Article|المادة)\s+(\d+)", t)
    return {"article": int(art.group(1)), "effective_from": eff.group(1), "kind": "regulation"}
    # END SOLUTION


# %%
# Check, with a safety net
try:
    META = {c["chunk_id"]: metadata(c) for c in CHUNKS}
except NotImplementedError:
    META = U.reference("D2_3_metadata")
U.save_reference("D2_3_metadata", META)
print(pd.DataFrame(META.values()).groupby("kind").size())
U.check(sum(m["kind"] == "amendment" for m in META.values()) == len(CHUNKS) - 60, "every circular is read as an amendment")

# %% [markdown]
# ## 3 · A validity interval for every version (5')
#
# A version of an article is valid from its effective date until the next version's effective date. A
# version whose effective date is after `as_of` is not valid yet, even if it is the newest document.
#
# ✏️ **Exercise** · `valid_on(as_of)` returns the set of chunk ids valid on that date: for each article,
# the version with the latest `effective_from` that is not after `as_of` (all its chunks, both languages).

# %%
def valid_on(as_of: str) -> set[str]:
    # BEGIN SOLUTION
    best = {}
    for cid, m in META.items():
        if m["effective_from"] <= as_of:
            a = m["article"]
            if a not in best or m["effective_from"] > best[a]:
                best[a] = m["effective_from"]
    return {cid for cid, m in META.items() if best.get(m["article"]) == m["effective_from"]}
    # END SOLUTION


try:
    VALID = valid_on(U.TODAY)
except NotImplementedError:
    VALID = set(U.reference("D2_3_valid"))
U.save_reference("D2_3_valid", sorted(VALID))
print(len(VALID), "of", len(CHUNKS), "chunks valid on", U.TODAY)

# %% [markdown]
# ## 4 · Near-duplicates by shingles (5')
#
# The copies circulated to the colleges repeat the text with a new number and a "copied to" line. Their
# word 5-gram sets (shingles) overlap almost completely. Keep the earliest document, drop the copies:
# they push other results out of the top 5.
#
# ✏️ **Exercise** · `jaccard(a, b)` on 5-word shingles of the normalised text, then `dedupe(ids)` keeps,
# among chunks of the same language whose Jaccard is 0.5 or more, only the first in id order.

# %%
def shingles(text, n=5):
    w = U.analyse(text)
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def jaccard(a: str, b: str) -> float:
    # BEGIN SOLUTION
    A, B = shingles(a), shingles(b)
    return len(A & B) / len(A | B) if A | B else 0.0
    # END SOLUTION


def dedupe(ids: set[str]) -> set[str]:
    # BEGIN SOLUTION
    by_id = {c["chunk_id"]: c for c in CHUNKS}
    keep = []
    for cid in sorted(ids):
        if not any(by_id[k]["lang"] == by_id[cid]["lang"] and jaccard(by_id[k]["text"], by_id[cid]["text"]) >= 0.5 for k in keep):
            keep.append(cid)
    return set(keep)
    # END SOLUTION


try:
    CLEAN = dedupe(VALID)
except NotImplementedError:
    CLEAN = set(U.reference("D2_3_clean"))
U.save_reference("D2_3_clean", sorted(CLEAN))
print(len(VALID) - len(CLEAN), "near-duplicate copies dropped")

# %% [markdown]
# ## 5 · Staleness tests (5')
#
# Five of the amendments exist only in Arabic. An English question alone ranks English pages first, so
# Nibras asks every question in both languages (`q` and `q_ar`) and fuses the rankings.
#
# ✏️ **Exercise** · `test_staleness(ids)` runs every changed question, in both languages, against the valid,
# deduplicated chunks and checks that the top result states the current rule and that none of the top 5
# states the stale one. Return the failures instead of stopping at the first.

# %%
def test_staleness(ids: set[str]) -> list[str]:
    # BEGIN SOLUTION
    pool = [c for c in CHUNKS if c["chunk_id"] in ids]
    failures = []
    for q in QS:
        top = search(pool, [q["q"], q["q_ar"]])
        if not current(top[0]["text"], q):
            failures.append(f"{q['qid']} top result is not the current rule")
        if any(q["stale"] in c["text"] or q["stale_ar"] in c["text"] for c in top):
            failures.append(f"{q['qid']} a stale version is in the top 5")
    return failures
    # END SOLUTION


try:
    FAIL = test_staleness(CLEAN)
except NotImplementedError:
    FAIL = ["not written yet"]
print("failures:", FAIL or "none")
after = sum(current(search([c for c in CHUNKS if c["chunk_id"] in CLEAN], [q["q"], q["q_ar"]])[0]["text"], q) for q in QS)
print(f"current rule on top: {before} of 15 before, {after} of 15 after")
U.check(not FAIL, "all 15 staleness tests pass")
note = next(q for q in QS if q["note"])
print(f"{note['qid']} ({note['article']}): the newest circular is {note['note']}; the answer stays '{note['expect']}'")

# %% [markdown]
# 🔍 Which rule was announced but is not in force yet? A search that ranks the newest document first would
# have given the wrong answer for two months.

# %% [markdown]
# ## ✱ Stretch · A question about the past
#
# A student appeals a decision taken in March 2024. Which attendance rule applied then? Run the same
# question with `as_of = "2024-03-01"`.

# %%
try:
    # BEGIN SOLUTION
    past = dedupe(valid_on("2024-03-01"))
    q = QS[0]
    top = search([c for c in CHUNKS if c["chunk_id"] in past], [q["q"], q["q_ar"]])[0]["text"]
    print("2024-03-01:", "the 25% rule" if q["stale"] in top or q["stale_ar"] in top else "the 20% rule", "|", top[:160])
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Why not keep only the newest document of each article?
#    <details><summary>Answer</summary>The newest may not be in force yet, and appeals ask about the rule that applied at a date.</details>
# 2. Why compare duplicates within one language only?
#    <details><summary>Answer</summary>A translation shares no shingles with its original; it is not a duplicate to drop, it serves the other language.</details>
# 3. Where do these tests run?
#    <details><summary>Answer</summary>On every re-index, in continuous integration: a new circular can make one fail.</details>
#
# **Carry forward.** The staleness tests are half of the LO4 evidence; Lab 2.4 writes the other half.
