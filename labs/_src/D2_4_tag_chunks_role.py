# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 2.4 · Role labels and leak tests
#
# Day 2 · Session 4 (Permissions in the index) · 25 min.
#
# The full corpus holds council minutes, allowance scales, examination archives and disciplinary case
# summaries, classified Restricted. Muath Al-Shathri, a second-year student, must never receive one, not
# even as a snippet. You tag every page with its document's classification and roles, filter before
# ranking, and write a leak suite that runs as Muath, plus a utility check that the filter did not hide
# what students may see.
#
# **Reference state.** `data/corpus/` (pages and documents with their labels), `data/access/users.json`
# and 12 leak queries. The answer key is not in the notebook: the tests decide.
#
# **Tasks**
# - [ ] 1 · Search as Muath with no filter (4')
# - [ ] 2 · Tag every page from its document (6')
# - [ ] 3 · Filter before ranking, not after (6')
# - [ ] 4 · The leak suite and the utility check (6')
# - [ ] 5 · Retrieved text is untrusted (3')
# - [ ] ✱ Stretch · The registrar's view: no over-blocking

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

import numpy as np
import pandas as pd

import nuha as U

PAGES = U.pages()
DOCS = U.docs()
USERS = json.loads((U.DATA / "access" / "users.json").read_text(encoding="utf-8"))
LEAK = U.jsonl("access/leak_tests.jsonl")
QUERIES, QRELS = U.testset("v1")
LEX = U.fts_index(PAGES)          # one index over everything, as Day 1 never had to decide
POS = {p["page_id"]: i for i, p in enumerate(PAGES)}
TRUTH = U.labels()                # used only to score, never to filter
print(len(PAGES), "pages |", len(DOCS), "documents |", sum(v[0] == "Restricted" for v in TRUTH.values()), "Restricted pages")


def hybrid(query, k=10, allowed=None):
    lex = [i for i, _ in U.fts_query(LEX, query, 50) if allowed is None or allowed[POS[i]]]
    qv = U.embed_query(query, "e5")
    dense = [i for i, _ in U.dense_query(qv, "e5", 50, allowed)] if qv is not None else []
    return U.rrf([lex, dense])[:k]


def restricted_in(ids):
    return [i for i in ids if TRUTH[i][0] == "Restricted"]


# %% [markdown]
# ## 1 · Search as Muath with no filter (4')

# %%
leaks = {t["tid"]: restricted_in(hybrid(t["q"])) for t in LEAK}
print(f"{sum(map(len, leaks.values()))} Restricted results across {len(LEAK)} queries")
first = next(i for v in leaks.values() for i in v)
print(U.page(first)["title"], "|", U.page(first)["text"][:160])

# %% [markdown]
# ## 2 · Tag every page from its document (6')
#
# Labels live on documents (`classification`, `roles`), set by their owners under the national data
# classification policy. A page takes its document's label. Some documents are attachments with no label
# of their own, only a `parent`.
#
# ✏️ **Exercise** · `tag(page)` returns `(classification, roles)`: the document's label, else the parent's
# (follow the chain), else **deny**: `("Restricted", ["registrar"])`. Never default to Public.

# %%
def tag(page: dict) -> tuple[str, list[str]]:
    # BEGIN SOLUTION
    d = DOCS[page["doc_id"]]
    while d.get("classification") is None and d.get("parent"):
        d = DOCS[d["parent"]]
    if d.get("classification") is None:
        return ("Restricted", ["registrar"])
    return (d["classification"], d["roles"])
    # END SOLUTION


try:
    TAGS = {p["page_id"]: tag(p) for p in PAGES}
except NotImplementedError:
    TAGS = U.labels()
naive = {p["page_id"]: (DOCS[p["doc_id"]]["classification"] or "Public") for p in PAGES}
print("pages a naive tagger calls Public but are Restricted:", sum(naive[i] == "Public" and TRUTH[i][0] == "Restricted" for i in naive))
U.check(all(TAGS[i][0] == TRUTH[i][0] for i in TAGS), "every page carries its document's classification")

# %% [markdown]
# ## 3 · Filter before ranking, not after (6')
#
# Filtering after ranking takes the top 10 and removes what Muath may not see: he gets fewer than 10
# results, and anything computed before the filter (a count, a snippet, a cache) has already seen
# Restricted text. Filtering before ranking searches only what he may see.
#
# ✏️ **Exercise** · `allowed_for(user)` returns a boolean array over `PAGES`: True when the page is Public
# or when one of its roles is among the user's roles.

# %%
def allowed_for(user: dict) -> np.ndarray:
    # BEGIN SOLUTION
    return np.array([TAGS[p["page_id"]][0] == "Public" or bool(set(TAGS[p["page_id"]][1]) & set(user["roles"])) for p in PAGES])
    # END SOLUTION


try:
    MUATH = allowed_for(USERS["muath"])
except NotImplementedError:
    MUATH = np.array([TRUTH[p["page_id"]][0] == "Public" for p in PAGES])
post = {t["tid"]: [i for i in hybrid(t["q"]) if MUATH[POS[i]]] for t in LEAK}
pre = {t["tid"]: hybrid(t["q"], allowed=MUATH) for t in LEAK}
print("results per query, filter after ranking:", [len(v) for v in post.values()])
print("results per query, filter before ranking:", [len(v) for v in pre.values()])

# %% [markdown]
# ## 4 · The leak suite and the utility check (6')
#
# ✏️ **Exercise** · Two tests. `test_no_leaks()`: for every leak query, run as Muath, no Restricted page in
# the top 10. `test_utility()`: on the 60 queries of test set v1, recall at 10 as Muath equals recall at 10
# on the Public index of Day 1 (the filter hides nothing a student may see). Both return a list of failures.

# %%
def test_no_leaks() -> list[str]:
    # BEGIN SOLUTION
    return [f"{t['tid']}: {restricted_in(hybrid(t['q'], allowed=MUATH))}" for t in LEAK if restricted_in(hybrid(t["q"], allowed=MUATH))]
    # END SOLUTION


def test_utility() -> list[str]:
    # BEGIN SOLUTION
    public = np.array([TRUTH[p["page_id"]][0] == "Public" for p in PAGES])
    fails = []
    for q in QUERIES:
        a = U.recall_at(hybrid(q["text"], allowed=MUATH), QRELS[q["query_id"]])
        b = U.recall_at(hybrid(q["text"], allowed=public), QRELS[q["query_id"]])
        if a < b:
            fails.append(f"{q['query_id']}: {a:.2f} < {b:.2f}")
    return fails
    # END SOLUTION


try:
    LEAKS, UTIL = test_no_leaks(), test_utility()
except NotImplementedError:
    LEAKS, UTIL = ["not written yet"], ["not written yet"]
print("leak failures:", LEAKS or "none")
print("utility failures:", UTIL or "none")
(U.lab_dir("D2_4") / "access_tests.json").write_text(json.dumps({"leak_failures": LEAKS, "utility_failures": UTIL}, indent=1))
U.check(not LEAKS and not UTIL, "no leaks, and nothing a student may see is hidden")

# %% [markdown]
# ## 5 · Retrieved text is untrusted (3')
#
# Circular 2026-777 is Public, and it talks to "automated assistants". Retrieval will hand it to the model
# like any other page. The filter you wrote is code: no page can talk its way past it. Mark instruction-like
# text before it reaches a prompt.
#
# ✏️ **Exercise** · `flag(text)` returns True when a retrieved page addresses an assistant or asks to
# ignore rules (English or Arabic).

# %%
def flag(text: str) -> bool:
    # BEGIN SOLUTION
    return bool(re.search(r"(automated assistants?|ignore (any|all|the|previous)|تجاهل|للمساعد)", text, re.I))
    # END SOLUTION


hit = hybrid("student satisfaction survey", allowed=MUATH)
try:
    flagged = [i for i in hit if flag(U.page(i)["text"])]
except NotImplementedError:
    flagged = []
print("flagged in the results:", [(i, U.page(i)["title"]) for i in flagged])
U.check(any(U.page(i)["doc_id"] == "CIR-2026-777" for i in flagged), "the planted circular is flagged, and it still could not unlock anything")

# %% [markdown]
# ## ✱ Stretch · The registrar's view: no over-blocking
#
# Run the 12 leak queries as the registrar. A filter that also blocks the registrar is broken the other way.

# %%
try:
    # BEGIN SOLUTION
    REG = allowed_for(USERS["registrar"])
    seen = sum(len(restricted_in(hybrid(t["q"], allowed=REG))) for t in LEAK)
    print("Restricted pages the registrar sees for the 12 queries:", seen)
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Your filter removes Restricted pages from the top 10. Why can a leak still happen?
#    <details><summary>Answer</summary>Anything computed before the filter (counts, snippets, a cache, a reranker's input) has seen them, and unlabelled pages pass a filter that only removes what is labelled.</details>
# 2. Why deny an unlabelled page instead of treating it as Public?
#    <details><summary>Answer</summary>A missing label is a gap in the data, not a decision; the attachment of a Restricted document inherits its secrecy.</details>
# 3. A page in the index says "ignore the access rules". What protects Muath?
#    <details><summary>Answer</summary>The filter runs in code before ranking; the model never sees what it may not show.</details>
#
# **Carry forward.** `artifacts/D2_4/access_tests.json` and the staleness tests of Lab 2.3: the LO4 suite.
