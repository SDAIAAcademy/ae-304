# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 1.1 · Graded judgements and a baseline score
#
# Day 1 · Session 1 (Retrieval is measured on its own) · 25 min.
#
# Nibras, the regulations assistant of Al-Nuha University, can only answer from what its retrieval
# layer finds. Before changing any search, you measure it. You judge the relevance of pooled results
# for 40 bilingual queries yourself, compare your grades with the reference judgements of test set v1,
# then score the supplied baseline with recall at 10, reciprocal rank and discounted cumulative gain,
# by language and by query type.
#
# **AI-off.** The judgements and the metric functions are done without an AI assistant.
#
# **Reference state.** Test set v1 (`data/testset/v1/`: 60 queries, graded judgements), the pools of
# 40 of its queries, and the baseline run (`data/runs/baseline.json`). Nothing from another lab.
#
# **Tasks**
# - [ ] 1 · The guideline and the pools (3')
# - [ ] 2 · Judge your 20 queries, in pairs (9')
# - [ ] 3 · Agreement with the reference judgements (4')
# - [ ] 4 · Recall at 10, reciprocal rank, nDCG at 10 (6')
# - [ ] 5 · By language and by query type (3')
# - [ ] ✱ Stretch · Five queries of your own as test set v1.1

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
import math

import pandas as pd

import nuha as U

QUERIES, QRELS = U.testset("v1")
POOLS = U.jsonl("testset/v1/pools.jsonl")
BASELINE = json.loads((U.DATA / "runs" / "baseline.json").read_text(encoding="utf-8"))
print(len(QUERIES), "queries |", sum(len(g) for g in QRELS.values()), "judged pairs |", len(POOLS), "pools |", "baseline:", BASELINE["method"])

# %% [markdown]
# ## 1 · The guideline and the pools (3')
#
# A query is judged against each page in its **pool**: the top results of several systems, so that
# no single system decides what gets judged. The grade answers one question: *would this page let
# Nibras answer the query?*
#
# | Grade | Meaning |
# |---|---|
# | 3 | Answers the query, in the query's language |
# | 2 | Answers it in the other language, or states the current version of the rule |
# | 1 | On the topic, but does not answer |
# | 0 | Not useful |
#
# Pair up. Partner A judges the 20 English queries, partner B the 20 Arabic ones.

# %%
MY_HALF = "A"  # change to "B" for the Arabic half
mine = [p for p in POOLS if (p["lang"] == "en") == (MY_HALF == "A")]


def show(pool):
    print(f"\n{pool['query_id']} [{pool['type']}] {pool['text']}")
    for i, pid in enumerate(pool["candidates"]):
        p = U.page(pid)
        print(f"  {i}. {p['title'][:70]} | {' '.join(p['text'].split())[:200]}")


for pool in mine[:3]:
    show(pool)

# %% [markdown]
# ## 2 · Judge your 20 queries, in pairs (9')
#
# ✏️ **Exercise** · Print each pool with `show(pool)` and write three grades per query, in the order of
# the candidates. Judge from the text, not from the title.

# %%
def my_judgements() -> dict:
    # BEGIN SOLUTION
    return {p["query_id"]: [QRELS[p["query_id"]].get(c, 0) for c in p["candidates"]] for p in mine}
    # END SOLUTION


# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    MINE = my_judgements()
except NotImplementedError:
    MINE = {p["query_id"]: [QRELS[p["query_id"]].get(c, 0) for c in p["candidates"]] for p in mine}
    print("(safety net: the reference grades stand in for yours)")
U.check(len(MINE) == 20 and all(len(v) == 3 for v in MINE.values()), "20 queries, three grades each")

# %% [markdown]
# ## 3 · Agreement with the reference judgements (4')
#
# ✏️ **Exercise** · Compare your grades with the reference grades of test set v1: exact agreement, and
# agreement within one grade. Then list the pairs where you differ by two or more.

# %%
def agreement(mine_: dict) -> dict:
    # BEGIN SOLUTION
    pairs = [(g, QRELS[q].get(c, 0), q, c) for q, gs in mine_.items()
             for g, c in zip(gs, next(p["candidates"] for p in POOLS if p["query_id"] == q))]
    exact = sum(a == b for a, b, *_ in pairs) / len(pairs)
    within = sum(abs(a - b) <= 1 for a, b, *_ in pairs) / len(pairs)
    far = [(q, c, a, b) for a, b, q, c in pairs if abs(a - b) >= 2]
    return {"pairs": len(pairs), "exact": round(exact, 2), "within_one": round(within, 2), "far": far}
    # END SOLUTION


try:
    AG = agreement(MINE)
except NotImplementedError:
    AG = {"pairs": 60, "exact": 1.0, "within_one": 1.0, "far": []}
print({k: v for k, v in AG.items() if k != "far"})
for q, c, a, b in AG["far"][:5]:
    print("differ:", q, c, "you", a, "reference", b)
grades = pd.Series([g for gs in MINE.values() for g in gs]).value_counts().sort_index()
print("your grades:", dict(grades))

# %% [markdown]
# 🔍 Where you disagree by two grades, read the page again. A judgement that two careful people
# cannot agree on is a sign the guideline needs one more sentence, not that one of you is careless.

# %% [markdown]
# ## 4 · Recall at 10, reciprocal rank, nDCG at 10 (6')
#
# A page is **relevant** for recall and reciprocal rank when its grade is 2 or 3. nDCG uses every grade:
# the gain of a page is 2^grade − 1, discounted by log2(rank + 1), and divided by the best possible DCG.
#
# ✏️ **Exercise** · Write the three functions. `ranked` is a list of page ids, `grades` a dict.

# %%
def recall_at(ranked, grades, k=10):
    # BEGIN SOLUTION
    rel = {p for p, g in grades.items() if g >= 2}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0
    # END SOLUTION


def reciprocal_rank(ranked, grades):
    # BEGIN SOLUTION
    return next((1 / i for i, p in enumerate(ranked, 1) if grades.get(p, 0) >= 2), 0.0)
    # END SOLUTION


def ndcg_at(ranked, grades, k=10):
    # BEGIN SOLUTION
    dcg = sum((2 ** grades.get(p, 0) - 1) / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0
    # END SOLUTION


# %%
# Check against the course's reference functions, with a safety net
toy = {"a": 3, "b": 2, "c": 1}
try:
    mine_scores = [recall_at(["x", "b", "a"], toy, 2), reciprocal_rank(["x", "b", "a"], toy), ndcg_at(["x", "b", "a"], toy)]
except NotImplementedError:
    recall_at, reciprocal_rank, ndcg_at = U.recall_at, U.rr, U.ndcg_at
    mine_scores = [recall_at(["x", "b", "a"], toy, 2), reciprocal_rank(["x", "b", "a"], toy), ndcg_at(["x", "b", "a"], toy)]
ref_scores = [U.recall_at(["x", "b", "a"], toy, 2), U.rr(["x", "b", "a"], toy), U.ndcg_at(["x", "b", "a"], toy)]
print("toy:", [round(x, 3) for x in mine_scores])
U.check(all(abs(a - b) < 1e-9 for a, b in zip(mine_scores, ref_scores)), "your metrics match the reference on a toy ranking")

rows = []
for q in QUERIES:
    r, g = BASELINE["run"][q["query_id"]], QRELS[q["query_id"]]
    rows.append({"query_id": q["query_id"], "lang": q["lang"], "type": q["type"],
                 "recall@10": recall_at(r, g), "RR": reciprocal_rank(r, g), "nDCG@10": ndcg_at(r, g)})
SCORES = pd.DataFrame(rows)
print(SCORES[["recall@10", "RR", "nDCG@10"]].mean().round(3).to_dict())

# %% [markdown]
# ## 5 · By language and by query type (3')
#
# One average hides the users it fails. Report the baseline by language and by query type: this table
# is the first row of the results table you keep all week.

# %%
by_lang = SCORES.groupby("lang")[["recall@10", "RR", "nDCG@10"]].mean().round(3)
by_type = SCORES.groupby("type")[["recall@10", "RR", "nDCG@10"]].mean().round(3)
print(by_lang, "\n")
print(by_type)
out = U.lab_dir("D1_1") / "baseline_report.json"
out.write_text(json.dumps({"test_set": "v1", "method": BASELINE["method"], "by_lang": by_lang.to_dict(), "by_type": by_type.to_dict()}, indent=1))
print("saved", out.relative_to(U.LABS))
U.check(by_lang.loc["en", "nDCG@10"] > by_lang.loc["ar", "nDCG@10"], "the baseline serves English queries better than Arabic ones")

# %% [markdown]
# 🔍 Which query type is worst? Read two of its queries and the baseline's top three. Is the problem the
# words (spelling, dialect) or the meaning?

# %% [markdown]
# ## ✱ Stretch · Five queries of your own as test set v1.1
#
# Write five queries a student would ask, judge the baseline's top five for each, and save them as
# `testset/v1.1`. Never change v1 in place: a score is only comparable on the same version.

# %%
try:
    # BEGIN SOLUTION
    new = [{"query_id": "N01", "lang": "en", "type": "regulation", "text": "can I take a course at another university"}]
    new_qrels = {"N01": {pid: 0 for pid in BASELINE["run"]["Q01"][:5]}}
    v11 = U.lab_dir("D1_1") / "testset_v1.1.json"
    v11.write_text(json.dumps({"version": "v1.1", "based_on": "v1", "queries": new, "qrels": new_qrels}, ensure_ascii=False, indent=1))
    print("saved", v11.relative_to(U.LABS))
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is the pool built from several systems and not from the baseline alone?
#    <details><summary>Answer</summary>Pages that only a better system finds would never be judged, and that system would be scored as if it returned junk.</details>
# 2. A method raises recall at 10 but lowers reciprocal rank. What changed for the user?
#    <details><summary>Answer</summary>More relevant pages are somewhere in the top ten, but the first relevant one appears lower: the assistant reads less useful text first.</details>
# 3. Why report Arabic and English separately?
#    <details><summary>Answer</summary>An index tuned on English can look fine on average while failing every Arabic user.</details>
#
# **Carry forward.** `artifacts/D1_1/baseline_report.json` is the first row of your results table. Every
# later lab scores on the same test set v1.
