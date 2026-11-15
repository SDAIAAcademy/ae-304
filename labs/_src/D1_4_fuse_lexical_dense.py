# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 1.4 · Fusion, a reranker and its latency
#
# Day 1 · Session 4 (Hybrid search and reranking) · 25 min.
#
# BM25 is good at codes and exact words, dense search at meaning and dialect. You fuse the two rankings,
# rerank the top 30 with a multilingual cross-encoder, and decide whether the gain is worth the latency.
# The result is the Day 1 results table: ranking metrics by language and 95th-percentile latency per method.
#
# **Reference state.** The corpus, test set v1, the supplied e5 vectors and the supplied cross-encoder
# scores (`data/reference/rerank_scores.json`, made with mmarco-mMiniLMv2-L12-H384-v1 on a laptop
# processor). No model is loaded here.
#
# **Tasks**
# - [ ] 1 · Reciprocal rank fusion (7')
# - [ ] 2 · Score fusion and the problem of scales (5')
# - [ ] 3 · Rerank the fused top 30 (6')
# - [ ] 4 · The gain against the added latency (7')
# - [ ] ✱ Stretch · The rerank depth that fits a latency budget

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
import time

import numpy as np
import pandas as pd

import nuha as U

PAGES = U.pages()
PUBLIC = np.array([U.labels()[p["page_id"]][0] == "Public" for p in PAGES])
QUERIES, QRELS = U.testset("v1")
CE = U.rerank_scores()
T = U.timings()
LEX = U.fts_index(U.public_pages())


def timed(fn):
    out, lat = {}, []
    for q in QUERIES:
        t = time.perf_counter()
        out[q["query_id"]] = fn(q)
        lat.append((time.perf_counter() - t) * 1000)
    return out, lat


bm25, lat_bm25 = timed(lambda q: [i for i, _ in U.fts_query(LEX, q["text"], 100)])
dense, lat_dense = timed(lambda q: [i for i, _ in U.dense_query(U.embed_query(q["text"], "e5"), "e5", 100, PUBLIC)])
bm25_scores = {q["query_id"]: dict(U.fts_query(LEX, q["text"], 100)) for q in QUERIES}
dense_scores = {q["query_id"]: dict(U.dense_query(U.embed_query(q["text"], "e5"), "e5", 100, PUBLIC)) for q in QUERIES}
print("first stages ready: BM25 normalised and dense e5, top 100 each")

# %% [markdown]
# ## 1 · Reciprocal rank fusion (7')
#
# RRF ignores the scores and uses the ranks: a page gets 1 / (k + rank) from each list it appears in, and
# k = 60 damps the difference between rank 1 and rank 2. Cormack, Clarke and Büttcher proposed it in 2009.
#
# ✏️ **Exercise** · Write `rrf(lists, k=60)` and fuse BM25 with dense e5 for every query.

# %%
def rrf(lists: list[list[str]], k: int = 60) -> list[str]:
    # BEGIN SOLUTION
    score = {}
    for ranked in lists:
        for rank, pid in enumerate(ranked, 1):
            score[pid] = score.get(pid, 0.0) + 1 / (k + rank)
    return sorted(score, key=lambda p: -score[p])
    # END SOLUTION


try:
    fused, lat_fuse = timed(lambda q: rrf([bm25[q["query_id"]], dense[q["query_id"]]]))
except NotImplementedError:
    fused, lat_fuse = timed(lambda q: U.rrf([bm25[q["query_id"]], dense[q["query_id"]]]))
for name, run in (("BM25", bm25), ("dense e5", dense), ("RRF", fused)):
    print(name, U.evaluate(run, QUERIES, QRELS).loc["all"].to_dict())
U.check(U.evaluate(fused, QUERIES, QRELS).loc["all", "nDCG@10"] >= max(U.evaluate(r, QUERIES, QRELS).loc["all", "nDCG@10"] for r in (bm25, dense)) - 0.02,
        "fusion is at least as good as the better of its two inputs (within 0.02)")

# %% [markdown]
# ## 2 · Score fusion and the problem of scales (5')
#
# Adding the raw scores is tempting. BM25 scores run into the tens; cosine similarities sit between about
# 0.7 and 0.9. Look at the ranges, then min-max both to [0, 1] before mixing them with a weight.
#
# ✏️ **Exercise** · `weighted(qid, alpha)`: min-max each score list of the query, then
# alpha × dense + (1 − alpha) × BM25, a page missing from a list scoring 0 there.

# %%
b = np.concatenate([list(s.values()) for s in bm25_scores.values()])
d = np.concatenate([list(s.values()) for s in dense_scores.values()])
print(f"BM25 scores {b.min():.1f} to {b.max():.1f} | cosine {d.min():.2f} to {d.max():.2f}")


def weighted(qid: str, alpha: float = 0.5) -> list[str]:
    # BEGIN SOLUTION
    def mm(s):
        lo, hi = min(s.values()), max(s.values())
        return {k: (v - lo) / (hi - lo or 1) for k, v in s.items()}
    bs, ds = mm(bm25_scores[qid]), mm(dense_scores[qid])
    comb = {p: alpha * ds.get(p, 0) + (1 - alpha) * bs.get(p, 0) for p in set(bs) | set(ds)}
    return sorted(comb, key=lambda p: -comb[p])
    # END SOLUTION


try:
    for a in (0.3, 0.5, 0.7):
        print("alpha", a, U.evaluate({q["query_id"]: weighted(q["query_id"], a) for q in QUERIES}, QUERIES, QRELS).loc["all"].to_dict())
except NotImplementedError:
    print("(skipped: weighted() not written yet)")

# %% [markdown]
# 🔍 A weight tuned on these 60 queries is tuned on your test set. RRF has no weight to overfit.

# %% [markdown]
# ## 3 · Rerank the fused top 30 (6')
#
# A cross-encoder reads the query and the page together and scores the pair. It is far better at judging
# than a vector comparison and far slower, so it only sees the top of the fused list.
#
# ✏️ **Exercise** · `rerank(qid, depth=30)`: reorder the first `depth` fused pages by `CE[qid][page]`, keep
# the rest in fused order behind them. A page without a supplied score keeps its fused position.

# %%
def rerank(qid: str, depth: int = 30) -> list[str]:
    # BEGIN SOLUTION
    head = fused[qid][:depth]
    scored = sorted([p for p in head if p in CE[qid]], key=lambda p: -CE[qid][p])
    unscored = [p for p in head if p not in CE[qid]]
    return scored + unscored + fused[qid][depth:]
    # END SOLUTION


try:
    reranked = {q["query_id"]: rerank(q["query_id"]) for q in QUERIES}
except NotImplementedError:
    reranked = {q: [p for p in sorted(fused[q][:30], key=lambda p: -CE[q].get(p, -99))] + fused[q][30:] for q in fused}
cover = np.mean([sum(p in CE[q] for p in fused[q][:30]) / 30 for q in fused])
print(f"supplied scores cover {cover:.0%} of the fused top 30")
print(U.evaluate(reranked, QUERIES, QRELS))

# %% [markdown]
# ## 4 · The gain against the added latency (7')
#
# ✏️ **Exercise** · Complete the results table: one row per method with nDCG at 10 by language, recall at
# 10, reciprocal rank and the 95th-percentile latency. Dense methods pay the query embedding
# (`T["e5_query_embed_ms_p95"]`); the reranker pays `T["rerank_30_pairs_ms_p95"]` on top of fusion.

# %%
def row(name, run, p95_ms):
    e = U.evaluate(run, QUERIES, QRELS)
    return {"method": name, "nDCG@10 ar": e.loc["ar", "nDCG@10"], "nDCG@10 en": e.loc["en", "nDCG@10"],
            "recall@10": e.loc["all", "recall@10"], "MRR": e.loc["all", "MRR"], "p95 ms": round(p95_ms, 1)}


def results_table():
    # BEGIN SOLUTION
    emb = T["e5_query_embed_ms_p95"]
    fuse_ms = U.p95([a + b + c for a, b, c in zip(lat_bm25, lat_dense, lat_fuse)]) + emb
    return pd.DataFrame([
        row("BM25 normalised", bm25, U.p95(lat_bm25)),
        row("dense e5", dense, U.p95(lat_dense) + emb),
        row("hybrid RRF", fused, fuse_ms),
        row("hybrid RRF + rerank 30", reranked, fuse_ms + T["rerank_30_pairs_ms_p95"]),
    ])
    # END SOLUTION


try:
    RESULTS = results_table()
except NotImplementedError:
    RESULTS = pd.DataFrame(U.reference("D1_4_results"))
print(RESULTS.to_string(index=False))
U.save_reference("D1_4_results", RESULTS.to_dict(orient="records"))
(U.lab_dir("D1_4") / "results.json").write_text(RESULTS.to_json(orient="records"))
gain = RESULTS.iloc[3]["nDCG@10 ar"] - RESULTS.iloc[2]["nDCG@10 ar"]
print(f"reranking adds {gain:+.3f} nDCG@10 on Arabic queries for {T['rerank_30_pairs_ms_p95']:.0f} ms more at p95")
U.check(len(RESULTS) == 4, "four methods in the results table")

# %% [markdown]
# 🔍 Nibras answers in a chat; the model's reply takes seconds. Is the reranker's cost visible to a student?
# Would it be in an autocomplete box?

# %% [markdown]
# ## ✱ Stretch · The rerank depth that fits a latency budget
#
# Assume the reranker's time grows with the number of pairs (30 pairs took the measured time). Which
# depth fits a 300 ms budget for the whole query, and what does it cost in nDCG?

# %%
try:
    # BEGIN SOLUTION
    per_pair = T["rerank_30_pairs_ms_p95"] / 30
    base_ms = RESULTS.iloc[2]["p95 ms"]
    for depth in (5, 10, 20, 30):
        run = {q["query_id"]: rerank(q["query_id"], depth) for q in QUERIES}
        print(depth, round(base_ms + per_pair * depth), "ms", U.evaluate(run, QUERIES, QRELS).loc["all", "nDCG@10"])
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Why can you not add a BM25 score to a cosine similarity?
#    <details><summary>Answer</summary>They are on different scales; the larger one decides the order alone.</details>
# 2. The reranker only sees the fused top 30. What does it do for a relevant page at rank 45?
#    <details><summary>Answer</summary>Nothing: reranking reorders candidates, it cannot recover what the first stage missed.</details>
# 3. When would you ship hybrid search without the reranker?
#    <details><summary>Answer</summary>When the latency budget is tight and the gain on your test set is small, or when the reranker is weaker on one language.</details>
#
# **Carry forward.** `artifacts/D1_4/results.json` is the Day 1 results table (LO2 evidence).
