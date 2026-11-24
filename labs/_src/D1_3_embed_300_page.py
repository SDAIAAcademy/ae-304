# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "faiss-cpu>=1.9"]
# ///
# %% [markdown]
# # Lab 1.3 · Two embedding models: quality, size and latency
#
# Day 1 · Session 3 (Dense retrieval and embedding choice) · 25 min.
#
# A vector search compares meanings, so it may find what BM25 missed in the dialect queries. You embed a
# 300-page sample with two open multilingual models (multilingual-e5-small and
# paraphrase-multilingual-MiniLM-L12-v2), see what e5's prefixes are for, load the supplied full
# indexes, and compare exact search with an approximate (HNSW) index on quality, index size and latency.
#
# **No model on your laptop.** In Colab the two models load and embed the sample live. On a laptop the
# notebook reads the supplied vectors of the same pages (`data/vectors/`, made with the same models) and
# the times measured when they were made. Set `AE304_MODELS=load` only if you accept a download of about
# 470 MB per model.
#
# **Tasks**
# - [ ] 1 · Embed the 300-page sample (6')
# - [ ] 2 · What e5's prefixes are for (4')
# - [ ] 3 · Load the full indexes and search exactly (6')
# - [ ] 4 · An HNSW index: recall against exact, size, latency (6')
# - [ ] 5 · One row per model in the results table (3')
# - [ ] ✱ Stretch · Store the vectors in eight bits

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

import faiss
import numpy as np
import pandas as pd

import nuha as U

PAGES = U.pages()
POS = {p["page_id"]: i for i, p in enumerate(PAGES)}
LABELS = U.labels()
PUBLIC = np.array([LABELS[p["page_id"]][0] == "Public" for p in PAGES])
SAMPLE = json.loads((U.VEC / "sample300.json").read_text())
QUERIES, QRELS = U.testset("v1")
T = U.timings()
print(len(PAGES), "pages,", int(PUBLIC.sum()), "Public | sample of", len(SAMPLE), "| models:", ", ".join(U.MODELS.values()))

# %% [markdown]
# ## 1 · Embed the 300-page sample (6')
#
# e5 was trained with a prefix on each side: `query: ` before a question, `passage: ` before a page.
# MiniLM has none. Vectors are normalised to length 1, so the dot product is the cosine similarity.
#
# ✏️ **Exercise** · `embed_sample(model)` returns the sample's vectors, the seconds it took and where they
# came from. Use `U.live_model(model)` (None on a laptop) and `U.PREFIX[model]`; without a live model,
# take the supplied rows of `U.page_vectors(model)` and the recorded time `T[f"{model}_embed_300_pages_s"]`.

# %%
def embed_sample(model: str):
    # BEGIN SOLUTION
    m = U.live_model(model)
    rows = [POS[i] for i in SAMPLE]
    if m is None:
        return U.page_vectors(model)[rows], T[f"{model}_embed_300_pages_s"], "supplied"
    texts = [U.PREFIX[model][1] + PAGES[r]["text"] for r in rows]
    t = time.perf_counter()
    v = m.encode(texts, normalize_embeddings=True, batch_size=32)
    return v.astype(np.float32), round(time.perf_counter() - t, 1), "live"
    # END SOLUTION


# %%
# Check, with a safety net
emb = {}
for model in U.MODELS:
    try:
        v, secs, how = embed_sample(model)
    except NotImplementedError:
        v, secs, how = U.page_vectors(model)[[POS[i] for i in SAMPLE]], T[f"{model}_embed_300_pages_s"], "supplied (safety net)"
    emb[model] = v
    same = float(np.mean(np.sum(v * U.page_vectors(model)[[POS[i] for i in SAMPLE]], axis=1)))
    print(f"{model}: {v.shape}, {secs} s for 300 pages ({how}); mean cosine with the supplied vectors {same:.3f}")
    U.check(v.shape == (300, 384) and same > 0.99, f"{model}: 300 vectors of 384 numbers that match the supplied index")
print("full corpus at that speed:", {m: f"{T[f'{m}_embed_300_pages_s'] * len(PAGES) / 300 / 60:.1f} min" for m in U.MODELS})

# %% [markdown]
# ## 2 · What e5's prefixes are for (4')
#
# The supplied file `e5_noprefix_pages.npy` holds e5 vectors of every page made **without** `passage: `,
# and `e5_noprefix_queries.npy` the 60 test queries without `query: `. Compare the two set-ups.

# %%
def evaluate_matrix(M, Q):
    run = {}
    for q, qv in zip(QUERIES, Q):
        sims = np.where(PUBLIC, M @ qv, -np.inf)
        run[q["query_id"]] = [PAGES[i]["page_id"] for i in np.argsort(-sims)[:20]]
    return U.evaluate(run, QUERIES, QRELS)


with_prefix = evaluate_matrix(U.page_vectors("e5"), np.stack([U.embed_query(q["text"], "e5") for q in QUERIES]))
without = evaluate_matrix(np.load(U.VEC / "e5_noprefix_pages.npy").astype(np.float32), np.load(U.VEC / "e5_noprefix_queries.npy").astype(np.float32))
print(pd.DataFrame({"with prefixes": with_prefix["nDCG@10"], "without": without["nDCG@10"]}))

# %% [markdown]
# 🔍 The model card asks for the prefixes. A difference here is the price of skipping one line of a model
# card; check the card of any model you adopt for the same kind of instruction.

# %% [markdown]
# ## 3 · Load the full indexes and search exactly (6')
#
# ✏️ **Exercise** · `dense_search(model, text, k)`: embed the query with `U.embed_query`, score every page
# by dot product, keep Public pages only, return the top `k` page ids.

# %%
def dense_search(model: str, text: str, k: int = 20) -> list[str]:
    # BEGIN SOLUTION
    qv = U.embed_query(text, model)
    sims = np.where(PUBLIC, U.page_vectors(model) @ qv, -np.inf)
    return [PAGES[i]["page_id"] for i in np.argsort(-sims)[:k]]
    # END SOLUTION


try:
    runs = {m: {q["query_id"]: dense_search(m, q["text"]) for q in QUERIES} for m in U.MODELS}
except NotImplementedError:
    runs = {m: {q["query_id"]: [i for i, _ in U.dense_query(U.embed_query(q["text"], m), m, 20, PUBLIC)] for q in QUERIES} for m in U.MODELS}
for m in U.MODELS:
    print(m)
    print(U.evaluate(runs[m], QUERIES, QRELS, by="type"))
DIALECT = {m: U.evaluate(runs[m], QUERIES, QRELS, by="type").loc["dialect", "recall@10"] for m in U.MODELS}
print("dialect recall@10:", DIALECT, "| normalised BM25 was 0.25")
U.check(max(DIALECT.values()) > 0.25, "one of the two models finds more dialect answers than normalised BM25 did (0.25)")

# %% [markdown]
# 🔍 The two models do not rank the same way: the smaller-sounding model is not the weaker one on every
# query type. Look at the dialect and course rows before you pick one from a leaderboard.

# %% [markdown]
# **If you finish early · One index, one model.** Both models give 384 numbers, so nothing stops a
# MiniLM query vector from being compared with e5 page vectors: no error, no warning.
#
# ✏️ **Exercise** · Embed the 60 queries with MiniLM, search the e5 page vectors with them
# (`evaluate_matrix`), and compare with e5 queries on the same index.

# %%
try:
    # BEGIN SOLUTION
    crossed = evaluate_matrix(U.page_vectors("e5"), np.stack([U.embed_query(q["text"], "minilm") for q in QUERIES]))
    # END SOLUTION
    print(pd.DataFrame({"e5 queries": with_prefix["nDCG@10"], "MiniLM queries": crossed["nDCG@10"]}))
    U.check(U.page_vectors("e5").shape[1] == U.embed_query(QUERIES[0]["text"], "minilm").shape[0], "the shapes agree, so nothing raised an error")
    U.check(crossed.loc["all", "nDCG@10"] < 0.1, "queries from another model find almost nothing")
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# 🔍 A vector means something only next to vectors from the same model. What must happen to the 2,977
# page vectors on the day Nibras changes its query model, and what should the index store so that a
# mismatch is caught before a query runs?

# %% [markdown]
# ## 4 · An HNSW index: recall against exact, size, latency (6')
#
# Exact search compares the query with every page. An approximate index (HNSW: a graph of neighbours you
# walk from coarse to fine) compares with a few hundred. On 3,000 pages exact search is already fast; the
# question is what you give up when the corpus is a million pages.
#
# ✏️ **Exercise** · Build `faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)` over the Public
# vectors, then measure for each query: the overlap of its top 10 with exact search, and the search time.

# %%
def hnsw_compare(model: str, ef: int = 64):  # efSearch: how many neighbours the walk keeps (32 gave 0.90 for e5)
    rows = np.where(PUBLIC)[0]
    X = np.ascontiguousarray(U.page_vectors(model)[rows])
    # BEGIN SOLUTION
    index = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef
    index.add(X)
    overlap, lat_ann, lat_exact = [], [], []
    for q in QUERIES:
        qv = U.embed_query(q["text"], model)[None, :]
        t = time.perf_counter()
        _, I = index.search(qv, 10)
        lat_ann.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        exact = np.argsort(-(X @ qv[0]))[:10]
        lat_exact.append((time.perf_counter() - t) * 1000)
        overlap.append(len(set(I[0]) & set(exact)) / 10)
    size = len(faiss.serialize_index(index))
    return {"model": model, "overlap@10": round(float(np.mean(overlap)), 3), "hnsw MB": round(size / 1e6, 2),
            "flat MB": round(X.nbytes / 1e6, 2), "hnsw p95 ms": round(U.p95(lat_ann), 3), "exact p95 ms": round(U.p95(lat_exact), 3)}
    # END SOLUTION


try:
    ANN = pd.DataFrame([hnsw_compare(m) for m in U.MODELS])
except NotImplementedError:
    ANN = pd.DataFrame(U.reference("D1_3_hnsw"))
print(ANN.to_string(index=False))
U.save_reference("D1_3_hnsw", ANN.to_dict(orient="records"))
U.check(bool((ANN["overlap@10"] > 0.9).all()), "HNSW keeps more than 90% of the exact top 10")

# %% [markdown]
# ## 5 · One row per model in the results table (3')
#
# Latency the user feels is the query embedding plus the search. The embedding times were measured once
# on a laptop processor (`T`); the search time is yours.

# %%
ROWS = []
for m in U.MODELS:
    e = U.evaluate(runs[m], QUERIES, QRELS)
    ann = ANN[ANN.model == m].iloc[0]
    ROWS.append({"method": f"dense {m}", "nDCG@10 ar": e.loc["ar", "nDCG@10"], "nDCG@10 en": e.loc["en", "nDCG@10"],
                 "recall@10": e.loc["all", "recall@10"], "index MB": ann["flat MB"],
                 "p95 ms": round(T[f"{m}_query_embed_ms_p95"] + ann["exact p95 ms"], 1)})
RESULTS = pd.DataFrame(ROWS)
print(RESULTS.to_string(index=False))
(U.lab_dir("D1_3") / "results.json").write_text(RESULTS.to_json(orient="records"))

# %% [markdown]
# 🔍 Which model would you ship for Nibras, and what would change your mind: a better score on one
# language, or a lower latency?

# %% [markdown]
# ## ✱ Stretch · Store the vectors in eight bits
#
# `faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)` stores
# each number in one byte. Measure the size and the overlap with exact search.

# %%
try:
    # BEGIN SOLUTION
    rows = np.where(PUBLIC)[0]
    X = np.ascontiguousarray(U.page_vectors("e5")[rows])
    sq = faiss.IndexScalarQuantizer(384, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq.train(X)
    sq.add(X)
    ov = []
    for q in QUERIES:
        qv = U.embed_query(q["text"], "e5")
        _, I = sq.search(qv[None, :], 10)
        ov.append(len(set(I[0]) & set(np.argsort(-(X @ qv))[:10])) / 10)
    print(f"8-bit: {len(faiss.serialize_index(sq)) / 1e6:.2f} MB, overlap@10 {np.mean(ov):.3f}")
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. You embed the pages with `passage: ` and forget `query: `. What do you expect?
#    <details><summary>Answer</summary>Lower scores: the query lands where the model puts passages, not questions. The model card states the convention.</details>
# 2. HNSW kept almost all of the exact top 10 here. Why is that not proof for a corpus of a million pages?
#    <details><summary>Answer</summary>Overlap depends on the graph settings and the corpus; measure it again at the real size.</details>
# 3. Which part of the latency does an approximate index not touch?
#    <details><summary>Answer</summary>The query embedding, which is most of it here.</details>
# 4. Nibras moves to a new embedding model. Can the old page vectors stay while new pages get new ones?
#    <details><summary>Answer</summary>No: queries and pages must come from one model. Re-embed the whole corpus into a new index, record the model's name with it, and switch the queries over only when it is complete.</details>
#
# **Carry forward.** `artifacts/D1_3/results.json`: two more rows of the results table.
