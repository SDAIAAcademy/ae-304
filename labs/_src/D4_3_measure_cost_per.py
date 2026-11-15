# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "duckdb>=1.2"]
# ///
# %% [markdown]
# # Lab 4.3 · Cost per question, tail latency and re-index time
#
# Day 4 · Session 3 (Cost, latency and freshness) · 25 min.
#
# Three numbers decide whether Nibras's retrieval layer can run for a whole university: what one question
# costs, how slow the slowest questions are, and how long a new circular takes to become searchable. You
# measure tokens and cost per question for three pipelines, the 95th-percentile latency of each stage of the
# retrieval layer, and the time to re-index after one circular changes, in full and incrementally.
#
# **Reference state.** The corpus, test set v1, the supplied vectors, the Day 4 questions, and
# `U.timings()` (embedding and reranking times measured once on a laptop processor). No model is called:
# cost is computed from token counts and the prices you set.
#
# **Tasks**
# - [ ] 1 · Tokens and cost per question, three pipelines (6')
# - [ ] 2 · Latency by stage, and its tail (7')
# - [ ] 3 · One circular changes: re-index everything (5')
# - [ ] 4 · Re-index only what changed (5')
# - [ ] 5 · One table for the operations review (2')
# - [ ] ✱ Stretch · A latency budget per stage

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

PAGES = U.public_pages()
ALL = U.pages()
POS = {p["page_id"]: i for i, p in enumerate(ALL)}
PUBLIC = np.zeros(len(ALL), dtype=bool)
PUBLIC[[POS[p["page_id"]] for p in PAGES]] = True
QUERIES, QRELS = U.testset("v1")
T = U.timings()
LEX = U.fts_index(PAGES)
# Prices per million tokens: set your provider's. A free model costs 0; the counts stay the same.
PRICE_IN, PRICE_OUT = float(os.environ.get("AE304_PRICE_IN", "0")), float(os.environ.get("AE304_PRICE_OUT", "0"))
print(len(PAGES), "Public pages | prices per million tokens:", PRICE_IN, PRICE_OUT)

# %% [markdown]
# ## 1 · Tokens and cost per question, three pipelines (6')
#
# - **Regulations question**: a system prompt, the question, the top 5 pages, an answer of about 150 tokens.
# - **Free text-to-SQL**: the schema prompt of Lab 4.1, the question, a query of about 80 tokens.
# - **Semantic layer**: the metric catalog of Lab 4.2, the question, a request of about 60 tokens.
#
# ✏️ **Exercise** · `cost_row(name, prompts, out_tokens)`: mean and 95th-percentile input tokens over the
# prompts, output tokens, and the cost per question and per 10,000 questions at `PRICE_IN` and `PRICE_OUT`.

# %%
SYSTEM = "You are Nibras, the regulations assistant of Al-Nuha University. Answer from the pages and cite them."
rag_prompts = [SYSTEM + q["text"] + " ".join(U.page(i)["text"] for i, _ in U.fts_query(LEX, q["text"], 5)) for q in QUERIES]
DB = U.warehouse()
schema = "\n".join(f"TABLE {t} (" + ", ".join(f"{c[0]} {c[1]}" for c in DB.sql(f"DESCRIBE {t}").fetchall()) + ")" for (t,) in DB.sql("SHOW TABLES").fetchall())
catalog = json.dumps(U.semantic_layer(DB), ensure_ascii=False)
sql_qs = [q["q"] for q in U.jsonl("warehouse/questions.jsonl")]


def cost_row(name: str, prompts: list[str], out_tokens: int) -> dict:
    # BEGIN SOLUTION
    toks = [U.count_tokens(p) for p in prompts]
    per_q = (np.mean(toks) * PRICE_IN + out_tokens * PRICE_OUT) / 1e6
    return {"pipeline": name, "input mean": int(np.mean(toks)), "input p95": int(np.percentile(toks, 95)), "output": out_tokens,
            "cost per question": round(per_q, 6), "cost per 10,000": round(per_q * 1e4, 2)}
    # END SOLUTION


try:
    COST = pd.DataFrame([cost_row("regulations, top 5 pages", rag_prompts, 150),
                         cost_row("free text-to-SQL", [schema + q for q in sql_qs], 80),
                         cost_row("semantic layer", [catalog + q for q in sql_qs], 60)])
except NotImplementedError:
    COST = pd.DataFrame(U.reference("D4_3_cost"))
U.save_reference("D4_3_cost", COST.to_dict(orient="records"))
print(COST.to_string(index=False))

# %% [markdown]
# 🔍 The pages dominate the regulations prompt. Halving k halves most of the cost: what does it do to recall
# at 5 (Day 1 numbers)?

# %% [markdown]
# ## 2 · Latency by stage, and its tail (7')
#
# ✏️ **Exercise** · `stage_times(q)` times, for one query, the lexical search, the dense search (matrix
# product over the supplied vectors) and the fusion, in milliseconds. The query embedding and the reranker
# were measured once (`T`): add their 95th percentiles to the total.

# %%
def stage_times(q: dict) -> dict:
    # BEGIN SOLUTION
    t0 = time.perf_counter()
    lex = [i for i, _ in U.fts_query(LEX, q["text"], 30)]
    t1 = time.perf_counter()
    dense = [i for i, _ in U.dense_query(U.embed_query(q["text"], "e5"), "e5", 30, PUBLIC)]
    t2 = time.perf_counter()
    U.rrf([lex, dense])
    t3 = time.perf_counter()
    return {"lexical": (t1 - t0) * 1000, "dense": (t2 - t1) * 1000, "fusion": (t3 - t2) * 1000}
    # END SOLUTION


try:
    LAT = pd.DataFrame([stage_times(q) for _ in range(U.pick(3, 1)) for q in QUERIES])
except NotImplementedError:
    LAT = pd.DataFrame(U.reference("D4_3_latency"))
U.save_reference("D4_3_latency", LAT.round(3).to_dict(orient="records"))
STAGES = pd.DataFrame({"p50 ms": LAT.median(), "p95 ms": LAT.quantile(0.95)}).round(2)
STAGES.loc["query embedding (measured once)"] = [T["e5_query_embed_ms_p50"], T["e5_query_embed_ms_p95"]]
STAGES.loc["rerank 30 (measured once)"] = [T["rerank_30_pairs_ms_p50"], T["rerank_30_pairs_ms_p95"]]
print(STAGES)
print("sum of the stage p95s:", round(STAGES["p95 ms"].sum(), 1), "ms (an upper bound on the end-to-end p95)")

# %% [markdown]
# 🔍 Which stage is the tail? Optimising the fusion code is pointless if the reranker is 90% of the time.

# %% [markdown]
# ## 3 · One circular changes: re-index everything (5')
#
# Circular 2026-060 (the summer term) is replaced by a corrected version that raises the limit from 10 to 12
# credit hours. The simplest pipeline rebuilds the whole index and re-embeds every page.

# %%
changed_id = next(p["page_id"] for p in PAGES if p["path"] == "circulars/2026/2026-060-ar.md")
new_text = U.page(changed_id)["text"].replace("أكثر من 10 ساعات", "أكثر من 12 ساعة")
assert new_text != U.page(changed_id)["text"]
per_page_s = T["e5_embed_300_pages_s"] / 300
t = time.perf_counter()
rows = [dict(p, text=new_text) if p["page_id"] == changed_id else p for p in PAGES]
FULL = U.fts_index(rows)
full_lex_s = time.perf_counter() - t
full = {"strategy": "full rebuild", "lexical s": round(full_lex_s, 2), "embedding s (estimated)": round(per_page_s * len(PAGES), 1)}
print(full)

# %% [markdown]
# ## 4 · Re-index only what changed (5')
#
# ✏️ **Exercise** · `incremental(con, page_id, text)`: delete the page's row from the FTS table and insert
# the new text (analysed), timed. Its vector is now stale: mark it so dense search skips the page until it is
# re-embedded (`PENDING`), rather than serve a vector of the old text. Then test that a search for the new
# rule finds the new text.

# %%
PENDING = np.zeros(len(ALL), dtype=bool)


def incremental(con, page_id: str, text: str) -> float:
    # BEGIN SOLUTION
    t = time.perf_counter()
    con.execute("DELETE FROM idx WHERE id = ?", (page_id,))
    con.execute("INSERT INTO idx VALUES (?, ?)", (page_id, " ".join(U.analyse(text))))
    con.commit()
    PENDING[POS[page_id]] = True
    return time.perf_counter() - t
    # END SOLUTION


try:
    inc_s = incremental(LEX, changed_id, new_text)
except NotImplementedError:
    inc_s = float("nan")
inc = {"strategy": "incremental", "lexical s": round(inc_s, 4), "embedding s (estimated)": round(per_page_s, 2)}
FRESH = pd.DataFrame([full, inc])
print(FRESH.to_string(index=False))
hits = [i for i, _ in U.fts_query(LEX, "الفصل الصيفي 12 ساعة", 3)]
U.check(changed_id in hits and "12 ساعة" in new_text, "the corrected circular is searchable at once")
dense_ok = PUBLIC & ~PENDING
print("dense search now skips", int((PUBLIC & PENDING).sum()), "page until it is re-embedded")

# %% [markdown]
# ## 5 · One table for the operations review (2')

# %%
REVIEW = {"cost": COST.to_dict(orient="records"), "latency_p95_ms": STAGES["p95 ms"].to_dict(), "freshness": FRESH.to_dict(orient="records")}
(U.lab_dir("D4_3") / "operations.json").write_text(json.dumps(REVIEW, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps(REVIEW["freshness"], indent=1))

# %% [markdown]
# ## ✱ Stretch · A latency budget per stage
#
# Nibras's budget for retrieval is 400 ms at p95. Which stages would you cut or move first, and which
# number from Day 1 tells you what that costs in quality?

# %%
try:
    # BEGIN SOLUTION
    budget = 400
    total = STAGES["p95 ms"].sum()
    print(f"total {total:.0f} ms against {budget} ms:", "within budget" if total <= budget else "over budget: shrink the rerank depth first (Lab 1.4 stretch)")
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Why report cost per question rather than price per token?
#    <details><summary>Answer</summary>The design decides how many tokens a question needs; a cheap token in a long prompt can cost more than an expensive one in a short prompt.</details>
# 2. Why is the 95th percentile the number to watch?
#    <details><summary>Answer</summary>One student in twenty waits that long or longer; the average hides them.</details>
# 3. A circular changed and its old vector is still in the index. What can go wrong?
#    <details><summary>Answer</summary>Dense search ranks the page by its old meaning; skip or re-embed changed pages, never serve a stale vector silently.</details>
#
# **Carry forward.** `artifacts/D4_3/operations.json`: cost, latency and freshness, the numbers the capstone README reports.
