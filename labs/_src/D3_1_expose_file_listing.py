# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "openai>=1.60", "python-dotenv>=1.0"]
# ///
# %% [markdown]
# # Lab 3.1 · Three search tools and a traced agent
#
# Day 3 · Session 1 (Search as tools for an agent) · 25 min.
#
# Instead of one search before every answer, Nibras can choose: list the files of a folder (cheap, names
# only), search by keyword, or search by meaning. You write the three tools with descriptions that tell a
# model when to use each, run 15 questions through a small agent loop, and read in the traces which tool
# it picked, how many calls it made and whether it answered.
#
# **The model.** With `OPENROUTER_API_KEY` set, a free hosted model chooses the tools. Without a key (or
# with `AE304_STANDIN=1`) a scripted stand-in (`nibras_tools.standin_pick`) follows the same loop: it picks
# the cheapest tool that fits, reads one result and answers. The notebook says which one ran.
#
# **Reference state.** The Public corpus, `data/agent/tool_questions.jsonl` (15 questions, the tool that
# fits best, the strings a correct answer contains). Nothing from Days 1 and 2 is needed.
#
# **Tasks**
# - [ ] 1 · The file tree, and listing it (5')
# - [ ] 2 · Keyword and vector search as tools (7')
# - [ ] 3 · Fifteen questions through the loop (6')
# - [ ] 4 · Read the traces: choice, calls, answer (7')
# - [ ] ✱ Stretch · A misleading description

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

import pandas as pd

import nuha as U
import nibras_tools as T

QS = U.jsonl("agent/tool_questions.jsonl")
print(U.model_mode(), "|", len(QS), "questions |", len(T.BY_PATH), "Public files")
SYSTEM = ("You are Nibras, the regulations assistant of Al-Nuha University. Answer from the tools only. "
          "Prefer the cheapest tool that can answer. Quote the numbers you found.")

# %% [markdown]
# ## 1 · The file tree, and listing it (5')
#
# Every page has a path: `regulations/en/article-17.md`, `circulars/2025/2025-118-ar.md`,
# `study-plans/eng/ce/year-2-en.md`, `guide/ar/housing.md`. "How many circulars in 2025?" needs no search
# at all, only a listing.
#
# ✏️ **Exercise** · `list_files(folder)`: the sorted paths under the folder (case-insensitive), at most
# 400, or a message that names the top folders when there is none. Its docstring is what the model reads.

# %%
def list_files(folder: str) -> list[str]:
    """List the files under a folder of the corpus, such as "circulars/2025", "study-plans/eng" or "guide/ar".
    Cheapest tool: names only, no text. Use it for questions about what exists or how many."""
    # BEGIN SOLUTION
    folder = folder.strip("/").lower()
    names = sorted(p for p in T.BY_PATH if p.lower().startswith(folder + "/"))
    return names[:400] if names else [f"no files under '{folder}'; top folders: " + ", ".join(sorted({p.split('/')[0] for p in T.BY_PATH}))]
    # END SOLUTION


try:
    print(list_files("circulars/2025")[:4], "...")
except NotImplementedError:
    list_files = T.list_files
    print("(safety net: the reference list_files)")
U.check(len(list_files("study-plans/eng")) == 40, "five Engineering plans, four years, two languages")

# %% [markdown]
# ## 2 · Keyword and vector search as tools (7')
#
# A tool's description is its user manual: say what it is good at, with an example of the input.
#
# ✏️ **Exercise** · Write both tools. Each returns at most `k` hits as `{"path", "title", "text"}` with the
# text cut to 400 characters. Use `U.fts_query(T._lex(), ...)` for keywords and `U.embed_query` with
# `U.dense_query(..., T.ALLOWED)` for meaning; when no query vector exists, fall back to keywords and say so.

# %%
def keyword_search(query: str, k: int = 5) -> list[dict]:
    """Keyword search (BM25, Arabic normalised). Best for exact words: course codes such as "CSC 301",
    circular numbers such as "2025-118", article numbers and names. English or Arabic."""
    # BEGIN SOLUTION
    return [T._hit(U.page(i)) for i, _ in U.fts_query(T._lex(), query, k)]
    # END SOLUTION


def vector_search(query: str, k: int = 5) -> list[dict]:
    """Search by meaning (multilingual embeddings). Best for questions in everyday words, paraphrases,
    dialect, or a question in one language whose answer may be in the other."""
    # BEGIN SOLUTION
    qv = U.embed_query(query, "e5")
    if qv is None:
        return [{"note": "no query vector on this machine: keyword results instead"}] + keyword_search(query, k)
    return [T._hit(U.page(i)) for i, _ in U.dense_query(qv, "e5", k, T.ALLOWED)]
    # END SOLUTION


try:
    print(keyword_search("CSC 301", 1)[0]["title"], "|", vector_search("Can I use a chatbot to help with my homework?", 1)[0]["title"])
except NotImplementedError:
    keyword_search, vector_search = T.keyword_search, T.vector_search
    print("(safety net: the reference search tools)")
print(json.dumps(U.tool_schema(keyword_search), indent=1)[:400])

# %% [markdown]
# ## 3 · Fifteen questions through the loop (6')
#
# `U.run_agent(question, tools, standin, ...)` sends the question and the three tool definitions to the
# model, runs the tool it asks for, sends back the result, and stops when it answers or after `max_steps`.
#
# ✏️ **Exercise** · Run all 15 questions and keep, for each: the first tool chosen, the number of calls,
# the estimated tokens sent, and whether the answer is correct (`U.graded(answer, expect)`).

# %%
def run_all() -> pd.DataFrame:
    rows = []
    for q in QS:
        # BEGIN SOLUTION
        r = U.run_agent(q["q"], [list_files, keyword_search, vector_search], T.standin_pick, system=SYSTEM, max_steps=4)
        rows.append({"qid": q["qid"], "best": q["best"], "first": r["calls"][0][0] if r["calls"] else "none",
                     "calls": len(r["calls"]), "tokens": r["tokens"], "correct": U.graded(r["answer"], q["expect"]), "model": r["model"],
                     "trace": " > ".join(f"{n}({json.dumps(a, ensure_ascii=False)[:40]})" for n, a in r["calls"])})
        # END SOLUTION
    return pd.DataFrame(rows)


try:
    TRACES = run_all()
except NotImplementedError:
    TRACES = pd.DataFrame(U.reference("D3_1_traces"))
U.save_reference("D3_1_traces", TRACES.to_dict(orient="records"))
print(TRACES[["qid", "best", "first", "calls", "tokens", "correct"]].to_string(index=False))
TRACES.to_json(U.lab_dir("D3_1") / "traces.json", orient="records", force_ascii=False)

# %% [markdown]
# ## 4 · Read the traces: choice, calls, answer (7')

# %%
summary = TRACES.assign(right_tool=TRACES.best == TRACES["first"]).groupby("best")[["right_tool", "correct", "calls", "tokens"]].mean().round(2)
print(summary)
print("\naccuracy:", round(TRACES.correct.mean(), 2), "| mean calls:", round(TRACES.calls.mean(), 2), "| mean tokens:", int(TRACES.tokens.mean()))
for t in TRACES[~TRACES.correct].itertuples():
    print("failed:", t.qid, t.trace)
U.check(TRACES.correct.mean() >= 0.6, "at least 9 of 15 questions answered")

# %% [markdown]
# 🔍 For each failure, was the wrong tool chosen, or the right tool given the wrong query, or did the answer
# ignore what the tool returned? Those are three different fixes: the description, the query, the prompt.

# %% [markdown]
# ## ✱ Stretch · A misleading description
#
# Swap the descriptions of the two search tools and run the five paraphrase questions again with the live
# model. A stand-in that never reads descriptions will not change: that is why the trace of a real model
# matters.

# %%
try:
    # BEGIN SOLUTION
    def vector_search_misleading(query: str, k: int = 5) -> list[dict]:
        """Keyword search. Best for exact course codes and circular numbers."""
        return vector_search(query, k)


    para = [q for q in QS if q["best"] == "vector_search"]
    for q in para:
        r = U.run_agent(q["q"], [list_files, keyword_search, vector_search_misleading], T.standin_pick, system=SYSTEM, max_steps=4)
        print(q["qid"], [c[0] for c in r["calls"]], U.graded(r["answer"], q["expect"]), r["model"])
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. "How many circulars in 2025?" Which tool should run first, and why?
#    <details><summary>Answer</summary>The listing: names are enough to count, and they cost a fraction of a search result.</details>
# 2. The agent always calls the first tool in its list. What do you change?
#    <details><summary>Answer</summary>The descriptions: say what each tool is for with an example input; then check the trace again.</details>
# 3. Why keep the trace of every run?
#    <details><summary>Answer</summary>The answer alone does not say whether the tool choice, the query or the reading failed.</details>
#
# **Carry forward.** `artifacts/D3_1/traces.json`: the first traces of the LO5 evidence.
