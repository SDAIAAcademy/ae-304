# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 2.2 · Two chunkings of the study plans
#
# Day 2 · Session 2 (Chunking by structure) · 25 min.
#
# A study plan is headings and tables. "In which semester is MTH 301?" needs the row of MTH 301 **and**
# the heading above its table in the same chunk. You chunk the 30 parsed study plans twice, by fixed
# windows of words and by headings with the heading path carried into each chunk, and compare how often
# the retrieved chunks can answer 20 table questions.
#
# **Reference state.** The parsed study plans in `data/plans/*.md` (Markdown, as a layout-aware parser
# gives them) and 20 questions with the strings an answer needs (`data/plans/table_questions.jsonl`).
#
# **Tasks**
# - [ ] 1 · One plan, as the parser gave it (3')
# - [ ] 2 · Fixed windows of 120 words (5')
# - [ ] 3 · Chunks by heading, with the heading path (8')
# - [ ] 4 · Retrieve and score on 20 table questions (6')
# - [ ] 5 · The recommendation (3')
# - [ ] ✱ Stretch · Window size against answerability

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

import pandas as pd

import nuha as U

PLANS = {f.stem.upper(): f.read_text(encoding="utf-8") for f in sorted((U.DATA / "plans").glob("*.md"))}
QS = U.jsonl("plans/table_questions.jsonl")
print(len(PLANS), "plans |", len(QS), "questions | e.g.", QS[0]["q"])

# %% [markdown]
# ## 1 · One plan, as the parser gave it (3')

# %%
print("\n".join(PLANS["CS"].splitlines()[:22]))

# %% [markdown]
# ## 2 · Fixed windows of 120 words (5')
#
# ✏️ **Exercise** · `windows(plan_id, text, size=120, overlap=20)`: split the text on whitespace and cut
# windows of `size` words that start every `size − overlap` words. Return dicts with `chunk_id`, `plan`
# and `text`, keeping the original line breaks out (join with spaces).

# %%
def windows(plan_id: str, text: str, size: int = 120, overlap: int = 20) -> list[dict]:
    # BEGIN SOLUTION
    w = text.split()
    step = size - overlap
    return [{"chunk_id": f"{plan_id}-w{i // step:03d}", "plan": plan_id, "text": " ".join(w[i:i + size])}
            for i in range(0, max(1, len(w) - overlap), step)]
    # END SOLUTION


try:
    FIXED = [c for pid, t in PLANS.items() for c in windows(pid, t)]
except NotImplementedError:
    FIXED = [{"chunk_id": f"{pid}-w{i:03d}", "plan": pid, "text": " ".join(t.split()[i * 100:i * 100 + 120])}
             for pid, t in PLANS.items() for i in range(len(t.split()) // 100 + 1)]
print(len(FIXED), "fixed windows | first:", FIXED[1]["text"][:160])

# %% [markdown]
# ## 3 · Chunks by heading, with the heading path (8')
#
# A chunk per `###` section (one semester and its table), never cut inside a table, and with its context
# written into the chunk: the plan title and the `##` heading above it. A chunk that says "Total | 15"
# without saying which plan and which semester answers nothing.
#
# ✏️ **Exercise** · `by_heading(plan_id, text)`: walk the lines, remember the current `#` and `##`
# headings, start a new chunk at each `###`, and begin each chunk's text with
# "<title> > <year> > <semester>".

# %%
def by_heading(plan_id: str, text: str) -> list[dict]:
    # BEGIN SOLUTION
    title = year = ""
    chunks, cur = [], None
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            year = line[3:].strip()
        elif line.startswith("### "):
            cur = {"chunk_id": f"{plan_id}-h{len(chunks):02d}", "plan": plan_id, "lines": [f"{title} > {year} > {line[4:].strip()}"]}
            chunks.append(cur)
        elif cur is not None and line.strip():
            cur["lines"].append(line)
    return [{"chunk_id": c["chunk_id"], "plan": c["plan"], "text": "\n".join(c["lines"])} for c in chunks]
    # END SOLUTION


try:
    HEAD = [c for pid, t in PLANS.items() for c in by_heading(pid, t)]
except NotImplementedError:
    HEAD = U.reference("D2_2_heading_chunks")
U.save_reference("D2_2_heading_chunks", HEAD)
print(len(HEAD), "heading chunks | first:\n" + HEAD[0]["text"][:300])
U.check(len(HEAD) == 8 * len(PLANS), "eight semester chunks per plan")

# %% [markdown]
# ## 4 · Retrieve and score on 20 table questions (6')
#
# Both chunk sets go into the same BM25 index type. A question counts as **answerable** when one of the
# top 3 chunks holds every string the answer needs (the semester heading and the row, or the total).
#
# ✏️ **Exercise** · `answerable_at3(chunks)`: index, search each question, return the share answerable.

# %%
def answerable_at3(chunks: list[dict]) -> float:
    # BEGIN SOLUTION
    con = U.fts_index(chunks)
    by_id = {c["chunk_id"]: c["text"] for c in chunks}
    ok = 0
    for q in QS:
        top = [by_id[i] for i, _ in U.fts_query(con, q["q"], 3)]
        ok += any(all(n in t for n in q["needs"]) for t in top)
    return ok / len(QS)
    # END SOLUTION


try:
    COMPARE = pd.DataFrame([{"chunking": "fixed 120 words", "chunks": len(FIXED), "answerable@3": answerable_at3(FIXED)},
                            {"chunking": "by heading + path", "chunks": len(HEAD), "answerable@3": answerable_at3(HEAD)}])
except NotImplementedError:
    COMPARE = pd.DataFrame(U.reference("D2_2_compare"))
print(COMPARE.to_string(index=False))
U.save_reference("D2_2_compare", COMPARE.to_dict(orient="records"))
U.check(COMPARE.iloc[1]["answerable@3"] > COMPARE.iloc[0]["answerable@3"], "heading chunks answer more table questions")

# %% [markdown]
# 🔍 Take one question that fails with fixed windows and print its top window. Is the row there? Is the
# heading?

# %%
con = U.fts_index(FIXED)
by_id = {c["chunk_id"]: c["text"] for c in FIXED}
for q in QS[:6]:
    top = [by_id[i] for i, _ in U.fts_query(con, q["q"], 3)]
    if not any(all(n in t for n in q["needs"]) for t in top):
        print(q["q"], "\n   needs", q["needs"], "\n   top window:", top[0][:200] if top else "none")
        break

# %% [markdown]
# ## 5 · The recommendation (3')
#
# ✏️ **Exercise** · Write the recommendation for Nibras's index in two sentences: which chunking, and the
# condition under which you would revisit it. Save it with the numbers.

# %%
try:
    # BEGIN SOLUTION
    RECOMMENDATION = ("Chunk study plans by semester heading with the plan title and year carried into each chunk, and never cut "
                      "inside a table. Revisit if plans gain long prose sections, where windows of a fixed size may retrieve better.")
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")
try:
    RECOMMENDATION
except NameError:
    RECOMMENDATION = "not written yet"
(U.lab_dir("D2_2") / "chunking.json").write_text(json.dumps({"results": COMPARE.to_dict(orient="records"), "recommendation": RECOMMENDATION}, indent=1))
print(RECOMMENDATION)

# %% [markdown]
# ## ✱ Stretch · Window size against answerability
#
# Try windows of 60, 120 and 240 words. Larger windows hold the heading more often; what do they cost?

# %%
try:
    # BEGIN SOLUTION
    for size in (60, 120, 240):
        ch = [c for pid, t in PLANS.items() for c in windows(pid, t, size, size // 6)]
        print(size, "words:", len(ch), "chunks, answerable@3", round(answerable_at3(ch), 2))
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. A window holds the row "MTH 301 | 3 | MTH 212" but not the heading. What can Nibras say?
#    <details><summary>Answer</summary>The credit hours and the prerequisite, not the semester: the fact that answers the question is in another chunk.</details>
# 2. Why write the plan title into every chunk instead of keeping it as metadata only?
#    <details><summary>Answer</summary>The ranker and the model read text; a chunk that names its plan matches "the Mathematics plan" in a query.</details>
# 3. When would fixed windows be the better choice?
#    <details><summary>Answer</summary>Long prose with no reliable headings, such as minutes or news.</details>
#
# **Carry forward.** `artifacts/D2_2/chunking.json`: the retrieval comparison and the recommendation (LO3).
