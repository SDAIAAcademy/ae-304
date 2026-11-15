# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 4.4 · Twenty failures, attributed by component
#
# Day 4 · Session 4 (Diagnosing failures by component) · 25 min.
#
# Students reported 20 wrong answers from Nibras. "The model is bad" is not a diagnosis. Each failure is
# attributed to the first component that lost the answer: parsing (the fact is not in the parsed text),
# retrieval (the page is not among the candidates), ranking (it is a candidate but not in the top 5) or
# generation (it was in the top 5 and the answer still missed it). Then you fix the weakest component and
# measure again.
#
# **Reference state.** `data/diagnosis/failures.jsonl`: for each failure the question, the page that answers
# it, the words the answer needs, the parsed text that was indexed, the 50 candidates, the top 5 and the
# answer. These traces are synthetic, written by `prepare_data.py`; the re-parse and the re-run are real.
#
# **Tasks**
# - [ ] 1 · The attribution ladder (4')
# - [ ] 2 · Attribute each failure (8')
# - [ ] 3 · Count by component (3')
# - [ ] 4 · Fix the weakest component and run again (6')
# - [ ] 5 · Before and after (4')
# - [ ] ✱ Stretch · The second-weakest component

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
import unicodedata

import numpy as np
import pandas as pd

import nuha as U

FAILS = U.jsonl("diagnosis/failures.jsonl")
PUBLIC_PAGES = U.public_pages()
print(len(FAILS), "failures | e.g.", FAILS[0]["q"])

# %% [markdown]
# ## 1 · The attribution ladder (4')
#
# Walk down the pipeline and stop at the first component that lost the answer. The order matters: a page
# that was never parsed correctly cannot be retrieved, so blaming retrieval would send you to the wrong fix.
#
# | Step | Question | If no |
# |---|---|---|
# | 1 | Are the needed words in the parsed text? | parsing |
# | 2 | Is the answering page among the 50 candidates? | retrieval |
# | 3 | Is it in the top 5 the model read? | ranking |
# | 4 | Does the answer contain the needed words? | generation |

# %%
f = FAILS[0]
print({k: (v[:120] if isinstance(v, str) else v[:5] if isinstance(v, list) else v) for k, v in f.items()})

# %% [markdown]
# ## 2 · Attribute each failure (8')
#
# ✏️ **Exercise** · `attribute(f)` returns `"parsing"`, `"retrieval"`, `"ranking"`, `"generation"` or
# `"none"` (no component lost it). Compare text after NFKC and `U.normalise`, as a parser's presentation
# forms would otherwise hide a match.

# %%
def contains(text: str, phrase: str) -> bool:
    return U.normalise(unicodedata.normalize("NFKC", phrase)) in U.normalise(unicodedata.normalize("NFKC", text))


def attribute(f: dict) -> str:
    # BEGIN SOLUTION
    if not contains(f["parsed_text"], f["gold_phrase"]):
        return "parsing"
    if f["gold_page"] not in f["candidates"]:
        return "retrieval"
    if f["gold_page"] not in f["top5"]:
        return "ranking"
    if not contains(f["answer"], f["gold_phrase"]):
        return "generation"
    return "none"
    # END SOLUTION


try:
    BEFORE = {f["fid"]: attribute(f) for f in FAILS}
except NotImplementedError:
    BEFORE = json.loads((U.DATA / "diagnosis" / "planted_components.json").read_text())
print(BEFORE)

# %% [markdown]
# ## 3 · Count by component (3')

# %%
COUNTS = pd.Series(BEFORE).value_counts()
print(COUNTS)
planted = json.loads((U.DATA / "diagnosis" / "planted_components.json").read_text())
U.check(BEFORE == planted, "your attribution matches the component each failure was built to fail in")
WEAKEST = COUNTS.index[0]
print("weakest component:", WEAKEST)

# %% [markdown]
# ## 4 · Fix the weakest component and run again (6')
#
# The parsing failures are pages that went through the text-layer parser: empty scans and reversed Arabic.
# The fix is the routing rule of Lab 2.1: re-parse them with the layout-aware parser. Here the re-parse
# gives the reference text of the page (`U.page(gold)["text"]`).
#
# ✏️ **Exercise** · `fix_parsing(f)` returns a new failure record: the re-parsed text, and the candidates
# and top 5 from a real search of the corrected Public index (`FIXED_INDEX`) for the question. The answer is
# unchanged: no model is re-run here.

# %%
FIXED_INDEX = U.fts_index(PUBLIC_PAGES)


def fix_parsing(f: dict) -> dict:
    # BEGIN SOLUTION
    hits = [i for i, _ in U.fts_query(FIXED_INDEX, f["q"], 50)]
    return dict(f, parsed_text=U.page(f["gold_page"])["text"], candidates=hits, top5=hits[:5])
    # END SOLUTION


try:
    FIXED = [fix_parsing(f) if BEFORE[f["fid"]] == "parsing" else f for f in FAILS]
    AFTER = {f["fid"]: attribute(f) for f in FIXED}
except NotImplementedError:
    FIXED = FAILS
    AFTER = U.reference("D4_4_after")
U.save_reference("D4_4_after", AFTER)
moved = {fid: (BEFORE[fid], AFTER[fid]) for fid in BEFORE if BEFORE[fid] != AFTER[fid]}
print("moved:", moved)

# %% [markdown]
# ## 5 · Before and after (4')

# %%
TABLE = pd.DataFrame({"before": pd.Series(BEFORE).value_counts(), "after": pd.Series(AFTER).value_counts()}).fillna(0).astype(int)
print(TABLE)
(U.lab_dir("D4_4") / "attribution.json").write_text(json.dumps({"before": BEFORE, "after": AFTER, "table": TABLE.to_dict()}, indent=1))
U.check(TABLE.loc["parsing", "after"] == 0 if "parsing" in TABLE.index else True, "no failure is left in parsing after the re-parse")

# %% [markdown]
# 🔍 The fixed parsing failures did not all become passes: some moved down the ladder to ranking or
# generation. That is progress, and it is the reason to attribute again after every fix.

# %% [markdown]
# ## ✱ Stretch · The second-weakest component
#
# Take the ranking failures and run their questions through the Day 1 hybrid search (BM25 and dense e5, RRF).
# Where does the answering page rank now?

# %%
try:
    # BEGIN SOLUTION
    pages = U.pages()
    allowed = np.array([U.labels()[p["page_id"]][0] == "Public" for p in pages])
    for f in FIXED:
        if AFTER[f["fid"]] == "ranking":
            lex = [i for i, _ in U.fts_query(FIXED_INDEX, f["q"], 50)]
            qv = U.embed_query(f["q"], "e5")
            dense = [i for i, _ in U.dense_query(qv, "e5", 50, allowed)] if qv is not None else []
            fused = U.rrf([lex, dense])
            print(f["fid"], "rank in hybrid:", fused.index(f["gold_page"]) + 1 if f["gold_page"] in fused else "not found")
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. A failure's page is not in the candidates, and its parsed text is empty. Which component do you fix?
#    <details><summary>Answer</summary>Parsing: retrieval cannot find text that was never extracted.</details>
# 2. Why attribute again after a fix?
#    <details><summary>Answer</summary>A fixed failure can move to the next component down instead of passing.</details>
# 3. Most failures are in generation. What do you check before changing the model?
#    <details><summary>Answer</summary>That the needed words really are in the top 5 the model read, and how the prompt presents them.</details>
#
# **Carry forward.** `artifacts/D4_4/attribution.json`: the diagnosis the capstone defence asks you to explain.
