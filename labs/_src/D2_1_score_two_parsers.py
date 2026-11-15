# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
# %% [markdown]
# # Lab 2.1 · Two parsers scored on 30 pages
#
# Day 2 · Session 1 (Parsing is a separate problem) · 25 min.
#
# Retrieval can only find text the parser produced. Al-Nuha's corpus mixes digital pages (a text layer
# exists) with scanned Arabic circulars (only an image). You score two parsers against reference text on
# 30 pages: parser A reads the text layer, parser B runs character recognition with layout analysis. You
# measure word error rate and table-cell accuracy by page type and write a routing rule.
#
# **About the parser outputs.** The files in `data/parsing/parser_a/` and `parser_b/` were written by
# `prepare_data.py` to reproduce the typical failures of the two kinds of parser on Arabic pages
# (presentation forms, visual word order, lost table cells, confused letters). The scoring code is the
# same you would run on the output of Docling, Unstructured or Tesseract.
#
# **Tasks**
# - [ ] 1 · One page, three ways (3')
# - [ ] 2 · Word error rate after normalisation (7')
# - [ ] 3 · Table-cell accuracy (6')
# - [ ] 4 · Scores by page type (5')
# - [ ] 5 · A routing rule, and its score (4')
# - [ ] ✱ Stretch · Repair the text layer and score it again

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
import unicodedata

import pandas as pd

import nuha as U

BASE = U.DATA / "parsing"
PAGES = U.jsonl("parsing/pages.jsonl")


def load(kind, pid):
    ext = "md" if kind == "reference" else "txt"
    return (BASE / kind / f"{pid}.{ext}").read_text(encoding="utf-8").strip()


DOCS = {p["page"]: {k: load(k, p["page"]) for k in ("reference", "parser_a", "parser_b")} for p in PAGES}
print(pd.DataFrame(PAGES).groupby(["source", "has_table"]).size().rename("pages"))

# %% [markdown]
# ## 1 · One page, three ways (3')

# %%
pid = next(p["page"] for p in PAGES if p["source"] == "digital" and p["lang"] == "ar" and p["has_table"])
for kind in ("reference", "parser_a", "parser_b"):
    print(f"--- {kind}\n" + "\n".join(DOCS[pid][kind].splitlines()[:6]))

# %% [markdown]
# 🔍 Parser A's Arabic looks right on screen, and is not: the letters are presentation forms (U+FExx) and
# the words run in visual order, last word first. A search index sees neither as the same text.

# %% [markdown]
# ## 2 · Word error rate after normalisation (7')
#
# Word error rate is the edit distance between the word sequences (insertions, deletions, substitutions)
# divided by the number of reference words. Normalise both sides first (NFKC, then `U.normalise`) so that
# you score the parse, not the spelling convention.
#
# ✏️ **Exercise** · Write `wer(reference, hypothesis)`. An empty hypothesis scores 1.0.

# %%
def words(text):
    return re.findall(r"\w+", U.normalise(unicodedata.normalize("NFKC", text)))


def wer(reference: str, hypothesis: str) -> float:
    # BEGIN SOLUTION
    r, h = words(reference), words(hypothesis)
    if not h:
        return 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return min(1.0, prev[-1] / max(1, len(r)))
    # END SOLUTION


# %%
# Check, with a safety net
try:
    toy = wer("the final examination", "the examination final")
except NotImplementedError:
    def wer(reference, hypothesis):
        import difflib
        r, h = words(reference), words(hypothesis)
        return 1.0 if not h else 1 - difflib.SequenceMatcher(a=r, b=h, autojunk=False).ratio()
    toy = wer("the final examination", "the examination final")
print("toy WER:", round(toy, 3))
U.check(abs(toy - 2 / 3) < 1e-9, "a swapped word pair costs two edits out of three words")

# %% [markdown]
# ## 3 · Table-cell accuracy (6')
#
# A table is only useful if each cell stays in its row and column. Read the reference table as rows of
# cells, and count the reference cells that appear in the same row of the parse at the same position.
#
# ✏️ **Exercise** · `cells(text)` returns the rows of every pipe table (skip the `|---|` rule), each row a
# list of normalised cell strings. `cell_accuracy(ref, hyp)` is the share of reference cells matched.

# %%
def cells(text: str) -> list[list[str]]:
    # BEGIN SOLUTION
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            rows.append([" ".join(words(c)) for c in line.strip().strip("|").split("|")])
    return rows
    # END SOLUTION


def cell_accuracy(ref: str, hyp: str) -> float | None:
    # BEGIN SOLUTION
    R, H = cells(ref), cells(hyp)
    if not R:
        return None
    hit = total = 0
    for i, row in enumerate(R):
        for j, c in enumerate(row):
            total += 1
            hit += i < len(H) and j < len(H[i]) and H[i][j] == c
    return hit / total
    # END SOLUTION


try:
    print(cell_accuracy(DOCS[pid]["reference"], DOCS[pid]["parser_b"]))
except NotImplementedError:
    def cell_accuracy(ref, hyp):
        return None
    print("(safety net: no table scores)")

# %% [markdown]
# ## 4 · Scores by page type (5')

# %%
rows = []
for p in PAGES:
    d = DOCS[p["page"]]
    for parser in ("parser_a", "parser_b"):
        rows.append({"page": p["page"], "source": p["source"], "table": p["has_table"], "lang": p["lang"], "parser": parser,
                     "WER": wer(d["reference"], d[parser]), "cells": cell_accuracy(d["reference"], d[parser])})
SCORES = pd.DataFrame(rows)
BY_TYPE = SCORES.groupby(["source", "table", "parser"])[["WER", "cells"]].mean().round(3)
print(BY_TYPE)
(U.lab_dir("D2_1") / "parsing_scores.json").write_text(SCORES.to_json(orient="records", force_ascii=False))
U.check(SCORES[(SCORES.source == "scanned") & (SCORES.parser == "parser_a")]["WER"].min() == 1.0, "the text layer of a scanned page is empty")

# %% [markdown]
# 🔍 Which parser wins on English digital pages? On Arabic digital pages? Which number would you never have
# seen if you had scored all 30 pages as one average?

# %% [markdown]
# ## 5 · A routing rule, and its score (4')
#
# ✏️ **Exercise** · `route(text_layer)` returns `"parser_a"` or `"parser_b"` from what the text layer
# looks like, without the reference: an empty layer, presentation forms (U+FB50 to U+FEFF) or a table
# that lost its pipes all send the page to parser B.

# %%
def route(text_layer: str) -> str:
    # BEGIN SOLUTION
    if not text_layer.strip():
        return "parser_b"
    if re.search("[\ufb50-\ufdff\ufe70-\ufefc]", text_layer):
        return "parser_b"
    if "|" not in text_layer and re.search(r"\b[A-Z]{3} \d{3}\b.*\b[A-Z]{3} \d{3}\b", text_layer):
        return "parser_b"
    return "parser_a"
    # END SOLUTION


try:
    chosen = {p["page"]: route(DOCS[p["page"]]["parser_a"]) for p in PAGES}
except NotImplementedError:
    chosen = {p["page"]: "parser_b" for p in PAGES}
routed = SCORES[[chosen[r.page] == r.parser for r in SCORES.itertuples()]]
summary = pd.DataFrame({"parser A only": SCORES[SCORES.parser == "parser_a"][["WER", "cells"]].mean(),
                        "parser B only": SCORES[SCORES.parser == "parser_b"][["WER", "cells"]].mean(),
                        "routed": routed[["WER", "cells"]].mean()}).round(3)
print(summary)
print("pages sent to parser A:", sum(v == "parser_a" for v in chosen.values()))
U.check(summary.loc["WER", "routed"] <= summary.loc["WER", "parser B only"], "routing is at least as good as parser B everywhere")

# %% [markdown]
# ## ✱ Stretch · Repair the text layer and score it again
#
# Fold the presentation forms with NFKC and reverse the word order of every Arabic line of parser A. How
# close does the repaired text layer get on digital Arabic pages? What does it still lose?

# %%
try:
    # BEGIN SOLUTION
    def repair(text):
        out = []
        for line in unicodedata.normalize("NFKC", text).splitlines():
            out.append(" ".join(reversed(line.split(" "))) if re.search("[\u0600-\u06ff]", line) else line)
        return "\n".join(out)


    dig_ar = [p["page"] for p in PAGES if p["source"] == "digital" and p["lang"] == "ar"]
    print("repaired text layer, digital Arabic WER:", round(sum(wer(DOCS[p]["reference"], repair(DOCS[p]["parser_a"])) for p in dig_ar) / len(dig_ar), 3))
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. A scanned circular is indexed and never retrieved. What does the parse look like?
#    <details><summary>Answer</summary>Empty, or a header line: the text layer of an image holds nothing.</details>
# 2. Why normalise before scoring a parser?
#    <details><summary>Answer</summary>To score what the parser got wrong, not a spelling convention both sides could share.</details>
# 3. Why route by page type instead of picking the best parser on average?
#    <details><summary>Answer</summary>Each parser wins on a different kind of page; the average hides both wins.</details>
#
# **Carry forward.** `artifacts/D2_1/parsing_scores.json`: the parsing score per page type (LO3 evidence).
