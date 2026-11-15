# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "openai>=1.60", "python-dotenv>=1.0"]
# ///
# %% [markdown]
# # Lab 3.2 · Single-shot against iterative search
#
# Day 3 · Session 2 (Multi-step questions) · 25 min.
#
# "What GPA do I need to transfer into the college that teaches CLS 334?" No page holds the answer: one
# says which college teaches CLS 334, another what that college asks for. You compare one search with an
# iterative loop that searches, reads, decides the next search and stops, on ten multi-hop questions, and
# count accuracy, searches and tokens.
#
# **The model.** The loop is yours. Deciding the next search is the model's job: a free hosted model with a
# key, otherwise the scripted stand-in `nibras_tools.standin_decide`, which follows the decomposition a
# model is asked to follow. The notebook says which one ran.
#
# **Reference state.** The Public corpus and `data/agent/multistep.jsonl` (15 questions; the first ten
# have two hops).
#
# **Tasks**
# - [ ] 1 · Single-shot: one search, then answer (5')
# - [ ] 2 · The iterative loop and its stop rules (9')
# - [ ] 3 · Ten questions both ways (6')
# - [ ] 4 · Accuracy, searches and tokens (5')
# - [ ] ✱ Stretch · Three hops

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

QS = U.jsonl("agent/multistep.jsonl")
TEN = QS[:10]
U.model_mode()
print(len(TEN), "two-hop questions, e.g.:", TEN[0]["q"])


def model_decide(question, notes):
    """The hosted model as the decider: it sees the question and the notes, and returns JSON."""
    prompt = ("Answer the question from the notes, or ask for one more search. Reply with JSON only: "
              '{"search": "<query>"} or {"answer": "<answer with the number or code>"}.\n'
              f"Question: {question}\nNotes:\n" + json.dumps(notes, ensure_ascii=False)[:6000])
    r = U.chat([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return json.loads(r["content"])


def decide(question, notes, final=False):
    if U.model_mode() == "live":
        try:
            return model_decide(question + (" (answer now, no more searches)" if final else ""), notes)
        except (U.ModelUnavailable, json.JSONDecodeError):
            pass
    return T.standin_decide(question, notes, final)


# %% [markdown]
# ## 1 · Single-shot: one search, then answer (5')
#
# ✏️ **Exercise** · `single_shot(question)`: one `T.hybrid_search(question, 5, chars=1500)`, then the decider
# answers from those notes only (`decide(question, notes, final=True)`). Return the answer, 1 search, and the
# estimated tokens of the notes.

# %%
def single_shot(question: str) -> dict:
    # BEGIN SOLUTION
    notes = [{"query": question, "top": T.hybrid_search(question, 5, chars=1500)}]
    d = decide(question, notes, final=True)
    return {"answer": d.get("answer", ""), "searches": 1, "tokens": U.count_tokens(notes)}
    # END SOLUTION


try:
    print(single_shot(TEN[0]["q"])["answer"][:200])
except NotImplementedError:
    def single_shot(question):
        notes = [{"query": question, "top": T.hybrid_search(question, 5, chars=1500)}]
        return {"answer": T.standin_decide(question, notes, final=True)["answer"], "searches": 1, "tokens": U.count_tokens(notes)}
    print("(safety net: single-shot without a decider)")

# %% [markdown]
# ## 2 · The iterative loop and its stop rules (9')
#
# The loop asks the decider for the next step. Three stop rules keep it honest: an answer, a search budget,
# and a repeated query (the decider is going round in circles).
#
# ✏️ **Exercise** · `iterative(question, max_searches=4)`: start with no notes; ask `decide`; on a search,
# run `T.hybrid_search(query, 3, chars=1500)` and append `{"query", "top"}` to the notes; stop on an answer, on a
# repeated query, or when the budget is spent (then answer from what you have). Count searches and tokens.

# %%
def iterative(question: str, max_searches: int = 4) -> dict:
    # BEGIN SOLUTION
    notes, tokens, asked = [], 0, set()
    while True:
        d = decide(question, notes)
        tokens += U.count_tokens(notes) + U.count_tokens(question)
        if "answer" in d:
            return {"answer": d["answer"], "searches": len(notes), "tokens": tokens, "queries": [n["query"] for n in notes]}
        q = d["search"]
        if q in asked or len(notes) >= max_searches:
            last = decide(question, notes, final=True)
            return {"answer": last.get("answer", ""), "searches": len(notes), "tokens": tokens, "queries": [n["query"] for n in notes]}
        asked.add(q)
        notes.append({"query": q, "top": T.hybrid_search(q, 3, chars=1500)})
    # END SOLUTION


try:
    r = iterative(TEN[0]["q"])
    print(r["queries"], "->", r["answer"])
except NotImplementedError:
    iterative = None
    print("(not written yet: step 3 uses the reference results)")

# %% [markdown]
# ## 3 · Ten questions both ways (6')

# %%
if iterative is not None:
    rows = []
    for q in TEN:
        s, it = single_shot(q["q"]), iterative(q["q"])
        rows.append({"qid": q["qid"], "single correct": U.graded(s["answer"], q["expect"]), "single tokens": s["tokens"],
                     "iter correct": U.graded(it["answer"], q["expect"]), "iter searches": it["searches"], "iter tokens": it["tokens"],
                     "queries": " | ".join(it["queries"])})
    RUNS = pd.DataFrame(rows)
else:
    RUNS = pd.DataFrame(U.reference("D3_2_runs"))
U.save_reference("D3_2_runs", RUNS.to_dict(orient="records"))
print(RUNS.drop(columns="queries").to_string(index=False))
RUNS.to_json(U.lab_dir("D3_2") / "single_vs_iterative.json", orient="records", force_ascii=False)

# %% [markdown]
# ## 4 · Accuracy, searches and tokens (5')

# %%
COMPARE = pd.DataFrame([
    {"method": "single-shot", "accuracy": RUNS["single correct"].mean(), "searches": 1.0, "tokens": RUNS["single tokens"].mean()},
    {"method": "iterative", "accuracy": RUNS["iter correct"].mean(), "searches": RUNS["iter searches"].mean(), "tokens": RUNS["iter tokens"].mean()},
]).round(2)
print(COMPARE.to_string(index=False), "\n(model:", U.model_mode() + ")")
U.check(COMPARE.iloc[1]["accuracy"] > COMPARE.iloc[0]["accuracy"], "iterative search answers more two-hop questions")

# %% [markdown]
# 🔍 Read the queries of one iterative run. Is the second query something the student would never have
# typed? That is the point: the first page tells you what to ask next.

# %% [markdown]
# ## ✱ Stretch · Three hops
#
# Questions M11 to M15 need three hops. Run them with `max_searches=2` and with 4. What does the budget cost?

# %%
try:
    # BEGIN SOLUTION
    for budget in (2, 4):
        ok = [U.graded(iterative(q["q"], budget)["answer"], q["expect"]) for q in QS[10:]]
        print("budget", budget, "->", sum(ok), "of", len(ok))
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Why can a bigger top k not replace the second search?
#    <details><summary>Answer</summary>The second query depends on what the first page says; no query typed before reading can ask for it.</details>
# 2. What does the repeated-query rule protect against?
#    <details><summary>Answer</summary>A decider that keeps asking the same thing and spends the budget without learning anything.</details>
# 3. When would you keep single-shot search?
#    <details><summary>Answer</summary>For one-hop questions, which are most of them: it is cheaper and faster, and iterative search adds nothing.</details>
#
# **Carry forward.** `artifacts/D3_2/single_vs_iterative.json`: the comparison of the LO5 evidence.
