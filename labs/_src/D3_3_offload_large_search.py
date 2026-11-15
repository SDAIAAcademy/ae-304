# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "openai>=1.60", "python-dotenv>=1.0"]
# ///
# %% [markdown]
# # Lab 3.3 · Offloading large results to files
#
# Day 3 · Session 3 (Paging and offloading) · 25 min.
#
# "How many courses of the Physics programme have no prerequisite?" The tool that can answer returns every
# course page of the programme, and a college returns sixty. Inside a context budget of 1,500 tokens the
# result is cut, and the count is wrong. You offload large results to a file, give the model a short stub
# and a tool to search the file, and measure tokens and accuracy on eight questions.
#
# **The model.** A free hosted model with a key; otherwise the scripted stand-in
# `nibras_tools.standin_count`, which counts in what it can see, or greps the file when it gets a stub.
#
# **Reference state.** The Public course pages and `data/agent/large_questions.jsonl` (8 questions).
#
# **Tasks**
# - [ ] 1 · How large is a large result? (4')
# - [ ] 2 · Eight questions inside the budget, no offloading (5')
# - [ ] 3 · Offload: a file and a stub (7')
# - [ ] 4 · A tool to search the file (5')
# - [ ] 5 · Tokens and accuracy, both ways (4')
# - [ ] ✱ Stretch · Paging instead of searching

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
import nibras_tools as T

QS = U.jsonl("agent/large_questions.jsonl")
BUDGET = 1500
OUT = U.lab_dir("D3_3")
SYSTEM = "You are Nibras. Answer from the tools only. Give the number you counted."
print(U.model_mode(), "|", len(QS), "questions | budget", BUDGET, "tokens")

# %% [markdown]
# ## 1 · How large is a large result? (4')

# %%
for scope in ({"programme": "PHY"}, {"college": "ENG"}):
    text = T.list_courses(**scope)
    print(scope, "->", len(text), "characters,", U.count_tokens(text), "estimated tokens")

# %% [markdown]
# ## 2 · Eight questions inside the budget, no offloading (5')
#
# `U.run_agent(..., budget=BUDGET)` cuts any tool result that does not fit in what is left of the budget,
# and says so at the cut, as many harnesses do.

# %%
def run(tools, label):
    rows = []
    for q in QS:
        r = U.run_agent(q["q"], tools, T.standin_count, system=SYSTEM, max_steps=4, budget=BUDGET)
        rows.append({"qid": q["qid"], "mode": label, "correct": U.graded(r["answer"], q["expect"]), "calls": len(r["calls"]),
                     "tokens": r["tokens"], "answer": r["answer"][:60], "expect": q["expect"][0]})
    return pd.DataFrame(rows)


PLAIN = run([T.list_courses], "in context")
print(PLAIN.to_string(index=False))

# %% [markdown]
# ## 3 · Offload: a file and a stub (7')
#
# ✏️ **Exercise** · `offload(fn)` wraps a tool: when its result is longer than 800 estimated tokens, write it
# to `OUT/<tool>_<n>.txt` and return a JSON stub instead: the file name, the number of lines and pages,
# the first 200 characters, and a hint to use `grep_file`. Short results pass through unchanged. Keep the
# wrapped tool's name and docstring (the model reads them).

# %%
def offload(fn):
    # BEGIN SOLUTION
    count = {"n": 0}

    def wrapped(**kw):
        result = fn(**kw)
        if U.count_tokens(result) <= 800:
            return result
        count["n"] += 1
        f = OUT / f"{fn.__name__}_{count['n']}.txt"
        f.write_text(result, encoding="utf-8")
        return json.dumps({"file": f.name, "lines": result.count("\n") + 1, "pages": result.count("\n\n") + 1,
                           "preview": result[:200], "hint": "the full result is in the file: use grep_file(file, pattern)"}, ensure_ascii=False)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    wrapped.__signature__ = __import__("inspect").signature(fn)
    return wrapped
    # END SOLUTION


try:
    list_courses_offloaded = offload(T.list_courses)
    stub = list_courses_offloaded(college="ENG")
except NotImplementedError:
    list_courses_offloaded, stub = None, "{}"
print(stub[:300])

# %% [markdown]
# ## 4 · A tool to search the file (5')
#
# ✏️ **Exercise** · `grep_file(file, pattern)`: the lines of the offloaded file that contain the pattern
# (case-insensitive), at most 50 of them, and their count. Refuse any file name outside `OUT`.

# %%
def grep_file(file: str, pattern: str) -> dict:
    """Search an offloaded result file for a pattern; returns the matching lines (at most 50) and their count."""
    # BEGIN SOLUTION
    path = (OUT / file).resolve()
    if path.parent != OUT.resolve() or not path.exists():
        return {"error": f"no offloaded file '{file}'"}
    hits = [l for l in path.read_text(encoding="utf-8").splitlines() if pattern.lower() in l.lower()]
    return {"count": len(hits), "lines": hits[:50]}
    # END SOLUTION


try:
    print(grep_file(json.loads(stub)["file"], "Credit hours: 4")["count"], "| outside:", grep_file("../../nuha.py", "def"))
except (NotImplementedError, KeyError, json.JSONDecodeError):
    grep_file = None
    print("(not written yet)")

# %% [markdown]
# ## 5 · Tokens and accuracy, both ways (4')

# %%
if list_courses_offloaded and grep_file:
    OFF = run([list_courses_offloaded, grep_file], "offloaded")
else:
    OFF = pd.DataFrame(U.reference("D3_3_offloaded"))
U.save_reference("D3_3_offloaded", OFF.to_dict(orient="records"))
BOTH = pd.concat([PLAIN, OFF])
TABLE = BOTH.groupby("mode")[["correct", "calls", "tokens"]].mean().round(2)
print(TABLE)
BOTH.to_json(OUT / "offload_compare.json", orient="records", force_ascii=False)
U.check(TABLE.loc["offloaded", "correct"] > TABLE.loc["in context", "correct"], "offloading answers more of the eight questions")

# %% [markdown]
# 🔍 The offloaded runs make one more call. Count the tokens anyway: which design spends less, and which one
# is right?

# %% [markdown]
# ## ✱ Stretch · Paging instead of searching
#
# Some questions cannot be answered by a pattern ("which of these courses is hardest?"). Write
# `read_lines(file, start, n)` and let the model page through the file. What does a full read cost?

# %%
try:
    # BEGIN SOLUTION
    def read_lines(file: str, start: int = 0, n: int = 40) -> str:
        """Read n lines of an offloaded file from line start."""
        lines = (OUT / file).read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[start:start + n]) + f"\n[lines {start} to {min(len(lines), start + n)} of {len(lines)}]"


    f = json.loads(stub)["file"]
    total = len((OUT / f).read_text(encoding="utf-8").splitlines())
    pages_needed = (total + 39) // 40
    print(f"{total} lines: {pages_needed} calls of 40 lines, about {U.count_tokens(read_lines(f, 0, 40)) * pages_needed} tokens to read it all")
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. A tool result is cut at the budget. What does the model know about the part it lost?
#    <details><summary>Answer</summary>Nothing, unless the harness says a cut happened; it will count what it sees and answer with confidence.</details>
# 2. What must a stub contain so the model can use the file?
#    <details><summary>Answer</summary>The file name, its size, a preview, and the tool that reads or searches it.</details>
# 3. Why refuse file names outside the folder in `grep_file`?
#    <details><summary>Answer</summary>A tool argument is input from the model, and retrieved text can steer the model: never let it name any file on the machine.</details>
#
# **Carry forward.** `artifacts/D3_3/offload_compare.json`: tokens and accuracy (LO5).
