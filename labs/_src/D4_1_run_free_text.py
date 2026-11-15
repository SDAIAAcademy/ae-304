# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "duckdb>=1.2", "openai>=1.60", "python-dotenv>=1.0"]
# ///
# %% [markdown]
# # Lab 4.1 · Free text-to-SQL on 20 statistical questions
#
# Day 4 · Session 1 (Questions over a warehouse) · 25 min.
#
# "How many students were enrolled in the first semester of 1447?" No page answers this: the number lives in
# the enrolment statistics warehouse of Abdulrahman Al-Duwaish's institutional research unit. You give a
# model the schema, ask it for SQL on 20 questions with reference answers, run what it writes and measure
# execution accuracy: the share of questions whose result equals the reference answer.
#
# **The model.** With a key, a free hosted model writes the SQL. Without one, the notebook replays recorded
# stand-in replies (`data/reference/standin_sql.json`), written by the course author to show the mistakes
# free text-to-SQL typically makes; they are not a model's output. The notebook says which one ran.
#
# **Reference state.** The warehouse (`data/warehouse/`, loaded into DuckDB by `U.warehouse()`) and
# `questions.jsonl` (20 questions, the unit's reference SQL and answers). The students table holds
# synthetic personal data classified Restricted.
#
# **Tasks**
# - [ ] 1 · The schema the model sees (4')
# - [ ] 2 · Ask for SQL on 20 questions (6')
# - [ ] 3 · Execution accuracy against the reference answers (7')
# - [ ] 4 · Read the failures by kind (5')
# - [ ] 5 · What the model was allowed to read (3')
# - [ ] ✱ Stretch · Business rules in the prompt

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

DB = U.warehouse()
QS = U.jsonl("warehouse/questions.jsonl")
STANDIN = json.loads((U.REF / "standin_sql.json").read_text(encoding="utf-8"))
U.model_mode()  # prints which model answers
print(len(QS), "questions | tables:", [r[0] for r in DB.sql("SHOW TABLES").fetchall()])

# %% [markdown]
# ## 1 · The schema the model sees (4')
#
# Schema grounding: the model can only use the tables, columns and values it is shown. The usual prompt
# gives the tables and columns, and a few example values.

# %%
def schema_prompt() -> str:
    parts = []
    for (t,) in DB.sql("SHOW TABLES").fetchall():
        cols = DB.sql(f"DESCRIBE {t}").fetchall()
        sample = DB.sql(f"SELECT * FROM {t} LIMIT 2").fetchall()
        parts.append(f"TABLE {t} (" + ", ".join(f"{c[0]} {c[1]}" for c in cols) + ")\n  e.g. " + "; ".join(map(str, sample)))
    return "\n".join(parts)


SCHEMA = schema_prompt()
print(SCHEMA)
print("\n", U.count_tokens(SCHEMA), "estimated tokens")

# %% [markdown]
# ## 2 · Ask for SQL on 20 questions (6')
#
# ✏️ **Exercise** · `ask_sql(q)`: with a live model, send the schema and the question and ask for one DuckDB
# query in a ```sql block; extract the SQL. Otherwise return the stand-in reply `STANDIN[q["qid"]]`.

# %%
def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";")


def ask_sql(q: dict) -> str:
    # BEGIN SOLUTION
    if U.model_mode() == "live":
        try:
            r = U.chat([{"role": "system", "content": "You write one DuckDB SQL query that answers the question. Reply with the query in a ```sql block."},
                        {"role": "user", "content": f"Schema:\n{SCHEMA}\n\nQuestion: {q['q']}"}], temperature=0)
            return extract_sql(r["content"])
        except U.ModelUnavailable:
            pass
    return STANDIN[q["qid"]]
    # END SOLUTION


try:
    SQL = {q["qid"]: ask_sql(q) for q in QS}
except NotImplementedError:
    SQL = dict(STANDIN)
print(SQL["S01"])

# %% [markdown]
# ## 3 · Execution accuracy against the reference answers (7')
#
# Execution accuracy compares results, not SQL text: two different queries can both be right. A number
# matches when it is within 0.001 of the reference (or 0.5% of it); a query that fails to run is wrong.
#
# ✏️ **Exercise** · `run_and_score(qid, sql)` returns `{"qid", "result", "answer", "correct", "error"}`.

# %%
def same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= max(0.001, 0.005 * abs(float(b)))
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


ANSWERS = {q["qid"]: q["answer"] for q in QS}


def run_and_score(qid: str, sql: str) -> dict:
    # BEGIN SOLUTION
    try:
        rows = DB.sql(sql).fetchall()
        result = rows[0][0] if rows else None
        return {"qid": qid, "result": result, "answer": ANSWERS[qid], "correct": result is not None and same(result, ANSWERS[qid]), "error": ""}
    except Exception as e:
        return {"qid": qid, "result": None, "answer": ANSWERS[qid], "correct": False, "error": str(e).splitlines()[0][:80]}
    # END SOLUTION


try:
    RESULTS = pd.DataFrame([run_and_score(qid, s) for qid, s in SQL.items()])
except NotImplementedError:
    RESULTS = pd.DataFrame(U.reference("D4_1_results"))
U.save_reference("D4_1_results", RESULTS.astype({"result": str}).to_dict(orient="records"))
print(RESULTS.to_string(index=False))
ACC = RESULTS.correct.mean()
print(f"\nexecution accuracy, free text-to-SQL ({U.model_mode()}): {ACC:.2f}")
RESULTS.to_json(U.lab_dir("D4_1") / "free_sql.json", orient="records")
U.check(0 < ACC < 1, "some questions right, some wrong: the interesting case")

# %% [markdown]
# ## 4 · Read the failures by kind (5')
#
# ✏️ **Exercise** · For each failure, compare the SQL with the reference SQL and tag the cause: `filter`
# (a status that should be excluded), `calendar` (a Hijri term written into a Gregorian column),
# `renamed` (a programme that was merged), `denominator`, `weighting` or `other`.

# %%
REF_SQL = {q["qid"]: q["reference_sql"] for q in QS}
for r in RESULTS[~RESULTS.correct].itertuples():
    print(f"{r.qid}: got {r.result}, expected {r.answer}\n   model: {SQL[r.qid]}\n   ref:   {REF_SQL[r.qid]}")


def tag_failures() -> dict:
    # BEGIN SOLUTION
    tags = {}
    for r in RESULTS[~RESULTS.correct].itertuples():
        s = SQL[r.qid]
        if re.search(r"'14\d\d-\d'", s) and "hijri" not in s:
            tags[r.qid] = "calendar"
        elif "programme_id = 'IS'" in s:
            tags[r.qid] = "renamed"
        elif "AVG(grade_points)" in s:
            tags[r.qid] = "weighting"
        elif "status = 'completed'" in s and "denied" not in s:
            tags[r.qid] = "denominator"
        elif "BETWEEN" in s:
            tags[r.qid] = "other"
        else:
            tags[r.qid] = "filter"
    return tags
    # END SOLUTION


try:
    TAGS = tag_failures()
except NotImplementedError:
    TAGS = {}
print(pd.Series(TAGS).value_counts())

# %% [markdown]
# 🔍 Which of these mistakes would a careful analyst also make on their first day? The rules that fix them
# (which statuses count, which calendar, which programmes merged) are not in the schema.

# %% [markdown]
# ## 5 · What the model was allowed to read (3')
#
# The students table is Restricted: synthetic, but shaped like personal data. Free text-to-SQL needs the
# raw tables, so every query can read rows about individual students.

# %%
touch = [qid for qid, s in SQL.items() if re.search(r"\bstudents\b", s)]
print(f"{len(touch)} of {len(SQL)} queries read the Restricted students table:", touch)

# %% [markdown]
# ## ✱ Stretch · Business rules in the prompt
#
# Add three rules to the system prompt (withdrawn enrolments do not count as enrolled; terms are stored as
# Gregorian ids and the terms table maps Hijri names; MIS was merged into IS). With a live model, rerun.

# %%
try:
    # BEGIN SOLUTION
    RULES = ("Rules: an enrolment with status 'withdrawn' does not make a student enrolled. term_id is Gregorian "
             "('2025-1'); terms.hijri holds the Hijri name ('1447-1'). Programme MIS was merged into IS: use programmes.merged_into.")
    if U.model_mode() == "live":
        rerun = []
        for q in QS:
            r = U.chat([{"role": "system", "content": "You write one DuckDB SQL query. Reply in a ```sql block. " + RULES},
                        {"role": "user", "content": f"Schema:\n{SCHEMA}\n\nQuestion: {q['q']}"}], temperature=0)
            rerun.append(run_and_score(q["qid"], extract_sql(r["content"]))["correct"])
        print("with rules:", sum(rerun) / len(rerun))
    else:
        print("stand-in mode: the recorded replies do not read the prompt; run with a key to measure the rules")
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. A query runs and returns a plausible number. Why is that not a pass?
#    <details><summary>Answer</summary>Execution accuracy compares the result with a reference answer; a plausible wrong number is the dangerous failure.</details>
# 2. "First semester of 1447" returned zero. What went wrong, and where is the fix?
#    <details><summary>Answer</summary>A Hijri name compared with a Gregorian id; the mapping belongs in a governed definition, not in each prompt.</details>
# 3. Why does free text-to-SQL raise a data-protection question here?
#    <details><summary>Answer</summary>The model's queries read the Restricted student rows directly, whatever the question.</details>
#
# **Carry forward.** `artifacts/D4_1/free_sql.json`: the first half of the LO6 comparison.
