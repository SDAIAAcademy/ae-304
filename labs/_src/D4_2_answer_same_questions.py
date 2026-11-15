# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "duckdb>=1.2", "sqlglot>=26.0", "openai>=1.60", "python-dotenv>=1.0"]
# ///
# %% [markdown]
# # Lab 4.2 · Governed metrics and validated SQL
#
# Day 4 · Session 2 (Semantic layers and validated SQL) · 25 min.
#
# The institutional research unit already defines "enrolled", "pass rate" and "weighted GPA" once, as
# version-controlled SQL views and a metric catalog. Instead of writing SQL, the model picks a metric,
# filters and a grouping; your code compiles that request to SQL over the governed views and validates it
# before it runs. You answer the same 20 questions, compare execution accuracy with free text-to-SQL, and
# write the decision memo.
#
# **The model.** With a key, a free hosted model writes the metric request (JSON). Without one, the notebook
# replays recorded stand-in requests (`data/reference/standin_semantic.json`), written by the course author,
# with two of the mistakes a model makes. The notebook says which one ran.
#
# **Reference state.** The warehouse, `data/semantic/` (views and catalog), the 20 questions, and the
# Lab 4.1 result on the same questions (`data/reference/D4_1_results.json`) if you did not run Lab 4.1.
#
# **Tasks**
# - [ ] 1 · The views and the metric catalog (4')
# - [ ] 2 · Compile a metric request to SQL (7')
# - [ ] 3 · Validate before running (6')
# - [ ] 4 · Twenty questions through the layer (5')
# - [ ] 5 · The decision memo (3')
# - [ ] ✱ Stretch · Small cells

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
import sqlglot
from sqlglot import exp

import nuha as U

DB = U.warehouse()
CAT = U.semantic_layer(DB)
QS = U.jsonl("warehouse/questions.jsonl")
ANSWERS = {q["qid"]: q["answer"] for q in QS}
STANDIN = json.loads((U.REF / "standin_semantic.json").read_text(encoding="utf-8"))
U.model_mode()  # prints which model answers
print(len(CAT["metrics"]), "metrics over", list(CAT["views"]))

# %% [markdown]
# ## 1 · The views and the metric catalog (4')

# %%
for name, m in CAT["metrics"].items():
    print(f"{name:20} {m['view']:16} {m['description']}")
print("\n", DB.sql("SELECT term, hijri_term, programme, COUNT(*) FROM enrolment_facts GROUP BY ALL ORDER BY 4 DESC LIMIT 3").df())

# %% [markdown]
# 🔍 The view does three things the 4.1 failures needed: it joins the Hijri name, maps the merged programme,
# and exposes no student identifier the model can ask for by name.

# %% [markdown]
# ## 2 · Compile a metric request to SQL (7')
#
# A request: `{"metric": "enrolled_students", "filters": {"term": "2025-1", "college": "ENG"},
# "group_by": [], "order": "desc", "limit": 1}`. A filter value can be a list (IN).
#
# ✏️ **Exercise** · `compile_request(req)`: the metric's view and SQL expression, a WHERE clause from the
# filters (the column `term` maps to the metric's `term_column` when it has one), GROUP BY, ORDER BY the
# value and LIMIT. Quote values safely: allow only letters, digits, spaces and `-/._`, reject anything else.

# %%
SAFE = re.compile(r"^[\w /.\-]+$")


def compile_request(req: dict) -> str:
    # BEGIN SOLUTION
    m = CAT["metrics"][req["metric"]]
    where = []
    for col, val in (req.get("filters") or {}).items():
        col = m.get("term_column", "term") if col == "term" else col
        vals = val if isinstance(val, list) else [val]
        if not all(SAFE.match(str(v)) for v in vals):
            raise ValueError(f"unsafe value in {col}")
        where.append(f"{col} IN (" + ", ".join(f"'{v}'" for v in vals) + ")")
    groups = req.get("group_by") or []
    sql = "SELECT " + "".join(f"{g}, " for g in groups) + f"{m['sql']} AS value FROM {m['view']}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if groups:
        sql += " GROUP BY " + ", ".join(groups) + f" ORDER BY value {'ASC' if req.get('order') == 'asc' else 'DESC'}"
    if req.get("limit"):
        sql += f" LIMIT {int(req['limit'])}"
    return sql
    # END SOLUTION


try:
    print(compile_request(STANDIN["S09"]))
except NotImplementedError:
    compile_request = None
    print("(not written yet)")

# %% [markdown]
# ## 3 · Validate before running (6')
#
# Validation is code that refuses before the warehouse runs anything. Two layers:
#
# - the request: a known metric, dimensions that exist on its view, filter values that exist in the data
#   (a Hijri name in the Gregorian `term` column is refused with a hint);
# - the SQL: one statement, a SELECT, reading only the governed views (parsed with sqlglot, not with a regex).
#
# ✏️ **Exercise** · `validate(req, sql)` returns a list of problems (empty means valid).

# %%
def validate(req: dict, sql: str) -> list[str]:
    # BEGIN SOLUTION
    problems = []
    m = CAT["metrics"].get(req.get("metric"))
    if m is None:
        return [f"unknown metric {req.get('metric')}"]
    dims = CAT["views"][m["view"]]
    for col, val in (req.get("filters") or {}).items():
        if col not in dims:
            problems.append(f"{col} is not a dimension of {m['view']}")
            continue
        real = m.get("term_column", "term") if col == "term" else col
        known = {str(r[0]) for r in DB.sql(f"SELECT DISTINCT {real} FROM {m['view']}").fetchall()}
        for v in (val if isinstance(val, list) else [val]):
            if str(v) not in known:
                hint = " (a Hijri name: use hijri_term)" if re.match(r"^14\d\d-\d$", str(v)) else ""
                problems.append(f"{col} = {v} does not exist{hint}")
    for g in req.get("group_by") or []:
        if g not in dims:
            problems.append(f"cannot group by {g}")
    try:
        trees = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        return problems + [f"SQL does not parse: {e}"]
    if len(trees) != 1 or not isinstance(trees[0], exp.Select):
        problems.append("only one SELECT statement is allowed")
    tables = {t.name for t in trees[0].find_all(exp.Table)}
    if not tables <= set(CAT["views"]):
        problems.append(f"reads tables outside the semantic layer: {sorted(tables - set(CAT['views']))}")
    return problems
    # END SOLUTION


try:
    print(validate(STANDIN["S17"], compile_request(STANDIN["S17"])))
    print(validate({"metric": "enrolments"}, "SELECT * FROM students"))
except (NotImplementedError, TypeError):
    validate = None
    print("(not written yet)")

# %% [markdown]
# ## 4 · Twenty questions through the layer (5')

# %%
def ask_request(q):
    if U.model_mode() == "live":
        try:
            catalog = {k: v["description"] for k, v in CAT["metrics"].items()}
            r = U.chat([{"role": "system", "content": "Answer with JSON only: a metric request {metric, filters, group_by, order, limit}. "
                         "Use only these metrics and dimensions. Terms are Gregorian ids such as 2025-1; Hijri names go in hijri_term."},
                        {"role": "user", "content": json.dumps({"metrics": catalog, "dimensions": CAT["views"], "question": q["q"]}, ensure_ascii=False)}],
                       response_format={"type": "json_object"}, temperature=0)
            return json.loads(r["content"])
        except (U.ModelUnavailable, json.JSONDecodeError):
            pass
    return STANDIN[q["qid"]]


if compile_request and validate:
    rows = []
    for q in QS:
        req = ask_request(q)
        try:
            sql = compile_request(req)
            problems = validate(req, sql)
        except (KeyError, ValueError) as e:
            sql, problems = "", [f"cannot compile: {e}"]
        result = None if problems else DB.sql(sql).fetchall()[0][0]
        rows.append({"qid": q["qid"], "result": result, "answer": ANSWERS[q["qid"]], "refused": bool(problems),
                     "correct": result is not None and U.same_value(result, ANSWERS[q["qid"]]), "problems": "; ".join(problems)})
    LAYER = pd.DataFrame(rows)
else:
    LAYER = pd.DataFrame(U.reference("D4_2_results"))
U.save_reference("D4_2_results", LAYER.astype({"result": str}).to_dict(orient="records"))
print(LAYER.to_string(index=False))
FREE = pd.DataFrame(U.reference("D4_1_results"))
COMPARE = pd.DataFrame([
    {"method": "free text-to-SQL", "execution accuracy": FREE.correct.mean(), "refused": 0, "wrong and confident": int((~FREE.correct).sum())},
    {"method": "semantic layer, validated", "execution accuracy": LAYER.correct.mean(), "refused": int(LAYER.refused.sum()),
     "wrong and confident": int((~LAYER.correct & ~LAYER.refused).sum())},
]).round(2)
print(COMPARE.to_string(index=False), f"\n(model: {U.model_mode()}; free text-to-SQL from the supplied Lab 4.1 result)")
LAYER.to_json(U.lab_dir("D4_2") / "semantic.json", orient="records")
U.check(COMPARE.iloc[1]["execution accuracy"] > COMPARE.iloc[0]["execution accuracy"], "the governed metrics answer more questions correctly")

# %% [markdown]
# 🔍 A refusal with a hint ("a Hijri name: use hijri_term") is an answer the assistant can act on: ask
# again, or tell the user. A confident wrong number is not.

# %% [markdown]
# ## 5 · The decision memo (3')
#
# ✏️ **Exercise** · Write the one-page memo for Dr. Lubna Al-Harithi: the decision, the evidence (the two
# accuracies and the wrong-and-confident counts), the cost of the layer (who maintains the views), and the
# condition under which you would revisit it.

# %%
def memo() -> str:
    # BEGIN SOLUTION
    f, s = COMPARE.iloc[0], COMPARE.iloc[1]
    return (f"# Decision: statistical questions go through the semantic layer\n\n"
            f"**Decision.** Nibras answers statistical questions only through the governed metrics, with validated SQL. Free text-to-SQL is not shipped.\n\n"
            f"**Evidence.** On 20 questions with reference answers, free text-to-SQL scored {f['execution accuracy']:.2f} execution accuracy "
            f"with {f['wrong and confident']} confident wrong numbers; the semantic layer scored {s['execution accuracy']:.2f}, "
            f"refused {s['refused']} and gave {s['wrong and confident']} confident wrong numbers ({U.model_mode()} model).\n\n"
            f"**Cost.** The institutional research unit maintains the views and the catalog as code; a new metric is a reviewed change.\n\n"
            f"**Data protection.** The model never reads student rows; queries touch the governed views only.\n\n"
            f"**Revisit** when users ask many questions no metric covers: measure them, then add metrics, not free SQL.")
    # END SOLUTION


try:
    MEMO = memo()
except NotImplementedError:
    MEMO = "not written yet"
(U.lab_dir("D4_2") / "decision_memo.md").write_text(MEMO, encoding="utf-8")
print(MEMO)

# %% [markdown]
# ## ✱ Stretch · Small cells
#
# Group `enrolled_students` in 2025-1 by programme and gender. Suppress any count below `CAT["small_cell"]`
# before it leaves the layer: a cell of three students can identify them.

# %%
try:
    # BEGIN SOLUTION
    df = DB.sql(compile_request({"metric": "enrolled_students", "filters": {"term": "2025-1", "college": "ART"}, "group_by": ["programme", "gender", "nationality"]})).df()
    df["shown"] = [str(v) if v >= CAT["small_cell"] else "fewer than 5" for v in df["value"]]
    print(df.sort_values("value").head(6).to_string(index=False))
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. The validator accepted a request. Does that make the answer right?
#    <details><summary>Answer</summary>No: validation proves the query is allowed and well formed; the metric definition decides whether it means the right thing.</details>
# 2. Why compile from a request instead of validating the model's own SQL?
#    <details><summary>Answer</summary>The definitions live in one reviewed place; the model chooses among them and cannot redefine "enrolled".</details>
# 3. A dean asks a question no metric covers. What do you do?
#    <details><summary>Answer</summary>Refuse with a clear message, log it, and add a metric if the question recurs; do not fall back to free SQL silently.</details>
#
# **Carry forward.** `artifacts/D4_2/decision_memo.md`: the one-page memo (LO6 evidence).
