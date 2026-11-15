# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "networkx>=3.4", "openai>=1.60", "python-dotenv>=1.0"]
# ///
# %% [markdown]
# # Lab 3.4 · A graph of courses, prerequisites and rules
#
# Day 3 · Session 4 (Knowledge graphs for linked facts) · 25 min.
#
# "Which courses must I pass, directly or indirectly, before PHY 401?" The answer is a chain that runs
# through four or five course pages; a search finds one link per call. You build a small graph of
# courses, prerequisites, programmes and colleges from the structured extraction of the course pages and
# plans, expose three graph queries as tools, and answer ten relationship questions with and without it.
#
# **The model.** A free hosted model with a key; otherwise the scripted stand-in: without the graph it
# follows prerequisites page by page (`nibras_tools.standin_without_graph`, at most six searches); with the
# graph it makes one graph call.
#
# **Reference state.** `data/graph/courses.jsonl` (368 courses with credits and prerequisites, extracted
# from the course pages), `plans.jsonl`, `colleges.jsonl`, and `relationship_questions.jsonl` (10).
#
# **Tasks**
# - [ ] 1 · Nodes and edges from the extraction (7')
# - [ ] 2 · Three graph queries as tools (7')
# - [ ] 3 · Ten questions without the graph (4')
# - [ ] 4 · The same ten with the graph (4')
# - [ ] 5 · Accuracy, calls and tokens (3')
# - [ ] ✱ Stretch · A rule edge: transfer requirements

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

import networkx as nx
import pandas as pd

import nuha as U
import nibras_tools as T

COURSES = U.jsonl("graph/courses.jsonl")
PLANS = U.jsonl("graph/plans.jsonl")
COLLEGES = U.jsonl("graph/colleges.jsonl")
QS = U.jsonl("graph/relationship_questions.jsonl")
SYSTEM = "You are Nibras. Answer from the tools only. List every course code the answer needs."
print(U.model_mode(), "|", len(COURSES), "courses |", len(QS), "questions")

# %% [markdown]
# ## 1 · Nodes and edges from the extraction (7')
#
# ✏️ **Exercise** · `build_graph()` returns a `networkx.DiGraph` with a node per course, programme and
# college (attribute `kind`), and edges `requires` (course to prerequisite), `includes` (programme to
# course, from the plans) and `in` (programme to college). Store credits and title on course nodes and the
# transfer GPA on college nodes.

# %%
def build_graph() -> nx.DiGraph:
    # BEGIN SOLUTION
    G = nx.DiGraph()
    for c in COLLEGES:
        G.add_node(c["college"], kind="college", name=c["name"], transfer_gpa=c["transfer_gpa"])
    for c in COURSES:
        G.add_node(c["code"], kind="course", title=c["title"], credits=c["credits"])
        for p in c["prereqs"]:
            G.add_edge(c["code"], p, rel="requires")
        if c["programme"]:
            G.add_node(c["programme"], kind="programme")
            G.add_edge(c["programme"], c["college"], rel="in")
    for row in PLANS:
        for code in row["courses"]:
            G.add_edge(row["programme"], code, rel="includes")
    return G
    # END SOLUTION


try:
    G = build_graph()
except NotImplementedError:
    G = nx.node_link_graph(U.reference("D3_4_graph"), edges="links")
U.save_reference("D3_4_graph", nx.node_link_data(G, edges="links"))
kinds = pd.Series(nx.get_node_attributes(G, "kind")).value_counts().to_dict()
print(kinds, "|", G.number_of_edges(), "edges")
U.check(kinds.get("course") == 368 and kinds.get("programme") == 30, "368 courses and 30 programmes")

# %% [markdown]
# ## 2 · Three graph queries as tools (7')
#
# ✏️ **Exercise** · Three tools, each one call: `prerequisites_of(code, transitive)` (follow `requires`
# edges, all the way when `transitive` is true), `programmes_including(code)` and `dependants_of(code)`
# (courses that list it as a direct prerequisite). Return sorted lists.

# %%
def edges(rel):
    return [(a, b) for a, b, d in G.edges(data=True) if d["rel"] == rel]


def prerequisites_of(code: str, transitive: bool = True) -> list[str]:
    """Prerequisites of a course from the course graph: direct ones, or all of them when transitive is true."""
    # BEGIN SOLUTION
    if code not in G:
        return []
    req = nx.DiGraph(edges("requires"))
    if code not in req:
        return []
    return sorted(nx.descendants(req, code)) if transitive else sorted(req.successors(code))
    # END SOLUTION


def programmes_including(code: str) -> list[str]:
    """Programmes whose study plan includes the course."""
    # BEGIN SOLUTION
    return sorted(a for a, b in edges("includes") if b == code)
    # END SOLUTION


def dependants_of(code: str) -> list[str]:
    """Courses that list this course as a direct prerequisite."""
    # BEGIN SOLUTION
    return sorted(a for a, b in edges("requires") if b == code)
    # END SOLUTION


try:
    print(QS[0]["q"], "->", prerequisites_of(QS[0]["q"].split()[-1].strip("?")))
    GRAPH_TOOLS = [prerequisites_of, programmes_including, dependants_of]
except NotImplementedError:
    GRAPH_TOOLS = None
    print("(not written yet)")

# %% [markdown]
# ## 3 · Ten questions without the graph (4')

# %%
def run(tools, standin, label):
    rows = []
    for q in QS:
        r = U.run_agent(q["q"], tools, standin, system=SYSTEM, max_steps=7, budget=4000)
        rows.append({"qid": q["qid"], "kind": q["kind"], "mode": label, "correct": U.graded(r["answer"], q["expect"]),
                     "calls": len(r["calls"]), "tokens": r["tokens"]})
    return pd.DataFrame(rows)


WITHOUT = run([T.keyword_search], T.standin_without_graph, "search only")
print(WITHOUT.to_string(index=False))

# %% [markdown]
# ## 4 · The same ten with the graph (4')
#
# ✏️ **Exercise** · `standin_with_graph(question, history)`: the stand-in's one graph call per question
# kind (all or direct prerequisites, programmes, dependants, "must ... before"), then the answer from the
# result. With a key, the hosted model chooses among the same tools by itself.

# %%
def standin_with_graph(question: str, history: list) -> dict:
    codes = re.findall(r"[A-Z]{3} \d{3}", question)
    # BEGIN SOLUTION
    target = codes[-1]
    if not history:
        if "study plan" in question:
            return {"tool": "programmes_including", "args": {"code": target}}
        if "how many courses list" in question.lower():
            return {"tool": "dependants_of", "args": {"code": target}}
        return {"tool": "prerequisites_of", "args": {"code": target, "transitive": True}}
    found = json.loads(history[-1]["result"])
    if question.lower().startswith("must"):
        return {"answer": "yes" if codes[0] in found else "no"}
    if "how many" in question.lower():
        return {"answer": f"{len(found)} courses: " + ", ".join(found)}
    return {"answer": ", ".join(found)}
    # END SOLUTION


if GRAPH_TOOLS:
    try:
        WITH = run(GRAPH_TOOLS, standin_with_graph, "with graph")
    except NotImplementedError:
        WITH = pd.DataFrame(U.reference("D3_4_with"))
else:
    WITH = pd.DataFrame(U.reference("D3_4_with"))
U.save_reference("D3_4_with", WITH.to_dict(orient="records"))
print(WITH.to_string(index=False))

# %% [markdown]
# ## 5 · Accuracy, calls and tokens (3')

# %%
BOTH = pd.concat([WITHOUT, WITH])
TABLE = BOTH.pivot_table(index="kind", columns="mode", values="correct", aggfunc="mean").round(2)
print(TABLE, "\n")
print(BOTH.groupby("mode")[["correct", "calls", "tokens"]].mean().round(2))
BOTH.to_json(U.lab_dir("D3_4") / "graph_compare.json", orient="records")
U.check(WITH.correct.mean() > WITHOUT.correct.mean(), "the graph answers more relationship questions")

# %% [markdown]
# 🔍 Which kind of question did search alone get right? A graph earns its upkeep only for the questions
# that follow links; everything else stays with search.

# %% [markdown]
# ## ✱ Stretch · A rule edge: transfer requirements
#
# Add an edge from each course to its college through its programme, and answer "what GPA does a student
# need to transfer into the college that teaches X?" in one graph call. Compare with Lab 3.2's two searches.

# %%
try:
    # BEGIN SOLUTION
    def transfer_gpa_for(code: str) -> float | None:
        """Transfer GPA of the college that teaches the course."""
        prog = next((c["programme"] for c in COURSES if c["code"] == code), None)
        col = next((b for a, b in edges("in") if a == prog), None)
        return G.nodes[col]["transfer_gpa"] if col else None


    for q in U.jsonl("agent/multistep.jsonl")[:4]:
        code = re.findall(r"[A-Z]{3} \d{3}", q["q"])[0]
        print(code, transfer_gpa_for(code), "expected", q["expect"][0])
    # END SOLUTION
except NotImplementedError:
    print("(not attempted yet: the reference solution is in solutions/)")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does search alone fail on "all prerequisites, directly or indirectly"?
#    <details><summary>Answer</summary>Each page holds one link; the chain needs one search per link, and the budget ends before the chain does.</details>
# 2. The extraction that fed the graph missed one prerequisite. What happens?
#    <details><summary>Answer</summary>The graph answers with confidence and is wrong; the graph is only as good as the extraction, so test it against the pages.</details>
# 3. When is a graph not worth building?
#    <details><summary>Answer</summary>When questions rarely follow links: one-hop facts are cheaper with search.</details>
#
# **Carry forward.** `artifacts/D3_4/graph_compare.json`, with Labs 3.1 to 3.3: the LO5 evidence.
