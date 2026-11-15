"""Nibras's search tools and the scripted stand-in model of Day 3 (import nibras_tools as T).

The tools are the reference versions of what Lab 3.1 asks you to write. The stand-in functions are
not a model: they are short scripts that follow the same steps a model is asked to follow (pick a
tool, read the result, search again, answer), so the labs run and are graded when no key for the
hosted model works. Numbers measured with them describe the stand-in, and say so.
"""

from __future__ import annotations

import json
import re

import numpy as np

import nuha as U

PUBLIC = [p for p in U.public_pages()]
BY_PATH = {p["path"]: p for p in PUBLIC}
_POS = {p["page_id"]: i for i, p in enumerate(U.pages())}
ALLOWED = np.zeros(len(_POS), dtype=bool)
ALLOWED[[_POS[p["page_id"]] for p in PUBLIC]] = True
_LEX = None
CODE = re.compile(r"\b[A-Z]{3} \d{3}\b")
CIRC = re.compile(r"\b20\d\d-\d{3}\b")


def _lex():
    global _LEX
    if _LEX is None:
        _LEX = U.fts_index(PUBLIC)
    return _LEX


def _hit(p, n=400):
    return {"path": p["path"], "title": p["title"], "text": " ".join(p["text"].split())[:n]}


# --------------------------------------------------------------------------- the tools

def list_files(folder: str) -> list[str]:
    """List the files under a folder of the corpus, such as "circulars/2025", "study-plans/eng" or "guide/ar".
    Cheapest tool: names only, no text. Use it for questions about what exists or how many."""
    folder = folder.strip("/").lower()
    names = sorted(p for p in BY_PATH if p.lower().startswith(folder + "/") or (folder == "" and "/" in p))
    return names[:400] if names else [f"no files under '{folder}'; top folders: " + ", ".join(sorted({p.split('/')[0] for p in BY_PATH}))]


def read_file(path: str) -> str:
    """Read one file of the corpus in full, by the path that list_files or a search returned."""
    p = BY_PATH.get(path)
    return p["text"] if p else f"no file '{path}'"


def keyword_search(query: str, k: int = 5) -> list[dict]:
    """Keyword search (BM25, Arabic normalised). Best for exact words: course codes such as "CSC 301",
    circular numbers such as "2025-118", article numbers and names. English or Arabic."""
    return [_hit(U.page(i)) for i, _ in U.fts_query(_lex(), query, k)]


def vector_search(query: str, k: int = 5) -> list[dict]:
    """Search by meaning (multilingual embeddings). Best for questions in everyday words, paraphrases,
    dialect, or a question in one language whose answer may be in the other."""
    qv = U.embed_query(query, "e5")
    if qv is None:
        return [{"note": "no query vector on this machine: keyword results instead"}] + keyword_search(query, k)
    return [_hit(U.page(i)) for i, _ in U.dense_query(qv, "e5", k, ALLOWED)]


def hybrid_search(query: str, k: int = 5, chars: int = 400) -> list[dict]:
    """Keyword and meaning search fused (RRF): the default single search."""
    lex = [i for i, _ in U.fts_query(_lex(), query, 30)]
    qv = U.embed_query(query, "e5")
    dense = [i for i, _ in U.dense_query(qv, "e5", 30, ALLOWED)] if qv is not None else []
    return [_hit(U.page(i), chars) for i in U.rrf([lex, dense])[:k]]


# --------------------------------------------------------------------------- stand-in: tool choice (Lab 3.1)

def standin_pick(question: str, history: list) -> dict:
    """Scripted stand-in for Lab 3.1: choose the cheapest tool that fits, read one result, answer."""
    q = question.lower()
    if not history:
        if re.search(r"\bhow many circulars\b", q):
            return {"tool": "list_files", "args": {"folder": "circulars/" + re.search(r"20\d\d", q).group()}}
        if "study plans" in q and "college of" in q:
            cid = {"engineering": "eng", "computing": "cci", "business": "bus", "health": "hsc", "sciences": "sci", "arts": "art"}
            key = next(v for k, v in cid.items() if k in q)
            return {"tool": "list_files", "args": {"folder": f"study-plans/{key}"}}
        if "guide pages" in q and "arabic" in q:
            return {"tool": "list_files", "args": {"folder": "guide/ar"}}
        if CODE.search(question) or CIRC.search(question):
            return {"tool": "keyword_search", "args": {"query": question}}
        return {"tool": "vector_search", "args": {"query": question}}
    last = history[-1]
    if last["tool"] == "list_files":
        names = json.loads(last["result"])
        if "circulars" in last["args"]["folder"]:
            nos = sorted({m.group() for n in names for m in [CIRC.search(n)] if m})
            return {"answer": f"{len(nos)} circulars: " + ", ".join(nos)}
        if "study-plans" in last["args"]["folder"]:
            progs = sorted({n.split("/")[2] for n in names})
            return {"answer": f"{len(progs)} study plans: " + ", ".join(progs)}
        return {"answer": "Arabic guide pages: " + ", ".join(sorted({n.split('/')[-1].split('-faq')[0].replace('.md', '') for n in names}))}
    hits = [h for h in json.loads(last["result"]) if "text" in h]
    return {"answer": hits[0]["text"] if hits else "I found nothing."}


# --------------------------------------------------------------------------- stand-in: iterative search (Lab 3.2)

def _texts(notes):
    return " \n".join(h["text"] for n in notes for h in n["top"])


def _facts(text):
    """What the notes say, read with the patterns of the course and plan pages."""
    def row(c):  # a study-plan row: | code | name | credits | prerequisites |
        m = re.search(r"\| " + re.escape(c) + r" \| [^|]+ \| (\d) \| ([^|]*)\|", text)
        return (m.group(1), (CODE.findall(m.group(2)) or ["none"])[0]) if m else (None, None)

    def prereq(c):
        m = re.search(re.escape(c) + r" · [^\n]*?Prerequisite: ([A-Z]{3} \d{3}|none)", text)
        return m.group(1) if m else row(c)[1]

    def credits(c):
        m = re.search(re.escape(c) + r" · [^\n]*?Credit hours: (\d)", text)
        return m.group(1) if m else row(c)[0]

    def college(c):
        m = re.search(re.escape(c) + r" · [^\n]*?Offered by the Department of [^,]+, (College of [^.]+)\.", text)
        return m.group(1) if m else None

    def gpa(col):
        m = re.search(re.escape(col) + r" if their cumulative grade point average is at least (\d\.\d\d)", text)
        return m.group(1) if m else None

    def semester(c):
        for chunk in text.split(" \n"):
            m = re.search(r"\| " + re.escape(c) + r" \| [A-Za-z]", chunk)  # the code cell of a row, not a prerequisite
            if chunk.startswith("Study plan") and m:
                sem = re.findall(r"Semester (\d)", chunk[: m.start()])
                if sem:
                    return sem[-1]
        return None
    return prereq, credits, college, gpa, semester


def standin_decide(question: str, notes: list, final: bool = False) -> dict:
    """Scripted stand-in for Lab 3.2: decompose a multi-step question into searches, one hop at a time.
    Returns {"search": query} or {"answer": text}; with final=True it answers from what it has."""
    code = CODE.search(question).group()
    prereq, credits, college, gpa, semester = _facts(_texts(notes))
    asked = {n["query"] for n in notes}

    def need(query, answer_if_stuck="I could not find it."):
        if not notes:  # the first search is the question itself
            return {"search": question}
        if final or query in asked:
            return {"answer": answer_if_stuck}
        return {"search": query}

    if "transfer into the college" in question:
        col = college(code)
        if not col:
            return need(f"{code} course description")
        g = gpa(col)
        return {"answer": f"At least {g} out of 5.00 ({col})."} if g else need(f"{col} handbook admission by transfer")
    pre = prereq(code)
    if not pre:
        return need(f"{code} course description prerequisite")
    if "credit hours is the first prerequisite" in question:
        ch = credits(pre)
        return {"answer": f"{pre} has {ch} credit hours."} if ch else need(f"{pre} course description credit hours")
    if "semester of its own programme" in question:
        s = semester(pre)
        return {"answer": f"{pre} is taught in Semester {s}."} if s else need(f"study plan {pre}")
    if "prerequisite of the first prerequisite" in question:
        p2 = prereq(pre)
        return {"answer": f"The prerequisite of {pre} is {p2}."} if p2 else need(f"{pre} course description prerequisite")
    return {"answer": "I could not find it."}


# --------------------------------------------------------------------------- tools that return a lot (Lab 3.3)

COURSE_PAGES = [p for p in PUBLIC if p["type"] == "course" and p["lang"] == "en"]
PROG_PREFIX = {p[0]: p[4] for p in __import__("nuha_content").PROGRAMMES}


def list_courses(programme: str = "", college: str = "") -> str:
    """Every course page of a programme (by id, such as "CS") or of a college (by id, such as "ENG"), in full.
    The result is large."""
    if programme:
        pre = PROG_PREFIX.get(programme.upper(), "")
        rows = [p for p in COURSE_PAGES if p["title"].startswith(pre + " ")]
    else:
        rows = [p for p in COURSE_PAGES if p["college"] == college.upper()]
    return "\n\n".join(p["text"] for p in rows)


def standin_count(question: str, history: list) -> dict:
    """Scripted stand-in for Lab 3.3: fetch the list, then count in what it can see (or grep the file)."""
    import nuha_content as C
    pattern = "Prerequisite: none" if "no prerequisite" in question else ("Credit hours: 4" if "4 credit" in question else "Credit hours: 1")
    if not history:
        prog = next((p[0] for p in C.PROGRAMMES if f"the {p[2]} programme" in question), "")
        col = next((c[0] for c in C.COLLEGES if c[1] in question), "")
        return {"tool": "list_courses", "args": {"programme": prog} if prog else {"college": col}}
    last = history[-1]
    if last["tool"] == "list_courses":
        try:
            stub = json.loads(last["result"])
        except json.JSONDecodeError:
            stub = None
        if isinstance(stub, dict) and "file" in stub:
            return {"tool": "grep_file", "args": {"file": stub["file"], "pattern": pattern}}
        return {"answer": f"{last['result'].count(pattern)} courses."}
    found = json.loads(last["result"])
    return {"answer": f"{found['count']} courses."}


# --------------------------------------------------------------------------- relationship questions (Lab 3.4)

def standin_without_graph(question: str, history: list) -> dict:
    """Scripted stand-in for Lab 3.4 with search only: follow prerequisites page by page, within the step budget."""
    codes = CODE.findall(question)
    target = codes[-1]
    seen = " \n".join(h["result"] for h in history)
    if "how many courses list" in question.lower():
        if not history:
            return {"tool": "keyword_search", "args": {"query": f"Prerequisite {target}", "k": 10}}
        hits = json.loads(history[-1]["result"])
        n = sum(re.search(r"Prerequisite: [^\n]*" + re.escape(target), h["text"]) is not None and not h["text"].startswith(target) for h in hits)
        return {"answer": f"{n} courses."}
    if "include" in question and "study plan" in question:
        if not history:
            return {"tool": "keyword_search", "args": {"query": f"study plan {target}", "k": 10}}
        hits = json.loads(history[-1]["result"])
        progs = sorted({h["path"].split("/")[2].upper() for h in hits if h["path"].startswith("study-plans/") and target in h["text"]})
        return {"answer": "Programmes: " + ", ".join(progs)}
    # prerequisites, followed breadth first from the course pages and plan rows the searches returned
    texts = seen.replace("\\u00b7", "·")  # results may be cut to the step budget, so read them as text

    def prereqs_of(c):
        m = re.search(re.escape(c) + r" · [^\n]*?Prerequisite: (.+?) Offered", texts)
        if m:
            return set(CODE.findall(m.group(1)))
        m = re.search(r"\| " + re.escape(c) + r" \| [^|]+ \| \d \| ([^|]*)\|", texts)
        return set(CODE.findall(m.group(1))) if m else None

    known, frontier, unread = set(), [target], []
    while frontier:
        c = frontier.pop(0)
        p = prereqs_of(c)
        if p is None:
            unread.append(c)
            continue
        for x in sorted(p - known):
            known.add(x)
            frontier.append(x)
    asked = {h["args"]["query"] for h in history}
    todo = [f"{c} course description prerequisite" for c in unread if f"{c} course description prerequisite" not in asked]
    if todo and len(history) < 6:
        return {"tool": "keyword_search", "args": {"query": todo[0], "k": 3}}
    if question.lower().startswith("must"):
        return {"answer": "yes" if codes[0] in known else "no, not found"}
    return {"answer": "Before " + target + ": " + ", ".join(sorted(known))}
