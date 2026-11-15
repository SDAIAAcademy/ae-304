"""Helpers shared by the AE-304 labs (import nuha as U).

The case study is Al-Nuha University and the retrieval layer of Nibras, its regulations
assistant. Everything here is plain Python you could write yourself: loading the data, Arabic
normalisation, SQLite full-text search, vector search over supplied vectors, metrics, a token
estimate, a cached call to a free hosted model with a scripted stand-in, and the warehouse.
Nothing downloads a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
VEC = DATA / "vectors"
CACHE = LABS / "cache"
ARTIFACTS = LABS / "artifacts"
QUICK = os.environ.get("AE304_QUICK", "") not in ("", "0")
IN_COLAB = "google.colab" in sys.modules
TODAY = "2026-11-01"

try:  # a local .env with OPENROUTER_API_KEY, if there is one
    from dotenv import load_dotenv
    load_dotenv(LABS / ".env")
except Exception:
    pass
if IN_COLAB and not os.environ.get("OPENROUTER_API_KEY"):  # Colab: the key from the Secrets panel
    try:
        from google.colab import userdata
        os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY") or ""
    except Exception:
        pass

MODEL = os.environ.get("AE304_MODEL", "google/gemma-4-31b-it:free")
FALLBACKS = [m for m in os.environ.get("AE304_FALLBACKS", "google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3.5-lightning:free").split(",") if m]
BASE_URL = os.environ.get("AE304_BASE_URL", "https://openrouter.ai/api/v1")


def pick(full, quick):
    """The full value, or the small one in quick mode (AE304_QUICK=1)."""
    return quick if QUICK else full


def check(ok: bool, what: str) -> None:
    """Print a self-check. With AE304_STRICT=1 (how the course author runs the solutions) a failed check stops the run."""
    print(("✅ " if ok else "❌ ") + what)
    if not ok and os.environ.get("AE304_STRICT") == "1":
        raise AssertionError(what)


def reference(name: str):
    """A supplied reference result: keeps a notebook running when an exercise is not done yet."""
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


def save_reference(name: str, value) -> None:
    """Solutions write their results here when AE304_WRITE_REF=1 (the course author's switch)."""
    if os.environ.get("AE304_WRITE_REF"):
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def lab_dir(name: str) -> Path:
    """A folder for one lab's outputs (labs/artifacts/<name>/, ignored by git)."""
    d = ARTIFACTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def jsonl(rel: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / rel).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- the corpus

_pages = None


def pages() -> list[dict]:
    """All 2,977 pages of the corpus, in a fixed order (the order of the supplied vectors)."""
    global _pages
    if _pages is None:
        _pages = jsonl("corpus/pages.jsonl")
    return _pages


def page(pid: str) -> dict:
    return PAGE_INDEX()[pid]


_pi = None


def PAGE_INDEX() -> dict:
    global _pi
    if _pi is None:
        _pi = {p["page_id"]: p for p in pages()}
    return _pi


def docs() -> dict:
    return {d["doc_id"]: d for d in jsonl("corpus/docs.jsonl")}


_labels = None


def labels() -> dict:
    """The reference labelling (Lab 2.4): each page gets its document's label, an attachment its
    parent's, and a page with no label anywhere is denied to everyone but the registrar."""
    global _labels
    if _labels is not None:
        return _labels
    ds = docs()
    out = {}
    for p in pages():
        d = ds[p["doc_id"]]
        while d.get("classification") is None and d.get("parent"):
            d = ds[d["parent"]]
        if d.get("classification") is None:
            out[p["page_id"]] = ("Restricted", ["registrar"])
        else:
            out[p["page_id"]] = (d["classification"], d["roles"])
    _labels = out
    return out


def public_pages() -> list[dict]:
    """The Public part of the corpus: what Day 1 indexes (Day 2 adds the rest behind role filters)."""
    lab = labels()
    return [p for p in pages() if lab[p["page_id"]][0] == "Public"]


def testset(version: str = "v1"):
    """(queries, qrels): 60 queries and their graded judgements {query_id: {page_id: grade}}."""
    qs = jsonl(f"testset/{version}/queries.jsonl")
    qrels: dict = {}
    for r in jsonl(f"testset/{version}/qrels.jsonl"):
        qrels.setdefault(r["query_id"], {})[r["page_id"]] = r["grade"]
    return qs, qrels


# --------------------------------------------------------------------------- Arabic text

DIACRITICS = re.compile("[\u064b-\u0652\u0670\u0640]")
AR_LETTER = re.compile("[\u0600-\u06ff]")
WORD = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    """The course's reference normalisation: presentation forms folded (NFKC), diacritics and
    tatweel removed, alef forms unified, ى to ي, ة to ه, hamza seats to their letter, lower case."""
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS.sub("", text)
    text = re.sub("[\u0623\u0625\u0622\u0671]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text.lower()


AR_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال")
AR_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "يه", "ه", "ي")


def light_stem(word: str) -> str:
    """Light stemming: strip one common prefix and one common suffix, keep at least three letters."""
    if AR_LETTER.match(word):
        for p in AR_PREFIXES:
            if word.startswith(p) and len(word) - len(p) >= 3:
                word = word[len(p):]
                break
        if word.startswith(("و", "ب", "ف", "ل")) and len(word) >= 5:
            word = word[1:]
        for s in AR_SUFFIXES:
            if word.endswith(s) and len(word) - len(s) >= 3:
                word = word[: -len(s)]
                break
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def analyse(text: str) -> list[str]:
    """Normalise, split into words, light-stem: what the normalised index stores."""
    return [light_stem(w) for w in WORD.findall(normalise(text))]


# --------------------------------------------------------------------------- lexical search

def fts_index(rows: list[dict], analysed: bool = True, path: str = ":memory:") -> sqlite3.Connection:
    """A SQLite FTS5 index over rows with "page_id" (or "chunk_id") and "text". With analysed=True
    the text is normalised and stemmed in Python first; otherwise FTS5's own tokeniser sees raw text."""
    con = sqlite3.connect(path)
    con.execute("DROP TABLE IF EXISTS idx")
    con.execute("CREATE VIRTUAL TABLE idx USING fts5(id UNINDEXED, body)")
    key = "page_id" if "page_id" in rows[0] else "chunk_id"
    con.executemany("INSERT INTO idx VALUES (?, ?)",
                    [(r[key], " ".join(analyse(r["text"])) if analysed else r["text"]) for r in rows])
    con.commit()
    return con


def fts_query(con: sqlite3.Connection, query: str, k: int = 10, analysed: bool = True) -> list[tuple[str, float]]:
    """Top k (id, score) by FTS5's bm25(), higher is better. Query words are OR-ed."""
    words = analyse(query) if analysed else WORD.findall(query.lower())
    words = [w for w in dict.fromkeys(words) if w]
    if not words:
        return []
    match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
    rows = con.execute("SELECT id, bm25(idx) FROM idx WHERE idx MATCH ? ORDER BY bm25(idx) LIMIT ?", (match, k)).fetchall()
    return [(i, -s) for i, s in rows]


# --------------------------------------------------------------------------- dense search

MODELS = {"e5": "intfloat/multilingual-e5-small", "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
PREFIX = {"e5": ("query: ", "passage: "), "minilm": ("", "")}
_vec, _qvec = {}, {}


def page_vectors(model: str) -> np.ndarray:
    """Supplied unit-length vectors of every page (rows in the order of pages()), float32."""
    if model not in _vec:
        _vec[model] = np.load(VEC / f"{model}_pages.npy").astype(np.float32)
    return _vec[model]


def _query_store(model: str):
    if model not in _qvec:
        texts = json.loads((VEC / "queries.json").read_text(encoding="utf-8"))
        _qvec[model] = ({t: i for i, t in enumerate(texts)}, np.load(VEC / f"{model}_queries.npy").astype(np.float32))
    return _qvec[model]


_live = {}


def live_model(model: str):
    """The real embedding model, loaded only in Colab or with AE304_MODELS=load (about 470 MB)."""
    if not (IN_COLAB or os.environ.get("AE304_MODELS") == "load"):
        return None
    if model not in _live:
        from sentence_transformers import SentenceTransformer
        _live[model] = SentenceTransformer(MODELS[model], device="cpu")
    return _live[model]


def embed_query(text: str, model: str = "e5") -> np.ndarray | None:
    """A query vector: from the supplied store when the text is known, live when a model is loaded,
    otherwise None (the caller falls back to lexical search and says so)."""
    index, mat = _query_store(model)
    if text in index:
        return mat[index[text]]
    m = live_model(model)
    if m is None:
        if os.environ.get("AE304_COLLECT"):  # the course author's switch: record texts that need a vector
            f = DATA / "agent" / "standin_queries.json"
            known = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
            if text not in known:
                f.write_text(json.dumps(known + [text], ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
        return None
    return m.encode([PREFIX[model][0] + text], normalize_embeddings=True)[0].astype(np.float32)


def dense_query(qvec: np.ndarray, model: str = "e5", k: int = 10, allowed: np.ndarray | None = None) -> list[tuple[str, float]]:
    sims = page_vectors(model) @ qvec
    if allowed is not None:
        sims = np.where(allowed, sims, -np.inf)
    top = np.argsort(-sims)[:k]
    ps = pages()
    return [(ps[i]["page_id"], float(sims[i])) for i in top if np.isfinite(sims[i])]


# --------------------------------------------------------------------------- fusion, metrics

def rrf(runs: list[list[str]], k: int = 60, depth: int = 100) -> list[str]:
    """Reciprocal rank fusion of ranked id lists."""
    score: dict = {}
    for run in runs:
        for rank, pid in enumerate(run[:depth], 1):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda p: -score[p])


def recall_at(ranked, grades, k=10, rel=2):
    relevant = {p for p, g in grades.items() if g >= rel}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def rr(ranked, grades, rel=2):
    return next((1.0 / i for i, p in enumerate(ranked, 1) if grades.get(p, 0) >= rel), 0.0)


def ndcg_at(ranked, grades, k=10):
    dcg = sum((2 ** grades.get(p, 0) - 1) / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def evaluate(run: dict, queries: list[dict], qrels: dict, by: str = "lang"):
    """Mean recall@10, MRR and nDCG@10 of a run {query_id: [page ids]}, overall and by a field."""
    import pandas as pd
    rows = []
    for q in queries:
        r = run.get(q["query_id"], [])
        g = qrels.get(q["query_id"], {})
        rows.append({by: q[by], "recall@10": recall_at(r, g), "MRR": rr(r, g), "nDCG@10": ndcg_at(r, g)})
    df = pd.DataFrame(rows)
    out = df.groupby(by).mean(numeric_only=True)
    out.loc["all"] = df.mean(numeric_only=True)
    return out.round(3)


def p95(ms: list[float]) -> float:
    return float(np.percentile(ms, 95)) if ms else 0.0


def rerank_scores() -> dict:
    """Supplied cross-encoder scores {query_id: {page_id: score}} for the candidates of each query."""
    return json.loads((DATA / "reference" / "rerank_scores.json").read_text(encoding="utf-8"))


def timings() -> dict:
    """Times measured once on a laptop processor when the supplied vectors were made."""
    return json.loads((DATA / "reference" / "timings.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- tokens

def count_tokens(obj) -> int:
    """An estimate of tokens: about four Latin letters, or two and a half Arabic letters, per token.
    Real tokenisers differ; the estimate is for comparing two designs, not for billing."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    n = 0
    for w in re.findall(r"\w+|[^\w\s]", text):
        if AR_LETTER.match(w):
            n += max(1, round(len(w) / 2.5))
        elif w.isdigit():
            n += math.ceil(len(w) / 3)
        elif w[0].isalnum() or w[0] == "_":
            n += max(1, round(len(w) / 4))
        else:
            n += 1
    return n


# --------------------------------------------------------------------------- the model

class ModelUnavailable(RuntimeError):
    pass


_state = {"mode": None}


def model_mode() -> str:
    """'live' when a key for the hosted model is set and works, otherwise 'standin'."""
    if _state["mode"] is None:
        _state["mode"] = "live" if os.environ.get("OPENROUTER_API_KEY") and os.environ.get("AE304_STANDIN") != "1" else "standin"
        if _state["mode"] == "standin":
            print("model: scripted stand-in (no OPENROUTER_API_KEY, or AE304_STANDIN=1). Numbers below come from the stand-in.")
        else:
            print("model:", MODEL)
    return _state["mode"]


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None, **kw) -> dict:
    """One cached Chat Completions call to the free hosted model, with fallbacks.
    Returns {"content", "tool_calls", "usage", "model"}. Raises ModelUnavailable when no key works."""
    if model_mode() != "live":
        raise ModelUnavailable("no working key")
    req = dict(messages=messages, tools=tools, **kw)
    key = hashlib.sha256(json.dumps([model or MODEL, req], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    f = CACHE / f"{key}.json"
    if f.exists() and os.environ.get("AE304_CACHE", "1") != "0":
        return json.loads(f.read_text(encoding="utf-8"))
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], timeout=60)
    last = None
    for m in [model or MODEL] + FALLBACKS:
        for attempt in range(3):
            try:
                args = dict(model=m, messages=messages, **kw)
                if tools:
                    args["tools"] = tools
                r = client.chat.completions.create(**args)
                msg = r.choices[0].message
                out = {"content": msg.content or "", "model": m,
                       "tool_calls": [{"id": t.id, "name": t.function.name, "arguments": t.function.arguments} for t in (msg.tool_calls or [])],
                       "usage": {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens} if r.usage else {}}
                CACHE.mkdir(exist_ok=True)
                f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                return out
            except Exception as e:  # rate limits and busy free models: wait, then the next model
                last = e
                if "401" in str(e) or "403" in str(e):
                    _state["mode"] = "standin"
                    print("model: the key was refused; switching to the scripted stand-in.")
                    raise ModelUnavailable(str(e))
                time.sleep(2 * (attempt + 1))
    raise ModelUnavailable(str(last))


def tool_schema(fn) -> dict:
    """A Chat Completions tool definition from a typed function with a docstring."""
    import inspect
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    props, req = {}, []
    for name, p in sig.parameters.items():
        ann = p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", "")
        t = {"int": "integer", "float": "number", "bool": "boolean"}.get(ann, "string")
        props[name] = {"type": t}
        if p.default is inspect.Parameter.empty:
            req.append(name)
    return {"type": "function", "function": {"name": fn.__name__, "description": doc.split("\n\n")[0],
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


def run_agent(question: str, tools: list, standin, system: str = "", max_steps: int = 6, budget: int | None = None) -> dict:
    """A small tool loop. The model is the hosted one when a key works, otherwise `standin`, a
    function (question, history) -> {"tool": name, "args": {...}} or {"answer": text}.
    Returns {"answer", "calls": [(name, args)], "tokens": estimated tokens sent, "model"}."""
    by_name = {t.__name__: t for t in tools}
    history, calls, sent = [], [], 0
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    used = "standin"
    for _ in range(max_steps):
        sent += count_tokens(messages)
        action = None
        if model_mode() == "live":
            try:
                r = chat(messages, tools=[tool_schema(t) for t in tools])
                used = r["model"]
                if r["tool_calls"]:
                    tc = r["tool_calls"][0]
                    action = {"tool": tc["name"], "args": json.loads(tc["arguments"] or "{}"), "id": tc["id"]}
                else:
                    action = {"answer": r["content"]}
            except ModelUnavailable:
                action = None
        if action is None:
            used = "standin"
            action = standin(question, history)
        if "answer" in action:
            return {"answer": action["answer"], "calls": calls, "tokens": sent, "model": used}
        name, args = action["tool"], action.get("args", {})
        try:
            result = by_name[name](**args)
        except Exception as e:  # an error the model can act on
            result = {"error": f"{type(e).__name__}: {e}"}
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if budget is not None:
            room = max(0, budget - count_tokens(messages) - 50)
            if count_tokens(text) > room:
                text = text[: max(0, int(len(text) * room / max(1, count_tokens(text))))] + " ... [cut to fit the context budget]"
        calls.append((name, args))
        history.append({"tool": name, "args": args, "result": text})
        tcid = action.get("id", f"call_{len(calls)}")
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": tcid, "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]})
        messages.append({"role": "tool", "tool_call_id": tcid, "content": text})
    return {"answer": "", "calls": calls, "tokens": sent, "model": used}


def graded(answer: str, expect: list[str]) -> bool:
    """True when every expected string appears in the answer (case and Arabic forms ignored)."""
    a = normalise(answer)
    return all(normalise(e) in a for e in expect)


# --------------------------------------------------------------------------- the warehouse

def warehouse(path: str = ":memory:"):
    """A DuckDB database with the enrolment statistics warehouse loaded from data/warehouse/."""
    import duckdb
    con = duckdb.connect(path)
    wh = (DATA / "warehouse").as_posix()
    for t in ("colleges", "programmes", "terms", "students"):
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_csv_auto('{wh}/{t}.csv', header=true)")
    con.execute(f"CREATE OR REPLACE TABLE enrolments AS SELECT * FROM read_csv_auto('{wh}/enrolments.csv.gz', header=true)")
    return con


def same_value(a, b) -> bool:
    """Execution accuracy's comparison: numbers within 0.001 or 0.5%, anything else as text."""
    try:
        return abs(float(a) - float(b)) <= max(0.001, 0.005 * abs(float(b)))
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def semantic_layer(con):
    """Create the governed views on a warehouse connection and return the metric catalog."""
    con.execute((DATA / "semantic" / "views.sql").read_text(encoding="utf-8"))
    return json.loads((DATA / "semantic" / "metrics.json").read_text(encoding="utf-8"))
