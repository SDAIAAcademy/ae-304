# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "sentence-transformers>=3.0"]
# ///
"""Write the supplied vectors, rerank scores and timings (the course author runs this once).

    uv run _tools/make_vectors.py            # everything: pages, queries, rerank scores, timings
    uv run _tools/make_vectors.py --queries  # only the query vectors (after new question files)
    uv run _tools/make_vectors.py --noprefix # only the e5 vectors made without prefixes

It loads two open multilingual embedding models and a multilingual cross-encoder with Sentence
Transformers, on a laptop processor. Participants never run it: the labs read what it writes.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import nuha as U  # noqa: E402

VEC = U.VEC
QUERY_FILES = ["testset/v1/queries.jsonl", "agent/tool_questions.jsonl", "agent/multistep.jsonl", "agent/large_questions.jsonl",
               "graph/relationship_questions.jsonl", "versions/changed_questions.jsonl", "access/leak_tests.jsonl", "plans/table_questions.jsonl"]


def query_texts() -> list[str]:
    texts = []
    for f in QUERY_FILES:
        for r in U.jsonl(f):
            texts.append(r.get("text") or r.get("q"))
            if r.get("q_ar"):
                texts.append(r["q_ar"])
    extra = U.DATA / "agent" / "standin_queries.json"
    if extra.exists():
        texts += json.loads(extra.read_text(encoding="utf-8"))
    return list(dict.fromkeys(texts))


def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    VEC.mkdir(parents=True, exist_ok=True)
    only_queries = "--queries" in sys.argv
    if "--noprefix" in sys.argv:
        e5 = SentenceTransformer(U.MODELS["e5"], device="cpu")
        qs, _ = U.testset()
        np.save(VEC / "e5_noprefix_pages.npy", e5.encode([p["text"] for p in U.pages()], normalize_embeddings=True, batch_size=32).astype(np.float16))
        np.save(VEC / "e5_noprefix_queries.npy", e5.encode([q["text"] for q in qs], normalize_embeddings=True).astype(np.float16))
        return
    models = {k: SentenceTransformer(v, device="cpu") for k, v in U.MODELS.items()}
    texts = query_texts()
    (VEC / "queries.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    for k, m in models.items():
        q = m.encode([U.PREFIX[k][0] + t for t in texts], normalize_embeddings=True, batch_size=32)
        np.save(VEC / f"{k}_queries.npy", q.astype(np.float16))
    print("queries:", len(texts))
    if only_queries:
        return
    pages = U.pages()
    timings = {}
    pub = [p["page_id"] for p in U.public_pages()]
    sample = sorted(random.Random(304).sample(pub, 300))
    (VEC / "sample300.json").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    by_id = {p["page_id"]: p for p in pages}
    for k, m in models.items():
        t = time.perf_counter()
        m.encode([U.PREFIX[k][1] + by_id[i]["text"] for i in sample], normalize_embeddings=True, batch_size=32)
        timings[f"{k}_embed_300_pages_s"] = round(time.perf_counter() - t, 1)
        v = m.encode([U.PREFIX[k][1] + p["text"] for p in pages], normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        np.save(VEC / f"{k}_pages.npy", v.astype(np.float16))
        lat = []
        qs, _ = U.testset()
        for q in qs:
            t = time.perf_counter()
            m.encode([U.PREFIX[k][0] + q["text"]], normalize_embeddings=True)
            lat.append((time.perf_counter() - t) * 1000)
        timings[f"{k}_query_embed_ms_p50"] = round(float(np.percentile(lat, 50)), 1)
        timings[f"{k}_query_embed_ms_p95"] = round(float(np.percentile(lat, 95)), 1)
        print(k, "pages", v.shape)
    # e5 without its prefixes, for the sample and the test set: the planted mistake of Lab 1.3
    e5 = models["e5"]
    np.save(VEC / "e5_noprefix_pages.npy", e5.encode([p["text"] for p in pages], normalize_embeddings=True, batch_size=32).astype(np.float16))
    qs, qrels = U.testset()
    np.save(VEC / "e5_noprefix_queries.npy", e5.encode([q["text"] for q in qs], normalize_embeddings=True).astype(np.float16))
    # rerank scores for each test query's candidates: the union of three first-stage top 30s
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")
    con = U.fts_index(U.public_pages())
    allowed = np.array([U.labels()[p["page_id"]][0] == "Public" for p in pages])
    scores, lat = {}, []
    for q in qs:
        cands = [i for i, _ in U.fts_query(con, q["text"], 30)]
        for k in ("e5", "minilm"):
            cands += [i for i, _ in U.dense_query(U.embed_query(q["text"], k), k, 30, allowed)]
        cands = list(dict.fromkeys(cands))
        pairs = [(q["text"], by_id[i]["text"][:1200]) for i in cands]
        t = time.perf_counter()
        s = ce.predict(pairs[:30], batch_size=30)
        lat.append((time.perf_counter() - t) * 1000)
        s = list(s) + list(ce.predict(pairs[30:], batch_size=32)) if len(pairs) > 30 else list(s)
        scores[q["query_id"]] = {i: round(float(x), 4) for i, x in zip(cands, s)}
    timings["rerank_30_pairs_ms_p50"] = round(float(np.percentile(lat, 50)), 1)
    timings["rerank_30_pairs_ms_p95"] = round(float(np.percentile(lat, 95)), 1)
    timings["measured_on"] = "a laptop processor, Sentence Transformers, batch of 32 (pages) or one query at a time"
    (U.REF).mkdir(parents=True, exist_ok=True)
    (U.REF / "rerank_scores.json").write_text(json.dumps(scores) + "\n", encoding="utf-8")
    (U.REF / "timings.json").write_text(json.dumps(timings, indent=1) + "\n", encoding="utf-8")
    print(timings)


if __name__ == "__main__":
    main()
