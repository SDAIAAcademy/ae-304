"""Figures for two Day 1 slides: rewriting a query with the corpus's words (Session 2) and an embedding
index caught half-way through a change of model (Session 3).

    uv run python _tools/rewrite_and_migration.py      (from labs/)

Reads the committed corpus, test set v1 and supplied vectors; writes nothing. Deterministic (seed 304).
The examples here are parallel to Labs 1.2 and 1.3: a dialect query that is not in test set v1, and a
partial migration, which the labs do not run.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
os.chdir(LABS)
sys.path.insert(0, str(LABS))

import numpy as np  # noqa: E402

import nuha as U  # noqa: E402


def vocabulary_gap() -> None:
    pages = U.public_pages()
    df = Counter()
    for p in pages:
        df.update(set(U.analyse(p["text"])))
    con = U.fts_index(pages)
    query = "ماده ما كملتها وش يكتبون بالسجل"  # incomplete grade, Gulf dialect; not in test set v1
    print("Session 2 · query words and the number of Public pages that hold each (after analysis)")
    print("  ", [(w, df[w]) for w in U.analyse(query)])
    for extra in ["", "ناقص", "معلق", "تقدير غير مكتمل"]:
        words = U.analyse(extra)
        top = [i for i, _ in U.fts_query(con, (query + " " + extra).strip(), 10)]
        rank = top.index("P0086") + 1 if "P0086" in top else None
        print(f"  + {extra or '(nothing)':<16} pages per added word {[df[w] for w in words]}  "
              f"top 1: {U.page(top[0])['title']}  | Article 43 at rank {rank}")


def partial_migration() -> None:
    P = U.pages()
    L = U.labels()
    public = np.array([L[p["page_id"]][0] == "Public" for p in P])
    queries, qrels = U.testset("v1")
    old, new = U.page_vectors("minilm"), U.page_vectors("e5")
    qv = np.stack([U.embed_query(q["text"], "e5") for q in queries])
    order = np.random.default_rng(304).permutation(len(P))
    print("Session 3 · moving the index from MiniLM to e5, queries already embedded with e5")
    for share in (0.25, 0.5, 0.75, 1.0):
        done = np.zeros(len(P), bool)
        done[order[: int(share * len(P))]] = True
        M = np.where(done[:, None], new, old)
        run, from_new = {}, []
        for q, v in zip(queries, qv):
            top = np.argsort(-np.where(public, M @ v, -np.inf))[:20]
            run[q["query_id"]] = [P[i]["page_id"] for i in top]
            from_new.append(done[top[:10]].mean())
        e = U.evaluate(run, queries, qrels).loc["all"]
        print(f"  {share:>4.0%} re-embedded: nDCG@10 {e['nDCG@10']:.3f}, recall@10 {e['recall@10']:.3f}, "
              f"top 10 from re-embedded pages {np.mean(from_new):.0%}")
    v = qv[0]
    for name, M in (("e5 pages", new), ("MiniLM pages", old)):
        s = M[public] @ v
        print(f"  cosine of query {queries[0]['query_id']} (e5) with {name}: {s.min():.2f} to {s.max():.2f}")


if __name__ == "__main__":
    vocabulary_gap()
    partial_migration()
