"""Write the supplied runs and pools that need no model (the course author runs this once,
after prepare_data.py and _tools/make_vectors.py).

    uv run python _tools/make_reference.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import nuha as U  # noqa: E402


def main():
    qs, qrels = U.testset()
    pub = U.public_pages()
    raw = U.fts_index(pub, analysed=False)
    base = {q["query_id"]: [i for i, _ in U.fts_query(raw, q["text"], 20, analysed=False)] for q in qs}
    runs = U.DATA / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "baseline.json").write_text(json.dumps({"method": "SQLite FTS5 BM25 on raw text, top 20", "test_set": "v1", "run": base}) + "\n", encoding="utf-8")
    allowed = np.array([U.labels()[p["page_id"]][0] == "Public" for p in U.pages()])
    r = random.Random(304)
    pools = []
    en = [q for q in qs if q["lang"] == "en"][:20]
    ar = [q for q in qs if q["lang"] == "ar"][:20]
    for q in en + ar:
        dense = [i for i, _ in U.dense_query(U.embed_query(q["text"], "e5"), "e5", 10, allowed)]
        cands = []
        for c in (base[q["query_id"]][:1] + dense[:1] + [x for x in base[q["query_id"]] + dense if qrels[q["query_id"]].get(x, 0) == 0][:1]
                  + base[q["query_id"]][1:] + dense[1:]):
            if c not in cands:
                cands.append(c)
            if len(cands) == 3:
                break
        r.shuffle(cands)
        pools.append({"query_id": q["query_id"], "lang": q["lang"], "type": q["type"], "text": q["text"], "candidates": cands})
    with (U.DATA / "testset" / "v1" / "pools.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for p in pools:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("baseline and", len(pools), "pools written")


if __name__ == "__main__":
    main()
