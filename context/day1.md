---
course: "Retrieval and Knowledge Engineering for Language Model Systems"
code: "AE-304"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-304/slides/day1.html"
---

# Retrieval and Knowledge Engineering for Language Model Systems

## Day 1: Measure and search

#### [day1#title] Measure and search

*SDAIA Academy · Retrieval and Knowledge Engineering for Language Model Systems*

Day 1

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-304

#### [day1#agenda] Measure and search

*Today*

- **Session 1 · Retrieval is measured on its own**: Judge relevance in grades, pool candidates, and score the retrieval layer apart from the answers, by language
- **Session 2 · Lexical search and Arabic text**: Inverted indexes and BM25 in SQLite, and what Arabic spelling and dialect do to word matching
- **Session 3 · Dense retrieval and embedding choice**: Multilingual embeddings, their conventions, exact and approximate indexes, and what each costs
- **Session 4 · Hybrid search and reranking**: Fuse lexical and dense rankings, rerank the top with a cross-encoder, and weigh the gain against the latency

### Session 1: Retrieval is measured on its own

#### [day1#s1] Retrieval is measured on its own

*Session 1*

- Judge graded relevance against a written guideline
- Compute recall at 10, reciprocal rank and nDCG at 10
- Report retrieval quality by language and query type

#### [day1#nibras-at-al-nuha] Nibras answers Al-Nuha University's students from 2,977 pages

*The case study for four days*

- **Regulations** in Arabic and English, amended by circulars over five years
- **Course pages and study plans** with tables, some of them scanned
- **Service guides** and news, written for students
- **A warehouse** of enrolments for statistical questions
- Some pages are **Restricted**: staff only

- **Your role.** Kawthar Al-Beladi, the retrieval engineer. The model is not your job; what it reads is.
- **The decision owner.** Dr. Lubna Al-Harithi, vice-rector for digital transformation, wants numbers before the pilot widens.

#### [day1#retrieval-apart-from-answers] A good answer can hide bad retrieval, and the reverse

*Why measure retrieval on its own*

1. **Question** "How many absences before I am denied the final exam?"
2. **Retrieval** the top 5 pages the model will read
3. **Generation** the answer written from those pages

- **Judging only answers.** A model can answer from memory with the page missing, or paraphrase a superseded rule fluently.
- **Judging retrieval.** Was the page that answers it in the top 10, and how high? Cheap, repeatable, no model needed.

#### [day1#graded-relevance] Relevance has degrees: grade it 0 to 3

*Relevance in grades*

| Grade | Meaning | Judged for "deadline to withdraw from a course" |
| --- | --- | --- |
| 3 | The page that answers it | Article 25: Withdrawal from a course |
| 2 | Answers part of it, or needs another page | Circular 2026-031, which amends Article 25 |
| 1 | On topic, does not answer | Article 26: Postponement of study |
| 0 | Not relevant, or never judged | Article 58: Transfer from another university |

> **Write the guideline first.** Two judges who read the same guideline disagree less, and the test set outlives the people who made it.

#### [day1#pooling] Judge a pool, not the whole corpus

*Which pages get judged*

1. **1 · Run several systems** BM25, dense, anything you have
2. **2 · Take each one's top results** merge, remove duplicates
3. **3 · Judge the pool** unjudged pages count as not relevant

**396** judged pairs for 60 queries in test set v1

> **The bias.** A new system that finds pages no pooled system found is scored as wrong on them. Add its top results to the pool before you compare.

#### [day1#three-metrics] Recall, reciprocal rank and nDCG measure different failures

*Three numbers, three questions*

- **Recall at 10.** Of the relevant pages, how many are in the top 10? The model cannot read what is not there.
- **Reciprocal rank.** 1 over the rank of the first relevant page. Does the answer come first?
- **nDCG at 10.** Rewards grade 3 above grade 1 and high ranks above low ones, scaled 0 to 1.

```python
# file: Lab 1.1 · the gain of one ranking
dcg = sum((2**grade[p] - 1) / log2(rank + 1) for rank, p in enumerate(top10, 1))
ndcg = dcg / ideal_dcg   # the same sum over the best possible order
```

#### [day1#baseline-by-language] The baseline serves English better than Arabic, and dialect hardly at all

*Measured in Lab 1.1*

| Raw BM25, top 20 | recall@10 | RR | nDCG@10 |
| --- | --- | --- | --- |
| Arabic queries | 0.340 | 0.550 | 0.415 |
| English queries | 0.432 | 0.722 | 0.532 |
| Dialect queries | 0.050 | 0.100 | 0.065 |
| All 60 | 0.386 | 0.636 | 0.473 |

| Item | Value |
| --- | --- |
| course | 0.674 |
| regulation | 0.574 |
| table | 0.483 |
| service | 0.451 |
| dialect | 0.065 |

nDCG@10 by query type on the right. Test set v1: 60 queries, 30 English and 30 Arabic, 10 of them in Gulf dialect.

#### [day1#report-by-slice] Report every metric by language and by query type

*What the outcome asks for*

- **Version the test set**: v1 is frozen; new queries make v1.1, so old numbers stay comparable
- **One file per run**: the ranking, the test-set version, the date, the settings
- **Slices before averages**: an overall gain can be a loss on Arabic
- **Small test sets are noisy**: a difference of 0.01 on 60 queries is not a finding

> **For a Saudi university.** An assistant for Arabic speakers that is only measured in English has not been measured. Half of test set v1 is Arabic, and a third of that is Gulf dialect.

#### [day1#check-graded-judgements] Which change to the evaluation would you make first?

*Checkpoint*

**Question:** Nibras's search scores nDCG@10 0.47 overall. A colleague proposes a new embedding model that scores 0.49 on the same 60 queries, judged only on pages the old system found. What do you do first?

- A. Adopt it: 0.49 is higher than 0.47
- B. Add the new model's top results to the pool, judge them, then compare by language
- C. Judge the model's answers instead of its rankings
- D. Double the test set to 120 queries and rerun

#### [day1#lab-1-1] Graded judgements and a baseline score

Lab 1.1 · 25 min

Judge graded relevance for 40 bilingual queries in pairs, measure agreement with the reference judgements of test set v1, and score the supplied baseline by language and query type.

[D1_1_judge_graded_relevance.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D1_1_judge_graded_relevance.ipynb)

- [ ] The guideline and the pools (3 min)
- [ ] Judge your 20 queries, in pairs (9 min)
- [ ] Agreement with the reference judgements (4 min)
- [ ] Recall at 10, reciprocal rank, nDCG at 10 (6 min)
- [ ] By language and by query type (3 min)
- [ ] ✱ Stretch: Five queries of your own as test set v1.1

### Session 2: Lexical search and Arabic text

#### [day1#s2] Lexical search and Arabic text

*Session 2*

- Explain how BM25 weighs rare and repeated words
- Normalise Arabic text and apply light stemming on both sides
- Measure the gain on Arabic and dialect queries

#### [day1#inverted-index] An inverted index maps each word to the pages that contain it

*Lexical search*

| Term | Pages |
| --- | --- |
| `withdraw` | Article 25, Circular 2026-031, guide/withdrawal |
| `postpone` | Article 26, guide/postponement |
| `semester` | 924 of the 2,617 Public pages |

- A query looks up its words, not every page: fast at any size
- Exact words only: "withdrawal" and "withdraw" are different keys unless you stem
- Every database has one: SQLite FTS5, PostgreSQL full-text search, OpenSearch

#### [day1#bm25-intuition] BM25 rewards rare words and saturates repeated ones

*How BM25 weighs words*

- **Rarity.** "postpone" in 9 pages weighs far more than "semester" in 924.
- **Saturation.** The tenth "postpone" in a page adds little over the third.
- **Length.** A match in a short page counts more than in a long one.

```sql
-- file: Lab 1.2 · SQLite FTS5
CREATE VIRTUAL TABLE idx USING fts5(id UNINDEXED, text);
SELECT id, bm25(idx) FROM idx
 WHERE idx MATCH '"withdraw" OR "course" OR "deadline"'
 ORDER BY bm25(idx) LIMIT 10;   -- lower is better in FTS5
```

#### [day1#arabic-spelling-variants] One Arabic word has several spellings, and BM25 sees each as a stranger

*What Arabic does to word matching*

| Variation | Written | Also written |
| --- | --- | --- |
| Hamza on alif | الأولى | الاولى |
| Final ya and alif maqsura | المستوى | المستوي |
| Ta marbuta and ha | الجامعة | الجامعه |
| Diacritics and tatweel | الدِّراسة | الدراسة |
| Clitics | والطلاب | الطلاب |

> **What goes wrong.** Students type fast on phones and skip the hamza; the regulation is typeset carefully. Raw matching punishes exactly the users you serve.

#### [day1#normalise-both-sides] Normalise and lightly stem both the pages and the query

*The fix*

1. **1 · Normalise** remove diacritics and tatweel; unify alif, ya, ta marbuta
2. **2 · Light stem** strip ال, و, ب and common suffixes
3. **3 · Index** the analysed words, not the raw text

- **The classic bug.** Normalising the pages but not the query, or the reverse: every match fails silently.
- **Light, not deep.** A root-based stemmer merges words with different meanings. Light stemming removes affixes only.

#### [day1#normalisation-gain] Normalisation lifts Arabic recall from 0.34 to 0.44 and dialect from 0.05 to 0.25

*Measured in Lab 1.2*

| Item | Value |
| --- | --- |
| Arabic, raw | 0.340 |
| Arabic, normalised | 0.435 |
| English, either | 0.432 |
| Dialect, raw | 0.05 |
| Dialect, normalised | 0.25 |

Recall at 10 on test set v1. nDCG@10 overall: 0.473 raw, 0.538 normalised. Query p95 latency 6.7 ms on a laptop.

#### [day1#dialect-failures] Dialect fails on vocabulary, which no spelling rule repairs

*Read the failures, not only the score*

**Student:** ابي انسحب من ماده متى اخر يوم

**Top result (weak example):** Circular 2026-012, on another subject

**Student:** ابي اوقف دراستي ترم كم مره يسمحون

**Top result (weak example):** Article 36, lateness to an exam

> **Takeaway.** اوقف دراستي means "postpone my study", but the regulation says تأجيل الدراسة. Matching words cannot bridge that; matching meaning might.

#### [day1#check-normalisation] Why did Arabic recall not move?

*Checkpoint*

**Question:** Kawthar adds a normaliser to the indexing pipeline and rebuilds the index. Arabic recall at 10 stays at 0.34. English is unchanged too. What is the most likely cause?

- A. BM25 cannot rank Arabic text
- B. The Arabic queries are all in dialect
- C. Normalisation needs a deep root stemmer to work
- D. The queries are not passed through the same normaliser

#### [day1#lab-1-2] BM25 with and without Arabic normalisation

Lab 1.2 · 25 min

Run BM25 in SQLite full-text search on raw text and after your own Arabic normalisation and light stemming, and compare recall on Arabic and dialect queries.

[D1_2_run_bm25_without.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D1_2_run_bm25_without.ipynb)

- [ ] FTS5 on raw text: the baseline again (4 min)
- [ ] Write the normaliser (7 min)
- [ ] Light stemming and a second index (6 min)
- [ ] Compare by language and on dialect queries (5 min)
- [ ] Read three dialect failures (3 min)
- [ ] ✱ Stretch: Give the title its own column and weight

### Session 3: Dense retrieval and embedding choice

#### [day1#s3] Dense retrieval and embedding choice

*Session 3*

- Embed pages and queries with an open multilingual model
- Follow a model card's input conventions and show why they matter
- Compare exact and approximate indexes on quality, size and latency

#### [day1#embeddings-as-meaning] An embedding places texts with similar meaning close together

*Dense retrieval*

- A model turns each page into **384 numbers**; the query too
- Search is a **dot product** of the query with every page
- **Multilingual** models put an Arabic question near an English answer
- No word has to match: "drop a course" can find "withdrawal"

- **The two models of Lab 1.3.** **multilingual-e5-small** and **paraphrase-multilingual-MiniLM-L12-v2**: open weights, 384 dimensions, small enough for a laptop processor or a free Colab runtime.

#### [day1#model-card-prefixes] e5 expects "query:" and "passage:" in front of its inputs

*Read the model card*

```python
# file: Lab 1.3 · step 2
model.encode("query: " + question)
model.encode("passage: " + page_text)
```

| nDCG@10, dense e5 | with prefixes | without |
| --- | --- | --- |
| Arabic | 0.375 | 0.354 |
| English | 0.639 | 0.615 |
| All | 0.507 | 0.484 |

> **What goes wrong.** Nothing fails loudly. Skip one line of the model card and you lose a few points everywhere, without an error message.

#### [day1#two-models-compared] The two models disagree, and neither wins every row

*Measured in Lab 1.3*

| Dense search, test set v1 | nDCG@10 ar | nDCG@10 en | recall@10 | dialect recall |
| --- | --- | --- | --- | --- |
| multilingual-e5-small | 0.375 | 0.639 | 0.401 | 0.07 |
| multilingual MiniLM | 0.419 | 0.582 | 0.579 | 0.325 |
| BM25 normalised (Lab 1.2) | 0.543 | 0.532 | 0.433 | 0.25 |

- **Unflattering and true.** On Arabic in standard spelling, normalised BM25 beats both models.
- **Where dense helps.** e5 on English; MiniLM on dialect, the row BM25 could not fix.

#### [day1#embedding-cost] Embedding the corpus takes minutes; each query adds tens of milliseconds

*What embedding costs*

- **7.3 min** to embed all 2,977 pages with e5, laptop processor
- **82 ms** e5 query embedding, 95th percentile
- **252 ms** MiniLM query embedding, 95th percentile

Measured once when the supplied vectors were made: 300 pages in 44.1 s (e5) and 27.4 s (MiniLM), on a shared laptop. Your Colab numbers will differ; the ratio between indexing and querying will not.

#### [day1#exact-or-approximate] Exact search compares with every page; HNSW walks a graph of neighbours

*Indexes for vectors*

1. **Exact (flat)** every page, every query: perfect recall, linear time
2. **HNSW** a layered graph, coarse to fine: a few hundred comparisons

| 2,617 Public pages | e5 | MiniLM |
| --- | --- | --- |
| Overlap with exact top 10 | 0.922 | 0.995 |
| HNSW index size | 4.73 MB | 4.73 MB |
| Flat index size | 4.02 MB | 4.02 MB |
| HNSW p95 | 0.53 ms | 0.52 ms |
| Exact p95 | 1.15 ms | 1.29 ms |

#### [day1#quantised-vectors] Eight-bit vectors take a quarter of the space and keep 99% of the top 10

*Stretch result*

- **4.02 → 1.01 MB** e5 index, 32-bit floats to 8-bit codes
- **0.987** overlap with the exact top 10

> **Takeaway.** Memory is usually the first limit of a vector index on modest hardware; scalar quantisation is the first lever, and its cost is measurable.

#### [day1#check-embedding-choice] Which conclusion does the evidence support?

*Checkpoint*

**Question:** On test set v1, normalised BM25 scores nDCG@10 0.543 on Arabic and dense e5 scores 0.375. On English, e5 scores 0.639 and BM25 0.532. What should Kawthar conclude?

- A. Each method wins on one language, so a combination is worth measuring
- B. Dense retrieval does not work for Arabic and should be dropped
- C. A larger embedding model will beat BM25 on both languages
- D. The two are equal, since their averages are close

#### [day1#lab-1-3] Two embedding models: quality, size and latency

Lab 1.3 · 25 min

Embed a 300-page sample with two open multilingual models, load the supplied full indexes and compare quality, index size and latency, exact and with HNSW.

[D1_3_embed_300_page.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D1_3_embed_300_page.ipynb)

- [ ] Embed the 300-page sample (6 min)
- [ ] What e5's prefixes are for (4 min)
- [ ] Load the full indexes and search exactly (6 min)
- [ ] An HNSW index: recall against exact, size, latency (6 min)
- [ ] One row per model in the results table (3 min)
- [ ] ✱ Stretch: Store the vectors in eight bits

### Session 4: Hybrid search and reranking

#### [day1#s4] Hybrid search and reranking

*Session 4*

- Fuse two rankings with reciprocal rank fusion
- Rerank candidates with a cross-encoder
- Decide on a reranker from its gain and its 95th-percentile latency

#### [day1#errors-differ] Lexical and dense search fail on different queries

*Hybrid search*

- **BM25 finds.** Course codes, circular numbers, article numbers, exact names: "CSC 301", "2025-118".
- **Dense finds.** Paraphrases, everyday words, the other language: "drop a course", a question in English about an Arabic page.

> **The idea.** Run both, then merge the two rankings into one. The question is how to merge two lists whose scores are on different scales.

#### [day1#score-scales] BM25 scores run from 0.3 to 28; cosine scores from 0.78 to 0.93

*Why you cannot add the scores*

| Item | Value |
| --- | --- |
| BM25 range | 0.3 to 28.4 |
| Cosine range | 0.78 to 0.93 |

| Weighted score fusion | nDCG@10 |
| --- | --- |
| alpha 0.3 | 0.570 |
| alpha 0.5 | 0.566 |
| alpha 0.7 | 0.556 |

> **What goes wrong.** Add raw scores and BM25 decides everything. Normalise them and you add a weight to tune, and it moves with every corpus change.

#### [day1#reciprocal-rank-fusion] Reciprocal rank fusion adds 1 / (60 + rank) from each list

*Use ranks, not scores*

```python
# file: Lab 1.4 · step 1
def rrf(rankings, k=60):
    score = defaultdict(float)
    for ranking in rankings:
        for rank, page in enumerate(ranking, 1):
            score[page] += 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)
```

| Test set v1 | recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| BM25 normalised | 0.433 | 0.729 | 0.538 |
| dense e5 | 0.401 | 0.691 | 0.507 |
| RRF of the two | 0.455 | 0.776 | 0.570 |

#### [day1#cross-encoder-reranker] A cross-encoder reads the query and each candidate together

*The second stage*

1. **1 · First stage** BM25 and dense, fused: top 30 in milliseconds
2. **2 · Rerank** a model scores each (query, page) pair: 30 model calls
3. **3 · Top 5** what the generator reads

- **Why it is better.** It sees both texts at once, so it can judge whether the page answers this question.
- **Why it is slow.** One forward pass per candidate, every query. Nothing can be computed ahead.

#### [day1#day1-results-table] Reranking gains the most and costs a hundred times the latency

*The Day 1 results table, measured*

| Method | nDCG@10 ar | nDCG@10 en | recall@10 | MRR | p95 ms |
| --- | --- | --- | --- | --- | --- |
| BM25 normalised | 0.543 | 0.532 | 0.433 | 0.729 | 6.9 |
| dense e5 | 0.375 | 0.639 | 0.401 | 0.691 | 84.1 |
| hybrid RRF | 0.510 | 0.630 | 0.455 | 0.776 | 91.1 |
| hybrid RRF + rerank 30 | 0.566 | 0.700 | 0.499 | 0.872 | 10,971 |

Latency includes the query embedding (measured once) and, for the last row, reranking 30 pairs on a shared laptop processor without a GPU.

#### [day1#rerank-depth] Rerank fewer candidates: most of the gain comes from the top 10

*Lab 1.4 stretch*

| Item | Value |
| --- | --- |
| depth 5 · 1.9 s | 0.608 |
| depth 10 · 3.7 s | 0.617 |
| depth 20 · 7.3 s | 0.633 |
| depth 30 · 11.0 s | 0.633 |

nDCG@10 overall by rerank depth, with the p95 latency of reranking on the laptop. Without a reranker: 0.570.

> **Takeaway.** Set the latency budget first, then pick the deepest rerank that fits it.

#### [day1#check-reranker-decision] Should Nibras ship the reranker?

*Checkpoint*

**Question:** Retrieval for Nibras must answer within 400 ms at the 95th percentile. Reranking the top 30 raises nDCG@10 from 0.570 to 0.633 and takes 11 s at p95 on the current servers, which have no GPU. What do you recommend?

- A. Ship it: quality matters more than speed
- B. Ship it with a rerank depth of 5
- C. Ship hybrid RRF now and measure the reranker on a GPU before deciding
- D. Ship BM25 alone, since reranking is too slow

#### [day1#lab-1-4] Fusion, a reranker and its latency

Lab 1.4 · 25 min

Fuse lexical and dense rankings, add a reranker on the fused top 30 and report the gain by language against the added latency in the Day 1 results table.

[D1_4_fuse_lexical_dense.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D1_4_fuse_lexical_dense.ipynb)

- [ ] Reciprocal rank fusion (7 min)
- [ ] Score fusion and the problem of scales (5 min)
- [ ] Rerank the fused top 30 (6 min)
- [ ] The gain against the added latency (7 min)
- [ ] ✱ Stretch: The rerank depth that fits a latency budget

#### [day1#wrap-up] Three things to keep

*Day 1 · wrap-up*

- **Measure retrieval alone.** A versioned test set with graded judgements, reported by language and query type.
- **Arabic needs analysis.** Normalise both sides. It lifted Arabic recall from 0.34 to 0.44; dialect needs meaning, not spelling.
- **Gain and latency in one row.** Fusion is nearly free; reranking gains most and may not fit the budget.

#### [day1#next] Day 2: Prepare the corpus

*Next*

Tomorrow the problems move upstream: scanned pages that parse to nothing, tables cut in half, rules amended by circulars, and pages a student must never see. Unfinished labs can be finished at home; each one starts from a supplied state.
