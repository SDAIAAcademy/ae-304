---
course: "Retrieval and Knowledge Engineering for Language Model Systems"
code: "AE-304"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-304/"
---

# Retrieval and Knowledge Engineering for Language Model Systems

Build and measure the retrieval layer of a language model system: search, corpus preparation, access, agents and structured data

Participants build and measure the retrieval layer of a language model system for a fictional university: bilingual regulations, study plans with tables, scanned Arabic circulars and an enrolment statistics warehouse. Starting from a graded test set, they build lexical, dense, hybrid and reranked search with Arabic normalisation, parse and chunk documents by structure, filter by date and role, give an agent search tools and a small knowledge graph, and answer statistical questions through a semantic layer instead of free text-to-SQL.

- **Audience:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.
- **Prerequisites:** AE-202 Deep Learning and Generative AI Fundamentals; AE-203 Building Applications on Language Models; Writes SQL joins and aggregations; Computes precision and recall; Has built a small grounded application with citations
- **Case study:** Al-Nuha University: Al-Nuha University is a fictional public university with six colleges whose regulations assistant, Nibras, answers students and staff from bilingual regulations, study plans with tables, scanned Arabic circulars, staff-only records and an enrolment statistics warehouse. Participants build and measure Nibras's retrieval layer on about 3,000 synthetic pages and a synthetic warehouse, generated with a fixed seed, with planted problems: dialect queries, broken parses, superseded rules, near-duplicate circulars, mislabelled staff pages and warehouse traps.

## Outcomes

- LO1 (Applied): Prepare a retrieval test set with graded relevance judgements and compute standard ranking metrics for each language, given 60 Arabic and English queries over the corpus
- LO2 (Applied): Implement lexical, dense, hybrid and reranked search with Arabic normalisation and compare them on quality and latency, given the corpus and the test set
- LO3 (Applied): Analyse parsing and chunking choices by scoring two parsers and two chunking schemes, given 30 scanned and digital pages with tables
- LO4 (Performance): Build retrieval that never returns a document the requester may not see or a superseded rule, given role labels and effective dates on every chunk
- LO5 (Performance): Design an agentic search component that chooses among search tools and a small knowledge graph within a context budget, given 15 multi-step questions, a graph of linked course rules and a token budget
- LO6 (Performance): Evaluate free text-to-SQL against a semantic layer with validated SQL and decide which to ship, given 20 statistical questions with reference answers

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

## Day 2: Prepare the corpus

#### [day2#title] Prepare the corpus

*SDAIA Academy · Retrieval and Knowledge Engineering for Language Model Systems*

Day 2

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-304

#### [day2#agenda] Prepare the corpus

*Today*

- **Session 1 · Parsing is a separate problem**: Score parsers apart from retrieval: word error rate and table cells by page type, and a rule that routes each page to the parser that reads it
- **Session 2 · Chunking by structure**: Chunk by the document's own structure, carry the heading path into each chunk, and test chunking by the questions it can answer
- **Session 3 · Dates, versions and duplicates**: Give every version a validity interval, drop near-duplicates, and test that the current rule comes first
- **Session 4 · Permissions in the index**: Carry role and classification on every chunk, filter before ranking, and test leaks from a student account as a suite

#### [day2#review-day1] What does yesterday's results table say?

*Review of Day 1*

**Question:** Dense e5 scored nDCG@10 0.375 on Arabic and 0.639 on English; normalised BM25 scored 0.543 and 0.532; their RRF fusion 0.510 and 0.630. Which reading is right?

- A. Fusion beats both inputs on both languages
- B. Dense retrieval cannot serve Arabic
- C. The three methods are equal, since their averages are close
- D. Fusion helps English a lot and costs a little on Arabic, so Arabic needs its own look

### Session 1: Parsing is a separate problem

#### [day2#s1] Parsing is a separate problem

*Session 1*

- Score parsed text against reference text with a word error rate
- Score tables cell by cell
- Route pages to a parser by page type

#### [day2#parsing-before-retrieval] Retrieval can only find what the parser wrote down

*Corpus preparation*

1. **PDF or scan** what the university publishes
2. **Parser** text and tables, or nothing
3. **Chunks** what gets indexed
4. **Search** Day 1's methods

> **What goes wrong.** A parsing failure looks like a retrieval failure: the page "is never found". Score the parser on its own, against reference text, before you tune search.

#### [day2#two-kinds-of-parser] A text-layer parser reads what the PDF stores; a layout-aware one reads the page

*Two families of parser*

- **Parser A · text layer.** Fast and cheap. Reads the characters the PDF holds, in the order it holds them. A scan holds no characters; Arabic may come out in presentation forms and reversed order.
- **Parser B · layout-aware.** Slower. Recognises text from the image and finds table structure. Makes its own errors: similar letters confused.

Examples of the families: pypdf and pdfminer for text layers; Docling, Unstructured, Tesseract and open-weight document vision models for layout and recognition.

#### [day2#specialised-or-general-reader] A small parsing model or a general vision model: score both on your pages

*Inside the layout-aware family*

- **Small specialised parsing model.** Trained only to read documents: layout, reading order, table structure, characters. Runs on modest hardware and costs little per page. Weak on layouts it never saw.
- **General vision model.** Reads the page image and follows an instruction, such as "return this table as Markdown". Flexible on odd layouts. Costs more per page and can write fluent text that is not on the page.

> **What goes wrong.** A misread letter looks like a mistake. An invented word looks like text. Only a score against reference pages, by page type, shows which one you have.

#### [day2#one-page-three-ways] The same study-plan page, as the text layer gives it

*Lab 2.1, step 1*

##### Reference

الخطة الدراسية: بكالوريوس الهندسة الصناعية، السنة الثالثة

`| IND 312 | ... | 3 | IND 212 |`

##### Parser A

ﺍﻝﺙﺍﻝﺙة ﺍﻝﺱﻥة ﺍﻝﺹﻥﺍﻉﻱة، ﺍﻝﻩﻥﺩﺱة ﺏﻙﺍﻝﻭﺭﻱﻭﺱ ﺍﻝﺩﺭﺍﺱﻱة: ﺍﻝﺥﻁة

`212 IND 3 ... 312 IND`

- **Presentation forms**: each letter stored in its joined shape, so no word matches
- **Reversed word order** in Arabic lines, and the table's pipes lost

#### [day2#word-error-rate] Word error rate: the edits that turn the output into the reference, per word

*Score the text*

```python
# file: Lab 2.1 · step 2
def words(text):
    text = unicodedata.normalize("NFKC", text)
    return re.findall(r"\w+", U.normalise(text))

def wer(ref, out):
    r, o = words(ref), words(out)
    return edit_distance(r, o) / len(r)
```

- Normalise first, or you score typography, not content
- Two swapped words cost two edits: a toy of three words scores 0.667
- An empty output scores 1.0, every word missing

#### [day2#table-cell-accuracy] A table is right only when each cell is in its row and column

*Score the tables*

- Compare the parsed table with the reference **cell by cell**: same row, same column, same text
- Text can be perfect and the table worthless: "3" and "IND 212" on one line do not say which course they belong to
- Study plans are mostly tables: a question like "how many credit hours is CSC 301" lives in one cell

> **Takeaway.** Report text and table scores separately. A parser can win one and lose the other.

#### [day2#scores-by-page-type] The text layer is fine on plain digital pages and useless on scans

*Measured in Lab 2.1*

| Page type (pages) | WER A | WER B | cells A | cells B |
| --- | --- | --- | --- | --- |
| Digital, no table (7) | 0.397 | 0.012 | - | - |
| Digital, table (8) | 0.278 | 0.001 | 0.000 | 0.998 |
| Scanned, no table (11) | 1.000 | 0.054 | - | - |
| Scanned, table (4) | 1.000 | 0.073 | 0.000 | 0.944 |

30 pages. Parser A's errors on digital pages are mostly the Arabic ones: reversed word order. Its English digital pages are nearly clean.

#### [day2#routing-rule] Route each page to the cheapest parser that reads it

*The decision the scores support*

1. **No text layer, or presentation forms?** parser B
2. **Looks like a table?** parser B
3. **Otherwise** parser A

| 30 pages | WER | cells |
| --- | --- | --- |
| Parser A only | 0.667 | 0.000 |
| Parser B only | 0.033 | 0.980 |
| Routed (4 pages to A) | 0.032 | 0.980 |

#### [day2#check-parser-choice] Where should Kawthar look first?

*Checkpoint*

**Question:** Students report that Nibras never finds the 2019 study plans, which the registrar scanned from paper. The plans are in the index. What should Kawthar check first?

- A. The weight of the title column in BM25
- B. The parsed text of those pages: a scan read by a text-layer parser is empty
- C. A better embedding model for Arabic
- D. The chunk size of the study plans

#### [day2#lab-2-1] Two parsers scored on 30 pages

Lab 2.1 · 25 min

Score a text-layer parser and a layout-aware parser on 30 scanned and digital pages with tables against reference text, by page type, and write the routing rule that picks one per page.

[D2_1_score_two_parsers.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D2_1_score_two_parsers.ipynb)

- [ ] One page, three ways (3 min)
- [ ] Word error rate after normalisation (7 min)
- [ ] Table-cell accuracy (6 min)
- [ ] Scores by page type (5 min)
- [ ] A routing rule, and its score (4 min)
- [ ] ✱ Stretch: Repair the text layer and score it again

### Session 2: Chunking by structure

#### [day2#s2] Chunking by structure

*Session 2*

- Chunk documents by headings and keep the heading path
- Compare chunkings by retrieval on table questions
- Recommend a chunking from the evidence

#### [day2#why-chunk] A chunk is the unit the model reads: big enough to answer, small enough to rank

*Chunking*

- **Too small.** A row without its header: "3 | IND 212" answers nothing.
- **Right.** One semester's table with the plan and year named.
- **Too large.** A whole plan: the right row is one of forty, and the score is diluted.

#### [day2#fixed-windows-cut-tables] Fixed windows cut tables wherever the word count falls

*The common default*

- window 1: ... | Total | 14 | | ### Year 1 · Semester 2 | Code | Course | Credit hours | Prerequisite | |---|---|---|---| | ACC 101 | Introduction to Financial ...
- window 2: ... Accounting | 3 | UNI 105 | | ACC 112 | ...

> **What goes wrong.** The window that holds "ACC 101" may not hold "Semester 2", and neither names the programme. The question "in which semester of the Accounting plan is ACC 101 taught?" has no single chunk that answers it.

#### [day2#chunk-by-heading] Cut at the document's headings and carry the heading path into each chunk

*Chunk by structure*

```markdown
<!-- file: Lab 2.2 · one heading chunk -->
Study plan: Bachelor of Accounting > Year 1 > Year 1 · Semester 1
| Code | Course | Credit hours | Prerequisite |
|---|---|---|---|
| ACC 112 | Foundations of Auditing | 3 | None |
| UNI 101 | Arabic Writing | 2 | None |
```

- One chunk per semester: **8 per plan**, 240 in all
- The path names the plan, the year and the semester, so the chunk answers on its own

#### [day2#answerable-at-3] Answerable at 3: does a top-3 chunk hold everything the answer needs?

*Score chunking by the questions it answers*

- Each of the 20 table questions lists the strings the answer needs: "Semester 6" and "MTH 301"
- A question is answerable when one of the top 3 chunks contains all of them
- No model needed, so the score is cheap and repeatable

- **Why not recall?** Chunkings make different units, so there are no shared relevance judgements. Answerability compares them on the same questions.

#### [day2#chunking-results] Heading chunks answer 0.65 of the table questions; 120-word windows 0.25

*Measured in Lab 2.2*

| Item | Value |
| --- | --- |
| windows of 60 words | 0.05 |
| windows of 120 words | 0.25 |
| windows of 240 words | 0.50 |
| by heading, with path | 0.65 |

Answerable at 3 on 20 table questions over 30 study plans. Window sizes from the stretch.

#### [day2#chunking-recommendation] A recommendation names the evidence and when to revisit it

*The recommendation, written down*

> **Recommendation.** Chunk study plans by semester heading with the plan title and year carried into each chunk, and never cut inside a table. Revisit if plans gain long prose sections, where windows of a fixed size may retrieve better.

- **Evidence**: answerable at 3, 0.65 against 0.25, on 20 questions
- **Scope**: study plans; regulations and news may chunk differently
- **Revisit**: a condition you can check, not "later"

#### [day2#check-chunking] Which change will answer more table questions?

*Checkpoint*

**Question:** Nibras uses 120-word windows on the study plans and answers 0.25 of the table questions in the top 3. Which change does the evidence support most?

- A. Shrink the windows to 60 words for precision
- B. Add a reranker on top of the windows
- C. Chunk by semester heading and carry the plan and year into each chunk
- D. Add a 20-word overlap between windows

#### [day2#lab-2-2] Two chunkings of the study plans

Lab 2.2 · 25 min

Chunk the study plans in fixed windows and by headings with their heading path, and compare how many of 20 table questions each chunking answers in the top 3.

[D2_2_chunk_study_plans.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D2_2_chunk_study_plans.ipynb)

- [ ] One plan, as the parser gave it (3 min)
- [ ] Fixed windows of 120 words (5 min)
- [ ] Chunks by heading, with the heading path (8 min)
- [ ] Retrieve and score on 20 table questions (6 min)
- [ ] The recommendation (3 min)
- [ ] ✱ Stretch: Window size against answerability

### Session 3: Dates, versions and duplicates

#### [day2#s3] Dates, versions and duplicates

*Session 3*

- Read effective dates and amended articles from circulars
- Keep a validity interval for every version
- Remove near-duplicates and write staleness tests

#### [day2#superseded-rules] A rule amended by a circular is still in the corpus, and it still ranks

*Dates and versions*

- The regulation's articles were written once
- Circulars amend them, each with an **issue date** and an **effective date**
- Both versions are relevant to the words of a question; only one is the answer today

**3 of 15** changed questions whose top result states the current rule, before this session's fix

#### [day2#issued-and-effective] The issue date says when a rule was written; the effective date says when it counts

*Read both dates*

- question: What is the maximum study load in a regular semester? (asked on 1 November 2026)
- Article: between 12 and 20 credit hours
- Circular 2026-140: issued 2026-10-05, effective 2027-01-01: between 12 and 18
- answer: 12 to 20 today; 18 from 1 January 2027

> **What goes wrong.** "Newest first" picks the circular, which is not yet in force.

#### [day2#validity-intervals] Give every version a validity interval and filter by today's date

*The fix*

| Article 17 version | valid from | valid to |
| --- | --- | --- |
| Regulation text | start | 2025 amendment |
| Circular 2025-118 | its effective date | open |

- Parse the circular for the article it amends and its effective date
- Close the previous version's interval when the next takes effect
- At query time keep chunks valid on the question's date

In the lab: 93 chunks, 33 of them amendments; 63 are valid on 1 November 2026.

#### [day2#near-duplicates] Near-duplicate copies crowd the top results with the same text

*Duplicates*

- Circulars are forwarded, re-posted by colleges, and copied into news
- Compare chunks as sets of **word shingles** (runs of 3 words) and their Jaccard overlap
- Copies measured 0.62 to 0.71 overlap; a threshold of 0.5 catches them

**8** near-duplicate copies dropped in Lab 2.3

> **Keep the original.** Drop the copy, keep the official circular's metadata: its dates are the ones that count.

#### [day2#staleness-tests] A staleness test asks a question and fails if the old answer comes first

*Make it a test*

```python
# file: Lab 2.3 · step 5
for t in CHANGED:                  # 15 questions, Arabic and English
    top = search(valid_on(TODAY), [t["q"], t["q_ar"]])[0]
    assert t["expect"] in top.text, t["qid"]
    assert t["stale"] not in top.text, t["qid"]
```

**3 → 15** of 15 tests passing, before and after

- **Run it on every rebuild.** A new circular, a re-parse or a new model can break it again.

#### [day2#check-effective-date] What should Nibras answer?

*Checkpoint*

**Question:** On 1 November 2026 a student asks for the maximum study load. The article says 20 hours. Circular 2026-140, issued in October, sets 18 hours from 1 January 2027. What is the right answer?

- A. 18 hours, since the circular is newer
- B. Both numbers, and let the student decide
- C. Neither, since the rule is changing
- D. 20 hours, and that 18 hours applies from 1 January 2027

#### [day2#lab-2-3] Effective dates, versions and duplicates

Lab 2.3 · 25 min

Read effective dates from the circulars, give every version a validity interval, drop near-duplicate copies, and pass 15 staleness tests whose answer changed in a newer circular.

[D2_3_add_effective_dates.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D2_3_add_effective_dates.ipynb)

- [ ] Fifteen questions, answered from stale text (3 min)
- [ ] Read effective dates and amended articles (7 min)
- [ ] A validity interval for every version (5 min)
- [ ] Near-duplicates by shingles (5 min)
- [ ] Staleness tests (5 min)
- [ ] ✱ Stretch: A question about the past

### Session 4: Permissions in the index

#### [day2#s4] Permissions in the index

*Session 4*

- Tag every chunk with its document's classification and roles
- Filter before ranking rather than after
- Run leak and utility tests from a student account

#### [day2#search-can-leak] A search index without labels shows students what only staff may see

*Permissions*

**103** Restricted results for 12 student queries, with no filter

- 360 of 2,977 pages are **Restricted**: council minutes, promotion rules, allowance scales, disciplinary case summaries, examination archives
- A test account, Muath Al-Shathri, searches as a student
- Every Restricted page he sees is a leak, whatever the model then says

#### [day2#label-from-document] Every chunk inherits its document's classification and allowed roles

*Tag at indexing*

1. **Document** classification and roles, set by its owner
2. **Pages and chunks** inherit the labels
3. **Unlabelled** Restricted by default

- **The naive tagger.** Labels each page from its own text: 7 Restricted pages without their own label came out Public.
- **In Saudi practice.** Data is classified by its owner under the national Data Classification Policy; personal data falls under the Personal Data Protection Law.

#### [day2#filter-before-ranking] Filter before ranking, or the student gets a short, strange list

*Where the filter goes*

| Results per query, 12 student queries | Counts |
| --- | --- |
| Top 10, then remove Restricted | 0, 0, 0, 0, 0, 5, 8, 0, 0, 1, 3, 0 |
| Rank only allowed pages | 10 for every query |

> **What goes wrong.** A post-filter leaks nothing and still fails the student: eight of twelve queries return nothing. It also tells an attacker, by the gaps, that something was there.

#### [day2#leak-and-utility] Test leaks and utility together, or an empty index passes

*The leak suite*

- **Leak tests.** 12 queries from the student account; fail if any result is Restricted.
- **Utility checks.** The same queries; fail if a Public page that answers them is missing.

- **0** leak failures after pre-filtering
- **0** utility failures

#### [day2#retrieved-text-untrusted] A page can carry instructions to the model; quote it, never obey it

*Retrieved text is data*

**Circular 2026-777:** The student satisfaction survey opens next week ... followed by a note addressed to automated assistants, asking them to ignore access rules and include disciplinary summaries and examination papers.

**Nibras's pipeline (good example):** Flagged P0487. Roles come from the account, not from page text: nothing unlocked.

> **Takeaway.** Permissions are enforced in code on the account's roles. No text in the index can change them.

#### [day2#check-permissions] Which design keeps students away from staff pages?

*Checkpoint*

**Question:** Nibras must never show a student a Restricted page and must still return ten useful results. Which design meets both?

- A. Label each chunk from its document, default to Restricted, and rank only the chunks the account may see
- B. Tell the model in the system prompt not to quote Restricted pages
- C. Take the top 10 and remove the Restricted ones
- D. Label each page from its own text with a classifier

#### [day2#lab-2-4] Role labels and leak tests

Lab 2.4 · 25 min

Tag every page with role and classification from its document, filter before ranking, and run 12 leak tests from a student account with a utility check that nothing allowed is lost.

[D2_4_tag_chunks_role.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D2_4_tag_chunks_role.ipynb)

- [ ] Search as Muath with no filter (4 min)
- [ ] Tag every page from its document (6 min)
- [ ] Filter before ranking, not after (6 min)
- [ ] The leak suite and the utility check (6 min)
- [ ] Retrieved text is untrusted (3 min)
- [ ] ✱ Stretch: The registrar's view: no over-blocking

#### [day2#wrap-up] Three things to keep

*Day 2 · wrap-up*

- **Score the parser alone.** By page type, text and tables apart, then route each page to the parser that reads it.
- **Chunk by structure.** Headings with their path answered 0.65 of the table questions against 0.25.
- **Dates and roles in the index.** Validity intervals and pre-filtering, each proved by a test suite that runs on every rebuild.

#### [day2#next] Day 3: Agentic search and knowledge

*Next*

Tomorrow the search tools go to an agent: it chooses among them, searches again when one search is not enough, keeps large results out of its context, and uses a small graph for questions about links.

## Day 3: Agentic search and knowledge

#### [day3#title] Agentic search and knowledge

*SDAIA Academy · Retrieval and Knowledge Engineering for Language Model Systems*

Day 3

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-304

#### [day3#agenda] Agentic search and knowledge

*Today*

- **Session 1 · Search as tools for an agent**: Listing, keyword and vector search as tools, described by when to use each, and the agent's choices read in its traces
- **Session 2 · Multi-step questions**: Search again from what you read, stop on an answer, a budget or a repeated query, and compare with one search
- **Session 3 · Paging and offloading**: Keep large results out of the context: a file, a stub and a tool to search it
- **Session 4 · Knowledge graphs for linked facts**: A small graph for facts that are links, queried with tools, measured against search

#### [day3#review-day2] Which suite catches the problem?

*Review of Day 2*

**Question:** After a re-index, a student asks about the attendance rule and Nibras quotes the regulation's original text, which Circular 2025-118 amended. Which test should have caught it before release?

- A. The leak suite from the student account
- B. The staleness suite, which fails when the old answer comes first
- C. nDCG@10 on test set v1
- D. Word error rate on the 30 parsing pages

### Session 1: Search as tools for an agent

#### [day3#s1] Search as tools for an agent

*Session 1*

- Write search tools whose descriptions say when to use them
- Trace an agent's tool choices
- Judge a choice by its cost and its answer

#### [day3#search-as-tools] Give the model search as tools, and it decides what to look for

*Agentic search*

- user: How many circulars did the university issue in 2025?
- tool call: `list_files("circulars/2025")`
- result: `["circulars/2025/2025-031-ar.md", ...]`
- final: The count and the circular numbers

- Yesterday the pipeline searched once with the question; today a **loop** calls tools
- The model reads each result and chooses the next call or answers
- Coding agents work this way over files: list, grep, read

#### [day3#three-tools] Three tools, each cheapest for one kind of question

*Lab 3.1*

| Tool | Returns | Best for |
| --- | --- | --- |
| `list_files(folder)` | file names only | what exists, how many |
| `keyword_search(query, k)` | top pages, BM25 normalised | codes, circular numbers, names |
| `vector_search(query, k)` | top pages, dense e5 | paraphrases, dialect, the other language |

> **All three search Public pages only.** The Day 2 filter sits inside the tools, so no tool call can reach a Restricted page.

#### [day3#tool-descriptions] A tool description tells the model when to call it, not only what it does

*The description is the interface*

```python
# file: nibras_tools.py
def keyword_search(query: str, k: int = 5) -> list[dict]:
    """Keyword search (BM25, Arabic normalised). Best for exact
    words: course codes such as "CSC 301", circular numbers such
    as "2025-118", article numbers and names. English or Arabic."""
```

- The function's docstring becomes the tool's description in the schema the model sees
- Examples of when to use it beat a list of features
- The stretch writes a misleading description and watches the choices change

#### [day3#tool-loop] Call the model, run the tool it asks for, append the result, repeat

*The loop, in twelve lines*

```python
# file: nuha.py · run_agent, simplified
messages = [system, {"role": "user", "content": question}]
for step in range(max_steps):
    reply = chat(messages, tools=schemas)
    if not reply.tool_calls:
        return reply.content                      # the answer
    for call in reply.tool_calls:
        result = TOOLS[call.name](**call.args)
        result = json.dumps(result)[:budget_chars] # the context budget
        messages.append(tool_message(call.id, result))
return "step budget spent"
```

#### [day3#read-the-traces] The right tool was chosen every time; the vector questions still failed most

*Measured in Lab 3.1, stand-in*

| Best tool | Questions | Right tool | Correct | Tokens |
| --- | --- | --- | --- | --- |
| list_files | 5 | 1.0 | 1.0 | 1,454 |
| keyword_search | 5 | 1.0 | 0.8 | 1,226 |
| vector_search | 5 | 1.0 | 0.4 | 1,212 |

> **Read the trace before blaming the agent.** The choice was right; e5 returned the wrong pages for short paraphrases. That is a Day 1 problem showing up in Day 3.

#### [day3#choice-has-a-cost] A wrong tool can still answer, at many times the tokens

*Judge a choice by its cost and its answer*

- **Listing for a count.** File names, one call: the count is exact.
- **Search for a count.** A top k of pages: the count is capped at k and every page's text is paid for.

> **Takeaway.** Log three numbers per question: the tool chosen, the calls and the tokens. Accuracy alone hides an agent that is right and expensive.

#### [day3#check-tool-choice] Which tool should Nibras call first?

*Checkpoint*

**Question:** A student writes in dialect: "ابي اعرف وش اسوي اذا مرضت يوم الاختبار" (what do I do if I am sick on the exam day). Which first call is best?

- A. list_files("regulations")
- B. keyword_search with the question as written
- C. vector_search with the question as written
- D. read_file on every article of the examination chapter

#### [day3#lab-3-1] Three search tools and a traced agent

Lab 3.1 · 25 min

Expose file listing, keyword search and vector search as tools, run 15 questions through a tool-calling loop and read the traces: which tool, how many calls, how many tokens, and whether the answer was right.

[D3_1_expose_file_listing.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D3_1_expose_file_listing.ipynb)

- [ ] The file tree, and listing it (5 min)
- [ ] Keyword and vector search as tools (7 min)
- [ ] Fifteen questions through the loop (6 min)
- [ ] Read the traces: choice, calls, answer (7 min)
- [ ] ✱ Stretch: A misleading description

### Session 2: Multi-step questions

#### [day3#s2] Multi-step questions

*Session 2*

- Write an iterative search loop with stop rules
- Compare single-shot and iterative search on accuracy, searches and tokens
- Set a search budget from the evidence

#### [day3#two-hop-questions] Some answers sit two pages apart, linked by a fact on the first

*Multi-step questions*

1. **Question** What GPA do I need to transfer into the college that teaches CLS 334?
2. **Hop 1** the course page names the College of Health Sciences
3. **Hop 2** that college's handbook: at least 4.00 out of 5.00

> **What goes wrong.** One search for the whole question returns the course page and some other college's transfer rule, and the answer quotes the wrong GPA.

#### [day3#iterative-loop] The iterative loop: search, read, decide the next search or answer

*Search again from what you read*

```python
# file: Lab 3.2 · step 2
notes, asked = [], set()
while True:
    d = decide(question, notes)             # the model, or the stand-in
    if "answer" in d:
        return d["answer"]
    q = d["search"]
    if q in asked or len(notes) >= max_searches:
        return decide(question, notes, final=True)["answer"]
    asked.add(q)
    notes.append({"query": q, "top": hybrid_search(q, 3)})
```

#### [day3#iterative-trace] The decider finds the college, then searches for its rule

*One question, three searches*

- search 1: the question as written
- notes: no course page in the top 3
- search 2: `CLS 334 course description`
- notes: CLS 334 · offered by the College of Health Sciences
- search 3: `College of Health Sciences handbook admission by transfer`
- answer: At least 4.00 out of 5.00 (College of Health Sciences)

#### [day3#single-against-iterative] Iterative search answers 9 of 10 two-hop questions, at twice the tokens

*Measured in Lab 3.2, stand-in*

| Ten two-hop questions | Accuracy | Searches | Tokens read |
| --- | --- | --- | --- |
| Single-shot, top 5 | 0.7 | 1.0 | 1,114 |
| Iterative, up to 4 searches | 0.9 | 2.1 | 2,363 |

> **Honest reading.** Single-shot already answers 7 of 10, because a top 5 with long excerpts often holds both hops. Iteration earns its cost on the rest.

#### [day3#search-budget] A search budget is a stop rule you set from evidence

*Lab 3.2 stretch*

- **3 of 5** three-hop questions answered with 2 searches
- **3 of 5** with 4 searches

Doubling the budget bought nothing on these five: the misses fail for another reason. Look at their traces before raising the budget.

#### [day3#check-iterative] What should the decider search next?

*Checkpoint*

**Question:** Question: "How many credit hours is the first prerequisite of CHM 423?" The first search returned the CHM 423 course page, which lists "Prerequisite: CHM 301". What is the best next search?

- A. The question again, as written
- B. "credit hours prerequisite"
- C. No search: answer with the hours on the CHM 423 page
- D. "CHM 301 course description credit hours"

#### [day3#lab-3-2] Single-shot against iterative search

Lab 3.2 · 25 min

Answer ten two-hop questions with one search and with an iterative loop that searches again from what it read, and compare accuracy, searches and tokens.

[D3_2_compare_single_shot.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D3_2_compare_single_shot.ipynb)

- [ ] Single-shot: one search, then answer (5 min)
- [ ] The iterative loop and its stop rules (9 min)
- [ ] Ten questions both ways (6 min)
- [ ] Accuracy, searches and tokens (5 min)
- [ ] ✱ Stretch: Three hops

### Session 3: Paging and offloading

#### [day3#s3] Paging and offloading

*Session 3*

- Measure what a large result costs in tokens
- Offload results to a file behind a stub
- Compare accuracy and tokens with and without offloading

#### [day3#large-results] Some results do not fit: every course page of a college is 8,500 tokens

*Paging and offloading*

- **1,594** tokens: the Physics programme's course pages
- **8,479** tokens: the College of Engineering's
- **1,500** tokens: the budget per tool result

Lab 3.3's tool `list_courses` returns every course page of a programme or a college in full. Token counts are estimates (four characters per token).

#### [day3#truncation-fails] Cut a long result to the budget and a count comes out wrong, confidently

*What truncation does*

| Question | Expected | In context, cut |
| --- | --- | --- |
| College of Engineering courses of 4 credit hours | 10 | 1 |
| College of Computing laboratory courses of 1 credit hour | 5 | 1 |
| Statistics programme courses with no prerequisite | 0 | 0 |

> **What goes wrong.** Nothing signals the cut. The agent counts what it can see and reports it as the answer.

#### [day3#offload-to-file] Offload: write the result to a file and return a stub

*The fix*

```python
# file: Lab 3.3 · step 3
def offload(result, limit=1500):
    if count_tokens(result) <= limit:
        return result
    path = OUT / f"list_courses_{n}.txt"
    path.write_text(result)
    return json.dumps({"file": path.name,
        "lines": n_lines, "pages": n_pages,
        "preview": result[:200],
        "hint": "search it with grep_file"})
```

- **Nothing is lost.** The full result is one tool call away.
- **A tool for the file.** `grep_file(file, pattern)` returns the matching lines and their count.

#### [day3#offload-results] Offloading doubles accuracy and roughly halves the tokens

*Measured in Lab 3.3, stand-in*

| Item | Value |
| --- | --- |
| accuracy, in context | 0.5 |
| accuracy, offloaded | 1.0 |
| tokens, in context | 1,724 |
| tokens, offloaded | 924 |

Eight counting questions, mean per question. Offloaded runs take two calls (the list, then the grep) instead of one.

#### [day3#paging-instead] Paging works, and costs more when the question needs a search

*Lab 3.3 stretch*

- **Paging.** Read the 359 lines in pages of 40: 9 calls and about 7,956 tokens to see them all.
- **Searching the file.** One grep for the pattern: one call and the count.

> **Takeaway.** Page when the agent must read everything; search the file when the question names what to look for.

#### [day3#check-offload] How should the agent handle this result?

*Checkpoint*

**Question:** A tool returns 60 course pages (about 8,500 tokens) for "how many Engineering courses carry 4 credit hours?". The result budget is 1,500 tokens. What design answers correctly with the smallest context?

- A. Cut the result to 1,500 tokens and count
- B. Raise the budget to 10,000 tokens
- C. Offload the result to a file, return a stub, and search the file for "Credit hours: 4"
- D. Ask the model to summarise the 60 pages first

#### [day3#lab-3-3] Offloading large results to files

Lab 3.3 · 25 min

Answer eight counting questions whose search results exceed the context budget, first in context and then with the results offloaded to a file and a tool to search it, and compare accuracy and tokens.

[D3_3_offload_large_search.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D3_3_offload_large_search.ipynb)

- [ ] How large is a large result? (4 min)
- [ ] Eight questions inside the budget, no offloading (5 min)
- [ ] Offload: a file and a stub (7 min)
- [ ] A tool to search the file (5 min)
- [ ] Tokens and accuracy, both ways (4 min)
- [ ] ✱ Stretch: Paging instead of searching

### Session 4: Knowledge graphs for linked facts

#### [day3#s4] Knowledge graphs for linked facts

*Session 4*

- Build a graph of courses, prerequisites and rules from extracted facts
- Expose graph queries as tools
- Compare relationship answers with and without the graph

#### [day3#facts-that-are-links] Some questions are about links between facts, not about any one page

*Knowledge graphs*

- "Which courses must I pass, **directly or indirectly**, before PHY 401?"
- "How many courses list CSC 201 as a prerequisite?"
- "Which study plans include DSC 223?"

> **The shape of the answer.** A chain or a set gathered from many pages. Search finds pages one at a time; a graph stores the links themselves.

#### [day3#small-graph] 368 courses, 30 programmes and 6 colleges, joined by 1,201 edges

*Lab 3.4, step 1*

1. **course** PHY 401
2. **requires** its direct prerequisites
3. **requires** theirs, on down the chain to UNI 105

- Nodes and edges come from the course pages and plans, already extracted (`data/graph/`)
- Edge kinds: `requires` (course to prerequisite), `includes` (programme to course), `in` (programme to college)
- Each college node carries a rule: the GPA its transfer rule requires
- NetworkX in memory; Neo4j or a SQL table of edges at scale

#### [day3#graph-queries-as-tools] Expose graph queries as tools, so the agent asks for a walk, not a page

*Lab 3.4, step 2*

```python
# file: Lab 3.4 · three tools
def prerequisites_of(code: str, transitive: bool = True) -> list[str]:
    """Prerequisites of a course from the course graph: direct ones,
    or all of them when transitive is true."""

def dependants_of(code: str) -> list[str]:
    """Courses that list this course as a direct prerequisite."""

def programmes_including(code: str) -> list[str]:
    """Programmes whose study plan includes the course."""
```

#### [day3#graph-results] With the graph, ten of ten; with search only, three, at fourteen times the tokens

*Measured in Lab 3.4, stand-in*

| Question kind | Search only | With the graph |
| --- | --- | --- |
| All prerequisites (3) | 0.0 | 1.0 |
| Dependants (2) | 0.0 | 1.0 |
| Must X come before Y (2) | 0.0 | 1.0 |
| Plans that include a course (3) | 1.0 | 1.0 |
| Mean calls, mean tokens | 2.2, 4,281 | 1.0, 307 |

#### [day3#when-a-graph] Build a graph for the relationships users ask about, not for the whole corpus

*When a graph earns its place*

- **Worth it.** Prerequisite chains, which plan includes what, which circular amends which article: few kinds of link, asked often.
- **Not worth it.** "What does the regulation say about appeals?": one passage answers it, and search finds it.

> **Takeaway.** A graph is a structured index for one family of questions. Keep it small, typed, and checked against its sources.

#### [day3#check-graph] Which question does the graph answer best?

*Checkpoint*

**Question:** Nibras has search tools and a small graph of courses, plans and prerequisites. Which question gains most from the graph?

- A. "Must I pass MTH 112 before PHY 401?"
- B. "What does the regulation say about appealing a grade?"
- C. "How many circulars were issued in 2024?"
- D. "What is the description of CSC 301?"

#### [day3#lab-3-4] A graph of courses, prerequisites and rules

Lab 3.4 · 25 min

Build a small graph of courses, prerequisites, programmes and each college's transfer rule, expose three graph queries as tools, and answer ten relationship questions with search only and with the graph.

[D3_4_build_small_graph.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D3_4_build_small_graph.ipynb)

- [ ] Nodes and edges from the extraction (7 min)
- [ ] Three graph queries as tools (7 min)
- [ ] Ten questions without the graph (4 min)
- [ ] The same ten with the graph (4 min)
- [ ] Accuracy, calls and tokens (3 min)
- [ ] ✱ Stretch: A rule edge: transfer requirements

#### [day3#wrap-up] Three things to keep

*Day 3 · wrap-up*

- **Tools need descriptions.** Say when to call each one; log the choice, the calls and the tokens.
- **Iterate with stop rules.** An answer, a budget, a repeated query. Compare with one search on accuracy and tokens.
- **Keep the context small.** Offload large results to files; put relationships in a small graph.

#### [day3#next] Day 4: Structured data and diagnosis

*Next*

Tomorrow: statistical questions over the enrolment warehouse, free text-to-SQL against governed metrics, what a question costs and how fast a change becomes searchable, and twenty failures attributed to the component that caused them.

## Day 4: Structured data and diagnosis

#### [day4#title] Structured data and diagnosis

*SDAIA Academy · Retrieval and Knowledge Engineering for Language Model Systems*

Day 4

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-304

#### [day4#agenda] Structured data and diagnosis

*Today*

- **Session 1 · Questions over a warehouse**: Let a model write SQL over the warehouse, score it by execution against reference answers, and read the failures by kind
- **Session 2 · Semantic layers and validated SQL**: Answer through governed metrics: the model picks a metric and filters, code compiles and validates the SQL, and small cells are suppressed
- **Session 3 · Cost, latency and freshness**: Cost per question, the 95th-percentile latency of each stage, and how fast a changed circular becomes searchable
- **Session 4 · Diagnosing failures by component**: Attribute each wrong answer to the first component that lost it, fix the weakest component, and attribute again

#### [day4#review-day3] Which design fits the question?

*Review of Day 3*

**Question:** A student asks: "How many courses of the College of Health Sciences carry 4 credit hours?" The tool that lists the college's course pages returns about 8,000 tokens; the result budget is 1,500. What should the agent do?

- A. Count in the first 1,500 tokens of the result
- B. Get a stub for the offloaded file and search it for the pattern
- C. Run vector search for "4 credit hour courses"
- D. Ask the prerequisite graph for the college's courses

### Session 1: Questions over a warehouse

#### [day4#s1] Questions over a warehouse

*Session 1*

- Give a model a schema and ask it for SQL
- Measure execution accuracy against reference answers
- Classify text-to-SQL failures and what the model could read

#### [day4#statistics-are-not-pages] No page holds "how many students enrolled last semester"

*Questions over a warehouse*

- The answer is computed from rows: a query over the enrolment warehouse
- Retrieval of documents cannot find it, however good
- The tempting shortcut: let the model write the SQL

- **Al-Nuha's warehouse.** Five tables: colleges, programmes, students, terms, enrolments. 6,967 students and 162,600 enrolments, synthetic. Owned by Abdulrahman Al-Duwaish, institutional research.

#### [day4#schema-prompt] The model sees the schema and two example rows per table: 518 tokens

*Lab 4.1, step 1*

```sql
-- file: Lab 4.1 · the schema prompt
TABLE enrolments (student_id BIGINT, term_id VARCHAR, course_code VARCHAR,
  credit_hours BIGINT, status VARCHAR, letter VARCHAR, grade_points DOUBLE)
  e.g. (4400007, '2022-2', 'CSC 101', 3, 'completed', 'C', 3.0)
TABLE terms (term_id VARCHAR, academic_year VARCHAR, semester BIGINT,
  hijri VARCHAR, name_en VARCHAR, name_ar VARCHAR)
  e.g. ('2022-1', '2022/2023', 1, '1444-1', 'First semester 2022/2023', ...)
```

> **What the schema does not say.** That a withdrawn enrolment does not make a student enrolled, that term ids are Gregorian, or that MIS was merged into IS.

#### [day4#execution-accuracy] Execution accuracy: run the query and compare the result with the reference

*How to score SQL*

- **Not the text.** Two different queries can both be right.
- **Not "it ran".** A query that runs can return a wrong number.
- **The result.** Equal to the reference answer, within rounding for rates.

Benchmarks such as BIRD report execution accuracy for the same reason.

#### [day4#free-sql-results] Free text-to-SQL got 12 of 20 right, and every wrong one looked right

*Measured in Lab 4.1, recorded replies*

**0.60** execution accuracy on 20 questions

| Item | Value |
| --- | --- |
| filter missing | 3 |
| renamed programme | 2 |
| calendar | 1 |
| denominator | 1 |
| weighting | 1 |

> **No query failed to run.** All eight wrong answers are numbers a reader would believe.

#### [day4#failure-examples] Each wrong query breaks a rule the schema never stated

*Read the failures by kind*

| Question | Model's query did | Got | Right |
| --- | --- | --- | --- |
| S01 enrolled students, a term | used the Hijri name as the term id | 0 | 5,210 |
| S06 GPA of CSC courses | averaged grade points, not credit-weighted | 3.58 | 3.66 |
| S09 credit hours, Engineering | counted withdrawn enrolments | 13,092 | 12,270 |
| S15 students in IS | missed the merged MIS programme | 70 | 145 |

#### [day4#what-sql-reads] 12 of 20 free queries read the Restricted students table

*Lab 4.1, step 5*

- To filter by college or programme, the queries join **students**, which holds gender, nationality and status per person
- The answer is an aggregate; the query still reads every student's row
- A model that may write any SQL may also write one that returns rows, not counts

> **In Saudi practice.** Student records are personal data under the Personal Data Protection Law; the lab treats the table as Restricted, in line with the national Data Classification Policy.

#### [day4#check-free-sql] What kind of failure is this?

*Checkpoint*

**Question:** "How many students were enrolled in the second semester of 2025/2026?" The model's query counts every student with an enrolment row in the term and returns 4,912. The reference returns 4,780. The query runs without error. What went wrong most likely?

- A. A syntax error the database ignored
- B. The model could not read the enrolments table
- C. A missing filter: withdrawn enrolments were counted as enrolled
- D. The model is too small for SQL

#### [day4#lab-4-1] Free text-to-SQL on 20 statistical questions

Lab 4.1 · 25 min

Ask a model for SQL on 20 statistical questions over the student warehouse, measure execution accuracy against the reference answers, classify each failure, and list which tables the queries read.

[D4_1_run_free_text.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D4_1_run_free_text.ipynb)

- [ ] The schema the model sees (4 min)
- [ ] Ask for SQL on 20 questions (6 min)
- [ ] Execution accuracy against the reference answers (7 min)
- [ ] Read the failures by kind (5 min)
- [ ] What the model was allowed to read (3 min)
- [ ] ✱ Stretch: Business rules in the prompt

### Session 2: Semantic layers and validated SQL

#### [day4#s2] Semantic layers and validated SQL

*Session 2*

- Define metrics as version-controlled views and a catalog
- Compile and validate metric requests before running them
- Write the decision memo from the comparison

#### [day4#governed-metrics] Define each metric once, as reviewed code, and let the model only choose

*Semantic layers*

1. **Views** enrolment_facts and student_facts: the rules applied once
2. **Catalog** 12 metrics, their dimensions, a small-cell rule
3. **Model** a request: metric, filters, group by
4. **Code** compiles, validates, runs

Tools that work this way: Cube, dbt's semantic layer, or plain SQL views and a metric file in version control, as in the lab.

#### [day4#metric-catalog] A metric's definition carries the rule the free query missed

*Lab 4.2, step 1*

| Metric | Definition in the catalog |
| --- | --- |
| `enrolled_students` | Students with at least one enrolment that was not withdrawn in the term |
| `credit_hours` | Credit hours registered, withdrawn courses excluded |
| `weighted_gpa` | Grade point average weighted by credit hours, completed courses only |
| `pass_rate` | Passed among graded enrolments; a denial counts as graded and failed |
| `graduates` | Students who graduated; term is the graduation term |

Five of the 12 metrics in data/semantic/metrics.json. The views also map the merged programme and both calendars.

#### [day4#compile-a-request] Code turns the request into SQL over the governed views only

*Lab 4.2, step 2*

```json
// file: the model's request
{"metric": "credit_hours",
 "filters": {"term": "2025-1",
             "college": "ENG"}}
```

```sql
-- file: compiled
SELECT SUM(credit_hours)
  FILTER (WHERE status <> 'withdrawn')
  AS value
FROM enrolment_facts
WHERE term IN ('2025-1')
  AND college IN ('ENG')
```

#### [day4#validate-before-running] Validate before running: known metric, known values, allowed tables

*Lab 4.2, step 3*

- Parse the SQL (sqlglot) and list every table it reads
- Refuse any table outside the semantic views
- Check each filter value exists: a term, a programme, a college
- Refuse with a reason the model or the user can act on

- refused: term = 1447-2 does not exist (a Hijri name: use hijri_term)
- refused: reads tables outside the semantic layer: ['students']

#### [day4#semantic-results] The semantic layer answered 19 of 20, refused one, and gave no wrong number

*Measured in Lab 4.2, recorded replies*

| 20 questions | Execution accuracy | Refused | Wrong and confident |
| --- | --- | --- | --- |
| Free text-to-SQL | 0.60 | 0 | 8 |
| Semantic layer, validated | 0.95 | 1 | 0 |

> **The refusal.** S17 asked about a term by its Hijri name; the request used it as a term id and the validator refused, with the reason. A refusal can be retried; a wrong number goes into a report.

#### [day4#small-cell-suppression] A cell of fewer than five students can identify them: suppress it

*Lab 4.2 stretch*

| Programme | Gender | Nationality | Value | Shown |
| --- | --- | --- | --- | --- |
| ARB | F | Non-Saudi | 2 | fewer than 5 |
| GEO | M | Non-Saudi | 4 | fewer than 5 |
| MED | F | Non-Saudi | 5 | 5 |

Enrolled students in the Arts college, first term of 2025/2026, by programme, gender and nationality. The threshold of 5 is set in the catalog.

#### [day4#decision-memo] The decision memo: one page, the evidence, the cost, when to revisit

*Lab 4.2, step 5*

> **Decision.** Nibras answers statistical questions only through the governed metrics, with validated SQL. Free text-to-SQL is not shipped.

- **Evidence**: 0.60 with 8 confident wrong numbers, against 0.95 with 1 refusal and none wrong
- **Cost**: institutional research maintains the views and the catalog; a new metric is a reviewed change
- **Data protection**: the model never reads student rows
- **Revisit** when users ask many questions no metric covers: measure them, then add metrics

#### [day4#check-semantic-layer] Does a valid request mean a right answer?

*Checkpoint*

**Question:** The validator accepts a request for `enrolled_students` in term 2025-1 for the College of Business. What does that acceptance prove?

- A. That the number answers the user's question
- B. That the result matches the reference answer
- C. That the model chose the right metric for the question
- D. That the query reads allowed views with filter values that exist

#### [day4#lab-4-2] Governed metrics and validated SQL

Lab 4.2 · 25 min

Answer the same 20 questions through a catalog of governed metrics with compiled and validated SQL, compare execution accuracy and confident wrong answers with Lab 4.1, and write the one-page decision memo.

[D4_2_answer_same_questions.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D4_2_answer_same_questions.ipynb)

- [ ] The views and the metric catalog (4 min)
- [ ] Compile a metric request to SQL (7 min)
- [ ] Validate before running (6 min)
- [ ] Twenty questions through the layer (5 min)
- [ ] The decision memo (3 min)
- [ ] ✱ Stretch: Small cells

### Session 3: Cost, latency and freshness

#### [day4#s3] Cost, latency and freshness

*Session 3*

- Compute tokens and cost per question for a pipeline
- Measure latency by stage and report its tail
- Re-index a changed document incrementally and mark its stale vector

#### [day4#three-operating-numbers] Three numbers decide whether Nibras can serve a whole university

*Cost, latency and freshness*

- **Cost per question.** Tokens in and out, times the price. The design sets the tokens.
- **Tail latency.** The 95th percentile, by stage: one student in twenty waits that long.
- **Freshness.** How long after a circular changes it becomes searchable.

#### [day4#tokens-per-question] The pages dominate a regulations question; the catalog dominates a statistics one

*Measured in Lab 4.3*

| Pipeline | Input tokens, mean | Input, p95 | Output |
| --- | --- | --- | --- |
| Regulations question, top 5 pages | 832 | 1,329 | 150 |
| Free text-to-SQL, schema prompt | 185 | 192 | 80 |
| Semantic layer, metric catalog | 1,080 | 1,087 | 60 |

Cost per question is these counts times your provider's prices (set in the lab); on a free model it is zero, and the counts are what a paid model would charge for.

#### [day4#latency-by-stage] The reranker is 99% of the retrieval tail on a laptop processor

*Measured in Lab 4.3*

| Item | Value |
| --- | --- |
| fusion | 0.09 ms |
| dense search | 4.4 ms |
| lexical search | 6.1 ms |
| query embedding | 82 ms |
| rerank 30 | 10,880 ms |

95th percentile per stage. Embedding and reranking were measured once when the supplied vectors were made, on a shared laptop without a GPU; the searches are measured in the lab.

#### [day4#one-circular-changes] A corrected circular: re-index everything, or only what changed

*Freshness*

| Circular 2026-060 corrected (summer load 10 to 12 hours) | Lexical | Embedding, estimated |
| --- | --- | --- |
| Full rebuild | 0.82 s | 385 s |
| Incremental: delete and insert one page | 0.003 s | 0.15 s |

> **What goes wrong.** The text is updated and its old vector stays. Dense search then ranks the page by what it used to say. Mark the vector stale and skip it until it is re-embedded.

#### [day4#operations-review] Report quality, cost, latency and freshness in the same document

*One table for the operations review*

- **Quality**: nDCG@10 by language (Day 1), staleness and leak suites (Day 2)
- **Cost**: input and output tokens per question by pipeline, and the price you assume
- **Latency**: p95 by stage, and the sum as an upper bound
- **Freshness**: time from a changed document to a searchable one

> **Takeaway.** A gain on one number is a decision only when the others are on the same page.

#### [day4#check-latency] Where do you cut first?

*Checkpoint*

**Question:** Nibras's retrieval p95 by stage: lexical 6 ms, dense 4 ms, fusion under 1 ms, query embedding 82 ms, rerank of the top 30 about 10,900 ms. The budget is 400 ms. What do you change first?

- A. The reranker: shrink its depth, move it to a GPU, or drop it, and measure the quality cost
- B. The fusion code
- C. A faster embedding model
- D. Lexical search

#### [day4#lab-4-3] Cost per question, tail latency and re-index time

Lab 4.3 · 25 min

Measure tokens and cost per question for three pipelines, the 95th-percentile latency of each retrieval stage, and the time to make a changed circular searchable, in full and incrementally.

[D4_3_measure_cost_per.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D4_3_measure_cost_per.ipynb)

- [ ] Tokens and cost per question, three pipelines (6 min)
- [ ] Latency by stage, and its tail (7 min)
- [ ] One circular changes: re-index everything (5 min)
- [ ] Re-index only what changed (5 min)
- [ ] One table for the operations review (2 min)
- [ ] ✱ Stretch: A latency budget per stage

### Session 4: Diagnosing failures by component

#### [day4#s4] Diagnosing failures by component

*Session 4*

- Attribute a failure to parsing, retrieval, ranking or generation
- Fix the weakest component and measure again
- Explain why failures move down the ladder

#### [day4#model-is-bad] "The model is bad" is not a diagnosis

*Diagnosing failures*

- Students reported **20 wrong answers** from Nibras
- Each has a trace: the parsed text, the 50 candidates, the top 5 the model read, the answer
- The question for each: which component lost the answer first?

- **Why it matters.** A new model does not fix an empty parse. A new parser does not fix a prompt. Fix the component that failed, then measure again.

#### [day4#attribution-ladder] Walk down the pipeline and stop at the first component that lost the answer

*The attribution ladder*

1. **1 · Parsing** are the needed words in the parsed text?
2. **2 · Retrieval** is the page among the 50 candidates?
3. **3 · Ranking** is it in the top 5 the model read?
4. **4 · Generation** does the answer contain the needed words?

> **Order matters.** A page that was never parsed cannot be retrieved; blaming retrieval would send you to the wrong fix.

#### [day4#failures-by-component] Parsing lost the most answers: 7 of 20

*Measured in Lab 4.4*

| Item | Value |
| --- | --- |
| parsing | 7 |
| retrieval | 5 |
| ranking | 5 |
| generation | 3 |

Attribution of the 20 reported failures; it matches the component each failure was built to fail in.

#### [day4#fix-and-reattribute] After the re-parse no failure is in parsing, and most moved down the ladder

*Fix the weakest component, then attribute again*

| Component | Before | After the re-parse |
| --- | --- | --- |
| parsing | 7 | 0 |
| retrieval | 5 | 6 |
| ranking | 5 | 5 |
| generation | 3 | 9 |

> **Progress, not passes.** Six fixed failures now reach the model with the right page; their recorded answers are still wrong because no model was re-run. One now fails at retrieval.

#### [day4#diagnosis-in-the-capstone] The defence asks you to explain one failure by component

*Into the capstone*

- Pick a failure from your own test set or your traces
- Show where it stopped on the ladder, with the evidence for each step
- Say what you changed, and the before and after counts

> **Takeaway.** The method is cheap: four checks per failure, no model needed. Run it every time the error count moves.

#### [day4#check-attribution] Which component lost this answer?

*Checkpoint*

**Question:** A failure's trace: the needed words are in the parsed text, the answering page is candidate 12 of 50, and the top 5 the model read do not include it. The answer says "I could not find a rule". Which component do you attribute it to?

- A. Generation
- B. Retrieval
- C. Ranking
- D. Parsing

#### [day4#lab-4-4] Twenty failures, attributed by component

Lab 4.4 · 25 min

Attribute 20 reported failures to parsing, retrieval, ranking or generation from their traces, re-parse the parsing failures, search again, and compare the counts before and after.

[D4_4_attribute_20_failures.ipynb](https://sdaiaacademy.github.io/ae-304/labs/D4_4_attribute_20_failures.ipynb)

- [ ] The attribution ladder (4 min)
- [ ] Attribute each failure (8 min)
- [ ] Count by component (3 min)
- [ ] Fix the weakest component and run again (6 min)
- [ ] Before and after (4 min)
- [ ] ✱ Stretch: The second-weakest component

#### [day4#wrap-up] Three things to keep

*Day 4 · wrap-up*

- **Govern the numbers.** Metrics as reviewed code, validated SQL: 0.95 and no confident wrong number, against 0.60 and eight.
- **Operate on three numbers.** Cost per question, p95 by stage, and the time from a change to a searchable page.
- **Diagnose by component.** The first component that lost the answer; fix it, attribute again.

#### [day4#next] Capstone: a measured retrieval layer

*Next*

Build a retrieval layer for Nibras, or for a synthetic or public corpus you bring, and publish it as a public repository whose README reports metrics by language, leak and staleness tests, latency and cost per question. Then an eight-minute defence and four minutes of questions.

## Capstone: Capstone project presentations

#### [capstone#title] Capstone project presentations

*SDAIA Academy · Retrieval and Knowledge Engineering for Language Model Systems*

Capstone

AI Engineering Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · AE-304

#### [capstone#agenda] How the presentations run

*Capstone*

- **Expectations and final prep**: the rubric, a last rehearsal, then the first presentations
- **Presentations**: each one timed, with questions after it
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#rubric] The rubric

*How presentations are scored*

| Item | Value |
| --- | --- |
| Retrieval quality and test set | 25% |
| Corpus preparation | 20% |
| Access and freshness tests | 20% |
| Agentic or structured-data component | 20% |
| Reproducibility | 15% |

> **Show results from at least two measured runs.** A claim without a number is a plan, not a result.

#### [capstone#format] 8 minutes to present, 4 to answer

*Format*

- **1 · The problem.** Who, what, why now. One slide.
- **2 · What you built.** Design, decisions, trade-offs. Then run one query live.
- **3 · What you measured.** Baseline, improvement by language, one failure explained by component, what comes next.

> **Peer feedback.** After each talk, everyone writes one strength, one question and one suggestion.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Project |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first presentation*

Rehearse once, record a backup, try the demo on this screen.

#### [capstone#now-presenting] Talk and demo

*Now presenting*

##### Questions

#### [capstone#review] What the presentations showed

*Trainer review*

- **Strongest patterns.** What several projects got right.
- **Common gaps.** What most projects would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Your project stays yours:** keep it reproducible and add the next improvement you named today
- **Next Academy course:** AE-390 AI Engineering Practicum, which builds on this course

#### [capstone#thanks] Thank you

*SDAIA Academy*

Retrieval and Knowledge Engineering for Language Model Systems
