<!-- academy:title -->
# Retrieval and Knowledge Engineering for Language Model Systems

هندسة الاسترجاع والمعرفة لأنظمة النماذج اللغوية

AI Engineering Track · SDAIA Academy · AE-304 · Level 300

REPLACE: one line under the title: what participants can do after the course

Participants build and measure the retrieval layer of a language model system for a fictional university: bilingual regulations, study plans with tables, scanned Arabic circulars and an enrolment statistics warehouse. Starting from a graded test set, they build lexical, dense, hybrid and reranked search with Arabic normalisation, parse and chunk documents by structure, filter by date and role, give an agent search tools and a small knowledge graph, and answer statistical questions through a semantic layer instead of free text-to-SQL.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ae-304/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Prepare a retrieval test set with graded relevance judgements and compute standard ranking metrics for each language, given 60 Arabic and English queries over the corpus | Applied | a versioned test set and a report of recall at 10, reciprocal rank and discounted cumulative gain by language |
| LO2 | Implement lexical, dense, hybrid and reranked search with Arabic normalisation and compare them on quality and latency, given the corpus and the test set | Applied | a results table with ranking metrics and 95th-percentile latency per method |
| LO3 | Analyse parsing and chunking choices by scoring two parsers and two chunking schemes, given 30 scanned and digital pages with tables | Applied | a parsing score per page type and a retrieval comparison with a recommendation |
| LO4 | Build retrieval that never returns a document the requester may not see or a superseded rule, given role labels and effective dates on every chunk | Performance | a passing suite of leak and staleness tests |
| LO5 | Design an agentic search component that chooses among search tools and a small knowledge graph within a context budget, given 15 multi-step questions, a graph of linked course rules and a token budget | Performance | traces and a comparison with single-shot search on accuracy, tool calls and tokens |
| LO6 | Evaluate free text-to-SQL against a semantic layer with validated SQL and decide which to ship, given 20 statistical questions with reference answers | Performance | execution accuracy for both and a one-page decision memo |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Measure and search | Retrieval is measured on its own; Lexical search and Arabic text; Dense retrieval and embedding choice; Hybrid search and reranking | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Prepare the corpus | Parsing is a separate problem; Chunking by structure; Dates, versions and duplicates; Permissions in the index | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Agentic search and knowledge | Search as tools for an agent; Multi-step questions; Paging and offloading; Knowledge graphs for linked facts | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Structured data and diagnosis | Questions over a warehouse; Semantic layers and validated SQL; Cost, latency and freshness; Diagnosing failures by component | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Judge graded relevance for 40 bilingual queries and score a provided baseline | [`D1_1_judge_graded_relevance.ipynb`](labs/D1_1_judge_graded_relevance.ipynb) |
| 1.2 | Day 1 · Session 2 | Run BM25 with and without Arabic normalisation and compare recall on Arabic and dialect queries | [`D1_2_run_bm25_without.ipynb`](labs/D1_2_run_bm25_without.ipynb) |
| 1.3 | Day 1 · Session 3 | Embed a 300-page sample with two open multilingual models, load the supplied full indexes and compare quality, index size and latency | [`D1_3_embed_300_page.ipynb`](labs/D1_3_embed_300_page.ipynb) |
| 1.4 | Day 1 · Session 4 | Fuse lexical and dense rankings, add a reranker and report the gain against the added latency | [`D1_4_fuse_lexical_dense.ipynb`](labs/D1_4_fuse_lexical_dense.ipynb) |
| 2.1 | Day 2 · Session 1 | Score two parsers on 30 pages against reference text and tables | [`D2_1_score_two_parsers.ipynb`](labs/D2_1_score_two_parsers.ipynb) |
| 2.2 | Day 2 · Session 2 | Chunk study plans by headings and by fixed windows and compare retrieval on table questions | [`D2_2_chunk_study_plans.ipynb`](labs/D2_2_chunk_study_plans.ipynb) |
| 2.3 | Day 2 · Session 3 | Add effective dates and deduplication and test 15 questions whose answer changed in a newer circular | [`D2_3_add_effective_dates.ipynb`](labs/D2_3_add_effective_dates.ipynb) |
| 2.4 | Day 2 · Session 4 | Tag chunks with role and classification and run leak tests from a student account | [`D2_4_tag_chunks_role.ipynb`](labs/D2_4_tag_chunks_role.ipynb) |
| 3.1 | Day 3 · Session 1 | Expose file listing, keyword and vector search as tools and trace the agent's choices on 15 questions | [`D3_1_expose_file_listing.ipynb`](labs/D3_1_expose_file_listing.ipynb) |
| 3.2 | Day 3 · Session 2 | Compare single-shot and iterative search on ten multi-hop questions | [`D3_2_compare_single_shot.ipynb`](labs/D3_2_compare_single_shot.ipynb) |
| 3.3 | Day 3 · Session 3 | Offload large search results to files and measure tokens and accuracy | [`D3_3_offload_large_search.ipynb`](labs/D3_3_offload_large_search.ipynb) |
| 3.4 | Day 3 · Session 4 | Build a small graph of courses, prerequisites and rules and answer ten relationship questions with and without it | [`D3_4_build_small_graph.ipynb`](labs/D3_4_build_small_graph.ipynb) |
| 4.1 | Day 4 · Session 1 | Run free text-to-SQL on 20 statistical questions and measure execution accuracy | [`D4_1_run_free_text.ipynb`](labs/D4_1_run_free_text.ipynb) |
| 4.2 | Day 4 · Session 2 | Answer the same questions through governed metrics with SQL validation and compare | [`D4_2_answer_same_questions.ipynb`](labs/D4_2_answer_same_questions.ipynb) |
| 4.3 | Day 4 · Session 3 | Measure cost per question, tail latency and re-index time after one circular changes | [`D4_3_measure_cost_per.ipynb`](labs/D4_3_measure_cost_per.ipynb) |
| 4.4 | Day 4 · Session 4 | Attribute 20 failures to parsing, retrieval, ranking or generation and fix the weakest component | [`D4_4_attribute_20_failures.ipynb`](labs/D4_4_attribute_20_failures.ipynb) |
<!-- /academy:labs -->

## Capstone

A measured retrieval layer for the university's assistant, or for a synthetic or public corpus the participant brings, submitted as a public repository whose README reports metrics by language, leak and staleness tests, latency and cost per question, then an eight-minute defence and four minutes of questions

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- AE-203 Building Applications on Language Models
- Writes SQL joins and aggregations
- Computes precision and recall
- Has built a small grounded application with citations

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Lexical search:** SQLite full-text search, OpenSearch, PostgreSQL full-text search
- **Vector index:** pgvector, Qdrant, LanceDB, FAISS
- **Embedding and reranking models:** open-weight multilingual models through Sentence Transformers, any embedding API
- **Document parsing and character recognition:** Docling, Unstructured, Tesseract, an open-weight document vision model
- **Arabic text processing:** CAMeL Tools, PyArabic
- **Analytical engine and semantic layer:** DuckDB, PostgreSQL, Cube Core, metric definitions kept as version-controlled SQL views
- **Graph store:** NetworkX, Neo4j
- **Retrieval metrics:** ranx, pytrec_eval
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- NDMO Data Classification Policy: REPLACE: the address of the official text
- NDMO Data Management and Personal Data Protection Standards: REPLACE: the address of the official text
- SDAIA Generative AI Guidelines: REPLACE: the address of the official text
