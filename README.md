<!-- academy:title -->
# Retrieval and Knowledge Engineering for Language Model Systems

هندسة الاسترجاع والمعرفة لأنظمة النماذج اللغوية

AI Engineering Track · SDAIA Academy · AE-304 · Level 300

Build and measure the retrieval layer of a language model system: search, corpus preparation, access, agents and structured data

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

**Al-Nuha University (جامعة النُّهى)**, a fictional public university with six colleges, is piloting **Nibras (نبراس)**, an assistant that answers students and staff from its regulations, circulars, course pages, study plans, service guides and an enrolment warehouse. Participants work as the retrieval engineer who must show the vice-rector for digital transformation, with numbers, what Nibras reads and why it answers wrongly when it does.

All data is synthetic and generated with a fixed seed by `labs/prepare_data.py`: 2,977 pages in Arabic and English (with Gulf-dialect queries, simulated scans, superseded rules, near-duplicate circulars and staff-only pages), a test set of 60 queries with graded judgements, and a warehouse of 6,967 students and 162,600 enrolments. No real person or record is in it. Vectors for the corpus are supplied, so no lab downloads a model on a laptop.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Graded judgements and a baseline score | [`D1_1_judge_graded_relevance.ipynb`](labs/D1_1_judge_graded_relevance.ipynb) |
| 1.2 | Day 1 · Session 2 | BM25 with and without Arabic normalisation | [`D1_2_run_bm25_without.ipynb`](labs/D1_2_run_bm25_without.ipynb) |
| 1.3 | Day 1 · Session 3 | Two embedding models: quality, size and latency | [`D1_3_embed_300_page.ipynb`](labs/D1_3_embed_300_page.ipynb) |
| 1.4 | Day 1 · Session 4 | Fusion, a reranker and its latency | [`D1_4_fuse_lexical_dense.ipynb`](labs/D1_4_fuse_lexical_dense.ipynb) |
| 2.1 | Day 2 · Session 1 | Two parsers scored on 30 pages | [`D2_1_score_two_parsers.ipynb`](labs/D2_1_score_two_parsers.ipynb) |
| 2.2 | Day 2 · Session 2 | Two chunkings of the study plans | [`D2_2_chunk_study_plans.ipynb`](labs/D2_2_chunk_study_plans.ipynb) |
| 2.3 | Day 2 · Session 3 | Effective dates, versions and duplicates | [`D2_3_add_effective_dates.ipynb`](labs/D2_3_add_effective_dates.ipynb) |
| 2.4 | Day 2 · Session 4 | Role labels and leak tests | [`D2_4_tag_chunks_role.ipynb`](labs/D2_4_tag_chunks_role.ipynb) |
| 3.1 | Day 3 · Session 1 | Three search tools and a traced agent | [`D3_1_expose_file_listing.ipynb`](labs/D3_1_expose_file_listing.ipynb) |
| 3.2 | Day 3 · Session 2 | Single-shot against iterative search | [`D3_2_compare_single_shot.ipynb`](labs/D3_2_compare_single_shot.ipynb) |
| 3.3 | Day 3 · Session 3 | Offloading large results to files | [`D3_3_offload_large_search.ipynb`](labs/D3_3_offload_large_search.ipynb) |
| 3.4 | Day 3 · Session 4 | A graph of courses, prerequisites and rules | [`D3_4_build_small_graph.ipynb`](labs/D3_4_build_small_graph.ipynb) |
| 4.1 | Day 4 · Session 1 | Free text-to-SQL on 20 statistical questions | [`D4_1_run_free_text.ipynb`](labs/D4_1_run_free_text.ipynb) |
| 4.2 | Day 4 · Session 2 | Governed metrics and validated SQL | [`D4_2_answer_same_questions.ipynb`](labs/D4_2_answer_same_questions.ipynb) |
| 4.3 | Day 4 · Session 3 | Cost per question, tail latency and re-index time | [`D4_3_measure_cost_per.ipynb`](labs/D4_3_measure_cost_per.ipynb) |
| 4.4 | Day 4 · Session 4 | Twenty failures, attributed by component | [`D4_4_attribute_20_failures.ipynb`](labs/D4_4_attribute_20_failures.ipynb) |
<!-- /academy:labs -->

## Capstone

A measured retrieval layer for the university's assistant, or for a synthetic or public corpus the participant brings, submitted as a public repository whose README reports metrics by language, leak and staleness tests, latency and cost per question, then an eight-minute defence and four minutes of questions

**Submission:** a public repository link. Its README reports, for the participant's retrieval layer, the ranking metrics by language and query type on a versioned test set, the leak and staleness test results, the 95th-percentile latency by stage and the cost per question, with the code, the test set and the run files committed. Then an eight-minute defence that runs one query live and explains one failure by component, and four minutes of questions.

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

### Measuring retrieval

- [Voorhees and Harman: Overview of the Eighth Text REtrieval Conference (TREC-8), pooling](https://trec.nist.gov/pubs/trec8/papers/overview_8.pdf)
- [MIRACL: a multilingual retrieval benchmark, reported per language](https://project-miracl.github.io/)

### Lexical, dense and hybrid search

- [SQLite FTS5 extension](https://www.sqlite.org/fts5.html)
- [Robertson and Zaragoza: The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Model card: intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [Model card: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [Model card: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- [Malkov and Yashunin: Efficient and robust approximate nearest neighbor search using HNSW graphs](https://arxiv.org/abs/1603.09320)
- [Cormack, Clarke and Buettcher: Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

### Corpus preparation, agents and context

- [Anthropic: Introducing contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [OWASP GenAI Security Project: LLM01 Prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

### Structured data

- [BIRD: a big bench for large-scale database grounded text-to-SQL evaluation](https://bird-bench.github.io/)
- [Cube documentation: data modeling overview](https://cube.dev/docs/product/data-modeling/overview)
- [sqlglot: SQL parser, transpiler and optimizer](https://github.com/tobymao/sqlglot)

### National references

- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [Data Classification Policy (National Data Management Office)](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [Data Management and Personal Data Protection Standards (National Data Management Office)](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataManagementPersonalDataProtectionStandards.pdf)
- [Generative AI Guideline for Government](https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf)
- [Generative AI Guideline for Public](https://sdaia.gov.sa/en/SDAIA/about/Files/GenerativeAIPublicEN.pdf)
<!-- /academy:sources -->

### National references

- Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf
- Implementing Regulation of the Personal Data Protection Law: https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
- NDMO Data Classification Policy: https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf
- NDMO Data Management and Personal Data Protection Standards: https://sdaia.gov.sa/en/SDAIA/about/Documents/DataManagementPersonalDataProtectionStandards.pdf
- SDAIA Generative AI Guideline for Government: https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf
- SDAIA Generative AI Guideline for Public: https://sdaia.gov.sa/en/SDAIA/about/Files/GenerativeAIPublicEN.pdf
