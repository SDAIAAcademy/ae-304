---
course: "Retrieval and Knowledge Engineering for Language Model Systems"
code: "AE-304"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-304/index.html"
---

# Retrieval and Knowledge Engineering for Language Model Systems

هندسة الاسترجاع والمعرفة لأنظمة النماذج اللغوية

AI Engineering Track · 4 teaching days + capstone day · 26 hours · Level 300

Build and measure the retrieval layer of a language model system: search, corpus preparation, access, agents and structured data

Participants build and measure the retrieval layer of a language model system for a fictional university: bilingual regulations, study plans with tables, scanned Arabic circulars and an enrolment statistics warehouse. Starting from a graded test set, they build lexical, dense, hybrid and reranked search with Arabic normalisation, parse and chunk documents by structure, filter by date and role, give an agent search tools and a small knowledge graph, and answer statistical questions through a semantic layer instead of free text-to-SQL.

**Who it is for:** Practitioners. Engineers who have completed the level 200 set or can show equivalent work: a deployed model or language model application they built and can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Al-Nuha University.** Al-Nuha University is a fictional public university with six colleges whose regulations assistant, Nibras, answers students and staff from bilingual regulations, study plans with tables, scanned Arabic circulars, staff-only records and an enrolment statistics warehouse. Participants build and measure Nibras's retrieval layer on about 3,000 synthetic pages and a synthetic warehouse, generated with a fixed seed, with planted problems: dialect queries, broken parses, superseded rules, near-duplicate circulars, mislabelled staff pages and warehouse traps.

## The course

### Day 1: Measure and search

Slides: https://sdaiaacademy.github.io/ae-304/slides/day1.md

- **Session 1: Retrieval is measured on its own**
  - Judge graded relevance against a written guideline
  - Compute recall at 10, reciprocal rank and nDCG at 10
  - Report retrieval quality by language and query type
  - [Lab 1.1: Graded judgements and a baseline score (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D1_1_judge_graded_relevance.ipynb)
- **Session 2: Lexical search and Arabic text**
  - Explain how BM25 weighs rare and repeated words
  - Normalise Arabic text and apply light stemming on both sides
  - Measure the gain on Arabic and dialect queries
  - [Lab 1.2: BM25 with and without Arabic normalisation (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D1_2_run_bm25_without.ipynb)
- **Session 3: Dense retrieval and embedding choice**
  - Embed pages and queries with an open multilingual model
  - Follow a model card's input conventions and show why they matter
  - Compare exact and approximate indexes on quality, size and latency
  - [Lab 1.3: Two embedding models: quality, size and latency (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D1_3_embed_300_page.ipynb)
- **Session 4: Hybrid search and reranking**
  - Fuse two rankings with reciprocal rank fusion
  - Rerank candidates with a cross-encoder
  - Decide on a reranker from its gain and its 95th-percentile latency
  - [Lab 1.4: Fusion, a reranker and its latency (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D1_4_fuse_lexical_dense.ipynb)

### Day 2: Prepare the corpus

Slides: https://sdaiaacademy.github.io/ae-304/slides/day2.md

- **Session 1: Parsing is a separate problem**
  - Score parsed text against reference text with a word error rate
  - Score tables cell by cell
  - Route pages to a parser by page type
  - [Lab 2.1: Two parsers scored on 30 pages (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D2_1_score_two_parsers.ipynb)
- **Session 2: Chunking by structure**
  - Chunk documents by headings and keep the heading path
  - Compare chunkings by retrieval on table questions
  - Recommend a chunking from the evidence
  - [Lab 2.2: Two chunkings of the study plans (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D2_2_chunk_study_plans.ipynb)
- **Session 3: Dates, versions and duplicates**
  - Read effective dates and amended articles from circulars
  - Keep a validity interval for every version
  - Remove near-duplicates and write staleness tests
  - [Lab 2.3: Effective dates, versions and duplicates (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D2_3_add_effective_dates.ipynb)
- **Session 4: Permissions in the index**
  - Tag every chunk with its document's classification and roles
  - Filter before ranking rather than after
  - Run leak and utility tests from a student account
  - [Lab 2.4: Role labels and leak tests (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D2_4_tag_chunks_role.ipynb)

### Day 3: Agentic search and knowledge

Slides: https://sdaiaacademy.github.io/ae-304/slides/day3.md

- **Session 1: Search as tools for an agent**
  - Write search tools whose descriptions say when to use them
  - Trace an agent's tool choices
  - Judge a choice by its cost and its answer
  - [Lab 3.1: Three search tools and a traced agent (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D3_1_expose_file_listing.ipynb)
- **Session 2: Multi-step questions**
  - Write an iterative search loop with stop rules
  - Compare single-shot and iterative search on accuracy, searches and tokens
  - Set a search budget from the evidence
  - [Lab 3.2: Single-shot against iterative search (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D3_2_compare_single_shot.ipynb)
- **Session 3: Paging and offloading**
  - Measure what a large result costs in tokens
  - Offload results to a file behind a stub
  - Compare accuracy and tokens with and without offloading
  - [Lab 3.3: Offloading large results to files (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D3_3_offload_large_search.ipynb)
- **Session 4: Knowledge graphs for linked facts**
  - Build a graph of courses, prerequisites and rules from extracted facts
  - Expose graph queries as tools
  - Compare relationship answers with and without the graph
  - [Lab 3.4: A graph of courses, prerequisites and rules (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D3_4_build_small_graph.ipynb)

### Day 4: Structured data and diagnosis

Slides: https://sdaiaacademy.github.io/ae-304/slides/day4.md

- **Session 1: Questions over a warehouse**
  - Give a model a schema and ask it for SQL
  - Measure execution accuracy against reference answers
  - Classify text-to-SQL failures and what the model could read
  - [Lab 4.1: Free text-to-SQL on 20 statistical questions (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D4_1_run_free_text.ipynb)
- **Session 2: Semantic layers and validated SQL**
  - Define metrics as version-controlled views and a catalog
  - Compile and validate metric requests before running them
  - Write the decision memo from the comparison
  - [Lab 4.2: Governed metrics and validated SQL (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D4_2_answer_same_questions.ipynb)
- **Session 3: Cost, latency and freshness**
  - Compute tokens and cost per question for a pipeline
  - Measure latency by stage and report its tail
  - Re-index a changed document incrementally and mark its stale vector
  - [Lab 4.3: Cost per question, tail latency and re-index time (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D4_3_measure_cost_per.ipynb)
- **Session 4: Diagnosing failures by component**
  - Attribute a failure to parsing, retrieval, ranking or generation
  - Fix the weakest component and measure again
  - Explain why failures move down the ladder
  - [Lab 4.4: Twenty failures, attributed by component (25 min)](https://sdaiaacademy.github.io/ae-304/labs/D4_4_attribute_20_failures.ipynb)

### Capstone: Capstone project presentations

Slides: https://sdaiaacademy.github.io/ae-304/slides/capstone.md

## Capstone

A measured retrieval layer for the university's assistant, or for a synthetic or public corpus the participant brings, submitted as a public repository whose README reports metrics by language, leak and staleness tests, latency and cost per question, then an eight-minute defence and four minutes of questions

**What you submit:** A public repository whose README reports, for the participant's retrieval layer, the ranking metrics by language and query type on a versioned test set, the leak and staleness test results, the 95th-percentile latency by stage and the cost per question, with the code, the test set and the run files committed. Then an eight-minute defence that runs one query live and explains one failure by component, and four minutes of questions.

**Rubric:**

- Retrieval quality and test set: 25%
- Corpus preparation: 20%
- Access and freshness tests: 20%
- Agentic or structured-data component: 20%
- Reproducibility: 15%

## Outcomes

- **LO1** Prepare a retrieval test set with graded relevance judgements and compute standard ranking metrics for each language, given 60 Arabic and English queries over the corpus (Applied) Evidence: a versioned test set and a report of recall at 10, reciprocal rank and discounted cumulative gain by language
- **LO2** Implement lexical, dense, hybrid and reranked search with Arabic normalisation and compare them on quality and latency, given the corpus and the test set (Applied) Evidence: a results table with ranking metrics and 95th-percentile latency per method
- **LO3** Analyse parsing and chunking choices by scoring two parsers and two chunking schemes, given 30 scanned and digital pages with tables (Applied) Evidence: a parsing score per page type and a retrieval comparison with a recommendation
- **LO4** Build retrieval that never returns a document the requester may not see or a superseded rule, given role labels and effective dates on every chunk (Performance) Evidence: a passing suite of leak and staleness tests
- **LO5** Design an agentic search component that chooses among search tools and a small knowledge graph within a context budget, given 15 multi-step questions, a graph of linked course rules and a token budget (Performance) Evidence: traces and a comparison with single-shot search on accuracy, tool calls and tokens
- **LO6** Evaluate free text-to-SQL against a semantic layer with validated SQL and decide which to ship, given 20 statistical questions with reference answers (Performance) Evidence: execution accuracy for both and a one-page decision memo

## Tools

- **Lexical search:** SQLite full-text search, OpenSearch, PostgreSQL full-text search
- **Vector index:** pgvector, Qdrant, LanceDB, FAISS
- **Embedding and reranking models:** open-weight multilingual models through Sentence Transformers, any embedding API
- **Document parsing and character recognition:** Docling, Unstructured, Tesseract, an open-weight document vision model
- **Arabic text processing:** CAMeL Tools, PyArabic
- **Analytical engine and semantic layer:** DuckDB, PostgreSQL, Cube Core, metric definitions kept as version-controlled SQL views
- **Graph store:** NetworkX, Neo4j
- **Retrieval metrics:** ranx, pytrec_eval

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- AE-203 Building Applications on Language Models
- Writes SQL joins and aggregations
- Computes precision and recall
- Has built a small grounded application with citations

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
- [Cormack, Clarke and Buettcher: Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

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

---

SDAIA Academy · AI Engineering Track · Retrieval and Knowledge Engineering for Language Model Systems

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit 79ea004
