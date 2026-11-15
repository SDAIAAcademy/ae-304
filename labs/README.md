# Retrieval and Knowledge Engineering for Language Model Systems: labs

Hands-on notebooks for the SDAIA Academy course **AE-304 Retrieval and Knowledge Engineering for Language Model Systems (هندسة الاسترجاع والمعرفة لأنظمة النماذج اللغوية)**. Every lab builds or measures a piece of the retrieval layer of **Nibras (نبراس)**, the regulations assistant of **Al-Nuha University (جامعة النُّهى)**, a fictional public university. Each lab starts from a supplied reference state (the corpus, the test set, the vectors, the parsed pages, the warehouse), so none depends on an earlier one.

Nothing runs a language model on your machine. Search, metrics, parsing scores, graphs and SQL run on a laptop processor or a Colab CPU runtime. The few steps that ask a model to decide (Day 3) or to write SQL (Day 4, Session 1) call a **free hosted model over an API**; without a working key they run a scripted stand-in and say so.

## Quick start

**1 · Optional: a free OpenRouter key.** Create one at <https://openrouter.ai/keys>. Free models (IDs ending in `:free`) have daily and per-minute limits. Identical requests are replayed from `labs/cache/`, so re-running a cell costs nothing. Without a key every notebook still runs to the end.

**2a · Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/ae-304/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/<notebook>.ipynb`.
2. Optional: in the 🔑 **Secrets** panel add `OPENROUTER_API_KEY` and turn on notebook access.
3. Run the first cell. It clones this repository and installs `requirements-colab.txt`. In Colab, Lab 1.3 embeds its 300-page sample live with the two embedding models.

**2b · Local**
```bash
git clone https://github.com/SDAIAAcademy/ae-304
cd ae-304/labs
cp .env.example .env           # optional: paste your key into .env
uv sync                        # creates .venv from pyproject.toml (uv.lock pins every version)
uv run python prepare_data.py  # optional: rebuilds data/ (committed) from the fixed seed
uv run jupyter lab
```
On Windows, if a native import fails in a deep folder (paths over 260 characters), set `UV_PROJECT_ENVIRONMENT` to a short folder such as `C:\venvs\ae304` before `uv sync`.

## Models and settings

| Setting (`.env`) | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | empty | without it, or when it is refused, the scripted stand-in runs |
| `AE304_MODEL` | `google/gemma-4-31b-it:free` | any free tool-calling model from <https://openrouter.ai/models?max_price=0&supported_parameters=tools> |
| `AE304_FALLBACKS` | `google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3.5-lightning:free` | tried in order when a free model is busy |
| `AE304_BASE_URL` | `https://openrouter.ai/api/v1` | any server with the Chat Completions shape |
| `AE304_CACHE` | `1` | `0` sends every request to the model |
| `AE304_QUICK` | `0` | `1` shrinks repeated timing runs |
| `AE304_STANDIN` | `0` | `1` uses the stand-in even with a key, for numbers that match the slides |
| `AE304_MODELS` | empty | `load` embeds live on a laptop (a download of about 470 MB per model) |
| `AE304_PRICE_IN`, `AE304_PRICE_OUT` | `0` | prices per million tokens for Lab 4.3 |

Free models change often. If the default disappears, pick another free model with tool support.

**Embedding and reranking models** (open weights): `intfloat/multilingual-e5-small`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and the cross-encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Their vectors and reranker scores for this corpus are supplied in `data/vectors/` and `data/reference/`, made once with `_tools/make_vectors.py`, with the times measured on a laptop processor in `data/reference/timings.json`.

**What the stand-in is.** `nibras_tools.py` holds short scripts that follow the steps a model is asked to follow (pick a tool, read the result, search again, answer), and Labs 4.1 and 4.2 read recorded replies (`data/reference/standin_sql.json`, `standin_semantic.json`). Numbers measured with the stand-in describe the stand-in; each notebook prints which one ran. Run with a key to measure a real model.

## Notebooks

| Day | Session | Notebook | Model calls |
|---|---|---|---|
| 1 | Retrieval is measured on its own | [`D1_1_judge_graded_relevance.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_1_judge_graded_relevance.ipynb) | none |
| 1 | Lexical search and Arabic text | [`D1_2_run_bm25_without.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_2_run_bm25_without.ipynb) | none |
| 1 | Dense retrieval and embedding choice | [`D1_3_embed_300_page.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_3_embed_300_page.ipynb) | embeddings, live in Colab only |
| 1 | Hybrid search and reranking | [`D1_4_fuse_lexical_dense.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_4_fuse_lexical_dense.ipynb) | none (supplied reranker scores) |
| 2 | Parsing is a separate problem | [`D2_1_score_two_parsers.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_1_score_two_parsers.ipynb) | none |
| 2 | Chunking by structure | [`D2_2_chunk_study_plans.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_2_chunk_study_plans.ipynb) | none |
| 2 | Dates, versions and duplicates | [`D2_3_add_effective_dates.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_3_add_effective_dates.ipynb) | none |
| 2 | Permissions in the index | [`D2_4_tag_chunks_role.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_4_tag_chunks_role.ipynb) | none |
| 3 | Search as tools for an agent | [`D3_1_expose_file_listing.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_1_expose_file_listing.ipynb) | tool calling, or the stand-in |
| 3 | Multi-step questions | [`D3_2_compare_single_shot.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_2_compare_single_shot.ipynb) | the next search, or the stand-in |
| 3 | Paging and offloading | [`D3_3_offload_large_search.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_3_offload_large_search.ipynb) | tool calling, or the stand-in |
| 3 | Knowledge graphs for linked facts | [`D3_4_build_small_graph.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_4_build_small_graph.ipynb) | tool calling, or the stand-in |
| 4 | Questions over a warehouse | [`D4_1_run_free_text.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_1_run_free_text.ipynb) | SQL, or recorded replies |
| 4 | Semantic layers and validated SQL | [`D4_2_answer_same_questions.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_2_answer_same_questions.ipynb) | a metric request, or recorded replies |
| 4 | Cost, latency and freshness | [`D4_3_measure_cost_per.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_3_measure_cost_per.ipynb) | none |
| 4 | Diagnosing failures by component | [`D4_4_attribute_20_failures.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_4_attribute_20_failures.ipynb) | none |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in; the cell after each one falls back to a supplied result, so the notebook runs to the end. Complete reference versions are in `solutions/`. Cells marked **✱ Stretch** are optional. Self-checks print ✅ or ❌.

The notebooks are generated from the percent-format sources in `_src/` by `_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions. Edit the sources, not the notebooks. Labs write their outputs to `artifacts/<lab>/`; delete `artifacts/` and `cache/` before running the course's `academy check`, which reports ignored files under `labs/`.

## The case study: Al-Nuha University

Synthetic and deterministic (seed 304), dated **1 November 2026** (the labs' "today"); no real person or record.

| Data | Folder | Contents |
|---|---|---|
| Corpus | `data/corpus/` | 2,977 pages of 2,075 documents in Arabic and English: the Undergraduate Study and Examinations Regulation (30 articles), 295 circular pages from 2023 to 2026 that amend it, 521 course pages, study plans of 30 programmes, college handbooks, service guides, news, and staff-only pages |
| Test set | `data/testset/v1/` | 60 queries (30 English, 30 Arabic of which 10 in Gulf dialect), graded judgements 0 to 3, and judging pools |
| Vectors | `data/vectors/` | both embedding models over every page (16-bit), the test-set and lab queries, a 300-page sample |
| Study plans | `data/plans/` | one Markdown plan per programme and 20 table questions (Lab 2.2) |
| Runs | `data/runs/` | the supplied baseline ranking (Lab 1.1) |
| Parsing | `data/parsing/` | 30 pages, scanned and digital, with reference text and tables and the outputs of two simulated parsers |
| Versions | `data/versions/` | 93 chunks of articles and circulars and 15 questions whose answer changed |
| Access | `data/access/` | accounts with roles and 12 leak tests |
| Agent | `data/agent/` | 15 tool-choice questions, 15 multi-step questions, 8 counting questions with large results |
| Graph | `data/graph/` | courses, programmes, colleges and 10 relationship questions |
| Warehouse | `data/warehouse/` | 6,967 students and 162,600 enrolments (DuckDB over CSV), 20 statistical questions with reference SQL and answers |
| Semantic layer | `data/semantic/` | governed views (`views.sql`) and a catalog of 12 metrics with a small-cell threshold of 5 |
| Diagnosis | `data/diagnosis/` | 20 failure traces (synthetic) and the component each was built to fail in |
| Reference states | `data/reference/` | baseline runs, reranker scores, timings, stand-in replies and the fallback results of the exercises |

## Planted problems (trainer's key)

| Lab | What is planted | What participants should find |
|---|---|---|
| 1.1 | Arabic queries, 10 in dialect | the raw BM25 baseline: nDCG@10 0.415 Arabic, 0.532 English, dialect recall 0.05 |
| 1.2 | spelling variants (hamza, ya, ta marbuta, diacritics, tatweel) between queries and pages | normalising both sides lifts Arabic recall 0.340 to 0.435, dialect 0.05 to 0.25 |
| 1.3 | e5 vectors made without their prefixes | nDCG@10 0.507 with prefixes, 0.484 without; MiniLM is better on dialect (0.325 recall) |
| 1.4 | scores on different scales; a slow reranker | RRF nDCG@10 0.570; rerank 30 reaches 0.633 at about 11 s p95 on a laptop processor |
| 2.1 | text layer: presentation forms, reversed Arabic word order, lost table pipes, empty scans; OCR: letter confusions | route scans and tables to the layout-aware parser (routed WER 0.032) |
| 2.2 | tables split by fixed windows | heading chunks answer 0.65 of the table questions in the top 3, 120-word windows 0.25 |
| 2.3 | amended articles, a rule not yet in force (study load, effective 1 January 2027), near-duplicate copies of circulars | 3 of 15 current answers on top before, 15 of 15 after; 8 copies dropped |
| 2.4 | staff pages whose own label is missing; a circular (2026-777) carrying an instruction to assistants | 103 Restricted results with no filter; pre-filtering ranks only allowed pages; the circular is flagged |
| 3.1 | tool descriptions; questions that need listing, codes or meaning | listing for "how many", keyword for codes, vector for paraphrases (stand-in accuracy 0.73) |
| 3.2 | two-hop questions (course to college to rule, course to prerequisite to credits or semester) | iterative 0.9 against single-shot 0.7, at about twice the tokens (stand-in) |
| 3.3 | results of 1,600 to 8,500 tokens against a 1,500-token budget | offloading: accuracy 0.5 to 1.0, tokens about 1,724 to 924 (stand-in) |
| 3.4 | prerequisite chains several links deep | with the graph 1.0, with search only 0.3 (stand-in) |
| 4.1 | warehouse traps: withdrawn enrolments, Gregorian term ids with Hijri names, MIS merged into IS, credit-weighted GPA, denominators | execution accuracy 0.60 with recorded replies; 12 of 20 queries read the students table |
| 4.2 | the same questions through governed metrics | 0.95, one refusal, no confident wrong numbers; small cells suppressed |
| 4.3 | one circular changed (2026-060, summer load 10 to 12 hours) | full rebuild against an incremental update; the stale vector skipped until re-embedded |
| 4.4 | 20 failures built to fail in a given component | 7 parsing, 5 retrieval, 5 ranking, 3 generation; after the re-parse, none in parsing |

## Standalone tools

| File | What it does |
|---|---|
| `prepare_data.py` | rebuilds `data/` from the fixed seed (standard library only) |
| `_tools/make_reference.py` | the baseline run and the judging pools |
| `_tools/make_sql.py` | the reference SQL, answers and recorded replies of Day 4 |
| `_tools/make_vectors.py` | the supplied vectors, reranker scores and timings ([PEP 723](https://peps.python.org/pep-0723/) script; downloads the models, so it is for the course author, not for participants) |
| `_tools/py2nb.py` | the notebooks and solutions from `_src/` |
