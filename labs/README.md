# Retrieval and Knowledge Engineering for Language Model Systems: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Retrieval is measured on its own | [`D1_1_judge_graded_relevance.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_1_judge_graded_relevance.ipynb) |
| Day 1 | Session 2 · Lexical search and Arabic text | [`D1_2_run_bm25_without.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_2_run_bm25_without.ipynb) |
| Day 1 | Session 3 · Dense retrieval and embedding choice | [`D1_3_embed_300_page.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_3_embed_300_page.ipynb) |
| Day 1 | Session 4 · Hybrid search and reranking | [`D1_4_fuse_lexical_dense.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D1_4_fuse_lexical_dense.ipynb) |
| Day 2 | Session 1 · Parsing is a separate problem | [`D2_1_score_two_parsers.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_1_score_two_parsers.ipynb) |
| Day 2 | Session 2 · Chunking by structure | [`D2_2_chunk_study_plans.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_2_chunk_study_plans.ipynb) |
| Day 2 | Session 3 · Dates, versions and duplicates | [`D2_3_add_effective_dates.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_3_add_effective_dates.ipynb) |
| Day 2 | Session 4 · Permissions in the index | [`D2_4_tag_chunks_role.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D2_4_tag_chunks_role.ipynb) |
| Day 3 | Session 1 · Search as tools for an agent | [`D3_1_expose_file_listing.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_1_expose_file_listing.ipynb) |
| Day 3 | Session 2 · Multi-step questions | [`D3_2_compare_single_shot.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_2_compare_single_shot.ipynb) |
| Day 3 | Session 3 · Paging and offloading | [`D3_3_offload_large_search.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_3_offload_large_search.ipynb) |
| Day 3 | Session 4 · Knowledge graphs for linked facts | [`D3_4_build_small_graph.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D3_4_build_small_graph.ipynb) |
| Day 4 | Session 1 · Questions over a warehouse | [`D4_1_run_free_text.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_1_run_free_text.ipynb) |
| Day 4 | Session 2 · Semantic layers and validated SQL | [`D4_2_answer_same_questions.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_2_answer_same_questions.ipynb) |
| Day 4 | Session 3 · Cost, latency and freshness | [`D4_3_measure_cost_per.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_3_measure_cost_per.ipynb) |
| Day 4 | Session 4 · Diagnosing failures by component | [`D4_4_attribute_20_failures.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-304/blob/main/labs/D4_4_attribute_20_failures.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
