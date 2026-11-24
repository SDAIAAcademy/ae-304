# Changes

## 2027.2.0 (2026-11-24)

- Added: query rewriting into the corpus's own words, on `vocabulary-gap` and `rewrite-from-corpus` (day 1, session 2), with an optional rewrite cell in Lab 1.2 step 5 and the practice question `re-lo2-rewrite-vocabulary`.
- Added: one embedding model per index, and re-embedding the whole corpus when the model changes, on `one-model-per-index` and `migrate-by-reembedding` (day 1, session 3), with an optional cell in Lab 1.3 step 3 and the practice question `re-lo2-model-change`.
- Added: the script `labs/_tools/rewrite_and_migration.py`, which computes the figures of the four new slides.
- Removed: `quantised-vectors` (day 1); the eight-bit stretch is introduced in the notes of `exact-or-approximate` and its result is in the notes of `lab-1-3`.
- Reworded: `dialect-failures` (one example, a new takeaway), the dialect card of `wrap-up` (day 1), Lab 1.2 checkpoint question 3; Lab 1.3 has a fourth checkpoint question.
- Notes: `one-circular-changes` (day 4) says incremental re-embedding holds only within one model.
