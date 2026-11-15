---
course: "Retrieval and Knowledge Engineering for Language Model Systems"
code: "AE-304"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-304/slides/day4.html"
---

# Retrieval and Knowledge Engineering for Language Model Systems

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
