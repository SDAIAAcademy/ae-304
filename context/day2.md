---
course: "Retrieval and Knowledge Engineering for Language Model Systems"
code: "AE-304"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-304/slides/day2.html"
---

# Retrieval and Knowledge Engineering for Language Model Systems

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
