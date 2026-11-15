---
course: "Retrieval and Knowledge Engineering for Language Model Systems"
code: "AE-304"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-304/slides/day3.html"
---

# Retrieval and Knowledge Engineering for Language Model Systems

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
