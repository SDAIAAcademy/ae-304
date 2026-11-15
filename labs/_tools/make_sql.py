"""Write the Day 4 questions, their reference SQL and answers, and the stand-in replies.

    uv run python _tools/make_sql.py

The reference SQL is the institutional research unit's definition of each number. The stand-in
replies are written by the course author to show the mistakes free text-to-SQL typically makes
(a missed filter, a wrong denominator, a calendar the data does not use, a renamed programme);
they are not a model's output. With a key, the labs ask the hosted model instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import nuha as U  # noqa: E402

E = "enrolments e"
SP = "JOIN students s USING (student_id) JOIN programmes p ON p.programme_id = s.programme_id"
PROG = "COALESCE(NULLIF(p.merged_into, ''), p.programme_id)"

# qid, question, reference SQL, free text-to-SQL stand-in reply, semantic-layer stand-in request
Q = [
    ("S01", "How many students were enrolled in the first semester of 1447?",
     f"SELECT COUNT(DISTINCT student_id) FROM {E} WHERE term_id = '2025-1' AND status <> 'withdrawn'",
     "SELECT COUNT(DISTINCT student_id) FROM enrolments WHERE term_id = '1447-1'",
     {"metric": "enrolled_students", "filters": {"hijri_term": "1447-1"}}),
    ("S02", "How many new students were admitted in the first semester of 2024/2025?",
     "SELECT COUNT(*) FROM students WHERE admit_term = '2024-1'",
     "SELECT COUNT(*) FROM students WHERE admit_term = '2024-1'",
     {"metric": "admitted_students", "filters": {"term": "2024-1"}}),
    ("S03", "How many new students did the Information Systems programme admit in 2023-1?",
     f"SELECT COUNT(*) FROM students s JOIN programmes p ON p.programme_id = s.programme_id WHERE {PROG} = 'IS' AND admit_term = '2023-1'",
     "SELECT COUNT(*) FROM students WHERE programme_id = 'IS' AND admit_term = '2023-1'",
     {"metric": "admitted_students", "filters": {"term": "2023-1", "programme": "IS"}}),
    ("S04", "What share of enrolments were withdrawn in 2025-2?",
     f"SELECT AVG(CASE WHEN status = 'withdrawn' THEN 1.0 ELSE 0 END) FROM {E} WHERE term_id = '2025-2'",
     "SELECT COUNT(*) FILTER (WHERE status = 'withdrawn') * 1.0 / COUNT(*) FROM enrolments WHERE term_id = '2025-2'",
     {"metric": "withdrawal_rate", "filters": {"term": "2025-2"}}),
    ("S05", "What was the pass rate in UNI 105 in 2025-1?",
     f"SELECT SUM(CASE WHEN status = 'completed' AND letter <> 'F' THEN 1.0 ELSE 0 END) / SUM(CASE WHEN status IN ('completed', 'denied') THEN 1 ELSE 0 END) FROM {E} WHERE term_id = '2025-1' AND course_code = 'UNI 105'",
     "SELECT AVG(CASE WHEN letter <> 'F' THEN 1.0 ELSE 0 END) FROM enrolments WHERE term_id = '2025-1' AND course_code = 'UNI 105' AND status = 'completed'",
     {"metric": "pass_rate", "filters": {"term": "2025-1", "course": "UNI 105"}}),
    ("S06", "What was the average grade point in CSC courses in 2024-1, weighted by credit hours?",
     f"SELECT SUM(grade_points * credit_hours) / SUM(credit_hours) FROM {E} WHERE term_id = '2024-1' AND course_code LIKE 'CSC %' AND status = 'completed'",
     "SELECT AVG(grade_points) FROM enrolments WHERE term_id = '2024-1' AND course_code LIKE 'CSC%'",
     {"metric": "weighted_gpa", "filters": {"term": "2024-1", "course_prefix": "CSC"}}),
    ("S07", "How many female students were enrolled in the second semester of 1446?",
     f"SELECT COUNT(DISTINCT e.student_id) FROM {E} JOIN students s USING (student_id) WHERE e.term_id = '2024-2' AND e.status <> 'withdrawn' AND s.gender = 'F'",
     "SELECT COUNT(DISTINCT e.student_id) FROM enrolments e JOIN students s ON e.student_id = s.student_id JOIN terms t ON t.term_id = e.term_id WHERE t.hijri = '1446-2' AND s.gender = 'F'",
     {"metric": "enrolled_students", "filters": {"hijri_term": "1446-2", "gender": "F"}}),
    ("S08", "How many non-Saudi students were enrolled in 2026-1?",
     f"SELECT COUNT(DISTINCT e.student_id) FROM {E} JOIN students s USING (student_id) WHERE e.term_id = '2026-1' AND e.status <> 'withdrawn' AND s.nationality = 'Non-Saudi'",
     "SELECT COUNT(DISTINCT e.student_id) FROM enrolments e JOIN students s ON e.student_id = s.student_id WHERE e.term_id = '2026-1' AND e.status <> 'withdrawn' AND s.nationality = 'Non-Saudi'",
     {"metric": "enrolled_students", "filters": {"term": "2026-1", "nationality": "Non-Saudi"}}),
    ("S09", "How many credit hours did students of the College of Engineering register in 2025-1?",
     f"SELECT SUM(e.credit_hours) FROM {E} {SP} WHERE e.term_id = '2025-1' AND e.status <> 'withdrawn' AND p.college_id = 'ENG'",
     "SELECT SUM(e.credit_hours) FROM enrolments e JOIN students s ON e.student_id = s.student_id JOIN programmes p ON s.programme_id = p.programme_id WHERE e.term_id = '2025-1' AND p.college_id = 'ENG'",
     {"metric": "credit_hours", "filters": {"term": "2025-1", "college": "ENG"}}),
    ("S10", "How many students graduated in 2026-1?",
     "SELECT COUNT(*) FROM students WHERE grad_term = '2026-1'",
     "SELECT COUNT(*) FROM students WHERE status = 'graduated' AND grad_term = '2026-1'",
     {"metric": "graduates", "filters": {"term": "2026-1"}}),
    ("S11", "Which programme had the most students enrolled in 2025-1?",
     f"SELECT {PROG} AS programme FROM {E} {SP} WHERE e.term_id = '2025-1' AND e.status <> 'withdrawn' GROUP BY 1 ORDER BY COUNT(DISTINCT e.student_id) DESC LIMIT 1",
     "SELECT s.programme_id FROM enrolments e JOIN students s ON e.student_id = s.student_id WHERE e.term_id = '2025-1' GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1",
     {"metric": "enrolled_students", "filters": {"term": "2025-1"}, "group_by": ["programme"], "order": "desc", "limit": 1}),
    ("S12", "What was the withdrawal rate in the College of Business in 2024-2?",
     f"SELECT AVG(CASE WHEN e.status = 'withdrawn' THEN 1.0 ELSE 0 END) FROM {E} {SP} WHERE e.term_id = '2024-2' AND p.college_id = 'BUS'",
     "SELECT AVG(CASE WHEN e.status = 'withdrawn' THEN 1.0 ELSE 0 END) FROM enrolments e JOIN students s ON e.student_id = s.student_id JOIN programmes p ON s.programme_id = p.programme_id WHERE e.term_id = '2024-2' AND p.college_id = 'BUS'",
     {"metric": "withdrawal_rate", "filters": {"term": "2024-2", "college": "BUS"}}),
    ("S13", "How many enrolments ended in denial of the final examination in 2025-1?",
     f"SELECT COUNT(*) FROM {E} WHERE term_id = '2025-1' AND status = 'denied'",
     "SELECT COUNT(*) FROM enrolments WHERE term_id = '2025-1' AND letter = 'DN'",
     {"metric": "denied_enrolments", "filters": {"term": "2025-1"}}),
    ("S14", "On average, how many courses did each enrolled student take in 2025-1?",
     f"SELECT COUNT(*) * 1.0 / COUNT(DISTINCT student_id) FROM {E} WHERE term_id = '2025-1' AND status <> 'withdrawn'",
     "SELECT COUNT(*) * 1.0 / COUNT(DISTINCT student_id) FROM enrolments WHERE term_id = '2025-1'",
     {"metric": "courses_per_student", "filters": {"term": "2025-1"}}),
    ("S15", "How many students of the Information Systems programme were enrolled in 2025-2?",
     f"SELECT COUNT(DISTINCT e.student_id) FROM {E} {SP} WHERE e.term_id = '2025-2' AND e.status <> 'withdrawn' AND {PROG} = 'IS'",
     "SELECT COUNT(DISTINCT e.student_id) FROM enrolments e JOIN students s ON e.student_id = s.student_id WHERE e.term_id = '2025-2' AND e.status <> 'withdrawn' AND s.programme_id = 'IS'",
     {"metric": "enrolled_students", "filters": {"term": "2025-2", "programme": "IS"}}),
    ("S16", "What was the pass rate across all courses in 2025-2?",
     f"SELECT SUM(CASE WHEN status = 'completed' AND letter <> 'F' THEN 1.0 ELSE 0 END) / SUM(CASE WHEN status IN ('completed', 'denied') THEN 1 ELSE 0 END) FROM {E} WHERE term_id = '2025-2'",
     "SELECT SUM(CASE WHEN status = 'completed' AND letter <> 'F' THEN 1.0 ELSE 0 END) / SUM(CASE WHEN status IN ('completed', 'denied') THEN 1 ELSE 0 END) FROM enrolments WHERE term_id = '2025-2'",
     {"metric": "pass_rate", "filters": {"term": "2025-2"}}),
    ("S17", "How many incomplete grades were recorded in the second semester of 1447?",
     f"SELECT COUNT(*) FROM {E} WHERE term_id = '2025-2' AND status = 'incomplete'",
     "SELECT COUNT(*) FROM enrolments e JOIN terms t ON t.term_id = e.term_id WHERE t.hijri = '1447-2' AND e.letter = 'IC'",
     {"metric": "incomplete_grades", "filters": {"term": "1447-2"}}),
    ("S18", "How many students did the College of Health Sciences admit in 2022/2023 and 2023/2024 together?",
     "SELECT COUNT(*) FROM students s JOIN programmes p ON p.programme_id = s.programme_id WHERE p.college_id = 'HSC' AND s.admit_term IN ('2022-1', '2023-1')",
     "SELECT COUNT(*) FROM students s JOIN programmes p ON p.programme_id = s.programme_id WHERE p.college_id = 'HSC' AND s.admit_term BETWEEN '2022-1' AND '2023-1'",
     {"metric": "admitted_students", "filters": {"college": "HSC", "academic_year": ["2022/2023", "2023/2024"]}}),
    ("S19", "What was the credit-weighted grade point average of College of Sciences students in 2025-1?",
     f"SELECT SUM(e.grade_points * e.credit_hours) / SUM(e.credit_hours) FROM {E} {SP} WHERE e.term_id = '2025-1' AND e.status = 'completed' AND p.college_id = 'SCI'",
     "SELECT SUM(e.grade_points * e.credit_hours) / SUM(e.credit_hours) FROM enrolments e JOIN students s ON e.student_id = s.student_id JOIN programmes p ON s.programme_id = p.programme_id WHERE e.term_id = '2025-1' AND p.college_id = 'SCI'",
     {"metric": "weighted_gpa", "filters": {"term": "2025-1", "college": "SCI"}}),
    ("S20", "How many students admitted in 2022-1 are still active?",
     "SELECT COUNT(*) FROM students WHERE admit_term = '2022-1' AND status = 'active'",
     "SELECT COUNT(*) FROM students WHERE admit_term = '2022-1' AND status = 'active'",
     {"metric": "active_students", "filters": {"admit_term": "2022-1"}}),
]


def main():
    con = U.warehouse()
    rows = []
    for qid, q, ref, *_ in Q:
        val = con.sql(ref).fetchall()[0][0]
        rows.append({"qid": qid, "q": q, "reference_sql": ref, "answer": round(val, 4) if isinstance(val, float) else val})
    with (U.DATA / "warehouse" / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (U.REF).mkdir(parents=True, exist_ok=True)
    (U.REF / "standin_sql.json").write_text(json.dumps({qid: sql for qid, _, _, sql, _ in Q}, indent=1) + "\n", encoding="utf-8")
    (U.REF / "standin_semantic.json").write_text(json.dumps({qid: req for qid, *_, req in Q}, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(r["qid"], r["answer"])


if __name__ == "__main__":
    main()
