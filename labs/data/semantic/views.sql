-- The semantic layer of the Al-Nuha warehouse: governed views, kept under version control.
-- Owner: the institutional research unit. A change here is reviewed like code.

CREATE OR REPLACE VIEW enrolment_facts AS
SELECT e.student_id,
       e.term_id AS term,
       t.hijri AS hijri_term,
       t.academic_year,
       e.course_code AS course,
       split_part(e.course_code, ' ', 1) AS course_prefix,
       e.credit_hours,
       e.status,
       e.letter,
       e.grade_points,
       COALESCE(NULLIF(p.merged_into, ''), p.programme_id) AS programme,  -- MIS was merged into IS in 2024
       p.college_id AS college,
       s.gender,
       s.nationality
FROM enrolments e
JOIN students s USING (student_id)
JOIN programmes p ON p.programme_id = s.programme_id
JOIN terms t ON t.term_id = e.term_id;

CREATE OR REPLACE VIEW student_facts AS
SELECT s.student_id,
       s.admit_term AS term,
       s.admit_term,
       t.hijri AS hijri_term,
       t.academic_year,
       NULLIF(s.grad_term, '') AS grad_term,
       s.status,
       COALESCE(NULLIF(p.merged_into, ''), p.programme_id) AS programme,
       p.college_id AS college,
       s.gender,
       s.nationality
FROM students s
JOIN programmes p ON p.programme_id = s.programme_id
JOIN terms t ON t.term_id = s.admit_term;
