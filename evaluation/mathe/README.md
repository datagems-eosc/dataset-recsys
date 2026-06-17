# MathE Document Recommender Evaluation

This folder describes offline evaluation ideas for the current MathE recommender.
The recommender currently returns document teaching materials only:

```text
platform_materials.type = 3
file_ext IN ('pdf', 'docx', 'pptx')
```

Videos and video transcripts are outside the scope of this evaluation.

## Benchmark Question Set

The MathE database contains historical question attempts in the `assessment`
table. We use these attempts to build an offline benchmark grounded in questions
that were actually used by students.

Specifically, we pick one representative question per topic/subtopic pool. Inside
each pool, we first select the question attempted by the largest number of
distinct students. Ties are then broken by the highest wrong-answer rate, and
then by the highest number of total attempts.

This matches the recommender setup, because the current MathE recommender also
operates inside a specific topic/subtopic pool. The benchmark therefore focuses
on real student questions from each curricular pool, especially questions where
students are more likely to benefit from useful material recommendations before
attempting similar questions again.

```sql
WITH question_stats AS (
  SELECT
    q.id AS question_id,
    q.question,
    q.topic AS _topic_id,
    q.subtopic AS _subtopic_id,
    t.name AS topic_name,
    s.name AS subtopic_name,
    ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords,
    COUNT(*) AS total_attempts,
    COUNT(DISTINCT a.student_id) AS distinct_students,
    1.0 - AVG(CASE WHEN a.answer = 1 THEN 1.0 ELSE 0.0 END) AS wrong_rate
  FROM assessment a
  JOIN platform__sna__questions q
    ON q.id = a.question_id
  LEFT JOIN platform__topic t
    ON q.topic = t.id
  LEFT JOIN platform__subtopic s
    ON q.subtopic = s.id
  LEFT JOIN platform_keyword_snaquestion qk
    ON q.id = qk.platformsnaquestionid
  LEFT JOIN platform__keywords k
    ON qk.platformkeywordid = k.id
  GROUP BY
    q.id,
    q.question,
    q.topic,
    q.subtopic,
    t.name,
    s.name
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY _topic_id, _subtopic_id
      ORDER BY distinct_students DESC, wrong_rate DESC, total_attempts DESC
    ) AS rn
  FROM question_stats
)
SELECT
  question_id,
  question,
  topic_name,
  subtopic_name,
  keywords,
  total_attempts,
  distinct_students,
  wrong_rate
FROM ranked
WHERE rn = 1
ORDER BY distinct_students DESC, wrong_rate DESC;
```

Interpretation:

```text
total_attempts     total historical assessment rows for the question
distinct_students  number of different students who attempted it
wrong_rate         proportion of attempts with answer != 1
topic_name         human-readable topic label
subtopic_name      human-readable subtopic label, when available
keywords           question keywords, when available
```

The benchmark can be exported to Excel with:

```bash
poetry run python evaluation/mathe/export_benchmark_questions.py
```

By default, this writes:

```text
evaluation/mathe/mathe_benchmark_questions.xlsx
```

Use `--output` to write somewhere else:

```bash
poetry run python evaluation/mathe/export_benchmark_questions.py \
  --output evaluation/mathe/my_benchmark.xlsx
```

Use `--only-feasible-recommendations` to remove benchmark questions whose
topic/subtopic pool has no eligible document recommendations:

```bash
poetry run python evaluation/mathe/export_benchmark_questions.py \
  --only-feasible-recommendations \
  --output evaluation/mathe/mathe_benchmark_questions_feasible.xlsx
```

### 1a. Click-Based Proxy Relevance

The inspected MathE database does not currently expose question-specific
material interaction logs. It does expose `platform_materials.clicks`, which is a
global click counter per material. We can use this as a weak proxy for historical
material usefulness.

This is not true ground truth. It is proxy ground truth:

```text
Within the same topic/subtopic document pool, materials with more historical
clicks are treated as more useful or more likely to be relevant.
```

Use the separate most-clicked proxy evaluation script to check whether the
recommender returns the most-clicked eligible document from each topic/subtopic
pool in its top-5 and top-k recommendations.

```bash
poetry run python evaluation/mathe/evaluate_most_clicked_proxy.py \
  --output evaluation/mathe/mathe_most_clicked_proxy_evaluation.xlsx
```

### 1b. Keyword-Based Proxy Relevance

The MathE database links both questions and materials to keywords:

```text
platform_keyword_snaquestion
platform_material_keyword
```

We can use those annotations as an offline proxy. This does not measure
human relevance directly. It measures whether the recommender ranks documents
whose keywords match the question keywords higher than documents with weaker or
no keyword overlap.

Run the keyword proxy evaluation with:

```bash
poetry run python evaluation/mathe/evaluate_keyword_proxy.py \
  --output evaluation/mathe/mathe_keyword_proxy_evaluation.xlsx
```
