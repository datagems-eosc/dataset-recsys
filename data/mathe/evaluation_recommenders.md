# MathE Recommender Evaluation Questions

This folder contains a small manually selected MathE question set for comparing
material recommendation strategies.

Input file:

```text
data/mathe/evaluation_questions.json
```

The comparison command can run all internal strategies, or only a selected
subset.

```text
metadata + OCR expansion
popular-seed baseline
question-embedding lookup
open-pool hybrid metadata + OCR + question-embedding reranking
curricular-pool ranker, restricted to same topic/subtopic materials
```

## Prerequisites

Start Redis port-forwarding in another terminal:

```bash
kubectl port-forward svc/<REDIS_SERVICE_NAME> -n <NAMESPACE> 6380:6379
```

## Smoke Test

Start with only a few questions because the question-embedding strategy loads
the embedding model and can be slow on CPU:

```bash
poetry run python -m dataset_recsys.utils.mathe_recsys_compare_cli \
  --questions-file data/mathe/evaluation_questions.json \
  --limit 3 \
  -n 10 \
  --redis-host localhost \
  --redis-port 6380
```

To compare only selected approaches, repeat `--approach`:

```bash
poetry run python -m dataset_recsys.utils.mathe_recsys_compare_cli \
  --questions-file data/mathe/evaluation_questions.json \
  --approach hybrid \
  --approach curricular_pool \
  --limit 3 \
  -n 10 \
  --redis-host localhost \
  --redis-port 6380
```

To validate recommendations for all questions under a topic/subtopic pool, use
the same command-line entry point:

```bash
poetry run python -m dataset_recsys.utils.mathe_recsys_compare_cli \
  --topic-subtopic "Integration" "Triple Integration" \
  --approach curricular_pool \
  -n 20
```

This writes:

```text
outputs/mathe_recsys_comparison.json
outputs/mathe_recsys_comparison.csv
```

The CSV is the easiest file to inspect because it has one row per:

```text
question + recommended material
```

## Full Evaluation

```bash
poetry run python -m dataset_recsys.utils.mathe_recsys_compare_cli \
  --questions-file data/mathe/evaluation_questions.json \
  -n 10 \
  --redis-host localhost \
  --redis-port 6380
```

## CSV Columns

```text
question_id
question_text
question_topic
question_subtopic
material_id
material_title
material_topic
material_subtopic
rank
```
