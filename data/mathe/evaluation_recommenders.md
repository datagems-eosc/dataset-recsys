# MathE Recommender Evaluation Questions

This folder contains a small manually selected MathE question set for comparing
material recommendation strategies.

Input file:

```text
data/mathe/evaluation_questions.json
```

The comparison command runs the three internal strategies:

```text
metadata + OCR expansion
popular-seed baseline
question-embedding lookup
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
poetry run python -m dataset_recsys.mathe_recommenders.compare_cli \
  --questions-file data/mathe/evaluation_questions.json \
  --limit 3 \
  -n 10 \
  --redis-host localhost \
  --redis-port 6380
```

This writes:

```text
temp/mathe_recommender_comparison.json
temp/mathe_recommender_comparison.csv
```

The CSV is the easiest file to inspect because it has one row per:

```text
question + strategy + recommended material
```

## Full Evaluation

```bash
poetry run python -m dataset_recsys.mathe_recommenders.compare_cli \
  --questions-file data/mathe/evaluation_questions.json \
  -n 10 \
  --redis-host localhost \
  --redis-port 6380
```

## CSV Columns

```text
strategy
question_id
question
material_id
title
metadata_score
rank
```
