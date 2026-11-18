# MathE Recommender Evaluation

Offline evaluation of the MathE document and video recommenders against expert
reference labels. Documents and videos are evaluated separately:

```text
documents  platform_materials.type = 3, file_ext IN ('pdf', 'docx', 'pptx')
videos     platform_materials.type IN (1, 2), with a link
```

```text
mathe_provider_reference_submission/
  mathe_reference_alltopics.xlsx   expert reference set (the fixed benchmark)
export_benchmark_questions.py      selects benchmark questions from student activity
expert_reference.py                reads the reference workbook, parses ID cells
ranking_variants.py                the three ranking configurations and their metrics
evaluate_ranking_variants.py       the evaluation (fetch, analyze)
outputs/                           generated snapshots and result workbooks (git-ignored)
```

## Benchmark questions

One question per topic/subtopic pool, chosen from historical student attempts
(`assessment`): the question with the most distinct students, then the highest
wrong-answer rate, then the most attempts. The recommenders rank materials
within the question's pool, so each pool is represented by a real, frequently
attempted question.

The selection query is `MatheMirrorClient.get_evaluation_benchmark_questions`.
It joins question keywords before counting attempts, so `total_attempts` is
multiplied by the question's keyword count. This affects only the last
tie-break, and the query is kept as is because it produced the benchmark.

```bash
poetry run python evaluation/mathe/export_benchmark_questions.py \
  [--output PATH] [--only-feasible-recommendations]
```

`--only-feasible-recommendations` drops questions whose pool has no
documents. It checks documents only, so it also drops questions that have
videos but no documents (918 and 992 in the benchmark); the benchmark was
exported without it. A re-export follows current student activity and can
select other questions, so the benchmark is the submitted workbook below, not
a re-export.

## Expert reference set

`mathe_provider_reference_submission/mathe_reference_alltopics.xlsx` is Ana's
15 July workbook, with document and video references. Its document IDs and
no-suitable-document markings match the earlier submission. The
`Reference Set` sheet has one row per benchmark question (30):

```text
related_document_material_ids   document IDs the experts consider relevant
no_suitable_documents           "yes" when no document fits the question
related_video_material_ids      video IDs the experts consider relevant
no_suitable_videos              "yes" when no video fits the question
```

- A blank ID cell means no expert ground truth for that content kind, and the
  question is excluded for it.
- IDs are read as runs of digits, so any separator works. Question 1297's
  `281.282` and `592.595` (stored by Excel as numbers) are read as 281, 282 and
  592, 595.
- Materials the experts did not list count as non-relevant.

Ground truth in the benchmark (30 questions):

| | Questions |
|---|---|
| Document ground truth | 17 |
| Video ground truth | 28 |
| Both | 16 |
| Documents only | 1 (558, marked no suitable videos) |
| Videos only | 12 |
| Neither | 1 (993, marked no suitable documents and videos) |

Of the 13 questions without document IDs, 12 are marked no suitable documents
and one (1167) is blank. Questions 918 and 992 have no documents in their
pool. The number evaluated can be lower than these counts if a pool is empty
when `fetch` runs; `run_info` reports the evaluated counts.

## Evaluation

Three ranking configurations, fixed in advance, are compared for documents and
for videos separately:

```text
keyword_only    keyword Jaccard alone
embedding_only  question-to-material cosine alone
combined        both signals normalised within the pool, then averaged
random          expected score of a random ordering of the pool (context only)
```

"Normalised within the pool" means that, for each question, each signal is
rescaled over that question's candidate materials so the best material gets 1
and the worst 0. Without it, keyword overlap (which moves in steps of 0.25-0.5)
would swamp cosine (which varies by about 0.1 within a pool). A configuration
leaves out materials with no positive signal it uses.

```bash
# Reads the deployment's MathE mirror and Qdrant (VPN / port-forwards).
poetry run python -m evaluation.mathe.evaluate_ranking_variants fetch \
  --environment-label "dev via VPN" \
  --snapshot evaluation/mathe/outputs/mathe_pool_scores_snapshot_dev.json

# Offline: reads only the snapshot.
poetry run python -m evaluation.mathe.evaluate_ranking_variants analyze \
  --snapshot evaluation/mathe/outputs/mathe_pool_scores_snapshot_dev.json \
  --output evaluation/mathe/outputs/mathe_ranking_evaluation_dev.xlsx
```

`fetch` scores every document and video in each reference question's pool
with the deployed scoring code: it embeds the database question text with
`BAAI/bge-m3` and compares it with the material embeddings in Qdrant. Per
material it keeps only the raw signals (`keyword_jaccard`; `similarity`, the
cosine as measured, 0.0 without an embedding; `has_embedding`). Nothing is
written to the services, and neither command overwrites a file without
`--overwrite`.

### Results and how to read them

- **Metrics:** nDCG@5 and hit@5 (at least one expert item in the top 5), with
  nDCG@10, precision@5 and MRR alongside. Relevance is binary: expert IDs are
  relevant, every other pool material is not. Ideal DCG uses all expert IDs.
- **Uncertainty:** 95% bootstrap CIs over questions; each configuration is
  also compared with `keyword_only` question by question (mean difference,
  CI, wins/ties/losses).
- **Populations:** `all_evaluated` (every question with expert IDs and a pool
  of that kind) and `expert_items_embedded` (the same without questions where
  an expert item has no embedding: the embedding signal cannot rank such an
  item, which biases the comparison against it).

Sheets, prefixed `doc_` or `video_`, plus a shared `run_info`:

```text
results              means and CIs per configuration, for both populations
vs_keyword_only      paired differences against keyword_only
per_question         every configuration's metrics and top-10 list
coverage             pool size, materials and expert items without embedding
excluded_questions   questions not evaluated, with the reason
```

### Response time (optional)

`fetch --timing-repeats 5` also times each stage of serving a ranking with the
deployed code, keeping the median of 5 runs per question, and adds
`latency_summary` and `latency_per_question` sheets. They compare live scoring
with embeddings (the API's fallback), live keyword-only scoring, and reading
the precomputed ranking (the API's normal path), with the speedup and the
share of questions whose stored ranking matches the question text. Timings
include the network round trips from the machine that ran `fetch`; they
exclude HTTP, authentication and model loading.
