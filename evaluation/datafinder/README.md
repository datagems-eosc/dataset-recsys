# DataFinder benchmark of the dataset recommender

One controlled comparison of the dataset-to-dataset (ds2ds) pipeline on the
DataFinder corpus (Viswanathan et al., 2023):

| System | Text embedded | Encoder / ranking |
| --- | --- | --- |
| S1 | `build_raw_embedding_text`: title. description. tasks | `allenai/specter2_base`, mean pooling, 512 tokens, cosine (`rank_similar_entities`) |
| S2 | `build_embedding_text`: `"<title>. <summary>"`, summary from `catalog_summary_v1` on `qwen3.6` (SCAYLE), cleaned with `LightTextPreprocessor` | same |

S2 vs S1 measures whether the summary step improves recommendations. Bedrock
is not used.

Two reference baselines need no model: **Random** (seeded uniform top 50, seed
42) gives chance level, and **Popularity** recommends the most-used datasets
(corpus `popularity`, Papers with Code usage counts) to everyone. Usage counts
come from papers, the same kind of evidence as the links, so Popularity is an
optimistic baseline: it shows how much a score can come from favouring hubs.

## Population and labels

- Candidates: all 5,548 datasets of the filtered corpus (`DataFinder.get()`).
- Query datasets: the 486 datasets with at least one link. None is excluded.
- Labels: `DataFinder.get_links_from_queries()`. Two datasets are linked when
  they appear together in the positives of one DataFinder paper (train and
  test splits); 1,815 undirected pairs. These are co-citation links, a proxy
  for relevance, not expert judgements. An unlinked pair is unknown, not
  irrelevant.
- DataFinder's `title` is the title of the paper that introduced a dataset;
  the dataset ID is not embedded. The raw description is cleaned with
  `LightTextPreprocessor` (as in `notebooks/ds2ds_demo.ipynb`).

## Long-tail settings

Links are concentrated on a few heavily used datasets: removing the 56 most
used (1% of the corpus) removes 53% of the linked pairs, and recommending the
most-used datasets to everyone already reaches Recall@10 0.18. Every system
is therefore also scored within the long-tail catalog:

| Setting | Removed (most used by `popularity`) | Query datasets | Linked pairs |
| --- | --- | --- | --- |
| `full` | none | 486 | 1,815 |
| `tail_excl_top1pct` | 56 (usage >= 498) | 400 | 857 |
| `tail_excl_top5pct` | 278 (usage >= 130) | 203 | 236 |

The removed datasets are dropped as candidates, as linked datasets and as
query datasets. Rankings are taken deep enough and filtered, which equals
ranking within the remaining catalog because cosine order is unchanged by
removing items. No extra embedding or SCAYLE call is needed.

## Metrics

Recall@k, nDCG@k (standard: ideal DCG from min(links, k) hits), HitRate@k,
capped Recall@k (hits / min(links, k)) and reciprocal rank within the top 50,
for k = 10, 20, 50, averaged over the 486 query datasets. `tndcg@k` is the
legacy `recs_metrics.item_item.tndcg_at_n`, kept only to compare with
`data/datafinder/results.csv`; it ignores missed links and is higher than
standard nDCG.

S2 - S1 is reported with a paired percentile bootstrap (10,000 resamples of
query datasets, seed 42) and win/tie/loss counts, overall and by number of
links per query dataset (1, 2-5, 6-20, >20). The BM25 and SPECTER rows of
`results.csv` are quoted as they stand for context; their input text was
lemmatized and stopword-filtered, so they are not a controlled comparison.

## Running it

Run from the repository root on a machine with the SCAYLE VPN, `SCAYLE_*` in
`.env`, and the `allenai/specter2_base` weights available to Hugging Face.

```bash
# Optional trial: 20 summaries, check the rate printed
poetry run python -m evaluation.datafinder.run_benchmark summarize --workers 8 --limit 20

# All summaries (5,548 SCAYLE calls). Resumable: rerun to continue or retry failures.
poetry run python -m evaluation.datafinder.run_benchmark summarize --workers 8

# Embed, rank and score
poetry run python -m evaluation.datafinder.run_benchmark evaluate
```

`evaluate` scores S1 at any time and S2 only when every dataset has a summary.
`--max-missing N` lets S2 run with up to N datasets that never got a summary;
those are embedded from raw metadata and counted in the manifest.

## Outputs (`outputs/`)

| File | Content |
| --- | --- |
| `summaries_scayle_qwen3.6_catalog_summary_v1.jsonl` | Every SCAYLE response (or error), one line per request |
| `metrics.csv` | Mean metrics per system, plus the `results.csv` reference rows |
| `metrics_by_link_count.csv` | Recall@10 and nDCG@10 by links per query dataset |
| `per_query.csv` | All metrics per system and query dataset |
| `comparison.json` | S2 - S1 bootstrap intervals and win/tie/loss counts |
| `manifest.json` | Git commit, raw data hashes, population, models, token lengths |
| `cache/*.npy` | Embedding cache keyed by text content (git-ignored) |

Summaries are sampled at temperature 0.7, so a new `summarize` run gives
slightly different text. The saved JSONL, not the prompt, defines an S2 result.
