"""Pure helpers for the DataFinder dataset-to-dataset benchmark.

Nothing here touches the network, an embedding model or an LLM, so every
function can be tested with small fakes. ``run_benchmark.py`` wires these
helpers to the service's embedding and SCAYLE summary code.

Labels: two corpus datasets are linked when they are cited together in the
``positives`` of the same DataFinder paper (``DataFinder.get_links_from_queries``).
These are co-citation links, a proxy for relevance, not expert judgements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile
from recs_metrics.item_item import recall_at_n, tndcg_at_n
from recs_metrics.ranked_list import weighted_ndcg_at_k

CUTOFFS: tuple[int, ...] = (10, 20, 50)
MAX_CUTOFF = max(CUTOFFS)
LINK_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-20", 6, 20),
    (">20", 21, None),
)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def _clean_str(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def datafinder_row_to_profile(row: Mapping[str, object]) -> DatasetProfile:
    """Map one DataFinder corpus row to the service's DatasetProfile.

    Same mapping as ``notebooks/ds2ds_demo.ipynb``: the paper title is the
    title (dataset ID when missing), the extracted tasks become keywords,
    and headline / field of science are empty because DataFinder has none.
    """
    dataset_id = _clean_str(row["id"])
    title = _clean_str(row.get("title")) or dataset_id
    tasks = row.get("tasks")
    keywords = ", ".join(str(t) for t in tasks if t) if isinstance(tasks, list) else ""
    return DatasetProfile(
        id=dataset_id,
        title=title,
        headline="",
        description=_clean_str(row.get("description")),
        keywords=keywords,
        field_of_science="",
        catalog_summary="",
    )


def build_profiles(corpus: pd.DataFrame) -> list[DatasetProfile]:
    """Build profiles in a deterministic order (ascending dataset ID)."""
    profiles = [datafinder_row_to_profile(row) for row in corpus.to_dict("records")]
    ids = [p.id for p in profiles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset IDs in the corpus.")
    return sorted(profiles, key=lambda p: p.id)


def restrict_links(
    links: Mapping[str, Iterable[str]], corpus_ids: Iterable[str]
) -> dict[str, set[str]]:
    """Keep links whose both ends are in the corpus; drop self-links and empty sets."""
    valid = set(corpus_ids)
    restricted: dict[str, set[str]] = {}
    for query_id, targets in links.items():
        if query_id not in valid:
            continue
        kept = {t for t in targets if t in valid and t != query_id}
        if kept:
            restricted[query_id] = kept
    return restricted


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_query_metrics(
    ranked_ids: Sequence[str],
    relevant: set[str],
    cutoffs: Sequence[int] = CUTOFFS,
) -> dict[str, float]:
    """Metrics for one query dataset. ``relevant`` must be non-empty."""
    if not relevant:
        raise ValueError("per_query_metrics needs at least one relevant dataset.")
    ranked = list(ranked_ids)
    gains = {item: 1.0 for item in relevant}
    row: dict[str, float] = {}
    for k in cutoffs:
        top_k = ranked[:k]
        hits = len(set(top_k) & relevant)
        row[f"recall@{k}"] = recall_at_n({"q": top_k}, {"q": relevant}, n=k)
        row[f"ndcg@{k}"] = float(weighted_ndcg_at_k(top_k, gains, k))
        row[f"tndcg@{k}"] = tndcg_at_n({"q": top_k}, {"q": relevant}, n=k)
        row[f"hit@{k}"] = float(hits > 0)
        row[f"capped_recall@{k}"] = hits / min(len(relevant), k)
    max_k = max(cutoffs)
    first = next((i for i, item in enumerate(ranked[:max_k], 1) if item in relevant), None)
    row[f"rr@{max_k}"] = 1.0 / first if first else 0.0
    return row


def random_rankings(
    ids: Sequence[str],
    query_ids: Iterable[str],
    top_k: int = MAX_CUTOFF,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Uniformly random top-k list per query dataset, excluding the query itself.

    Deterministic for a given seed, ID order and query set: query datasets are
    processed in ascending ID order from one generator.
    """
    pool = list(ids)
    if top_k > len(pool) - 1:
        raise ValueError("top_k must be smaller than the number of datasets.")
    rng = np.random.default_rng(seed)
    rankings: dict[str, list[str]] = {}
    for query_id in sorted(query_ids):
        # Draw one extra so dropping the query itself still leaves top_k items.
        picks = rng.choice(len(pool), size=top_k + 1, replace=False)
        ranked = [pool[i] for i in picks if pool[i] != query_id]
        rankings[query_id] = ranked[:top_k]
    return rankings


def popularity_rankings(
    popularity: Mapping[str, float],
    query_ids: Iterable[str],
    top_k: int = MAX_CUTOFF,
) -> dict[str, list[str]]:
    """The same most-used datasets for every query dataset, excluding the query itself.

    Order: usage count descending, then dataset ID ascending; a missing count
    counts as 0.
    """
    if top_k > len(popularity) - 1:
        raise ValueError("top_k must be smaller than the number of datasets.")

    def count(item: str) -> float:
        value = popularity[item]
        return 0.0 if value is None or value != value else float(value)  # NaN -> 0

    order = sorted(popularity, key=lambda item: (-count(item), item))
    head = order[: top_k + 1]
    return {q: [d for d in head if d != q][:top_k] for q in sorted(query_ids)}


def most_used_datasets(popularity: Mapping[str, float], fraction: float) -> set[str]:
    """The ceil(fraction * N) most-used datasets (usage descending, then ID ascending)."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1).")

    def count(item: str) -> float:
        value = popularity[item]
        return 0.0 if value is None or value != value else float(value)

    n_head = math.ceil(fraction * len(popularity))
    return set(sorted(popularity, key=lambda item: (-count(item), item))[:n_head])


def restrict_to_tail(
    rankings: Mapping[str, Sequence[str]],
    links: Mapping[str, set[str]],
    head: set[str],
    top_k: int = MAX_CUTOFF,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Remove the ``head`` datasets as candidates, linked datasets and query datasets.

    Equivalent to ranking within the long-tail catalog, because removing items
    from a ranking does not change the order of the rest. Query datasets left
    without a link are dropped. Rankings must be deep enough to hold ``top_k``
    tail items after the removal; a shorter list is an error.
    """
    tail_links = {q: {t for t in v if t not in head} for q, v in links.items() if q not in head}
    tail_links = {q: v for q, v in tail_links.items() if v}
    tail_rankings: dict[str, list[str]] = {}
    for query_id in tail_links:
        kept = [item for item in rankings[query_id] if item not in head]
        if len(kept) < top_k:
            raise ValueError(f"Ranking for {query_id} is too short after removing the head.")
        tail_rankings[query_id] = kept[:top_k]
    return tail_rankings, tail_links


def link_bucket(n_links: int) -> str:
    for name, low, high in LINK_BUCKETS:
        if n_links >= low and (high is None or n_links <= high):
            return name
    raise ValueError(f"No bucket for {n_links} links.")


def evaluate_rankings(
    system: str,
    rankings: Mapping[str, Sequence[str]],
    links: Mapping[str, set[str]],
    cutoffs: Sequence[int] = CUTOFFS,
) -> pd.DataFrame:
    """Per-query metrics for every query dataset in ``links``.

    A query dataset without a ranking is an error, not a skipped row, so the
    population is always the full set of linked query datasets.
    """
    missing = sorted(set(links) - set(rankings))
    if missing:
        raise ValueError(f"{system}: no ranking for {len(missing)} query datasets, e.g. {missing[:3]}")
    rows = []
    for query_id in sorted(links):
        ranked = list(rankings[query_id])
        if query_id in ranked:
            raise ValueError(f"{system}: {query_id} is ranked as its own neighbour.")
        relevant = links[query_id]
        rows.append(
            {
                "system": system,
                "query_id": query_id,
                "n_links": len(relevant),
                "link_bucket": link_bucket(len(relevant)),
                **per_query_metrics(ranked, relevant, cutoffs),
            }
        )
    return pd.DataFrame(rows)


def metric_columns(per_query: pd.DataFrame) -> list[str]:
    return [c for c in per_query.columns if "@" in c]


def _group_keys(per_query: pd.DataFrame) -> list[str]:
    return [c for c in ("setting", "system") if c in per_query.columns]


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric per (setting,) system, with the number of query datasets."""
    keys = _group_keys(per_query)
    cols = metric_columns(per_query)
    grouped = per_query.groupby(keys, sort=False)
    summary = grouped[cols].mean()
    summary.insert(0, "n_queries", grouped.size())
    return summary.reset_index()


def summarize_by_bucket(per_query: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    order = {name: i for i, (name, _, _) in enumerate(LINK_BUCKETS)}
    keys = _group_keys(per_query)
    grouped = per_query.groupby([*keys, "link_bucket"], sort=False)
    out = grouped[list(metrics)].mean()
    out.insert(0, "n_queries", grouped.size())
    out = out.reset_index()
    out["_order"] = out["link_bucket"].map(order)
    return out.sort_values([*keys, "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    baseline: str
    candidate: str
    n_queries: int
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int


def paired_bootstrap(
    per_query: pd.DataFrame,
    baseline: str,
    candidate: str,
    metric: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> PairedComparison:
    """Paired percentile bootstrap of mean(candidate - baseline) over query datasets."""
    pivot = per_query.pivot(index="query_id", columns="system", values=metric)
    if pivot[[baseline, candidate]].isna().any().any():
        raise ValueError("Both systems must be scored on the same query datasets.")
    base = pivot[baseline].to_numpy(dtype=float)
    cand = pivot[candidate].to_numpy(dtype=float)
    diff = cand - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    boot = diff[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return PairedComparison(
        metric=metric,
        baseline=baseline,
        candidate=candidate,
        n_queries=len(diff),
        baseline_mean=float(base.mean()),
        candidate_mean=float(cand.mean()),
        mean_difference=float(diff.mean()),
        ci_low=float(low),
        ci_high=float(high),
        wins=int((diff > 0).sum()),
        ties=int((diff == 0).sum()),
        losses=int((diff < 0).sum()),
    )


__all__ = [
    "CUTOFFS",
    "LINK_BUCKETS",
    "MAX_CUTOFF",
    "PairedComparison",
    "build_profiles",
    "datafinder_row_to_profile",
    "evaluate_rankings",
    "link_bucket",
    "metric_columns",
    "most_used_datasets",
    "paired_bootstrap",
    "per_query_metrics",
    "popularity_rankings",
    "random_rankings",
    "restrict_links",
    "restrict_to_tail",
    "summarize",
    "summarize_by_bucket",
]
