"""The three MathE ranking configurations and their metrics.

Pure functions over a pool-score snapshot: no database, Qdrant or
embedding-model calls. A pool candidate is a dict with:

- ``material_id``: platform material ID as a string;
- ``keyword_jaccard``: question/material keyword Jaccard in [0, 1];
- ``similarity``: question-to-material cosine as measured (can be negative;
  0.0 when the material has no stored embedding, as in the deployed ranker);
- ``has_embedding``: whether Qdrant returned a vector for the material.

Pools keep the order returned by the deployed scorer (ascending material ID),
and every sort is stable, so exact ties fall back to that order.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from recs_metrics.ranked_list import weighted_ndcg_at_k


Pool = list[dict[str, Any]]


def _clipped_similarity(candidate: dict[str, Any]) -> float:
    """A negative cosine carries no relevance meaning, so it counts as 0."""
    return max(float(candidate["similarity"]), 0.0)


def rank_keyword_only(pool: Pool) -> list[str]:
    """Keyword Jaccard alone. Materials with Jaccard 0 are left out."""
    kept = [c for c in pool if c["keyword_jaccard"] > 0.0]
    ordered = sorted(kept, key=lambda c: c["keyword_jaccard"], reverse=True)
    return [c["material_id"] for c in ordered]


def rank_embedding_only(pool: Pool) -> list[str]:
    """Question-to-material cosine alone. Materials with no positive cosine
    (including those without an embedding) are left out."""
    kept = [c for c in pool if _clipped_similarity(c) > 0.0]
    ordered = sorted(kept, key=lambda c: c["similarity"], reverse=True)
    return [c["material_id"] for c in ordered]


def _min_max(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low <= 0.0:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rank_combined(pool: Pool) -> list[str]:
    """Equal-weight combination of both signals.

    The raw signals live on different scales (coarse Jaccard steps vs a narrow
    cosine band), so each is min-max normalised within the pool first, then
    the two are averaged. Materials with neither signal are left out. Ties
    break on cosine, then Jaccard, then pool order.
    """
    if not pool:
        return []
    jaccards = _min_max([float(c["keyword_jaccard"]) for c in pool])
    similarities = _min_max([_clipped_similarity(c) for c in pool])
    scored = [
        (0.5 * n_jaccard + 0.5 * n_similarity, float(c["similarity"]), c)
        for c, n_jaccard, n_similarity in zip(pool, jaccards, similarities)
        if c["keyword_jaccard"] > 0.0 or _clipped_similarity(c) > 0.0
    ]
    ordered = sorted(
        scored,
        key=lambda item: (item[0], item[1], item[2]["keyword_jaccard"]),
        reverse=True,
    )
    return [item[2]["material_id"] for item in ordered]


CONFIGURATIONS: dict[str, Callable[[Pool], list[str]]] = {
    "keyword_only": rank_keyword_only,
    "embedding_only": rank_embedding_only,
    "combined": rank_combined,
}
REFERENCE_CONFIGURATION = "keyword_only"
RANDOM = "random"


def _first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for rank, material_id in enumerate(ranked, start=1):
        if material_id in relevant:
            return rank
    return None


METRIC_NAMES = (
    "ndcg_at_5",
    "ndcg_at_10",
    "precision_at_5",
    "hit_at_5",
    "hit_at_10",
    "reciprocal_rank",
    "list_length",
)


def list_metrics(ranked: list[str], expert_ids: list[str]) -> dict[str, float]:
    """Metrics for one ranked list against expert IDs (binary relevance).

    nDCG uses every expert ID in the ideal ranking, including IDs outside the
    pool. Precision@5 divides by 5, not by the list length.
    """
    relevant = set(expert_ids)
    relevance = {material_id: 1.0 for material_id in relevant}
    first = _first_relevant_rank(ranked, relevant)
    return {
        "ndcg_at_5": weighted_ndcg_at_k(ranked, relevance, 5) or 0.0,
        "ndcg_at_10": weighted_ndcg_at_k(ranked, relevance, 10) or 0.0,
        "precision_at_5": len(set(ranked[:5]) & relevant) / 5,
        "hit_at_5": float(first is not None and first <= 5),
        "hit_at_10": float(first is not None and first <= 10),
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "list_length": float(len(ranked)),
    }


def expected_random_metrics(
    pool: Pool,
    expert_ids: list[str],
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """Mean metrics over seeded random orderings of the whole pool."""
    material_ids = [c["material_id"] for c in pool]
    rng = random.Random(seed)
    totals = {name: 0.0 for name in METRIC_NAMES}
    for _ in range(permutations):
        shuffled = material_ids[:]
        rng.shuffle(shuffled)
        for name, value in list_metrics(shuffled, expert_ids).items():
            totals[name] += value
    return {name: total / permutations for name, total in totals.items()}


def bootstrap_mean_ci(
    values: list[float],
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for a mean (over questions)."""
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2
    low = means[max(math.floor(alpha * resamples), 0)]
    high = means[min(math.ceil((1.0 - alpha) * resamples) - 1, resamples - 1)]
    return low, high


def wins_ties_losses(differences: list[float]) -> tuple[int, int, int]:
    wins = sum(d > 1e-12 for d in differences)
    losses = sum(d < -1e-12 for d in differences)
    return wins, len(differences) - wins - losses, losses
