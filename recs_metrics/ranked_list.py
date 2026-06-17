from __future__ import annotations

import math
from statistics import mean


def weighted_dcg(relevances: list[float]) -> float:
    """Compute discounted cumulative gain for one ordered relevance list."""
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def weighted_ndcg_at_k(
    recommended_ids: list[str],
    relevance_by_item: dict[str, float],
    k: int,
) -> float | None:
    """Compute weighted nDCG@k for one ranked recommendation list."""
    ideal_relevances = sorted(relevance_by_item.values(), reverse=True)[:k]
    ideal_dcg = weighted_dcg(ideal_relevances)
    if ideal_dcg == 0:
        return None

    actual_relevances = [
        relevance_by_item.get(item_id, 0.0)
        for item_id in recommended_ids[:k]
    ]
    return weighted_dcg(actual_relevances) / ideal_dcg


def binary_precision_at_k(
    recommended_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float | None:
    """Compute binary Precision@k for one ranked recommendation list."""
    if not relevant_ids:
        return None

    top_k_ids = recommended_ids[:k]
    if not top_k_ids:
        return None

    return len(set(top_k_ids) & relevant_ids) / len(top_k_ids)


def mean_defined_metric(values: list[float | bool | None]) -> float | None:
    """Average metric values, skipping undefined cases represented by None."""
    defined = [float(value) for value in values if value is not None]
    return mean(defined) if defined else None


__all__ = [
    "binary_precision_at_k",
    "mean_defined_metric",
    "weighted_dcg",
    "weighted_ndcg_at_k",
]
