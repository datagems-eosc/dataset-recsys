from recs_metrics.ranked_list import (
    binary_precision_at_k,
    mean_defined_metric,
    weighted_ndcg_at_k,
)


def test_binary_precision_at_k_for_one_recommendation_list():
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d", "x"}

    assert binary_precision_at_k(recommended, relevant, 2) == 1 / 2
    assert binary_precision_at_k(recommended, relevant, 4) == 1 / 2


def test_ranked_list_metrics_are_undefined_without_relevant_items():
    recommended = ["a", "b"]

    assert binary_precision_at_k(recommended, set(), 2) is None
    assert weighted_ndcg_at_k(recommended, {"a": 0.0, "b": 0.0}, 2) is None


def test_weighted_ndcg_uses_weighted_relevance():
    relevance_by_item = {
        "a": 0.5,
        "b": 1.0,
        "c": 0.0,
    }

    assert weighted_ndcg_at_k(["b", "a", "c"], relevance_by_item, 3) == 1.0
    assert weighted_ndcg_at_k(["c", "a", "b"], relevance_by_item, 3) < 1.0


def test_mean_defined_metric_skips_none_and_accepts_bools():
    assert mean_defined_metric([1.0, None, 0.0, True]) == 2 / 3
    assert mean_defined_metric([None]) is None
