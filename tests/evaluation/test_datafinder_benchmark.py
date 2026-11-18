import json
import math

import pandas as pd
import pytest

from evaluation.datafinder.benchmark import (
    build_profiles,
    evaluate_rankings,
    link_bucket,
    most_used_datasets,
    paired_bootstrap,
    per_query_metrics,
    popularity_rankings,
    random_rankings,
    restrict_links,
    restrict_to_tail,
    summarize,
)
from evaluation.datafinder import run_benchmark
from evaluation.datafinder.run_benchmark import read_summaries


def test_per_query_metrics_standard_vs_legacy_ndcg():
    # Two linked datasets, only one retrieved, at rank 1.
    row = per_query_metrics(["a", "x", "y"], {"a", "b"}, cutoffs=(10,))

    assert row["recall@10"] == 0.5
    assert row["hit@10"] == 1.0
    assert row["capped_recall@10"] == 0.5
    assert row["rr@10"] == 1.0
    # Legacy tndcg ignores the missed link; standard nDCG does not.
    assert row["tndcg@10"] == 1.0
    assert row["ndcg@10"] == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_per_query_metrics_no_hit_and_cutoff():
    row = per_query_metrics(["x", "y", "a"], {"a"}, cutoffs=(2, 3))

    assert row["recall@2"] == 0.0 and row["ndcg@2"] == 0.0 and row["tndcg@2"] == 0.0
    assert row["recall@3"] == 1.0
    assert row["rr@3"] == pytest.approx(1 / 3)


def test_capped_recall_for_hub_with_more_links_than_k():
    relevant = {f"r{i}" for i in range(5)}
    row = per_query_metrics(["r0", "r1"], relevant, cutoffs=(2,))

    assert row["recall@2"] == pytest.approx(2 / 5)
    assert row["capped_recall@2"] == 1.0


def test_per_query_metrics_rejects_empty_labels():
    with pytest.raises(ValueError):
        per_query_metrics(["a"], set())


def test_restrict_links_drops_outside_corpus_self_links_and_empty_sets():
    links = {"a": {"b", "z", "a"}, "c": {"z"}, "z": {"a"}}

    assert restrict_links(links, ["a", "b", "c"]) == {"a": {"b"}}


def test_build_profiles_maps_fields_and_sorts_by_id():
    corpus = pd.DataFrame(
        [
            {"id": "MNIST", "title": "Gradient-based learning", "description": "Digits.",
             "tasks": ["Image Classification", "Clustering"]},
            {"id": "CIFAR-10", "title": pd.NA, "description": "Tiny images.", "tasks": pd.NA},
        ]
    )
    profiles = build_profiles(corpus)

    assert [p.id for p in profiles] == ["CIFAR-10", "MNIST"]
    assert profiles[0].title == "CIFAR-10"  # falls back to the ID
    assert profiles[0].keywords == ""
    assert profiles[1].keywords == "Image Classification, Clustering"
    assert profiles[1].headline == "" and profiles[1].catalog_summary == ""


def test_build_profiles_rejects_duplicate_ids():
    corpus = pd.DataFrame([{"id": "A", "title": "t", "description": "d", "tasks": ["x"]}] * 2)
    with pytest.raises(ValueError):
        build_profiles(corpus)


def test_evaluate_rankings_requires_every_query_and_no_self_match():
    links = {"a": {"b"}, "b": {"a"}}

    with pytest.raises(ValueError, match="no ranking"):
        evaluate_rankings("S", {"a": ["b"]}, links)
    with pytest.raises(ValueError, match="own neighbour"):
        evaluate_rankings("S", {"a": ["a", "b"], "b": ["a"]}, links)


def test_evaluate_and_summarize_count_every_query():
    links = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    rankings = {"a": ["b", "c"], "b": ["c", "a"], "c": ["a", "b"]}
    per_query = evaluate_rankings("S", rankings, links, cutoffs=(1, 2))
    summary = summarize(per_query)

    assert summary.loc[0, "n_queries"] == 3
    assert summary.loc[0, "recall@1"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert list(per_query["link_bucket"]) == ["1", "2-5", "1"]


def test_link_bucket_edges():
    assert [link_bucket(n) for n in (1, 2, 5, 6, 20, 21, 139)] == ["1", "2-5", "2-5", "6-20", "6-20", ">20", ">20"]


def test_paired_bootstrap_identical_systems_and_counts():
    per_query = pd.DataFrame(
        {"system": ["S1", "S1", "S2", "S2"], "query_id": ["a", "b", "a", "b"],
         "ndcg@10": [0.2, 0.5, 0.2, 0.9]}
    )
    same = paired_bootstrap(per_query[per_query.system == "S1"].assign(system="S1")
                            .pipe(lambda d: pd.concat([d, d.assign(system="S2")])), "S1", "S2", "ndcg@10")
    assert same.mean_difference == 0 and same.ci_low == 0 and same.ci_high == 0

    result = paired_bootstrap(per_query, "S1", "S2", "ndcg@10", n_resamples=200)
    assert result.mean_difference == pytest.approx(0.2)
    assert (result.wins, result.ties, result.losses) == (1, 1, 0)
    assert result.ci_low <= result.mean_difference <= result.ci_high


def test_read_summaries_keeps_success_over_later_failure(tmp_path):
    path = tmp_path / "s.jsonl"
    lines = [
        {"id": "a", "summary": "", "error": "timeout"},
        {"id": "a", "summary": "A summary.", "error": ""},
        {"id": "a", "summary": "", "error": "timeout"},
        {"id": "b", "summary": "", "error": "empty response"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    records = read_summaries(path)
    assert records["a"]["summary"] == "A summary."
    assert records["b"]["summary"] == ""
    assert read_summaries(tmp_path / "missing.jsonl") == {}


def test_random_rankings_are_deterministic_unique_and_exclude_self():
    ids = [f"d{i}" for i in range(30)]
    first = random_rankings(ids, ["d0", "d5"], top_k=29, seed=42)

    assert first == random_rankings(ids, ["d5", "d0"], top_k=29, seed=42)
    for query_id, ranked in first.items():
        assert query_id not in ranked
        assert len(ranked) == len(set(ranked)) == 29
    assert first != random_rankings(ids, ["d0", "d5"], top_k=29, seed=7)
    with pytest.raises(ValueError):
        random_rankings(ids, ["d0"], top_k=30)


def test_popularity_rankings_order_ties_missing_and_self_exclusion():
    popularity = {"a": 5, "b": 9, "c": 5, "d": float("nan"), "e": 0}
    ranked = popularity_rankings(popularity, ["b", "c"], top_k=3)

    assert ranked["c"] == ["b", "a", "d"]  # ties by ID; c excluded
    assert ranked["b"] == ["a", "c", "d"]  # NaN counts as 0, before "e" by ID
    with pytest.raises(ValueError):
        popularity_rankings(popularity, ["a"], top_k=5)


def test_most_used_datasets_rounds_up_and_breaks_ties_by_id():
    popularity = {"a": 5, "b": 9, "c": 5, "d": 1}

    assert most_used_datasets(popularity, 0.0) == set()
    assert most_used_datasets(popularity, 0.25) == {"b"}
    assert most_used_datasets(popularity, 0.3) == {"b", "a"}  # ceil(1.2) = 2, a before c


def test_restrict_to_tail_removes_head_everywhere():
    links = {"q": {"hub", "t1"}, "hub": {"q"}, "r": {"hub"}}
    rankings = {"q": ["hub", "x", "t1", "y"], "hub": ["q"], "r": ["hub", "x", "y"]}
    tail_rankings, tail_links = restrict_to_tail(rankings, links, {"hub"}, top_k=2)

    assert tail_links == {"q": {"t1"}}  # hub query dropped; r left without links
    assert tail_rankings == {"q": ["x", "t1"]}
    with pytest.raises(ValueError, match="too short"):
        restrict_to_tail(rankings, links, {"hub"}, top_k=4)


def test_summarize_groups_by_setting_when_present():
    per_query = pd.DataFrame(
        {"setting": ["full", "full", "tail"], "system": ["S", "S", "S"],
         "query_id": ["a", "b", "a"], "recall@10": [1.0, 0.0, 0.5]}
    )
    summary = summarize(per_query)

    assert summary[["setting", "n_queries", "recall@10"]].values.tolist() == [["full", 2, 0.5], ["tail", 1, 0.5]]


def test_no_tasks_variant_drops_keywords_only(monkeypatch):
    corpus = pd.DataFrame(
        [{"id": "MNIST", "title": "Gradient-based learning", "description": "**Digits**.",
          "tasks": ["Image Classification"]}]
    )

    class FakeDataFinder:
        dataset_folder = "unused"

        def get(self):
            return {"corpus": corpus}

        def get_links_from_queries(self):
            return {}

    monkeypatch.setattr(run_benchmark, "DataFinder", FakeDataFinder)
    with_tasks, _, _ = run_benchmark.load_benchmark("with_tasks")
    no_tasks, _, _ = run_benchmark.load_benchmark("no_tasks")

    assert with_tasks[0].keywords == "Image Classification"
    assert no_tasks[0].keywords == ""
    assert no_tasks[0].description == with_tasks[0].description == "Digits."
    with pytest.raises(ValueError):
        run_benchmark.load_benchmark("other")
