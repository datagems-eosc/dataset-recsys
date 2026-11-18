import pytest

from evaluation.mathe.evaluate_ranking_variants import (
    CONTENT_KINDS,
    analyze_snapshot,
    evaluation_status,
    latency_sheets,
    measure_latency,
)
from evaluation.mathe.ranking_variants import (
    bootstrap_mean_ci,
    expected_random_metrics,
    list_metrics,
    rank_combined,
    rank_embedding_only,
    rank_keyword_only,
    wins_ties_losses,
)

DOCUMENTS, VIDEOS = CONTENT_KINDS


def candidate(material_id, jaccard, similarity, has_embedding=True):
    return {
        "material_id": material_id,
        "keyword_jaccard": jaccard,
        "similarity": similarity,
        "has_embedding": has_embedding,
    }


POOL = [
    candidate("1", 0.5, 0.40),
    candidate("2", 0.5, 0.60),
    candidate("3", 0.0, 0.70),
    candidate("4", 0.0, -0.10),
    candidate("5", 0.0, 0.0, has_embedding=False),
    candidate("6", 0.25, 0.30),
]


# --- configurations --------------------------------------------------------


def test_keyword_only_breaks_ties_on_pool_order_and_drops_zero_overlap():
    assert rank_keyword_only(POOL) == ["1", "2", "6"]


def test_embedding_only_ignores_keywords_and_drops_non_positive_cosine():
    assert rank_embedding_only(POOL) == ["3", "2", "1", "6"]


def test_combined_averages_signals_normalised_within_the_pool():
    pool = [candidate("a", 0.5, 0.40), candidate("b", 0.0, 0.55), candidate("c", 0.25, 0.52)]
    # Jaccard -> a 1, b 0, c .5; cosine -> a 0, b 1, c .8; averages .5, .5, .65.
    # a and b tie at .5 and break on cosine.
    assert rank_combined(pool) == ["c", "b", "a"]


def test_combined_drops_materials_without_any_signal():
    assert rank_combined(POOL) == ["2", "1", "3", "6"]
    assert rank_combined([]) == []


def test_combined_handles_a_constant_signal():
    pool = [candidate("1", 0.2, 0.5), candidate("2", 0.2, 0.3)]
    assert rank_combined(pool) == ["1", "2"]


# --- metrics and statistics -------------------------------------------------


def test_list_metrics_precision_divides_by_k_and_counts_outside_pool_ids():
    metrics = list_metrics(["1", "2"], ["2", "99"])
    assert metrics["precision_at_5"] == pytest.approx(1 / 5)
    assert metrics["hit_at_5"] == 1.0
    assert metrics["reciprocal_rank"] == pytest.approx(0.5)
    assert metrics["list_length"] == 2.0
    assert 0.0 < metrics["ndcg_at_5"] < 1.0  # "99" stays in the ideal DCG


def test_list_metrics_empty_ranking_scores_zero():
    metrics = list_metrics([], ["1"])
    assert metrics["ndcg_at_5"] == 0.0
    assert metrics["hit_at_10"] == 0.0


def test_expected_random_metrics_is_deterministic():
    assert expected_random_metrics(POOL, ["1"], 100, seed=7) == expected_random_metrics(
        POOL, ["1"], 100, seed=7
    )
    assert expected_random_metrics(POOL[:2], ["1", "2"], 20, seed=1)["ndcg_at_5"] == pytest.approx(1.0)


def test_bootstrap_ci_and_wins_ties_losses():
    assert bootstrap_mean_ci([0.2] * 5, 100, seed=0) == (pytest.approx(0.2), pytest.approx(0.2))
    assert bootstrap_mean_ci([], 100, seed=0) == (None, None)
    assert wins_ties_losses([0.1, 0.0, -0.2, 0.3]) == (2, 1, 1)


# --- analysis ---------------------------------------------------------------


def snapshot_question(question_id, expert_ids, documents, expert_video_ids=(), videos=()):
    return {
        "question_id": question_id,
        "topic_name": "Topic",
        "subtopic_name": "",
        "question_text": "q",
        "expert_document_ids": list(expert_ids),
        "no_suitable_documents": not expert_ids,
        "expert_video_ids": list(expert_video_ids),
        "no_suitable_videos": not expert_video_ids,
        "documents": list(documents),
        "videos": list(videos),
    }


def test_evaluation_status_is_per_content_kind():
    question = snapshot_question(1, [], [], expert_video_ids=["2"], videos=POOL)
    assert evaluation_status(question, DOCUMENTS) == "no_document_candidates"
    assert evaluation_status(question, VIDEOS) == "evaluated"


def test_analyze_snapshot_reports_three_configurations_and_two_populations():
    snapshot = {
        "schema_version": 1,
        "questions": [
            snapshot_question(10, ["2"], POOL, expert_video_ids=["3"], videos=POOL),
            snapshot_question(11, ["5"], POOL),  # expert item 5 has no embedding
            snapshot_question(12, [], POOL, expert_video_ids=["1"], videos=POOL),
            snapshot_question(13, ["1"], [], expert_video_ids=["2"], videos=POOL),
        ],
    }
    sheets = analyze_snapshot(snapshot)

    results = sheets["doc_results"]
    assert {r["configuration"] for r in results} == {"keyword_only", "embedding_only", "combined", "random"}
    by_population = {(r["population"], r["configuration"]): r for r in results}
    assert by_population[("all_evaluated", "combined")]["questions"] == 2
    assert by_population[("expert_items_embedded", "combined")]["questions"] == 1
    assert {r["configuration"] for r in sheets["doc_vs_keyword_only"]} == {"embedding_only", "combined"}
    assert {r["question_id"] for r in sheets["doc_excluded_questions"]} == {12, 13}
    coverage = {r["question_id"]: r for r in sheets["doc_coverage"]}
    assert coverage[11]["expert_ids_without_embedding"] == "5"
    assert sheets["video_results"][0]["questions"] == 3
    run_info = {r["field"]: r["value"] for r in sheets["run_info"]}
    assert run_info["document_questions_with_expert_items_without_embedding"] == "11"
    assert all(len(name) <= 31 for name in sheets)


def test_analyze_snapshot_skips_a_content_kind_without_labels_and_rejects_bad_input():
    sheets = analyze_snapshot({"schema_version": 1, "questions": [snapshot_question(1, ["2"], POOL)]})
    assert "doc_results" in sheets and not any(name.startswith("video_") for name in sheets)
    with pytest.raises(ValueError):
        analyze_snapshot({"schema_version": 99, "questions": []})
    with pytest.raises(ValueError):
        analyze_snapshot({"schema_version": 1, "questions": [snapshot_question(1, [], POOL)]})


# --- response time ------------------------------------------------------------


class FakeLatencyMathe:
    def get_question_metadata(self, question_id):
        return {"keywords": ["a", "b"]}

    def get_document_materials_for_question_topic_subtopic(self, question_id):
        return [{"material_id": 1, "keywords": ["a"]}, {"material_id": " 2 ", "keywords": []}]

    def get_videos_for_question(self, question_id):
        return [{"material_id": 7, "keywords": ["b"]}]


class FakeLatencyQdrant:
    def __init__(self, payload):
        self.payload = payload

    def get_mathe_question_ranking(self, question_id, content_type):
        return self.payload


def run_latency(kind_name, payload, repeats=3, lookups=None):
    lookups = [] if lookups is None else lookups
    state = {"now": 0.0}

    def clock():
        state["now"] += 1.0
        return state["now"]

    def similarity_lookup(embedding, ids, qdrant, application):
        lookups.append(list(ids))
        return {"1": 0.4}

    return measure_latency(
        question_id=5,
        question_text="q",
        kind_name=kind_name,
        mathe_client=FakeLatencyMathe(),
        qdrant_client=FakeLatencyQdrant(payload),
        encode=lambda text: [0.1],
        similarity_lookup=similarity_lookup,
        keyword_jaccard=lambda q, m: len(set(q) & set(m or [])) / len(set(q) | set(m or [])),
        application="app",
        repeats=repeats,
        question_hash=lambda text: f"hash:{text}",
        clock=clock,
    )


def test_measure_latency_times_each_stage_and_checks_freshness():
    timing = run_latency("document", {"document": ["1"], "question_hash": "hash:q"})
    assert timing["pool_query_seconds"] == 1.0 and timing["precomputed_lookup_seconds"] == 1.0
    assert timing["precomputed_ranking_found"] and timing["precomputed_hash_matches"]
    stale = run_latency("document", {"document": ["1"], "question_hash": "other"})
    assert not stale["precomputed_hash_matches"]
    assert run_latency("document", None, repeats=0) is None


def test_measure_latency_looks_up_video_transcript_ids():
    lookups = []
    run_latency("video", None, repeats=1, lookups=lookups)
    assert lookups == [["7", "7.txt"]]


def test_latency_sheets_compare_live_and_precomputed_paths():
    def timing(stage, precomputed, fresh=True):
        return {
            "pool_query_seconds": stage, "keyword_scoring_seconds": stage,
            "question_embedding_seconds": stage, "vector_lookup_seconds": stage,
            "ranking_seconds": stage, "precomputed_lookup_seconds": precomputed,
            "pool_size": 6, "precomputed_ranking_found": True,
            "precomputed_hash_matches": fresh, "repeats": 3,
        }

    first = snapshot_question(1, ["2"], POOL)
    first["latency"] = {"document": timing(0.1, 0.005), "video": None}
    second = snapshot_question(2, [], POOL)
    second["latency"] = {"document": timing(0.2, 0.010, fresh=False), "video": None}
    summary = {r["measure"]: r for r in latency_sheets([first, second], DOCUMENTS)["latency_summary"]}
    assert summary["live_with_embeddings_total"]["median_ms"] == pytest.approx(750.0)
    assert summary["median_speedup_precomputed_vs_live"]["value"] == pytest.approx(100.0)
    assert summary["share_with_fresh_precomputed_ranking"]["value"] == pytest.approx(0.5)
    assert latency_sheets([first], VIDEOS) is None
