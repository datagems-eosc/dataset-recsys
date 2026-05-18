from dataset_recsys.mathe_recommenders.seed_scoring import (
    compute_keyword_jaccard,
    score_pdf_seed_candidates,
)
from dataset_recsys.mathe_recommenders.metadata_ocr import (
    rank_expanded_candidates,
    recommend_pdf_seeds_for_question,
    recommend_from_metadata_seeds,
)
from dataset_recsys.mathe_recommenders.popular_seed import (
    recommend_from_popular_seed,
)
from dataset_recsys.storage.mathe_mirror_client import (
    MatheMirrorClient,
    material_id_to_redis_id,
    redis_id_to_material_id,
)


def test_compute_keyword_jaccard_treats_none_as_empty():
    assert compute_keyword_jaccard(None, None) == 0.0
    assert compute_keyword_jaccard(["algebra"], None) == 0.0


def test_compute_keyword_jaccard_uses_sets():
    assert compute_keyword_jaccard(
        ["algebra", "matrix", "matrix"],
        ["matrix", "calculus"],
    ) == 1 / 3


def test_mathe_material_redis_id_maps_to_db_material_id():
    assert material_id_to_redis_id(221) == "221.pdf"
    assert material_id_to_redis_id("221.pdf") == "221.pdf"
    assert redis_id_to_material_id("221.pdf") == 221
    assert redis_id_to_material_id("ChainRule.pdf") is None


def test_score_pdf_seed_candidates_enriches_and_sorts_by_metadata_score():
    question_metadata = {
        "question_id": 42,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["algebra", "matrix"],
    }
    seed_candidates = [
        {
            "material_id": 1,
            "topic_ids": [10],
            "subtopic_ids": [99],
            "keywords": ["geometry"],
        },
        {
            "material_id": 2,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["algebra", "matrix"],
        },
        {
            "material_id": 3,
            "topic_ids": [99],
            "subtopic_ids": [99],
            "keywords": ["matrix", "calculus"],
        },
    ]

    scored = score_pdf_seed_candidates(question_metadata, seed_candidates)

    assert [candidate["material_id"] for candidate in scored] == [2, 1, 3]
    assert scored[0]["keyword_jaccard"] == 1.0
    assert scored[0]["same_subtopic"] == 1
    assert scored[0]["same_topic"] == 1
    assert scored[0]["metadata_score"] == 1.0
    assert scored[1]["metadata_score"] == 1 / 3
    assert scored[2]["metadata_score"] == 1 / 9


def test_score_pdf_seed_candidates_matches_plural_topic_and_subtopic_ids():
    question_metadata = {
        "question_id": 42,
        "topic_id": 2,
        "subtopic_id": 3,
        "keywords": ["Derivatives"],
    }
    seed_candidates = [
        {
            "material_id": 818,
            "topic_ids": [1, 2],
            "subtopic_ids": [1, 3],
            "keywords": ["Derivatives", "Partial Differentiation"],
        }
    ]

    scored = score_pdf_seed_candidates(question_metadata, seed_candidates)

    assert scored[0]["same_topic"] == 1
    assert scored[0]["same_subtopic"] == 1
    assert scored[0]["metadata_score"] == (0.5 + 1 + 1) / 3


def test_recommend_pdf_seeds_for_question_scores_and_limits_candidates():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["algebra"],
    }
    client.get_pdf_seed_candidates = lambda question_id: [
        {
            "material_id": 1,
            "topic_ids": [99],
            "subtopic_ids": [99],
            "keywords": ["algebra"],
        },
        {
            "material_id": 2,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": [],
        },
    ]

    recommendations = recommend_pdf_seeds_for_question(42, k=1, mathe_mirror_client=client)

    assert len(recommendations) == 1
    assert recommendations[0]["material_id"] == 2
    assert recommendations[0]["metadata_score"] == 2 / 3


def test_recommend_pdf_seeds_for_question_returns_empty_when_question_missing():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.get_question_metadata = lambda question_id: None
    client.get_pdf_seed_candidates = lambda question_id: [
        {"material_id": 1, "topic_ids": [10], "subtopic_ids": [20], "keywords": []}
    ]

    assert recommend_pdf_seeds_for_question(404, k=10, mathe_mirror_client=client) == []
    assert recommend_pdf_seeds_for_question(404, k=0, mathe_mirror_client=client) == []


class FakeRecommendationClient:
    def __init__(self, recommendations):
        self.recommendations = recommendations

    def get_recommendations(self, application, entity_id, limit=None):
        recommendations = self.recommendations.get(entity_id, [])
        if limit is None:
            return recommendations
        return recommendations[:limit]

    def get_recommendations_with_scores(self, application, entity_id, limit=None):
        recommendations = self.get_recommendations(application, entity_id, limit)
        return [
            item if isinstance(item, tuple) else (item, float(len(recommendations) - rank))
            for rank, item in enumerate(recommendations)
        ]


def test_rank_expanded_candidates_returns_metadata_seeds_when_enough_exist():
    seeds = [
        {"material_id": 6, "material_redis_id": "6.pdf", "metadata_score": 1.0},
        {"material_id": 7, "material_redis_id": "7.pdf", "metadata_score": 0.8},
        {"material_id": 8, "material_redis_id": "8.pdf", "metadata_score": 0.3},
    ]

    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=2,
        recommendation_client=FakeRecommendationClient({}),
    )

    assert [candidate["material_redis_id"] for candidate in candidates] == ["6.pdf", "7.pdf"]


def test_rank_expanded_candidates_uses_material_redis_id_when_present():
    seeds = [
        {
            "material_id": 818,
            "material_redis_id": "derivatives_intro.pdf",
            "metadata_score": 1.0,
        }
    ]
    recommendation_client = FakeRecommendationClient(
        {"derivatives_intro.pdf": ["chain_rule.pdf"]}
    )

    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=2,
        recommendation_client=recommendation_client,
        neighbors_per_seed=1,
    )

    assert [candidate["material_redis_id"] for candidate in candidates] == [
        "derivatives_intro.pdf",
        "chain_rule.pdf",
    ]


def test_recommend_from_metadata_seeds_returns_db_material_ids_after_redis_expansion():
    mathe_client = MatheMirrorClient.__new__(MatheMirrorClient)
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["derivatives"],
    }
    mathe_client.get_pdf_seed_candidates = lambda question_id: [
        {
            "material_id": 818,
            "material_redis_id": "derivatives_intro.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        }
    ]
    mathe_client.get_pdf_material_metadata_by_redis_ids = lambda material_ids: [
        {
            "material_id": 818,
            "material_redis_id": "derivatives_intro.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
        {
            "material_id": 900,
            "material_redis_id": "chain_rule.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
    ]
    mathe_client.get_pdf_material_details = lambda material_ids: [
        {"material_id": 818, "material_redis_id": "derivatives_intro.pdf"},
        {"material_id": 900, "material_redis_id": "chain_rule.pdf"},
    ]
    recommendation_client = FakeRecommendationClient(
        {"derivatives_intro.pdf": ["chain_rule.pdf"]}
    )

    recommendations = recommend_from_metadata_seeds(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=recommendation_client,
    )

    assert recommendations == ["818", "900"]


def test_rank_expanded_candidates_scores_neighbor_metadata():
    question_metadata = {
        "question_id": 42,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["chain rule"],
    }
    seeds = [
        {
            "material_id": 818,
            "material_redis_id": "818.pdf",
            "metadata_score": 1.0,
        }
    ]
    mathe_client = MatheMirrorClient.__new__(MatheMirrorClient)
    mathe_client.get_pdf_material_metadata_by_redis_ids = lambda material_ids: [
        {
            "material_id": 818,
            "material_redis_id": "818.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
        {
            "material_id": 900,
            "material_redis_id": "900.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
    ]
    recommendation_client = FakeRecommendationClient(
        {"818.pdf": ["900.pdf"]}
    )

    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=2,
        recommendation_client=recommendation_client,
        question_metadata=question_metadata,
        mathe_mirror_client=mathe_client,
    )

    neighbor = next(
        candidate for candidate in candidates
        if candidate["material_redis_id"] == "900.pdf"
    )
    assert neighbor["same_topic"] == 1
    assert neighbor["same_subtopic"] == 1
    assert neighbor["metadata_score"] == 1.0


def test_rank_expanded_candidates_uses_stored_redis_similarity_scores():
    seeds = [{"material_id": 6, "material_redis_id": "6.pdf", "metadata_score": 1.0}]
    recommendation_client = FakeRecommendationClient(
        {"6.pdf": [("10.pdf", 0.42), ("11.pdf", 0.91), ("12.pdf", 0.5)]}
    )

    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=3,
        recommendation_client=recommendation_client,
        embedding_weight=1.0,
        neighbors_per_seed=3,
    )

    assert [candidate["material_redis_id"] for candidate in candidates] == [
        "6.pdf",
        "11.pdf",
        "12.pdf",
    ]


def test_recommend_from_popular_seed_uses_previous_most_clicked_seed_flow():
    mathe_client = MatheMirrorClient.__new__(MatheMirrorClient)
    mathe_client.get_material_by_question_id = lambda question_id: {
        "id": 6,
        "material_redis_id": "popular_derivatives.pdf",
    }
    recommendation_client = FakeRecommendationClient(
        {"popular_derivatives.pdf": ["10.pdf", "11.pdf", "12.pdf"]}
    )

    recommendations = recommend_from_popular_seed(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=recommendation_client,
    )

    assert recommendations == ["10.pdf", "11.pdf"]


def test_recommend_from_popular_seed_returns_empty_when_no_seed_exists():
    mathe_client = MatheMirrorClient.__new__(MatheMirrorClient)
    mathe_client.get_material_by_question_id = lambda question_id: None

    recommendations = recommend_from_popular_seed(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient({}),
    )

    assert recommendations == []
