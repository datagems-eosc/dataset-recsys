from dataset_recsys.mathe_recommenders.metadata_ocr import (
    rank_expanded_candidates,
    recommend_from_metadata_seeds,
    recommend_document_seeds_for_question,
)

from fakes import FakeRecommendationClient, fake_mathe_client


def test_recommend_document_seeds_for_question_scores_and_limits_candidates():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["algebra"],
    }
    mathe_client.get_document_seed_candidates = lambda question_id: [
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

    recommendations = recommend_document_seeds_for_question(
        42,
        k=1,
        mathe_mirror_client=mathe_client,
    )

    assert len(recommendations) == 1
    assert recommendations[0]["material_id"] == 2
    assert recommendations[0]["metadata_score"] == 2 / 3


def test_recommend_document_seeds_for_question_returns_empty_when_question_missing():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: None
    mathe_client.get_document_seed_candidates = lambda question_id: [
        {"material_id": 1, "topic_ids": [10], "subtopic_ids": [20], "keywords": []}
    ]

    assert recommend_document_seeds_for_question(404, k=10, mathe_mirror_client=mathe_client) == []
    assert recommend_document_seeds_for_question(404, k=0, mathe_mirror_client=mathe_client) == []


def test_rank_expanded_candidates_uses_seed_and_document_similarity_scores():
    seeds = [{"material_id": 6, "metadata_score": 1.0}]
    recommendation_client = FakeRecommendationClient(
        {"6": [("10", 0.42), ("11", 0.91), ("12", 0.5)]}
    )

    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=3,
        recommendation_client=recommendation_client,
        material_similarity_weight=1.0,
        neighbors_per_seed=3,
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "6",
        "11",
        "12",
    ]


def test_rank_expanded_candidates_scores_neighbor_metadata():
    mathe_client = fake_mathe_client()
    question_metadata = {
        "question_id": 42,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["chain rule"],
    }
    mathe_client.get_document_material_metadata_by_ids = lambda material_ids: [
        {
            "material_id": 818,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
        {
            "material_id": 900,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
    ]

    candidates = rank_expanded_candidates(
        seeds=[{"material_id": 818, "metadata_score": 1.0}],
        k=2,
        recommendation_client=FakeRecommendationClient({"818": ["900"]}),
        question_metadata=question_metadata,
        mathe_mirror_client=mathe_client,
    )

    neighbor = next(
        candidate
        for candidate in candidates
        if candidate["material_id"] == "900"
    )
    assert neighbor["same_topic"] == 1
    assert neighbor["same_subtopic"] == 1
    assert neighbor["metadata_score"] == 1.0


def test_recommend_from_metadata_seeds_returns_platform_material_ids():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["derivatives"],
    }
    mathe_client.get_document_seed_candidates = lambda question_id: [
        {
            "material_id": 818,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        }
    ]
    mathe_client.get_document_material_metadata_by_ids = lambda material_ids: [
        {
            "material_id": 818,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
        {
            "material_id": 900,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
    ]
    recommendations = recommend_from_metadata_seeds(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient(
            {"818": ["900"]}
        ),
    )

    assert recommendations == ["818", "900"]
