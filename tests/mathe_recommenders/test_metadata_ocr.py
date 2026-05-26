from dataset_recsys.mathe_recommenders.metadata_ocr import (
    rank_expanded_candidates,
    recommend_from_metadata_seeds,
    recommend_pdf_seeds_for_question,
)

from fakes import FakeRecommendationClient, fake_mathe_client


def test_recommend_pdf_seeds_for_question_scores_and_limits_candidates():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["algebra"],
    }
    mathe_client.get_pdf_seed_candidates = lambda question_id: [
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

    recommendations = recommend_pdf_seeds_for_question(
        42,
        k=1,
        mathe_mirror_client=mathe_client,
    )

    assert len(recommendations) == 1
    assert recommendations[0]["material_id"] == 2
    assert recommendations[0]["metadata_score"] == 2 / 3


def test_recommend_pdf_seeds_for_question_returns_empty_when_question_missing():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: None
    mathe_client.get_pdf_seed_candidates = lambda question_id: [
        {"material_id": 1, "topic_ids": [10], "subtopic_ids": [20], "keywords": []}
    ]

    assert recommend_pdf_seeds_for_question(404, k=10, mathe_mirror_client=mathe_client) == []
    assert recommend_pdf_seeds_for_question(404, k=0, mathe_mirror_client=mathe_client) == []


def test_rank_expanded_candidates_uses_seed_and_redis_similarity_scores():
    seeds = [{"material_id": 6, "material_redis_id": "6.pdf", "metadata_score": 1.0}]
    recommendation_client = FakeRecommendationClient(
        {"6.pdf": [("10.pdf", 0.42), ("11.pdf", 0.91), ("12.pdf", 0.5)]}
    )

    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=3,
        recommendation_client=recommendation_client,
        material_similarity_weight=1.0,
        neighbors_per_seed=3,
    )

    assert [candidate["material_redis_id"] for candidate in candidates] == [
        "6.pdf",
        "11.pdf",
        "12.pdf",
    ]


def test_rank_expanded_candidates_scores_neighbor_metadata():
    mathe_client = fake_mathe_client()
    question_metadata = {
        "question_id": 42,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["chain rule"],
    }
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

    candidates = rank_expanded_candidates(
        seeds=[{"material_id": 818, "material_redis_id": "818.pdf", "metadata_score": 1.0}],
        k=2,
        recommendation_client=FakeRecommendationClient({"818.pdf": ["900.pdf"]}),
        question_metadata=question_metadata,
        mathe_mirror_client=mathe_client,
    )

    neighbor = next(
        candidate
        for candidate in candidates
        if candidate["material_redis_id"] == "900.pdf"
    )
    assert neighbor["same_topic"] == 1
    assert neighbor["same_subtopic"] == 1
    assert neighbor["metadata_score"] == 1.0


def test_recommend_from_metadata_seeds_returns_db_material_ids_after_redis_expansion():
    mathe_client = fake_mathe_client()
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

    recommendations = recommend_from_metadata_seeds(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient(
            {"derivatives_intro.pdf": ["chain_rule.pdf"]}
        ),
    )

    assert recommendations == ["818", "900"]
