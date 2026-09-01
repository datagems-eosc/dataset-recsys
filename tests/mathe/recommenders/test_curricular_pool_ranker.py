from pytest import approx

from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    rank_curricular_pool_candidates,
    recommend_from_curricular_pool,
)

from fakes import FakeEmbeddingClient, fake_mathe_client


def test_rank_curricular_pool_candidates_scores_only_the_same_pool():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["product rule"],
    }
    mathe_client.get_document_materials_for_question_topic_subtopic = lambda question_id: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["product rule"],
        },
        {
            "material_id": 101,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": [],
        },
    ]
    embedding_client = FakeEmbeddingClient(
        [("100", 0.2), ("101", 0.9), ("900", 1.0)]
    )

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate x sin(x).",
        k=2,
        mathe_mirror_client=mathe_client,
        embedding_client=embedding_client,
        keyword_weight=0.4,
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "101",
        "100",
    ]
    assert candidates[0]["final_score"] == approx(0.54)
    assert candidates[1]["final_score"] == approx(0.52)
    assert embedding_client.calls == [
        {
            "method": "find_similar_by_ids",
            "application": "mathe_documents",
            "query_embedding": [0.1, 0.2],
            "entity_ids": ["100", "101"],
            "table": embedding_client.TABLE_MATHE,
        }
    ]


def test_rank_curricular_pool_candidates_keeps_missing_embeddings_at_zero():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["chain rule"],
    }
    mathe_client.get_document_materials_for_question_topic_subtopic = lambda question_id: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
        {
            "material_id": 101,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["product rule"],
        },
    ]
    embedding_client = FakeEmbeddingClient([("101", 0.8)])

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate a composition.",
        k=2,
        mathe_mirror_client=mathe_client,
        embedding_client=embedding_client,
        question_embedding=[0.1, 0.2],
    )

    scores = {
        candidate["material_id"]: candidate["question_to_material_similarity"]
        for candidate in candidates
    }
    assert scores["100"] == 0.0
    assert scores["101"] == 0.8


def test_recommend_from_curricular_pool_returns_db_material_ids():
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": [],
    }
    mathe_client.get_document_materials_for_question_topic_subtopic = lambda question_id: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": [],
        }
    ]

    recommendations = recommend_from_curricular_pool(
        question_id=42,
        question="Differentiate x.",
        k=1,
        mathe_mirror_client=mathe_client,
        embedding_client=FakeEmbeddingClient([("100", 0.9)]),
        question_embedding=[0.1, 0.2],
    )

    assert recommendations == ["100"]
