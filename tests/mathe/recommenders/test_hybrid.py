from pytest import approx

from dataset_recsys.mathe_recommenders import hybrid

from fakes import FakeEmbeddingClient, FakeRecommendationClient, fake_mathe_client


def test_recommend_hybrid_candidates_merges_metadata_ocr_and_question_sources(
    monkeypatch,
):
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["derivatives"],
    }
    mathe_client.get_document_seed_candidates = lambda question_id: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        }
    ]
    mathe_client.get_document_material_metadata_by_ids = lambda material_ids: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
        {
            "material_id": 101,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["chain rule"],
        },
        {
            "material_id": 200,
            "topic_ids": [99],
            "subtopic_ids": [99],
            "keywords": [],
        },
    ]
    monkeypatch.setattr(
        "dataset_recsys.mathe_recommenders.question_embedding.encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )
    embedding_client = FakeEmbeddingClient(
        [("200", 0.95), ("100", 0.7), ("101", 0.6)]
    )

    candidates = hybrid.recommend_hybrid_candidates(
        question_id=42,
        question="Differentiate y = (2x^3 - 5x)^5.",
        k=3,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient(
            {"100": [("101", 0.8)]}
        ),
        embedding_client=embedding_client,
        metadata_weight=0.6,
        material_ocr_weight=0.25,
        question_weight=0.15,
        question_candidate_limit=2,
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "100",
        "101",
        "200",
    ]
    assert candidates[0]["material_to_material_similarity"] == 0.0
    assert candidates[0]["question_to_material_similarity"] == 0.7
    assert candidates[0]["final_score"] == approx(0.705)
    assert candidates[1]["material_to_material_similarity"] == 0.8
    assert candidates[1]["question_to_material_similarity"] == 0.6
    assert candidates[2]["question_to_material_similarity"] == 0.95


def test_recommend_hybrid_candidates_ranks_metadata_seeds_with_question_similarity(
    monkeypatch,
):
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["derivatives"],
    }
    mathe_client.get_document_seed_candidates = lambda question_id: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
        {
            "material_id": 101,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
    ]
    monkeypatch.setattr(
        "dataset_recsys.mathe_recommenders.question_embedding.encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )
    embedding_client = FakeEmbeddingClient([("100", 0.5), ("101", 0.99)])

    candidates = hybrid.recommend_hybrid_candidates(
        question_id=42,
        question="differentiate x^2",
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient({"100": ["900"]}),
        embedding_client=embedding_client,
        metadata_weight=0.6,
        question_weight=0.15,
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "101",
        "100",
    ]
    assert embedding_client.calls[0]["method"] == "find_similar_by_ids"
    assert candidates[0]["question_to_material_similarity"] == 0.99


def test_recommend_hybrid_candidates_keeps_question_candidate_limit_small(
    monkeypatch,
):
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["derivatives"],
    }
    mathe_client.get_document_seed_candidates = lambda question_id: []
    mathe_client.get_document_material_metadata_by_ids = lambda material_ids: [
        {
            "material_id": 220,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        }
    ]
    monkeypatch.setattr(
        "dataset_recsys.mathe_recommenders.question_embedding.encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )
    embedding_client = FakeEmbeddingClient(
        [("220", 0.91), ("221", 0.88), ("222", 0.85)]
    )

    candidates = hybrid.recommend_hybrid_candidates(
        question_id=42,
        question="differentiate x^2",
        k=10,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient({}),
        embedding_client=embedding_client,
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "220",
        "221",
        "222",
    ]
    assert embedding_client.calls[0]["top_k"] == hybrid.MATHE_HYBRID_QUESTION_CANDIDATES


def test_recommend_hybrid_candidates_handles_materials_without_embeddings(
    monkeypatch,
):
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["derivatives"],
    }
    mathe_client.get_document_seed_candidates = lambda question_id: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        }
    ]
    mathe_client.get_document_material_metadata_by_ids = lambda material_ids: [
        {
            "material_id": 100,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
        {
            "material_id": 101,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
    ]
    monkeypatch.setattr(
        "dataset_recsys.mathe_recommenders.question_embedding.encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )
    embedding_client = FakeEmbeddingClient([("100", 0.9)])

    candidates = hybrid.recommend_hybrid_candidates(
        question_id=42,
        question="differentiate x^2",
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient(
            {"100": [("101", 0.8)]}
        ),
        embedding_client=embedding_client,
        question_candidate_limit=0,
    )

    scores = {
        candidate["material_id"]: candidate["question_to_material_similarity"]
        for candidate in candidates
    }
    assert scores["100"] == 0.9
    assert scores["101"] == 0.0
