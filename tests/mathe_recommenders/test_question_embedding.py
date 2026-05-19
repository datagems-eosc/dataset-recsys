from pytest import approx

from dataset_recsys.mathe_recommenders import question_embedding

from fakes import FakeEmbeddingClient, fake_mathe_client


def test_recommend_from_question_embedding_queries_mathe_vector_table(
    monkeypatch,
):
    mathe_client = fake_mathe_client()
    monkeypatch.setattr(
        question_embedding,
        "encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )
    embedding_client = FakeEmbeddingClient(
        [("220.pdf", 0.91), ("222.pdf", 0.85)]
    )
    mathe_client.get_pdf_material_metadata_by_redis_ids = lambda material_ids: [
        {
            "material_id": 220,
            "material_redis_id": "220.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
        {
            "material_id": 222,
            "material_redis_id": "222.pdf",
            "topic_ids": [99],
            "subtopic_ids": [99],
            "keywords": [],
        },
    ]

    recommendations = question_embedding.recommend_from_question_embedding(
        question="differentiate x^2",
        k=2,
        question_metadata={
            "topic_id": 10,
            "subtopic_id": 20,
            "keywords": ["derivatives"],
        },
        mathe_mirror_client=mathe_client,
        embedding_client=embedding_client,
        candidate_limit=2,
    )

    assert recommendations[0]["material_redis_id"] == "220.pdf"
    assert recommendations[0]["metadata_score"] == 1.0
    assert recommendations[0]["question_to_material_similarity"] == 0.91
    assert recommendations[0]["total_score"] == approx(0.955)
    assert embedding_client.calls == [
        {
            "application": "mathe",
            "query_embedding": [0.1, 0.2, 0.3],
            "top_k": 2,
            "table": "mathe_embeddings",
        }
    ]


def test_recommend_from_question_embedding_reranks_with_metadata(monkeypatch):
    mathe_client = fake_mathe_client()
    monkeypatch.setattr(
        question_embedding,
        "encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )
    embedding_client = FakeEmbeddingClient(
        [("weak_metadata.pdf", 0.95), ("strong_metadata.pdf", 0.8)]
    )
    mathe_client.get_pdf_material_metadata_by_redis_ids = lambda material_ids: [
        {
            "material_id": 1,
            "material_redis_id": "weak_metadata.pdf",
            "topic_ids": [99],
            "subtopic_ids": [99],
            "keywords": [],
        },
        {
            "material_id": 2,
            "material_redis_id": "strong_metadata.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        },
    ]

    recommendations = question_embedding.recommend_from_question_embedding(
        question="differentiate x^2",
        k=1,
        embedding_client=embedding_client,
        question_metadata={
            "topic_id": 10,
            "subtopic_id": 20,
            "keywords": ["derivatives"],
        },
        mathe_mirror_client=mathe_client,
        similarity_weight=0.5,
        candidate_limit=2,
    )

    assert recommendations[0]["material_redis_id"] == "strong_metadata.pdf"
    assert recommendations[0]["metadata_score"] == 1.0
    assert recommendations[0]["question_to_material_similarity"] == 0.8
    assert recommendations[0]["total_score"] == 0.9
