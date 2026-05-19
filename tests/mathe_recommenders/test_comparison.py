from pytest import approx

from dataset_recsys.mathe_recommenders import question_embedding
from dataset_recsys.mathe_recommenders.comparison import compare_question_recommenders

from fakes import FakeEmbeddingClient, FakeRecommendationClient, fake_mathe_client


def test_compare_question_recommenders_includes_question_embedding_scores(
    monkeypatch,
):
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "topic": "Differentiation",
        "subtopic": "Derivatives",
        "keywords": ["derivatives"],
    }
    mathe_client.get_pdf_seed_candidates = lambda question_id: []
    mathe_client.get_pdf_material_metadata_by_redis_ids = lambda material_ids: [
        {
            "material_id": 220,
            "material_redis_id": "220.pdf",
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["derivatives"],
        }
    ]
    mathe_client.get_material_by_question_id = lambda question_id: None
    mathe_client.get_pdf_material_details = lambda material_ids: [
        {
            "material_id": 220,
            "material_redis_id": "220.pdf",
            "title": "Partial Derivatives",
        }
    ]
    monkeypatch.setattr(
        question_embedding,
        "encode_texts",
        lambda texts, model_name: [[0.1, 0.2, 0.3]],
    )

    report = compare_question_recommenders(
        question_id=42,
        k=1,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient({}),
        question_text="differentiate x^2",
        embedding_client=FakeEmbeddingClient([("220.pdf", 0.91)]),
    )

    recommendation = report["strategies"]["question_embedding"]["recommendations"][0]
    assert recommendation["material_id"] == 220
    assert recommendation["scores"] == {
        "keyword_jaccard": 1.0,
        "same_subtopic": 1,
        "same_topic": 1,
        "metadata_score": 1.0,
        "question_to_material_similarity": 0.91,
        "total_score": approx(0.955),
    }
