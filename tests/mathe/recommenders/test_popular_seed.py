from dataset_recsys.mathe_recommenders.popular_seed import recommend_from_popular_seed

from fakes import FakeRecommendationClient, fake_mathe_client


def test_recommend_from_popular_seed_uses_document_platform_ids():
    mathe_client = fake_mathe_client()
    mathe_client.get_popular_document_for_question = (
        lambda question_id: {"material_id": 6}
    )
    recommendation_client = FakeRecommendationClient(
        {"6": ["10", "11", "12"]}
    )

    recommendations = recommend_from_popular_seed(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=recommendation_client,
    )

    assert recommendations == ["10", "11"]
    assert recommendation_client.calls == [
        {
            "application": "mathe_documents",
            "entity_id": "6",
            "limit": 2,
        }
    ]
