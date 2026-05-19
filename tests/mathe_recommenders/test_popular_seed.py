from dataset_recsys.mathe_recommenders.popular_seed import recommend_from_popular_seed

from fakes import FakeRecommendationClient, fake_mathe_client


def test_recommend_from_popular_seed_uses_previous_most_clicked_seed_flow():
    mathe_client = fake_mathe_client()
    mathe_client.get_material_by_question_id = lambda question_id: {
        "id": 6,
        "material_redis_id": "popular_derivatives.pdf",
    }

    recommendations = recommend_from_popular_seed(
        question_id=42,
        k=2,
        mathe_mirror_client=mathe_client,
        recommendation_client=FakeRecommendationClient(
            {"popular_derivatives.pdf": ["10.pdf", "11.pdf", "12.pdf"]}
        ),
    )

    assert recommendations == ["10.pdf", "11.pdf"]
