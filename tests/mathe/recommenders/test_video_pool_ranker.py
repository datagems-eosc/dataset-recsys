from dataset_recsys.mathe_recommenders.video_pool_ranker import (
    rank_video_pool_candidates,
    recommend_videos_for_question,
)

from fakes import FakeEmbeddingClient, fake_mathe_client


def test_rank_video_pool_uses_only_video_namespace_and_embedded_pool_members():
    mathe_client = fake_mathe_client()
    requested_question_ids = []

    def get_videos_for_question(question_id):
        requested_question_ids.append(question_id)
        return [
            {"material_id": 901, "platform_type": 1},
            {"material_id": 902, "platform_type": 2},
            {"material_id": 903, "platform_type": 1},
        ]

    mathe_client.get_videos_for_question = get_videos_for_question
    embedding_client = FakeEmbeddingClient(
        [("901", 0.6), ("902", 0.9), ("999", 1.0)]
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=3,
        mathe_mirror_client=mathe_client,
        embedding_client=embedding_client,
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == ["902", "901"]
    assert [candidate["platform_type"] for candidate in candidates] == [2, 1]
    assert candidates[0]["question_to_video_similarity"] == 0.9
    assert requested_question_ids == [42]
    assert embedding_client.calls == [
        {
            "method": "find_similar_by_ids",
            "application": "mathe_videos",
            "query_embedding": [0.1, 0.2],
            "entity_ids": ["901", "902", "903"],
            "table": embedding_client.TABLE_MATHE,
        }
    ]


def test_recommend_videos_returns_ranked_platform_ids():
    mathe_client = fake_mathe_client()
    mathe_client.get_videos_for_question = lambda question_id: [
        {"material_id": 901, "platform_type": 1},
        {"material_id": 902, "platform_type": 2},
    ]

    recommendations = recommend_videos_for_question(
        question_id=42,
        question="Explain the chain rule.",
        k=1,
        mathe_mirror_client=mathe_client,
        embedding_client=FakeEmbeddingClient([("901", 0.6), ("902", 0.9)]),
        question_embedding=[0.1, 0.2],
    )

    assert recommendations == ["902"]


def test_rank_video_pool_returns_empty_without_pool_or_embeddings():
    mathe_client = fake_mathe_client()
    mathe_client.get_videos_for_question = lambda question_id: []

    assert (
        rank_video_pool_candidates(
            question_id=42,
            question="Explain the chain rule.",
            k=3,
            mathe_mirror_client=mathe_client,
            embedding_client=FakeEmbeddingClient([]),
            question_embedding=[0.1, 0.2],
        )
        == []
    )

    mathe_client.get_videos_for_question = lambda question_id: [
        {"material_id": 901, "platform_type": 1}
    ]
    assert (
        rank_video_pool_candidates(
            question_id=42,
            question="Explain the chain rule.",
            k=3,
            mathe_mirror_client=mathe_client,
            embedding_client=FakeEmbeddingClient([]),
            question_embedding=[0.1, 0.2],
        )
        == []
    )
