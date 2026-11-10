from pytest import approx

from dataset_recsys.mathe_recommenders import video_pool_ranker
from dataset_recsys.mathe_recommenders.video_pool_ranker import (
    rank_video_pool_candidates,
    recommend_videos_for_question,
    score_video_pool_candidates,
)

from fakes import FakeQdrantStorageClient, fake_mathe_client


def _mathe_client(videos, question_keywords=None):
    mathe_client = fake_mathe_client()
    mathe_client.get_videos_for_question = lambda question_id: videos
    mathe_client.get_question_metadata = lambda question_id: {
        "keywords": question_keywords or [],
    }
    return mathe_client


def test_rank_video_pool_scores_only_pool_members_in_the_video_namespace():
    requested_question_ids = []
    mathe_client = fake_mathe_client()

    def get_videos_for_question(question_id):
        requested_question_ids.append(question_id)
        return [
            {"material_id": 901, "platform_type": 1, "keywords": []},
            {"material_id": 902, "platform_type": 2, "keywords": []},
            {"material_id": 903, "platform_type": 1, "keywords": []},
        ]

    mathe_client.get_videos_for_question = get_videos_for_question
    mathe_client.get_question_metadata = lambda question_id: {"keywords": []}
    qdrant_client = FakeQdrantStorageClient(
        similar_results=[("901", 0.6), ("902", 0.9), ("999", 1.0)]
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=3,
        mathe_mirror_client=mathe_client,
        qdrant_client=qdrant_client,
        question_embedding=[0.1, 0.2],
    )

    # 999 has the best similarity but is outside the pool, so it never appears.
    # 903 has neither an embedding nor a shared keyword, so it is left out.
    assert [candidate["material_id"] for candidate in candidates] == ["902", "901"]
    assert [candidate["platform_type"] for candidate in candidates] == [2, 1]
    assert candidates[0]["question_to_video_similarity"] == 0.9
    assert requested_question_ids == [42]
    assert qdrant_client.calls == [
        {
            "method": "find_similar_by_ids",
            "application": "mathe_videos",
            "query_embedding": [0.1, 0.2],
            "entity_ids": ["901", "901.txt", "902", "902.txt", "903", "903.txt"],
            "collection_name": qdrant_client.COLLECTION_MATHE,
        }
    ]


def test_rank_video_pool_combines_keyword_overlap_and_similarity():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": ["Chain rule"]},
            {"material_id": 902, "platform_type": 2, "keywords": ["Integrals"]},
        ],
        question_keywords=["Chain rule", "Power rule"],
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=2,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("901", 0.2), ("902", 0.9)]
        ),
        keyword_weight=0.6,
        question_embedding=[0.1, 0.2],
    )

    # 901: 0.6 * (1 shared / 2 distinct) + 0.4 * 0.2 = 0.38
    # 902: 0.6 * 0.0 + 0.4 * 0.9 = 0.36
    assert [candidate["material_id"] for candidate in candidates] == ["901", "902"]
    assert candidates[0]["keyword_jaccard"] == approx(0.5)
    assert candidates[0]["final_score"] == approx(0.38)
    assert candidates[1]["keyword_jaccard"] == 0.0
    assert candidates[1]["final_score"] == approx(0.36)


def test_video_without_embedding_is_ranked_on_keyword_overlap_alone():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": ["Integrals"]},
            {"material_id": 902, "platform_type": 1, "keywords": ["Chain rule"]},
        ],
        question_keywords=["Chain rule"],
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=2,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(similar_results=[("901", 0.5)]),
        keyword_weight=0.6,
        question_embedding=[0.1, 0.2],
    )

    # 902 has no embedding: 0.6 * 1.0 + 0.4 * 0.0 = 0.6, above 901's 0.2.
    assert [candidate["material_id"] for candidate in candidates] == ["902", "901"]
    assert candidates[0]["question_to_video_similarity"] == 0.0
    assert candidates[0]["final_score"] == approx(0.6)
    assert candidates[1]["final_score"] == approx(0.2)


def test_videos_with_equal_scores_keep_pool_order():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": []},
            {"material_id": 902, "platform_type": 2, "keywords": None},
            {"material_id": 903, "platform_type": 1},
        ],
        question_keywords=["Chain rule"],
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=3,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("903", 0.5), ("902", 0.5), ("901", 0.5)]
        ),
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "901",
        "902",
        "903",
    ]


# A video with no embedding and no shared keyword has no ranking signal.
# Returning it would place it in an arbitrary position.
def test_videos_without_embedding_or_shared_keyword_are_left_out():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": ["Integrals"]},
            {"material_id": 902, "platform_type": 2, "keywords": []},
            {"material_id": 903, "platform_type": 1, "keywords": ["Chain rule"]},
        ],
        question_keywords=["Chain rule"],
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=3,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(similar_results=[]),
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == ["903"]

    no_signal_client = _mathe_client(
        videos=[{"material_id": 901, "platform_type": 1, "keywords": ["Integrals"]}],
        question_keywords=["Chain rule"],
    )
    assert (
        rank_video_pool_candidates(
            question_id=42,
            question="Explain the chain rule.",
            k=3,
            mathe_mirror_client=no_signal_client,
            qdrant_client=FakeQdrantStorageClient(similar_results=[]),
            question_embedding=[0.1, 0.2],
        )
        == []
    )


# A negative cosine similarity counts as 0.0 in the weighted sum, so it never
# cancels out part of the keyword score.
def test_negative_similarity_does_not_reduce_the_video_keyword_score():
    mathe_client = _mathe_client(
        videos=[{"material_id": 901, "platform_type": 1, "keywords": ["Chain rule"]}],
        question_keywords=["Chain rule"],
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=1,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(similar_results=[("901", -0.5)]),
        keyword_weight=0.6,
        question_embedding=[0.1, 0.2],
    )

    assert candidates[0]["final_score"] == approx(0.6)
    assert candidates[0]["question_to_video_similarity"] == -0.5


# An embedded video with a zero or negative similarity and no shared keyword
# scores zero as well, so it is left out like an unembedded one.
def test_embedded_video_with_zero_score_is_left_out():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": []},
            {"material_id": 902, "platform_type": 2, "keywords": []},
            {"material_id": 903, "platform_type": 1, "keywords": []},
        ],
        question_keywords=["Chain rule"],
    )

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=3,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("901", -0.1), ("902", 0.0), ("903", 0.2)]
        ),
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == ["903"]


def test_score_video_pool_candidates_reports_zero_score_videos():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": ["Integrals"]},
            {"material_id": 902, "platform_type": 2, "keywords": ["Chain rule"]},
            {"material_id": 903, "platform_type": 1, "keywords": []},
        ],
        question_keywords=["Chain rule"],
    )

    scored = score_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(similar_results=[("903", -0.2)]),
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in scored] == ["901", "902", "903"]
    assert {
        candidate["material_id"]: candidate["exclusion_reason"]
        for candidate in scored
    } == {
        "901": "zero_score_no_embedding",
        "902": None,
        "903": "zero_score_with_embedding",
    }
    assert [candidate["has_embedding"] for candidate in scored] == [False, False, True]


def test_video_pool_ranker_logs_zero_score_videos(caplog):
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": ["Integrals"]},
            {"material_id": 902, "platform_type": 2, "keywords": ["Chain rule"]},
        ],
        question_keywords=["Chain rule"],
    )

    with caplog.at_level("INFO"):
        rank_video_pool_candidates(
            question_id=42,
            question="Explain the chain rule.",
            k=2,
            mathe_mirror_client=mathe_client,
            qdrant_client=FakeQdrantStorageClient(similar_results=[]),
            question_embedding=[0.1, 0.2],
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "video ranker left out 1 of 2 pool materials" in message
    assert "question 42" in message
    assert "'901': 'zero_score_no_embedding'" in message


def test_rank_video_pool_ranks_by_similarity_when_question_metadata_is_missing():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": ["Chain rule"]},
            {"material_id": 902, "platform_type": 2, "keywords": ["Chain rule"]},
        ],
    )
    mathe_client.get_question_metadata = lambda question_id: None

    candidates = rank_video_pool_candidates(
        question_id=42,
        question="Explain the chain rule.",
        k=2,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("901", 0.3), ("902", 0.7)]
        ),
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == ["902", "901"]
    assert [candidate["keyword_jaccard"] for candidate in candidates] == [0.0, 0.0]


def test_video_keyword_weight_defaults_to_the_document_weight():
    assert video_pool_ranker.MATHE_VIDEO_KEYWORD_WEIGHT == 0.6


def test_recommend_videos_returns_ranked_platform_ids():
    mathe_client = _mathe_client(
        videos=[
            {"material_id": 901, "platform_type": 1, "keywords": []},
            {"material_id": 902, "platform_type": 2, "keywords": []},
        ],
    )

    recommendations = recommend_videos_for_question(
        question_id=42,
        question="Explain the chain rule.",
        k=1,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("901", 0.6), ("902", 0.9)]
        ),
        question_embedding=[0.1, 0.2],
    )

    assert recommendations == ["902"]


def test_rank_video_pool_returns_empty_without_a_pool_or_a_question():
    assert (
        rank_video_pool_candidates(
            question_id=42,
            question="Explain the chain rule.",
            k=3,
            mathe_mirror_client=_mathe_client(videos=[]),
            qdrant_client=FakeQdrantStorageClient(similar_results=[]),
            question_embedding=[0.1, 0.2],
        )
        == []
    )

    mathe_client = _mathe_client(
        videos=[{"material_id": 901, "platform_type": 1, "keywords": []}],
    )
    for question, k in [("   ", 3), ("Explain the chain rule.", 0)]:
        assert (
            rank_video_pool_candidates(
                question_id=42,
                question=question,
                k=k,
                mathe_mirror_client=mathe_client,
                qdrant_client=FakeQdrantStorageClient(similar_results=[]),
                question_embedding=[0.1, 0.2],
            )
            == []
        )
