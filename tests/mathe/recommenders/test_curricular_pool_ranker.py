from pytest import approx

from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    rank_curricular_pool_candidates,
    recommend_from_curricular_pool,
    score_curricular_pool_candidates,
)

from fakes import FakeQdrantStorageClient, fake_mathe_client


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
    qdrant_client = FakeQdrantStorageClient(
        similar_results=[("100", 0.2), ("101", 0.9), ("900", 1.0)]
    )

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate x sin(x).",
        k=2,
        mathe_mirror_client=mathe_client,
        qdrant_client=qdrant_client,
        keyword_weight=0.4,
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == [
        "101",
        "100",
    ]
    assert candidates[0]["final_score"] == approx(0.54)
    assert candidates[1]["final_score"] == approx(0.52)
    assert qdrant_client.calls == [
        {
            "method": "find_similar_by_ids",
            "application": "mathe_documents",
            "query_embedding": [0.1, 0.2],
            "entity_ids": ["100", "101"],
            "collection_name": qdrant_client.COLLECTION_MATHE,
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
    qdrant_client = FakeQdrantStorageClient(similar_results=[("101", 0.8)])

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate a composition.",
        k=2,
        mathe_mirror_client=mathe_client,
        qdrant_client=qdrant_client,
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
        qdrant_client=FakeQdrantStorageClient(similar_results=[("100", 0.9)]),
        question_embedding=[0.1, 0.2],
    )

    assert recommendations == ["100"]


def _document_pool_client(materials, question_keywords):
    mathe_client = fake_mathe_client()
    mathe_client.get_question_metadata = lambda question_id: {
        "question_id": question_id,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": question_keywords,
    }
    mathe_client.get_document_materials_for_question_topic_subtopic = (
        lambda question_id: materials
    )
    return mathe_client


# A document with no embedding and no shared keyword has no ranking signal.
# Returning it would place it in an arbitrary position.
def test_documents_without_embedding_or_shared_keyword_are_left_out():
    mathe_client = _document_pool_client(
        materials=[
            {"material_id": 100, "keywords": ["product rule"]},
            {"material_id": 101, "keywords": ["chain rule"]},
            {"material_id": 102, "keywords": []},
            {"material_id": 103, "keywords": ["integrals"]},
        ],
        question_keywords=["chain rule"],
    )

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate a composition.",
        k=4,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(similar_results=[("103", 0.3)]),
        question_embedding=[0.1, 0.2],
    )

    # 101 shares a keyword; 103 has an embedding; 100 and 102 have neither.
    assert [candidate["material_id"] for candidate in candidates] == ["101", "103"]


def test_document_pool_without_any_signal_returns_no_recommendations():
    mathe_client = _document_pool_client(
        materials=[{"material_id": 100, "keywords": ["product rule"]}],
        question_keywords=["chain rule"],
    )

    assert (
        recommend_from_curricular_pool(
            question_id=42,
            question="Differentiate a composition.",
            k=3,
            mathe_mirror_client=mathe_client,
            qdrant_client=FakeQdrantStorageClient(similar_results=[]),
            question_embedding=[0.1, 0.2],
        )
        == []
    )


# A negative cosine similarity counts as 0.0 in the weighted sum, so it never
# cancels out part of the keyword score.
def test_negative_similarity_does_not_reduce_the_document_keyword_score():
    mathe_client = _document_pool_client(
        materials=[{"material_id": 100, "keywords": ["chain rule"]}],
        question_keywords=["chain rule"],
    )

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate a composition.",
        k=1,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(similar_results=[("100", -0.5)]),
        keyword_weight=0.6,
        question_embedding=[0.1, 0.2],
    )

    assert candidates[0]["final_score"] == approx(0.6)
    assert candidates[0]["question_to_material_similarity"] == -0.5


# An embedded document with a zero or negative similarity and no shared keyword
# scores zero as well, so it is left out like an unembedded one.
def test_embedded_document_with_zero_score_is_left_out():
    mathe_client = _document_pool_client(
        materials=[
            {"material_id": 100, "keywords": []},
            {"material_id": 101, "keywords": []},
            {"material_id": 102, "keywords": []},
        ],
        question_keywords=["chain rule"],
    )

    candidates = rank_curricular_pool_candidates(
        question_id=42,
        question="Differentiate a composition.",
        k=3,
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("100", -0.1), ("101", 0.0), ("102", 0.2)]
        ),
        question_embedding=[0.1, 0.2],
    )

    assert [candidate["material_id"] for candidate in candidates] == ["102"]


def test_score_curricular_pool_candidates_reports_zero_score_materials():
    mathe_client = _document_pool_client(
        materials=[
            {"material_id": 100, "keywords": ["product rule"]},
            {"material_id": 101, "keywords": ["chain rule"]},
            {"material_id": 102, "keywords": []},
            {"material_id": 103, "keywords": []},
        ],
        question_keywords=["chain rule"],
    )

    scored = score_curricular_pool_candidates(
        question_id=42,
        question="Differentiate a composition.",
        mathe_mirror_client=mathe_client,
        qdrant_client=FakeQdrantStorageClient(
            similar_results=[("102", -0.2), ("103", 0.5)]
        ),
        question_embedding=[0.1, 0.2],
    )

    # The whole pool comes back in pool order, including left-out materials.
    assert [candidate["material_id"] for candidate in scored] == [
        "100",
        "101",
        "102",
        "103",
    ]
    assert {
        candidate["material_id"]: candidate["exclusion_reason"]
        for candidate in scored
    } == {
        "100": "zero_score_no_embedding",
        "101": None,
        "102": "zero_score_with_embedding",
        "103": None,
    }
    assert [candidate["has_embedding"] for candidate in scored] == [
        False,
        False,
        True,
        True,
    ]
    assert [candidate["final_score"] for candidate in scored] == [
        0.0,
        approx(0.6),
        0.0,
        approx(0.2),
    ]


def test_curricular_pool_ranker_logs_zero_score_materials(caplog):
    mathe_client = _document_pool_client(
        materials=[
            {"material_id": 100, "keywords": ["product rule"]},
            {"material_id": 101, "keywords": ["chain rule"]},
        ],
        question_keywords=["chain rule"],
    )

    with caplog.at_level("INFO"):
        rank_curricular_pool_candidates(
            question_id=42,
            question="Differentiate a composition.",
            k=2,
            mathe_mirror_client=mathe_client,
            qdrant_client=FakeQdrantStorageClient(similar_results=[]),
            question_embedding=[0.1, 0.2],
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "document ranker left out 1 of 2 pool materials" in message
    assert "question 42" in message
    assert "'100': 'zero_score_no_embedding'" in message


def test_curricular_pool_ranker_does_not_log_when_every_material_is_ranked(caplog):
    mathe_client = _document_pool_client(
        materials=[{"material_id": 101, "keywords": ["chain rule"]}],
        question_keywords=["chain rule"],
    )

    with caplog.at_level("INFO"):
        rank_curricular_pool_candidates(
            question_id=42,
            question="Differentiate a composition.",
            k=2,
            mathe_mirror_client=mathe_client,
            qdrant_client=FakeQdrantStorageClient(similar_results=[]),
            question_embedding=[0.1, 0.2],
        )

    assert caplog.records == []
