from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient

from dataset_recsys.mathe_recommenders import precomputed_rankings as rankings
from dataset_recsys.mathe_recommenders.curricular_pool_ranker import recommend_from_curricular_pool
from dataset_recsys.mathe_recommenders.video_pool_ranker import recommend_videos_for_question
from dataset_recsys.storage.qdrant_client import QdrantStorageClient


QUESTION = "Explain the chain rule."
COMPLETED = {"sync_status": "completed", "last_sync_started_at": "first-sync"}


def test_precomputed_rankings_are_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MATHE_PRECOMPUTED_RANKINGS", raising=False)
    assert rankings.precomputed_rankings_enabled()
    monkeypatch.setenv("MATHE_PRECOMPUTED_RANKINGS", "false")
    assert not rankings.precomputed_rankings_enabled()


@pytest.fixture
def services(monkeypatch):
    qdrant = QdrantStorageClient(client=QdrantClient(":memory:"))
    qdrant.store_embeddings(
        application="mathe_documents",
        dataset_ids=["11"],
        embeddings=np.array([[1.0, 0.0]]),
        embedding_inputs=["document"],
        embedding_model="test-model",
    )
    qdrant.store_embeddings(
        application="mathe_videos",
        dataset_ids=["21.txt"],
        embeddings=np.array([[1.0, 0.0]]),
        embedding_inputs=["video"],
        embedding_model="test-model",
    )
    mirror = SimpleNamespace(
        get_recommendation_questions=lambda: [
            {"question_id": 42, "question": QUESTION},
            {"question_id": 43, "question": "A question without candidates"},
        ],
        get_question_metadata=lambda _: {"keywords": ["Chain rule"]},
        get_document_materials_for_question_topic_subtopic=lambda question_id: [
            {"material_id": 11, "keywords": []},
            {"material_id": 12, "keywords": ["Chain rule"]},
            {"material_id": 13, "keywords": []},
        ] if question_id == 42 else [],
        get_videos_for_question=lambda question_id: [
            {"material_id": 21, "keywords": []},
            {"material_id": 22, "keywords": ["Chain rule"]},
        ] if question_id == 42 else [],
    )
    monkeypatch.setattr(rankings, "encode_texts", lambda texts, _: np.tile([1.0, 0.0], (len(texts), 1)))
    yield mirror, qdrant
    qdrant.close()


def test_precomputed_rankings_match_both_live_rankers(services):
    mirror, qdrant = services
    assert rankings.get_precomputed_ranking(qdrant, 42, QUESTION, "document", 10, COMPLETED) is None
    assert rankings.refresh_question_rankings(mirror, qdrant, lambda: COMPLETED) == 2

    for content_type, recommend in (
        ("document", recommend_from_curricular_pool),
        ("video", recommend_videos_for_question),
    ):
        expected = recommend(42, QUESTION, 10, mirror, qdrant, question_embedding=[1.0, 0.0])
        assert len(expected) == 2
        assert rankings.get_precomputed_ranking(qdrant, 42, QUESTION, content_type, 10, COMPLETED) == expected
        assert rankings.get_precomputed_ranking(qdrant, 42, f" {QUESTION} ", content_type, 1, COMPLETED) == expected[:1]

    point = qdrant.client.retrieve(qdrant.MATHE_QUESTION_RANKINGS, ids=[42])[0]
    scores = {row["material_id"]: row for row in point.payload["document_scores"]}
    assert scores["13"]["exclusion_reason"] == "zero_score_no_embedding"
    assert scores["11"]["similarity"] == pytest.approx(1.0)
    assert rankings.get_precomputed_ranking(
        qdrant, 43, "A question without candidates", "document", 10, COMPLETED
    ) == []


def test_precomputed_rankings_skip_stale_or_unknown_questions(services, monkeypatch):
    mirror, qdrant = services
    monkeypatch.setattr(rankings, "time", lambda: 1000)
    rankings.refresh_question_rankings(mirror, qdrant, lambda: COMPLETED)
    assert rankings.get_precomputed_ranking(qdrant, 99, QUESTION, "document", 10, COMPLETED) is None
    assert rankings.get_precomputed_ranking(qdrant, 42, "Changed text", "document", 10, COMPLETED) is None

    for status in (
        {**COMPLETED, "sync_status": "running"},
        {**COMPLETED, "sync_status": "failed"},
        {**COMPLETED, "last_sync_started_at": "second-sync"},
    ):
        assert rankings.get_precomputed_ranking(qdrant, 42, QUESTION, "document", 10, status) is None

    monkeypatch.setattr(rankings, "time", lambda: 1000 + 365 * 24 * 60 * 60)
    assert rankings.get_precomputed_ranking(qdrant, 42, QUESTION, "document", 10, COMPLETED) == ["12", "11"]


def test_interrupted_refresh_keeps_the_active_snapshot(services, monkeypatch):
    mirror, qdrant = services
    rankings.refresh_question_rankings(mirror, qdrant, lambda: COMPLETED)
    previous = qdrant.client.get_aliases().aliases[0].collection_name
    monkeypatch.setattr(rankings, "BATCH_SIZE", 1)
    encode = rankings.encode_texts

    def fail_second_batch(texts, model):
        if texts != [QUESTION]:
            raise RuntimeError("embedding unavailable")
        return encode(texts, model)

    monkeypatch.setattr(rankings, "encode_texts", fail_second_batch)
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        rankings.refresh_question_rankings(mirror, qdrant, lambda: COMPLETED)
    assert qdrant.client.get_aliases().aliases[0].collection_name == previous
    assert {row.name for row in qdrant.client.get_collections().collections} == {
        qdrant.COLLECTION_MATHE, previous,
    }

    monkeypatch.setattr(rankings, "encode_texts", encode)
    running = {"sync_status": "running", "last_sync_started_at": "second-sync"}
    rankings.refresh_question_rankings(mirror, qdrant, lambda: running, sync_generation="second-sync")
    assert rankings.get_precomputed_ranking(qdrant, 42, QUESTION, "document", 10, COMPLETED) is None
    assert rankings.get_precomputed_ranking(
        qdrant, 42, QUESTION, "document", 10, {**running, "sync_status": "completed"}
    ) == ["12", "11"]
