"""Tests for QdrantStorageClient against Qdrant's in-memory mode.

No Qdrant server is needed: ``QdrantClient(":memory:")`` runs the same client
API inside the test process.
"""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from dataset_recsys.storage.qdrant_client import QdrantStorageClient

DATASETS = QdrantStorageClient.COLLECTION_DATASET
MATHE = QdrantStorageClient.COLLECTION_MATHE


@pytest.fixture
def client():
    return QdrantStorageClient(client=QdrantClient(":memory:"))


def _store(client, application, vectors, collection_name=None):
    """Store one embedding per entity ID from an ``{id: vector}`` mapping."""
    return client.store_embeddings(
        application=application,
        dataset_ids=list(vectors),
        embeddings=np.array(list(vectors.values()), dtype=float),
        embedding_inputs=[f"text of {entity_id}" for entity_id in vectors],
        embedding_model="test-model",
        collection_name=collection_name,
        run_id="run-1",
    )


def test_constructor_wraps_an_injected_client():
    qdrant = QdrantClient(":memory:")

    assert QdrantStorageClient(client=qdrant).client is qdrant


# ---------------------------------------------------------------------------
# Before the first rebuild Qdrant has no collections. Reads must report
# "nothing stored" instead of failing, so the API does not return 500.
# ---------------------------------------------------------------------------


def test_reads_on_a_fresh_qdrant_report_nothing_stored(client):
    assert client.exists("a") is False
    assert client.get_entity_status("ds2ds", "a") == "NOT_FOUND"
    assert client.check_existence_batch("ds2ds", ["a", "b"]) == {"a": False, "b": False}
    assert client.get_recommendations_with_scores("ds2ds", "a") == []
    assert client.list_entities("ds2ds") == []
    assert client.find_similar("ds2ds", query_embedding=[1.0, 0.0]) == []
    assert client.find_similar_by_ids("ds2ds", [1.0, 0.0], ["a"]) == []
    assert client.find_entities_recommending("ds2ds", "a") == []


def test_deletes_on_a_fresh_qdrant_do_nothing(client):
    assert client.delete_single_embedding("a") == 0
    assert client.delete_application("ds2ds") == 0
    client.update_neighbor_recs("ds2ds", "a", "b", 0.5)
    client.remove_from_neighbor_recs("ds2ds", "a", "b")


# ---------------------------------------------------------------------------
# Embeddings and similarity search
# ---------------------------------------------------------------------------


def test_store_embeddings_routes_by_application_and_reports_existence(client):
    assert _store(client, "ds2ds", {"a": [1.0, 0.0]}) == 1
    assert _store(client, "mathe_documents", {"221": [1.0, 0.0]}) == 1

    assert client.exists("a") is True
    assert client.exists("221") is False
    assert client.exists("221", collection_name=MATHE) is True
    assert client.check_existence_batch("ds2ds", ["a", "missing"]) == {
        "a": True,
        "missing": False,
    }


def test_store_embeddings_with_no_rows_stores_nothing(client):
    assert _store(client, "ds2ds", {}) == 0
    assert client.list_entities("ds2ds") == []


def test_find_similar_returns_best_first_dicts_within_one_application(client):
    _store(
        client,
        "mathe_documents",
        {"1": [1.0, 0.0], "2": [0.6, 0.8], "3": [0.0, 1.0]},
        collection_name=MATHE,
    )
    _store(client, "mathe_videos", {"900": [1.0, 0.0]}, collection_name=MATHE)

    matches = client.find_similar(
        "mathe_documents",
        query_embedding=[1.0, 0.0],
        top_k=2,
        collection_name=MATHE,
    )

    assert [match["dataset_id"] for match in matches] == ["1", "2"]
    assert matches[0]["score"] == pytest.approx(1.0)
    assert matches[1]["score"] == pytest.approx(0.6)
    assert matches[0]["payload"]["application"] == "mathe_documents"


def test_find_similar_without_top_k_returns_the_complete_ranking(client):
    vectors = {
        f"entity-{index:02d}": [1.0, index / 10.0] for index in range(25)
    }
    _store(client, "ds2ds", vectors)
    _store(client, "other", {"elsewhere": [1.0, 0.0]})

    matches = client.find_similar("ds2ds", query_embedding=[1.0, 0.0], top_k=None)

    assert [match["dataset_id"] for match in matches] == sorted(vectors)
    assert client.count_entities("ds2ds") == 25
    assert client.count_entities("ds2ds", collection_name="missing_collection") == 0
    assert client.find_similar("empty-app", query_embedding=[1.0, 0.0], top_k=None) == []


def test_find_similar_accepts_query_vector_and_requires_a_vector(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})

    assert [m["dataset_id"] for m in client.find_similar("ds2ds", query_vector=[1.0, 0.0])] == ["a"]
    with pytest.raises(ValueError, match="query_embedding"):
        client.find_similar("ds2ds")


def test_find_similar_by_ids_scores_only_requested_stored_ids(client):
    _store(
        client,
        "mathe_documents",
        {"1": [1.0, 0.0], "2": [0.6, 0.8], "3": [0.0, 1.0]},
        collection_name=MATHE,
    )
    _store(client, "mathe_videos", {"900": [1.0, 0.0]}, collection_name=MATHE)

    scores = dict(
        client.find_similar_by_ids(
            "mathe_documents",
            [1.0, 0.0],
            ["2", "3", "900", "missing"],
            collection_name=MATHE,
        )
    )

    # "1" was not requested, "900" belongs to another application and
    # "missing" has no embedding, so none of them is scored.
    assert set(scores) == {"2", "3"}
    assert scores["2"] == pytest.approx(0.6)
    assert scores["3"] == pytest.approx(0.0)
    assert client.find_similar_by_ids("mathe_documents", [1.0, 0.0], [], collection_name=MATHE) == []


# A full rebuild is authoritative: entities removed from the catalog must not
# stay stored and recommendable.
def test_store_embeddings_with_replace_removes_entities_missing_from_the_new_set(client):
    _store(
        client,
        "mathe_documents",
        {"221": [1.0, 0.0], "222": [0.6, 0.8], "223": [0.0, 1.0]},
        collection_name=MATHE,
    )
    _store(client, "mathe_videos", {"900": [1.0, 0.0]}, collection_name=MATHE)
    client.store_recommendations(
        "mathe_documents",
        {"221": {"222": 0.6, "223": 0.1}, "223": {"222": 0.8}},
        collection_name=MATHE,
    )

    stored = client.store_embeddings(
        application="mathe_documents",
        dataset_ids=["221", "222"],
        embeddings=np.array([[1.0, 0.0], [0.6, 0.8]]),
        embedding_inputs=["a", "b"],
        embedding_model="test-model",
        collection_name=MATHE,
        replace=True,
    )

    assert stored == 2
    assert client.list_entities("mathe_documents", MATHE) == ["221", "222"]
    assert client.exists("223", collection_name=MATHE) is False
    assert client.get_recommendations_with_scores("mathe_documents", "223", collection_name=MATHE) == []
    # The other application in the same collection is untouched.
    assert client.list_entities("mathe_videos", MATHE) == ["900"]


def test_store_embeddings_without_replace_keeps_other_entities(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0], "b": [0.0, 1.0]})

    _store(client, "ds2ds", {"c": [0.6, 0.8]})

    assert client.list_entities("ds2ds") == ["a", "b", "c"]


def test_empty_batch_with_replace_deletes_nothing(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})

    stored = client.store_embeddings(
        application="ds2ds",
        dataset_ids=[],
        embeddings=np.empty((0, 2)),
        embedding_inputs=[],
        embedding_model="test-model",
        replace=True,
    )

    assert stored == 0
    assert client.list_entities("ds2ds") == ["a"]


def test_delete_entities_not_in_reports_how_many_were_removed(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.6, 0.8]})
    _store(client, "other", {"z": [1.0, 0.0]})

    assert client.delete_entities_not_in("ds2ds", ["a"]) == 2
    assert client.delete_entities_not_in("ds2ds", ["a"]) == 0
    assert client.list_entities("ds2ds") == ["a"]
    assert client.list_entities("other") == ["z"]
    assert client.delete_entities_not_in("ds2ds", ["a"], collection_name="missing") == 0


# A collection also holds other applications' embeddings and recommendations,
# so a write with the wrong vector size must be refused, not answered by
# deleting and recreating the collection.
def test_store_embeddings_refuses_a_different_vector_size_and_keeps_the_data(client):
    _store(client, "mathe_documents", {"1": [1.0, 0.0]}, collection_name=MATHE)
    _store(client, "mathe_videos", {"900": [0.0, 1.0]}, collection_name=MATHE)
    client.store_recommendations("mathe_documents", {"1": {"2": 0.9}}, collection_name=MATHE)

    with pytest.raises(ValueError, match="vectors of size 2.*size 3"):
        _store(client, "mathe_videos", {"901": [1.0, 0.0, 0.0]}, collection_name=MATHE)

    assert client.list_entities("mathe_documents", MATHE) == ["1"]
    assert client.list_entities("mathe_videos", MATHE) == ["900"]
    assert client.get_recommendations("mathe_documents", "1", collection_name=MATHE) == ["2"]
    assert client.get_schema_overview(MATHE)["vectors_config"]["size"] == 2


# ---------------------------------------------------------------------------
# Stored recommendations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"a": {"b": 0.9, "c": 0.4}}, [("b", 0.9), ("c", 0.4)]),
        ({"a": [("c", 0.4), ("b", 0.9)]}, [("b", 0.9), ("c", 0.4)]),
        ({"a": [{"id": "b", "score": 0.9}, {"id": "c", "score": 0.4}]}, [("b", 0.9), ("c", 0.4)]),
        # Ranked IDs without scores get descending rank scores, keeping the order.
        ({"a": ["b", "c"]}, [("b", 2.0), ("c", 1.0)]),
        (
            [{"id": "a", "recommendations": [{"id": "b", "score": 0.9}, {"id": "c", "score": 0.4}]}],
            [("b", 0.9), ("c", 0.4)],
        ),
    ],
    ids=["score-map", "pairs", "dicts", "ranked-ids", "catalog-list"],
)
def test_store_recommendations_accepts_each_input_shape(client, stored, expected):
    _store(client, "ds2ds", {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.6, 0.8]})

    assert client.store_recommendations("ds2ds", stored) == 1

    assert client.get_recommendations_with_scores("ds2ds", "a") == expected
    assert client.get_recommendations("ds2ds", "a") == [rec_id for rec_id, _ in expected]


def test_store_recommendations_with_no_data_stores_nothing(client):
    assert client.store_recommendations("ds2ds", {}) == 0
    assert client.store_recommendations("ds2ds", None) == 0


def test_recommendation_reads_respect_limit(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})
    client.store_recommendations("ds2ds", {"a": {"b": 0.9, "c": 0.4, "d": 0.1}})

    assert client.get_recommendations("ds2ds", "a", limit=2) == ["b", "c"]
    assert client.get_recommendations_with_scores("ds2ds", "a", limit=1) == [("b", 0.9)]
    assert client.get_recommendations("ds2ds", "a", limit=0) == []
    assert client.get_recommendations("ds2ds", "", limit=5) == []


def test_entity_status_distinguishes_missing_empty_and_available(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    client.store_recommendations("ds2ds", {"a": {"b": 0.9}})

    assert client.get_entity_status("ds2ds", "a") == "AVAILABLE"
    assert client.get_entity_status("ds2ds", "b") == "NO_RECOMMENDATIONS"
    assert client.get_entity_status("ds2ds", "missing") == "NOT_FOUND"
    assert client.get_entity_status("ds2ds", "") == "NOT_FOUND"


# The MathE seed expansion treats a material without stored neighbours as
# having none, as the Redis client did, while a direct lookup of an unknown
# entity is an error.
def test_unknown_entity_has_no_scored_recommendations_but_direct_lookup_raises(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})

    assert client.get_recommendations_with_scores("ds2ds", "missing") == []
    assert client.get_recommendations("ds2ds", "a") == []
    with pytest.raises(KeyError, match="missing"):
        client.get_recommendations("ds2ds", "missing")


def test_update_single_entity_recs_replaces_the_list(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})
    client.store_recommendations("ds2ds", {"a": {"b": 0.9}})

    client.update_single_entity_recs("ds2ds", "a", {"c": 0.7})

    assert client.get_recommendations_with_scores("ds2ds", "a") == [("c", 0.7)]


def test_update_neighbor_recs_inserts_in_score_order_and_trims_to_limit(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})
    client.store_recommendations("ds2ds", {"a": {"b": 0.9, "c": 0.4}})

    client.update_neighbor_recs("ds2ds", "a", "new", 0.6, limit=None)
    assert client.get_recommendations("ds2ds", "a") == ["b", "new", "c"]

    client.update_neighbor_recs("ds2ds", "a", "best", 0.95, limit=2)
    assert client.get_recommendations("ds2ds", "a") == ["best", "b"]


def test_remove_from_neighbor_recs_and_remove_single_entity_recs(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})
    client.store_recommendations("ds2ds", {"a": {"b": 0.9, "c": 0.4}})

    client.remove_from_neighbor_recs("ds2ds", "a", "b")
    assert client.get_recommendations("ds2ds", "a") == ["c"]

    client.remove_single_entity_recs("ds2ds", "a")
    assert client.get_entity_status("ds2ds", "a") == "NO_RECOMMENDATIONS"


def test_remove_dataset_deletes_the_point_and_references_to_it(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.6, 0.8]})
    client.store_recommendations(
        "ds2ds",
        {"a": {"b": 0.9, "c": 0.4}, "b": {"a": 0.9}, "c": {"b": 0.8, "a": 0.4}},
    )

    assert client.find_entities_recommending("ds2ds", "b") == ["a", "c"]
    client.remove_dataset("ds2ds", "b")

    assert client.exists("b") is False
    assert client.list_entities("ds2ds") == ["a", "c"]
    assert client.get_recommendations("ds2ds", "a") == ["c"]
    assert client.get_recommendations("ds2ds", "c") == ["a"]


def test_delete_application_removes_only_that_application(client):
    _store(client, "mathe_documents", {"1": [1.0, 0.0]}, collection_name=MATHE)
    _store(client, "mathe_videos", {"900": [1.0, 0.0]}, collection_name=MATHE)

    client.delete_application("mathe_documents", collection_name=MATHE)

    assert client.list_entities("mathe_documents", MATHE) == []
    assert client.list_entities("mathe_videos", MATHE) == ["900"]


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_list_entities_reads_every_page(client, monkeypatch):
    monkeypatch.setattr(QdrantStorageClient, "SCROLL_PAGE_SIZE", 2)
    entity_ids = [f"entity-{index}" for index in range(7)]
    _store(client, "ds2ds", {entity_id: [1.0, float(index)] for index, entity_id in enumerate(entity_ids)})
    _store(client, "other", {"elsewhere": [1.0, 0.0]})

    assert client.list_entities("ds2ds") == sorted(entity_ids)


def test_schema_overview_and_connection_check(client):
    _store(client, "ds2ds", {"a": [1.0, 0.0]})

    overview = client.get_schema_overview()
    assert overview["collection_name"] == DATASETS
    assert overview["vectors_config"]["size"] == 2
    assert "error" in client.get_schema_overview("missing_collection")
    assert client.check_connection() is True
