"""Tests for adding one dataset to the DataGEMS dataset index."""

import asyncio

import numpy as np
import pytest
from qdrant_client import QdrantClient

from dataset_recsys import embeddings
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile
from dataset_recsys.storage.qdrant_client import QdrantStorageClient
from dataset_recsys.workflows import incremental_update


@pytest.fixture
def qdrant():
    return QdrantStorageClient(client=QdrantClient(":memory:"))


@pytest.fixture
def add_dataset(monkeypatch, qdrant):
    """Add a dataset with a chosen embedding, without calling an LLM or a model."""
    vectors_by_id = {}

    monkeypatch.setattr(incremental_update, "enrich_batch", lambda profiles: profiles)
    monkeypatch.setattr(embeddings, "build_embedding_text", lambda profile: profile.id)
    monkeypatch.setattr(
        embeddings,
        "encode_texts",
        lambda texts, model_name: np.array([vectors_by_id[text] for text in texts]),
    )

    def add(dataset_id, vector):
        vectors_by_id[dataset_id] = vector
        profile = DatasetProfile(
            id=dataset_id,
            title=f"Dataset {dataset_id}",
            headline="",
            description="",
            keywords=[],
            field_of_science=[],
        )
        return asyncio.run(
            incremental_update.process_incremental_update(
                profile,
                application="ds2ds",
                qdrant_client=qdrant,
            )
        )

    return add


def test_first_dataset_is_added_to_a_fresh_index(add_dataset, qdrant):
    assert add_dataset("a", [1.0, 0.0]) is True

    assert qdrant.list_entities("ds2ds") == ["a"]
    assert qdrant.get_entity_status("ds2ds", "a") == "NO_RECOMMENDATIONS"


def test_existing_dataset_is_not_added_again(add_dataset, qdrant):
    add_dataset("a", [1.0, 0.0])

    assert add_dataset("a", [0.0, 1.0]) is False
    assert qdrant.list_entities("ds2ds") == ["a"]


def test_new_dataset_and_its_neighbours_are_linked_in_score_order(add_dataset, qdrant):
    add_dataset("a", [1.0, 0.0])
    add_dataset("b", [0.0, 1.0])

    add_dataset("c", [0.8, 0.6])

    assert qdrant.get_recommendations_with_scores("ds2ds", "c") == [
        ("a", pytest.approx(0.8)),
        ("b", pytest.approx(0.6)),
    ]
    assert qdrant.get_recommendations("ds2ds", "a") == ["c", "b"]
    assert qdrant.get_recommendations("ds2ds", "b") == ["c", "a"]


# /recommend filters a dataset's stored list down to the datasets the user may
# see, so the list must hold the full ranking, not only the nearest hundred.
def test_new_dataset_stores_its_ranking_against_every_other_dataset(add_dataset, qdrant):
    existing_ids = [f"dataset-{index:03d}" for index in range(120)]
    qdrant.store_embeddings(
        application="ds2ds",
        dataset_ids=existing_ids,
        embeddings=np.array([[1.0, index / 100.0] for index in range(120)]),
        embedding_inputs=existing_ids,
        embedding_model="test-model",
    )

    add_dataset("new", [1.0, 0.0])

    assert qdrant.get_recommendations("ds2ds", "new") == existing_ids
    for dataset_id in (existing_ids[0], existing_ids[-1]):
        assert qdrant.get_recommendations("ds2ds", dataset_id) == ["new"]
