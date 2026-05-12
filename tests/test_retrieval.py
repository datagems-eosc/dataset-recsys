import numpy as np
import pytest

from dataset_recsys.retrieval import rank_similar_entities

def test_rank_similar_entities_returns_all_neighbors_in_ranked_order():
    entity_ids = ["6.pdf", "7.pdf", "8.pdf"]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]],
        dtype=float,
    )

    recommendations = rank_similar_entities(entity_ids, embeddings)

    assert recommendations == {
        "6.pdf": [("8.pdf", 0.8), ("7.pdf", 0.0)],
        "7.pdf": [("8.pdf", 0.6), ("6.pdf", 0.0)],
        "8.pdf": [("6.pdf", 0.8), ("7.pdf", 0.6)],
    }

def test_rank_similar_entities_can_limit_top_k():
    entity_ids = ["6.pdf", "7.pdf", "8.pdf"]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]],
        dtype=float,
    )

    recommendations = rank_similar_entities(entity_ids, embeddings, top_k=1)

    assert recommendations == {
        "6.pdf": [("8.pdf", 0.8)],
        "7.pdf": [("8.pdf", 0.6)],
        "8.pdf": [("6.pdf", 0.8)],
    }

def test_rank_similar_entities_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        rank_similar_entities(["6.pdf"], np.array([[1.0]]), top_k=0)

# Bash: pytest -v tests/test_retrieval.py