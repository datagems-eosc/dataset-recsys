from __future__ import annotations
import numpy as np

RankedNeighbors = dict[str, list[tuple[str, float]]]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def rank_similar_entities(
    entity_ids: list[str],
    embeddings: np.ndarray,
    top_k: int | None = None,
) -> RankedNeighbors:
    """Rank similar entities by cosine similarity."""
    if len(entity_ids) != len(embeddings):
        raise ValueError("The number of entity ids must match the embedding count.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or None.")

    similarity = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(similarity, -np.inf)
    neighbor_count = max(len(entity_ids) - 1, 0)
    result_count = neighbor_count if top_k is None else min(top_k, neighbor_count)
    ranked_indices = np.argsort(-similarity, axis=1)[:, :result_count]

    return {
        entity_id: [
            (entity_ids[j], float(similarity[i, j]))
            for j in ranked_indices[i]
        ]
        for i, entity_id in enumerate(entity_ids)
    }


__all__ = [
    "RankedNeighbors",
    "cosine_similarity_matrix",
    "rank_similar_entities",
]
