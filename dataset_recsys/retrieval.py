from __future__ import annotations

from collections.abc import Hashable, Sequence

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
    groups: Sequence[Hashable] | None = None,
) -> RankedNeighbors:
    """Rank similar entities by cosine similarity.

    ``groups``, when given, holds one key per entity; entities that share a
    key (for example, MathE materials that point to the same video) are not
    ranked as each other's neighbours. Without it, only the entity itself is
    excluded.
    """
    if len(entity_ids) != len(embeddings):
        raise ValueError("The number of entity ids must match the embedding count.")
    if groups is not None and len(groups) != len(entity_ids):
        raise ValueError("The number of groups must match the entity id count.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or None.")

    similarity = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(similarity, -np.inf)
    if groups is not None:
        same_group = np.array(
            [[a == b for b in groups] for a in groups], dtype=bool
        ).reshape(len(groups), len(groups))
        similarity[same_group] = -np.inf

    ranked_indices = np.argsort(-similarity, axis=1)

    rankings: RankedNeighbors = {}
    for i, entity_id in enumerate(entity_ids):
        neighbours = [
            (entity_ids[j], float(similarity[i, j]))
            for j in ranked_indices[i]
            if similarity[i, j] != -np.inf
        ]
        rankings[entity_id] = neighbours if top_k is None else neighbours[:top_k]
    return rankings


__all__ = [
    "RankedNeighbors",
    "cosine_similarity_matrix",
    "rank_similar_entities",
]
