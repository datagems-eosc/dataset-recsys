from __future__ import annotations
import numpy as np
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    return normalized @ normalized.T

def build_recommendations(
    profiles: list[DatasetProfile],
    embeddings: np.ndarray,
    top_k: int | None = None,
) -> list[dict]:
    """
    If `top_k` is None, all other datasets are returned in ranked order.
    """
    if len(profiles) != len(embeddings):
        raise ValueError(
            "The number of profiles must match the number of embedding vectors."
        )
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or None.")

    similarity = cosine_similarity_matrix(embeddings)
    recommendations: list[dict] = []

    for i, profile in enumerate(profiles):
        ranked_indices = np.argsort(similarity[i])[::-1]
        ranked_indices = [j for j in ranked_indices if j != i]
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        recommendations.append(
            {
                "id": profile.id,
                "title": profile.title,
                "recommendations": [
                    {
                        "id": profiles[j].id,
                        "title": profiles[j].title,
                        "score": float(similarity[i, j]),
                    }
                    for j in ranked_indices
                ],
            }
        )

    return recommendations


__all__ = ["build_recommendations", "cosine_similarity_matrix"]