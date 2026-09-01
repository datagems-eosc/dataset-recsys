import os

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.mathe_recommenders.seed_scoring import (
    score_document_seed_candidates,
)
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


MATHE_MATERIAL_SIMILARITY_WEIGHT = min(
    max(float(os.getenv("MATHE_EMBEDDING_WEIGHT", "0.5")), 0.0),
    1.0,
)
MATHE_NEIGHBORS_PER_SEED = int(os.getenv("MATHE_NEIGHBORS_PER_SEED", "20"))


def seed_material_id(seed: dict) -> str:
    return str(seed["material_id"]).strip()


def add_metadata_scores(
    candidates: dict[str, dict],
    question_metadata: dict,
    mathe_mirror_client: MatheMirrorClient,
) -> None:
    material_metadata = mathe_mirror_client.get_document_material_metadata_by_ids(
        list(candidates.keys())
    )
    scored_metadata = score_document_seed_candidates(
        question_metadata,
        material_metadata,
    )

    for metadata in scored_metadata:
        material_id = str(metadata["material_id"]).strip()
        candidate = candidates.get(material_id)
        if candidate:
            candidate.update(metadata)
            candidate["material_id"] = material_id


def rank_expanded_candidates(
    seeds: list[dict],
    k: int,
    recommendation_client: RecommendationClient,
    question_metadata: dict | None = None,
    mathe_mirror_client: MatheMirrorClient | None = None,
    material_similarity_weight: float = MATHE_MATERIAL_SIMILARITY_WEIGHT,
    neighbors_per_seed: int = MATHE_NEIGHBORS_PER_SEED,
) -> list[dict]:
    """
    Rank MathE document recommendations from metadata seeds, expanding with
    embedding neighbors only when fewer than k metadata seeds exist.
    """
    seed_by_entity_id = {
        seed_material_id(seed): seed
        for seed in seeds
    }

    if k <= 0:
        return []

    if len(seed_by_entity_id) >= k:
        return [
            {
                **seed,
                "material_id": entity_id,
                "material_to_material_similarity": 0.0,
                "final_score": float(seed.get("metadata_score", 0.0)),
            }
            for entity_id, seed in list(seed_by_entity_id.items())[:k]
        ]

    candidates: dict[str, dict] = {}
    for entity_id, seed in seed_by_entity_id.items():
        metadata_score = float(seed.get("metadata_score", 0.0))
        candidates[entity_id] = {
            "material_id": entity_id,
            "metadata_score": metadata_score,
            "material_to_material_similarity": 1.0,
        }

    for seed_entity_id in seed_by_entity_id:
        # Failed-OCR materials have no stored recommendation neighbors. They
        # stay in the candidate pool as metadata seeds, but do not expand.
        neighbors = recommendation_client.get_recommendations_with_scores(
            application=MatheApplication.DOCUMENTS,
            entity_id=seed_entity_id,
            limit=neighbors_per_seed,
        )

        for neighbor_id, material_to_material_similarity in neighbors:
            neighbor_id = str(neighbor_id).strip()
            candidate = candidates.setdefault(
                neighbor_id,
                {
                    "material_id": neighbor_id,
                    "metadata_score": 0.0,
                    "material_to_material_similarity": 0.0,
                },
            )
            candidate["material_to_material_similarity"] = max(
                float(candidate["material_to_material_similarity"]),
                material_to_material_similarity,
            )

    if question_metadata and mathe_mirror_client:
        add_metadata_scores(
            candidates,
            question_metadata,
            mathe_mirror_client,
        )

    for candidate in candidates.values():
        candidate["final_score"] = (
            material_similarity_weight
            * float(candidate["material_to_material_similarity"])
            + (1.0 - material_similarity_weight)
            * float(candidate["metadata_score"])
        )

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate["final_score"],
            candidate["metadata_score"],
            candidate["material_to_material_similarity"],
        ),
        reverse=True,
    )
    return ranked_candidates[:k]


def recommend_document_seeds_for_question(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
) -> list[dict]:
    """Return top-k metadata-scored document seed materials for a question."""
    if k <= 0:
        return []

    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return []

    seed_candidates = mathe_mirror_client.get_document_seed_candidates(question_id)
    scored_candidates = score_document_seed_candidates(
        dict(question_metadata),
        [dict(candidate) for candidate in seed_candidates],
    )

    return scored_candidates[:k]


def recommend_from_metadata_seeds(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
) -> list[str]:
    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return []

    seeds = recommend_document_seeds_for_question(
        question_id,
        k,
        mathe_mirror_client,
    )
    candidates = rank_expanded_candidates(
        seeds=seeds,
        k=k,
        recommendation_client=recommendation_client,
        question_metadata=dict(question_metadata),
        mathe_mirror_client=mathe_mirror_client,
    )
    return [str(candidate["material_id"]) for candidate in candidates]
