import os

from dataset_recsys.mathe_recommenders.seed_scoring import score_pdf_seed_candidates
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


MATHE_APPLICATION = "mathe"
MATHE_EMBEDDING_WEIGHT = min(
    max(float(os.getenv("MATHE_EMBEDDING_WEIGHT", "0.5")), 0.0),
    1.0,
)
MATHE_NEIGHBORS_PER_SEED = int(os.getenv("MATHE_NEIGHBORS_PER_SEED", "20"))


def seed_redis_id(seed: dict) -> str:
    return str(seed["material_redis_id"]).strip()


def _score_candidate_metadata(
    candidates: dict[str, dict],
    question_metadata: dict,
    mathe_mirror_client: MatheMirrorClient,
) -> None:
    material_metadata = mathe_mirror_client.get_pdf_material_metadata_by_redis_ids(
        list(candidates.keys())
    )
    scored_metadata = score_pdf_seed_candidates(
        question_metadata,
        material_metadata,
    )

    for metadata in scored_metadata:
        material_redis_id = str(metadata["material_redis_id"]).strip()
        candidate = candidates.get(material_redis_id)
        if candidate:
            candidate.update(metadata)


def rank_expanded_candidates(
    seeds: list[dict],
    k: int,
    recommendation_client: RecommendationClient,
    question_metadata: dict | None = None,
    mathe_mirror_client: MatheMirrorClient | None = None,
    embedding_weight: float = MATHE_EMBEDDING_WEIGHT,
    neighbors_per_seed: int = MATHE_NEIGHBORS_PER_SEED,
) -> list[dict]:
    """
    Rank MathE PDF recommendations from metadata seeds, expanding with OCR
    embedding neighbors only when fewer than k metadata seeds exist.
    """
    seed_by_entity_id = {
        seed_redis_id(seed): seed
        for seed in seeds
    }

    if k <= 0:
        return []

    if len(seed_by_entity_id) >= k:
        return [
            {
                **seed,
                "material_redis_id": entity_id,
                "embedding_score": 0.0,
                "final_score": float(seed.get("metadata_score", 0.0)),
            }
            for entity_id, seed in list(seed_by_entity_id.items())[:k]
        ]

    candidates: dict[str, dict] = {}
    for entity_id, seed in seed_by_entity_id.items():
        metadata_score = float(seed.get("metadata_score", 0.0))
        candidates[entity_id] = {
            "material_id": seed.get("material_id"),
            "material_redis_id": entity_id,
            "metadata_score": metadata_score,
            "embedding_score": 1.0,
        }

    for seed_entity_id in seed_by_entity_id:
        # Failed-OCR materials simply have no Redis recommendation key. They stay
        # in the candidate pool as metadata seeds, but do not expand.
        neighbors = recommendation_client.get_recommendations_with_scores(
            application=MATHE_APPLICATION,
            entity_id=seed_entity_id,
            limit=neighbors_per_seed,
        )

        for neighbor_id, embedding_score in neighbors:
            neighbor_id = str(neighbor_id).strip()
            candidate = candidates.setdefault(
                neighbor_id,
                {
                    "material_id": None,
                    "material_redis_id": neighbor_id,
                    "metadata_score": 0.0,
                    "embedding_score": 0.0,
                },
            )
            candidate["embedding_score"] = max(
                float(candidate["embedding_score"]),
                embedding_score,
            )

    if question_metadata and mathe_mirror_client:
        _score_candidate_metadata(
            candidates,
            question_metadata,
            mathe_mirror_client,
        )

    for candidate in candidates.values():
        candidate["final_score"] = (
            embedding_weight * float(candidate["embedding_score"])
            + (1.0 - embedding_weight) * float(candidate["metadata_score"])
        )

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate["final_score"],
            candidate["metadata_score"],
            candidate["embedding_score"],
        ),
        reverse=True,
    )
    return ranked_candidates[:k]


def resolve_db_material_ids(
    candidates: list[dict],
    mathe_mirror_client: MatheMirrorClient,
) -> list[str]:
    unresolved_redis_ids = [
        candidate["material_redis_id"]
        for candidate in candidates
        if (
            candidate.get("material_redis_id")
            and candidate.get("material_id") is None
        )
    ]
    details_by_redis_id = {
        str(material["material_redis_id"]).strip(): material
        for material in mathe_mirror_client.get_pdf_material_details(
            unresolved_redis_ids
        )
    }

    db_material_ids: list[str] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        material_id = candidate.get("material_id")
        if material_id is None:
            material = details_by_redis_id.get(candidate.get("material_redis_id"))
            material_id = material.get("material_id") if material else None

        if material_id is None:
            continue

        material_id = str(material_id)
        if material_id in seen_ids:
            continue

        db_material_ids.append(material_id)
        seen_ids.add(material_id)

    return db_material_ids


def recommend_pdf_seeds_for_question(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
) -> list[dict]:
    """Return top-k metadata-scored PDF seed materials for a question."""
    if k <= 0:
        return []

    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return []

    seed_candidates = mathe_mirror_client.get_pdf_seed_candidates(question_id)
    scored_candidates = score_pdf_seed_candidates(
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

    seeds = recommend_pdf_seeds_for_question(
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
    return resolve_db_material_ids(candidates, mathe_mirror_client)
