import os
from typing import Any

from dataset_recsys.mathe_recommenders.metadata_ocr import (
    MATHE_APPLICATION,
    MATHE_NEIGHBORS_PER_SEED,
    add_metadata_scores,
    recommend_pdf_seeds_for_question,
    resolve_db_material_ids,
    seed_redis_id,
)
from dataset_recsys.mathe_recommenders.question_embedding import (
    DEFAULT_MATHE_EMBEDDING_MODEL,
    encode_question,
    score_question_similarity_for_material_ids,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


def _env_weight(name: str, default: str) -> float:
    return min(max(float(os.getenv(name, default)), 0.0), 1.0)


MATHE_HYBRID_METADATA_WEIGHT = _env_weight("MATHE_HYBRID_METADATA_WEIGHT", "0.6")
MATHE_HYBRID_MATERIAL_OCR_WEIGHT = _env_weight(
    "MATHE_HYBRID_MATERIAL_OCR_WEIGHT",
    "0.25",
)
MATHE_HYBRID_QUESTION_WEIGHT = _env_weight("MATHE_HYBRID_QUESTION_WEIGHT", "0.15")
MATHE_HYBRID_QUESTION_CANDIDATES = int(
    os.getenv("MATHE_HYBRID_QUESTION_CANDIDATES", "3")
)


def _add_candidate(
    candidates: dict[str, dict[str, Any]],
    material_redis_id: str,
    material_id: Any = None,
    material_to_material_similarity: float = 0.0,
) -> None:
    """
    Add a material to the candidate pool.

    - Normalize/store the material_redis_id.
    - Create the candidate if it is not already in the pool.
    - Keep the max OCR similarity across all metadata seeds.
    """
    candidate = candidates.setdefault(
        str(material_redis_id).strip(),
        {
            "material_id": material_id,
            "material_redis_id": str(material_redis_id).strip(),
            "metadata_score": 0.0,
            "material_to_material_similarity": 0.0,
            "question_to_material_similarity": 0.0,
        },
    )
    candidate["material_to_material_similarity"] = max(
        float(candidate["material_to_material_similarity"]),
        float(material_to_material_similarity),
    )


def _add_question_similarities(
    candidates: dict[str, dict[str, Any]],
    question_embedding: list[float],
    embedding_client: EmbeddingClient,
) -> None:
    similarities = score_question_similarity_for_material_ids(
        question_embedding,
        list(candidates),
        embedding_client,
    )
    for material_redis_id, similarity in similarities.items():
        # Only candidates present in the embedding table get a question score;
        # missing embeddings keep the candidate default of 0.0.
        if material_redis_id in candidates:
            candidates[material_redis_id]["question_to_material_similarity"] = similarity


def _rank_candidates(
    candidates: dict[str, dict[str, Any]],
    k: int,
    metadata_weight: float,
    material_ocr_weight: float,
    question_weight: float,
) -> list[dict[str, Any]]:
    for candidate in candidates.values():
        candidate["final_score"] = (
            metadata_weight * float(candidate.get("metadata_score", 0.0))
            + material_ocr_weight
            * float(candidate.get("material_to_material_similarity", 0.0))
            + question_weight
            * float(candidate.get("question_to_material_similarity", 0.0))
        )

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate["final_score"],
            candidate["metadata_score"],
            candidate["material_to_material_similarity"],
            candidate["question_to_material_similarity"],
        ),
        reverse=True,
    )[:k]


def recommend_hybrid_candidates(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
    embedding_client: EmbeddingClient | None = None,
    metadata_weight: float = MATHE_HYBRID_METADATA_WEIGHT,
    material_ocr_weight: float = MATHE_HYBRID_MATERIAL_OCR_WEIGHT,
    question_weight: float = MATHE_HYBRID_QUESTION_WEIGHT,
    question_candidate_limit: int = MATHE_HYBRID_QUESTION_CANDIDATES,
    neighbors_per_seed: int = MATHE_NEIGHBORS_PER_SEED,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Recommend MathE PDF materials by merging the current metadata/OCR candidate
    source with a small set of question-text embedding candidates.
    """
    question = question.strip()
    if k <= 0 or not question:
        return []

    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return []

    embedding_client = embedding_client or EmbeddingClient()
    question_embedding = question_embedding or encode_question(question, embedding_model)
    metadata_seeds = recommend_pdf_seeds_for_question(
        question_id=question_id,
        k=k,
        mathe_mirror_client=mathe_mirror_client,
    )
    if len(metadata_seeds) >= k:
        candidates = {}
        for seed in metadata_seeds:
            _add_candidate(
                candidates,
                seed_redis_id(seed),
                material_id=seed.get("material_id"),
            )
            candidates[seed_redis_id(seed)].update(seed)
        _add_question_similarities(
            candidates,
            question_embedding,
            embedding_client,
        )
        return _rank_candidates(
            candidates,
            k,
            metadata_weight,
            0.0,
            question_weight,
        )

    candidates: dict[str, dict[str, Any]] = {}
    for seed in metadata_seeds:
        _add_candidate(
            candidates,
            seed_redis_id(seed),
            material_id=seed.get("material_id"),
        )
        neighbors = recommendation_client.get_recommendations_with_scores(
            application=MATHE_APPLICATION,
            entity_id=seed_redis_id(seed),
            limit=neighbors_per_seed,
        )
        for neighbor_id, material_to_material_similarity in neighbors:
            _add_candidate(
                candidates,
                str(neighbor_id).strip(),
                material_to_material_similarity=material_to_material_similarity,
            )

    question_matches = embedding_client.find_similar(
        application=MATHE_APPLICATION,
        query_embedding=question_embedding,
        top_k=question_candidate_limit,
        table=embedding_client.TABLE_MATHE,
    )
    for material_id, _similarity in question_matches:
        _add_candidate(candidates, str(material_id).strip())

    add_metadata_scores(
        candidates,
        dict(question_metadata),
        mathe_mirror_client,
    )
    _add_question_similarities(
        candidates,
        question_embedding,
        embedding_client,
    )

    return _rank_candidates(
        candidates,
        k,
        metadata_weight,
        material_ocr_weight,
        question_weight,
    )


def recommend_from_hybrid(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
    embedding_client: EmbeddingClient | None = None,
) -> list[str]:
    """
    Run hybrid MathE recommender.

    Flow:
    1. Retrieve metadata seeds for the question.
    2. If there are enough metadata seeds, rank only those seeds with
       metadata_score and question_to_material_similarity.
    3. Otherwise, build a candidate pool from metadata seeds, OCR neighbors,
       and a small set of question-nearest materials.
    4. Score the expanded pool with metadata_score,
       material_to_material_similarity, and question_to_material_similarity.
    5. Return the top-k candidates resolved to platform material IDs.
    """
    candidates = recommend_hybrid_candidates(
        question_id=question_id,
        question=question,
        k=k,
        mathe_mirror_client=mathe_mirror_client,
        recommendation_client=recommendation_client,
        embedding_client=embedding_client,
    )
    return resolve_db_material_ids(candidates, mathe_mirror_client)
