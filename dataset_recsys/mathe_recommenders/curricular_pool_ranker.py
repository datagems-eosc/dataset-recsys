import os
from typing import Any

from dataset_recsys.mathe_recommenders.metadata_ocr import resolve_db_material_ids
from dataset_recsys.mathe_recommenders.question_embedding import (
    DEFAULT_MATHE_EMBEDDING_MODEL,
    encode_question,
    score_question_similarity_for_material_ids,
)
from dataset_recsys.mathe_recommenders.seed_scoring import compute_keyword_jaccard
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


def _env_weight(name: str, default: str) -> float:
    return min(max(float(os.getenv(name, default)), 0.0), 1.0)


MATHE_CURRICULAR_KEYWORD_WEIGHT = _env_weight(
    "MATHE_CURRICULAR_KEYWORD_WEIGHT",
    "0.6",
)


def _rank_candidates(
    candidates: list[dict[str, Any]],
    k: int,
    keyword_weight: float,
) -> list[dict[str, Any]]:
    question_weight = 1.0 - keyword_weight
    for candidate in candidates:
        candidate["final_score"] = (
            keyword_weight * float(candidate.get("keyword_jaccard", 0.0))
            + question_weight
            * float(candidate.get("question_to_material_similarity", 0.0))
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["final_score"],
            candidate["question_to_material_similarity"],
            candidate["keyword_jaccard"],
        ),
        reverse=True,
    )[:k]


def rank_curricular_pool_candidates(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    embedding_client: EmbeddingClient | None = None,
    keyword_weight: float = MATHE_CURRICULAR_KEYWORD_WEIGHT,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank document teaching materials restricted to the question's same topic/subtopic pool.

    The pool is a hard curricular filter. Materials outside the same
    topic/subtopic are never added. Ranking uses only keyword overlap and
    question-to-material similarity.
    """
    question = question.strip()
    if k <= 0 or not question:
        return []

    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return []

    pool = mathe_mirror_client.get_document_materials_for_question_topic_subtopic(
        question_id
    )
    if not pool:
        return []

    candidates_by_redis_id: dict[str, dict[str, Any]] = {}
    for material in pool:
        material_redis_id = str(material["material_redis_id"]).strip()
        keyword_jaccard = compute_keyword_jaccard(
            question_metadata.get("keywords"),
            material.get("keywords"),
        )
        candidates_by_redis_id[material_redis_id] = {
            **material,
            "material_redis_id": material_redis_id,
            "keyword_jaccard": keyword_jaccard,
            "metadata_score": keyword_jaccard,
            "question_to_material_similarity": 0.0,
        }

    embedding_client = embedding_client or EmbeddingClient()
    question_embedding = question_embedding or encode_question(
        question,
        embedding_model,
    )
    question_similarities = score_question_similarity_for_material_ids(
        question_embedding,
        list(candidates_by_redis_id),
        embedding_client,
    )
    for material_redis_id, similarity in question_similarities.items():
        if material_redis_id in candidates_by_redis_id:
            candidates_by_redis_id[material_redis_id][
                "question_to_material_similarity"
            ] = similarity

    return _rank_candidates(
        list(candidates_by_redis_id.values()),
        k,
        keyword_weight,
    )


def recommend_from_curricular_pool(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    embedding_client: EmbeddingClient | None = None,
    question_embedding: list[float] | None = None,
) -> list[str]:
    """Return top-k document material IDs from the same topic/subtopic pool."""
    candidates = rank_curricular_pool_candidates(
        question_id=question_id,
        question=question,
        k=k,
        mathe_mirror_client=mathe_mirror_client,
        embedding_client=embedding_client,
        question_embedding=question_embedding,
    )
    return resolve_db_material_ids(candidates, mathe_mirror_client)
