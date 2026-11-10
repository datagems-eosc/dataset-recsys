import logging
import os
from typing import Any

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.mathe_recommenders.question_embedding import (
    DEFAULT_MATHE_EMBEDDING_MODEL,
    encode_question,
    score_question_similarity_for_material_ids,
)
from dataset_recsys.mathe_recommenders.seed_scoring import compute_keyword_jaccard
from dataset_recsys.storage.qdrant_client import QdrantStorageClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


def _env_weight(name: str, default: str) -> float:
    return min(max(float(os.getenv(name, default)), 0.0), 1.0)


MATHE_CURRICULAR_KEYWORD_WEIGHT = _env_weight(
    "MATHE_CURRICULAR_KEYWORD_WEIGHT",
    "0.6",
)

logger = logging.getLogger(__name__)

# Why a pool material with a final score of zero is left out of the ranking.
ZERO_SCORE_NO_EMBEDDING = "zero_score_no_embedding"
ZERO_SCORE_WITH_EMBEDDING = "zero_score_with_embedding"


def _score_candidates(
    candidates: list[dict[str, Any]],
    keyword_weight: float,
) -> None:
    """Set ``final_score`` and ``exclusion_reason`` on each pool candidate.

    Keyword overlap lies in [0, 1]. A cosine similarity can be negative, which
    carries no meaning as a relevance contribution, so it counts as 0.0 in the
    weighted sum. The reported ``question_to_material_similarity`` keeps the
    measured value.

    A candidate whose final score is zero has no ranking signal: any position
    given to it would be arbitrary. It is marked with the reason so it can be
    left out of the ranking and reported.
    """
    question_weight = 1.0 - keyword_weight
    for candidate in candidates:
        similarity = float(candidate.get("question_to_material_similarity", 0.0))
        candidate["final_score"] = (
            keyword_weight * float(candidate.get("keyword_jaccard", 0.0))
            + question_weight * max(similarity, 0.0)
        )
        if candidate["final_score"] > 0.0:
            candidate["exclusion_reason"] = None
        elif candidate.get("has_embedding"):
            candidate["exclusion_reason"] = ZERO_SCORE_WITH_EMBEDDING
        else:
            candidate["exclusion_reason"] = ZERO_SCORE_NO_EMBEDDING


def _rank_candidates(
    candidates: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Return the top-k scored candidates, leaving out zero-score ones."""
    ranked = [
        candidate
        for candidate in candidates
        if candidate["exclusion_reason"] is None
    ]
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate["final_score"],
            candidate["question_to_material_similarity"],
            candidate["keyword_jaccard"],
        ),
        reverse=True,
    )[:k]


def score_curricular_pool_candidates(
    question_id: int,
    question: str,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    keyword_weight: float = MATHE_CURRICULAR_KEYWORD_WEIGHT,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Score every document teaching material in the question's topic/subtopic pool.

    The pool is a hard curricular filter. Materials outside the same
    topic/subtopic are never added. Scoring uses only keyword overlap and
    question-to-material similarity.

    The whole pool is returned in pool order (ascending material ID), including
    the materials that the ranking leaves out. Each candidate carries
    ``has_embedding``, ``final_score`` and ``exclusion_reason``, which is
    ``None`` for a ranked material and otherwise names why its score is zero.
    Use this to report zero-score materials; use
    ``rank_curricular_pool_candidates`` for recommendations.
    """
    question = question.strip()
    if not question:
        return []

    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return []

    pool = mathe_mirror_client.get_document_materials_for_question_topic_subtopic(
        question_id
    )
    if not pool:
        return []

    candidates_by_material_id: dict[str, dict[str, Any]] = {}
    for material in pool:
        material_index_id = str(material["material_id"]).strip()
        keyword_jaccard = compute_keyword_jaccard(
            question_metadata.get("keywords"),
            material.get("keywords"),
        )
        candidates_by_material_id[material_index_id] = {
            **material,
            "material_id": material_index_id,
            "keyword_jaccard": keyword_jaccard,
            "metadata_score": keyword_jaccard,
            "question_to_material_similarity": 0.0,
            "has_embedding": False,
        }

    qdrant_client = qdrant_client or QdrantStorageClient()
    question_embedding = question_embedding or encode_question(
        question,
        embedding_model,
    )
    question_similarities = score_question_similarity_for_material_ids(
        question_embedding,
        list(candidates_by_material_id),
        qdrant_client,
        application=MatheApplication.DOCUMENTS,
    )
    for material_index_id, similarity in question_similarities.items():
        if material_index_id in candidates_by_material_id:
            candidate = candidates_by_material_id[material_index_id]
            candidate["question_to_material_similarity"] = similarity
            candidate["has_embedding"] = True

    candidates = list(candidates_by_material_id.values())
    _score_candidates(candidates, keyword_weight)
    return candidates


def rank_curricular_pool_candidates(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    keyword_weight: float = MATHE_CURRICULAR_KEYWORD_WEIGHT,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank document teaching materials restricted to the question's same topic/subtopic pool.

    Materials with a final score of zero are left out, so the result can be
    shorter than ``k``. They are logged, and
    ``score_curricular_pool_candidates`` returns them with the reason.
    """
    if k <= 0:
        return []

    candidates = score_curricular_pool_candidates(
        question_id=question_id,
        question=question,
        mathe_mirror_client=mathe_mirror_client,
        qdrant_client=qdrant_client,
        keyword_weight=keyword_weight,
        embedding_model=embedding_model,
        question_embedding=question_embedding,
    )
    _log_zero_score_candidates("document", question_id, candidates)
    return _rank_candidates(candidates, k)


def _log_zero_score_candidates(
    content_kind: str,
    question_id: int,
    candidates: list[dict[str, Any]],
) -> None:
    excluded = {
        candidate["material_id"]: candidate["exclusion_reason"]
        for candidate in candidates
        if candidate["exclusion_reason"] is not None
    }
    if excluded:
        logger.info(
            "MathE %s ranker left out %d of %d pool materials with a zero "
            "score for question %s: %s",
            content_kind,
            len(excluded),
            len(candidates),
            question_id,
            excluded,
        )


def recommend_from_curricular_pool(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    question_embedding: list[float] | None = None,
) -> list[str]:
    """Return top-k document material IDs from the same topic/subtopic pool."""
    candidates = rank_curricular_pool_candidates(
        question_id=question_id,
        question=question,
        k=k,
        mathe_mirror_client=mathe_mirror_client,
        qdrant_client=qdrant_client,
        question_embedding=question_embedding,
    )
    return [str(candidate["material_id"]) for candidate in candidates]
