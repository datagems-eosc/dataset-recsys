import os
from typing import Any

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    ZERO_SCORE_NO_EMBEDDING,
    ZERO_SCORE_WITH_EMBEDDING,
    _log_zero_score_candidates,
)
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


# Same default as the document ranker's MATHE_CURRICULAR_KEYWORD_WEIGHT. It has
# not been validated on video relevance labels.
MATHE_VIDEO_KEYWORD_WEIGHT = _env_weight(
    "MATHE_VIDEO_KEYWORD_WEIGHT",
    "0.6",
)


def score_video_pool_candidates(
    question_id: int,
    question: str,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    keyword_weight: float = MATHE_VIDEO_KEYWORD_WEIGHT,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Score every video in the question's topic/subtopic pool.

    The pool is a hard curricular filter. Each video is scored like a document
    in the curricular pool ranker:

        final_score = keyword_weight * keyword_jaccard
                      + (1 - keyword_weight) * max(question_to_video_similarity, 0)

    A negative cosine similarity carries no meaning as a relevance
    contribution, so it counts as 0.0 in the sum; the reported
    ``question_to_video_similarity`` keeps the measured value. A video without
    a published ``mathe_videos`` embedding has a similarity of 0.0.

    The whole pool is returned in pool order (ascending platform material ID),
    including the videos that the ranking leaves out. Each candidate carries
    ``has_embedding``, ``final_score`` and ``exclusion_reason``, which is
    ``None`` for a ranked video and otherwise names why its score is zero.
    """
    question = question.strip()
    if not question:
        return []

    pool = mathe_mirror_client.get_videos_for_question(question_id)
    if not pool:
        return []

    question_metadata = mathe_mirror_client.get_question_metadata(question_id) or {}
    question_keywords = question_metadata.get("keywords")

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for video in pool:
        material_id = str(video["material_id"]).strip()
        candidates_by_id[material_id] = {
            **video,
            "material_id": material_id,
            "keyword_jaccard": compute_keyword_jaccard(
                question_keywords,
                video.get("keywords"),
            ),
            "question_to_video_similarity": 0.0,
            "has_embedding": False,
        }

    qdrant_client = qdrant_client or QdrantStorageClient()
    question_embedding = question_embedding or encode_question(
        question,
        embedding_model,
    )

    # Pass both raw ID and ID with extension to Qdrant lookup
    lookup_ids = []
    for material_id in candidates_by_id:
        lookup_ids.extend([material_id, f"{material_id}.txt"])

    similarities_raw = score_question_similarity_for_material_ids(
        question_embedding,
        lookup_ids,
        qdrant_client,
        application=MatheApplication.VIDEOS,
    )

    question_weight = 1.0 - keyword_weight
    for material_id, candidate in candidates_by_id.items():
        # Map results back to the clean platform material_id
        candidate["has_embedding"] = (
            material_id in similarities_raw
            or f"{material_id}.txt" in similarities_raw
        )
        candidate["question_to_video_similarity"] = similarities_raw.get(
            material_id,
            similarities_raw.get(f"{material_id}.txt", 0.0),
        )
        candidate["final_score"] = (
            keyword_weight * candidate["keyword_jaccard"]
            + question_weight * max(candidate["question_to_video_similarity"], 0.0)
        )
        if candidate["final_score"] > 0.0:
            candidate["exclusion_reason"] = None
        elif candidate["has_embedding"]:
            candidate["exclusion_reason"] = ZERO_SCORE_WITH_EMBEDDING
        else:
            candidate["exclusion_reason"] = ZERO_SCORE_NO_EMBEDDING

    return list(candidates_by_id.values())


def rank_video_pool_candidates(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    keyword_weight: float = MATHE_VIDEO_KEYWORD_WEIGHT,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Rank the videos in the question's topic/subtopic pool.

    Videos with a final score of zero have no ranking signal and are left out,
    so the result can be shorter than ``k``. They are logged, and
    ``score_video_pool_candidates`` returns them with the reason. Ties keep the
    pool order, which is ascending platform material ID.
    """
    if k <= 0:
        return []

    candidates = score_video_pool_candidates(
        question_id=question_id,
        question=question,
        mathe_mirror_client=mathe_mirror_client,
        qdrant_client=qdrant_client,
        keyword_weight=keyword_weight,
        embedding_model=embedding_model,
        question_embedding=question_embedding,
    )
    _log_zero_score_candidates("video", question_id, candidates)

    candidate_order = {
        candidate["material_id"]: index
        for index, candidate in enumerate(candidates)
    }
    ranked = [
        candidate
        for candidate in candidates
        if candidate["exclusion_reason"] is None
    ]
    return sorted(
        ranked,
        key=lambda candidate: (
            -candidate["final_score"],
            -candidate["question_to_video_similarity"],
            -candidate["keyword_jaccard"],
            candidate_order[candidate["material_id"]],
        ),
    )[:k]


def recommend_videos_for_question(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    question_embedding: list[float] | None = None,
) -> list[str]:
    """Return ranked MathE video platform IDs for a question."""
    candidates = rank_video_pool_candidates(
        question_id=question_id,
        question=question,
        k=k,
        mathe_mirror_client=mathe_mirror_client,
        qdrant_client=qdrant_client,
        question_embedding=question_embedding,
    )
    return [candidate["material_id"] for candidate in candidates]


__all__ = ["recommend_videos_for_question"]
