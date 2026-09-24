from typing import Any

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.mathe_recommenders.question_embedding import (
    DEFAULT_MATHE_EMBEDDING_MODEL,
    encode_question,
    score_question_similarity_for_material_ids,
)
from dataset_recsys.storage.qdrant_client import QdrantStorageClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


def rank_video_pool_candidates(
    question_id: int,
    question: str,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient | None = None,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    question_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Rank embedded videos from the question's topic/subtopic pool."""
    question = question.strip()
    if k <= 0 or not question:
        return []

    pool = mathe_mirror_client.get_videos_for_question(question_id)
    if not pool:
        return []

    candidates_by_id = {
        str(video["material_id"]).strip(): {
            **video,
            "material_id": str(video["material_id"]).strip(),
        }
        for video in pool
    }
    candidate_order = {
        material_id: index
        for index, material_id in enumerate(candidates_by_id)
    }

    qdrant_client = qdrant_client or QdrantStorageClient()
    question_embedding = question_embedding or encode_question(
        question,
        embedding_model,
    )
    similarities = score_question_similarity_for_material_ids(
        question_embedding,
        list(candidates_by_id),
        qdrant_client,
        application=MatheApplication.VIDEOS,
    )

    ranked_ids = sorted(
        similarities,
        key=lambda material_id: (
            -similarities[material_id],
            candidate_order[material_id],
        ),
    )[:k]
    return [
        {
            **candidates_by_id[material_id],
            "question_to_video_similarity": similarities[material_id],
        }
        for material_id in ranked_ids
    ]


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
