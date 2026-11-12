"""Build and serve versioned question-to-material rankings for MathE."""

import hashlib
import logging
import os
from collections.abc import Callable
from time import time
from uuid import uuid4

from qdrant_client import models

from dataset_recsys.embeddings import encode_texts
from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    MATHE_CURRICULAR_KEYWORD_WEIGHT,
    rank_scored_document_candidates,
    score_curricular_pool_candidates,
)
from dataset_recsys.mathe_recommenders.question_embedding import DEFAULT_MATHE_EMBEDDING_MODEL
from dataset_recsys.mathe_recommenders.video_pool_ranker import (
    MATHE_VIDEO_KEYWORD_WEIGHT,
    rank_scored_video_candidates,
    score_video_pool_candidates,
)
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.qdrant_client import QdrantStorageClient

logger = logging.getLogger(__name__)
BATCH_SIZE = 16
# Bump the first element when scoring, ordering, or stored data changes.
SCORING_VERSION = [
    1,
    DEFAULT_MATHE_EMBEDDING_MODEL,
    MATHE_CURRICULAR_KEYWORD_WEIGHT,
    MATHE_VIDEO_KEYWORD_WEIGHT,
]


def precomputed_rankings_enabled() -> bool:
    return os.getenv("MATHE_PRECOMPUTED_RANKINGS", "true").strip().lower() in {"1", "true", "yes"}


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def get_precomputed_ranking(
    qdrant: QdrantStorageClient,
    question_id: int,
    question: str,
    content_type: str,
    k: int,
    sync_status: dict,
) -> list[str] | None:
    """Return a fresh ranking, or None when the request needs live scoring."""
    if sync_status.get("sync_status") not in {"completed", "never_run"}:
        return None
    payload = qdrant.get_mathe_question_ranking(question_id, content_type)
    if not payload:
        return None
    if (
        payload.get("scoring_version") != SCORING_VERSION
        or payload.get("sync_generation") != sync_status.get("last_sync_started_at")
        or payload.get("question_hash") != _question_hash(question)
    ):
        return None
    ranking = payload.get(content_type)
    return ranking[:k] if isinstance(ranking, list) else None


def _candidate_scores(candidates: list[dict], similarity_field: str) -> list[dict]:
    return [
        {
            "material_id": row["material_id"],
            "similarity": row[similarity_field],
            "keyword_jaccard": row["keyword_jaccard"],
            "final_score": row["final_score"],
            "has_embedding": row["has_embedding"],
            "exclusion_reason": row["exclusion_reason"],
        }
        for row in candidates
    ]


def _make_point(
    question_id: int,
    question: str,
    embedding: list[float],
    mirror: MatheMirrorClient,
    qdrant: QdrantStorageClient,
    generation: str | None,
    built_at: float,
) -> models.PointStruct:
    documents = score_curricular_pool_candidates(
        question_id, question, mirror, qdrant, question_embedding=embedding
    )
    videos = score_video_pool_candidates(
        question_id, question, mirror, qdrant, question_embedding=embedding
    )
    ranked_documents = rank_scored_document_candidates(documents, len(documents))
    ranked_videos = rank_scored_video_candidates(videos, len(videos))
    return models.PointStruct(
        id=question_id,
        vector={},
        payload={
            "question_hash": _question_hash(question),
            "sync_generation": generation,
            "scoring_version": SCORING_VERSION,
            "built_at": built_at,
            "document_scores": _candidate_scores(documents, "question_to_material_similarity"),
            "video_scores": _candidate_scores(videos, "question_to_video_similarity"),
            "document": [row["material_id"] for row in ranked_documents],
            "video": [row["material_id"] for row in ranked_videos],
        },
    )


def _alias_target(qdrant: QdrantStorageClient) -> str | None:
    return next(
        (
            alias.collection_name
            for alias in qdrant.client.get_aliases().aliases
            if alias.alias_name == qdrant.MATHE_QUESTION_RANKINGS
        ),
        None,
    )


def _delete_inactive_snapshot(qdrant: QdrantStorageClient, collection: str) -> None:
    """Keep any collection that may still be active after a failed request."""
    try:
        if (
            collection.startswith(f"{qdrant.MATHE_QUESTION_RANKINGS}_")
            and _alias_target(qdrant) != collection
        ):
            qdrant.client.delete_collection(collection)
    except Exception:
        logger.warning("Could not remove inactive MathE ranking snapshot %s", collection, exc_info=True)


def refresh_question_rankings(
    mirror: MatheMirrorClient,
    qdrant: QdrantStorageClient,
    read_sync_status: Callable[[], dict],
    *,
    sync_generation: str | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> int:
    """Publish a complete snapshot after building it in a separate collection."""
    state = read_sync_status()
    phase = state.get("sync_status")
    generation = state.get("last_sync_started_at")
    if sync_generation is not None:
        if phase != "running" or generation != sync_generation:
            raise RuntimeError("The MathE sync changed before ranking refresh could start")
    elif phase not in {"completed", "never_run"}:
        raise RuntimeError("Wait for a successful MathE sync before refreshing question rankings")

    questions = [row for row in mirror.get_recommendation_questions() if row["question"].strip()]
    if not questions:
        raise RuntimeError("No MathE questions were found")
    alias = qdrant.MATHE_QUESTION_RANKINGS
    collection = f"{alias}_{uuid4().hex}"
    built_at = time()
    try:
        if not qdrant.client.create_collection(collection, vectors_config={}):
            raise RuntimeError("Qdrant did not create the MathE ranking snapshot")
        for offset in range(0, len(questions), BATCH_SIZE):
            batch = questions[offset : offset + BATCH_SIZE]
            texts = [row["question"].strip() for row in batch]
            embeddings = encode_texts(texts, DEFAULT_MATHE_EMBEDDING_MODEL)
            points = [
                _make_point(row["question_id"], text, vector.tolist(), mirror, qdrant, generation, built_at)
                for row, text, vector in zip(batch, texts, embeddings, strict=True)
            ]
            qdrant.client.upsert(collection, points, wait=True)
            if heartbeat is not None:
                heartbeat()

        current = read_sync_status()
        if (current.get("sync_status"), current.get("last_sync_started_at")) != (phase, generation):
            raise RuntimeError("The MathE sync changed while question rankings were being built")

        previous = _alias_target(qdrant)
        operations = []
        if previous is not None:
            operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        operations.append(models.CreateAliasOperation(
            create_alias=models.CreateAlias(alias_name=alias, collection_name=collection)
        ))
        if not qdrant.client.update_collection_aliases(operations):
            raise RuntimeError("Qdrant did not publish the MathE ranking snapshot")
    except Exception:
        _delete_inactive_snapshot(qdrant, collection)
        raise

    if previous is not None:
        _delete_inactive_snapshot(qdrant, previous)
    logger.info("Published precomputed MathE rankings for %d questions", len(questions))
    return len(questions)
