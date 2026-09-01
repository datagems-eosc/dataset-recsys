import os
from typing import Any

from dataset_recsys.embeddings import encode_texts
from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.mathe_recommenders.seed_scoring import (
    score_document_seed_candidates,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


DEFAULT_MATHE_EMBEDDING_MODEL = os.getenv(
    "MATHE_EMBEDDING_MODEL",
    "BAAI/bge-m3",
)
MATHE_QUESTION_EMBEDDING_WEIGHT = min(
    max(float(os.getenv("MATHE_QUESTION_EMBEDDING_WEIGHT", "0.5")), 0.0),
    1.0,
)
MATHE_QUESTION_EMBEDDING_CANDIDATES = int(
    os.getenv("MATHE_QUESTION_EMBEDDING_CANDIDATES", "50")
)


def encode_question(
    question: str,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
) -> list[float]:
    question_embedding = encode_texts(
        [question],
        model_name=embedding_model,
    )[0]
    return (
        question_embedding.tolist()
        if hasattr(question_embedding, "tolist")
        else question_embedding
    )


def score_question_similarity_for_material_ids(
    question_embedding: list[float],
    material_ids: list[str],
    embedding_client: EmbeddingClient,
    application: MatheApplication,
) -> dict[str, float]:
    """
    Score requested materials against the question embedding.

    Materials without a stored MathE embedding are absent from the returned
    mapping; callers decide the default score for those candidates.
    """
    similarities = embedding_client.find_similar_by_ids(
        application=application,
        query_embedding=question_embedding,
        entity_ids=material_ids,
        table=embedding_client.TABLE_MATHE,
    )
    return {
        str(material_id).strip(): float(similarity)
        for material_id, similarity in similarities
    }


def recommend_from_question_embedding(
    question: str,
    k: int,
    question_metadata: dict[str, Any],
    mathe_mirror_client: MatheMirrorClient,
    embedding_client: EmbeddingClient | None = None,
    embedding_model: str = DEFAULT_MATHE_EMBEDDING_MODEL,
    similarity_weight: float = MATHE_QUESTION_EMBEDDING_WEIGHT,
    candidate_limit: int = MATHE_QUESTION_EMBEDDING_CANDIDATES,
) -> list[dict[str, Any]]:
    """Recommend MathE materials by blending question-vector and metadata scores."""
    question = question.strip()
    if not question or k <= 0:
        return []

    embedding_client = embedding_client or EmbeddingClient()
    top_k = max(k, candidate_limit)
    question_embedding = encode_question(question, embedding_model)

    results = embedding_client.find_similar(
        application=MatheApplication.DOCUMENTS,
        query_embedding=question_embedding,
        top_k=top_k,
        table=embedding_client.TABLE_MATHE,
    )

    candidates = {
        str(material_id).strip(): {
            "material_id": str(material_id).strip(),
            "question_to_material_similarity": float(similarity),
            "metadata_score": 0.0,
        }
        for material_id, similarity in results
    }

    material_metadata = mathe_mirror_client.get_document_material_metadata_by_ids(
        list(candidates)
    )
    scored_metadata = score_document_seed_candidates(
        question_metadata,
        material_metadata,
    )

    for metadata in scored_metadata:
        material_id = str(metadata["material_id"]).strip()
        candidate = candidates.get(material_id)
        if candidate:
            candidate.update(
                {
                    "keyword_jaccard": metadata.get("keyword_jaccard", 0.0),
                    "same_subtopic": metadata.get("same_subtopic", 0),
                    "same_topic": metadata.get("same_topic", 0),
                    "metadata_score": metadata.get("metadata_score", 0.0),
                }
            )

    for candidate in candidates.values():
        candidate["total_score"] = (
            similarity_weight * candidate["question_to_material_similarity"]
            + (1.0 - similarity_weight) * candidate["metadata_score"]
        )

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate["total_score"],
            candidate["metadata_score"],
            candidate["question_to_material_similarity"],
        ),
        reverse=True,
    )[:k]
