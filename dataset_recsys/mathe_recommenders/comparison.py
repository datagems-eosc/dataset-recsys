from dataset_recsys.mathe_recommenders.metadata_ocr import (
    recommend_pdf_seeds_for_question,
    rank_expanded_candidates,
    seed_redis_id,
)
from dataset_recsys.mathe_recommenders.question_embedding import (
    recommend_from_question_embedding,
)
from dataset_recsys.mathe_recommenders.hybrid import recommend_hybrid_candidates
from dataset_recsys.mathe_recommenders.popular_seed import recommend_from_popular_seed
from dataset_recsys.mathe_recommenders.seed_scoring import score_pdf_seed_candidates
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


METADATA_SCORE_KEYS = (
    "keyword_jaccard",
    "same_subtopic",
    "same_topic",
    "metadata_score",
)


def _details_by_material_id(
    material_ids: list[str],
    mathe_mirror_client: MatheMirrorClient,
) -> dict[str, dict]:
    details = mathe_mirror_client.get_pdf_material_details(material_ids)
    return {
        str(material["material_redis_id"]).strip(): dict(material)
        for material in details
    }


def _enrich_recommendations(
    material_ids: list[str],
    details_by_id: dict[str, dict],
    scores_by_id: dict[str, dict] | None = None,
) -> list[dict]:
    enriched = []
    scores_by_id = scores_by_id or {}

    for rank, material_id in enumerate(material_ids, start=1):
        material_id = str(material_id).strip()
        material = details_by_id.get(material_id, {})
        enriched.append(
            {
                "rank": rank,
                "material_id": material.get("material_id"),
                "material_redis_id": material_id,
                "scores": scores_by_id.get(material_id, {}),
                "title": material.get("title"),
                "author": material.get("author"),
                "description": material.get("description"),
                "file_name": material.get("file_name"),
                "topics": material.get("topics", []),
                "subtopics": material.get("subtopics", []),
                "keywords": material.get("keywords", []),
            }
        )

    return enriched


def _metadata_scores_for_materials(
    material_ids: list[str],
    question_metadata: dict,
    mathe_mirror_client: MatheMirrorClient,
) -> dict[str, dict]:
    if not material_ids:
        return {}

    material_metadata = mathe_mirror_client.get_pdf_material_metadata_by_redis_ids(
        material_ids
    )
    scored_materials = score_pdf_seed_candidates(
        question_metadata,
        material_metadata,
    )

    return {
        str(material["material_redis_id"]).strip(): {
            key: material[key]
            for key in METADATA_SCORE_KEYS
            if key in material
        }
        for material in scored_materials
    }


def _metadata_score_fields(candidate: dict) -> dict:
    return {
        key: candidate[key]
        for key in METADATA_SCORE_KEYS
        if key in candidate
    }


def compare_question_recommenders(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
    question_text: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> dict:
    """Build an internal comparison report for MathE recommender strategies."""
    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return {
            "question": None,
            "strategies": {},
        }

    metadata_seeds = recommend_pdf_seeds_for_question(
        question_id,
        k,
        mathe_mirror_client,
    )
    metadata_candidates = rank_expanded_candidates(
        seeds=metadata_seeds,
        k=k,
        recommendation_client=recommendation_client,
        question_metadata=dict(question_metadata),
        mathe_mirror_client=mathe_mirror_client,
    )
    metadata_ids = [
        seed_redis_id(candidate)
        for candidate in metadata_candidates
    ]
    metadata_scores = {}
    for candidate in metadata_candidates:
        material_id = seed_redis_id(candidate)
        scores = _metadata_score_fields(candidate)
        if "material_to_material_similarity" in candidate:
            scores["material_to_material_similarity"] = candidate[
                "material_to_material_similarity"
            ]
        if "final_score" in candidate:
            scores["total_score"] = candidate["final_score"]
        metadata_scores[material_id] = scores

    popular_ids = recommend_from_popular_seed(
        question_id=question_id,
        k=k,
        mathe_mirror_client=mathe_mirror_client,
        recommendation_client=recommendation_client,
    )
    popular_seed = mathe_mirror_client.get_material_by_question_id(question_id)
    popular_seed_id = (
        seed_redis_id(popular_seed)
        if popular_seed
        else None
    )
    question_embedding_candidates = (
        recommend_from_question_embedding(
            question_text,
            k,
            embedding_client=embedding_client,
            question_metadata=dict(question_metadata),
            mathe_mirror_client=mathe_mirror_client,
        )
        if question_text
        else []
    )
    question_embedding_ids = [
        str(candidate["material_redis_id"]).strip()
        for candidate in question_embedding_candidates
    ]
    question_embedding_scores = {}
    for candidate in question_embedding_candidates:
        material_id = seed_redis_id(candidate)
        scores = _metadata_score_fields(candidate)
        scores["question_to_material_similarity"] = candidate.get(
            "question_to_material_similarity"
        )
        scores["total_score"] = candidate.get("total_score")
        question_embedding_scores[material_id] = scores

    hybrid_candidates = (
        recommend_hybrid_candidates(
            question_id=question_id,
            question=question_text,
            k=k,
            mathe_mirror_client=mathe_mirror_client,
            recommendation_client=recommendation_client,
            embedding_client=embedding_client,
        )
        if question_text
        else []
    )
    hybrid_ids = [
        str(candidate["material_redis_id"]).strip()
        for candidate in hybrid_candidates
    ]
    hybrid_scores = {}
    for candidate in hybrid_candidates:
        material_id = seed_redis_id(candidate)
        scores = _metadata_score_fields(candidate)
        scores["material_to_material_similarity"] = candidate.get(
            "material_to_material_similarity"
        )
        scores["question_to_material_similarity"] = candidate.get(
            "question_to_material_similarity"
        )
        scores["total_score"] = candidate.get("final_score")
        hybrid_scores[material_id] = scores

    popular_material_ids = popular_ids + ([popular_seed_id] if popular_seed_id else [])
    popular_scores = _metadata_scores_for_materials(
        popular_material_ids,
        dict(question_metadata),
        mathe_mirror_client,
    )
    material_ids_for_details = list(
        dict.fromkeys(
            metadata_ids
            + popular_ids
            + question_embedding_ids
            + hybrid_ids
            + ([popular_seed_id] if popular_seed_id else [])
        )
    )
    details_by_id = _details_by_material_id(
        material_ids_for_details,
        mathe_mirror_client,
    )

    question_details = {
        "question_id": question_metadata["question_id"],
        "topic": question_metadata.get("topic"),
        "subtopic": question_metadata.get("subtopic"),
        "keywords": question_metadata.get("keywords", []),
    }

    return {
        "question": question_details,
        "strategies": {
            "metadata": {
                "description": "Current production flow: metadata seed candidates, expanded with Redis OCR neighbors only when needed.",
                "recommendations": _enrich_recommendations(
                    metadata_ids,
                    details_by_id,
                    metadata_scores,
                ),
            },
            "popular_seed": {
                "description": "Previous experimental flow: most-clicked PDF seed under the question topic, then Redis OCR neighbors.",
                "seed": (
                    _enrich_recommendations(
                        [popular_seed_id],
                        details_by_id,
                        popular_scores,
                    )[0]
                    if popular_seed_id
                    else None
                ),
                "recommendations": _enrich_recommendations(
                    popular_ids,
                    details_by_id,
                    popular_scores,
                ),
            },
            "question_embedding": {
                "description": "Experimental flow: embed provided question text, then query MathE material embeddings in pgvector.",
                "recommendations": _enrich_recommendations(
                    question_embedding_ids,
                    details_by_id,
                    question_embedding_scores,
                ),
                "available": bool(question_text),
            },
            "hybrid": {
                "description": "Experimental flow: merge metadata/OCR candidates with a small set of question-text embedding candidates, then rerank with metadata, material OCR, and question similarity scores.",
                "recommendations": _enrich_recommendations(
                    hybrid_ids,
                    details_by_id,
                    hybrid_scores,
                ),
                "available": bool(question_text),
            },
        },
    }
