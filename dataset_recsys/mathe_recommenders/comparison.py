from dataset_recsys.mathe_recommenders.metadata_ocr import (
    recommend_pdf_seeds_for_question,
    rank_expanded_candidates,
    seed_redis_id,
)
from dataset_recsys.mathe_recommenders.popular_seed import recommend_from_popular_seed
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


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


def compare_question_recommenders(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
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
    metadata_scores = {
        seed_redis_id(candidate): {
            key: candidate[key]
            for key in (
                "keyword_jaccard",
                "same_subtopic",
                "same_topic",
                "metadata_score",
                "embedding_score",
                "final_score",
            )
            if key in candidate
        }
        for candidate in metadata_candidates
    }

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
    material_ids_for_details = list(
        dict.fromkeys(
            metadata_ids
            + popular_ids
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
                    )[0]
                    if popular_seed_id
                    else None
                ),
                "recommendations": _enrich_recommendations(
                    popular_ids,
                    details_by_id,
                ),
            },
        },
    }
