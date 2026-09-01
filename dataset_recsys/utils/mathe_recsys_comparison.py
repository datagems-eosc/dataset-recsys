from dataset_recsys.mathe_recommenders.metadata_ocr import (
    recommend_document_seeds_for_question,
    rank_expanded_candidates,
    seed_material_id,
)
from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.mathe_recommenders.question_embedding import (
    recommend_from_question_embedding,
)
from dataset_recsys.mathe_recommenders.hybrid import recommend_hybrid_candidates
from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    rank_curricular_pool_candidates,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


AVAILABLE_STRATEGIES = (
    "metadata",
    "popular_seed",
    "question_embedding",
    "hybrid",
    "curricular_pool",
)


def _selected_strategies(strategies: list[str] | tuple[str, ...] | None) -> set[str]:
    selected = set(strategies or AVAILABLE_STRATEGIES)
    unknown = selected - set(AVAILABLE_STRATEGIES)
    if unknown:
        raise ValueError(
            "Unknown recommender strategies: "
            + ", ".join(sorted(unknown))
            + ". Available strategies: "
            + ", ".join(AVAILABLE_STRATEGIES)
        )
    return selected


def _details_by_material_id(
    material_ids: list[str],
    mathe_mirror_client: MatheMirrorClient,
) -> dict[str, dict]:
    details = mathe_mirror_client.get_document_material_details_by_ids(material_ids)
    return {
        str(material["material_id"]).strip(): dict(material)
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


def _pick_scores(candidate: dict, keys: tuple[str, ...]) -> dict:
    return {
        key: candidate[key]
        for key in keys
        if key in candidate
    }


def _candidate_id(candidate: dict) -> str:
    return str(candidate["material_id"]).strip()


def _candidate_scores(
    candidates: list[dict],
    keys: tuple[str, ...],
    total_score_key: str = "final_score",
) -> dict[str, dict]:
    scores_by_id = {}
    for candidate in candidates:
        scores = _pick_scores(candidate, keys)
        if total_score_key in candidate:
            scores["total_score"] = candidate[total_score_key]
        scores_by_id[_candidate_id(candidate)] = scores
    return scores_by_id


def _add_strategy(
    strategy_payloads: dict,
    material_ids_for_details: list[str],
    name: str,
    ids: list[str],
    scores: dict[str, dict],
    seed_id: str | None = None,
) -> None:
    strategy_payloads[name] = {
        "ids": ids,
        "scores": scores,
        **({"seed_id": seed_id} if seed_id else {}),
    }
    material_ids_for_details.extend(ids + ([seed_id] if seed_id else []))


def compare_question_recommenders(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
    question_text: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    strategies: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Build an internal comparison report for MathE recommender strategies."""
    selected = _selected_strategies(strategies)
    question_metadata = mathe_mirror_client.get_question_metadata(question_id)
    if not question_metadata:
        return {
            "question": None,
            "strategies": {},
        }

    strategy_payloads = {}
    material_ids_for_details = []

    if "metadata" in selected:
        metadata_seeds = recommend_document_seeds_for_question(
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
        _add_strategy(
            strategy_payloads,
            material_ids_for_details,
            "metadata",
            [_candidate_id(candidate) for candidate in metadata_candidates],
            _candidate_scores(
                metadata_candidates,
                ("metadata_score", "material_to_material_similarity"),
            ),
        )

    if "popular_seed" in selected:
        popular_seed = mathe_mirror_client.get_popular_document_for_question(
            question_id
        )
        popular_seed_id = seed_material_id(popular_seed) if popular_seed else None
        popular_neighbors = (
            recommendation_client.get_recommendations_with_scores(
                application=MatheApplication.DOCUMENTS,
                entity_id=popular_seed_id,
                limit=k,
            )
            if popular_seed_id
            else []
        )
        popular_ids = [str(material_id).strip() for material_id, _ in popular_neighbors]
        popular_scores = {
            str(material_id).strip(): {
                "material_to_material_similarity": float(similarity)
            }
            for material_id, similarity in popular_neighbors
        }
        _add_strategy(
            strategy_payloads,
            material_ids_for_details,
            "popular_seed",
            popular_ids,
            popular_scores,
            seed_id=popular_seed_id,
        )

    if "question_embedding" in selected:
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
        _add_strategy(
            strategy_payloads,
            material_ids_for_details,
            "question_embedding",
            [_candidate_id(candidate) for candidate in question_embedding_candidates],
            _candidate_scores(
                question_embedding_candidates,
                ("metadata_score", "question_to_material_similarity"),
                total_score_key="total_score",
            ),
        )

    if "hybrid" in selected:
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
        _add_strategy(
            strategy_payloads,
            material_ids_for_details,
            "hybrid",
            [_candidate_id(candidate) for candidate in hybrid_candidates],
            _candidate_scores(
                hybrid_candidates,
                (
                    "metadata_score",
                    "material_to_material_similarity",
                    "question_to_material_similarity",
                ),
            ),
        )

    if "curricular_pool" in selected:
        curricular_candidates = (
            rank_curricular_pool_candidates(
                question_id=question_id,
                question=question_text,
                k=k,
                mathe_mirror_client=mathe_mirror_client,
                embedding_client=embedding_client,
            )
            if question_text
            else []
        )
        _add_strategy(
            strategy_payloads,
            material_ids_for_details,
            "curricular_pool",
            [_candidate_id(candidate) for candidate in curricular_candidates],
            _candidate_scores(
                curricular_candidates,
                ("keyword_jaccard", "question_to_material_similarity"),
            ),
        )

    material_ids_for_details = list(dict.fromkeys(material_ids_for_details))
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

    strategies_output = {}
    if "metadata" in strategy_payloads:
        strategies_output["metadata"] = {
            "description": "Metadata/OCR flow: document seeds expanded with stored document neighbors only when needed.",
            "recommendations": _enrich_recommendations(
                strategy_payloads["metadata"]["ids"],
                details_by_id,
                strategy_payloads["metadata"]["scores"],
            ),
        }
    if "popular_seed" in strategy_payloads:
        popular_seed_id = strategy_payloads["popular_seed"].get("seed_id")
        strategies_output["popular_seed"] = {
            "description": "Previous experimental flow: most-clicked document seed in the question pool, then stored document neighbors.",
            "seed": (
                _enrich_recommendations(
                    [popular_seed_id],
                    details_by_id,
                    strategy_payloads["popular_seed"]["scores"],
                )[0]
                if popular_seed_id
                else None
            ),
            "recommendations": _enrich_recommendations(
                strategy_payloads["popular_seed"]["ids"],
                details_by_id,
                strategy_payloads["popular_seed"]["scores"],
            ),
        }
    if "question_embedding" in strategy_payloads:
        strategies_output["question_embedding"] = {
            "description": "Question-text flow: embed provided question text, then query MathE material embeddings in pgvector.",
            "recommendations": _enrich_recommendations(
                strategy_payloads["question_embedding"]["ids"],
                details_by_id,
                strategy_payloads["question_embedding"]["scores"],
            ),
            "available": bool(question_text),
        }
    if "hybrid" in strategy_payloads:
        strategies_output["hybrid"] = {
            "description": "Open-pool hybrid flow: merge metadata/OCR candidates with a small set of question-text embedding candidates, then rerank with metadata, material OCR, and question similarity scores.",
            "recommendations": _enrich_recommendations(
                strategy_payloads["hybrid"]["ids"],
                details_by_id,
                strategy_payloads["hybrid"]["scores"],
            ),
            "available": bool(question_text),
        }
    if "curricular_pool" in strategy_payloads:
        strategies_output["curricular_pool"] = {
            "description": "Current production flow: hard same-topic/same-subtopic document pool, ranked with keyword overlap and question-to-material similarity.",
            "recommendations": _enrich_recommendations(
                strategy_payloads["curricular_pool"]["ids"],
                details_by_id,
                strategy_payloads["curricular_pool"]["scores"],
            ),
            "available": bool(question_text),
        }

    return {
        "question": question_details,
        "strategies": strategies_output,
    }
