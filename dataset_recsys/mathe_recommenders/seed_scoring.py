from __future__ import annotations

from typing import Any, Iterable


def _keyword_set(keywords: Iterable[Any] | None) -> set[str]:
    if keywords is None:
        return set()

    return {str(keyword) for keyword in keywords if keyword is not None}


def compute_keyword_jaccard(
    question_keywords: Iterable[Any] | None,
    material_keywords: Iterable[Any] | None,
) -> float:
    """Compute Jaccard similarity between question and material keyword sets."""
    question_keyword_set = _keyword_set(question_keywords)
    material_keyword_set = _keyword_set(material_keywords)
    union = question_keyword_set | material_keyword_set

    if not union:
        return 0.0

    return len(question_keyword_set & material_keyword_set) / len(union)


def score_pdf_seed_candidates(
    question_metadata: dict[str, Any],
    seed_candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score PDF seed candidates using keyword, topic, and subtopic metadata."""
    question_keywords = question_metadata.get("keywords")
    question_topic_id = question_metadata.get("topic_id")
    question_subtopic_id = question_metadata.get("subtopic_id")
    scored_candidates: list[dict[str, Any]] = []

    for candidate in seed_candidates:
        keyword_jaccard = compute_keyword_jaccard(
            question_keywords,
            candidate.get("keywords"),
        )
        candidate_subtopic_ids = {
            subtopic_id
            for subtopic_id in candidate.get("subtopic_ids", [])
            if subtopic_id is not None
        }
        candidate_topic_ids = {
            topic_id
            for topic_id in candidate.get("topic_ids", [])
            if topic_id is not None
        }
        same_subtopic = int(
            question_subtopic_id is not None
            and question_subtopic_id in candidate_subtopic_ids
        )
        same_topic = int(
            question_topic_id is not None
            and question_topic_id in candidate_topic_ids
        )

        scored_candidates.append(
            {
                **candidate,
                "keyword_jaccard": keyword_jaccard,
                "same_subtopic": same_subtopic,
                "same_topic": same_topic,
                "metadata_score": (
                    keyword_jaccard + same_subtopic + same_topic
                ) / 3,
            }
        )

    return sorted(
        scored_candidates,
        key=lambda candidate: candidate["metadata_score"],
        reverse=True,
    )


__all__ = [
    "compute_keyword_jaccard",
    "score_pdf_seed_candidates",
]
