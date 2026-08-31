from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    rank_curricular_pool_candidates,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from recs_metrics.ranked_list import (
    binary_precision_at_k,
    mean_defined_metric,
    weighted_ndcg_at_k,
)


DEFAULT_REFERENCE_WORKBOOK = Path(__file__).with_name("mathe_reference_alltopics.xlsx")
DEFAULT_OUTPUT = Path(__file__).with_name("mathe_expert_reference_evaluation.xlsx")
REFERENCE_SHEET_NAME = "Reference Set"
SAVED_RECOMMENDATIONS_SHEET_NAME = "per_question"


def parse_material_ids(value: Any) -> list[str]:
    """Parse comma/semicolon/whitespace-separated MathE material IDs from Excel."""
    if value is None or pd.isna(value):
        return []

    if isinstance(value, int):
        return [str(value)]

    if isinstance(value, float):
        if not math.isfinite(value):
            return []
        if value.is_integer():
            return [str(int(value))]

    text = str(value).strip()
    if not text:
        return []

    if re.fullmatch(r"\d+\.0+", text):
        return [text.split(".", maxsplit=1)[0]]

    material_ids = re.findall(r"\d+", text)
    return list(dict.fromkeys(material_ids))


def _is_yes(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _median_defined(values: list[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return median(defined) if defined else None


def _percentile_defined(values: list[float | None], percentile: float) -> float | None:
    defined = sorted(float(value) for value in values if value is not None)
    if not defined:
        return None

    index = math.ceil((percentile / 100) * len(defined)) - 1
    return defined[min(max(index, 0), len(defined) - 1)]


def load_reference_rows(reference_workbook: Path) -> list[dict[str, Any]]:
    reference_df = pd.read_excel(reference_workbook, sheet_name=REFERENCE_SHEET_NAME)
    return reference_df.to_dict(orient="records")


def load_saved_recommendation_rows(
    recommendations_workbook: Path,
) -> dict[int, dict[str, Any]]:
    recommendations_df = pd.read_excel(
        recommendations_workbook,
        sheet_name=SAVED_RECOMMENDATIONS_SHEET_NAME,
    )
    return {
        int(row["question_id"]): row
        for row in recommendations_df.to_dict(orient="records")
    }


def _first_relevant_rank(
    recommended_ids: list[str],
    relevant_ids: set[str],
) -> int | None:
    for index, material_id in enumerate(recommended_ids, start=1):
        if material_id in relevant_ids:
            return index
    return None


def evaluate_expert_reference(
    reference_rows: list[dict[str, Any]],
    mathe_client: MatheMirrorClient,
    embedding_client: EmbeddingClient,
    recommendations_k: int,
) -> list[dict[str, Any]]:
    """Evaluate MathE recommendations against expert-provided document IDs."""
    evaluation_rows = []
    for reference in reference_rows:
        question_id = int(reference["question_id"])
        expert_ids = parse_material_ids(reference.get("related_document_material_ids"))
        relevant_ids = set(expert_ids)
        relevance_by_material = {material_id: 1.0 for material_id in relevant_ids}

        document_pool = mathe_client.get_document_materials_for_question_topic_subtopic(
            question_id
        )
        candidate_ids = {
            str(material["material_id"]).strip()
            for material in document_pool
            if material.get("material_id") is not None
        }

        recommendation_start = time.perf_counter()
        ranked_candidates = rank_curricular_pool_candidates(
            question_id=question_id,
            question=_safe_text(reference.get("question")),
            k=recommendations_k,
            mathe_mirror_client=mathe_client,
            embedding_client=embedding_client,
        )
        recommendation_time_seconds = time.perf_counter() - recommendation_start

        recommended_ids = [
            str(candidate["material_id"]).strip()
            for candidate in ranked_candidates
            if candidate.get("material_id") is not None
        ]
        first_relevant_rank = _first_relevant_rank(recommended_ids, relevant_ids)

        if not document_pool:
            status = "no_document_candidates"
        elif not relevant_ids:
            status = "no_expert_document_ground_truth"
        else:
            status = "evaluated"

        expert_ids_in_candidate_pool = [
            material_id for material_id in expert_ids if material_id in candidate_ids
        ]
        expert_ids_outside_candidate_pool = [
            material_id for material_id in expert_ids if material_id not in candidate_ids
        ]

        evaluation_rows.append(
            {
                "question_id": question_id,
                "question": reference.get("question"),
                "topic_name": reference.get("topic_name"),
                "subtopic_name": reference.get("subtopic_name"),
                "question_keywords": reference.get("question_keywords"),
                "provided_document_material_ids": "; ".join(expert_ids),
                "no_suitable_documents": _is_yes(
                    reference.get("no_suitable_documents")
                ),
                "optional_comment": reference.get("optional_comment"),
                "candidate_material_count": len(document_pool),
                "expert_reference_material_count": len(expert_ids),
                "expert_reference_materials_in_candidate_pool": "; ".join(
                    expert_ids_in_candidate_pool
                ),
                "expert_reference_materials_outside_candidate_pool": "; ".join(
                    expert_ids_outside_candidate_pool
                ),
                "evaluation_status": status,
                "recommendation_time_seconds": recommendation_time_seconds,
                "weighted_ndcg_at_5": weighted_ndcg_at_k(
                    recommended_ids,
                    relevance_by_material,
                    5,
                ),
                f"weighted_ndcg_at_{recommendations_k}": weighted_ndcg_at_k(
                    recommended_ids,
                    relevance_by_material,
                    recommendations_k,
                ),
                "precision_at_5": binary_precision_at_k(
                    recommended_ids,
                    relevant_ids,
                    5,
                ),
                f"precision_at_{recommendations_k}": binary_precision_at_k(
                    recommended_ids,
                    relevant_ids,
                    recommendations_k,
                ),
                "hit_at_5": (
                    bool(first_relevant_rank and first_relevant_rank <= 5)
                    if relevant_ids
                    else None
                ),
                f"hit_at_{recommendations_k}": (
                    bool(
                        first_relevant_rank
                        and first_relevant_rank <= recommendations_k
                    )
                    if relevant_ids
                    else None
                ),
                "first_relevant_rank": first_relevant_rank,
                f"recommender_top_{recommendations_k}_material_ids": "; ".join(
                    recommended_ids
                ),
                f"recommender_top_{recommendations_k}_expert_hits": "; ".join(
                    material_id for material_id in recommended_ids if material_id in relevant_ids
                ),
            }
        )

    return evaluation_rows


def evaluate_expert_reference_from_saved_recommendations(
    reference_rows: list[dict[str, Any]],
    saved_recommendation_rows: dict[int, dict[str, Any]],
    recommendations_k: int,
) -> list[dict[str, Any]]:
    """Evaluate expert IDs using a previously exported recommendation workbook."""
    recommendation_column = f"recommender_top_{recommendations_k}_material_ids"
    evaluation_rows = []
    for reference in reference_rows:
        question_id = int(reference["question_id"])
        saved = saved_recommendation_rows.get(question_id, {})
        expert_ids = parse_material_ids(reference.get("related_document_material_ids"))
        relevant_ids = set(expert_ids)
        relevance_by_material = {material_id: 1.0 for material_id in relevant_ids}
        recommended_ids = parse_material_ids(saved.get(recommendation_column))
        first_relevant_rank = _first_relevant_rank(recommended_ids, relevant_ids)
        candidate_material_count = int(saved.get("candidate_material_count") or 0)

        if not saved:
            status = "missing_saved_recommendations"
        elif candidate_material_count == 0:
            status = "no_document_candidates"
        elif not relevant_ids:
            status = "no_expert_document_ground_truth"
        else:
            status = "evaluated"

        evaluation_rows.append(
            {
                "question_id": question_id,
                "question": reference.get("question"),
                "topic_name": reference.get("topic_name"),
                "subtopic_name": reference.get("subtopic_name"),
                "question_keywords": reference.get("question_keywords"),
                "provided_document_material_ids": "; ".join(expert_ids),
                "no_suitable_documents": _is_yes(
                    reference.get("no_suitable_documents")
                ),
                "optional_comment": reference.get("optional_comment"),
                "candidate_material_count": candidate_material_count,
                "expert_reference_material_count": len(expert_ids),
                "expert_reference_materials_in_candidate_pool": "",
                "expert_reference_materials_outside_candidate_pool": "",
                "evaluation_status": status,
                "recommendation_time_seconds": saved.get(
                    "recommendation_time_seconds"
                ),
                "weighted_ndcg_at_5": weighted_ndcg_at_k(
                    recommended_ids,
                    relevance_by_material,
                    5,
                ),
                f"weighted_ndcg_at_{recommendations_k}": weighted_ndcg_at_k(
                    recommended_ids,
                    relevance_by_material,
                    recommendations_k,
                ),
                "precision_at_5": binary_precision_at_k(
                    recommended_ids,
                    relevant_ids,
                    5,
                ),
                f"precision_at_{recommendations_k}": binary_precision_at_k(
                    recommended_ids,
                    relevant_ids,
                    recommendations_k,
                ),
                "hit_at_5": (
                    bool(first_relevant_rank and first_relevant_rank <= 5)
                    if relevant_ids
                    else None
                ),
                f"hit_at_{recommendations_k}": (
                    bool(
                        first_relevant_rank
                        and first_relevant_rank <= recommendations_k
                    )
                    if relevant_ids
                    else None
                ),
                "first_relevant_rank": first_relevant_rank,
                f"recommender_top_{recommendations_k}_material_ids": "; ".join(
                    recommended_ids
                ),
                f"recommender_top_{recommendations_k}_expert_hits": "; ".join(
                    material_id for material_id in recommended_ids if material_id in relevant_ids
                ),
            }
        )

    return evaluation_rows


def summarize(rows: list[dict[str, Any]], recommendations_k: int) -> list[dict]:
    recommendation_times = [row["recommendation_time_seconds"] for row in rows]
    no_document_ids = [
        str(row["question_id"])
        for row in rows
        if row["evaluation_status"] == "no_document_candidates"
    ]
    no_expert_ground_truth_ids = [
        str(row["question_id"])
        for row in rows
        if row["evaluation_status"] == "no_expert_document_ground_truth"
    ]
    missing_saved_recommendation_ids = [
        str(row["question_id"])
        for row in rows
        if row["evaluation_status"] == "missing_saved_recommendations"
    ]

    return [
        {"metric": "reference_questions", "value": len(rows)},
        {
            "metric": "questions_with_expert_document_ids",
            "value": sum(row["expert_reference_material_count"] > 0 for row in rows),
        },
        {
            "metric": "expert_evaluable_questions",
            "value": sum(row["evaluation_status"] == "evaluated" for row in rows),
        },
        {
            "metric": "excluded_questions",
            "value": len(no_document_ids)
            + len(no_expert_ground_truth_ids)
            + len(missing_saved_recommendation_ids),
        },
        {
            "metric": "questions_without_document_candidates",
            "value": len(no_document_ids),
        },
        {
            "metric": "questions_without_document_candidates_ids",
            "value": ", ".join(no_document_ids),
        },
        {
            "metric": "questions_without_expert_document_ground_truth",
            "value": len(no_expert_ground_truth_ids),
        },
        {
            "metric": "questions_without_expert_document_ground_truth_ids",
            "value": ", ".join(no_expert_ground_truth_ids),
        },
        {
            "metric": "questions_missing_saved_recommendations",
            "value": len(missing_saved_recommendation_ids),
        },
        {
            "metric": "questions_missing_saved_recommendations_ids",
            "value": ", ".join(missing_saved_recommendation_ids),
        },
        {
            "metric": "mean_weighted_ndcg_at_5",
            "value": mean_defined_metric(
                [row["weighted_ndcg_at_5"] for row in rows]
            ),
        },
        {
            "metric": f"mean_weighted_ndcg_at_{recommendations_k}",
            "value": mean_defined_metric(
                [row[f"weighted_ndcg_at_{recommendations_k}"] for row in rows]
            ),
        },
        {
            "metric": "mean_precision_at_5",
            "value": mean_defined_metric([row["precision_at_5"] for row in rows]),
        },
        {
            "metric": f"mean_precision_at_{recommendations_k}",
            "value": mean_defined_metric(
                [row[f"precision_at_{recommendations_k}"] for row in rows]
            ),
        },
        {
            "metric": "mean_hit_at_5",
            "value": mean_defined_metric([row["hit_at_5"] for row in rows]),
        },
        {
            "metric": f"mean_hit_at_{recommendations_k}",
            "value": mean_defined_metric(
                [row[f"hit_at_{recommendations_k}"] for row in rows]
            ),
        },
        {
            "metric": "mean_recommendation_time_seconds",
            "value": mean_defined_metric(recommendation_times),
        },
        {
            "metric": "median_recommendation_time_seconds",
            "value": _median_defined(recommendation_times),
        },
        {
            "metric": "p95_recommendation_time_seconds",
            "value": _percentile_defined(recommendation_times, 95),
        },
    ]


def excluded_questions(rows: list[dict[str, Any]]) -> list[dict]:
    excluded_rows = []
    for row in rows:
        status = row["evaluation_status"]
        if status == "evaluated":
            continue

        excluded_rows.append(
            {
                "question_id": row["question_id"],
                "exclusion_reason": status,
                "question": row["question"],
                "topic_name": row["topic_name"],
                "subtopic_name": row["subtopic_name"],
                "question_keywords": row["question_keywords"],
                "provided_document_material_ids": row[
                    "provided_document_material_ids"
                ],
                "no_suitable_documents": row["no_suitable_documents"],
                "candidate_material_count": row["candidate_material_count"],
                "expert_reference_material_count": row[
                    "expert_reference_material_count"
                ],
            }
        )
    return excluded_rows


def run_evaluation(
    reference_workbook: Path,
    output_path: Path,
    recommendations_k: int = 10,
    recommendations_workbook: Path | None = None,
) -> int:
    if recommendations_workbook is not None:
        reference_rows = load_reference_rows(reference_workbook)
        saved_recommendation_rows = load_saved_recommendation_rows(
            recommendations_workbook
        )
        rows = evaluate_expert_reference_from_saved_recommendations(
            reference_rows,
            saved_recommendation_rows,
            recommendations_k,
        )
        summary_rows = summarize(rows, recommendations_k)
        excluded_rows = excluded_questions(rows)
        return write_evaluation_workbook(output_path, rows, summary_rows, excluded_rows)

    mathe_client = MatheMirrorClient()
    embedding_client = EmbeddingClient()
    try:
        reference_rows = load_reference_rows(reference_workbook)
        rows = evaluate_expert_reference(
            reference_rows,
            mathe_client,
            embedding_client,
            recommendations_k,
        )
        summary_rows = summarize(rows, recommendations_k)
        excluded_rows = excluded_questions(rows)
    finally:
        embedding_client.close()
        mathe_client.close()

    return write_evaluation_workbook(output_path, rows, summary_rows, excluded_rows)


def write_evaluation_workbook(
    output_path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        pd.DataFrame(rows).to_excel(writer, sheet_name="per_question", index=False)
        pd.DataFrame(excluded_rows).to_excel(
            writer,
            sheet_name="excluded_questions",
            index=False,
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MathE recommendations against expert-provided document IDs."
    )
    parser.add_argument(
        "--reference-workbook",
        type=Path,
        default=DEFAULT_REFERENCE_WORKBOOK,
        help="Workbook containing the expert Reference Set sheet.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recommendations-workbook",
        type=Path,
        default=None,
        help=(
            "Optional existing evaluation workbook containing per_question "
            "recommendation lists. When provided, no live MATHE services are used."
        ),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--recommendations-k",
        type=int,
        default=10,
        help="Number of recommendations to generate for expert reference evaluation.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    count = run_evaluation(
        args.reference_workbook,
        args.output,
        recommendations_k=args.recommendations_k,
        recommendations_workbook=args.recommendations_workbook,
    )
    print(f"Wrote {count} expert reference evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
