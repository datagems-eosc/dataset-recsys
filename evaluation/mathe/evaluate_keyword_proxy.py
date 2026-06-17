from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    rank_curricular_pool_candidates,
)
from dataset_recsys.mathe_recommenders.seed_scoring import compute_keyword_jaccard
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from recs_metrics.ranked_list import (
    binary_precision_at_k,
    mean_defined_metric,
    weighted_ndcg_at_k,
)


DEFAULT_OUTPUT = Path(__file__).with_name("mathe_keyword_proxy_evaluation.xlsx")


def _keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item is not None]


def _keyword_set(value: Any) -> set[str]:
    return set(_keywords(value))


def _median_defined(values: list[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return median(defined) if defined else None


def _percentile_defined(values: list[float | None], percentile: float) -> float | None:
    defined = sorted(float(value) for value in values if value is not None)
    if not defined:
        return None

    index = math.ceil((percentile / 100) * len(defined)) - 1
    return defined[min(max(index, 0), len(defined) - 1)]


def evaluate_keyword_proxy(
    benchmark_rows: list[dict],
    mathe_client: MatheMirrorClient,
    embedding_client: EmbeddingClient,
    recommendations_k: int,
) -> list[dict]:
    """Evaluate ranking agreement with question-material keyword overlap."""
    evaluation_rows = []
    for question in benchmark_rows:
        question_id = int(question["question_id"])
        question_keywords = _keyword_set(question.get("keywords"))
        document_pool = mathe_client.get_document_materials_for_question_topic_subtopic(
            question_id
        )

        relevance_by_material = {}
        shared_keywords_by_material = {}
        for material in document_pool:
            material_redis_id = str(material["material_redis_id"]).strip()
            material_keywords = _keyword_set(material.get("keywords"))
            shared_keywords = question_keywords & material_keywords
            relevance_by_material[material_redis_id] = compute_keyword_jaccard(
                question_keywords,
                material_keywords,
            )
            shared_keywords_by_material[material_redis_id] = sorted(shared_keywords)

        relevant_ids = {
            material_id
            for material_id, relevance in relevance_by_material.items()
            if relevance > 0
        }

        recommendation_start = time.perf_counter()
        ranked_candidates = rank_curricular_pool_candidates(
            question_id=question_id,
            question=str(question.get("question") or ""),
            k=recommendations_k,
            mathe_mirror_client=mathe_client,
            embedding_client=embedding_client,
        )
        recommendation_time_seconds = time.perf_counter() - recommendation_start
        recommended_ids = [
            str(candidate["material_redis_id"]).strip()
            for candidate in ranked_candidates
        ]
        recommended_relevances = [
            relevance_by_material.get(material_id, 0.0)
            for material_id in recommended_ids
        ]

        if not document_pool:
            status = "no_document_candidates"
        elif not relevant_ids:
            status = "no_keyword_proxy_ground_truth"
        else:
            status = "evaluated"

        evaluation_rows.append(
            {
                "question_id": question.get("question_id"),
                "question": question.get("question"),
                "topic_name": question.get("topic_name"),
                "subtopic_name": question.get("subtopic_name"),
                "question_keywords": "; ".join(sorted(question_keywords)),
                "wrong_rate": question.get("wrong_rate"),
                "distinct_students": question.get("distinct_students"),
                "candidate_material_count": len(document_pool),
                "keyword_matching_material_count": len(relevant_ids),
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
                f"recommender_top_{recommendations_k}_material_ids": "; ".join(
                    recommended_ids
                ),
                f"recommender_top_{recommendations_k}_keyword_jaccards": "; ".join(
                    f"{relevance:.4f}" for relevance in recommended_relevances
                ),
                f"recommender_top_{recommendations_k}_shared_keywords": "; ".join(
                    ", ".join(shared_keywords_by_material.get(material_id, []))
                    or "-"
                    for material_id in recommended_ids
                ),
            }
        )

    return evaluation_rows


def summarize(rows: list[dict], recommendations_k: int) -> list[dict]:
    recommendation_times = [row["recommendation_time_seconds"] for row in rows]
    no_document_ids = [
        str(row["question_id"])
        for row in rows
        if row["evaluation_status"] == "no_document_candidates"
    ]
    no_keyword_ground_truth_ids = [
        str(row["question_id"])
        for row in rows
        if row["evaluation_status"] == "no_keyword_proxy_ground_truth"
    ]

    return [
        {"metric": "benchmark_questions", "value": len(rows)},
        {
            "metric": "questions_with_document_candidates",
            "value": sum(row["candidate_material_count"] > 0 for row in rows),
        },
        {
            "metric": "keyword_evaluable_questions",
            "value": sum(row["evaluation_status"] == "evaluated" for row in rows),
        },
        {
            "metric": "excluded_questions",
            "value": len(no_document_ids) + len(no_keyword_ground_truth_ids),
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
            "metric": "questions_without_keyword_proxy_ground_truth",
            "value": len(no_keyword_ground_truth_ids),
        },
        {
            "metric": "questions_without_keyword_proxy_ground_truth_ids",
            "value": ", ".join(no_keyword_ground_truth_ids),
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


def excluded_questions(rows: list[dict]) -> list[dict]:
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
                "candidate_material_count": row["candidate_material_count"],
                "keyword_matching_material_count": row[
                    "keyword_matching_material_count"
                ],
            }
        )
    return excluded_rows


def run_evaluation(output_path: Path, recommendations_k: int = 10) -> int:
    mathe_client = MatheMirrorClient()
    embedding_client = EmbeddingClient()
    try:
        benchmark_rows = mathe_client.get_evaluation_benchmark_questions()
        rows = evaluate_keyword_proxy(
            benchmark_rows,
            mathe_client,
            embedding_client,
            recommendations_k,
        )
        summary_rows = summarize(rows, recommendations_k)
        excluded_rows = excluded_questions(rows)
    finally:
        embedding_client.close()
        mathe_client.close()

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
        description="Evaluate MathE recommendations against keyword-overlap proxy relevance."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--recommendations-k",
        type=int,
        default=10,
        help="Number of recommendations to generate for proxy evaluation.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    count = run_evaluation(args.output, recommendations_k=args.recommendations_k)
    print(f"Wrote {count} keyword proxy evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
