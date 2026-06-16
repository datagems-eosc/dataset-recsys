from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    recommend_from_curricular_pool,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


DEFAULT_OUTPUT = Path(__file__).with_name("mathe_most_clicked_proxy_evaluation.xlsx")


def _benchmark_questions_with_document_candidates(
    client: MatheMirrorClient,
) -> list[dict]:
    rows = client.get_evaluation_benchmark_questions()
    return [
        row
        for row in rows
        if client.get_document_materials_for_question_topic_subtopic(
            int(row["question_id"])
        )
    ]


def evaluate_most_clicked_proxy(
    benchmark_rows: list[dict],
    mathe_client: MatheMirrorClient,
    embedding_client: EmbeddingClient,
    recommendations_k: int,
) -> list[dict]:
    """Evaluate whether each pool's most-clicked document appears in recommendations."""
    evaluation_rows = []
    for question in benchmark_rows:
        question_id = int(question["question_id"])
        most_clicked = mathe_client.get_most_popular_document_material_for_question_topic_subtopic(
            question_id
        ) or {}
        most_clicked_id = (
            str(most_clicked["material_id"])
            if most_clicked.get("material_id") is not None
            else None
        )
        recommendations = recommend_from_curricular_pool(
            question_id=question_id,
            question=str(question.get("question") or ""),
            k=recommendations_k,
            mathe_mirror_client=mathe_client,
            embedding_client=embedding_client,
        )
        recommender_rank = (
            recommendations.index(most_clicked_id) + 1
            if most_clicked_id in recommendations
            else None
        )

        evaluation_rows.append(
            {
                "question_id": question.get("question_id"),
                "question": question.get("question"),
                "topic_name": question.get("topic_name"),
                "subtopic_name": question.get("subtopic_name"),
                "wrong_rate": question.get("wrong_rate"),
                "distinct_students": question.get("distinct_students"),
                "most_clicked_material_id": most_clicked.get("material_id"),
                "most_clicked_material_redis_id": most_clicked.get("material_redis_id"),
                "most_clicked_title": most_clicked.get("title"),
                "file_ext": most_clicked.get("file_ext"),
                "clicks": most_clicked.get("clicks"),
                "recommender_rank": recommender_rank,
                "hit_at_5": bool(recommender_rank and recommender_rank <= 5),
                f"hit_at_{recommendations_k}": bool(
                    recommender_rank and recommender_rank <= recommendations_k
                ),
                f"recommender_top_{recommendations_k}_material_ids": "; ".join(
                    recommendations
                ),
            }
        )
    return evaluation_rows


def run_evaluation(output_path: Path, recommendations_k: int = 10) -> int:
    mathe_client = MatheMirrorClient()
    embedding_client = EmbeddingClient()
    try:
        benchmark_rows = _benchmark_questions_with_document_candidates(mathe_client)
        rows = evaluate_most_clicked_proxy(
            benchmark_rows,
            mathe_client,
            embedding_client,
            recommendations_k,
        )
    finally:
        embedding_client.close()
        mathe_client.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MathE recommendations against the most-clicked proxy."
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
    print(f"Wrote {count} most-clicked proxy evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
