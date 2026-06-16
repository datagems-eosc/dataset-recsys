from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


DEFAULT_OUTPUT = Path(__file__).with_name("mathe_benchmark_questions.xlsx")


def keep_questions_with_feasible_document_recommendations(
    rows: list[dict],
    client: MatheMirrorClient,
) -> list[dict]:
    """Keep benchmark questions that have at least one eligible document candidate."""
    feasible_rows = []
    for row in rows:
        document_pool = client.get_document_materials_for_question_topic_subtopic(
            int(row["question_id"])
        )
        if document_pool:
            feasible_rows.append(row)
    return feasible_rows


def export_benchmark_questions(
    output_path: Path,
    only_feasible_recommendations: bool = False,
) -> tuple[int, int]:
    client = MatheMirrorClient()
    try:
        rows = client.get_evaluation_benchmark_questions()
        original_count = len(rows)
        if only_feasible_recommendations:
            rows = keep_questions_with_feasible_document_recommendations(rows, client)
    finally:
        client.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output_path, index=False)
    return original_count, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MathE evaluation benchmark questions to Excel."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--only-feasible-recommendations",
        action="store_true",
        help="Drop benchmark questions whose topic/subtopic pool has no eligible document recommendations.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    total_count, exported_count = export_benchmark_questions(
        args.output,
        only_feasible_recommendations=args.only_feasible_recommendations,
    )
    if args.only_feasible_recommendations:
        removed_count = total_count - exported_count
        print(
            f"Wrote {exported_count} benchmark questions to {args.output} "
            f"({removed_count} without feasible document recommendations removed)"
        )
    else:
        print(f"Wrote {exported_count} benchmark questions to {args.output}")


if __name__ == "__main__":
    main()
