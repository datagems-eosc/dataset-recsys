import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from dataset_recsys.utils.mathe_recsys_comparison import (
    AVAILABLE_STRATEGIES,
    compare_question_recommenders,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


DEFAULT_JSON_OUTPUT = Path("outputs/mathe_recsys_comparison.json")
DEFAULT_TABLE_OUTPUT = Path("outputs/mathe_recsys_comparison.csv")
TABLE_COLUMNS = [
    "question_id",
    "question_text",
    "question_topic",
    "question_subtopic",
    "material_id",
    "material_title",
    "material_topic",
    "material_subtopic",
    "rank",
]


def _join_names(values: list[Any] | None) -> str:
    return "; ".join(str(value) for value in values or [] if value)


def _load_question_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Question file must contain a JSON list.")

    cases = []
    for item in payload:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        cases.append(
            {
                "question_id": int(item["id"]),
                "question_text": str(item.get("question") or "").strip(),
                "level": item.get("level"),
            }
        )
    return cases


def _load_topic_cases(
    topic_subtopics: list[tuple[str, str]],
    mathe_client: MatheMirrorClient,
) -> list[dict]:
    questions = mathe_client.get_questions_by_topic_subtopics(topic_subtopics)
    return [
        {
            "question_id": int(question["question_id"]),
            "question_text": str(question.get("question") or "").strip(),
            "topic": question.get("topic_name"),
            "subtopic": question.get("subtopic_name"),
        }
        for question in questions
    ]


def _table_rows(results: list[dict]) -> list[dict]:
    rows = []

    for result in results:
        input_case = result.get("input") or {}
        comparison = result.get("comparison") or {}
        question = comparison.get("question") or {}

        for strategy in (comparison.get("strategies") or {}).values():
            for rec in strategy.get("recommendations", []):
                rows.append(
                    {
                        "question_id": input_case.get("question_id") or question.get("question_id"),
                        "question_text": input_case.get("question_text", ""),
                        "question_topic": input_case.get("topic") or question.get("topic"),
                        "question_subtopic": input_case.get("subtopic") or question.get("subtopic"),
                        "material_id": rec.get("material_id", ""),
                        "material_title": rec.get("title", ""),
                        "material_topic": _join_names(rec.get("topics")),
                        "material_subtopic": _join_names(rec.get("subtopics")),
                        "rank": rec.get("rank", ""),
                    }
                )

    return rows


def _write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_table(report: dict, path: Path) -> None:
    rows = _table_rows(report.get("results", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare internal MathE material recommendation strategies."
    )
    parser.add_argument("question_id", type=int, nargs="?")
    parser.add_argument("-n", "--num-recommendations", type=int, default=10)
    parser.add_argument(
        "--approach",
        action="append",
        choices=AVAILABLE_STRATEGIES,
        help="Strategy to run. Repeat to compare a subset, e.g. --approach hybrid --approach curricular_pool.",
    )
    parser.add_argument("--question", help="Question text for single-question embedding comparison.")
    parser.add_argument("--questions-file", type=Path, help="JSON file with question cases for batch comparison.")
    parser.add_argument(
        "--topic-subtopic",
        nargs=2,
        action="append",
        metavar=("TOPIC", "SUBTOPIC"),
        help="Topic/subtopic pair to export. Repeat for multiple pairs.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N batch cases.")
    parser.add_argument("--json-output", type=Path, help="Full JSON output path. Batch default: outputs/mathe_recsys_comparison.json")
    parser.add_argument("--table-output", type=Path, help="CSV table output path. Batch default: outputs/mathe_recsys_comparison.csv")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--redis-host")
    parser.add_argument("--redis-port")
    args = parser.parse_args()

    input_modes = sum(
        bool(value)
        for value in (args.questions_file, args.topic_subtopic, args.question_id is not None)
    )
    if input_modes != 1:
        parser.error("Provide exactly one input mode: question_id, --questions-file, or --topic-subtopic.")

    if load_dotenv:
        load_dotenv(args.env_file)
    if args.redis_host:
        os.environ["REDIS_HOST"] = args.redis_host
    if args.redis_port:
        os.environ["REDIS_PORT"] = str(args.redis_port)

    batch_mode = args.questions_file is not None or args.topic_subtopic is not None
    json_output = args.json_output or (DEFAULT_JSON_OUTPUT if batch_mode else None)
    table_output = args.table_output or (DEFAULT_TABLE_OUTPUT if batch_mode else None)

    mathe_client = MatheMirrorClient()
    recs_client = RecommendationClient()
    embedding_client = EmbeddingClient()

    try:
        if batch_mode:
            if args.questions_file:
                cases = _load_question_cases(args.questions_file)
                source = str(args.questions_file)
            else:
                cases = _load_topic_cases(args.topic_subtopic, mathe_client)
                source = "topic_subtopic"
            if args.limit:
                cases = cases[:args.limit]

            report = {
                "source": source,
                "approaches": args.approach or list(AVAILABLE_STRATEGIES),
                "count": len(cases),
                "results": [],
            }
            for index, case in enumerate(cases, start=1):
                print(f"[{index}/{len(cases)}] question_id={case['question_id']}", flush=True)
                comparison = compare_question_recommenders(
                    question_id=case["question_id"],
                    k=args.num_recommendations,
                    mathe_mirror_client=mathe_client,
                    recommendation_client=recs_client,
                    question_text=case["question_text"],
                    embedding_client=embedding_client,
                    strategies=args.approach,
                )
                report["results"].append({"input": case, "comparison": comparison})
                if json_output:
                    _write_json(report, json_output)
                if table_output:
                    _write_table(report, table_output)
        else:
            report = compare_question_recommenders(
                question_id=args.question_id,
                k=args.num_recommendations,
                mathe_mirror_client=mathe_client,
                recommendation_client=recs_client,
                question_text=args.question,
                embedding_client=embedding_client,
                strategies=args.approach,
            )
    finally:
        mathe_client.close()
        embedding_client.close()

    if json_output:
        _write_json(report, json_output)
        print(f"Wrote JSON: {json_output}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if table_output:
        _write_table(report, table_output)
        print(f"Wrote table: {table_output}")


if __name__ == "__main__":
    main()
