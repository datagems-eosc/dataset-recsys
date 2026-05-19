import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from dataset_recsys.mathe_recommenders.comparison import compare_question_recommenders
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


DEFAULT_JSON_OUTPUT = Path("outputs/mathe_recommender_comparison.json")
DEFAULT_TABLE_OUTPUT = Path("outputs/mathe_recommender_comparison.csv")
TABLE_COLUMNS = [
    "strategy",
    "question_id",
    "question",
    "material_id",
    "title",
    "metadata_score",
    "rank",
]


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


def _table_rows(results: list[dict]) -> list[dict]:
    rows = []

    for result in results:
        input_case = result.get("input") or {}
        comparison = result.get("comparison") or {}
        question = comparison.get("question") or {}

        for strategy_name, strategy in (comparison.get("strategies") or {}).items():
            for rec in strategy.get("recommendations", []):
                scores = rec.get("scores") or {}
                rows.append(
                    {
                        "strategy": strategy_name,
                        "question_id": input_case.get("question_id") or question.get("question_id"),
                        "question": input_case.get("question_text", ""),
                        "material_id": rec.get("material_id", ""),
                        "title": rec.get("title", ""),
                        "metadata_score": scores.get("metadata_score", ""),
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
    parser.add_argument("--question", help="Question text for single-question embedding comparison.")
    parser.add_argument("--questions-file", type=Path, help="JSON file with question cases for batch comparison.")
    parser.add_argument("--limit", type=int, help="Process only the first N cases from --questions-file.")
    parser.add_argument("--json-output", type=Path, help="Full JSON output path. Batch default: outputs/mathe_recommender_comparison.json")
    parser.add_argument("--table-output", type=Path, help="CSV table output path. Batch default: outputs/mathe_recommender_comparison.csv")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--redis-host")
    parser.add_argument("--redis-port")
    args = parser.parse_args()

    if args.questions_file and args.question_id is not None:
        parser.error("Use either question_id or --questions-file, not both.")
    if not args.questions_file and args.question_id is None:
        parser.error("Provide question_id or --questions-file.")

    if load_dotenv:
        load_dotenv(args.env_file)
    if args.redis_host:
        os.environ["REDIS_HOST"] = args.redis_host
    if args.redis_port:
        os.environ["REDIS_PORT"] = str(args.redis_port)

    batch_mode = args.questions_file is not None
    json_output = args.json_output or (DEFAULT_JSON_OUTPUT if batch_mode else None)
    table_output = args.table_output or (DEFAULT_TABLE_OUTPUT if batch_mode else None)

    mathe_client = MatheMirrorClient()
    recs_client = RecommendationClient()
    embedding_client = EmbeddingClient()

    try:
        if batch_mode:
            cases = _load_question_cases(args.questions_file)
            if args.limit:
                cases = cases[:args.limit]

            report = {"source": str(args.questions_file), "count": len(cases), "results": []}
            for index, case in enumerate(cases, start=1):
                print(f"[{index}/{len(cases)}] question_id={case['question_id']}", flush=True)
                comparison = compare_question_recommenders(
                    question_id=case["question_id"],
                    k=args.num_recommendations,
                    mathe_mirror_client=mathe_client,
                    recommendation_client=recs_client,
                    question_text=case["question_text"],
                    embedding_client=embedding_client,
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
