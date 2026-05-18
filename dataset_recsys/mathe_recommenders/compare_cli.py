import argparse
import json
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from dataset_recsys.mathe_recommenders.comparison import compare_question_recommenders
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare internal MathE material recommendation strategies."
    )
    parser.add_argument("question_id", type=int)
    parser.add_argument("-n", "--num-recommendations", type=int, default=10)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--redis-host", default=None)
    parser.add_argument("--redis-port", default=None)
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv(args.env_file)

    if args.redis_host:
        os.environ["REDIS_HOST"] = args.redis_host
    if args.redis_port:
        os.environ["REDIS_PORT"] = str(args.redis_port)

    mathe_mirror_client = MatheMirrorClient()
    try:
        report = compare_question_recommenders(
            question_id=args.question_id,
            k=args.num_recommendations,
            mathe_mirror_client=mathe_mirror_client,
            recommendation_client=RecommendationClient(),
        )
    finally:
        mathe_mirror_client.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

