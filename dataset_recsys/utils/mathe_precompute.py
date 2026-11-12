"""Refresh the precomputed MathE question rankings in the configured Qdrant."""

import argparse
import json
import logging
import os
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    load_dotenv()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO)

    from dataset_recsys.mathe_recommenders.precomputed_rankings import refresh_question_rankings
    from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
    from dataset_recsys.storage.qdrant_client import QdrantStorageClient

    status_file = Path(os.getenv("MATHE_PATH", "s3/default")) / "sync_status.json"

    def read_sync_status() -> dict:
        try:
            return json.loads(status_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"sync_status": "never_run", "last_sync_started_at": None}

    with (
        closing(MatheMirrorClient()) as mirror,
        closing(QdrantStorageClient()) as qdrant,
    ):
        refresh_question_rankings(mirror, qdrant, read_sync_status)


if __name__ == "__main__":
    main()
