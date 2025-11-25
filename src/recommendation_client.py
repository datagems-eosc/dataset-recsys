import json
import os
from typing import List, Set, Dict, Optional
import redis
import os
from dotenv import load_dotenv

def load_repo_env():
    """Walk up directories until .env is found and load it."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(".env file not found in any parent directory.")
        current_dir = parent_dir

# Load .env once when the module is imported
load_repo_env()

REDIS_URL = os.getenv("REDIS_URL")

class RecommendationClient:
    """
    Client for ingesting and querying PDF recommendation mappings using Redis Sets.

    Redis key pattern:
        recommendations:<usecase>:<pdf_name>
    """

    def __init__(self):
        self.r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    # -------------------------
    # INGESTION
    # -------------------------
    def ingest_json(self, json_path: str, usecase: Optional[str] = None):
        """
        Load a JSON file where keys are pdf names and values are lists of recommended PDFs.

        Example JSON:
        {
            "6.pdf": ["7.pdf", "9.pdf"],
            "65.pdf": ["67.pdf", "66.pdf"]
        }
        """
        if usecase is None:
            usecase = os.path.basename(json_path).split(".")[0]

        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON must contain a dictionary at its top level.")

        count = 0
        for pdf, recs in data.items():
            if not isinstance(recs, list):
                raise ValueError(f"Value for '{pdf}' must be a list.")

            key = self._key(usecase, pdf)

            if recs:
                self.r.sadd(key, *recs)
            else:
                # ensure empty sets exist
                self.r.sadd(key, "")

            count += 1

        return f"Ingested {count} PDFs into usecase '{usecase}'."

    # -------------------------
    # QUERYING
    # -------------------------
    def get_recommendations(self, usecase: str, pdf: str) -> Set[str]:
        """Return all recommended PDFs for the given input PDF."""
        key = self._key(usecase, pdf)
        recs = self.r.smembers(key)
        return {r for r in recs if r != ""}  # filter placeholder

    def list_pdfs(self, usecase: str) -> List[str]:
        """List all PDFs in a given usecase."""
        pattern = f"recommendations:{usecase}:*"
        keys = self.r.keys(pattern)
        return [key.split(":", 2)[-1] for key in keys]

    def list_usecases(self) -> List[str]:
        """List all available usecases."""
        keys = self.r.keys("recommendations:*")
        usecases = {key.split(":")[1] for key in keys}
        return sorted(usecases)

    def find_entries_recommending(self, usecase: str, pdf: str) -> Set[str]:
        """
        Return a set of all PDFs that list `pdf` as a recommendation.

        Example:
            If 7.pdf appears inside:
                - recommendations:taxguides:6.pdf
                - recommendations:taxguides:22.pdf

            then calling:
                find_entries_recommending("taxguides", "7.pdf")
            returns:
                {"6.pdf", "22.pdf"}
        """
        pattern = f"recommendations:{usecase}:*"
        keys = self.r.keys(pattern)

        referring_pdfs = set()

        for key in keys:
            source_pdf = key.split(":", 2)[-1]
            recs = self.r.smembers(key)

            if pdf in recs:
                referring_pdfs.add(source_pdf)

        return referring_pdfs
    

    # -------------------------
    # INTERNAL UTILITIES
    # -------------------------
    @staticmethod
    def _key(usecase: str, pdf: str) -> str:
        return f"recommendations:{usecase}:{pdf}"
