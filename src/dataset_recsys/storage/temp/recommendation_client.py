import json
import os
from typing import List, Set, Optional, Union
import redis

class RecommendationClient:
    """
    Client for ingesting and querying item recommendation mappings using Redis Sets.

    Redis key pattern:
        recommendations:<dataset_id>:<item_id>
    """

    def __init__(self):
        self.r = self.get_redis_client()

    def get_redis_client(self) -> redis.Redis:
        REDIS_HOST = os.getenv("REDIS_HOST", "redis")
        REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        REDIS_DB = int(os.getenv("REDIS_DB", "0"))

        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        return client        

    # -------------------------
    # INGESTION
    # -------------------------
    def ingest_dataset(self, json_path: str, dataset_id: Optional[str] = None):
        """
        Load a JSON file. The key in Redis will be the dataset_id, 
        and the value will be a Set of recommended dataset IDs found in the JSON.
        """
        if dataset_id is None:
            dataset_id = os.path.basename(json_path).split(".")[0]

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract unique recommended IDs from the JSON (works for lists or dict values)
        recs = []
        if isinstance(data, list):
            recs = data
        elif isinstance(data, dict):
            # If the JSON is still in the old item_id: [recs] format, 
            # we flatten all recommendations into a single set for the dataset.
            for value in data.values():
                if isinstance(value, list):
                    recs.extend(value)
                else:
                    recs.append(value)

        # Filter out empty strings/None and cast to set for uniqueness
        recs_to_add = {str(r) for r in recs if r}

        if recs_to_add:
            # Use the dataset_id directly as the key
            self.r.sadd(dataset_id, *recs_to_add)
        
        return f"Ingested {len(recs_to_add)} recommendations into dataset key '{dataset_id}'."

    # -------------------------
    # QUERYING
    # -------------------------
    def get_recommendations(self, dataset_id: str) -> Set[str]:
        """Return all recommended IDs stored under this dataset_id key."""
        return self.r.smembers(dataset_id)

    def list_datasets(self) -> List[str]:
        """List all keys in the current Redis DB."""
        # Note: In a dedicated DB, this returns all dataset_ids.
        return sorted(self.r.keys("*"))

    def find_items_recommending(self, target_dataset_id: str) -> Set[str]:
        """Find which dataset keys contain target_dataset_id in their Set."""
        all_keys = self.r.keys("*")
        referring_datasets = set()

        for key in all_keys:
            if self.r.sismember(key, target_dataset_id):
                referring_datasets.add(key)

        return referring_datasets
   
    # -------------------------
    # UTILITIES
    # -------------------------
    def remove_old_recommendations(self, dataset_id: str) -> bool:
        """Deletes the dataset key."""
        return bool(self.r.delete(dataset_id))

    def check_connection(self) -> bool:
        try:
            return bool(self.r.ping())
        except redis.exceptions.ConnectionError:
            return False