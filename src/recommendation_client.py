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
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
        REDIS_DB = int(os.getenv("REDIS_DB", "0"))

        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        return client        

    # -------------------------
    # INGESTION
    # -------------------------
    def ingest_dataset(self, json_path: str, dataset_id: Optional[str] = None):
        """
        Load a JSON file where keys are item_ids and values are lists of recommended item_ids.

        Example JSON:
        {
            "item_123": ["item_456", "item_789"],
            "item_999": ["item_100", "item_101"]
        }
        """
        # Default dataset_id to filename if not provided
        if dataset_id is None:
            dataset_id = os.path.basename(json_path).split(".")[0]

        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON must contain a dictionary at its top level.")

        count = 0
        for item_id, recs in data.items():
            if not isinstance(recs, list):
                raise ValueError(f"Value for '{item_id}' must be a list.")

            key = self._key(dataset_id, item_id)

            if recs:
                # Add all recommended item_ids to the Redis set
                self.r.sadd(key, *recs)
            else:
                # Ensure empty sets exist via a placeholder
                self.r.sadd(key, "")

            count += 1

        return f"Ingested {count} items into dataset '{dataset_id}'."

    # -------------------------
    # QUERYING
    # -------------------------
    def get_recommendations(self, dataset_id: str, item_id: str) -> Set[str]:
        """
        Return all recommended item_ids for the given input item.
        Accepts a single dataset_id (str).
        """
        print(f"🔍 Fetching recommendations for dataset_id='{dataset_id}', item_id='{item_id}'...")
        key = self._key(dataset_id, item_id)
        recs = self.r.smembers(key)
        return {r for r in recs if r != ""}

    def list_items(self, dataset_id: str) -> List[str]:
        """List all item_ids in a given dataset."""
        pattern = f"recommendations:{dataset_id}:*"
        keys = self.r.keys(pattern)
        # Split by ':' and take the last part (item_id)
        return [key.split(":", 2)[-1] for key in keys]

    def list_datasets(self) -> List[str]:
        """List all available dataset IDs."""
        keys = self.r.keys("recommendations:*")
        datasets = {key.split(":")[1] for key in keys}
        return sorted(datasets)

    def find_items_recommending(self, dataset_id: str, target_item_id: str) -> Set[str]:
        """
        Return a set of all item_ids that list `target_item_id` as a recommendation.
        """
        pattern = f"recommendations:{dataset_id}:*"
        keys = self.r.keys(pattern)

        referring_items = set()

        for key in keys:
            source_item_id = key.split(":", 2)[-1]
            recs = self.r.smembers(key)

            if target_item_id in recs:
                referring_items.add(source_item_id)

        return referring_items

    def check_connection(self) -> bool:
        """Pings the Redis server and returns True if successful."""
        try:
            if self.r.ping():
                print(f"✅ Successfully connected to Redis")
                return True
            else:
                print("❌ Redis connection failed.")
                return False
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Redis Connection Error: {e}")
            return False
        
    def remove_old_recommendations(self, dataset_id: str) -> int:
        """
        Removes all keys following the old pattern 'recommendations:<dataset_id>:*'.
        Returns the total number of keys deleted.
        """
        pattern = f"recommendations:{dataset_id}:*"
        cursor = 0
        total_deleted = 0
        
        print(f"🧹 Starting cleanup for dataset: {dataset_id}...")

        while True:
            # SCAN is safer than KEYS in production as it doesn't block the server
            cursor, keys = self.r.scan(cursor=cursor, match=pattern, count=100)
            
            if keys:
                # Delete the batch of keys
                deleted_count = self.r.delete(*keys)
                total_deleted += deleted_count
            
            if cursor == 0:
                break
                
        print(f"✅ Cleanup complete. Removed {total_deleted} keys.")
        return total_deleted
    
    # -------------------------
    # INTERNAL UTILITIES
    # -------------------------
    @staticmethod
    def _key(dataset_id: str, item_id: str) -> str:
        return f"recommendations:{dataset_id}:{item_id}"