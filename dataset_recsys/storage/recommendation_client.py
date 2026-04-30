import json
import os
from typing import Dict, List

import redis


class RecommendationClient:
    """
    Redis client for storing and querying entity-to-entity recommendations.

    Storage model:
        Redis key   -> recs:{application}:{entity_id}
        Redis value -> Sorted Set (ZSET) of recommended entity IDs scored by relevance

    Index model:
        Redis key   -> recs:index:{application}
        Redis value -> Set of entity IDs stored for that application

    Example usage:
        recs:mathe:6.pdf -> {7.pdf: 0.91, 9.pdf: 0.87, 221.pdf: 0.72}
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", "6380"))
        self.db = int(db or os.getenv("REDIS_DB", "0"))
        self.r = self.get_redis_client()

    def get_redis_client(self) -> redis.Redis:
        return redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
        )

    def _recommendation_key(self, application: str, entity_id: str) -> str:
        return f"recs:{application}:{entity_id}"

    def _index_key(self, application: str) -> str:
        return f"recs:index:{application}"

    # TODO: In the future, we will keep only dicts of entity_id -> score, 
    # and remove support for ranked lists without explicit scores.
    def _normalize_recommendations(self, items) -> Dict[str, float]:
        """Convert input into {entity_id: score} mapping."""
        if isinstance(items, dict):
            return {str(k): float(v) for k, v in items.items() if k}
        elif isinstance(items, list):
            n = len(items)
            return {str(k): float(n - i) for i, k in enumerate(items) if k}
        else:
            raise ValueError(
                "Expected recommendations to be a dict of entity_id -> score, "
                "or a list of entity_ids ranked by relevance."
            )

    def store_recommendations(self, application: str, data) -> int:
        """Store scored recommendations for one application."""
        self.delete_application(application)
        index_key = self._index_key(application)
        stored_entities = 0

        # --- POLYMORPHIC PRE-PROCESSING ---
        # If input is the Claude-style list: [{"id": "...", "recommendations": [...]}, ...]
        if isinstance(data, list):
            items_to_process = {}
            for entry in data:
                eid = entry.get("id")
                recs = entry.get("recommendations", [])
                
                # If the nested recs are objects with scores, 
                # convert them to a dict before passing to normalize
                if recs and isinstance(recs[0], dict):
                    items_to_process[eid] = {
                        str(r["id"]): float(r.get("score", 0.0)) 
                        for r in recs if "id" in r
                    }
                else:
                    # Otherwise, just pass the list (e.g., list of strings)
                    items_to_process[eid] = recs
        
        # If input is the Legacy dict: {"6.pdf": ["7.pdf", ...]}
        elif isinstance(data, dict):
            items_to_process = data
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")

        # --- STORAGE LOOP ---
        for entity_id, recommended_items in items_to_process.items():
            if not entity_id:
                continue

            rec_key = self._recommendation_key(application, str(entity_id))
            recs_to_add = self._normalize_recommendations(recommended_items)

            self.r.sadd(index_key, str(entity_id))
            stored_entities += 1

            if recs_to_add:
                self.r.zadd(rec_key, recs_to_add)

        return stored_entities

    # -------------------------
    # INGESTION
    # -------------------------
    def ingest_dataset(self, json_path: str, application: str) -> str:
        """
        Load a JSON file containing entity-to-entity recommendations for one application.

        The JSON is expected to have one of the following forms:
            entity_id -> list of recommended entity IDs
            entity_id -> dict of recommended entity ID -> score

        Existing recommendations for the application are replaced on each ingestion.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stored_entities = self.store_recommendations(application, data)

        return (
            f"Stored recommendations for {stored_entities} entities "
            f"under application '{application}'."
        )

    def update_single_entity_recs(self, application: str, entity_id: str, recommendations: Dict[str, float]):
        """Update only one ZSET and ensure the ID is in the application index."""
        rec_key = self._recommendation_key(application, entity_id)
        index_key = self._index_key(application)
        
        with self.r.pipeline() as pipe:
            pipe.sadd(index_key, entity_id)
            pipe.delete(rec_key) # Refresh this specific entity
            if recommendations:
                pipe.zadd(rec_key, recommendations)
            pipe.execute()

    def update_neighbor_recs(self, application: str, neighbor_id: str, new_entity_id: str, score: float, limit: int = 20):
        """
        Inject a new entity into an existing neighbor's recommendation list.
        Keeps the ZSET capped at 'limit'.
        """
        rec_key = self._recommendation_key(application, neighbor_id)
        
        with self.r.pipeline() as pipe:
            pipe.zadd(rec_key, {new_entity_id: score})
            # Remove elements outside the top N to keep Redis lean
            pipe.zremrangebyrank(rec_key, 0, -(limit + 1)) 
            pipe.execute()

    def remove_single_entity_recs(self, application: str, entity_id: str):
        """Deletes an entity's own rec list and removes it from the application index."""
        rec_key = self._recommendation_key(application, entity_id)
        index_key = self._index_key(application)
        
        with self.r.pipeline() as pipe:
            pipe.delete(rec_key)
            pipe.srem(index_key, entity_id)
            pipe.execute()

    def remove_from_neighbor_recs(self, application: str, neighbor_id: str, target_id: str):
        """Removes target_id from a specific neighbor's recommendation ZSET."""
        rec_key = self._recommendation_key(application, neighbor_id)
        self.r.zrem(rec_key, target_id)

    # -------------------------
    # QUERYING
    # -------------------------
    def get_recommendations(self, application: str, entity_id: str) -> List[str]:
        """
        Return recommended entity IDs for one entity, ordered by score descending.
        Returns an empty list if the entity_id does not exist or has no recs.
        """
        if not entity_id:
            return []
            
        key = self._recommendation_key(application, entity_id)
        # ZREVRANGE returns items from highest score to lowest
        return self.r.zrevrange(key, 0, -1)

    # TODO: Check whether this returns only the entity IDs that have recommendations, or also those with empty recommendation ZSETs. If the latter, we may want to filter those out.
    def list_entities(self, application: str) -> List[str]:
        """List entity IDs currently stored for an application."""
        return sorted(self.r.smembers(self._index_key(application)))

    def find_entities_recommending(self, application: str, target_entity_id: str) -> List[str]:
        """Find which entities recommend a given target entity within an application."""
        referring_entities = []

        for entity_id in self.r.smembers(self._index_key(application)):
            rec_key = self._recommendation_key(application, entity_id)
            if self.r.zscore(rec_key, target_entity_id) is not None:
                referring_entities.append(entity_id)

        return sorted(referring_entities)

    def check_existence_batch(self, application: str, entity_ids: List[str]) -> Dict[str, bool]:
        """
        Checks if a list of entity_ids exist in the recommendation catalog for the given application.
        Returns a dictionary mapping entity_id to a boolean.
        """
        index_key = self._index_key(application)
        
        # Use a pipeline for atomic/batch execution
        with self.r.pipeline() as pipe:
            for eid in entity_ids:
                pipe.sismember(index_key, eid)
            results = pipe.execute()
        
        return {eid: bool(res) for eid, res in zip(entity_ids, results)}

    # -------------------------
    # UTILITIES
    # -------------------------

    # TODO: If recommendation recomputation is introduced later, call it after cleanup.
    def remove_dataset(self, application: str, entity_id: str) -> int:
        """Remove one dataset completely from an application.

        This deletes:
        - its own recommendation key
        - its entry in the application index
        - any references to it inside other entities' recommendation sets

        Returns the number of Redis keys deleted for the entity itself
        (0 or 1). References removed from other ZSETs are not counted here.
        """
        if not entity_id:
            return 0

        deleted = int(self.r.delete(self._recommendation_key(application, entity_id)))
        self.r.srem(self._index_key(application), entity_id)

        for other_entity_id in self.r.smembers(self._index_key(application)):
            rec_key = self._recommendation_key(application, other_entity_id)
            self.r.zrem(rec_key, entity_id)

        return deleted

    def delete_application(self, application: str) -> int:
        """Delete all recommendation keys stored for one application."""
        entity_ids = self.r.smembers(self._index_key(application))

        deleted = 0
        for entity_id in entity_ids:
            deleted += int(self.r.delete(self._recommendation_key(application, entity_id)))

        if entity_ids:
            self.r.delete(self._index_key(application))

        return deleted

    def check_connection(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return bool(self.r.ping())
        except redis.exceptions.ConnectionError:
            return False