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

    def __init__(self):
        self.r = self.get_redis_client()

    def get_redis_client(self) -> redis.Redis:
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))

        return redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
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

    def store_recommendations(self, application: str, recommendations: Dict[str, Dict[str, float]]) -> int:
        """Store scored recommendations for one application."""
        self.delete_application(application)
        index_key = self._index_key(application)
        stored_entities = 0

        for entity_id, recommended_items in recommendations.items():
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

        if not isinstance(data, dict):
            raise ValueError(
                "Expected JSON to be a dict."
            )

        stored_entities = self.store_recommendations(application, data)

        return (
            f"Stored recommendations for {stored_entities} entities "
            f"under application '{application}'."
        )

    # -------------------------
    # QUERYING
    # -------------------------
    def get_recommendations(self, application: str, entity_id: str) -> List[str]:
        """Return recommended entity IDs for one entity within an application."""
        return self.r.zrevrange(
            self._recommendation_key(application, entity_id), 0, -1, 
        )

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

    # -------------------------
    # UTILITIES
    # -------------------------
    def delete_entity(self, application: str, entity_id: str) -> bool:
        """Delete all recommendations stored for one entity."""
        deleted = bool(self.r.delete(self._recommendation_key(application, entity_id)))
        self.r.srem(self._index_key(application), entity_id)
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