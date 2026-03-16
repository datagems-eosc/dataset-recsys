import json
import os
from typing import List, Set

import redis


class RecommendationClient:
    """
    Redis client for storing and querying entity-to-entity recommendations.

    Storage model:
        Redis key   -> recs:{application}:{entity_id}
        Redis value -> Set of recommended entity IDs

    Index model:
        Redis key   -> recs:index:{application}
        Redis value -> Set of entity IDs stored for that application

    Examples:
        recs:mathe:6.pdf -> {7.pdf, 9.pdf, 221.pdf}
        recs:portal:meteo_era5land -> {weather_stations_climpact, wikipedia}
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

    # -------------------------
    # INGESTION
    # -------------------------
    def ingest_dataset(self, json_path: str, application: str) -> str:
        """
        Load a JSON file containing entity-to-entity recommendations for one application.

        The JSON is expected to have the form:
            entity_id -> list of recommended entity IDs

        Existing recommendations for the application are replaced on each ingestion.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "Expected recommendation JSON to be a dict of entity_id -> recommended entity IDs."
            )

        self.delete_application(application)

        index_key = self._index_key(application)
        stored_entities = 0

        for entity_id, recommended_ids in data.items():
            rec_key = self._recommendation_key(application, str(entity_id))

            if isinstance(recommended_ids, list):
                recs_to_add = {str(rec_id) for rec_id in recommended_ids if rec_id}
            else:
                recs_to_add = {str(recommended_ids)} if recommended_ids else set()

            self.r.sadd(index_key, str(entity_id))
            stored_entities += 1

            if recs_to_add:
                self.r.sadd(rec_key, *recs_to_add)

        return (
            f"Stored recommendations for {stored_entities} entities "
            f"under application '{application}'."
        )

    # -------------------------
    # QUERYING
    # -------------------------
    def get_recommendations(self, application: str, entity_id: str) -> Set[str]:
        """Return recommended entity IDs for one entity within an application."""
        return self.r.smembers(self._recommendation_key(application, entity_id))

    # TODO: Check whether this returns only the entity ids that have recommendations, or also those with empty recommendation sets. If the latter, we may want to filter those out.
    def list_entities(self, application: str) -> List[str]:
        """List entity IDs currently stored for an application."""
        return sorted(self.r.smembers(self._index_key(application)))

    def find_entities_recommending(self, application: str, target_entity_id: str) -> Set[str]:
        """Find which entities recommend a given target entity within an application."""
        referring_entities = set()

        for entity_id in self.r.smembers(self._index_key(application)):
            rec_key = self._recommendation_key(application, entity_id)
            if self.r.sismember(rec_key, target_entity_id):
                referring_entities.add(entity_id)

        return referring_entities

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