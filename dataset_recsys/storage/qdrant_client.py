import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStorageClient:
    """
    Qdrant client for storing vector embeddings and metadata, 
    serving vector similarity searches, and storing precomputed recommendations.
    """

    COLLECTION_DATASET = "dataset_embeddings"
    COLLECTION_MATHE = "mathe_embeddings"
    MATHE_QUESTION_RANKINGS = "mathe_question_rankings"
    SCROLL_PAGE_SIZE = 1000

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        """Connect to Qdrant, or wrap an existing ``client`` (used by tests)."""
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", None)

        self.client = client or QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )
        # Collections already seen to exist, so reads only ask Qdrant once.
        self._known_collections: set[str] = set()

    def _collection_exists(self, collection_name: str) -> bool:
        """Return whether a collection exists, remembering positive answers.

        Before the first rebuild Qdrant has no collections. Reads treat a
        missing collection as "nothing stored" instead of failing.
        """
        if collection_name in self._known_collections:
            return True
        if self.client.collection_exists(collection_name):
            self._known_collections.add(collection_name)
            return True
        return False

    def _retrieve_point(self, collection_name: str, entity_id: str) -> Any | None:
        """Return one entity's point with its payload, or None if not stored."""
        if not self._collection_exists(collection_name):
            return None
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=[self._generate_point_id(entity_id)],
            with_payload=True,
        )
        return points[0] if points else None

    def get_mathe_question_ranking(
        self, question_id: int, content_type: str
    ) -> dict[str, Any] | None:
        """Read the active question ranking without transferring its candidate scores."""
        collection = self.MATHE_QUESTION_RANKINGS
        if not 0 <= question_id < 2**64 or not self._collection_exists(collection):
            return None
        points = self.client.retrieve(
            collection_name=collection,
            ids=[question_id],
            with_vectors=False,
            with_payload=[
                "question_hash", "sync_generation", "scoring_version", content_type
            ],
        )
        return points[0].payload if points else None

    def _resolve_collection(self, application: str, explicit_collection: Optional[str] = None) -> str:
        """Route MathE applications to COLLECTION_MATHE by default if unspecified."""
        if explicit_collection:
            return explicit_collection
        if str(application).lower().startswith("mathe"):
            return self.COLLECTION_MATHE
        return self.COLLECTION_DATASET

    def _generate_point_id(self, entity_id: str) -> str:
        """Generate a deterministic UUID v5 from an arbitrary entity_id string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, entity_id))

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        distance: models.Distance = models.Distance.COSINE,
    ):
        """Create the collection if it is missing.

        An existing collection is never deleted here. If its vector size
        differs from ``vector_size``, the write is refused with a ValueError:
        the collection also holds other applications' embeddings and
        recommendations, so replacing it must be a deliberate step (drop the
        collection, then run a full rebuild).
        """
        collections = [col.name for col in self.client.get_collections().collections]

        if collection_name in collections and vector_size is not None:
            info = self.client.get_collection(collection_name)
            
            # Extract current vector size safely
            vectors_config = info.config.params.vectors
            current_size = getattr(vectors_config, "size", None)

            # If named vectors dict structure is used by Qdrant
            if current_size is None and isinstance(vectors_config, dict):
                first_vector = next(iter(vectors_config.values()), None)
                current_size = getattr(first_vector, "size", None)

            if current_size is not None and current_size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{collection_name}' stores vectors of size "
                    f"{current_size}, but embeddings of size {vector_size} were "
                    "provided. Nothing was written. Check that the embeddings are "
                    "sent to the right collection; to change the embedding model, "
                    "drop the collection deliberately and run a full rebuild."
                )

        if collection_name not in collections:
            # Fallback to 768 if vector_size was not specified on initial creation
            target_size = vector_size or 768
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=target_size,
                    distance=distance,
                ),
            )
            # Re-create payload index for fast application filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="application",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    # -------------------------------------------------------------------------
    # 1. EMBEDDING STORAGE & PARITY (Replaces pgvector)
    # -------------------------------------------------------------------------

    def store_embeddings(
        self,
        application: str,
        dataset_ids: List[str],
        embeddings: Any,
        embedding_inputs: List[str],
        embedding_model: str,
        table: Optional[str] = None,  # pgvector parity
        collection_name: Optional[str] = None,
        run_id: Optional[str] = None,
        replace: bool = False,
        **kwargs,
    ) -> int:
        """Upsert vector embeddings with metadata into Qdrant.

        By default this only adds or overwrites the given entities. A full
        rebuild passes ``replace=True``: after the new embeddings are written,
        the application's entities that are not in ``dataset_ids`` are deleted,
        so entities removed from the catalog do not stay recommendable. Other
        applications in the collection are never touched.

        An empty batch writes and deletes nothing, even with ``replace=True``;
        call ``delete_application`` to clear an application on purpose.
        """
        if len(embeddings) == 0:
            return 0

        target_collection = self._resolve_collection(application, collection_name or table)
        vector_size = len(embeddings[0])
        self.ensure_collection(target_collection, vector_size=vector_size)

        points = []
        for entity_id, embedding, text in zip(dataset_ids, embeddings, embedding_inputs):
            point_id = self._generate_point_id(entity_id)
            vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

            payload = {
                "application": application,
                "dataset_id": entity_id,
                "embedding_input": text,
                "embedding_model": embedding_model,
                "enrichment_llm": kwargs.get("enrichment_llm", "none"),
                "prompt_version": kwargs.get("prompt_version", "none"),
                "run_id": run_id,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=target_collection, points=points)
        if replace:
            self.delete_entities_not_in(application, dataset_ids, target_collection)
        return len(points)

    def delete_entities_not_in(
        self,
        application: str,
        keep_entity_ids: List[str],
        collection_name: str = COLLECTION_DATASET,
    ) -> int:
        """Delete an application's entities whose IDs are not in ``keep_entity_ids``.

        Returns the number of entities removed.
        """
        if not self._collection_exists(collection_name):
            return 0

        conditions: Dict[str, Any] = {
            "must": [
                models.FieldCondition(
                    key="application",
                    match=models.MatchValue(value=application),
                )
            ]
        }
        if keep_entity_ids:
            conditions["must_not"] = [
                models.HasIdCondition(
                    has_id=[self._generate_point_id(eid) for eid in keep_entity_ids]
                )
            ]
        stale_filter = models.Filter(**conditions)

        stale_count = self.client.count(
            collection_name=collection_name,
            count_filter=stale_filter,
            exact=True,
        ).count
        if stale_count:
            self.client.delete(
                collection_name=collection_name,
                points_selector=stale_filter,
            )
        return stale_count

    def upsert_single_embedding(
        self, 
        application: str, 
        dataset_id: str, 
        embedding: List[float], 
        embedding_input: str,
        metadata: dict,
        collection_name: str = COLLECTION_DATASET,
    ) -> None:
        """Upsert a single vector and metadata (pgvector parity)."""
        # The first entity may arrive before any rebuild created the collection.
        self.ensure_collection(collection_name, vector_size=len(embedding))
        point_id = self._generate_point_id(dataset_id)
        payload = {
            "application": application,
            "dataset_id": dataset_id,
            "embedding_input": embedding_input,
            "embedding_model": metadata.get("model"),
            "enrichment_llm": metadata.get("llm"),
            "prompt_version": metadata.get("prompt"),
            "run_id": metadata.get("run_id"),
        }
        self.client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(id=point_id, vector=embedding, payload=payload)]
        )

    def exists(self, dataset_id: str, collection_name: str = COLLECTION_DATASET) -> bool:
        """Check if a dataset_id already has an embedding record."""
        return self._retrieve_point(collection_name, dataset_id) is not None

    def find_similar(
        self,
        application: str,
        query_embedding: Optional[List[float]] = None,
        query_vector: Optional[List[float]] = None,
        top_k: Optional[int] = 10,
        table: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search an application namespace for the entities most similar to a vector.

        With an integer ``top_k`` this is an approximate nearest-neighbour
        search. ``top_k=None`` returns the complete ranking of every entity in
        the application, computed exactly, which is what the incremental
        updates store so that they match a full rebuild.
        """
        target_collection = collection_name or table or self.COLLECTION_DATASET
        vector = query_embedding if query_embedding is not None else query_vector

        if vector is None:
            raise ValueError("Either 'query_embedding' or 'query_vector' must be provided.")
        if not self._collection_exists(target_collection):
            return []

        search_params = None
        if top_k is None:
            top_k = self.count_entities(application, target_collection)
            if top_k == 0:
                return []
            search_params = models.SearchParams(exact=True)

        # query_points returns a QueryResponse object
        response = self.client.query_points(
            collection_name=target_collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="application",
                        match=models.MatchValue(value=application),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            search_params=search_params,
        )

        # Access the scored points via `response.points`
        return [
            {
                "dataset_id": (res.payload or {}).get("dataset_id"),
                "score": res.score,
                "payload": res.payload,
            }
            for res in response.points
        ]

    def count_entities(self, application: str, collection_name: str = COLLECTION_DATASET) -> int:
        """Count the entities stored for an application."""
        if not self._collection_exists(collection_name):
            return 0
        return self.client.count(
            collection_name=collection_name,
            count_filter=models.Filter(must=[
                models.FieldCondition(key="application", match=models.MatchValue(value=application))
            ]),
            exact=True,
        ).count

    def find_similar_by_ids(
        self,
        application: str,
        query_embedding: List[float],
        entity_ids: List[str],
        table: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Tuple[Optional[str], float]]:
        """Return query-vector similarities for the requested IDs."""
        if not entity_ids:
            return []

        target_collection = collection_name or table or self.COLLECTION_DATASET
        if not self._collection_exists(target_collection):
            return []
        point_ids = [self._generate_point_id(eid) for eid in entity_ids]

        response = self.client.query_points(
            collection_name=target_collection,
            query=query_embedding,  # Renamed from `query_vector` to `query`
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="application",
                        match=models.MatchValue(value=application),
                    ),
                    models.HasIdCondition(has_id=point_ids),
                ]
            ),
            limit=len(entity_ids),
            with_payload=True,
        )

        return [
            ((res.payload or {}).get("dataset_id"), res.score)
            for res in response.points  # Access `.points` attribute
        ]

    def delete_single_embedding(self, dataset_id: str, collection_name: str = COLLECTION_DATASET) -> int:
        """Delete a single embedding from the database."""
        if not self._collection_exists(collection_name):
            return 0
        point_id = self._generate_point_id(dataset_id)
        res = self.client.delete(collection_name=collection_name, points_selector=[point_id])
        return 1 if res.status == models.UpdateStatus.COMPLETED else 0

    def delete_application(self, application: str, table: Optional[str] = None, collection_name: Optional[str] = None) -> int:
        """Delete all points and recommendations belonging to an application."""
        target_collection = collection_name or table or self.COLLECTION_DATASET
        if not self._collection_exists(target_collection):
            return 0
        res = self.client.delete(
            collection_name=target_collection,
            points_selector=models.Filter(must=[
                models.FieldCondition(key="application", match=models.MatchValue(value=application))
            ])
        )
        return 1 if res.status == models.UpdateStatus.COMPLETED else 0

    # -------------------------------------------------------------------------
    # 2. RECOMMENDATION STORAGE & PARITY (Replaces Redis ZSETs)
    # -------------------------------------------------------------------------

    def store_recommendations(
        self,
        application: str,
        data: Optional[Any] = None,
        recommendations: Optional[Any] = None,
        collection_name: str = COLLECTION_DATASET,
    ) -> int:
        """Store top-N precomputed recommendations (Redis parity)."""
        input_data = data if data is not None else recommendations
        if not input_data:
            return 0

        items_to_process = {}
        if isinstance(input_data, list):
            for entry in input_data:
                eid = entry.get("id")
                recs = entry.get("recommendations", [])
                if recs and isinstance(recs[0], dict):
                    items_to_process[eid] = {str(r["id"]): float(r.get("score", 0.0)) for r in recs if "id" in r}
                else:
                    items_to_process[eid] = recs
        elif isinstance(input_data, dict):
            items_to_process = input_data
        else:
            raise ValueError(f"Unsupported JSON structure: {type(input_data)}")

        stored_entities = 0
        for entity_id, recs in items_to_process.items():
            if not entity_id:
                continue

            point_id = self._generate_point_id(str(entity_id))
            formatted_recs = []
            
            # Normalize recommendations
            if isinstance(recs, dict):
                formatted_recs = [{"id": str(k), "score": float(v)} for k, v in recs.items()]
            elif isinstance(recs, list):
                for rank, r in enumerate(recs):
                    if isinstance(r, dict):
                        formatted_recs.append({"id": str(r.get("id")), "score": float(r.get("score", 1.0))})
                    elif isinstance(r, (tuple, list)):
                        formatted_recs.append({"id": str(r[0]), "score": float(r[1] if len(r) > 1 else 1.0)})
                    else:
                        formatted_recs.append({"id": str(r), "score": float(len(recs) - rank)}) # Legacy rank score

            self.client.set_payload(
                collection_name=collection_name,
                payload={"recommendations": {application: formatted_recs}},
                points=[point_id],
            )
            stored_entities += 1

        return stored_entities

    def ingest_dataset(self, json_path: str, application: str, collection_name: str = COLLECTION_DATASET) -> str:
        """Load a JSON file containing entity-to-entity recommendations."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stored = self.store_recommendations(application, data=data, collection_name=collection_name)
        return f"Stored recommendations for {stored} entities under application '{application}'."

    def update_single_entity_recs(self, application: str, entity_id: str, recommendations: Dict[str, float], collection_name: str = COLLECTION_DATASET):
        """Update only one recommendation list."""
        point_id = self._generate_point_id(entity_id)
        formatted = [{"id": str(k), "score": float(v)} for k, v in recommendations.items()]
        
        self.client.set_payload(
            collection_name=collection_name,
            payload={"recommendations": {application: formatted}},
            points=[point_id]
        )

    def update_neighbor_recs(
        self,
        application: str,
        neighbor_id: str,
        new_entity_id: str,
        score: float,
        limit: Optional[int] = None,
        collection_name: str = COLLECTION_DATASET,
    ):
        """Inject a new entity into an existing neighbor's recommendation list."""
        point_id = self._generate_point_id(neighbor_id)
        point = self._retrieve_point(collection_name, neighbor_id)

        if point is None:
            return

        current_recs = point.payload.get("recommendations", {}).get(application, [])
        recs_map = {str(r["id"]): float(r["score"]) for r in current_recs}
        recs_map[str(new_entity_id)] = float(score)

        sorted_recs = sorted(recs_map.items(), key=lambda item: item[1], reverse=True)
        if limit is not None:
            sorted_recs = sorted_recs[:limit]

        formatted = [{"id": k, "score": v} for k, v in sorted_recs]
        self.client.set_payload(
            collection_name=collection_name,
            payload={"recommendations": {application: formatted}},
            points=[point_id]
        )

    def remove_single_entity_recs(self, application: str, entity_id: str, collection_name: str = COLLECTION_DATASET):
        """Deletes an entity's own rec list by overwriting the payload."""
        point_id = self._generate_point_id(entity_id)
        self.client.set_payload(
            collection_name=collection_name,
            payload={"recommendations": {application: []}},
            points=[point_id]
        )

    def remove_from_neighbor_recs(self, application: str, neighbor_id: str, target_id: str, collection_name: str = COLLECTION_DATASET):
        """Removes target_id from a specific neighbor's recommendation payload."""
        point_id = self._generate_point_id(neighbor_id)
        point = self._retrieve_point(collection_name, neighbor_id)

        if point is None:
            return

        current_recs = point.payload.get("recommendations", {}).get(application, [])
        filtered = [r for r in current_recs if str(r["id"]) != str(target_id)]
        
        self.client.set_payload(
            collection_name=collection_name,
            payload={"recommendations": {application: filtered}},
            points=[point_id]
        )

    def remove_dataset(self, application: str, entity_id: str, collection_name: str = COLLECTION_DATASET) -> int:
        """Remove one dataset completely from an application (Point deletion + neighbor cleanup)."""
        # Remove point entirely
        deleted = self.delete_single_embedding(entity_id, collection_name)

        # Find entities recommending this target and remove references
        referencing_entities = self.find_entities_recommending(application, entity_id, collection_name)
        for neighbor in referencing_entities:
            self.remove_from_neighbor_recs(application, neighbor, entity_id, collection_name)

        return deleted

    # -------------------------------------------------------------------------
    # 3. RECOMMENDATION RETRIEVAL
    # -------------------------------------------------------------------------

    def get_entity_status(self, application: str, entity_id: str, collection_name: str = COLLECTION_DATASET) -> str:
        """Evaluates the existence and completeness of an entity."""
        if not entity_id:
            return "NOT_FOUND"

        point = self._retrieve_point(collection_name, entity_id)
        if point is None:
            return "NOT_FOUND"

        recs = (point.payload or {}).get("recommendations", {}).get(application, [])
        return "AVAILABLE" if recs else "NO_RECOMMENDATIONS"

    def _stored_recommendations(
        self,
        application: str,
        entity_id: str,
        collection_name: str,
    ) -> Optional[List[tuple[str, float]]]:
        """Return an entity's scored recommendations, best first.

        Returns None when the entity has no stored point, and an empty list
        when it is stored without recommendations for the application.
        """
        point = self._retrieve_point(collection_name, entity_id)
        if point is None:
            return None

        recs_by_app = (point.payload or {}).get("recommendations", {})
        app_recs = recs_by_app.get(application, [])
        return sorted(
            [(str(r["id"]), float(r["score"])) for r in app_recs],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_recommendations(
        self,
        application: str,
        entity_id: str,
        limit: Optional[int] = None,
        collection_name: str = COLLECTION_DATASET,
    ) -> List[str]:
        """Return recommended entity IDs for one entity (Redis parity).

        Raises ``KeyError`` when the entity is not stored at all. An entity
        that is stored but has no recommendations returns an empty list.
        """
        if not entity_id or (limit is not None and limit <= 0):
            return []

        recs = self._stored_recommendations(application, entity_id, collection_name)
        if recs is None:
            raise KeyError(f"Entity '{entity_id}' does not exist in backend.")
        return [rec_id for rec_id, _ in (recs[:limit] if limit else recs)]

    def get_recommendations_with_scores(
        self,
        application: str,
        entity_id: str,
        limit: Optional[int] = None,
        collection_name: str = COLLECTION_DATASET,
    ) -> List[tuple[str, float]]:
        """Return recommended entity IDs with their scores, best first.

        Returns an empty list for an entity that is not stored (Redis parity):
        callers such as the MathE seed expansion treat a material without
        stored neighbours as having none.
        """
        if not entity_id or (limit is not None and limit <= 0):
            return []

        recs = self._stored_recommendations(application, entity_id, collection_name) or []
        return recs[:limit] if limit else recs

    def _scroll_application(
        self,
        application: str,
        collection_name: str,
        payload_fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Return the requested payload fields of every entity of an application."""
        if not self._collection_exists(collection_name):
            return []

        payloads = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="application", match=models.MatchValue(value=application))
                ]),
                limit=self.SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=models.PayloadSelectorInclude(include=payload_fields),
                with_vectors=False,
            )
            payloads.extend(rec.payload for rec in records if rec.payload)
            if offset is None:
                break
        return payloads

    def list_entities(self, application: str, collection_name: str = COLLECTION_DATASET) -> List[str]:
        """List every entity ID currently stored for an application."""
        payloads = self._scroll_application(application, collection_name, ["dataset_id"])
        return sorted(
            payload["dataset_id"] for payload in payloads if "dataset_id" in payload
        )

    def find_entities_recommending(self, application: str, target_entity_id: str, collection_name: str = COLLECTION_DATASET) -> List[str]:
        """Find which entities recommend a given target entity within an application."""
        referring_entities = []
        entities = self.list_entities(application, collection_name)
        
        # In a massive dataset, this scroll check would need optimization via a dedicated lookup index. 
        # For parity, we iterate over application entities.
        for entity_id in entities:
            recs = self.get_recommendations(application, entity_id, collection_name=collection_name)
            if target_entity_id in recs:
                referring_entities.append(entity_id)

        return sorted(referring_entities)

    def check_existence_batch(self, application: str, entity_ids: List[str], collection_name: str = COLLECTION_DATASET) -> Dict[str, bool]:
        """Checks if a list of entity_ids exist in the catalog."""
        if not entity_ids or not self._collection_exists(collection_name):
            return {eid: False for eid in entity_ids}
        point_ids = [self._generate_point_id(eid) for eid in entity_ids]
        points = self.client.retrieve(collection_name=collection_name, ids=point_ids)
        
        found_point_ids = {p.id for p in points}
        return {eid: (self._generate_point_id(eid) in found_point_ids) for eid in entity_ids}

    # -------------------------------------------------------------------------
    # 4. UTILITIES
    # -------------------------------------------------------------------------

    def get_schema_overview(self, collection_name: str = COLLECTION_DATASET) -> dict:
        """Return collection configuration simulating a database schema."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "collection_name": collection_name,
                "vectors_config": info.config.params.vectors.model_dump()
            }
        except Exception as e:
            return {"error": str(e)}

    def check_connection(self) -> bool:
        """Healthcheck for Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the underlying client connection (pgvector parity)."""
        self.client.close()
