import json
import os
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantStorageClient
from qdrant_client.http import models


class QdrantStorageClient:
    """
    Qdrant client for storing vector embeddings and metadata, 
    serving vector similarity searches, and storing precomputed recommendations.
    """

    COLLECTION_DATASET = "dataset_embeddings"
    COLLECTION_MATHE = "mathe_embeddings"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", None)

        self.client = QdrantStorageClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
        )

    def _generate_point_id(self, entity_id: str) -> str:
        """Generate a deterministic UUID v5 from an arbitrary entity_id string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, entity_id))

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: models.Distance = models.Distance.COSINE,
    ):
        """Create collection if it does not already exist."""
        collections = [col.name for col in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
            # Create payload index on application for fast filtered lookups
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
        **kwargs,
    ) -> int:
        """Upsert vector embeddings with metadata into Qdrant."""
        if len(embeddings) == 0:
            return 0

        target_collection = collection_name or table or self.COLLECTION_DATASET
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
        return len(points)

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
        point_id = self._generate_point_id(dataset_id)
        points = self.client.retrieve(collection_name=collection_name, ids=[point_id])
        return len(points) > 0

    def find_similar(
        self,
        application: str,
        query_embedding: Optional[List[float]] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        table: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform ANN search filtered by application namespace."""
        target_collection = collection_name or table or self.COLLECTION_DATASET
        vector = query_embedding if query_embedding is not None else query_vector

        results = self.client.search(
            collection_name=target_collection,
            query_vector=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="application",
                        match=models.MatchValue(value=application),
                    )
                ]
            ),
            limit=top_k,
        )

        return [
            {
                "dataset_id": res.payload.get("dataset_id"),
                "score": res.score,
                "payload": res.payload,
            }
            for res in results
        ]

    def find_similar_by_ids(
        self,
        application: str,
        query_embedding: List[float],
        entity_ids: List[str],
        table: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Return query-vector similarities for the requested IDs."""
        if not entity_ids:
            return []

        target_collection = collection_name or table or self.COLLECTION_DATASET
        point_ids = [self._generate_point_id(eid) for eid in entity_ids]

        results = self.client.search(
            collection_name=target_collection,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="application", match=models.MatchValue(value=application)),
                    models.HasIdCondition(has_id=point_ids)
                ]
            ),
            limit=len(entity_ids),
        )
        return [(res.payload.get("dataset_id"), res.score) for res in results]

    def delete_single_embedding(self, dataset_id: str, collection_name: str = COLLECTION_DATASET) -> int:
        """Delete a single embedding from the database."""
        point_id = self._generate_point_id(dataset_id)
        res = self.client.delete(collection_name=collection_name, points_selector=[point_id])
        return 1 if res.status == models.UpdateStatus.COMPLETED else 0

    def delete_application(self, application: str, table: Optional[str] = None, collection_name: Optional[str] = None) -> int:
        """Delete all points and recommendations belonging to an application."""
        target_collection = collection_name or table or self.COLLECTION_DATASET
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
        points = self.client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=True)
        
        if not points:
            return

        current_recs = points[0].payload.get("recommendations", {}).get(application, [])
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
        points = self.client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=True)
        
        if not points:
            return

        current_recs = points[0].payload.get("recommendations", {}).get(application, [])
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

        point_id = self._generate_point_id(entity_id)
        points = self.client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=True)
        
        if not points:
            return "NOT_FOUND"
            
        recs = points[0].payload.get("recommendations", {}).get(application, [])
        return "AVAILABLE" if recs else "NO_RECOMMENDATIONS"

    def get_recommendations(
        self,
        application: str,
        entity_id: str,
        limit: Optional[int] = None,
        collection_name: str = COLLECTION_DATASET,
    ) -> List[str]:
        """Return recommended entity IDs for one entity (Redis parity)."""
        recs_with_scores = self.get_recommendations_with_scores(application, entity_id, limit, collection_name)
        return [rec_id for rec_id, _ in recs_with_scores]

    def get_recommendations_with_scores(
        self,
        application: str,
        entity_id: str,
        limit: Optional[int] = None,
        collection_name: str = COLLECTION_DATASET,
    ) -> List[tuple[str, float]]:
        """Return recommended entity IDs with their scores."""
        if not entity_id or (limit is not None and limit <= 0):
            return []

        point_id = self._generate_point_id(entity_id)
        points = self.client.retrieve(collection_name=collection_name, ids=[point_id], with_payload=True)

        if not points or not points[0].payload:
            raise KeyError(f"Entity '{entity_id}' does not exist in backend.")

        recs_by_app = points[0].payload.get("recommendations", {})
        app_recs = recs_by_app.get(application, [])

        sorted_recs = sorted([(str(r["id"]), float(r["score"])) for r in app_recs], key=lambda x: x[1], reverse=True)
        return sorted_recs[:limit] if limit else sorted_recs

    def list_entities(self, application: str, collection_name: str = COLLECTION_DATASET) -> List[str]:
        """List entity IDs currently stored for an application."""
        records, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="application", match=models.MatchValue(value=application))
            ]),
            limit=10000, 
            with_payload=True,
            with_vectors=False
        )
        return sorted([rec.payload.get("dataset_id") for rec in records if "dataset_id" in rec.payload])

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