import os
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
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

        self.client = QdrantClient(
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
    # 1. EMBEDDING STORAGE (Replaces pgvector)
    # -------------------------------------------------------------------------

    def store_embeddings(
        self,
        application: str,
        dataset_ids: List[str],
        embeddings: Any,
        embedding_inputs: List[str],
        embedding_model: str,
        collection_name: str = COLLECTION_DATASET,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> int:
        """Upsert vector embeddings with metadata into Qdrant."""
        if len(embeddings) == 0:
            return 0

        vector_size = len(embeddings[0])
        self.ensure_collection(collection_name, vector_size=vector_size)

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

        # Batch upsert points
        self.client.upsert(collection_name=collection_name, points=points)
        return len(points)

    # -------------------------------------------------------------------------
    # 2. RECOMMENDATION STORAGE (Replaces Redis ZSETs)
    # -------------------------------------------------------------------------

    def store_recommendations(
        self,
        application: str,
        recommendations: Dict[str, List[Any]],
        collection_name: str = COLLECTION_DATASET,
    ):
        """
        Store top-N precomputed recommendations directly inside entity payload points per application.
        Example payload created: {"recommendations": {"ds2ds": [{"id": "rec_id", "score": 0.95}, ...]}}
        """
        for entity_id, recs in recommendations.items():
            point_id = self._generate_point_id(entity_id)

            formatted_recs = []
            for r in recs:
                if isinstance(r, dict):
                    rec_id = r.get("id")
                    rec_score = r.get("score", 1.0)
                elif isinstance(r, (tuple, list)):
                    rec_id = r[0]
                    rec_score = r[1] if len(r) > 1 else 1.0
                else:
                    rec_id = r
                    rec_score = 1.0

                formatted_recs.append({
                    "id": str(rec_id),
                    "score": float(rec_score),
                })

            # Set payload nested under application name (e.g., payload["recommendations"]["ds2ds"])
            self.client.set_payload(
                collection_name=collection_name,
                payload={
                    "recommendations": {
                        application: formatted_recs
                    }
                },
                points=[point_id],
            )

    def get_recommendations(
        self,
        entity_id: str,
        application: str,
        collection_name: str = COLLECTION_DATASET,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-N precomputed recommendations for a given entity and application.
        """
        point_id = self._generate_point_id(entity_id)

        points = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
        )

        if not points or not points[0].payload:
            return []

        # Safely retrieve from payload.recommendations.<application>
        recs_by_app = points[0].payload.get("recommendations", {})
        return recs_by_app.get(application, [])

    # -------------------------------------------------------------------------
    # 3. VECTOR RETRIEVAL / QUERYING
    # -------------------------------------------------------------------------

    def find_similar(
        self,
        application: str,
        query_vector: List[float],
        top_k: int = 10,
        collection_name: str = COLLECTION_DATASET,
    ) -> List[Dict[str, Any]]:
        """Perform ANN search filtered by application namespace."""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
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

    def check_connection(self) -> bool:
        """Healthcheck for Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False