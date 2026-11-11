from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


def fake_mathe_client():
    return MatheMirrorClient.__new__(MatheMirrorClient)


class FakeQdrantStorageClient:
    """Stand-in for QdrantStorageClient that mirrors its return shapes.

    ``recommendations`` maps an entity ID to its stored neighbours, either as
    ``(id, score)`` pairs or as a ranked list of IDs. ``similar_results`` is the
    ranked ``(id, score)`` list a vector search would return.

    Like the real client, ``find_similar`` returns dicts and
    ``find_similar_by_ids`` returns ``(id, score)`` tuples. For an entity with
    no stored point, ``get_recommendations_with_scores`` returns an empty list
    and ``get_recommendations`` raises ``KeyError``.
    """

    COLLECTION_DATASET = "dataset_embeddings"
    COLLECTION_MATHE = "mathe_embeddings"

    def __init__(self, recommendations=None, similar_results=None):
        self.recommendations = recommendations or {}
        self.similar_results = similar_results or []
        self.calls = []
        self.recommendation_calls = []

    def find_similar(
        self,
        application,
        query_embedding=None,
        query_vector=None,
        top_k=10,
        table=None,
        collection_name=None,
    ):
        vector = query_embedding if query_embedding is not None else query_vector
        self.calls.append(
            {
                "application": application,
                "query_embedding": vector,
                "top_k": top_k,
                "collection_name": collection_name or table,
            }
        )
        return [
            {
                "dataset_id": material_id,
                "score": similarity,
                "payload": {"dataset_id": material_id},
            }
            for material_id, similarity in self.similar_results[:top_k]
        ]

    def find_similar_by_ids(
        self,
        application,
        query_embedding,
        entity_ids,
        table=None,
        collection_name=None,
    ):
        self.calls.append(
            {
                "method": "find_similar_by_ids",
                "application": application,
                "query_embedding": query_embedding,
                "entity_ids": entity_ids,
                "collection_name": collection_name or table,
            }
        )
        scores_by_id = {
            str(material_id).strip(): similarity
            for material_id, similarity in self.similar_results
        }
        return [
            (entity_id, scores_by_id[entity_id])
            for entity_id in entity_ids
            if entity_id in scores_by_id
        ]

    def get_recommendations_with_scores(
        self,
        application,
        entity_id,
        limit=None,
        collection_name=COLLECTION_DATASET,
    ):
        self.recommendation_calls.append(
            {
                "application": application,
                "entity_id": entity_id,
                "limit": limit,
                "collection_name": collection_name,
            }
        )
        if not entity_id or (limit is not None and limit <= 0):
            return []

        stored = self.recommendations.get(entity_id, [])
        scored = [
            (str(item[0]), float(item[1]))
            if isinstance(item, tuple)
            else (str(item), float(len(stored) - rank))
            for rank, item in enumerate(stored)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit] if limit else scored

    def get_recommendations(
        self,
        application,
        entity_id,
        limit=None,
        collection_name=COLLECTION_DATASET,
    ):
        if entity_id and entity_id not in self.recommendations:
            if limit is None or limit > 0:
                raise KeyError(f"Entity '{entity_id}' does not exist in backend.")
        return [
            recommended_id
            for recommended_id, _score in self.get_recommendations_with_scores(
                application, entity_id, limit, collection_name
            )
        ]
