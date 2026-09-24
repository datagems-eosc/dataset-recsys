from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


def fake_mathe_client():
    return MatheMirrorClient.__new__(MatheMirrorClient)


class FakeRecommendationClient:
    def __init__(self, recommendations):
        self.recommendations = recommendations
        self.calls = []

    def get_recommendations(self, application, entity_id, limit=None):
        self.calls.append(
            {
                "application": application,
                "entity_id": entity_id,
                "limit": limit,
            }
        )
        recommendations = self.recommendations.get(entity_id, [])
        return recommendations if limit is None else recommendations[:limit]

    def get_recommendations_with_scores(self, application, entity_id, limit=None):
        recommendations = self.get_recommendations(application, entity_id, limit)
        return [
            item if isinstance(item, tuple) else (item, float(len(recommendations) - rank))
            for rank, item in enumerate(recommendations)
        ]


class FakeEmbeddingClient:
    TABLE_MATHE = "mathe_embeddings"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def find_similar(self, application, query_embedding, top_k, table):
        self.calls.append(
            {
                "application": application,
                "query_embedding": query_embedding,
                "top_k": top_k,
                "table": table,
            }
        )
        return self.results[:top_k]

    def find_similar_by_ids(self, application, query_embedding, entity_ids, table):
        self.calls.append(
            {
                "method": "find_similar_by_ids",
                "application": application,
                "query_embedding": query_embedding,
                "entity_ids": entity_ids,
                "table": table,
            }
        )
        scores_by_id = {
            str(material_id).strip(): similarity
            for material_id, similarity in self.results
        }
        return [
            (entity_id, scores_by_id[entity_id])
            for entity_id in entity_ids
            if entity_id in scores_by_id
        ]

class FakeQdrantClient:
    def __init__(self):
        self.calls = []
        self.payloads = {
            "6.pdf": {
                "recommendations": {"8.pdf": 0.8, "7.pdf": 0.7, "5.pdf": 0.6}
            }
        }

    def set_payload(self, *args, **kwargs):
        self.calls.append(("set_payload", args, kwargs))

    def retrieve(self, *args, **kwargs):
        self.calls.append(("retrieve", args, kwargs))
        ids = kwargs.get("ids", [])
        results = []
        for point_id in ids:
            if point_id in self.payloads:
                class PointRecord:
                    payload = self.payloads[point_id]
                results.append(PointRecord())
        return results
