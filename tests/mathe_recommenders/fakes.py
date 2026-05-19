from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


def fake_mathe_client():
    return MatheMirrorClient.__new__(MatheMirrorClient)


class FakeRecommendationClient:
    def __init__(self, recommendations):
        self.recommendations = recommendations

    def get_recommendations(self, application, entity_id, limit=None):
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
