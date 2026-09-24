from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.qdrant_client import QdrantStorageClient


def fake_mathe_client():
    return MatheMirrorClient.__new__(MatheMirrorClient)


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


def test_normalize_recommendations_accepts_scored_pairs():
    client = QdrantStorageClient.__new__(QdrantStorageClient)

    normalized = client._normalize_recommendations(
        [("8.pdf", 0.8), ("7.pdf", 0.0)]
    )

    assert normalized == {"8.pdf": 0.8, "7.pdf": 0.0}


# Backward compatibility: if only ordered IDs are provided,
# assign descending rank-derived scores so Qdrant preserves the order.
def test_normalize_recommendations_still_accepts_ranked_id_lists():
    client = QdrantStorageClient.__new__(QdrantStorageClient)

    normalized = client._normalize_recommendations(["8.pdf", "7.pdf"])

    assert normalized == {"8.pdf": 2.0, "7.pdf": 1.0}


def test_normalize_recommendations_still_accepts_score_maps():
    client = QdrantStorageClient.__new__(QdrantStorageClient)

    normalized = client._normalize_recommendations({"8.pdf": 0.8, "7.pdf": 0.0})

    assert normalized == {"8.pdf": 0.8, "7.pdf": 0.0}


def test_normalize_recommendations_accepts_empty_inputs():
    client = QdrantStorageClient.__new__(QdrantStorageClient)

    assert client._normalize_recommendations([]) == {}
    assert client._normalize_recommendations({}) == {}


# This test ensures that when the limit is None, the method does not attempt to trim recommendations.
def test_update_neighbor_recs_does_not_trim_when_limit_is_none():
    client = QdrantStorageClient.__new__(QdrantStorageClient)
    client.client = FakeQdrantClient()

    updated_payloads = []
    client.set_recommendations_payload = lambda app, neighbor_id, recs: updated_payloads.append((neighbor_id, recs))
    client.get_recommendations_with_scores = lambda app, neighbor_id: [("7.pdf", 0.7)]

    client.update_neighbor_recs(
        application="mathe",
        neighbor_id="6.pdf",
        new_entity_id="8.pdf",
        score=0.8,
        limit=None,
    )

    assert len(updated_payloads) == 1
    neighbor_id, recs = updated_payloads[0]
    assert neighbor_id == "6.pdf"
    assert recs == {"8.pdf": 0.8, "7.pdf": 0.7}


# This test ensures that when a limit is provided, get_recommendations restricts the results returned.
def test_get_recommendations_can_limit_range():
    client = QdrantStorageClient.__new__(QdrantStorageClient)
    client.get_recommendations_with_scores = lambda application, entity_id, limit=None: [
        ("8.pdf", 0.8),
        ("7.pdf", 0.7),
        ("5.pdf", 0.6),
    ][:limit]

    recommendations = client.get_recommendations(
        application="mathe",
        entity_id="6.pdf",
        limit=2,
    )

    assert recommendations == ["8.pdf", "7.pdf"]


# This test ensures recommendations with scores are correctly retrieved from Qdrant payloads.
def test_get_recommendations_with_scores_returns_qdrant_scores():
    client = QdrantStorageClient.__new__(QdrantStorageClient)
    client.client = FakeQdrantClient()

    recommendations = client.get_recommendations_with_scores(
        application="mathe",
        entity_id="6.pdf",
        limit=2,
    )

    assert recommendations == [("8.pdf", 0.8), ("7.pdf", 0.7)]