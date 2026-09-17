import asyncio

from dataset_recsys.api.routes import health


def test_health_clients_are_not_created_during_module_import():
    assert health.recs_client is None
    assert health.embedding_client is None


def test_health_check_creates_clients_lazily(monkeypatch):
    class ConnectedClient:
        def check_connection(self):
            return True

    redis_client = ConnectedClient()
    embedding_client = ConnectedClient()

    monkeypatch.setattr(health, "recs_client", None)
    monkeypatch.setattr(health, "embedding_client", None)
    monkeypatch.setattr(health, "RecommendationClient", lambda: redis_client)
    monkeypatch.setattr(health, "EmbeddingClient", lambda: embedding_client)

    response = asyncio.run(health.health_check())

    assert health.recs_client is redis_client
    assert health.embedding_client is embedding_client
    assert response == {
        "status": "ok",
        "redis": "connected",
        "vector_db": "connected",
    }


def test_schema_endpoint_creates_embedding_client_lazily(monkeypatch):
    class SchemaClient:
        def get_schema_overview(self):
            return {"table": ["column"]}

    embedding_client = SchemaClient()

    monkeypatch.setattr(health, "embedding_client", None)
    monkeypatch.setattr(health, "EmbeddingClient", lambda: embedding_client)

    response = asyncio.run(health.get_schema())

    assert health.embedding_client is embedding_client
    assert response == {"status": "ok", "schema": {"table": ["column"]}}
