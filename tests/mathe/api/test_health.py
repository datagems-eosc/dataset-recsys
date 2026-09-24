import asyncio

from dataset_recsys.api.routes import health


def test_health_clients_are_not_created_during_module_import():
    assert health.qdrant_client is None


def test_health_check_creates_clients_lazily(monkeypatch):
    class ConnectedClient:
        def check_connection(self):
            return True

    qdrant_client = ConnectedClient()

    monkeypatch.setattr(health, "qdrant_client", None)
    monkeypatch.setattr(health, "QdrantStorageClient", lambda: qdrant_client)

    response = asyncio.run(health.health_check())

    assert health.qdrant_client is qdrant_client
    assert response == {
        "status": "ok",
        "qdrant": "connected",
    }


def test_schema_endpoint_creates_qdrant_client_lazily(monkeypatch):
    class SchemaClient:
        def get_schema_overview(self):
            return {"collections": ["ds2ds"]}

    qdrant_client = SchemaClient()

    monkeypatch.setattr(health, "qdrant_client", None)
    monkeypatch.setattr(health, "QdrantStorageClient", lambda: qdrant_client)

    response = asyncio.run(health.get_schema())

    assert health.qdrant_client is qdrant_client
    assert response == {"status": "ok", "schema": {"collections": ["ds2ds"]}}