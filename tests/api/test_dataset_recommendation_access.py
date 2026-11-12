"""Dataset authorization at both recommendation API entry points."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dataset_recsys.api.routes import datagems
from dataset_recsys.api.security import security


@pytest.fixture
def api(monkeypatch):
    app = FastAPI()
    app.include_router(datagems.router)
    state = {"claims": {"sub": "test-user", "realm_access": {"roles": ["dg_user"]}}, "grants": {}}
    app.dependency_overrides[security.get_current_user_claims] = lambda: state["claims"]

    class Storage:
        def get_entity_status(self, application, entity_id):
            return "AVAILABLE"

        def get_recommendations(self, application, entity_id, limit):
            return ["restricted", "allowed", "wrong-role", "also-allowed"]

    async def grants(claims, token):
        return None if security.has_privileged_dataset_role(claims) else state["grants"]

    monkeypatch.setattr(datagems, "qdrant_client", Storage())
    monkeypatch.setattr(security, "get_authorized_entity_ids", grants)
    monkeypatch.setattr(datagems, "write_request_log_to_sqlite", lambda **kwargs: None)
    return TestClient(app), state


def request(client, endpoint, authenticated=True):
    headers = {"Authorization": "Bearer test-token"} if authenticated else {}
    if endpoint == "direct":
        return client.post("/dataset-recsys/recommend", params={"entity_id": "source", "n": 2}, headers=headers)
    return client.post(
        "/dataset-recsys/recommend/ap",
        headers=headers,
        json={
            "nodes": [
                {"id": "op", "labels": ["DatasetRecommender_Operator"], "properties": {"n": 2}},
                {"id": "source", "labels": ["sc:Dataset"], "properties": {}},
            ],
            "edges": [{"from": "source", "to": "op", "labels": ["input"]}],
        },
    )


def recommendation_ids(response, endpoint):
    body = response.json()
    if endpoint == "direct":
        return [item["entity_id"] for item in body["recommendations"]]
    return [edge["to"] for edge in body["edges"] if "output" in edge["labels"]]


@pytest.mark.parametrize("endpoint", ["direct", "ap"])
@pytest.mark.parametrize("role", ["dg_admin", "dg_dataset-curator", "dg_system"])
def test_privileged_roles_keep_global_access(api, endpoint, role):
    client, state = api
    state["claims"]["realm_access"]["roles"] = [role]
    response = request(client, endpoint)
    assert response.status_code == 200
    assert recommendation_ids(response, endpoint) == ["restricted", "allowed"]


@pytest.mark.parametrize("endpoint", ["direct", "ap"])
def test_browse_grants_allow_source_and_filter_results_before_limit(api, endpoint):
    client, state = api
    state["grants"] = {
        "source": "dg_ds-browse", "allowed": "dg_ds-browse",
        "wrong-role": "other-role", "also-allowed": "dg_ds-browse",
    }
    response = request(client, endpoint)
    assert response.status_code == 200
    assert recommendation_ids(response, endpoint) == ["allowed", "also-allowed"]


@pytest.mark.parametrize("endpoint", ["direct", "ap"])
def test_missing_source_grant_is_forbidden(api, endpoint):
    client, state = api
    state["grants"] = {"allowed": "dg_ds-browse"}
    assert request(client, endpoint).status_code == 403


@pytest.mark.parametrize("endpoint", ["direct", "ap"])
def test_user_without_platform_role_is_forbidden_even_with_browse_grant(api, endpoint):
    client, state = api
    state["claims"]["realm_access"]["roles"] = []
    state["grants"] = {"source": "dg_ds-browse", "allowed": "dg_ds-browse"}
    assert request(client, endpoint).status_code == 403


@pytest.mark.parametrize("endpoint", ["direct", "ap"])
def test_authentication_is_required(api, endpoint):
    client, _ = api
    assert request(client, endpoint, authenticated=False).status_code == 401
