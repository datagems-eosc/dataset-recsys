"""DMM_BASE_URL selects the dataset catalog; a missing value must stop the fetch."""

import pytest

from dataset_recsys.ingestion import fetch_gems_datasets, moma_dataset
from dataset_recsys.ingestion.moma_dataset import MomaDataset


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if anything is requested over HTTP."""

    def fail(*args, **kwargs):
        raise AssertionError("No HTTP request may be sent")

    monkeypatch.setattr(fetch_gems_datasets.requests, "get", fail)
    monkeypatch.setattr(fetch_gems_datasets.requests, "post", fail)
    monkeypatch.setattr(moma_dataset.requests, "get", fail)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_dmm_base_url_is_an_error(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("DMM_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("DMM_BASE_URL", value)

    with pytest.raises(RuntimeError, match="DMM_BASE_URL is not set"):
        fetch_gems_datasets.get_dmm_base_url()


def test_dmm_base_url_is_read_without_a_trailing_slash(monkeypatch):
    monkeypatch.setenv("DMM_BASE_URL", " https://dmm.example.org/dmm/ ")

    assert fetch_gems_datasets.get_dmm_base_url() == "https://dmm.example.org/dmm"
    assert moma_dataset.moma_search_url() == (
        "https://dmm.example.org/dmm/api/v1/dataset/search"
    )


# A rebuild must stop before it asks for a token or any catalog data.
def test_catalog_fetch_stops_before_any_request_when_url_is_missing(
    monkeypatch, no_network
):
    monkeypatch.delenv("DMM_BASE_URL", raising=False)

    def fail_token():
        raise AssertionError("No token may be requested")

    monkeypatch.setattr(fetch_gems_datasets, "get_access_token", fail_token)

    with pytest.raises(RuntimeError, match="DMM_BASE_URL is not set"):
        fetch_gems_datasets.fetch_catalog()
    with pytest.raises(RuntimeError, match="DMM_BASE_URL is not set"):
        fetch_gems_datasets.fetch_search("token")


def test_adding_a_dataset_stops_before_any_request_when_url_is_missing(
    monkeypatch, no_network
):
    monkeypatch.delenv("DMM_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DMM_BASE_URL is not set"):
        MomaDataset(user_token="token").get_dataset_properties("dataset-1")


def test_catalog_search_uses_the_configured_instance(monkeypatch):
    monkeypatch.setenv("DMM_BASE_URL", "https://dmm.example.org/dmm")
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return FakeResponse({"datasets": []})

    monkeypatch.setattr(fetch_gems_datasets.requests, "get", fake_get)

    fetch_gems_datasets.fetch_search("token")

    assert requested == ["https://dmm.example.org/dmm/api/v1/dataset/search"]


def test_single_dataset_lookup_uses_the_configured_instance(monkeypatch):
    monkeypatch.setenv("DMM_BASE_URL", "https://dmm.example.org/dmm")
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return FakeResponse(
            {"datasets": [{"nodes": [{"properties": {"name": "Dataset 1"}}]}]}
        )

    monkeypatch.setattr(moma_dataset.requests, "get", fake_get)

    properties = MomaDataset(user_token="token").get_dataset_properties("dataset-1")

    assert properties == {"name": "Dataset 1"}
    assert requested == ["https://dmm.example.org/dmm/api/v1/dataset/search"]
