import json
from pathlib import Path
from fastapi.testclient import TestClient
from src.services.dataset_recs_api import app
import pytest

client = TestClient(app)

def get_available_datasets():
    """Return a list of available datasets for testing."""
    # TODO: Extend this list as more datasets are added.
    return ["mathe"]

def load_recos(dataset="mathe"):
    """Load recommendations for the given dataset."""
    path = Path(f"data/{dataset}/{dataset}_top20_recommendations.json")
    # TODO: Replace this with database retrieval logic as needed.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.mark.parametrize("dataset", get_available_datasets())
def test_recos_structure(dataset):
    """Verify the structure of recommendations is a dict with string keys and list of strings as values."""
    recos = load_recos(dataset)
    assert isinstance(recos, dict)
    for k, v in recos.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert all(isinstance(x, str) for x in v)

@pytest.mark.parametrize("dataset", get_available_datasets())
def test_recos_reference_consistency(dataset):
    """Ensure all recommended items exist as keys, maintaining internal consistency."""
    recos = load_recos(dataset)
    for _, v in recos.items():
        for rec in v:
            assert rec in recos, f"{rec} not found as key in recommendations."

@pytest.mark.parametrize("dataset", get_available_datasets())
def test_recos_no_self_recommendations(dataset):
    """Check that no item recommends itself."""
    recos = load_recos(dataset)
    for k, v in recos.items():
        assert k not in v, f"{k} recommends itself."

# @pytest.mark.parametrize("dataset", get_available_datasets())
# def test_get_valid_recommendations(dataset):
#     """Test API returns valid recommendations for a known item id."""
#     recos = load_recos(dataset)
#     valid_id = next(iter(recos.keys()))
#     response = client.get(f"/recommendations?dataset={dataset}&iid={valid_id}&n=5")
#     assert response.status_code == 200
#     data = response.json()
#     assert "recommendations" in data
#     assert len(data["recommendations"]) <= 5

@pytest.mark.parametrize("dataset", get_available_datasets())
def test_get_invalid_id(dataset):
    """Verify API returns 404 for a non-existent item id."""
    response = client.get(f"/recommendations?dataset={dataset}&iid=nonexistent.pdf")
    assert response.status_code == 404

def test_get_invalid_dataset():
    """Verify API returns 404 for an unknown dataset."""
    response = client.get("/recommendations?dataset=unknown&iid=6.pdf")
    assert response.status_code == 404

# python -m pytest -v