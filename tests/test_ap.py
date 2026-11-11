"""Integration test of the Analytical Pattern flow against a running test app.

It needs the local test app from tests/internal_ap_flow.py (and the Qdrant it
reads from) to be running:

    uvicorn internal_ap_flow:app --app-dir tests

When that app is not reachable the test is skipped, so the unit test run does
not depend on it.
"""
import json
from pathlib import Path
from typing import Dict

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
# Resolved next to this file, so the test can be run from any folder.
AP_REQUEST_FILE = Path(__file__).resolve().parent / "ap_request.json"


def load_sample_ap_request() -> Dict:
    """Loads the Analytical Pattern request from the local JSON file."""
    with open(AP_REQUEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_ap_request() -> Dict:
    return load_sample_ap_request()


def test_recommendation_ap_flow(sample_ap_request):
    """
    Test the full AP flow:
    1. Send AP JSON to /dataset-recsys/recommend/ap
    2. Verify recommendation sc:Dataset nodes were added
    3. Verify the operator is linked to the results via ranked output edges
    """
    with httpx.Client(base_url=BASE_URL) as client:
        # Call the endpoint
        try:
            response = client.post("/test/recommend/ap", json=sample_ap_request)
        except httpx.TransportError as error:
            pytest.skip(
                f"The AP test app is not reachable at {BASE_URL} ({error}); "
                "start it with: uvicorn internal_ap_flow:app --app-dir tests"
            )

        # Check HTTP status
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        updated_ap = response.json()
        print("\nUpdated Analytical Pattern Response:")
        print(json.dumps(updated_ap, indent=2))
        is_wrapped = "ap" in updated_ap
        nodes = updated_ap["nodes"] if not is_wrapped else updated_ap["ap"]["nodes"]
        edges = updated_ap["edges"] if not is_wrapped else updated_ap["ap"]["edges"]

        # Verify recommended dataset nodes
        # We expect recommendation sc:Dataset nodes
        item_nodes = [n for n in nodes if "sc:Dataset" in n["labels"]]
        assert len(item_nodes) > 0, "No recommendation items (sc:Dataset) were added to the AP."
        
        # Verify the Operator has 'output' edges
        output_edges = [e for e in edges if "output" in e["labels"]]
        assert len(output_edges) > 0, "The operator is not linked to the results via 'output' edges."
        assert all("rank" in e.get("properties", {}) for e in output_edges), (
            "All output edges should include a rank property."
        )
        print(
            f"\nFound {len(item_nodes)} recommended datasets in the updated Analytical Pattern."
        )

if __name__ == "__main__":
    # If running directly without pytest
    test_recommendation_ap_flow(load_sample_ap_request())
