import json
from typing import Dict

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
AP_REQUEST_FILE = "ap_request.json"

@pytest.fixture
def sample_ap_request() -> Dict:
    """Loads the Analytical Pattern request from the local JSON file."""
    with open(AP_REQUEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.mark.asyncio
async def test_recommendation_ap_flow(sample_ap_request):
    """
    Test the full AP flow:
    1. Send AP JSON to /dataset-recsys/recommend/ap
    2. Verify recommendation sc:Dataset nodes were added
    3. Verify the operator is linked to the results via ranked output edges
    """
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Call the endpoint
        response = await client.post("/dataset-recsys/recommend/ap", json=sample_ap_request)
        
        # Check HTTP status
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        updated_ap = response.json()
        nodes = updated_ap["nodes"]
        edges = updated_ap["edges"]

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
    import asyncio
    asyncio.run(test_recommendation_ap_flow(sample_ap_request()))