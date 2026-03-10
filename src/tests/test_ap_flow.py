import json
import pytest
import httpx
from typing import Dict

# Configuration - update these to match your local dev server
BASE_URL = "http://127.0.0.1:8000"
AP_REQUEST_FILE = "ap_search_request.json"

@pytest.fixture
def sample_ap_request() -> Dict:
    """Loads the Analytical Pattern request from the local JSON file."""
    with open(AP_REQUEST_FILE, "r") as f:
        return json.load(f)

@pytest.mark.asyncio
async def test_recommendation_ap_flow(sample_ap_request):
    """
    Test the full AP flow: 
    1. Send AP JSON to /test/recommend/ap
    2. Verify the Operator added cr:FileObject nodes
    3. Verify the distribution edges link to the correct dataset_id
    """
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 1. Call the endpoint
        response = await client.post("/test/recommend/ap", json=sample_ap_request)
        
        # Check HTTP status
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        updated_ap = response.json()
        nodes = updated_ap["ap"]["nodes"]
        edges = updated_ap["ap"]["edges"]

        # 2. Verify Output Nodes (Items)
        # We expect cr:FileObject nodes to exist now
        item_nodes = [n for n in nodes if "cr:FileObject" in n["labels"]]
        assert len(item_nodes) > 0, "No recommendation items (cr:FileObject) were added to the AP."
        
        # Check that the first item has the required properties
        first_item = item_nodes[0]
        assert "item_id" in first_item["properties"]
        assert "dataset_id" in first_item["properties"]

        # 3. Verify Graph Topology (Edges)
        # Look for the 'distribution' edges we talked about
        dist_edges = [e for e in edges if "distribution" in e["labels"]]
        assert len(dist_edges) > 0, "The 'distribution' edges linking items to datasets are missing."
        
        # Verify the Operator has 'output' edges
        output_edges = [e for e in edges if "output" in e["labels"]]
        assert len(output_edges) > 0, "The operator is not linked to the results via 'output' edges."

        # 4. Verify Metadata
        assert "metadata" in updated_ap
        assert "recommendations" in updated_ap["metadata"]
        print(f"\n✅ Success: Found {len(item_nodes)} items in the updated Analytical Pattern.")

if __name__ == "__main__":
    # If running directly without pytest
    import asyncio
    asyncio.run(test_recommendation_ap_flow(sample_ap_request()))