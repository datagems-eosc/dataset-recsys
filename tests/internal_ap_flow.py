"""
DEV-ONLY TEST FILE: It bypasses authentication and authorization.

Used for:
- testing recommendation logic without auth
- testing Analytical Pattern transformations locally

How to run locally:
1. Ensure Redis is reachable locally, e.g. through port-forward:
   kubectl port-forward svc/dataset-recsys-redis -n athenarc 6380:6379

2. Start this test app (not the main API):
   REDIS_HOST=localhost REDIS_PORT=6380 uvicorn internal_ap_flow:app --reload --app-dir tests

3. Test the direct recommendation flow:
   curl -X POST "http://127.0.0.1:8000/test/recommend" \
     -H "Content-Type: application/json" \
     -d '{"entity_id": "9b25bc46-8bd3-4f7f-94b4-52dbc38c130f", "n": 5}'

4. Test the full AP flow:
   curl -X POST "http://127.0.0.1:8000/test/recommend/ap" \
     -H "Content-Type: application/json" \
     -d @tests/ap_request.json
"""

from typing import Dict

from fastapi import FastAPI, HTTPException

from dataset_recsys.api.analytical_patterns.models import Recommendation, RecsRequest, RecsResponse
from dataset_recsys.api.analytical_patterns.ap_handling import (
    parse_recommendation_request_ap,
    create_recommendation_response_ap,
)
from dataset_recsys.storage.recommendation_client import RecommendationClient

app = FastAPI()
recs_client = RecommendationClient()

async def internal_recommend_logic(request: RecsRequest):
    """
    The core recommendation engine logic without authorization checks.
    """
    entity_id = request.entity_id
    n = request.n

    try:
        recs_set = recs_client.get_recommendations(application="ds2ds", entity_id=entity_id)

        if not recs_set:
            return RecsResponse(
                entity_id=entity_id,
                recommendations=[]
            )

        recs_list = list(recs_set)

        print(f"Generated {len(recs_list)} recommendations for {entity_id}.")
        print(f"Recommendations: {recs_list[:n]}")

        recs = [Recommendation(entity_id=rec_id) for rec_id in recs_list[:n]]

        return RecsResponse(
            entity_id=entity_id,
            recommendations=recs
        )

    except Exception as e:
        print(f"Error in recommendation logic: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Test Endpoints ---

@app.post("/test/recommend", response_model=RecsResponse)
async def test_get_recommendations(request: RecsRequest):
    """
    Direct endpoint to test the recommendation engine.
    """
    return await internal_recommend_logic(request)

@app.post("/test/recommend/ap", response_model=Dict)
async def test_get_recommendations_ap(analytical_pattern: Dict):
    """
    Endpoint to test the full Analytical Pattern (AP) flow: 
    Parse AP -> Get Recs -> Update AP.
    """
    try:
        print("Received AP request for recommendations.")
        # 1. Extract request params from the incoming Graph JSON
        search_request = parse_recommendation_request_ap(analytical_pattern)
        
        # 2. Get the recommendations
        search_response = await internal_recommend_logic(search_request)
        
        # 3. Inject the results back into the Graph JSON
        updated_ap = create_recommendation_response_ap(analytical_pattern, search_response)

        # Write the updated ap to a file for inspection
        # with open("updated_ap.json", "w") as f:
        #     import json
        #     json.dump(updated_ap, f, indent=2)
        
        return updated_ap
        
    except Exception as e:
        print(f"Error processing AP: {e}")
        raise HTTPException(status_code=500, detail=f"AP processing failed: {e}")