import time
from typing import Dict
from fastapi import FastAPI, HTTPException
import time
from fastapi import FastAPI, HTTPException
from dataset_recsys.api.analytical_patterns.models import Recommendation, RecsRequest, RecsResponse
from dataset_recsys.api.analytical_patterns.ap_handling import parse_recommendation_request_ap, create_recommendation_response_ap
from dataset_recsys.storage.recommendation_client import RecommendationClient

app = FastAPI()
recs_client = RecommendationClient()

async def internal_recommend_logic(request: RecsRequest):
    """
    The core recommendation engine logic without authorization checks.
    """
    start_time = time.time()
    application = request.application
    entity_id = request.entity_id
    n = request.n

    try:
        # Fetch from your Redis/Database client
        recs_set = recs_client.get_recommendations(application=application, entity_id=entity_id)
        
        if not recs_set:
            return RecsResponse(
                query_time=time.time() - start_time, 
                application=application,
                entity_id=entity_id,
                recommendations=[]
            )

        recs_list = list(recs_set)
        query_time = time.time() - start_time
        
        print(f"Generated {len(recs_list)} recommendations for {application}:{entity_id} in {query_time:.2f} seconds.")
        print(f"Recommendations: {recs_list[:n]}")  # Print only top N for brevity
        
        recs = [Recommendation(id=rec_id) for rec_id in recs_list[:n]]
        
        # Build the response model
        final_response = RecsResponse(
            query_time=query_time,
            application=application,
            entity_id=entity_id,
            recommendations=recs  # Return only top N recommendations
        )
        return final_response

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