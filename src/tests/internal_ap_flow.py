import time
from typing import Dict
from fastapi import FastAPI, HTTPException
import time
from fastapi import FastAPI, HTTPException
from src.services.models import SearchRequest, SearchResponse, API_SearchResult
from src.ap_handling import parse_recommendation_request_ap, create_recommendation_response_ap
from src.recommendation_client import RecommendationClient

app = FastAPI()
recs_client = RecommendationClient()

# --- Core Logic Functions (Simplified for Testing) ---

async def internal_recommend_logic(request: SearchRequest):
    """
    The core recommendation engine logic without authorization checks.
    """
    start_time = time.time()
    n = request.n
    # In testing, we treat the requested dataset_id as the target directly
    target_dataset = request.dataset_id

    try:
        # Fetch from your Redis/Database client
        recs_set = recs_client.get_recommendations(dataset_id=target_dataset)
        
        if not recs_set:
            return SearchResponse(
                query_time=time.time() - start_time, 
                dataset_id=target_dataset, 
                recommendations=[]
            )

        recs_list = list(recs_set)
        query_time = time.time() - start_time
        
        # Build the response model
        final_response = SearchResponse(
            query_time=query_time,
            dataset_id=target_dataset,
            recommendations=[
                API_SearchResult(item_id=rec) 
                for rec in recs_list[:n]
            ]
        )
        return final_response

    except Exception as e:
        print(f"Error in recommendation logic: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Test Endpoints ---

@app.post("/test/recommend", response_model=SearchResponse)
async def test_get_recommendations(request: SearchRequest):
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
        # 1. Extract request params from the incoming Graph JSON
        search_request = parse_recommendation_request_ap(analytical_pattern)
        
        # 2. Get the recommendations
        search_response = await internal_recommend_logic(search_request)
        
        # 3. Inject the results back into the Graph JSON
        updated_ap = create_recommendation_response_ap(analytical_pattern, search_response)
        
        print(f"AP processing complete. Found {len(search_response.recommendations)} recommendations for dataset_id='{search_response.dataset_id}' in {search_response.query_time:.2f} seconds.")
        
        # Print pretty JSON for debugging
        import json
        print("Updated Analytical Pattern with Recommendations:")
        print(json.dumps(updated_ap, indent=2))
        
        return updated_ap
        
    except Exception as e:
        print(f"Error processing AP: {e}")
        raise HTTPException(status_code=500, detail=f"AP processing failed: {e}")