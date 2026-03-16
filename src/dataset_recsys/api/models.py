from typing import List
from enum import Enum
from fastapi import Query
from pydantic import BaseModel, Field, field_validator

class SearchRequest(BaseModel):
    dataset_id: str = Query(..., description="The dataset identifier (UUID) to search within")
    n: int = Query(10, gt=0, le=20, description="Number of similar items to return")
class API_SearchResult(BaseModel):
    item_id: str = Field(..., description="The identifier for the recommended item")

class SearchResponse(BaseModel):
    query_time: float = Field(..., description="Time taken to process the recommendation request in seconds")
    dataset_id: str = Field(..., description="The dataset identifier (UUID) that was searched")
    recommendations: List[API_SearchResult] = Field(..., description="List of recommended items with their corresponding dataset identifiers")