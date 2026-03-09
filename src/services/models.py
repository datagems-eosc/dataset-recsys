from typing import List
from enum import Enum
from fastapi import Query
from pydantic import BaseModel, Field, field_validator

class SearchRequest(BaseModel):
    iid: str = Query(..., description="The item identifier within the selected dataset"),
    n: int = Query(10, gt=0, le=20, description="Number of similar items to return")    
    dataset_ids: List[str] = Field(
        default=None,
        description="A list of dataset identifiers (UUIDs) to restrict the search to.",
    )
    @field_validator("dataset_ids")
    @classmethod
    def check_dataset_ids_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("dataset_ids cannot be an empty list.")
        return v
class RecommendationResponse(BaseModel):
    query_time: float = Field(..., description="Time taken to process the recommendation request in seconds")
    dataset_ids: List[str] = Field(..., description="List of dataset identifiers that were searched to generate the recommendations")
    item_ids: List[str] = Field(..., description="List of recommended items")
    