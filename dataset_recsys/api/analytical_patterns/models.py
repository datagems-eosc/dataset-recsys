from typing import List
from enum import Enum
from fastapi import Query
from pydantic import BaseModel, Field, field_validator
SUPPORTED_APPLICATIONS = ["portal", "mathe"]

class RecsRequest(BaseModel):
    application: str = Query(..., description="The scope within which we are searching for recommendations (e.g., mathe or portal)", enum=SUPPORTED_APPLICATIONS, required=True)
    entity_id: str = Query(..., description="The ID of the entity for which we want recommendations (e.g., a specific dataset or a specific material)", required=True)
    n: int = Query(10, gt=0, le=20, description="Number of similar items to return")

class Recommendation(BaseModel):
    id: str = Field(..., description="The recommended entity ID")

class RecsResponse(BaseModel):
    query_time: float = Field(..., description="Time taken to process the recommendation request in seconds")
    application: str = Field(..., description="The scope within which we are searching for recommendations (e.g., mathe or portal)")
    entity_id: str = Field(..., description="The ID of the entity for which we want recommendations (e.g., a specific dataset or a specific material)")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")