from typing import List
from pydantic import BaseModel, Field

class RecsRequest(BaseModel):
    entity_id: str = Field(
        ...,
        description="The ID of the entity for which we want recommendations",
    )
    n: int = Field(
        10,
        gt=0,
        le=20,
        description="Number of similar items to return",
    )

class Recommendation(BaseModel):
    entity_id: str = Field(..., description="The recommended entity ID")

class RecsResponse(BaseModel):
    # query_time: float = Field(
    #     ...,
    #     description="Time taken to process the recommendation request in seconds",
    # )
    entity_id: str = Field(
        ...,
        description="The ID of the entity for which we want recommendations",
    )
    recommendations: List[Recommendation] = Field(
        ...,
        description="List of recommendations",
    )