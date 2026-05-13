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
        description="Number of similar entities to return",
    )


class Recommendation(BaseModel):
    entity_id: str = Field(..., description="The recommended entity ID")


class RecsResponse(BaseModel):
    entity_id: str = Field(
        ...,
        description="The ID of the entity for which we want recommendations",
    )
    recommendations: List[Recommendation] = Field(
        ...,
        description="List of recommendations",
    )


class MatheRecommendation(BaseModel):
    material_id: str = Field(..., description="The recommended MathE material ID")


class MatheRecsResponse(BaseModel):
    question_id: str = Field(
        ...,
        description="The MathE question ID used to generate the recommendations",
    )
    recommendations: List[MatheRecommendation] = Field(
        ...,
        description="List of recommended MathE materials",
    )
