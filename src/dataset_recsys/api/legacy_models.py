from pydantic import BaseModel, Field
from typing import List

class ItemToItemRecsResponse(BaseModel):
    """The legacy response format exactly as it was."""
    dataset: str = Field(..., description="The dataset/application name")
    iid: str = Field(..., description="The item identifier within the selected dataset")
    recommendations: List[str] = Field(..., description="List of recommended item identifiers")