"""
services contains external API integrations 
"""

from .bedrock import (
    get_bedrock_client_for_model,
    test_bedrock_model_access,
    build_prompt,
    call_bedrock,
    enrich_dataset_from_json,
    batch_enrich
)

__all__ = [
    "get_bedrock_client_for_model",
    "test_bedrock_model_access",
    "build_prompt",
    "call_bedrock",
    "enrich_dataset_from_json",
    "batch_enrich"
]