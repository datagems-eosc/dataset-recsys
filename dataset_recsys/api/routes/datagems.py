

import json
import time
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from dataset_recsys.api.analytical_patterns.ap_handling import (
    create_recommendation_response_ap,
    parse_recommendation_request_ap,
)
from dataset_recsys.api.analytical_patterns.models import (
    Recommendation,
    RecsRequest,
    RecsResponse,
)
from dataset_recsys.api.security import security
from dataset_recsys.ingestion.moma_dataset import MomaDataset
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.workflows.incremental_update import process_incremental_update

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")
router = APIRouter(prefix="/dataset-recsys", tags=["DataGEMS Recommendation Service"])
recs_client = RecommendationClient()

DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/valid_examples.json")
DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/error_examples.json")
AP_DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_valid_examples.json")
AP_DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_error_examples.json")
AP_REQUEST_EXAMPLE_PATH = Path("dataset_recsys/api/api_docs/ap_request_example.json")


def load_json_file(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"File '{path}' does not exist.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load file from {path}: {e}")
        return {}

examples_data, errors_data = (load_json_file(DOCS_VALID_EXAMPLES_PATH), load_json_file(DOCS_ERROR_EXAMPLES_PATH))
ap_examples_data, ap_errors_data = (load_json_file(AP_DOCS_VALID_EXAMPLES_PATH), load_json_file(AP_DOCS_ERROR_EXAMPLES_PATH))
ap_request_example = load_json_file(AP_REQUEST_EXAMPLE_PATH)

@router.post(
    "/recommend",
    response_model=RecsResponse,
    summary="Get recommendations",
    description="""
Retrieve the top-N recommendations for a given dataset.
    """,
    responses={
        200: {
            "description": "Successful retrieval of related datasets",
            "content": {"application/json": {"examples": examples_data}},
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": {
                        "Missing Entity ID": errors_data.get("422_missing_entity_id")
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing token",
            "content": {
                "application/json": {
                    "examples": {"Invalid Token": errors_data.get("401")}
                }
            },
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "examples": {"Access Denied": errors_data.get("403")}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {"System Failure": errors_data.get("500")}
                }
            },
        },
    },
)
async def get_recommendations(
    entity_id: str = Query(..., description="The dataset identifier.", required=True),
    n: int = Query(10, gt=0, le=20, description="Number of similar items to return"),
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme),
):
    request = RecsRequest(entity_id=entity_id, n=n)
    start_time = time.time()
    user_subject = claims.get("sub")
    log = logger.bind(item_id=request.entity_id, UserId=user_subject)

    accounting_logger.info(
        "Recommendation Request Received",
        UserId=user_subject,
        Action="get_recommendations",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    request.entity_id = request.entity_id.strip()
    if not request.entity_id:
        log.warning("Missing entity_id.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Entity ID is required.",
        )

    lookup_id = request.entity_id
    try:
        authorized_list = await security.get_authorized_entity_ids(token)
        authorized_set = set(authorized_list)

        if lookup_id not in authorized_set:
            log.warning(f"User {user_subject} not authorized for source entity {lookup_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for the requested entity_id.",
            )

        raw_recs = recs_client.get_recommendations(
            application="portal", entity_id=request.entity_id
        )

        filtered_recs = [Recommendation(entity_id=item) for item in raw_recs if item in authorized_set]

        query_time = time.time() - start_time
        log.info(
            f"Returning {len(filtered_recs[:request.n])} recs for {lookup_id} in {query_time:.3f}s"
        )

        return RecsResponse(
            entity_id=request.entity_id,
            recommendations=filtered_recs[:request.n],
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/recommend/ap",
    summary="Get recommendations via Analytical Pattern",
    description="""
Processes an Analytical Pattern (AP) request by extracting the seed dataset and returning its top-N recommendations as part of an enriched graph.
    """
# 1. **Parameter Extraction**: Retrieves the requested number of recommendations from the operator properties.
# 2. **Seed Identification**: Identifies the seed dataset by tracing the incoming input edge to the operator.
# 3. **Recommendation Request**: Queries the recommendation engine to obtain the most relevant datasets.
# 4. **Graph Injection**: Adds the recommended datasets as new sc:Dataset nodes.
# 5. **Output Linking**: Connects the operator to each recommended dataset via output edges, assigning a rank property to preserve the order of relevance.
    ,
    responses={
        200: {
            "description": "Successful graph transformation",
            "content": {"application/json": {"examples": ap_examples_data}},
        },
        403: {
            "description": "Authorization Failure",
            "content": {
                "application/json": {
                    "examples": {
                        "Insufficient Permissions": errors_data.get("403")
                    }
                }
            },
        },
        422: {
            "description": "Malformed AP Graph",
            "content": {
                "application/json": {
                    "examples": {
                        "Portal Missing Edge": ap_errors_data.get("422_ap_missing_input_edge"),
                        "Operator Missing": ap_errors_data.get("422_ap_missing_operator"),
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {"Server Error": errors_data.get("500")}
                }
            },
        },
    },
)
async def get_recommendations_ap(
    analytical_pattern: dict = Body(
        ...,
        description="The Analytical Pattern graph in JSON format",
        openapi_examples={
            "Portal Request": {
                "summary": "Standard Portal Request",
                "description": "Infers entity_id from the incoming 'input' edge.",
                "value": ap_request_example,
            }
        },
    ),
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme),
):
    try:
        search_request = parse_recommendation_request_ap(analytical_pattern)
        search_response = await get_recommendations(
            search_request.entity_id,
            search_request.n,
            claims,
            token,
        )
        updated_ap = create_recommendation_response_ap(analytical_pattern, search_response)
        return updated_ap
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing recommendation AP request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post(
    "/dataset/add",
    summary="Add a new dataset to the system",
    description="""
Fetch a dataset from external sources, generate its embeddings, and integrate it into the recommendation engine.
If the dataset already exists, the system will skip the addition to avoid duplicates.
    """,
    responses={
        200: {
            "description": "Dataset processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Dataset ds_123 successfully added and recommendations updated."
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing token",
            "content": {
                "application/json": {
                    "examples": {"Invalid Token": errors_data.get("401")}
                }
            },
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "examples": {"Access Denied": errors_data.get("403")}
                }
            },
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": {
                        "Missing Entity ID": errors_data.get("422_missing_entity_id")
                    }
                }
            },
        },
        404: {
            "description": "Not Found - Dataset not found in external source",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found in external repository."}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {"System Failure": errors_data.get("500")}
                }
            },
        },
    },
)
async def add_dataset(
    entity_id: str = Query(..., description="The dataset identifier to add."),
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme),
):
    user_subject = claims.get("sub")
    log = logger.bind(item_id=entity_id, UserId=user_subject)

    accounting_logger.info(
        "Dataset Addition Request Received",
        UserId=user_subject,
        Action="add_dataset",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        ItemId=entity_id,
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    entity_id = entity_id.strip()
    if not entity_id:
        log.warning("Missing entity_id.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dataset ID is required.",
        )
    
    # It is more efficient to reuse the global recs_client or define these outside 
    # the request scope, but keeping local as per your original snippet logic.
    embedding_client = EmbeddingClient()
    
    try:
        moma = MomaDataset(user_token=token)
        # Assuming get_from_external might raise an error if not found
        moma.get_from_external(entity_id)
        profile = moma.to_dataset_profile()

        was_added = await process_incremental_update(
            profile, 
            "portal", 
            recs_client=recs_client, 
            emb_client=embedding_client
        )

        if not was_added:
            return {
                "status": "ignored",
                "message": f"Dataset {entity_id} is already in the system.",
            }

        return {
            "status": "success",
            "message": f"Dataset {entity_id} successfully added and recommendations updated.",
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error adding dataset {entity_id}: {e}", exc_info=True)
        # Providing a cleaner error message for the user while logging the full exception
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while adding the dataset."
        )