import time
from datetime import datetime

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
from dataset_recsys.api.api_docs_loader import (
    AP_DOCS_ERROR_EXAMPLES_PATH,
    AP_DOCS_VALID_EXAMPLES_PATH,
    AP_REQUEST_EXAMPLE_PATH,
    DOCS_ERROR_EXAMPLES_PATH,
    DOCS_VALID_EXAMPLES_PATH,
    load_json_file,
)
from dataset_recsys.api.security import security
from dataset_recsys.storage.recommendation_client import RecommendationClient

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")
router = APIRouter(prefix="/dataset-recsys", tags=["DataGEMS Recommendation Service"])
recs_client = RecommendationClient()

examples_data = load_json_file(DOCS_VALID_EXAMPLES_PATH)
errors_data = load_json_file(DOCS_ERROR_EXAMPLES_PATH)
ap_examples_data = load_json_file(AP_DOCS_VALID_EXAMPLES_PATH)
ap_errors_data = load_json_file(AP_DOCS_ERROR_EXAMPLES_PATH)
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
    n: int = Query(10, gt=0, description="Number of similar items to return"),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system", "dg_ds-browse"])),
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
        
        log.warning(
            "Fetched authorized entity IDs for user",
            authorized_count=len(authorized_set),
            authorized_ids=list(authorized_set)  # Show a sample of authorized IDs for debugging
        )

        if lookup_id not in authorized_set:
            log.warning(f"User {user_subject} not authorized for source entity {lookup_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for the requested entity_id.",
            )

        raw_recs = recs_client.get_recommendations(
            application="ds2ds",
            entity_id=request.entity_id,
            limit=None,  # Fetch all and filter/slice in-memory to ensure we can apply auth filtering before limiting
        )

        log.warning(
            "Fetched raw recommendations from Redis", 
            raw_count=len(raw_recs),
            raw_sample=raw_recs[:5]  # Shows the top 5 raw IDs to see what Redis returned
        )

        # Filter against authorized sets first
        authorized_recs = [item for item in raw_recs if item in authorized_set]

        dropped_count = len(raw_recs) - len(authorized_recs)
        if dropped_count > 0:
            log.info(
                "Filter applied: unauthorized recommendations removed",
                unauthorized_dropped_count=dropped_count,
                authorized_count=len(authorized_recs)
            )


        # Then slice to the requested 'n' and convert to Pydantic models
        filtered_recs = [
            Recommendation(entity_id=item)
            for item in authorized_recs[:request.n]
        ]

        log.warning(
            "Applied pagination/limit slicing",
            requested_n=request.n,
            final_returned_count=len(filtered_recs)
        )
        
        query_time = time.time() - start_time
        log.info(
            f"Returning {len(filtered_recs)} recs for {lookup_id} in {query_time:.3f}s"
        )

        return RecsResponse(
            entity_id=request.entity_id,
            recommendations=filtered_recs,
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
                        "Missing Edge": ap_errors_data.get("422_ap_missing_input_edge"),
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
            "Request": {
                "summary": "Standard Request",
                "description": "Infers entity_id from the incoming 'input' edge.",
                "value": ap_request_example,
            }
        },
    ),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
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
