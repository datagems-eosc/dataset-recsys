from ast import Dict
from email.mime import application
import time
from datetime import datetime
import logging
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status, Query, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.api.logging.logging_config import (
    request_response_logging_middleware,
    correlation_id_middleware,
    setup_logging,
)
from dataset_recsys.api.logging.exceptions import (
    ErrorResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
    FailedDependencyMessage,
    FailedDependencyResponse,
    FailedDependencyException,
)
from dataset_recsys.api.security import security
from dataset_recsys.api.analytical_patterns.ap_handling import (
    parse_recommendation_request_ap,
    create_recommendation_response_ap,
)
from dataset_recsys.api.analytical_patterns.models import RecsRequest, RecsResponse, Recommendation
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

# Configure logging
setup_logging()
logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")
app_state = {}
SUPPORTED_APPLICATIONS = ["portal", "mathe"]
MATHE_DATASET_ID = "b551f361-3f61-4ccf-a001-7c28d065c30d"

app = FastAPI(
    openapi_url="/dataset-recsys/openapi.json",
    docs_url="/dataset-recsys/docs",
    redoc_url="/dataset-recsys/redoc",
)
recs_client = RecommendationClient()
# --- Middleware ---
app.middleware("http")(request_response_logging_middleware)
app.middleware("http")(correlation_id_middleware)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
AP_DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_valid_examples.json")
AP_DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_error_examples.json")
AP_REQUEST_EXAMPLE_PATH = Path("dataset_recsys/api/api_docs/ap_request_example.json")
AP_REQUEST_MATHE_EXAMPLE_PATH = Path("dataset_recsys/api/api_docs/ap_request_pilot.json")
DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/valid_examples.json")
DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/error_examples.json")

def load_json_file(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"File '{path}' does not exist.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load file from {path}: {e}")
        return {}

examples_data, errors_data = (load_json_file(DOCS_VALID_EXAMPLES_PATH), load_json_file(DOCS_ERROR_EXAMPLES_PATH))
ap_examples_data, ap_errors_data = (load_json_file(AP_DOCS_VALID_EXAMPLES_PATH), load_json_file(AP_DOCS_ERROR_EXAMPLES_PATH))
ap_request_example = load_json_file(AP_REQUEST_EXAMPLE_PATH)
ap_request_mathe_example = load_json_file(AP_REQUEST_MATHE_EXAMPLE_PATH)

# --- API Endpoints ---
@app.post(
    "/dataset-recsys/recommend",
    response_model=RecsResponse,
    summary="Get recommendations",
    description="""
Retrieve the top-N recommendations for a given entity within an application.

The meaning of *entity* varies by application:
- **MathE** — entities are educational materials (PDFs), whose identifiers correspond to filenames with `.pdf` extension (e.g., `{entity_id}.pdf`).
- **Portal** — entities are datasets, identified by their unique dataset IDs (e.g., `9b25bc46-8bd3-4f7f-94b4-52dbc38c130f`).
    """,
    tags=["Dataset Recommendation Service"],
    responses={
        200: {"description": "Successful retrieval of related datasets", "content": {"application/json": {"examples": examples_data}}},
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": {
                        "Missing Application": errors_data.get("422_missing_application"),
                        "Missing Entity ID": errors_data.get("422_missing_entity_id")
                    }
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing token",
            "content": {
                "application/json": {
                    "examples": {
                        "Invalid Token": errors_data.get("401")
                    }
                }
            }
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "examples": {
                        "Access Denied": errors_data.get("403")
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {
                        "System Failure": errors_data.get("500")
                    }
                }
            }
        }
    }
)
async def get_recommendations(
    application: str = Query(..., description="", enum=SUPPORTED_APPLICATIONS, required=True),
    entity_id: str = Query(..., description="The entity identifier within the selected application.", required=True),
    n: int = Query(10, le=20, description="Number of similar items to return"),
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme),     
):
    start_time = time.time()
    user_subject = claims.get("sub")
    log = logger.bind(item_id=MATHE_DATASET_ID if application == "mathe" else entity_id, UserId=user_subject)
    accounting_logger.info(
        "Recommendation Request Received",
        UserId=user_subject,
        Action="get_recommendations",
        Resource="DatasetRecommendations",
        Type="+",
        Value=1,
        Measure="Unit",
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    # 1. Validate application name and entity_id format
    application = application.lower()
    if application not in SUPPORTED_APPLICATIONS:
        logger.warning(f"Requested application '{application}' is not supported.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=f"Application '{application}' is not supported."
        )
    
    entity_id = entity_id.strip()
    if not entity_id:
        logger.warning("Entity ID is empty after stripping.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Entity ID cannot be empty."
        )

    # 2. Logic for Mathe vs Portal
    # If application is mathe, we force the lookup to the fixed MATHE_DATASET_ID
    lookup_id = MATHE_DATASET_ID if application.lower() == "mathe" else entity_id
    try:
        # 3. Fetch authorized IDs for the user from the security module
        authorized_list = await security.get_authorized_entity_ids(token)
        authorized_set = set(authorized_list)

        # 4. Check if user is authorized to even see the source entity
        if lookup_id not in authorized_set:
            logger.warning(f"User {user_subject} not authorized for source entity {lookup_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for the requested entity_id."
            )

        # 5. Get ranked recs from Redis
        raw_recs = recs_client.get_recommendations(application=application, entity_id=entity_id)

        # 6. Filter recs by authorization while PRESERVING ORDER    
        filtered_recs = [Recommendation(id=item) for item in raw_recs if item in authorized_set]

        query_time = time.time() - start_time
        logger.info(f"Returning {len(filtered_recs[:n])} recs for {lookup_id} in {query_time:.3f}s")

        return RecsResponse(
            dataset=application,
            iid=entity_id,
            recommendations=filtered_recs[:n]
        )        
    except HTTPException:
        # Re-raise HTTP exceptions so they aren't caught by the general Exception block
        raise
    except Exception as e:
        logger.error("Unexpected error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post(
    "/dataset-recsys/recommend/ap", 
    response_model=RecsResponse,
    tags=["Analytical Pattern Handling"],
    summary="Get recommendations via Analytical Pattern",
description="""
Processes a graph-based **Analytical Pattern (AP)** request through the following flow:

1. **Extraction**: Identifies the `application`, `n`, and (optionally) `entity_id` from the **DatasetRecommender_Operator** node properties.
2. **Identification**: 
   - For **MathE**: Uses the `entity_id` provided directly in the operator's properties.
   - For **Portal**: Identifies the seed dataset by tracing the incoming **input** edge to the operator.
3. **Engine Query**: Calls the recommendation engine to find related entities.
4. **Graph Injection**: Creates new nodes for recommendations:
   - **MathE**: Generates **cr:FileObject** nodes with unique UUIDs.
   - **Portal**: Generates **sc:Dataset** nodes.
5. **Relationship Mapping**: Connects the Operator to each new node via **output** edges, assigning a `rank` property to preserve the order of relevance.
    """,
    responses={
        200: {"description": "Successful graph transformation", "content": {"application/json": {"examples": ap_examples_data}}},
        403: {
                "description": "Authorization Failure",
                "content": {
                    "application/json": {
                        "examples": {
                            "Insufficient Permissions": errors_data.get("403")
                        }
                    }
                }
            },
            422: {
                "description": "Malformed AP Graph",
                "content": {
                    "application/json": {
                        "examples": {
                            "MathE Missing Property": ap_errors_data.get("422_ap_missing_mathe_id"),
                            "Portal Missing Edge": ap_errors_data.get("422_ap_missing_input_edge"),
                            "Operator Missing": ap_errors_data.get("422_ap_missing_operator")
                        }
                    }
                }
            },
            500: {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {
                        "examples": {
                            "Server Error": errors_data.get("500")
                        }
                    }
                }
            }
        }
)
async def get_recommendations_ap(
    analytical_pattern: dict = Body(
        ...,
        description="The Analytical Pattern graph in JSON format",
        openapi_examples={
            "Portal Request": {
                "summary": "Standard Portal Request",
                "description": "Infers entity_id from the incoming 'input' edge.",
                "value": ap_request_example  # Your existing portal JSON
            },
            "MathE Request": {
                "summary": "MathE Pilot",
                "description": "Uses the 'entity_id' property from the operator node.",
                "value": ap_request_mathe_example  # Your MathE-specific JSON
            }
        }
    ),
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme),
):
    try:
        search_request = parse_recommendation_request_ap(analytical_pattern)
        search_response = await get_recommendations(
            search_request.application, 
            search_request.entity_id, 
            search_request.n, 
            claims, 
            token
        )
        updated_ap = create_recommendation_response_ap(analytical_pattern, search_response)
        return updated_ap
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing recommendation AP request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get(
    "/dataset-recsys/health",
    summary="Health check",
    description="Check if the API, Redis, and vector database are responsive.",
    tags=["Service Health"],
)
async def health_check():
    try:
        # --- Redis check ---
        is_redis_up = recs_client.check_connection()

        # --- pgvector / Postgres check ---
        embedding_client = EmbeddingClient()
        is_vector_db_up = embedding_client.check_connection()

        if not is_redis_up or not is_vector_db_up:
            logger.error(
                "Health check failed",
                redis=is_redis_up,
                vector_db=is_vector_db_up,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service Unavailable",
                    "redis": is_redis_up,
                    "vector_db": is_vector_db_up,
                },
            )

        return {
            "status": "ok",
            "redis": "connected",
            "vector_db": "connected",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.get(
    "/dataset-recsys/",
    summary="Root endpoint",
    description="Root endpoint to verify that the service is running.",
    tags=["Service Health"],
)
async def root():
    return {"status": "ok", "message": "Dataset Recommendation Service is running."}

@app.get(
    "/dataset-recsys/debug/schema",
    summary="Get database schema",
    description="Retrieve the database schema for the embedding storage.",
    tags=["Service Health"],
)
async def get_schema():
    try:
        embedding_client = EmbeddingClient()
        schema = embedding_client.get_schema_overview()
        return {"status": "ok", "schema": schema}
    except Exception as e:
        logger.error(f"Error fetching schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Exception Handlers ---
@app.exception_handler(FailedDependencyException)
async def failed_dependency_exception_handler(
    request: Request, exc: FailedDependencyException
):
    response_content = FailedDependencyResponse(
        code=104,
        error="error communicating with underpinning service",
        message=FailedDependencyMessage(
            statusCode=exc.downstream_status_code,
            source=exc.source,
            correlationId=exc.correlation_id,
            payload=exc.downstream_payload,
        ),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content.model_dump(exclude_none=True),
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=exc.status_code, error=exc.detail).model_dump(),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        ValidationErrorDetail(
            Key=".".join(map(str, err.get("loc", []))), Value=[err.get("msg", "")]
        )
        for err in exc.errors()
    ]
    response_content = ValidationErrorResponse(
        code=102, error="Validation Error", message=details
    )
    # Changed from 400 to 422 to match OpenAPI docs
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
        content=response_content.model_dump()
    )

# Enable port forwarding to Redis before running the app:
# export KUBECONFIG=~/path/to/.kubeconfig
# kubectl port-forward pod/dataset-recsys-redis-5547b598b7-mngqk -n athenarc 6380:6379

# Run the API with:
# uvicorn dataset_recsys.api.main:app --reload
# http://127.0.0.1:8000/dataset-recsys/redoc
# http://127.0.0.1:8000/dataset-recsys/docs

# Test api && redis connection:
# curl -X GET "http://127.0.0.1:8000/dataset-recsys/health" -v

# Test legacy recommendations:
# curl -G "http://127.0.0.1:8000/dataset-recsys/recommend" --data-urlencode "dataset=mathe" --data-urlencode "iid=6.pdf" --data-urlencode "n=10"

# Test v2 recommendations:
# curl -X POST "http://127.0.0.1:8000/dataset-recsys/v2/recommend" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN_HERE" -d '{"iid": "56.pdf", "n": 5, "dataset_ids": ["9b25bc46-8bd3-4f7f-94b4-52dbc38c130f"]}'