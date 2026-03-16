from ast import Dict
import time
from datetime import datetime
import logging
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status, Query, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.recommendation_client import RecommendationClient
from src.legacy_client import LegacyClient
from src.configs.logging_config import (
    request_response_logging_middleware,
    correlation_id_middleware,
    setup_logging,
)
from src.configs.exceptions import (
    ErrorResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
    FailedDependencyMessage,
    FailedDependencyResponse,
    FailedDependencyException,
)
from src.configs import security
from dataset_recsys.api.models import SearchRequest, SearchResponse, API_SearchResult
from dataset_recsys.api.legacy_models import ItemToItemRecsResponse
from src.ap_handling import parse_recommendation_request_ap, create_recommendation_response_ap
import json
from pathlib import Path

# Configure logging
setup_logging()
logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")
app_state = {}

logging.basicConfig(level=logging.INFO)
legacy_logger = logging.getLogger("dataset_recs_api")
SUPPORTED_DATASETS = ["mathe"]

app = FastAPI(
    openapi_url="/dataset-recsys/openapi.json",
    docs_url="/dataset-recsys/docs",
    redoc_url="/dataset-recsys/redoc",
)
recs_client = RecommendationClient()
legacy_recs_client = LegacyClient()

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
DOCS_VALID_EXAMPLES_PATH = Path("src/services/api_docs/valid_examples.json")
DOCS_ERROR_EXAMPLES_PATH = Path("src/services/api_docs/error_examples.json")
AP_DOCS_VALID_EXAMPLES_PATH = Path("src/services/api_docs/ap_valid_examples.json")
AP_DOCS_ERROR_EXAMPLES_PATH = Path("src/services/api_docs/ap_error_examples.json")
AP_DOCS_REQ_EXAMPLE_PATH = Path("src/services/api_docs/ap_request_example.json")
LEGACY_VALID_EXAMPLES_PATH = Path("src/services/api_docs/legacy_valid_examples.json")
LEGACY_ERROR_EXAMPLES_PATH = Path("src/services/api_docs/legacy_error_examples.json")

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
legacy_examples_data, legacy_errors_data = (load_json_file(LEGACY_VALID_EXAMPLES_PATH), load_json_file(LEGACY_ERROR_EXAMPLES_PATH))
ap_examples_data, ap_errors_data = (load_json_file(AP_DOCS_VALID_EXAMPLES_PATH), load_json_file(AP_DOCS_ERROR_EXAMPLES_PATH))
ap_request_example = load_json_file(AP_DOCS_REQ_EXAMPLE_PATH)
# --- API Endpoints ---
@app.get(
    "/dataset-recsys/recommend",
    response_model=ItemToItemRecsResponse,
    summary="Get recommendations",
    description="""
Retrieve the top-N recommendations for a given item in a dataset.

The meaning of *item* varies by dataset:
- **MathE** — items are educational materials (PDFs), whose identifiers correspond to filenames with `.pdf` extension (e.g., `{item_id}.pdf`).
    """,
    tags=["Dataset Recommendation Service"],
    responses={
        200: {
            # "description": "Successful response examples for supported datasets",
            "content": {
                "application/json": {
                    "examples": legacy_examples_data
                }
            }
        },
        404: legacy_errors_data.get("404"),
        422: legacy_errors_data.get("422")
    }
)
def get_recommendations_legacy(
    dataset: str = Query(..., description="The dataset/application name", enum=SUPPORTED_DATASETS),
    iid: str = Query(..., description="The item identifier within the selected dataset"),
    n: int = Query(10, le=20, description="Number of similar items to return")
):
    try:
        available_usecases = legacy_recs_client.list_usecases()     
        if dataset not in available_usecases:
            legacy_logger.warning(f"Dataset '{dataset}' not found")
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")

        recs_set = legacy_recs_client.get_recommendations(usecase=dataset, pdf=iid)
        if not recs_set:
            legacy_logger.warning(f"Item ID '{iid}' not found in dataset '{dataset}'")
            raise HTTPException(status_code=404, detail=f"Item ID '{iid}' not found in dataset '{dataset}'")

        recs_list = list(recs_set)[:n]
        legacy_logger.info(f"Returning {len(recs_list)} recommendations for dataset={dataset}, iid={iid}")
        return ItemToItemRecsResponse(
            dataset=dataset,
            iid=iid,
            recommendations=recs_list
        )
    except HTTPException:
        raise
    except Exception as e:
        legacy_logger.error(f"Unexpected error while getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post(
    "/dataset-recsys/v2/recommend",
    response_model=SearchResponse,
    tags=["V2 Recommendation Service"],
    summary="Get Related Datasets",
    description="""
Given a source `dataset_id`, retrieve a list of related or recommended **datasets**.
All results are filtered based on the user's authorization claims.
    """,
    responses={
        200: {
            "description": "Successful retrieval of related datasets",
            "content": {
                "application/json": {
                    "examples": examples_data  # Loaded from valid_examples.json
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": errors_data.get("422")
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing token",
            "content": {
                "application/json": {
                    "examples": errors_data.get("401")
                }
            }
        },
        403: {
            "description": "Forbidden - Insufficient permissions for the requested dataset",
            "content": {
                "application/json": {
                    "examples": errors_data.get("403")
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": errors_data.get("500")
                }
            }
        }
    }
)
async def get_recommendations_v2(
    request: SearchRequest,
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme), 
):
    start_time = time.time()
    user_subject = claims.get("sub")
    log = logger.bind(item_id=request.dataset_id, UserId=user_subject)
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
    dataset_id = request.dataset_id
    n = request.n
    try:
        authorized_dataset_ids = await security.get_authorized_dataset_ids(token)
        
        target_dataset = list(set([dataset_id]).intersection(authorized_dataset_ids)) if request.dataset_id else authorized_dataset_ids
        if not target_dataset:
            log = log.bind(requested_dataset_ids=dataset_id)
            log.warning(
                    "User requested datasets they are not authorized for. Returning empty results."
                )
            return SearchResponse(query_time=0, dataset_id=request.dataset_id, recommendations=[])

        recs_set = recs_client.get_recommendations(dataset_id=target_dataset)
        if not recs_set:
            logger.info(f"No recommendations found for dataset_id='{dataset_id}'")
            return SearchResponse(query_time=time.time() - start_time, dataset_id=target_dataset, recommendations=[])

        recs_list = list(recs_set).intersection(authorized_dataset_ids)
        if not recs_list:
            log = log.bind(recommended_datasets=recs_list)
            log.warning(
                    "Recommendations found but user is not authorized to access any of them. Returning empty results."
                )
            return SearchResponse(query_time=time.time() - start_time, dataset_id=target_dataset, recommendations=[])


        query_time = time.time() - start_time
        final_response = SearchResponse(
            query_time=query_time,
            dataset_id=target_dataset,
            recommendations= [API_SearchResult(item_id=rec) for rec in recs_list[:n]]
        )
        logger.info(f"Found {len(recs_list)} total recommendations for dataset_id='{dataset_id}' in dataset '{target_dataset}' (returning top {n}) in {query_time:.2f} seconds")
        return final_response

    except Exception as e:
        log.error(
            "An unexpected error occurred during search.", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )

@app.post(
    "/dataset-recsys/recommend/ap", 
    response_model=dict, 
    tags=["Analytical Pattern Handling"],
    summary="Get recommendations via Analytical Pattern",
    description="""
Processes a graph-based **Analytical Pattern (AP)** request. 

1. Extracts the `dataset_id` and `n` from the **DatasetRecommender_Operator** node.
2. Queries the recommendation engine.
3. Injects the recommended datasets as new **sc:Dataset** nodes.
4. Links the Operator to the new nodes via **output** edges with a `rank` property.
    """,
    responses={
        200: {
            "description": "Successful graph transformation",
            "content": {
                "application/json": {
                    "examples": {
                        "Dataset-to-Dataset AP Success": {
                            "summary": "Successful AP transformation",
                            "value": ap_examples_data  # Ensure this variable loads your AP valid JSON
                        }
                    }
                }
            }
        },
        403: {
            "description": "Authorization Failure",
            "content": {
                "application/json": {
                    "example": {
                        "code": 403,
                        "error": "Forbidden",
                        "message": "User not authorized for the requested dataset."
                    }
                }
            }
        },
        422: {
            "description": "Malformed AP Graph",
            "content": {
                "application/json": {
                    "example": {
                        "code": 422,
                        "error": "Unprocessable Entity",
                        "message": "Required Operator node or dataset_id property missing in AP."
                    }
                }
            }
        }
    }    
)
async def get_recommendations_ap(
    analytical_pattern: dict = Body(..., description="The Analytical Pattern graph in JSON format", example=ap_request_example),  # Ensure this variable loads your AP valid JSON
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    token: str = Depends(security.oauth2_scheme),
):
    try:
        search_request = parse_recommendation_request_ap(analytical_pattern)
        search_response = await get_recommendations_v2(search_request, claims, token)
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
    description="Check if the API and the underlying Redis database are responsive.",
    tags=["Service Health"],
)
async def health_check():
    try:
        # 1. Check Redis connectivity via the client
        is_redis_up = legacy_recs_client.check_connection()
        
        if not is_redis_up:
            logger.error("Health check failed: Redis is unreachable.")
            raise HTTPException(
                status_code=503, 
                detail="Service Unavailable: Database connection failed"
            )

        return {
            "status": "ok",
            "database": "connected",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check encountered an unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.get(
    "/dataset-recsys/",
    summary="Root endpoint",
    description="Root endpoint to verify that the service is running.",
    tags=["Service Health"],
)
async def root():
    return {"status": "ok", "message": "Dataset Recommendation Service is running."}

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
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content=response_content.model_dump()
    )

# Enable port forwarding to Redis before running the app:
# export KUBECONFIG=~/path/to/.kubeconfig
# kubectl port-forward pod/dataset-recsys-redis-5547b598b7-mngqk -n athenarc 6380:6379

# Run the API with:
# uvicorn src.services.dataset_recs_api:app --reload
# http://127.0.0.1:8000/dataset-recsys/redoc
# http://127.0.0.1:8000/dataset-recsys/docs

# Test api && redis connection:
# curl -X GET "http://127.0.0.1:8000/dataset-recsys/health" -v

# Test legacy recommendations:
# curl -G "http://127.0.0.1:8000/dataset-recsys/recommend" --data-urlencode "dataset=mathe" --data-urlencode "iid=6.pdf" --data-urlencode "n=10"

# Test v2 recommendations:
# curl -X POST "http://127.0.0.1:8000/dataset-recsys/v2/recommend" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN_HERE" -d '{"iid": "56.pdf", "n": 5, "dataset_ids": ["9b25bc46-8bd3-4f7f-94b4-52dbc38c130f"]}'