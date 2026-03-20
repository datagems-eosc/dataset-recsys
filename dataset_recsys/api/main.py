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
class Recommendation(BaseModel):
    id: str = Field(..., description="The recommended entity ID")

class RecsResponse(BaseModel):
    """Response model for recommendations."""
    query_time: float = Field(..., description="Time taken to process the recommendation query in seconds")
    application: str = Field(..., description="The application/dataset for which recommendations are returned")
    entity_id: str = Field(..., description="The input entity ID for which recommendations were requested")
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")

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
async def get_recommendations(
    application: str = Query(..., description="", enum=SUPPORTED_APPLICATIONS, required=True),
    entity_id: str = Query(..., description="The item identifier within the selected dataset", required=True),
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

    # 1. Validate application name
    application = application.lower()
    if application not in SUPPORTED_APPLICATIONS:
        logger.warning(f"Requested application '{application}' is not supported.")
        raise HTTPException(status_code=404, detail=f"Application '{application}' not found")

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
            return RecsResponse(dataset=application, iid=entity_id, recommendations=[])
        
        # 5. Get ranked recs from Redis
        raw_recs = recs_client.get_recommendations(application=application, entity_id=entity_id)

        # 6. Filter recs by authorization while PRESERVING ORDER    
        filtered_recs = [item for item in raw_recs if item in authorized_set]

        query_time = time.time() - start_time
        logger.info(f"Returning {len(filtered_recs[:n])} recs for {lookup_id} in {query_time:.3f}s")

        return RecsResponse(
            dataset=application,
            iid=entity_id,
            recommendations=filtered_recs[:n]
        )        

    except Exception as e:
        logger.error("Unexpected error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get(
    "/dataset-recsys/health",
    summary="Health check",
    description="Check if the API and the underlying Redis database are responsive.",
    tags=["Service Health"],
)
async def health_check():
    try:
        # 1. Check Redis connectivity via the client
        is_redis_up = recs_client.check_connection()
        
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
# uvicorn dataset_recsys.api.main:app --reload
# http://127.0.0.1:8000/dataset-recsys/redoc
# http://127.0.0.1:8000/dataset-recsys/docs

# Test api && redis connection:
# curl -X GET "http://127.0.0.1:8000/dataset-recsys/health" -v

# Test legacy recommendations:
# curl -G "http://127.0.0.1:8000/dataset-recsys/recommend" --data-urlencode "dataset=mathe" --data-urlencode "iid=6.pdf" --data-urlencode "n=10"

# Test v2 recommendations:
# curl -X POST "http://127.0.0.1:8000/dataset-recsys/v2/recommend" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN_HERE" -d '{"iid": "56.pdf", "n": 5, "dataset_ids": ["9b25bc46-8bd3-4f7f-94b4-52dbc38c130f"]}'