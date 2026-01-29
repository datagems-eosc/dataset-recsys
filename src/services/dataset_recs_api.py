import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.recommendation_client import RecommendationClient
from typing import List
import json
from pathlib import Path
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_recs_api")

app = FastAPI(
    openapi_url="/dataset-recsys/openapi.json",
    docs_url="/dataset-recsys/docs",
    redoc_url="/dataset-recsys/redoc",
)
recs_client = RecommendationClient()

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
SUPPORTED_DATASETS = ["mathe"]  # Extend this list to support more datasets

class ItemToItemRecsResponse(BaseModel):
    dataset: str = Field(..., description="The dataset/application name")
    iid: str = Field(..., description="The item identifier within the selected dataset")
    recommendations: List[str] = Field(
        ...,
        description="List of recommended item identifiers"
    )


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
                    "examples": examples_data
                }
            }
        },
        404: errors_data.get("404"),
        422: errors_data.get("422")
    }
)
def get_recommendations(
    dataset: str = Query(..., description="The dataset/application name", enum=SUPPORTED_DATASETS),
    iid: str = Query(..., description="The item identifier within the selected dataset"),
    n: int = Query(10, le=20, description="Number of similar items to return")
):
    try:
        available_usecases = recs_client.list_usecases()     
        if dataset not in available_usecases:
            logger.warning(f"Dataset '{dataset}' not found")
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")

        recs_set = recs_client.get_recommendations(usecase=dataset, pdf=iid)
        if not recs_set:
            logger.warning(f"Item ID '{iid}' not found in dataset '{dataset}'")
            raise HTTPException(status_code=404, detail=f"Item ID '{iid}' not found in dataset '{dataset}'")

        recs_list = list(recs_set)[:n]
        logger.info(f"Returning {len(recs_list)} recommendations for dataset={dataset}, iid={iid}")
        return ItemToItemRecsResponse(
            dataset=dataset,
            iid=iid,
            recommendations=recs_list
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error while getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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

        # 2. Optional: Check if any data exists at all
        usecases = recs_client.list_usecases()
        
        return {
            "status": "ok",
            "database": "connected",
            "available_datasets": len(usecases),
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

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail} (path: {request.url.path})")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "errors": [
                {
                    "code": exc.status_code,
                    "detail": exc.detail
                }
            ]
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"RequestValidationError: {exc.errors()} (path: {request.url.path})")
    return JSONResponse(
        status_code=422,
        content={
            "errors": [
                {
                    "code": 422,
                    "detail": exc.errors()
                }
            ]
        }
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

# Test recommendations:
# curl -G "http://127.0.0.1:8000/dataset-recsys/recommend" --data-urlencode "dataset=mathe" --data-urlencode "iid=6.pdf" --data-urlencode "n=10"