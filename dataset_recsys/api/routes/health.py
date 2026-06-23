import structlog
from fastapi import APIRouter, HTTPException

from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/dataset-recsys", tags=["Service Health"])
recs_client = RecommendationClient()
embedding_client = EmbeddingClient()

@router.get(
    "/health",
    summary="Health check",
    description="Check if the API, Redis, and vector database are responsive.",
    tags=["Service Health"],
)
async def health_check():
    try:
        is_redis_up = recs_client.check_connection()
        is_vector_db_up = embedding_client.check_connection()

        if not is_redis_up or not is_vector_db_up:
            logger.error(
                "Health check failed",
                redis=is_redis_up,
                vector_db=is_vector_db_up,
            )
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable",
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


@router.get(
    "/",
    summary="Root endpoint",
    description="Root endpoint to verify that the service is running.",
    tags=["Service Health"],
)
async def root():
    return {"status": "ok", "message": "Dataset Recommendation Service is running."}


@router.get(
    "/debug/schema",
    summary="Get database schema",
    description="Retrieve the database schema for the embedding storage.",
    tags=["Service Health"],
)
async def get_schema():
    try:
        if embedding_client is None:
            raise HTTPException(status_code=503, detail="Embedding client not initialized")
        schema = embedding_client.get_schema_overview()
        return {"status": "ok", "schema": schema}
    except Exception as e:
        logger.error(f"Error fetching schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")