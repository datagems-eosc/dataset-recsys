import structlog
from fastapi import APIRouter, HTTPException

from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/dataset-recsys", tags=["Service Health"])
recs_client: RecommendationClient | None = None
embedding_client: EmbeddingClient | None = None


def get_recommendation_client() -> RecommendationClient:
    """Create the Redis client only when a health check needs it."""
    global recs_client
    if recs_client is None:
        recs_client = RecommendationClient()
    return recs_client


def get_embedding_client() -> EmbeddingClient:
    """Connect to PostgreSQL only when an endpoint needs it."""
    global embedding_client
    if embedding_client is None:
        embedding_client = EmbeddingClient()
    return embedding_client


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API, Redis, and vector database are responsive.",
    tags=["Service Health"],
)
async def health_check():
    try:
        is_redis_up = get_recommendation_client().check_connection()
        is_vector_db_up = get_embedding_client().check_connection()

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
        schema = get_embedding_client().get_schema_overview()
        return {"status": "ok", "schema": schema}
    except Exception as e:
        logger.error(f"Error fetching schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
