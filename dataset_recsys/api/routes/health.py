import structlog
from fastapi import APIRouter, HTTPException

from dataset_recsys.storage.qdrant_client import QdrantStorageClient

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/dataset-recsys", tags=["Service Health"])
qdrant_client: QdrantStorageClient | None = None

def get_qdrant_client() -> QdrantStorageClient:
    """Connect to Qdrant only when an endpoint needs it."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantStorageClient()
    return qdrant_client

@router.get(
    "/health",
    summary="Health check",
    description="Check if the API, Redis, and vector database are responsive.",
    tags=["Service Health"],
)
async def health_check():
    try:
        is_qdrant_up = get_qdrant_client().check_connection()
        if not is_qdrant_up:
            logger.error(
                "Health check failed",
                qdrant=is_qdrant_up,
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
        schema = get_qdrant_client().get_schema_overview()
        return {"status": "ok", "schema": schema}
    except Exception as e:
        logger.error(f"Error fetching schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
