import os
import sys
from pathlib import Path
import time
from datetime import datetime

from pydantic import BaseModel
import structlog
from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks

from dataset_recsys.api.analytical_patterns.models import Recommendation, RecsResponse
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.utils.mathe_syncer import MathE_Syncer

MATHE_PATH = Path(os.getenv("MATHE_PATH", "/mnt/s3/default"))
MATHE_PDF_PATH = MATHE_PATH / "pdfs"
syncer = MathE_Syncer(base_dir=MATHE_PDF_PATH)

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")

router = APIRouter(prefix="/dataset-recsys/mathe", tags=["MathE Recommendation Service"])
recs_client = RecommendationClient()

@router.post(
    "/recommend",
    response_model=RecsResponse,
    summary="Get recommendations",
    description="""
Retrieve the top-N recommendations for a given educational material (only PDFs are currently supported).
    """,
)
async def get_recommendations(
    entity_id: str = Query(
        ...,
        description="The MathE material identifier (for example, `6.pdf`).",
        required=True,
    ),
    n: int = Query(10, gt=0, le=20, description="Number of similar items to return"),
):
    start_time = time.time()

    log = logger.bind(item_id=entity_id)
    accounting_logger.info(
        "Recommendation Request Received",
        Action="get_recommendations",
        Resource="dataset2dataset_recommender",
        Domain="mathe",
        ItemId=entity_id,
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    entity_id = entity_id.strip()
    if not entity_id:
        log.warning("Missing entity_id.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Entity ID is required.",
        )

    try:
        raw_recs = recs_client.get_recommendations(
            application="ds2ds_mathe", entity_id=entity_id
        )

        filtered_recs = [Recommendation(entity_id=item) for item in raw_recs]

        query_time = time.time() - start_time
        log.info(
            f"Returning {len(filtered_recs[:n])} MathE recs for {entity_id} in {query_time:.3f}s"
        )

        return RecsResponse(
            entity_id=entity_id,
            recommendations=filtered_recs[:n],
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error in MathE recommendations", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/sync")
async def sync_data(background_tasks: BackgroundTasks):
    """
    Triggers the sync and OCR process.
    Uses BackgroundTasks so the API returns immediately.
    """
    # Trigger the lifecycle as a background task
    background_tasks.add_task(syncer.sync_and_process)
    
    return {
        "message": "Sync and OCR process initiated.",
        "status": "Accepted",
        "details": "The system is now discovering new PDFs and processing them in the background."
    }

@router.get("/status")
async def get_status():
    """Returns the current state of the dataset."""
    df = syncer.get()
    total = len(df)
    completed = int(df[df['status'] == 'completed'].shape[0])
    failed = int(df[df['status'] == 'failed'].shape[0])
    
    # Calculate percentage
    progress = (completed + failed) / total * 100 if total > 0 else 100
    
    return {
        "progress_percent": round(progress, 2),
        "total_files": total,
        "completed": completed,
        "failed": failed,
        "is_syncing": syncer.is_running
    }