import os
import sys
from pathlib import Path
import time
from datetime import datetime

from pydantic import BaseModel
import structlog
from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks

from dataset_recsys.api.analytical_patterns.models import MatheRecommendation, MatheRecsResponse
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.utils.mathe_syncer import MathE_Syncer
from dataset_recsys.workflows.mathe_sync_pipeline import run_mathe_pipeline

MATHE_PATH = Path(os.getenv("MATHE_PATH", "/mnt/s3/default"))
MATHE_PDF_PATH = MATHE_PATH / "pdfs"
syncer = MathE_Syncer(base_dir=MATHE_PDF_PATH)

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")

router = APIRouter(prefix="/dataset-recsys/mathe", tags=["MathE Recommendation Service"])
recs_client = RecommendationClient()
mathe_client = MatheMirrorClient()

@router.post(
    "/recommend",
    response_model=MatheRecsResponse,
    summary="Get recommendations",
    description="""
Retrieve the top-N recommendations for a given educational material (only PDFs are currently supported).
    """,
)
async def get_recommendations(
    question_id: str = Query(
        ...,
        description="The MathE question identifier (for example, `6`).",
        required=True,
    ),
    n: int = Query(10, gt=0, description="Number of similar items to return"),
):
    start_time = time.time()

    log = logger.bind(item_id=question_id)
    accounting_logger.info(
        "Recommendation Request Received",
        Action="get_recommendations",
        Resource="dataset2dataset_recommender",
        Domain="mathe",
        ItemId=question_id,
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    question_id = question_id.strip()
    if not question_id:
        log.warning("Missing question_id.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question ID is required.",
        )

    try:
        material_id = mathe_client.get_material_id_by_question_id(question_id)
        # Cast to string if not None, else keep as None
        material_id = str(material_id) if material_id is not None else None
        if not material_id:
            log.warning(f"No material found for question_id {question_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No material found for question_id {question_id}",
            )

        raw_recs = recs_client.get_recommendations(
            application="mathe", entity_id= material_id + ".pdf"
        )

        filtered_recs = [MatheRecommendation(material_id=item) for item in raw_recs]

        query_time = time.time() - start_time
        log.info(
            f"Returning {len(filtered_recs[:n])} MathE recs for {material_id} in {query_time:.3f}s"
        )

        return MatheRecsResponse(
            question_id=question_id,
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
    Triggers the sync, OCR, and recommendation refresh process.
    Uses BackgroundTasks so the API returns immediately.
    """
    # Trigger the lifecycle as a background task
    background_tasks.add_task(run_mathe_pipeline, syncer)
    
    return {
        "message": "Sync, OCR, and recommendation refresh initiated.",
        "status": "Accepted",
        "details": "The system is now discovering new PDFs, processing OCR, and rebuilding MathE recommendations in the background."
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
