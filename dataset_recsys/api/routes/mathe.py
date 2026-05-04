import os
import sys
from pathlib import Path
import time
from datetime import datetime

from pydantic import BaseModel
import structlog
from fastapi import APIRouter, HTTPException, Query, status

from dataset_recsys.api.analytical_patterns.models import Recommendation, RecsResponse
from dataset_recsys.storage.recommendation_client import RecommendationClient

project_root = "/app" 
if project_root not in sys.path:
    sys.path.append(project_root)
from data.mathe.syncer import MathE_Syncer

MATHE_PDF_PATH = Path(os.getenv("MATHE_PDF_PATH", "/mnt/s3/default"))
syncer = MathE_Syncer(base_dir=MATHE_PDF_PATH)

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")

router = APIRouter(prefix="/dataset-recsys/mathe", tags=["MathE Recommendation Service"])
recs_client = RecommendationClient()

class SyncResponse(BaseModel):
    message: str
    datasets_found: int
    source_folder: str

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

@router.post(
    "/sync",
    response_model=SyncResponse,
    summary="Synchronize MathE dataset",
    description="Loads the MathE dataset metadata from the mounted S3 volume and returns the count of items."
)
async def sync_datasets():
    try:
        # get() triggers _init_data() if self.data is None
        df = syncer.get()
        count = len(df)
        
        return SyncResponse(
            message="Synchronization successful.",
            datasets_found=count,
            source_folder=str(MATHE_PDF_PATH)
        )
    except FileNotFoundError as e:
        logger.error("Sync failed: File not found", error=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Sync failed: Unexpected error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during sync.")
