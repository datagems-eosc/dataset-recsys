import time
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from dataset_recsys.api.analytical_patterns.models import Recommendation, RecsResponse
from dataset_recsys.storage.recommendation_client import RecommendationClient

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
            application="mathe", entity_id=entity_id
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