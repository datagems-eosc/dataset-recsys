import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, status, BackgroundTasks

from dataset_recsys.api.analytical_patterns.models import (
    MatheRecommendation,
    MatheRecsRequest,
    MatheRecsResponse,
)
from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
    recommend_from_curricular_pool,
)
from dataset_recsys.mathe_recommenders.video_pool_ranker import (
    recommend_videos_for_question,
)
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.utils.mathe_syncer import MathE_Syncer
from dataset_recsys.workflows.mathe_sync_pipeline import run_mathe_pipeline
from dataset_recsys.api.security import security

MATHE_PATH = Path(os.getenv("MATHE_PATH", "s3/default"))
MATHE_DATASET_ID = "9b25bc46-8bd3-4f7f-94b4-52dbc38c130f"
SYNC_HEARTBEAT_STALE_AFTER = timedelta(minutes=30)
syncer = MathE_Syncer(base_dir=MATHE_PATH)

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")

router = APIRouter(prefix="/dataset-recsys/mathe", tags=["MathE Recommendation Service"])
mathe_client: MatheMirrorClient | None = None
embedding_client: EmbeddingClient | None = None


def get_mathe_client() -> MatheMirrorClient:
    global mathe_client
    if mathe_client is None:
        mathe_client = MatheMirrorClient()
    return mathe_client


def get_embedding_client() -> EmbeddingClient:
    global embedding_client
    if embedding_client is None:
        embedding_client = EmbeddingClient()
    return embedding_client


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_sync_status(sync_status: dict) -> str:
    status = sync_status.get("sync_status", "never_run")
    if status != "running":
        return status

    heartbeat_at = _parse_utc_timestamp(sync_status.get("last_sync_heartbeat_at"))
    if heartbeat_at is None:
        return "stale"

    if datetime.now(timezone.utc) - heartbeat_at > SYNC_HEARTBEAT_STALE_AFTER:
        return "stale"

    return "running"


async def _get_content_recommendations(
    request: MatheRecsRequest,
    claims: dict,
    token: str,
    content_type: str,
    recommender: Callable[..., list[str]],
) -> MatheRecsResponse:
    start_time = time.time()
    user_subject = claims.get("sub")
    question_id = request.question_id.strip()
    question = request.question.strip()
    log = logger.bind(
        question_id=question_id,
        content_type=content_type,
        UserId=user_subject,
    )

    accounting_logger.info(
        "Recommendation Request Received",
        Action="get_recommendations",
        Resource="question_to_material_recommender",
        Domain="mathe",
        ContentType=content_type,
        QuestionId=question_id,
        UserId=user_subject,
        Timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    if not question_id:
        log.warning("Missing question_id.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question ID is required.",
        )
    if not question:
        log.warning("Missing question.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question text is required.",
        )

    try:
        try:
            question_id_int = int(question_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Question ID must be an integer.",
            )

        authorized_entities = await security.get_authorized_entity_ids(token)
        if MATHE_DATASET_ID not in authorized_entities:
            log.warning(
                "User %s not authorized for MathE dataset %s",
                user_subject,
                MATHE_DATASET_ID,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access MathE recommendations.",
            )

        recommended_material_ids = recommender(
            question_id=question_id_int,
            question=question,
            k=request.n,
            mathe_mirror_client=get_mathe_client(),
            embedding_client=get_embedding_client(),
        )
        if not recommended_material_ids:
            log.warning(
                "No %s recommendations found for question_id %s",
                content_type,
                question_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No {content_type} recommendations found for "
                    f"question_id {question_id}"
                ),
            )

        recommendations = [
            MatheRecommendation(material_id=material_id)
            for material_id in recommended_material_ids[: request.n]
        ]
        log.info(
            "Returning %d MathE %s recommendations in %.3fs",
            len(recommendations),
            content_type,
            time.time() - start_time,
        )
        return MatheRecsResponse(
            question_id=question_id,
            recommendations=recommendations,
        )
    except HTTPException:
        raise
    except Exception as error:
        log.error(
            "Unexpected error in MathE recommendations",
            error=str(error),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")


# TODO: Remove this compatibility route after every API client has migrated to
# /dataset-recsys/mathe/recommend/documents.
@router.post(
    "/recommend",
    response_model=MatheRecsResponse,
    summary="Get document material recommendations",
    description="""
Backward-compatible alias of `/dataset-recsys/mathe/recommend/documents`.
    """,
    deprecated=True,
)
@router.post(
    "/recommend/documents",
    response_model=MatheRecsResponse,
    summary="Get document material recommendations for a math question",
    description="""
Given a MathE question ID, return a list of recommended document teaching materials.
    """,
)
async def get_document_recommendations(
    request: MatheRecsRequest = Body(...),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
    token: str = Depends(security.oauth2_scheme),
):
    return await _get_content_recommendations(
        request=request,
        claims=claims,
        token=token,
        content_type="document",
        recommender=recommend_from_curricular_pool,
    )


@router.post(
    "/recommend/videos",
    response_model=MatheRecsResponse,
    summary="Get video recommendations for a math question",
    description="""
Given a MathE question ID, return video lessons and reviews from its curricular pool.
    """,
)
async def get_video_recommendations(
    request: MatheRecsRequest = Body(...),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
    token: str = Depends(security.oauth2_scheme),
):
    return await _get_content_recommendations(
        request=request,
        claims=claims,
        token=token,
        content_type="video",
        recommender=recommend_videos_for_question,
    )


@router.post("/sync")
async def sync_data(
    background_tasks: BackgroundTasks,
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
    token: str = Depends(security.oauth2_scheme),
):
    """
    Triggers the sync, OCR, and recommendation refresh process.
    Uses BackgroundTasks so the API returns immediately.
    """
    claims = claims if isinstance(claims, dict) else {}
    user_subject = claims.get("sub")
    accounting_logger.info(
        "Sync Pipeline Triggered",
        Action="sync_data",
        Domain="mathe",
        UserId=user_subject,
        Timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )    
    # Trigger the lifecycle as a background task
    background_tasks.add_task(run_mathe_pipeline, syncer)
    
    return {
        "message": "Sync, OCR, and recommendation refresh initiated.",
        "status": "Accepted",
        "details": "The system is now discovering new PDFs, processing OCR, and rebuilding MathE recommendations in the background."
    }

@router.get("/status")
async def get_status(
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
    token: str = Depends(security.oauth2_scheme),    
):
    """Returns the current MathE sync status and metadata about the materials being processed."""
    claims = claims if isinstance(claims, dict) else {}
    user_subject = claims.get("sub")
    accounting_logger.info(
        "Sync Status Requested",
        Action="get_status",
        Domain="mathe",
        UserId=user_subject,
        Timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    df = syncer.get()
    total = len(df)
    completed = int(df[df["status"] == "completed"].shape[0]) if "status" in df else 0
    pending = int(df[df["status"] == "pending"].shape[0]) if "status" in df else 0
    failed_material_ids = (
        df.loc[df["status"] == "failed", "material_id"].dropna().astype(str).tolist()
        if {"status", "material_id"}.issubset(df.columns)
        else []
    )
    sync_status = syncer.get_sync_status()
    
    return {
        "sync_status": _resolve_sync_status(sync_status),
        "last_sync_started_at": sync_status.get("last_sync_started_at"),
        "last_sync_heartbeat_at": sync_status.get("last_sync_heartbeat_at"),
        "last_sync_completed_at": sync_status.get("last_sync_completed_at"),
        "total_materials": total,
        "ocr_completed_materials": completed,
        "ocr_pending_materials": pending,
        "ocr_failed_material_ids": failed_material_ids,
        "embeddings_created": sync_status.get("embeddings_created", 0),
    }
