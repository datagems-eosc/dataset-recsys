import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone

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
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.utils.mathe_syncer import MathE_Syncer
from dataset_recsys.workflows.mathe_sync_pipeline import run_mathe_pipeline
from dataset_recsys.api.security import security

MATHE_PATH = Path(os.getenv("MATHE_PATH", "/mnt/s3/default"))
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


@router.post(
    "/recommend",
    response_model=MatheRecsResponse,
    summary="Get material recommendations for a math question",
    description="""
Given a MathE question ID, return a list of recommended PDF materials. 
    """,
)
async def get_recommendations(
    request: MatheRecsRequest = Body(...),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
    token: str = Depends(security.oauth2_scheme),
):
    start_time = time.time()
    user_subject = claims.get("sub")
    question_id = request.question_id

    log = logger.bind(question_id=question_id, UserId=user_subject)
    accounting_logger.info(
        "Recommendation Request Received",
        Action="get_recommendations",
        Resource="question_to_material_recommender",
        Domain="mathe",
        QuestionId=question_id,
        UserId=user_subject,
        Timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    question_id = question_id.strip()
    question = request.question.strip()
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

        # Fetch authorized items from security context
        authorized_list = await security.get_authorized_entity_ids(token)
        authorized_set = set(authorized_list)
        
        # Checj if the authorized set contains the mathe dataset ID
        if MATHE_DATASET_ID not in authorized_set:
            log.warning(f"User {user_subject} not authorized for MathE dataset {MATHE_DATASET_ID}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access MathE recommendations.",
            )

        recommended_material_ids = recommend_from_curricular_pool(
            question_id=question_id_int,
            question=question,
            k=request.n,
            mathe_mirror_client=get_mathe_client(),
            embedding_client=get_embedding_client(),
        )

        if not recommended_material_ids:
            log.warning(f"No PDF recommendations found for question_id {question_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No PDF recommendations found for question_id {question_id}",
            )

        filtered_recs = [
            MatheRecommendation(material_id=material_id)
            for material_id in recommended_material_ids
        ]

        query_time = time.time() - start_time
        log.info(
            f"Returning {len(filtered_recs[:request.n])} MathE material recs for question "
            f"{question_id} in {query_time:.3f}s"
        )

        return MatheRecsResponse(
            question_id=question_id,
            recommendations=filtered_recs[:request.n],
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Unexpected error in MathE recommendations", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
