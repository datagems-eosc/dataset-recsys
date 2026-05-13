from datetime import datetime
from typing import List

import structlog
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from dataset_recsys.api.api_docs_loader import DOCS_ERROR_EXAMPLES_PATH, load_json_file
from dataset_recsys.api.security import security
from dataset_recsys.ingestion.moma_dataset import MomaDataset
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.workflows.incremental_update import process_incremental_update

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")
router = APIRouter(prefix="/dataset-recsys", tags=["DataGEMS Dataset Management"])
recs_client = RecommendationClient()

DEFAULT_APPLICATION = "ds2ds"

errors_data = load_json_file(DOCS_ERROR_EXAMPLES_PATH)

@router.post(
    "/dataset/add",
    summary="Add a dataset",
    description="""
Adds a dataset to the DataGEMS recommender.

The service retrieves the dataset metadata, builds its embedding, stores it in the vector database, and updates the recommendation index. If the dataset already exists, the request is ignored to avoid duplicate entries.
    """,
    responses={
        200: {
            "description": "Dataset processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Dataset ds_123 successfully added and recommendations updated."
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing token",
            "content": {
                "application/json": {
                    "examples": {"Invalid Token": errors_data.get("401")}
                }
            },
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "examples": {"Access Denied": errors_data.get("403")}
                }
            },
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": {
                        "Missing Entity ID": errors_data.get("422_missing_entity_id")
                    }
                }
            },
        },
        404: {
            "description": "Not Found - Dataset not found in external source",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found in external repository."}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {"System Failure": errors_data.get("500")}
                }
            },
        },
    },
)
async def add_dataset(
    entity_id: str = Query(..., description="The dataset identifier to add."),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
    token: str = Depends(security.oauth2_scheme),
):
    user_subject = claims.get("sub")
    log = logger.bind(item_id=entity_id, UserId=user_subject)

    accounting_logger.info(
        "Dataset Addition Request Received",
        UserId=user_subject,
        Action="add_dataset",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        DatasetId=entity_id,
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    entity_id = entity_id.strip()
    if not entity_id:
        log.warning("Missing entity_id.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dataset ID is required.",
        )
    
    # It is more efficient to reuse the global recs_client or define these outside 
    # the request scope, but keeping local as per your original snippet logic.
    embedding_client = EmbeddingClient()
    
    try:
        moma = MomaDataset(user_token=token)
        # Assuming get_from_external might raise an error if not found
        moma.get_from_external(entity_id)
        profile = moma.to_dataset_profile()

        was_added = await process_incremental_update(
            profile, 
            application=DEFAULT_APPLICATION, 
            recs_client=recs_client, 
            emb_client=embedding_client
        )

        if not was_added:
            return {
                "status": "ignored",
                "message": f"Dataset {entity_id} is already in the system.",
            }

        return {
            "status": "success",
            "message": f"Dataset {entity_id} successfully added and recommendations updated.",
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error adding dataset {entity_id}: {e}", exc_info=True)
        # Providing a cleaner error message for the user while logging the full exception
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while adding the dataset."
        )

@router.post(
    "/dataset/remove",
    summary="Remove a dataset",
    description="""
Removes a dataset from the DataGEMS recommender.

The service deletes the dataset embedding, removes its recommendation list, removes the dataset from the recommendation index, and cleans references to it from other recommendation lists.
    """,
    responses={
        200: {
            "description": "Dataset removed successfully",
            "content": {
                "application/json": {
                    "example": {"status": "success", "message": "Dataset ds_123 removed."}
                }
            },
        },
        404: {
            "description": "Not Found",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset not found in the recommendation engine."}
                }
            },
        },
        500: {"description": "Internal Server Error"}
    }
)
async def remove_dataset(
    entity_id: str = Query(..., description="The unique identifier of the dataset to be removed."),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
):
    user_subject = claims.get("sub")
    log = logger.bind(item_id=entity_id, UserId=user_subject)

    accounting_logger.info(
        "Dataset Removal Request Received",
        UserId=user_subject,
        Action="remove_dataset",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        DatasetId=entity_id,
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )

    # Validate input
    entity_id = entity_id.strip()
    if not entity_id:
        raise HTTPException(status_code=422, detail="Entity ID is required.")

    try:
        from dataset_recsys.workflows.dataset_removal import dataset_removal
        
        # Dependency check: ensuring we have our clients ready
        embedding_client = EmbeddingClient()
        
        was_removed = await dataset_removal(
            entity_id=entity_id, 
            application=DEFAULT_APPLICATION,
            recs_client=recs_client, 
            emb_client=embedding_client
        )

        if not was_removed:
            log.warning("Removal failed: dataset does not exist.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {entity_id} not found in the system."
            )

        accounting_logger.info(
            "Dataset Removal Processed",
            UserId=user_subject,
            Action="remove_dataset",
            DatasetId=entity_id,
            Timestamp=datetime.utcnow().isoformat() + "Z",
        )

        return {
            "status": "success",
            "message": f"Dataset {entity_id} removed from the system.",
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error during removal of {entity_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during removal.")

@router.post(
    "/dataset/exist",
    summary="Check dataset existence",
    description="""
Checks whether one or more datasets are currently registered in the DataGEMS recommendation index.

The endpoint receives a list of dataset IDs and returns a mapping from each ID to a boolean value indicating whether it exists in the recommender.
    """,
    responses={
        200: {
            "description": "Existence check successful",
            "content": {
                "application/json": {
                    "example": {"ds_123": True, "ds_456": False}
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing token",
            "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {"application/json": {"example": {"detail": "Access denied"}}},
        },
        422: {
            "description": "Validation Error - Missing or empty list",
            "content": {"application/json": {"example": {"detail": "List of entity_ids cannot be empty."}}},
        },
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal system failure"}}},
        },
    },
)
async def check_existence(
    entity_ids: List[str] = Body(
        ..., 
        description="A list of dataset IDs to verify.",
        example=["ds_123", "ds_456"]
    ),
    claims: dict = Depends(security.require_role(["user", "dg_user", "dg_system"])),
):
    user_subject = claims.get("sub")
    log = logger.bind(UserId=user_subject, DatasetCount=len(entity_ids))
    accounting_logger.info(
        "Existence Check Request Received",
        UserId=user_subject,
        Action="check_existence",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        DatasetCount=len(entity_ids),
        Timestamp=datetime.utcnow().isoformat() + "Z",
    )
    
    # Basic validation
    if not entity_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="List of entity_ids cannot be empty."
        )

    # Perform batch check
    existence_map = recs_client.check_existence_batch(DEFAULT_APPLICATION, entity_ids)
    
    return existence_map
