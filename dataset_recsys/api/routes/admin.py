

import json
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from dataset_recsys.api.security import security
from dataset_recsys.ingestion.moma_dataset import MomaDataset
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.workflows.incremental_update import process_incremental_update
from typing import List

logger = structlog.get_logger(__name__)
accounting_logger = structlog.get_logger("accounting")
router = APIRouter(prefix="/dataset-recsys", tags=["Administrative Operations"])
recs_client = RecommendationClient()

DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/valid_examples.json")
DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/error_examples.json")
AP_DOCS_VALID_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_valid_examples.json")
AP_DOCS_ERROR_EXAMPLES_PATH = Path("dataset_recsys/api/api_docs/ap_error_examples.json")
AP_REQUEST_EXAMPLE_PATH = Path("dataset_recsys/api/api_docs/ap_request_example.json")
DEFAULT_APPLICATION = "ds2ds"


def load_json_file(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"File '{path}' does not exist.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load file from {path}: {e}")
        return {}

examples_data, errors_data = (load_json_file(DOCS_VALID_EXAMPLES_PATH), load_json_file(DOCS_ERROR_EXAMPLES_PATH))
ap_examples_data, ap_errors_data = (load_json_file(AP_DOCS_VALID_EXAMPLES_PATH), load_json_file(AP_DOCS_ERROR_EXAMPLES_PATH))
ap_request_example = load_json_file(AP_REQUEST_EXAMPLE_PATH)

@router.post(
    "/dataset/add",
    summary="Add a new dataset to the system",
    description="""
Fetch a dataset from external sources, generate its embeddings, and integrate it into the recommendation engine.
If the dataset already exists, the system will skip the addition to avoid duplicates.
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
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
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
        ItemId=entity_id,
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
    summary="Remove a dataset from the system",
    description="""
Performs a clean, incremental removal of a dataset from the recommendation engine. 

**Workflow:**
1. **Inbound Cleanup**: Scans the Redis index to find every other dataset currently recommending this ID and removes the reference.
2. **Outbound Cleanup**: Deletes the specific recommendation list (ZSET) for this dataset.
3. **Index Cleanup**: Removes the dataset ID from the application's global index.
4. **Vector Cleanup**: Deletes the embedding record from the PostgreSQL vector database.
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
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
):
    user_subject = claims.get("sub")
    log = logger.bind(item_id=entity_id, UserId=user_subject)

    accounting_logger.info(
        "Dataset Removal Request Received",
        UserId=user_subject,
        Action="remove_dataset",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        ItemId=entity_id,
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
            ItemId=entity_id,
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
    summary="Check dataset existence in catalog",
    description="""
Validates the existence of multiple dataset IDs within the application's recommendation index.

**Process:**
1. Receives a list of entity IDs.
2. Executes a batch Redis pipeline (`SISMEMBER`) to check availability for each ID simultaneously.
3. Returns a dictionary mapping each queried ID to a boolean status (`true` if present, `false` otherwise).
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
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
):
    user_subject = claims.get("sub")
    log = logger.bind(UserId=user_subject, ItemIds=entity_ids)
    accounting_logger.info(
        "Existence Check Request Received",
        UserId=user_subject,
        Action="check_existence",
        Resource="dataset2dataset_recommender",
        Domain="datagems",
        ItemIds=entity_ids,
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
