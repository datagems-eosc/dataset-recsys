from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, TypedDict

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.retrieval import rank_similar_entities
from dataset_recsys.utils.mathe_index_migrations import (
    include_legacy_mathe_collection,
)
from dataset_recsys.utils.mathe_syncer import MathE_Syncer
from dataset_recsys.embeddings import encode_texts
import numpy as np

logger = logging.getLogger(__name__)

# TODO: Remove this compatibility summary value together with legacy index
# publication after every consumer uses a split application namespace.
MATHE_APPLICATION = MatheApplication.LEGACY
MATHE_SPLIT_APPLICATIONS = (
    MatheApplication.DOCUMENTS,
    MatheApplication.VIDEOS,
)
DEFAULT_MATHE_EMBEDDING_MODEL = os.getenv(
    "MATHE_EMBEDDING_MODEL",
    "BAAI/bge-m3",
)


class CompletedMaterial(TypedDict):
    """Completed sync entry used to build the MathE indexes."""
    sync_entry_id: str
    platform_material_id: str | None  # None only for legacy unresolved rows
    type: str
    text: str


def _get_completed_materials_from_db(
    syncer: MathE_Syncer,
) -> list[CompletedMaterial]:
    """Queries completed OCR and transcript materials directly from the SQLite database."""
    query = """
        SELECT id, type, platform_material_id, claude_ocr_text
        FROM sync_entries
        WHERE status = 'completed'
          AND type IN ('document', 'video')
          AND claude_ocr_text IS NOT NULL
          AND claude_ocr_text != ''
        ORDER BY type, id
    """
    materials: list[CompletedMaterial] = []

    try:
        with syncer._get_sqlite_conn() as conn:
            rows = conn.execute(query).fetchall()
            
        for row in rows:
            sync_entry_id = str(row["id"] or "").strip()
            material_type = str(row["type"]).strip().lower()
            platform_material_id = (
                str(row["platform_material_id"] or "").strip() or None
            )
            ocr_text = str(row["claude_ocr_text"]).strip()

            if not sync_entry_id or not ocr_text:
                continue

            materials.append(
                CompletedMaterial(
                    sync_entry_id=sync_entry_id,
                    platform_material_id=platform_material_id,
                    type=material_type,
                    text=ocr_text,
                )
            )

    except Exception:
        logger.exception("Failed to load completed materials from SQLite database.")
        
    return materials


def _build_collection_indices(
    materials: list[CompletedMaterial],
) -> dict[MatheApplication, dict[str, int]]:
    """Map each permanent split-collection ID to its shared embedding row.
    Documents are stored and compared only with documents.
    Videos are stored and compared only with videos."""
    collection_indices: dict[MatheApplication, dict[str, int]] = {
        application: {} for application in MATHE_SPLIT_APPLICATIONS
    }

    for index, material in enumerate(materials):
        platform_material_id = material.get("platform_material_id")
        # TODO: After the legacy index migration, require platform_material_id in
        # the loader, change CompletedMaterial.platform_material_id to str, and
        # remove this missing-ID compatibility branch.
        if not platform_material_id:
            logger.warning(
                "Skipping %s from split MathE indexes because platform_material_id is missing",
                material["sync_entry_id"],
            )
            continue

        application = (
            MatheApplication.DOCUMENTS
            if material["type"] == "document"
            else MatheApplication.VIDEOS
        )
        if platform_material_id in collection_indices[application]:
            logger.warning(
                "Duplicate MathE platform material %s in %s; keeping the last completed row",
                platform_material_id,
                application,
            )
        collection_indices[application][platform_material_id] = index

    return collection_indices


def _replace_embedding_and_recommendation_collection(
    application: MatheApplication,
    entity_indices: dict[str, int],
    materials: list[CompletedMaterial],
    embeddings: np.ndarray,
    embedding_client: EmbeddingClient,
    recommendation_client: RecommendationClient,
    run_id: str,
) -> dict[str, int]:
    """Replace one pgvector and Redis collection without crossing content types."""
    entity_ids = list(entity_indices)
    if not entity_ids:
        embedding_client.delete_application(
            application,
            table=embedding_client.TABLE_MATHE,
        )
        recommendation_client.store_recommendations(application, {})
        return {
            "processed_materials": 0,
            "embeddings_stored": 0,
            "redis_keys_updated": 0,
        }

    indices = list(entity_indices.values())
    collection_embeddings = embeddings[indices]
    embedding_inputs = [materials[index]["text"] for index in indices]
    embeddings_stored = embedding_client.store_embeddings(
        application=application,
        dataset_ids=entity_ids,
        embeddings=collection_embeddings,
        embedding_inputs=embedding_inputs,
        embedding_model=DEFAULT_MATHE_EMBEDDING_MODEL,
        table=embedding_client.TABLE_MATHE,
        run_id=run_id,
    )
    recommendations = rank_similar_entities(entity_ids, collection_embeddings)
    redis_keys_updated = recommendation_client.store_recommendations(
        application=application,
        data=recommendations,
    )
    return {
        "processed_materials": len(entity_ids),
        "embeddings_stored": embeddings_stored,
        "redis_keys_updated": redis_keys_updated,
    }

def _build_recommendation_client():
    return RecommendationClient()


def _build_embedding_client():
    return EmbeddingClient()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0 0%"

    filled = round(width * current / total)
    percent = round(100 * current / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


def _save_sync_status(syncer: MathE_Syncer, **updates: Any) -> dict[str, Any]:
    current_status = {}
    if hasattr(syncer, "get_sync_status"):
        try:
            current_status = syncer.get_sync_status()
        except Exception:
            logger.exception("Could not read MathE sync status")

    current_status.update(updates)

    if hasattr(syncer, "save_sync_status"):
        syncer.save_sync_status(current_status)

    return current_status


def run_mathe_pipeline(syncer: MathE_Syncer) -> dict:
    """Run MathE sync/OCR, rebuild recommendations, and publish them to Redis."""
    logger.info("Starting MathE online refresh pipeline")
    started_at = _utc_now()
    _save_sync_status(
        syncer,
        sync_status="running",
        last_sync_started_at=started_at,
        last_sync_heartbeat_at=started_at,
        last_sync_completed_at=None,
        embeddings_created=0,
    )

    try:
        syncer.sync_and_process()
        _save_sync_status(syncer, last_sync_heartbeat_at=_utc_now())
        logger.info("MathE sync and OCR complete, loading data for recommendation refresh")

        materials = _get_completed_materials_from_db(syncer)
        logger.info("Loaded %d completed materials from SQLite", len(materials))

        if not materials:
            logger.warning("No completed MathE materials with OCR text found")
            completed_at = _utc_now()
            _save_sync_status(
                syncer,
                sync_status="skipped",
                last_sync_heartbeat_at=completed_at,
                last_sync_completed_at=completed_at,
                embeddings_created=0,
            )
            return {
                "status": "skipped",
                "processed_materials": 0,
                "redis_keys_updated": 0,
                "reason": "No completed materials with OCR text found.",
            }

        texts = [material["text"] for material in materials]

        logger.info("Generating MathE OCR text embeddings for %d materials", len(materials))

        batch_size = 32
        all_embeddings = []
        embeddings_created = 0
        embedding_client = _build_embedding_client()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_number = i // batch_size + 1
            print(f"Encoding batch {batch_number} with {len(batch)} materials...")
            batch_encodings = encode_texts(batch, model_name=DEFAULT_MATHE_EMBEDDING_MODEL)
            all_embeddings.append(batch_encodings)
            embeddings_created += len(batch_encodings)
            _save_sync_status(
                syncer,
                embeddings_created=embeddings_created,
                last_sync_heartbeat_at=_utc_now(),
            )
            print(f"Embedding progress {_progress_bar(embeddings_created, len(texts))}")
            logger.info(
                "MathE embedding progress %s batch=%s",
                _progress_bar(embeddings_created, len(texts)),
                batch_number,
            )

        embeddings = np.vstack(all_embeddings)
        collection_indices = _build_collection_indices(materials)
        collection_indices = include_legacy_mathe_collection(
            collection_indices,
            materials,
        )
        recommendation_client = _build_recommendation_client()
        collection_summaries = {}
        for application, entity_indices in collection_indices.items():
            logger.info(
                "Refreshing MathE embedding and Redis collection %s with %d materials",
                application,
                len(entity_indices),
            )
            collection_summaries[str(application)] = (
                _replace_embedding_and_recommendation_collection(
                    application=application,
                    entity_indices=entity_indices,
                    materials=materials,
                    embeddings=embeddings,
                    embedding_client=embedding_client,
                    recommendation_client=recommendation_client,
                    run_id=started_at,
                )
            )

        redis_keys_updated = sum(
            collection["redis_keys_updated"]
            for collection in collection_summaries.values()
        )
        completed_at = _utc_now()
        _save_sync_status(
            syncer,
            sync_status="completed",
            last_sync_heartbeat_at=completed_at,
            last_sync_completed_at=completed_at,
            embeddings_created=embeddings_created,
        )

        summary = {
            "status": "completed",
            "processed_materials": len(materials),
            "embeddings_created": embeddings_created,
            "redis_keys_updated": redis_keys_updated,
            "application": MATHE_APPLICATION,
            "collections": collection_summaries,
        }
        logger.info("Finished MathE online refresh pipeline: %s", summary)
        return summary
    except Exception:
        failed_at = _utc_now()
        _save_sync_status(
            syncer,
            sync_status="failed",
            last_sync_heartbeat_at=failed_at,
            last_sync_completed_at=failed_at,
        )
        logger.exception("MathE online refresh pipeline failed")
        raise


__all__ = ["run_mathe_pipeline"]
