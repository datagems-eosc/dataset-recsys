from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.retrieval import rank_similar_entities
from dataset_recsys.utils.mathe_syncer import MathE_Syncer
from dataset_recsys.embeddings import encode_texts
import numpy as np

logger = logging.getLogger(__name__)

MATHE_APPLICATION = "mathe"
DEFAULT_MATHE_EMBEDDING_MODEL = os.getenv(
    "MATHE_EMBEDDING_MODEL",
    "BAAI/bge-m3",
)


def _normalize_material_id(material_id: Any) -> str:
    """Normalize MathE ids from data.json to the API-facing PDF filename."""
    return Path(str(material_id).strip()).name


def _load_mathe_data(json_file: Path) -> list[dict[str, Any]]:
    if not json_file.exists():
        logger.warning("MathE OCR data file is missing: %s", json_file)
        return []

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        logger.exception("MathE OCR data file is not valid JSON: %s", json_file)
        return []

    if not payload:
        logger.warning("MathE OCR data file is empty: %s", json_file)
        return []
    if not isinstance(payload, list):
        logger.error(
            "MathE OCR data file has unsupported structure: %s (%s)",
            json_file,
            type(payload).__name__,
        )
        return []

    return payload


def _completed_materials(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    materials: list[dict[str, str]] = []

    for entry in entries:
        material_id = _normalize_material_id(entry.get("id", ""))
        ocr_text = str(entry.get("claude_ocr_text") or "").strip()

        if not material_id or entry.get("status") != "completed" or not ocr_text:
            continue

        materials.append({"id": material_id, "text": ocr_text})

    return materials


def _build_recommendation_client():
    return RecommendationClient()


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


def _delete_ocr_data_file(json_file: Path) -> bool:
    if not json_file.exists():
        return False

    json_file.unlink()
    logger.info("Deleted MathE OCR data file after Redis refresh: %s", json_file)
    return True


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

        entries = _load_mathe_data(syncer.json_file)
        logger.info("Loaded %d entries from MathE OCR data file", len(entries))
        materials = _completed_materials(entries)
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

        material_ids = [material["id"] for material in materials]
        texts = [material["text"] for material in materials]

        logger.info("Generating MathE OCR text embeddings for %d materials", len(materials))

        batch_size = 32
        all_embeddings = []
        embeddings_created = 0
        client = EmbeddingClient()

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
        client.store_embeddings(
            application=MATHE_APPLICATION,
            dataset_ids=material_ids,
            embeddings=embeddings,
            embedding_inputs=texts,
            embedding_model=DEFAULT_MATHE_EMBEDDING_MODEL,
            table=client.TABLE_MATHE,
            run_id=started_at,
        )

        logger.info("Computing full MathE nearest-neighbor recommendations")
        recommendations = rank_similar_entities(material_ids, embeddings)

        logger.info("Storing MathE recommendations in Redis")
        logger.info("Recommendations data: %s", recommendations)
        recs_client = _build_recommendation_client()
        redis_keys_updated = recs_client.store_recommendations(
            application=MATHE_APPLICATION,
            data=recommendations,
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


__all__ = ["MATHE_APPLICATION", "run_mathe_pipeline"]
