from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dataset_recsys.storage.recommendation_client import RecommendationClient
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


def _delete_ocr_data_file(json_file: Path) -> bool:
    if not json_file.exists():
        return False

    json_file.unlink()
    logger.info("Deleted MathE OCR data file after Redis refresh: %s", json_file)
    return True


def run_mathe_pipeline(syncer: MathE_Syncer) -> dict:
    """Run MathE sync/OCR, rebuild recommendations, and publish them to Redis."""
    logger.info("Starting MathE online refresh pipeline")
    syncer.sync_and_process()
    logger.info("MathE sync and OCR complete, loading data for recommendation refresh")

    entries = _load_mathe_data(syncer.json_file)
    logger.info("Loaded %d entries from MathE OCR data file", len(entries))
    materials = _completed_materials(entries)
    if not materials:
        logger.warning("No completed MathE materials with OCR text found")
        return {
            "status": "skipped",
            "processed_materials": 0,
            "redis_keys_updated": 0,
            "reason": "No completed materials with OCR text found.",
        }

    material_ids = [material["id"] for material in materials]
    texts = [material["text"] for material in materials]

    logger.info("Generating MathE OCR text embeddings for %d materials", len(materials))

    # Conceptual fix inside run_mathe_pipeline
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_encodings = encode_texts(batch, model_name=DEFAULT_MATHE_EMBEDDING_MODEL)
        all_embeddings.append(batch_encodings)

    # Combine results
    embeddings = np.vstack(all_embeddings)
    # embeddings = encode_texts(texts, model_name=DEFAULT_MATHE_EMBEDDING_MODEL)

    logger.info("Computing full MathE nearest-neighbor recommendations")
    recommendations = rank_similar_entities(material_ids, embeddings)

    logger.info("Storing MathE recommendations in Redis")
    logger.info("Recommendations data: %s", recommendations)
    recs_client = _build_recommendation_client()
    redis_keys_updated = recs_client.store_recommendations(
        application=MATHE_APPLICATION,
        data=recommendations,
    )
    # ocr_data_deleted = _delete_ocr_data_file(syncer.json_file)

    summary = {
        "status": "completed",
        "processed_materials": len(materials),
        "redis_keys_updated": redis_keys_updated,
        "application": MATHE_APPLICATION,
        # "ocr_data_deleted": ocr_data_deleted,
    }
    logger.info("Finished MathE online refresh pipeline: %s", summary)
    return summary


__all__ = ["MATHE_APPLICATION", "run_mathe_pipeline"]
