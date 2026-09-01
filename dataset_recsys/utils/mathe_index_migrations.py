"""Transitional compatibility for MathE's legacy mixed recommendation index.

Delete this module, its import, and the ``include_legacy_mathe_collection`` call
from ``mathe_sync_pipeline.py`` after every consumer reads from the permanent
``mathe_documents`` or ``mathe_videos`` application namespace.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from dataset_recsys.mathe_recommenders.constants import MatheApplication


_YOUTUBE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def include_legacy_mathe_collection(
    collection_indices: dict[MatheApplication, dict[str, int]],
    materials: Sequence[Mapping[str, object]],
) -> dict[MatheApplication, dict[str, int]]:
    """Prepend the old mixed ``mathe`` collection to the split collections."""
    legacy_indices = {
        _normalize_legacy_index_id(material["sync_entry_id"]): index
        for index, material in enumerate(materials)
    }
    return {
        MatheApplication.LEGACY: legacy_indices,
        **collection_indices,
    }


def _normalize_legacy_index_id(sync_entry_id: object) -> str:
    """Convert a processing ID to the format expected by the old index."""
    clean_id = Path(str(sync_entry_id or "").strip()).name
    if _YOUTUBE_ID_PATTERN.fullmatch(clean_id):
        return f"{clean_id}.txt"
    return clean_id


__all__ = ["include_legacy_mathe_collection"]
