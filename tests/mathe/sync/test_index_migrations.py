"""Tests for transitional MathE legacy-index compatibility."""

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.utils.mathe_index_migrations import (
    include_legacy_mathe_collection,
)


def test_include_legacy_collection_normalizes_document_and_video_processing_ids():
    split_indices = {
        MatheApplication.DOCUMENTS: {"6": 0},
        MatheApplication.VIDEOS: {"901": 1},
    }
    materials = [
        {
            "sync_entry_id": "./6.pdf",
            "platform_material_id": "6",
            "type": "document",
            "text": "document text",
        },
        {
            "sync_entry_id": "abcdefghijk",
            "platform_material_id": "901",
            "type": "video",
            "text": "video transcript",
        },
    ]

    collection_indices = include_legacy_mathe_collection(
        split_indices,
        materials,
    )

    assert list(collection_indices) == [
        MatheApplication.LEGACY,
        MatheApplication.DOCUMENTS,
        MatheApplication.VIDEOS,
    ]
    assert collection_indices[MatheApplication.LEGACY] == {
        "6.pdf": 0,
        "abcdefghijk.txt": 1,
    }
    assert collection_indices[MatheApplication.DOCUMENTS] == {"6": 0}
    assert collection_indices[MatheApplication.VIDEOS] == {"901": 1}
