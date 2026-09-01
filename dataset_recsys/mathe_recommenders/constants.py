"""Controlled identifiers used by the MathE recommendation domain."""

from enum import StrEnum


class MatheApplication(StrEnum):
    """Logical pgvector and Redis application namespaces."""

    LEGACY = "mathe"
    DOCUMENTS = "mathe_documents"
    VIDEOS = "mathe_videos"


VIDEO_TYPE_TO_SUBTYPE = {
    1: "video_lesson",
    2: "video_review",
}


__all__ = [
    "MatheApplication",
    "VIDEO_TYPE_TO_SUBTYPE",
]
