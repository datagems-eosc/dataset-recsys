from __future__ import annotations

import re
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin


class BaseTextPreprocessor(BaseEstimator, TransformerMixin):
    """Reusable base preprocessor compatible with sklearn pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [self.clean(text) for text in X]

    def clean(self, text: str) -> str:
        raise NotImplementedError


class LightTextPreprocessor(BaseTextPreprocessor):
    """
    Lightweight cleaner for enriched natural-language text used in retrieval
    and embedding pipelines.

    Design goals:
    - remove markdown / formatting artifacts introduced by LLM enrichment
    """

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = str(text)

        # Remove full markdown heading lines (e.g., "## Dataset Summary")
        # Example: "## Dataset Summary" -> removed entirely
        text = re.sub(r"^\s*#+\s+.*$", "", text, flags=re.MULTILINE)

        # Remove markdown emphasis while preserving the content
        # Examples:
        # "**important dataset**" -> "important dataset"
        # "*keyword*" -> "keyword"
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        # Replace markdown links with their anchor text
        # Example: "[dataset page](http://example.com)" -> "dataset page"
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Flatten bullet-like formatting while keeping the content
        # Example:
        # "- first item" -> "first item"
        # "• second item" -> "second item"
        text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)

        # Normalize whitespace
        # Example: "This   is\n\ntext" -> "This is text"
        text = re.sub(r"\s+", " ", text).strip()

        return text

def build_embedding_text(profile: dict, cleaner: BaseTextPreprocessor | None = None) -> str:
    """Build a single cleaned text block from a dataset profile for embedding."""
    title = profile.get("title").strip()
    description = profile.get("enriched_description").strip()

    if cleaner is not None:
        description = cleaner.clean(description)

    if title and description.lower().startswith(title.lower()):
        return description

    return f"{title}. {description}"