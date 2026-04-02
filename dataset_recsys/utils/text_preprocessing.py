from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile


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

        # Remove bare brackets but keep their contents
        # Example: "[dataset]" -> "dataset"
        text = re.sub(r"\[(.*?)\]", r"\1", text)

        # Flatten bullet-like formatting while keeping the content
        # Example:
        # "- first item" -> "first item"
        # "• second item" -> "second item"
        text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)

        # Normalize whitespace
        # Example: "This   is\n\ntext" -> "This is text"
        text = re.sub(r"\s+", " ", text).strip()

        return text


def get_dataset_profile_fields() -> set[str]:
    """Return the available DatasetProfile dataclass field names."""
    return set(DatasetProfile.__dataclass_fields__.keys())


def preprocess_profiles_field(
    profiles: list[DatasetProfile],
    field_name: str,
    cleaner: BaseTextPreprocessor | None = None,
) -> list[DatasetProfile]:
    cleaner = cleaner or LightTextPreprocessor()
    available_fields = get_dataset_profile_fields()

    if field_name not in available_fields:
        raise AttributeError(
            f"DatasetProfile has no field '{field_name}'. Available fields: {sorted(available_fields)}"
        )

    processed_profiles: list[DatasetProfile] = []
    for data_profile in profiles:
        processed_profiles.append(
            replace(
                data_profile,
                **{field_name: cleaner.clean(getattr(data_profile, field_name))},
            )
        )

    return processed_profiles


def preprocess_catalog(
    catalog: list[DatasetProfile],
    cleaner: BaseTextPreprocessor | None = None,
) -> list[DatasetProfile]:
    """Clean the catalog summaries in the dataset profiles using the provided cleaner."""
    cleaner = cleaner or LightTextPreprocessor()

    processed_catalog: list[DatasetProfile] = []
    for data_profile in catalog:
        processed_catalog.append(
            replace(
                data_profile,
                catalog_summary=cleaner.clean(data_profile.catalog_summary),
            )
        )

    return processed_catalog


__all__ = [
    "BaseTextPreprocessor",
    "LightTextPreprocessor",
    "preprocess_catalog",
    "preprocess_profiles_field",
]