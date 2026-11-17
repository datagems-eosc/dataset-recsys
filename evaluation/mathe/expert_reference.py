from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

REFERENCE_SHEET_NAME = "Reference Set"


def parse_material_ids(value: Any) -> list[str]:
    """Parse comma/semicolon/whitespace-separated MathE material IDs from Excel."""
    if value is None or pd.isna(value):
        return []

    if isinstance(value, int):
        return [str(value)]

    if isinstance(value, float):
        if not math.isfinite(value):
            return []
        if value.is_integer():
            return [str(int(value))]

    text = str(value).strip()
    if not text:
        return []

    if re.fullmatch(r"\d+\.0+", text):
        return [text.split(".", maxsplit=1)[0]]

    material_ids = re.findall(r"\d+", text)
    return list(dict.fromkeys(material_ids))


def _is_yes(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def load_reference_rows(reference_workbook: Path) -> list[dict[str, Any]]:
    reference_df = pd.read_excel(reference_workbook, sheet_name=REFERENCE_SHEET_NAME)
    return reference_df.to_dict(orient="records")


