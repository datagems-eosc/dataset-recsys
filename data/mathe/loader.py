from typing import List, Dict, Optional
from pathlib import Path
import json

import pandas as pd
from huggingface_hub import snapshot_download

HUGGINGFACE_REPO_ID = "DARELab/cross-dataset-assets"

class MathE:
    """
    Usage:
        from mathe import MathE
        mathe = MathE()
        data = mathe.get()
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir: Optional[Path] = base_dir
        self.data = None

    def _init_data(self) -> None:
        if self._base_dir is None:
            local_dir = snapshot_download(
                repo_id=HUGGINGFACE_REPO_ID,
                repo_type="dataset",
                local_dir_use_symlinks=False,
                allow_patterns=["mathe/**"], # downloads PDFs + OCR + indexes
            )
            print("Assets stored under:", local_dir)
            self._base_dir = Path(local_dir) / "mathe"

        ocr_path = self._base_dir / "data.json"
        with open(ocr_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Keep only entries whose PDF filename is numeric (e.g. '962.pdf')
        self.data = [
            entry
            for entry in self.data
            if (
                (name := Path(entry["id"]).name).lower().endswith(".pdf")
                and name[:-4].isnumeric()
            )
        ]

        print(len(self.data), "numeric materials found")

    def get_info(self) -> Dict[str, str]:
        """
        Returns high-level information about MathE OCR materials.
        """
        return {
            "name": "MathE",
            "description": (
                "MathE is a collection of higher-education mathematics materials originating from "
                "the MathE platform, an adaptive and open-access e-learning environment designed to enrich the "
                "mathematical learning experiences of students and lecturers in higher education."
                "Each material is provided as a PDF plus OCR-extracted text, "
                "intended to support content-based educational recommendations."
            ),
            "source": "DARELab/cross-dataset-assets (Hugging Face dataset, 'mathe' folder)",
            "formats": ["pdf", "json"],
            "dataset_folder": str(self._base_dir) if self._base_dir is not None else "NOT_YET_LOADED",
        }

    def get(self) -> pd.DataFrame:
        """
        Returns the main MathE OCR materials table as a DataFrame.

        Returns:
            pd.DataFrame with columns:
                - id:          relative path to PDF (e.g., 'materials/56.pdf')
                - contents:    OCR text
                - material_id: file name (e.g., '56.pdf')
                - pdf_path:    absolute local path to the PDF
        """
        if self.data is None:
            self._init_data()
        df = pd.DataFrame(self.data)
        df["material_id"] = df["id"].apply(lambda p: Path(p).name)
        df["pdf_path"] = df["id"].apply(lambda p: str(self._base_dir / p))
        df = df.replace("", pd.NA)
        return df

    def get_raw(self) -> List[Dict]:
        """
        Returns the raw JSON list as loaded from data.json.
        """
        if self.data is None:
            self._init_data()
        return list(self.data)