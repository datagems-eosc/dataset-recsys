from typing import List, Dict, Optional
from pathlib import Path
import json
import sqlite3

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

        self.db_path = self._base_dir / "syncer.db"
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}. Run the syncer first.")

    def _get_sqlite_conn(self):
        """Helper to create a read-only or standard connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
            "formats": ["pdf", "sqlite"],
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
        if self.db_path is None:
            self._init_data()

        # Query entries where type is a document, status is completed, and ID is a numeric pdf filename
        query = """
            SELECT id, claude_ocr_text AS contents, internal_pdf_path AS pdf_path
            FROM sync_entries 
            WHERE type = 'document' 
              AND status = 'completed'
        """

        with self._get_sqlite_conn() as conn:
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return pd.DataFrame(columns=["id", "contents", "material_id", "pdf_path"])
            
        def is_valid_numeric_pdf(id_val: str) -> bool:
            p = Path(id_val)
            return p.suffix.lower() == ".pdf" and p.stem.isnumeric()

        df = df[df["id"].apply(is_valid_numeric_pdf)].copy()
        
        # Populate clean fields
        df["material_id"] = df["id"].apply(lambda p: Path(p).name)
        
        # Overwrite relative path logic to ensure absolute paths resolve locally if needed
        # (Fall back to self._base_dir / id if internal_pdf_path isn't absolute)
        def get_absolute_path(row):
            local_p = Path(row["pdf_path"]) if row["pdf_path"] else Path(row["id"])
            if local_p.is_absolute():
                return str(local_p)
            return str(self._base_dir / local_p)

        df["pdf_path"] = df.apply(get_absolute_path, axis=1)
        
        print(len(df), "numeric materials found")
        return df.replace("", pd.NA)

    def get_raw(self) -> List[Dict]:
        """
        Returns the raw SQLite records as loaded from the database.
        """
        if self.db_path is None:
            self._init_data()
            
        with self._get_sqlite_conn() as conn:
            rows = conn.execute("SELECT * FROM sync_entries").fetchall()
            
        raw_list = [dict(row) for row in rows]
        
        # Keep only entries whose PDF filename is numeric
        return [
            entry
            for entry in raw_list
            if (
                (name := Path(entry["id"]).name).lower().endswith(".pdf")
                and name[:-4].isnumeric()
            )
        ]