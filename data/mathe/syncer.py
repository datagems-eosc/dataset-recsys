from typing import List, Dict, Optional
from pathlib import Path
import json
import pandas as pd

class MathE_Syncer:
    """
    Usage:
        # In production, point to the mounted S3 volume path
        mathe = MathE_Syncer(base_dir=Path("/your/path/to/data"))
        data = mathe.get()
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir: Path = base_dir
        self.data: Optional[List[Dict]] = None

    def _init_data(self) -> None:
        """Loads data from the specified directory."""
        ocr_path = self._base_dir / "data.json"
        
        if not ocr_path.exists():
            raise FileNotFoundError(f"data.json not found at {ocr_path}. Check your path or mount configuration.")

        with open(ocr_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Filtering logic: keep only entries where the PDF filename is numeric
        self.data = [
            entry
            for entry in self.data
            if (
                (name := Path(entry["id"]).name).lower().endswith(".pdf")
                and name[:-4].isnumeric()
            )
        ]
        print(f"Successfully loaded {len(self.data)} items from {self._base_dir}")

    def get_info(self) -> Dict[str, str]:
        """Returns high-level information about MathE OCR materials."""
        return {
            "name": "MathE",
            "source": "Mounted Filesystem/S3",
            "dataset_folder": str(self._base_dir),
        }

    def get(self) -> pd.DataFrame:
        """Returns the main MathE OCR materials table as a DataFrame."""
        if self.data is None:
            self._init_data()
            
        df = pd.DataFrame(self.data)
        df["material_id"] = df["id"].apply(lambda p: Path(p).name)
        # Ensure path points to the absolute path within your mounted volume
        df["pdf_path"] = df["id"].apply(lambda p: str(self._base_dir / p))
        return df.replace("", pd.NA)

    def get_raw(self) -> List[Dict]:
        """Returns the raw JSON list as loaded from data.json."""
        if self.data is None:
            self._init_data()
        return list(self.data)