import signal
import subprocess
import boto3
import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict, Optional

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
logger = logging.getLogger(__name__)


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0 0%"

    filled = round(width * current / total)
    percent = round(100 * current / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


def _has_completed_ocr(entry: Dict[str, Any]) -> bool:
    ocr_text = str(entry.get("claude_ocr_text") or "")
    return entry.get("status") == "completed" or (
        bool(ocr_text) and not ocr_text.startswith("OCR Failed")
    )

class MathE_Syncer:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._pdf_dir = self._base_dir / "pdfs"
        self._docx_dir = self._base_dir / "docxs"
        self._ppt_dir = self._base_dir / "ppts"
        self.json_file = self._base_dir / "data.json"
        self.status_file = self._base_dir / "sync_status.json"
        self.data: Optional[List[Dict]] = None
        
        # Claude 4.5 Global Configuration
        self.model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.region = "eu-central-1"
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.region, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        
        # Graceful shutdown handling
        self.is_running = False  # Add this line        
        self.keep_running = True
        signal.signal(signal.SIGTERM, self._handle_exit)                

    def _handle_exit(self, signum, frame):
        print("Received SIGTERM, finishing current file...")
        self.keep_running = False

    def _libreoffice_convert(self, file_path: Path, output_dir: Path) -> bool:
        """
        Internal helper to execute the LibreOffice headless conversion command.
        """
        unique_profile = f"file:///tmp/libo_profile_{file_path.stem}"

        cmd = [
            'libreoffice',
            '--headless',
            '-env:UserInstallation=' + unique_profile, 
            '--convert-to', 'pdf',
            '--outdir', str(output_dir),
            str(file_path)
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0

    def _init_data(self) -> None:
        """Loads data.json and performs discovery across designated raw inputs."""
        if self.json_file.exists():
            with open(self.json_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = []

        existing_ids = {entry["id"] for entry in self.data}
        new_found = False

        if not self._base_dir.exists():
            print(f"MathE base directory does not exist: {self._base_dir}")
            return

        tmp_build_dir = Path("/tmp/libo_out")
        tmp_build_dir.mkdir(parents=True, exist_ok=True)

        # 1. Preprocess Word documents -> Store location context as local /tmp
        if self._docx_dir.exists():
            for f in self._docx_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".docx" and f.stem.isnumeric():
                    original_id = f.name  # e.g., "3.docx"
                    
                    if original_id not in existing_ids:
                        local_pdf_target = tmp_build_dir / f"{f.stem}_docx.pdf"
                        
                        if not local_pdf_target.exists():
                            print(f"Converting DOCX to local memory sandbox: {f.name}")
                            if self._libreoffice_convert(f, tmp_build_dir):
                                standard_out = tmp_build_dir / f"{f.stem}.pdf"
                                try:
                                    standard_out.rename(local_pdf_target)
                                except Exception as e:
                                    print(f"❌ Failed to rename local PDF for {f.name}: {e}")
                                    continue

                        print(f"Queueing converted DOCX target: {original_id}")
                        self.data.append({
                            "id": original_id,
                            "internal_pdf_path": str(local_pdf_target),
                            "claude_ocr_text": None,
                            "status": "pending"
                        })
                        new_found = True

        # 2. Preprocess PowerPoint presentations -> Store location context as local /tmp
        if self._ppt_dir.exists():
            for f in self._ppt_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".pptx" and f.stem.isnumeric():
                    original_id = f.name  # e.g., "3.pptx"
                    
                    if original_id not in existing_ids:
                        local_pdf_target = tmp_build_dir / f"{f.stem}_pptx.pdf"
                        
                        if not local_pdf_target.exists():
                            print(f"Converting PPTX to local memory sandbox: {f.name}")
                            if self._libreoffice_convert(f, tmp_build_dir):
                                standard_out = tmp_build_dir / f"{f.stem}.pdf"
                                try:
                                    standard_out.rename(local_pdf_target)
                                except Exception as e:
                                    print(f"❌ Failed to rename local PDF for {f.name}: {e}")
                                    continue

                        print(f"Queueing converted PPTX target: {original_id}")
                        self.data.append({
                            "id": original_id,
                            "internal_pdf_path": str(local_pdf_target),
                            "claude_ocr_text": None,
                            "status": "pending"
                        })
                        new_found = True

        # 3. Discover native pre-existing material PDFs directly from server directory
        if self._pdf_dir.exists():
            for f in self._pdf_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".pdf" and f.stem.isnumeric():
                    original_id = f.name  # e.g., "3.pdf"
                    if original_id not in existing_ids:
                        print(f"Discovered native production target: {original_id}")
                        self.data.append({
                            "id": original_id,
                            "internal_pdf_path": str(f),  # Reads directly from source pdf dir
                            "claude_ocr_text": None,
                            "status": "pending"
                        })
                        new_found = True
        
        if new_found:
            self._save_state()

    def _save_state(self) -> None:
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def get_sync_status(self) -> Dict[str, Any]:
        if not self.status_file.exists():
            return {
                "sync_status": "never_run",
                "last_sync_started_at": None,
                "last_sync_completed_at": None,
            }

        with open(self.status_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_sync_status(self, status: Dict[str, Any]) -> None:
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4)

    # --- Data Access Methods ---

    def get(self) -> pd.DataFrame:
        """Returns the main table as a DataFrame."""
        if self.data is None:
            self._init_data()

        if not self.data:
            return pd.DataFrame(
                columns=["id", "source_type", "claude_ocr_text", "status", "material_id", "pdf_path"]
            )
            
        df = pd.DataFrame(self.data)
        df["material_id"] = df["id"]
        
        # Extract source_type directly from the clean file extension (suffix)
        #    '.docx' -> 'docx', '.pptx' -> 'pptx', '.pdf' -> 'pdf'
        df["source_type"] = df["id"].apply(lambda p: Path(p).suffix.lstrip('.').lower())
        
        # Use your tracking 'internal_pdf_path' instead of assuming its location relative to id
        df["pdf_path"] = df["internal_pdf_path"]
        
        return df.replace("", pd.NA)

    def get_raw(self) -> List[Dict]:
        """Returns the raw JSON list."""
        if self.data is None:
            self._init_data()
        return list(self.data)

    def count_available_pdfs(self) -> int:
        """
        Counts tracked files that have a valid converted or native PDF living on disk.
        """
        if self.data is None:
            self._init_data()
            
        # Iterate over our known tracking catalog paths to verify actual existence 
        # (This accommodates both the local container /tmp and server pdf_dir files securely)
        valid_count = 0
        for entry in self.data:
            path_str = entry.get("internal_pdf_path")
            if path_str and Path(path_str).exists():
                valid_count += 1
                
        return valid_count

    def get_info(self) -> Dict[str, str]:
        """Returns high-level info."""
        return {
            "name": "MathE",
            "source": "Mounted Filesystem/S3",
            "dataset_folder": str(self._base_dir),
        }

    def sync_and_process(self, limit: Optional[int] = None):
        if self.is_running:
            print("Sync job is already in progress.")
            return
        
        self.is_running = True
        try:
            print("Starting sync/process lifecycle...")
            self._init_data()
            print(f"Discovered {len(self.data)} total entries, with {self.count_available_pdfs()} available PDFs.")
            # limit = 1 # For testing, process only 1 file at a time. Remove or adjust this for full batch processing.
            self.run_batch_ocr(limit=limit)
            print("Lifecycle complete.")
        except Exception as e:
            print(f"Error during sync/process lifecycle: {e}")
        finally:
            self.is_running = False  # Ensure it always releases the lock

    # --- OCR Logic ---

    def run_batch_ocr(self, limit: Optional[int] = None):
        if self.data is None:
            self._init_data()
        
        pending_entries = [
            entry
            for entry in self.data
            if not _has_completed_ocr(entry) and entry.get("status") != "failed"
        ]
        if limit is not None:
            pending_entries = pending_entries[:limit]

        skipped = len(self.data) - len(pending_entries)
        logger.info(
            "Starting MathE OCR batch: %s pending, %s already completed/failed",
            len(pending_entries),
            skipped,
        )
        print(
            f"Starting batch OCR process: {len(pending_entries)} pending, "
            f"{skipped} already completed/failed."
        )
        processed = 0
        for entry in pending_entries:
            if not self.keep_running:
                print("Shutdown signaled. Saving and exiting.")
                break
            
            print(
                f"OCR progress {_progress_bar(processed + 1, len(pending_entries))} "
                f"processing {entry['id']}"
            )
            print(f"Current status: {entry.get('status')}, OCR text length: {len(str(entry.get('claude_ocr_text') or ''))}")
                
            full_path = Path(entry["internal_pdf_path"])
            try:
                entry["claude_ocr_text"] = self._perform_claude_call(full_path)
                entry["status"] = "completed"
            except Exception as e:
                print(f"Error processing {entry['id']}: {e}")
                entry["claude_ocr_text"] = f"OCR Failed: {str(e)}"
                entry["status"] = "failed"
            
            print(f"Finished processing {entry['id']}. Status: {entry['status']}, OCR text length: {len(str(entry.get('claude_ocr_text') or ''))}")
            logger.info(
                "MathE OCR progress %s material=%s status=%s",
                _progress_bar(processed + 1, len(pending_entries)),
                Path(entry["id"]).name,
                entry["status"],
            )
            self._save_state()
            processed += 1

        if not pending_entries:
            logger.info("MathE OCR progress %s; no new OCR work needed", _progress_bar(0, 0))

        if not self.data or all(e['status'] in ['completed', 'failed'] for e in self.data):
            self._send_notification("OCR process finished for all files.")
        
    def _send_notification(self, msg: str):
        # Placeholder for notification logic (e.g., email, Slack)
        print(f"NOTIFICATION: {msg}")

    def _perform_claude_call(self, p: Path) -> str:
        # (Same logic as previous step)
        with open(p, "rb") as f:
            pdf_bytes = f.read()

        response = self.bedrock.converse(
            modelId=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"document": {"name": p.stem, "format": "pdf", "source": {"bytes": pdf_bytes}}},
                    {"text": "Extract all text from this math document. Use LaTeX for equations. Follow the native language of the document. Do not add any commentary or explanations, just return the raw extracted text."}
                ]
            }],
            inferenceConfig={"temperature": 0.0}
        )
        return response['output']['message']['content'][0]['text']
