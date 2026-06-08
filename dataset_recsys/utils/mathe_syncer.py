import signal
import subprocess
import boto3
import json
import logging
import os
import re
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict, Optional
import yt_dlp
from faster_whisper import WhisperModel
import psycopg2

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
logger = logging.getLogger(__name__)


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0 0%"

    filled = round(width * current / total)
    percent = round(100 * current / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


def _has_completed_processing(entry: Dict[str, Any]) -> bool:
    """Verifies if an entry already has successfully parsed text content."""
    text = str(entry.get("claude_ocr_text") or "")
    return entry.get("status") == "completed" or (
        bool(text) and not (text.startswith("OCR Failed") or text.startswith("Transcription Failed"))
    )

class MathE_Syncer:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._pdf_dir = self._base_dir / "pdfs"
        self._docx_dir = self._base_dir / "docxs"
        self._ppt_dir = self._base_dir / "ppts"
        self._transcript_dir = self._base_dir / "transcripts"
        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        self.json_file = self._base_dir / "data.json"
        self.status_file = self._base_dir / "sync_status.json"
        self.cookie_file = self._base_dir / "cookies.txt"
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

    def _get_db_connection(self):
        return psycopg2.connect(
            host=os.getenv("DATAGEMS_POSTGRES_HOST"),
            port=int(os.getenv("DATAGEMS_POSTGRES_PORT", "5432")),
            dbname=os.getenv("DB_DS_NAME"),
            user=os.getenv("DB_DS_USER"),
            password=os.getenv("DB_DS_PASSWORD"),
            options=f"-c search_path={os.getenv('DATAGEMS_POSTGRES_SCHEMA', 'public')}"
        )

    def _is_youtube_asset(self, link_value: str) -> bool:
        """Determines if the raw string input points to a video stream asset."""
        if len(link_value) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', link_value):
            return True
        return "youtube.com" in link_value or "youtu.be" in link_value

    def _extract_video_id(self, link_value: str) -> str:
        if len(link_value) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', link_value):
            return link_value
        id_match = re.search(r'(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})', link_value)
        if id_match:
            return id_match.group(1)
        raise ValueError(f"Could not parse valid YouTube identifier: {link_value}")

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

        # --- ROUTE 1: Discover Video Streams from PostgreSQL ---
        print("Syncing video assets from PostgreSQL platform registry...")
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT link FROM platform_materials WHERE type < 3;")
                    rows = cursor.fetchall()
                    db_links = [row[0] for row in rows if row[0]]
        except Exception as e:
            print(f"Warning: PostgreSQL lookup failed. Relying on local data cache. Exception: {e}")
            db_links = []

        for link in db_links:
            if self._is_youtube_asset(link):
                try:
                    video_id = self._extract_video_id(link)
                    
                    # If a transcript file already exists in the folder, sync state instantly
                    backup_text_file = self._transcript_dir / f"{video_id}.txt"
                    
                    if video_id not in existing_ids:
                        status = "completed" if backup_text_file.exists() else "pending"
                        ocr_text = None
                        
                        if backup_text_file.exists():
                            with open(backup_text_file, "r", encoding="utf-8") as f:
                                ocr_text = f.read()

                        entry = {
                            "id": video_id,
                            "type": "audio",
                            "source_value": link,
                            "claude_ocr_text": ocr_text,
                            "status": status
                        }
                        self.data.append(entry)
                        existing_ids[video_id] = entry
                        new_found = True
                    else:
                        # Self-healing backup check for tracking cache mapping
                        entry = existing_ids[video_id]
                        if entry.get("status") == "pending" and backup_text_file.exists():
                            with open(backup_text_file, "r", encoding="utf-8") as f:
                                entry["claude_ocr_text"] = f.read()
                            entry["status"] = "completed"
                            new_found = True
                except ValueError:
                    continue

        # --- ROUTE 2: Discover Structural Office Documents via Local Directories ---
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
                            "type": "document",
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
                            "type": "document",
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
                            "type": "document",
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
        df["pdf_path"] = df.get("internal_pdf_path", pd.NA)
        
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
            self.run_hybrid_batch_processing(limit=limit)
            print("Lifecycle complete.")
        except Exception as e:
            print(f"Error during sync/process lifecycle: {e}")
        finally:
            self.is_running = False  # Ensure it always releases the lock

    # --- OCR Logic ---

    def run_hybrid_batch_processing(self, limit: Optional[int] = None):
        if self.data is None:
            self._init_data()
        
        pending_entries = [
            entry
            for entry in self.data
            if not _has_completed_processing(entry) and entry.get("status") != "failed"
        ]
        if limit is not None:
            pending_entries = pending_entries[:limit]

        if not pending_entries:
            print("No pending work detected across audio formats or documents.")
            return

        whisper_model = None
        if any(e.get("type") == "audio" for e in pending_entries):
            print("Initializing local Whisper extraction runtime engines...")
            whisper_model = WhisperModel("base", device="cpu", compute_type="float32")

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

            if entry.get("type") == "audio":
                self._process_audio_entry(entry, whisper_model)
            else:
                self._process_document_entry(entry)
            
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

    def _process_audio_entry(self, entry: Dict, whisper_model: WhisperModel):
        video_id = entry["id"]

        backup_text_file = self._transcript_dir / f"{video_id}.txt"
        if backup_text_file.exists():
            print(f"-> Found local backup inside transcripts/ folder for video {video_id}. Restoring state...")
            with open(backup_text_file, "r", encoding="utf-8") as f:
                entry["claude_ocr_text"] = f.read()
            entry["status"] = "completed"
            return

        local_audio_path = None
        try:
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            output_file = self._base_dir / f"{video_id}.m4a"
            
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best', 
                'outtmpl': str(self._base_dir / f"{video_id}.%(ext)s"),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                }],
                'quiet': True,
            }
            if self.cookie_file.exists():
                ydl_opts['cookiefile'] = str(self.cookie_file)
                
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            local_audio_path = output_file
            
            # Extract Text with VAD filter checks
            segments, _ = whisper_model.transcribe(
                str(local_audio_path), beam_size=5, vad_filter=True, 
                vad_parameters=dict(min_speech_duration_ms=250)
            )
            transcript_text = " ".join([segment.text for segment in segments])
            entry["claude_ocr_text"] = transcript_text
            entry["status"] = "completed"
            with open(backup_text_file, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            print(f"Successfully transcribed audio segment {video_id}")
        except Exception as e:
            print(f"Failed transcription pipeline for {video_id}: {e}")
            entry["claude_ocr_text"] = f"Transcription Failed: {str(e)}"
            entry["status"] = "failed"
        finally:
            if local_audio_path and local_audio_path.exists():
                os.remove(local_audio_path)

    def _process_document_entry(self, entry: Dict):
        doc_path = Path(entry["internal_pdf_path"])
        try:
            if not doc_path.exists():
                raise FileNotFoundError(f"Underlying converted PDF file artifact missing from path: {doc_path}")
                
            with open(doc_path, "rb") as f:
                pdf_bytes = f.read()

            response = self.bedrock.converse(
                modelId=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"document": {"name": doc_path.stem, "format": "pdf", "source": {"bytes": pdf_bytes}}},
                        {"text": "Extract all text from this math document. Use LaTeX for equations. Follow the native language of the document. Do not add any commentary or explanations, just return the raw extracted text."}
                    ]
                }],
                inferenceConfig={"temperature": 0.0}
            )
            entry["claude_ocr_text"] = response['output']['message']['content'][0]['text']
            entry["status"] = "completed"
            print(f"Successfully executed Claude OCR for document: {entry['id']}")
        except Exception as e:
            print(f"Failed Bedrock Claude OCR processing for {entry['id']}: {e}")
            entry["claude_ocr_text"] = f"OCR Failed: {str(e)}"
            entry["status"] = "failed"
 
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
