import signal
import subprocess
import boto3
import json
import logging
import os
import re
import sqlite3
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
        self._ppt_dir = self._base_dir / "pptxs/"
        self._transcript_dir = self._base_dir / "transcripts"
        self._transcript_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self._base_dir / "syncer.db"
        self._init_db()        

        self.status_file = self._base_dir / "sync_status.json"
        self.cookie_file = self._base_dir / "cookies.txt"
        
        # Claude 4.5 Global Configuration
        self.model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.region = "eu-central-1"
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.region, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        
        # Graceful shutdown handling
        self.is_running = False  # Add this line        
        self.keep_running = True
        signal.signal(signal.SIGTERM, self._handle_exit)                

    def _get_sqlite_conn(self):
        """Returns a connection context manager yielding dict-like rows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Allows accessing columns by name
        return conn

    def _init_db(self):
        """Creates the sync tracking table if it doesn't already exist."""
        with self._get_sqlite_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_entries (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    source_value TEXT,
                    internal_pdf_path TEXT,
                    claude_ocr_text TEXT,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)
            conn.commit()

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
        """Loads and processes discovery directly into the SQLite Engine."""
        if not self._base_dir.exists():
            print(f"MathE base directory does not exist: {self._base_dir}")
            return

        # Query existing state indexes directly from local DB
        with self._get_sqlite_conn() as conn:
            rows = conn.execute("SELECT id, status, internal_pdf_path FROM sync_entries").fetchall()
            state_map = {row["id"]: {"status": row["status"], "internal_pdf_path": row["internal_pdf_path"]} for row in rows}
            existing_ids = set(state_map.keys())

        # --- ROUTE 1: Discover Video Streams from PostgreSQL ---
        print("Syncing video assets from PostgreSQL platform registry...")
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # MathE material_type: 1 = Video Lesson, 2 = Video Review.
                    cursor.execute(
                        "SELECT link FROM platform_materials WHERE type IN (1, 2);"
                    )
                    rows = cursor.fetchall()
                    db_links = [row[0] for row in rows if row[0]]
        except Exception as e:
            print(f"Warning: PostgreSQL lookup failed. Relying on local data cache. Exception: {e}")
            db_links = []

        with self._get_sqlite_conn() as conn:
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

                            conn.execute(
                                "INSERT INTO sync_entries (id, type, source_value, claude_ocr_text, status) VALUES (?, ?, ?, ?, ?)",
                                (video_id, "audio", link, ocr_text, status)
                            )
                            existing_ids.add(video_id)
                        else:
                            # Self-healing backup check for tracking cache mapping
                            entry = state_map.get(video_id)
                            if entry and entry.get("status") == "pending" and backup_text_file.exists():
                                with open(backup_text_file, "r", encoding="utf-8") as f:
                                    ocr_text = f.read()
                                conn.execute(
                                    "UPDATE sync_entries SET status = 'completed', claude_ocr_text = ? WHERE id = ?",
                                    (ocr_text, video_id)
                                )
                    except ValueError:
                        continue
            conn.commit()

        # --- ROUTE 2: Discover Structural Office Documents via Local Directories ---
        tmp_build_dir = Path("/tmp/libo_out")
        tmp_build_dir.mkdir(parents=True, exist_ok=True)

        with self._get_sqlite_conn() as conn:
            # 1. Preprocess Word documents -> Store location context as local /tmp
            if self._docx_dir.exists():
                for f in self._docx_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".docx" and f.stem.isnumeric():
                        original_id = f.name  # e.g., "3.docx"
                        local_pdf_target = tmp_build_dir / f"{f.stem}_docx.pdf"                    
                        
                        # Recover sandboxed file if it missing from /tmp on server reload
                        if original_id in existing_ids:
                            entry = state_map.get(original_id)
                            if entry and entry.get("status") == "pending" and not Path(entry.get("internal_pdf_path", "")).exists():
                                print(f"Regenerating vanished temporary sandboxed PDF for: {f.name}")
                                if self._libreoffice_convert(f, tmp_build_dir):
                                    (tmp_build_dir / f"{f.stem}.pdf").rename(local_pdf_target)
                        else:
                            if not local_pdf_target.exists():
                                print(f"Converting DOCX to local memory sandbox: {f.name}")
                                if self._libreoffice_convert(f, tmp_build_dir):
                                    try:
                                        (tmp_build_dir / f"{f.stem}.pdf").rename(local_pdf_target)
                                    except Exception as e:
                                        print(f"❌ Failed to rename local PDF for {f.name}: {e}")
                                        continue

                            print(f"Queueing converted DOCX target: {original_id}")
                            conn.execute(
                                "INSERT INTO sync_entries (id, type, internal_pdf_path, status) VALUES (?, ?, ?, ?)",
                                (original_id, "document", str(local_pdf_target), "pending")
                            )
                            existing_ids.add(original_id)

            # 2. Preprocess PowerPoint presentations -> Store location context as local /tmp
            if self._ppt_dir.exists():
                for f in self._ppt_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".pptx" and f.stem.isnumeric():
                        original_id = f.name  # e.g., "3.pptx"
                        local_pdf_target = tmp_build_dir / f"{f.stem}_pptx.pdf"
                        
                        # Recover sandboxed file if it missing from /tmp on server reload
                        if original_id in existing_ids:
                            entry = state_map.get(original_id)
                            if entry and entry.get("status") == "pending" and not Path(entry.get("internal_pdf_path", "")).exists():
                                print(f"Regenerating vanished temporary sandboxed PDF for: {f.name}")
                                if self._libreoffice_convert(f, tmp_build_dir):
                                    (tmp_build_dir / f"{f.stem}.pdf").rename(local_pdf_target)
                        else:
                            if not local_pdf_target.exists():
                                print(f"Converting PPTX to local memory sandbox: {f.name}")
                                if self._libreoffice_convert(f, tmp_build_dir):
                                    try:
                                        (tmp_build_dir / f"{f.stem}.pdf").rename(local_pdf_target)
                                    except Exception as e:
                                        print(f"❌ Failed to rename local PDF for {f.name}: {e}")
                                        continue

                            print(f"Queueing converted PPTX target: {original_id}")
                            conn.execute(
                                "INSERT INTO sync_entries (id, type, internal_pdf_path, status) VALUES (?, ?, ?, ?)",
                                (original_id, "document", str(local_pdf_target), "pending")
                            )
                            existing_ids.add(original_id)

            # 3. Discover native pre-existing material PDFs directly from server directory
            if self._pdf_dir.exists():
                for f in self._pdf_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".pdf" and f.stem.isnumeric():
                        original_id = f.name  # e.g., "3.pdf"
                        if original_id not in existing_ids:
                            print(f"Discovered native production target: {original_id}")
                            conn.execute(
                                "INSERT INTO sync_entries (id, type, internal_pdf_path, status) VALUES (?, ?, ?, ?)",
                                (original_id, "document", str(f), "pending")
                            )
                            existing_ids.add(original_id)
            conn.commit()

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
        """Returns the main table directly from SQLite as a DataFrame."""
        with self._get_sqlite_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM sync_entries", conn)

        if df.empty:
            return pd.DataFrame(
                columns=["id", "source_type", "claude_ocr_text", "status", "material_id", "pdf_path"]
            )
            
        df["material_id"] = df["id"]
        df["source_type"] = df["id"].apply(lambda p: Path(p).suffix.lstrip('.').lower())
        df["pdf_path"] = df["internal_pdf_path"]
        
        return df.replace("", pd.NA)

    def get_raw(self) -> List[Dict]:
        """Returns the raw database records as a list of dictionaries."""
        with self._get_sqlite_conn() as conn:
            rows = conn.execute("SELECT * FROM sync_entries").fetchall()
            return [dict(row) for row in rows]

    def count_available_pdfs(self) -> int:
        """Counts files that have a valid converted or native PDF living on disk."""
        with self._get_sqlite_conn() as conn:
            rows = conn.execute("SELECT internal_pdf_path FROM sync_entries WHERE internal_pdf_path IS NOT NULL").fetchall()
            
        valid_count = 0
        for row in rows:
            path_str = row["internal_pdf_path"]
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
            # --- TO THIS SAFE BLOCK ---
            with self._get_sqlite_conn() as conn:
                total_db_entries = conn.execute("SELECT COUNT(*) FROM sync_entries").fetchone()[0]
            print(f"Discovered {total_db_entries} total entries, with {self.count_available_pdfs()} available PDFs.")
            # limit = 1 # For testing, process only 1 file at a time. Remove or adjust this for full batch processing.
            self.run_hybrid_batch_processing(limit=limit)
            print("Lifecycle complete.")
        except Exception as e:
            print(f"Error during sync/process lifecycle: {e}")
        finally:
            self.is_running = False  # Ensure it always releases the lock

    # --- OCR Logic ---

    def run_hybrid_batch_processing(self, limit: Optional[int] = None):
        with self._get_sqlite_conn() as conn:
            query = "SELECT * FROM sync_entries WHERE status != 'completed' AND status != 'failed'"
            if limit is not None:
                query += f" LIMIT {limit}"
            pending_entries = [dict(row) for row in conn.execute(query).fetchall()]

        if not pending_entries:
            print("No pending work detected across audio formats or documents.")
            return

        whisper_model = None
        if any(e.get("type") == "audio" for e in pending_entries):
            print("Initializing local Whisper extraction runtime engines...")
            whisper_model = WhisperModel("base", device="cpu", compute_type="float32")

        with self._get_sqlite_conn() as conn:
            total_entries = conn.execute("SELECT COUNT(*) FROM sync_entries").fetchone()[0]
        skipped = total_entries - len(pending_entries)

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

            with self._get_sqlite_conn() as conn:
                conn.execute(
                    "UPDATE sync_entries SET status = ?, claude_ocr_text = ? WHERE id = ?",
                    (entry["status"], entry["claude_ocr_text"], entry["id"])
                )
                conn.commit()
            
            print(f"Finished processing {entry['id']}. Status: {entry['status']}, OCR text length: {len(str(entry.get('claude_ocr_text') or ''))}")
            logger.info(
                "MathE OCR progress %s material=%s status=%s",
                _progress_bar(processed + 1, len(pending_entries)),
                Path(entry["id"]).name,
                entry["status"],
            )
            processed += 1

        with self._get_sqlite_conn() as conn:
            unfinished = conn.execute("SELECT COUNT(*) FROM sync_entries WHERE status NOT IN ('completed', 'failed')").fetchone()[0]
        if unfinished == 0:
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
                # Use extract_info with download=True to fetch metadata along with the audio file
                info_dict = ydl.extract_info(youtube_url, download=True)
                
                # Safely parse fields with defaults if they are missing
                title = info_dict.get("title", "Unknown Title")
                description = info_dict.get("description", "No Description Provided")
                tags_list = info_dict.get("tags", [])
                keywords = ", ".join(tags_list) if tags_list else "No keywords found."

            local_audio_path = output_file
            
            # Extract Text with VAD filter checks
            segments, _ = whisper_model.transcribe(
                str(local_audio_path), beam_size=5, vad_filter=True, 
                vad_parameters=dict(min_speech_duration_ms=250)
            )
            transcript_text = " ".join([segment.text for segment in segments])
            formatted_output = (
                f"VIDEO TITLE: {title}\n"
                f"VIDEO KEYWORDS: {keywords}\n"
                f"VIDEO DESCRIPTION:\n{description}\n"
                f"{'='*40}\n"
                f"TRANSCRIPT:\n{transcript_text}"
            )
            entry["claude_ocr_text"] = formatted_output
            entry["status"] = "completed"
            with open(backup_text_file, "w", encoding="utf-8") as f:
                f.write(formatted_output)
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
