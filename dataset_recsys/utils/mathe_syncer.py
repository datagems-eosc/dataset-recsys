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

from dataset_recsys.mathe_recommenders.constants import VIDEO_TYPE_TO_SUBTYPE
from dataset_recsys.storage.mathe_sync_catalog import (
    DocumentSyncSource,
    MathESyncCatalog,
    VideoSyncSource,
    has_completed_processing,
)
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
logger = logging.getLogger(__name__)


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0 0%"

    filled = round(width * current / total)
    percent = round(100 * current / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


class MathE_Syncer:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._pdf_dir = self._base_dir / "pdfs"
        self._docx_dir = self._base_dir / "docxs"
        self._ppt_dir = self._base_dir / "pptxs/"
        self._transcript_dir = self._base_dir / "transcripts"
        self._transcript_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self._base_dir / "syncer.db"
        self.catalog = MathESyncCatalog(self.db_path)

        self.status_file = self._base_dir / "sync_status.json"
        self.cookie_file = self._base_dir / "cookies.txt"
        
        # Claude 4.5 Global Configuration
        self.model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.region = "eu-central-1"
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.region, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        
        # Graceful shutdown handling
        # The top-level MathE pipeline owns this lifecycle guard.
        self.is_running = False
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

    def _get_mathe_client(self) -> MatheMirrorClient:
        return MatheMirrorClient(
            schema=os.getenv("DATAGEMS_POSTGRES_SCHEMA", "public")
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

    def _read_transcript_backup(self, video_id: str) -> Optional[str]:
        """Returns a usable transcript backup; empty files are not completed work."""
        backup_path = self._transcript_dir / f"{video_id}.txt"
        if not backup_path.exists():
            return None

        try:
            transcript = backup_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            logger.warning(
                "Could not read transcript backup %s: %s",
                backup_path,
                error,
            )
            return None
        return transcript if transcript.strip() else None

    def _convert_office_document(
        self,
        source_file: Path,
        output_dir: Path,
        target_pdf: Path,
    ) -> bool:
        if not self._libreoffice_convert(source_file, output_dir):
            return False

        generated_pdf = output_dir / f"{source_file.stem}.pdf"
        try:
            generated_pdf.rename(target_pdf)
        except OSError as error:
            logger.warning(
                "Failed to move converted PDF for %s: %s",
                source_file.name,
                error,
            )
            return False
        return True

    def _discover_office_documents(
        self,
        source_dir: Path,
        subtype: str,
        output_dir: Path,
        existing_entries: dict[str, Dict[str, Any]],
        discovered_sources: list[DocumentSyncSource],
    ) -> bool:
        """Discovers one available office-document source directory."""
        if not source_dir.exists():
            return False

        for source_file in source_dir.iterdir():
            if not (
                source_file.is_file()
                and source_file.suffix.lower() == f".{subtype}"
                and source_file.stem.isnumeric()
            ):
                continue

            filename = source_file.name
            target_pdf = output_dir / f"{source_file.stem}_{subtype}.pdf"
            existing_entry = existing_entries.get(filename)

            if existing_entry is not None:
                entry_id = str(existing_entry["id"])
                internal_path = existing_entry.get("internal_pdf_path")
                if (
                    existing_entry.get("status") == "pending"
                    and (not internal_path or not Path(internal_path).exists())
                ):
                    print(
                        "Regenerating vanished temporary sandboxed PDF for: "
                        f"{filename}"
                    )
                    self._convert_office_document(
                        source_file,
                        output_dir,
                        target_pdf,
                    )
            else:
                entry_id = filename
                if not target_pdf.exists():
                    print(
                        f"Converting {subtype.upper()} to local memory sandbox: "
                        f"{filename}"
                    )
                    if not self._convert_office_document(
                        source_file,
                        output_dir,
                        target_pdf,
                    ):
                        logger.warning(
                            "Could not create a PDF for MathE source %s",
                            filename,
                        )
                        continue
                print(f"Queueing converted {subtype.upper()} target: {filename}")

            discovered_sources.append(
                DocumentSyncSource(
                    entry_id=entry_id,
                    internal_pdf_path=str(target_pdf),
                    platform_material_id=source_file.stem,
                    content_subtype=subtype,
                )
            )

        return True

    def reconcile_sources(self) -> None:
        """Discover current MathE sources and reconcile the local catalog."""
        if not self._base_dir.exists():
            print(f"MathE base directory does not exist: {self._base_dir}")
            return

        rows = self.catalog.list_entries()

        # A deployed legacy catalog may store a document as "./221.pdf". Match
        # by filename so reconciliation preserves its completed OCR instead of
        # inserting a second row for the same platform material.
        document_entries_by_filename: dict[str, Dict[str, Any]] = {}
        for row in rows:
            if row.get("type") != "document":
                continue
            filename = Path(str(row["id"])).name
            current = document_entries_by_filename.get(filename)
            if current is None or (
                has_completed_processing(row)
                and not has_completed_processing(current)
            ):
                document_entries_by_filename[filename] = row

        # --- ROUTE 1: Discover Video Streams from PostgreSQL ---
        print("Syncing video assets from PostgreSQL platform registry...")
        mathe_client = None
        video_catalog_loaded = False
        try:
            mathe_client = self._get_mathe_client()
            db_videos = [
                {
                    "platform_material_id": str(row["platform_material_id"]),
                    "link": str(row["link"]).strip(),
                    "platform_type": int(row["platform_type"]),
                }
                for row in mathe_client.get_video_materials()
            ]
            video_catalog_loaded = True
        except Exception as e:
            print(f"Warning: PostgreSQL lookup failed. Relying on local data cache. Exception: {e}")
            db_videos = []
        finally:
            if mathe_client is not None:
                try:
                    mathe_client.close()
                except Exception as e:
                    logger.warning("Failed to close MathE PostgreSQL client: %s", e)

        seen_video_ids: dict[str, str] = {}
        video_sources: list[VideoSyncSource] = []
        for video in db_videos:
            link = video["link"]
            if not self._is_youtube_asset(link):
                continue
            try:
                video_id = self._extract_video_id(link)
            except ValueError:
                continue

            platform_material_id = video["platform_material_id"]
            if video_id in seen_video_ids:
                logger.warning(
                    "Multiple MathE materials reference YouTube video %s; "
                    "keeping platform material %s and skipping %s",
                    video_id,
                    seen_video_ids[video_id],
                    platform_material_id,
                )
                continue

            seen_video_ids[video_id] = platform_material_id
            video_sources.append(
                VideoSyncSource(
                    entry_id=video_id,
                    source_value=link,
                    platform_material_id=platform_material_id,
                    content_subtype=VIDEO_TYPE_TO_SUBTYPE.get(
                        video["platform_type"]
                    ),
                    transcript_backup=self._read_transcript_backup(video_id),
                )
            )

        # An empty successful registry result means there are no videos. A
        # failed query is not authoritative and must preserve the local cache.
        if video_catalog_loaded:
            removed_count = self.catalog.reconcile_videos(video_sources)
            if removed_count:
                logger.info(
                    "Removed %s MathE videos no longer present in the platform registry",
                    removed_count,
                )

        # --- ROUTE 2: Discover Structural Office Documents via Local Directories ---
        tmp_build_dir = Path("/tmp/libo_out")
        tmp_build_dir.mkdir(parents=True, exist_ok=True)

        document_sources: list[DocumentSyncSource] = []
        authoritative_document_subtypes: set[str] = set()

        # Office source directories are authoritative only when available.
        if self._discover_office_documents(
            self._docx_dir,
            "docx",
            tmp_build_dir,
            document_entries_by_filename,
            document_sources,
        ):
            authoritative_document_subtypes.add("docx")

        if self._discover_office_documents(
            self._ppt_dir,
            "pptx",
            tmp_build_dir,
            document_entries_by_filename,
            document_sources,
        ):
            authoritative_document_subtypes.add("pptx")

        # Discover native pre-existing material PDFs directly from server directory.
        if self._pdf_dir.exists():
            authoritative_document_subtypes.add("pdf")
            for source_file in self._pdf_dir.iterdir():
                if not (
                    source_file.is_file()
                    and source_file.suffix.lower() == ".pdf"
                    and source_file.stem.isnumeric()
                ):
                    continue

                existing_entry = document_entries_by_filename.get(source_file.name)
                entry_id = (
                    str(existing_entry["id"])
                    if existing_entry is not None
                    else source_file.name
                )
                if existing_entry is None:
                    print(
                        "Discovered native production target: "
                        f"{source_file.name}"
                    )
                document_sources.append(
                    DocumentSyncSource(
                        entry_id=entry_id,
                        internal_pdf_path=str(source_file),
                        platform_material_id=source_file.stem,
                        content_subtype="pdf",
                    )
                )

        removed_count = self.catalog.reconcile_documents(
            document_sources,
            authoritative_document_subtypes,
        )
        if removed_count:
            logger.info(
                "Removed %s MathE documents no longer present in mounted source directories",
                removed_count,
            )

        total_entries = self.catalog.count_entries()
        print(
            f"Discovered {total_entries} total entries, with "
            f"{self.count_available_pdfs()} available PDFs."
        )

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
        df = pd.DataFrame(self.catalog.list_entries())

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "id",
                    "type",
                    "platform_material_id",
                    "content_subtype",
                    "claude_ocr_text",
                    "status",
                    "material_id",
                    "source_type",
                    "pdf_path",
                ]
            )
            
        df["material_id"] = df["id"]
        df["source_type"] = df["id"].apply(lambda p: Path(p).suffix.lstrip('.').lower())
        df["pdf_path"] = df["internal_pdf_path"]
        
        return df.replace("", pd.NA)

    def get_raw(self) -> List[Dict]:
        """Returns the raw database records as a list of dictionaries."""
        return self.catalog.list_entries()

    def get_completed_materials(self) -> List[Dict]:
        """Return completed document OCR and video transcript entries."""
        return self.catalog.list_completed_materials()

    def count_available_pdfs(self) -> int:
        """Counts files that have a valid converted or native PDF living on disk."""
        return sum(
            Path(path_str).exists()
            for path_str in self.catalog.list_internal_pdf_paths()
        )

    def get_info(self) -> Dict[str, str]:
        """Returns high-level info."""
        return {
            "name": "MathE",
            "source": "Mounted Filesystem/S3",
            "dataset_folder": str(self._base_dir),
        }

    # --- OCR Logic ---

    def process_pending_materials(self, limit: Optional[int] = None) -> None:
        """Run document OCR or video transcription for pending catalog entries."""
        pending_entries = self.catalog.list_pending_entries(limit=limit)

        if not pending_entries:
            print("No pending work detected across videos or documents.")
            return

        whisper_model = None
        if any(e.get("type") == "video" for e in pending_entries):
            print("Initializing local Whisper extraction runtime engines...")
            whisper_model = WhisperModel("base", device="cpu", compute_type="float32")

        total_entries = self.catalog.count_entries()
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

            if entry.get("type") == "video":
                self._process_video_entry(entry, whisper_model)
            else:
                self._process_document_entry(entry)

            self.catalog.update_processing_result(
                entry_id=entry["id"],
                status=entry["status"],
                text=entry["claude_ocr_text"],
            )
            
            print(f"Finished processing {entry['id']}. Status: {entry['status']}, OCR text length: {len(str(entry.get('claude_ocr_text') or ''))}")
            logger.info(
                "MathE OCR progress %s material=%s status=%s",
                _progress_bar(processed + 1, len(pending_entries)),
                Path(entry["id"]).name,
                entry["status"],
            )
            processed += 1

        unfinished = self.catalog.count_unfinished_entries()
        if unfinished == 0:
            self._send_notification("OCR process finished for all files.")

    def _process_video_entry(self, entry: Dict, whisper_model: WhisperModel):
        video_id = entry["id"]

        backup_text_file = self._transcript_dir / f"{video_id}.txt"
        backup_text = self._read_transcript_backup(video_id)
        if backup_text is not None:
            print(f"-> Found local backup inside transcripts/ folder for video {video_id}. Restoring state...")
            entry["claude_ocr_text"] = backup_text
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
