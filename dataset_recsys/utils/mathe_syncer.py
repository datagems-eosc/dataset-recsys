import signal
from dataclasses import replace
import subprocess
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
    MISSING_SOURCE_FILE,
    NOT_IN_REGISTRY,
    UNSUPPORTED_EXTENSION,
    DocumentRegistryGap,
    DocumentSyncSource,
    MathESyncCatalog,
    VideoLinkGap,
    VideoMaterialLink,
    VideoSyncSource,
    has_completed_processing,
)
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.utils.scayle import OCR_LLM, ocr_file

logger = logging.getLogger(__name__)

# Document OCR always runs on SCAYLE. SCAYLE receives one rendered page per
# request.
SCAYLE_OCR_PROMPT = (
    "Extract all text from this math document page. Use LaTeX for equations. "
    "Follow the native language of the document. Do not add any commentary or "
    "explanations, just return the raw extracted text."
)

# MATHE_OCR_BACKEND used to select SCAYLE or Bedrock. It is no longer read; a
# leftover value only triggers a warning, so an old setting cannot route OCR
# to another provider.
_RETIRED_OCR_BACKEND_ENV = "MATHE_OCR_BACKEND"


_RAW_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
# Forms seen in MathE records that omit the YouTube domain: a raw ID followed
# by URL parameters ("odkaPgWPQGo&t"), and a bare watch path
# ("watch?v=KQ5tyLxCgb8"). Matched against the whole value, so fragments of
# other text are not taken for video IDs.
_RAW_VIDEO_ID_WITH_PARAMS = re.compile(r"([A-Za-z0-9_-]{11})[&?#].*", re.DOTALL)
_BARE_WATCH_PATH = re.compile(
    r"/?watch\?(?:[^#\s]*?&(?:amp;)?)?v=([A-Za-z0-9_-]{11})(?:[&#].*)?",
    re.IGNORECASE | re.DOTALL,
)
_YOUTUBE_URL_VIDEO_ID = re.compile(
    r"(?:"
    r"youtu\.be/"
    r"|youtube(?:-nocookie)?\.com/(?:embed|v|e|shorts|live)/"
    r"|youtube(?:-nocookie)?\.com/(?:watch|attribution_link)?\?(?:[^#\s]*?&(?:amp;)?)?v="
    r")([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)

# Document file types the sync can convert and OCR, as stored in
# platform_materials.file_ext.
SUPPORTED_DOCUMENT_EXTENSIONS = ("pdf", "docx", "pptx")


def _warn_if_retired_ocr_backend_is_set() -> None:
    value = (os.getenv(_RETIRED_OCR_BACKEND_ENV) or "").strip().lower()
    if value and value != "scayle":
        logger.warning(
            "%s=%s is ignored: MathE document OCR always uses SCAYLE.",
            _RETIRED_OCR_BACKEND_ENV,
            value,
        )


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
        
        # Document OCR runs on SCAYLE; credentials and URL come from the
        # SCAYLE_* environment variables (see dataset_recsys/utils/scayle.py).
        # ``ocr_model`` names the model that produces the OCR text, so runs
        # can report it for reproducibility.
        _warn_if_retired_ocr_backend_is_set()
        self.ocr_model = OCR_LLM
        
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
        value = link_value.strip()
        if (
            _RAW_VIDEO_ID.fullmatch(value)
            or _RAW_VIDEO_ID_WITH_PARAMS.fullmatch(value)
            or _BARE_WATCH_PATH.fullmatch(value)
        ):
            return True
        lowered = link_value.lower()
        return "youtube" in lowered or "youtu.be" in lowered

    def _extract_video_id(self, link_value: str) -> str:
        """Return the 11-character YouTube video ID of a link or raw ID.

        Accepts watch URLs (``v=`` anywhere in the query), ``youtu.be``
        short links, and ``/embed/``, ``/v/``, ``/e/``, ``/shorts/`` and
        ``/live/`` paths, on youtube.com, its ``www``/``m``/``music``
        subdomains and youtube-nocookie.com, with or without a scheme. Also
        accepts a raw ID followed by URL parameters (``odkaPgWPQGo&t``) and a
        watch path without the domain (``watch?v=KQ5tyLxCgb8``).
        """
        value = link_value.strip()
        if _RAW_VIDEO_ID.fullmatch(value):
            return value
        for pattern in (_RAW_VIDEO_ID_WITH_PARAMS, _BARE_WATCH_PATH):
            bare_match = pattern.fullmatch(value)
            if bare_match:
                return bare_match.group(1)
        id_match = _YOUTUBE_URL_VIDEO_ID.search(value)
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

    def require_source_directories(self) -> None:
        """Fail a sync early when a MathE material directory is missing.

        This runs when a sync starts, not in ``__init__``: the API creates the
        syncer at import time, so raising there would stop the whole service
        from starting.
        """
        missing = [
            str(directory)
            for directory in (
                self._pdf_dir,
                self._docx_dir,
                self._ppt_dir,
                self._transcript_dir,
            )
            if not directory.is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "MathE sync cannot start; required directories are missing: "
                + ", ".join(missing)
            )

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
        registry_documents: Optional[list[Dict[str, str]]] = None
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
            # The document list only serves to report registry documents the
            # sync cannot process, so its failure must not stop video sync.
            try:
                registry_documents = [
                    {
                        "platform_material_id": str(row["platform_material_id"]),
                        "file_ext": str(row["file_ext"] or "").strip().lower(),
                    }
                    for row in mathe_client.get_document_materials()
                ]
            except Exception as e:
                logger.warning(
                    "Could not load MathE document registry; keeping the "
                    "previous list of documents without a source file: %s",
                    e,
                )
        except Exception as e:
            print(f"Warning: PostgreSQL lookup failed. Relying on local data cache. Exception: {e}")
            db_videos = []
        finally:
            if mathe_client is not None:
                try:
                    mathe_client.close()
                except Exception as e:
                    logger.warning("Failed to close MathE PostgreSQL client: %s", e)

        # One source per YouTube video, so each video is transcribed once,
        # linked to every MathE material that points to it. The registry is
        # ordered by material ID, so the lowest ID is the video's primary.
        video_sources_by_id: dict[str, VideoSyncSource] = {}
        video_link_gaps: list[VideoLinkGap] = []
        for video in db_videos:
            link = video["link"]
            platform_material_id = video["platform_material_id"]
            try:
                if not self._is_youtube_asset(link):
                    raise ValueError("not a YouTube URL or ID")
                video_id = self._extract_video_id(link)
            except ValueError:
                logger.warning(
                    "Skipping MathE video material %s: link %r names no "
                    "YouTube video",
                    platform_material_id,
                    link,
                )
                video_link_gaps.append(VideoLinkGap(platform_material_id, link))
                continue

            material_link = VideoMaterialLink(
                platform_material_id=platform_material_id,
                content_subtype=VIDEO_TYPE_TO_SUBTYPE.get(video["platform_type"]),
            )
            existing = video_sources_by_id.get(video_id)
            if existing is not None:
                logger.info(
                    "MathE materials %s and %s share YouTube video %s; "
                    "indexing both with one transcript",
                    existing.platform_material_id,
                    platform_material_id,
                    video_id,
                )
                video_sources_by_id[video_id] = replace(
                    existing,
                    linked_materials=existing.linked_materials + (material_link,),
                )
                continue

            video_sources_by_id[video_id] = VideoSyncSource(
                entry_id=video_id,
                source_value=link,
                platform_material_id=platform_material_id,
                content_subtype=material_link.content_subtype,
                transcript_backup=self._read_transcript_backup(video_id),
                linked_materials=(material_link,),
            )
        video_sources = list(video_sources_by_id.values())

        # An empty successful registry result means there are no videos. A
        # failed query is not authoritative and must preserve the local cache.
        if video_catalog_loaded:
            self.catalog.replace_video_link_gaps(video_link_gaps)
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

        if registry_documents is not None:
            self._record_document_registry_gaps(registry_documents)

        total_entries = self.catalog.count_entries()
        print(
            f"Discovered {total_entries} total entries, with "
            f"{self.count_available_pdfs()} available PDFs."
        )

    def _record_document_registry_gaps(
        self,
        registry_documents: list[Dict[str, str]],
    ) -> None:
        """Record documents that are in only one of the registry and the catalog.

        Documents are discovered from the mounted source directories by file
        name (``<platform_material_id>.<ext>``), not from the registry. Without
        this check, a registry document without such a file would be absent
        from the index, and a file without a registry document would be
        indexed, both without any trace. Indexing itself is not changed.
        """
        catalogued_extensions: dict[str, set[str]] = {}
        for row in self.catalog.list_entries():
            material_id = row.get("platform_material_id")
            if row.get("type") != "document" or not material_id:
                continue
            extension = str(
                row.get("content_subtype")
                or Path(str(row["id"])).suffix.lstrip(".")
            ).lower()
            catalogued_extensions.setdefault(str(material_id), set()).add(extension)

        registry_ids = {document["platform_material_id"] for document in registry_documents}
        gaps = [
            DocumentRegistryGap(
                platform_material_id=document["platform_material_id"],
                file_ext=document["file_ext"],
                reason=(
                    MISSING_SOURCE_FILE
                    if document["file_ext"] in SUPPORTED_DOCUMENT_EXTENSIONS
                    else UNSUPPORTED_EXTENSION
                ),
            )
            for document in registry_documents
            if document["platform_material_id"] not in catalogued_extensions
        ]
        gaps.extend(
            DocumentRegistryGap(
                platform_material_id=material_id,
                file_ext=", ".join(sorted(extensions)),
                reason=NOT_IN_REGISTRY,
            )
            for material_id, extensions in catalogued_extensions.items()
            if material_id not in registry_ids
        )
        self.catalog.replace_document_registry_gaps(gaps)

        for reason in (MISSING_SOURCE_FILE, UNSUPPORTED_EXTENSION, NOT_IN_REGISTRY):
            material_ids = sorted(
                (gap.platform_material_id for gap in gaps if gap.reason == reason),
                key=lambda material_id: (len(material_id), material_id),
            )
            if material_ids:
                logger.warning(
                    "%d MathE documents are in only one of the registry and "
                    "the catalog (%s): %s",
                    len(material_ids),
                    reason,
                    ", ".join(material_ids),
                )

    def get_video_registry_gaps(self) -> List[Dict[str, Any]]:
        """Return registry video materials that are not indexed, and why."""
        return self.catalog.list_video_registry_gaps()

    def get_document_registry_gaps(self) -> List[Dict[str, Any]]:
        """Return documents in only one of the registry and the catalog."""
        return self.catalog.list_document_registry_gaps()

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
                    "ocr_model",
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

    def requeue_documents_for_reprocessing(self) -> int:
        """Mark every document entry pending so the next OCR pass redoes it.

        Existing text is kept until the new OCR result overwrites it, so a
        shutdown mid-run does not leave a document with no text.
        """
        requeued = 0
        for row in self.catalog.list_entries():
            if row.get("type") != "document":
                continue
            self.catalog.update_processing_result(
                entry_id=row["id"],
                status="pending",
                text=row.get("claude_ocr_text") or "",
                ocr_model=None,
            )
            requeued += 1
        return requeued

    def process_pending_materials(self, limit: Optional[int] = None, retry_failed: bool = True) -> None:
        """Run document OCR or video transcription for pending catalog entries."""

        # TODO: Add a configurable delay for retrying transient failures; see the
        # retry policy TODO in docs/source/mathe_workflows.md.
        if retry_failed:
            requeued = self.catalog.requeue_failed_entries()
            if requeued:
                print(f"Retrying {requeued} previously failed entries.")
                logger.info("Requeued %s failed MathE entries for retry", requeued)

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

            # Only a successful document OCR records the model that produced
            # the text; videos are transcribed, not OCR'd.
            entry["ocr_model"] = None
            if entry.get("type") == "video":
                self._process_video_entry(entry, whisper_model)
            else:
                self._process_document_entry(entry)

            self.catalog.update_processing_result(
                entry_id=entry["id"],
                status=entry["status"],
                text=entry["claude_ocr_text"],
                ocr_model=entry.get("ocr_model"),
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

        # 1. Check if the in-memory entry passed to this method is already completed
        if has_completed_processing(entry):
            print(f"-> Video {video_id} is already completed in syncer.db. Skipping transcription.")
            return

        # 2. Check direct database record in syncer.db
        catalog_entry = self.catalog.get_entry(video_id)
        if catalog_entry and has_completed_processing(catalog_entry):
            print(f"-> Found completed record for video {video_id} in syncer.db. Restoring state...")
            entry["claude_ocr_text"] = catalog_entry["claude_ocr_text"]
            entry["status"] = "completed"
            return

        # 3. Check local transcript backup file on disk
        backup_text_file = self._transcript_dir / f"{video_id}.txt"
        backup_text = self._read_transcript_backup(video_id)
        if backup_text is not None:
            print(f"-> Found local backup inside transcripts/ folder for video {video_id}. Restoring state...")
            entry["claude_ocr_text"] = backup_text
            entry["status"] = "completed"
            return

        # 4. Perform fresh download & Whisper transcription if not found above
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
                # Workarounds for YouTube 403 Forbidden & JS Cipher challenges
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'ios', 'mweb', 'web'],
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                }
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

    def _ocr_document_with_scayle(self, doc_path: Path) -> str:
        # SCAYLE only takes images, so ocr_file renders each PDF page and OCRs it.
        # The PDF text layer is skipped on purpose: it garbles maths, and the
        # LaTeX output from the vision model is what we want here.
        text = ocr_file(
            str(doc_path),
            llm=self.ocr_model,
            prefer_text_layer=False,
            prompt=SCAYLE_OCR_PROMPT,
        )
        if not text.strip():
            raise ValueError("SCAYLE OCR returned empty text")
        return text

    def _process_document_entry(self, entry: Dict):
        doc_path = Path(entry["internal_pdf_path"])
        try:
            if not doc_path.exists():
                raise FileNotFoundError(f"Underlying converted PDF file artifact missing from path: {doc_path}")

            text = self._ocr_document_with_scayle(doc_path)

            # The column keeps its legacy name ("claude_ocr_text") to avoid a schema change.
            entry["claude_ocr_text"] = text
            entry["ocr_model"] = self.ocr_model
            entry["status"] = "completed"
            print(
                f"Successfully executed SCAYLE OCR ({self.ocr_model}) "
                f"for document: {entry['id']}"
            )
        except Exception as e:
            print(f"Failed SCAYLE OCR processing for {entry['id']}: {e}")
            entry["claude_ocr_text"] = f"OCR Failed: {str(e)}"
            entry["ocr_model"] = None
            entry["status"] = "failed"

    def _send_notification(self, msg: str):
        # Placeholder for notification logic (e.g., email, Slack)
        print(f"NOTIFICATION: {msg}")
