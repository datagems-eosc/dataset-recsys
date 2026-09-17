"""
SQLite persistence for the local MathE synchronization catalog.

MathESyncCatalog
├── Owns syncer.db and its schema
├── Executes all sync_entries SQL
├── Stores processing results
└── Reconciles additions, updates and removals
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

from dataset_recsys.utils.mathe_sync_migrations import migrate_sync_catalog


@dataclass(frozen=True)
class DocumentSyncSource:
    """One document currently present in an authoritative source directory."""

    entry_id: str
    internal_pdf_path: str
    platform_material_id: str
    content_subtype: str


@dataclass(frozen=True)
class VideoSyncSource:
    """One video currently present in the MathE platform registry."""

    entry_id: str
    source_value: str
    platform_material_id: str
    content_subtype: str | None
    transcript_backup: str | None


def has_completed_processing(entry: Mapping[str, Any]) -> bool:
    """Return whether an entry contains a successful completed text result."""
    text = str(entry.get("claude_ocr_text") or "").strip()
    return entry.get("status") == "completed" and bool(text) and not (
        text.startswith("OCR Failed") or text.startswith("Transcription Failed")
    )


class MathESyncCatalog:
    """Store and reconcile MathE OCR/transcript processing state in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a transaction with dictionary-style result rows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        """Create or migrate the catalog without discarding processing state."""
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_entries (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    source_value TEXT,
                    internal_pdf_path TEXT,
                    claude_ocr_text TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    platform_material_id TEXT,
                    content_subtype TEXT
                )
                """
            )
            # TRANSITIONAL: Remove with mathe_sync_migrations after every
            # deployed sync catalog has been upgraded and backfilled.
            migrate_sync_catalog(conn)

    def list_entries(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM sync_entries").fetchall()
        return [dict(row) for row in rows]

    def list_completed_materials(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, type, platform_material_id, claude_ocr_text
                FROM sync_entries
                WHERE status = 'completed'
                  AND type IN ('document', 'video')
                  AND claude_ocr_text IS NOT NULL
                  AND claude_ocr_text != ''
                ORDER BY type, id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def reconcile_videos(self, sources: Sequence[VideoSyncSource]) -> int:
        """Upsert current videos and remove rows absent from a loaded registry."""
        active_ids = {source.entry_id for source in sources}
        with self.connection() as conn:
            stored_entries = {
                str(row["id"]): dict(row)
                for row in conn.execute("SELECT * FROM sync_entries").fetchall()
            }

            for source in sources:
                existing = stored_entries.get(source.entry_id)
                if existing is None:
                    status = (
                        "completed"
                        if source.transcript_backup is not None
                        else "pending"
                    )
                    conn.execute(
                        """
                        INSERT INTO sync_entries (
                            id,
                            type,
                            source_value,
                            claude_ocr_text,
                            status,
                            platform_material_id,
                            content_subtype
                        ) VALUES (?, 'video', ?, ?, ?, ?, ?)
                        """,
                        (
                            source.entry_id,
                            source.source_value,
                            source.transcript_backup,
                            status,
                            source.platform_material_id,
                            source.content_subtype,
                        ),
                    )
                    continue

                conn.execute(
                    """
                    UPDATE sync_entries
                    SET type = 'video',
                        source_value = ?,
                        platform_material_id = ?,
                        content_subtype = ?
                    WHERE id = ?
                    """,
                    (
                        source.source_value,
                        source.platform_material_id,
                        source.content_subtype,
                        source.entry_id,
                    ),
                )
                if (
                    source.transcript_backup is not None
                    and not has_completed_processing(existing)
                ):
                    conn.execute(
                        """
                        UPDATE sync_entries
                        SET status = 'completed', claude_ocr_text = ?
                        WHERE id = ?
                        """,
                        (source.transcript_backup, source.entry_id),
                    )
                elif (
                    existing.get("status") == "completed"
                    and not str(existing.get("claude_ocr_text") or "").strip()
                ):
                    conn.execute(
                        """
                        UPDATE sync_entries
                        SET status = 'pending', claude_ocr_text = NULL
                        WHERE id = ?
                        """,
                        (source.entry_id,),
                    )

            stored_video_ids = {
                str(row["id"])
                for row in conn.execute(
                    "SELECT id FROM sync_entries WHERE type = 'video'"
                ).fetchall()
            }
            removed_ids = sorted(stored_video_ids - active_ids)
            conn.executemany(
                "DELETE FROM sync_entries WHERE id = ? AND type = 'video'",
                [(entry_id,) for entry_id in removed_ids],
            )

        return len(removed_ids)

    def reconcile_documents(
        self,
        sources: Sequence[DocumentSyncSource],
        authoritative_subtypes: set[str],
    ) -> int:
        """Upsert current documents and reconcile available source directories."""
        active_ids = {source.entry_id for source in sources}
        with self.connection() as conn:
            for source in sources:
                conn.execute(
                    """
                    INSERT INTO sync_entries (
                        id,
                        type,
                        internal_pdf_path,
                        status,
                        platform_material_id,
                        content_subtype
                    ) VALUES (?, 'document', ?, 'pending', ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        type = 'document',
                        internal_pdf_path = EXCLUDED.internal_pdf_path,
                        platform_material_id = EXCLUDED.platform_material_id,
                        content_subtype = EXCLUDED.content_subtype
                    """,
                    (
                        source.entry_id,
                        source.internal_pdf_path,
                        source.platform_material_id,
                        source.content_subtype,
                    ),
                )

            removed_ids = []
            rows = conn.execute(
                """
                SELECT id, content_subtype
                FROM sync_entries
                WHERE type = 'document'
                """
            ).fetchall()
            for row in rows:
                entry_id = str(row["id"])
                subtype = str(row["content_subtype"] or "").lower()
                if not subtype:
                    subtype = Path(entry_id).suffix.lower().lstrip(".")
                if subtype in authoritative_subtypes and entry_id not in active_ids:
                    removed_ids.append(entry_id)

            conn.executemany(
                "DELETE FROM sync_entries WHERE id = ? AND type = 'document'",
                [(entry_id,) for entry_id in removed_ids],
            )

        return len(removed_ids)

    def list_pending_entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT *
            FROM sync_entries
            WHERE status NOT IN ('completed', 'failed')
        """
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_processing_result(
        self,
        entry_id: str,
        status: str,
        text: str | None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sync_entries
                SET status = ?, claude_ocr_text = ?
                WHERE id = ?
                """,
                (status, text, entry_id),
            )

    def count_entries(self) -> int:
        with self.connection() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM sync_entries").fetchone()[0]
            )

    def count_unfinished_entries(self) -> int:
        with self.connection() as conn:
            return int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM sync_entries
                    WHERE status NOT IN ('completed', 'failed')
                    """
                ).fetchone()[0]
            )

    def list_internal_pdf_paths(self) -> list[str]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT internal_pdf_path
                FROM sync_entries
                WHERE internal_pdf_path IS NOT NULL
                """
            ).fetchall()
        return [str(row["internal_pdf_path"]) for row in rows]


__all__ = [
    "DocumentSyncSource",
    "MathESyncCatalog",
    "VideoSyncSource",
    "has_completed_processing",
]
