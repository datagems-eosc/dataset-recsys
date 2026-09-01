"""Transitional migrations for MathE's deployed SQLite sync catalog.

Delete this module and its single call from ``MathE_Syncer._init_db`` after all
deployed catalogs contain and have backfilled the Stage 1 content metadata.
"""

import sqlite3
from pathlib import Path

SYNC_ENTRY_METADATA_COLUMNS = {
    "platform_material_id": "TEXT",
    "content_subtype": "TEXT",
}


def migrate_sync_catalog(conn: sqlite3.Connection) -> None:
    """Add Stage 1 columns and classify rows created by the legacy syncer."""
    _add_sync_entry_metadata_columns(conn)
    _backfill_sync_entry_metadata(conn)


def _add_sync_entry_metadata_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(sync_entries)").fetchall()
    }
    for column_name, column_type in SYNC_ENTRY_METADATA_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE sync_entries ADD COLUMN {column_name} {column_type}"
            )


def _backfill_sync_entry_metadata(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT * FROM sync_entries").fetchall()
    for row in rows:
        entry = dict(row)
        entry_id = str(entry["id"]).strip()
        legacy_type = str(entry.get("type") or "").strip().lower()
        is_video = legacy_type in {"audio", "video"}

        if is_video:
            defaults = {
                "platform_material_id": None,
                "content_subtype": None,
            }
            content_type = "video"
        else:
            source_path = Path(Path(entry_id).name)
            defaults = {
                "platform_material_id": (
                    source_path.stem if source_path.stem.isnumeric() else None
                ),
                "content_subtype": (
                    source_path.suffix.lower().lstrip(".") or "document"
                ),
            }
            content_type = "document"

        values = {
            column_name: entry.get(column_name) or default_value
            for column_name, default_value in defaults.items()
        }
        if is_video and entry.get("content_subtype") == "video_unknown":
            values["content_subtype"] = None

        conn.execute(
            """
            UPDATE sync_entries
            SET type = ?,
                platform_material_id = ?,
                content_subtype = ?
            WHERE id = ?
            """,
            (
                content_type,
                values["platform_material_id"],
                values["content_subtype"],
                entry["id"],
            ),
        )


__all__ = ["SYNC_ENTRY_METADATA_COLUMNS", "migrate_sync_catalog"]
