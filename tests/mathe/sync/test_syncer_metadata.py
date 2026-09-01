"""Tests for MathE sync catalog metadata and material discovery."""

import sqlite3

from dataset_recsys.utils import mathe_sync_migrations
import dataset_recsys.utils.mathe_syncer as mathe_syncer_module


LEGACY_SYNC_SCHEMA = """
    CREATE TABLE sync_entries (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        source_value TEXT,
        internal_pdf_path TEXT,
        claude_ocr_text TEXT,
        status TEXT NOT NULL DEFAULT 'pending'
    )
"""


def _disable_bedrock(monkeypatch):
    monkeypatch.setattr(
        mathe_syncer_module.boto3,
        "client",
        lambda *args, **kwargs: object(),
    )


def test_syncer_passes_configured_schema_to_mathe_client(tmp_path, monkeypatch):
    _disable_bedrock(monkeypatch)
    monkeypatch.setenv("DATAGEMS_POSTGRES_SCHEMA", "mathe_dev")
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    client = object()
    captured = {}

    def fake_mathe_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(mathe_syncer_module, "MatheMirrorClient", fake_mathe_client)

    assert syncer._get_mathe_client() is client
    assert captured == {"schema": "mathe_dev"}


def test_sync_catalog_migrates_and_backfills_legacy_entries(tmp_path, monkeypatch):
    db_path = tmp_path / "syncer.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(LEGACY_SYNC_SCHEMA)
        conn.executemany(
            "INSERT INTO sync_entries VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "./221.pdf",
                    "document",
                    None,
                    "./221.pdf",
                    "document text",
                    "completed",
                ),
                (
                    "222.docx",
                    "document",
                    None,
                    "/tmp/222_docx.pdf",
                    "word document text",
                    "completed",
                ),
                (
                    "223.pptx",
                    "document",
                    None,
                    "/tmp/223_pptx.pdf",
                    "presentation text",
                    "completed",
                ),
                (
                    "abcdefghijk",
                    "audio",
                    "https://www.youtube.com/watch?v=abcdefghijk",
                    None,
                    "video transcript",
                    "completed",
                ),
            ],
        )

    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    with syncer._get_sqlite_conn() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sync_entries)").fetchall()
        }
        entries = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM sync_entries").fetchall()
        }

    assert set(mathe_sync_migrations.SYNC_ENTRY_METADATA_COLUMNS) <= columns
    assert {
        "content_kind",
        "source_asset_id",
        "source_url",
        "processing_kind",
    }.isdisjoint(columns)
    expected_document_metadata = {
        "type": "document",
        "platform_material_id": "221",
        "content_subtype": "pdf",
    }
    expected_video_metadata = {
        "type": "video",
        "platform_material_id": None,
        "content_subtype": None,
    }
    assert {
        key: entries["./221.pdf"][key]
        for key in expected_document_metadata
    } == expected_document_metadata
    assert {
        key: entries["abcdefghijk"][key]
        for key in expected_video_metadata
    } == expected_video_metadata
    assert entries["abcdefghijk"]["source_value"] == (
        "https://www.youtube.com/watch?v=abcdefghijk"
    )
    assert entries["abcdefghijk"]["claude_ocr_text"] == "video transcript"
    assert entries["abcdefghijk"]["status"] == "completed"
    assert entries["./221.pdf"]["internal_pdf_path"] == "./221.pdf"
    assert entries["./221.pdf"]["claude_ocr_text"] == "document text"
    assert entries["./221.pdf"]["status"] == "completed"
    assert entries["222.docx"]["platform_material_id"] == "222"
    assert entries["222.docx"]["content_subtype"] == "docx"
    assert entries["223.pptx"]["platform_material_id"] == "223"
    assert entries["223.pptx"]["content_subtype"] == "pptx"

    # Transitional Stage 1 placeholder values are normalized in place.
    with syncer._get_sqlite_conn() as conn:
        conn.execute(
            """
            UPDATE sync_entries
            SET content_subtype = 'video_unknown'
            WHERE id = 'abcdefghijk'
            """
        )
        conn.commit()

    # Re-running the migration must preserve state and remain idempotent.
    syncer._init_db()
    assert len(syncer.get_raw()) == 4
    migrated_video = {
        entry["id"]: entry for entry in syncer.get_raw()
    }["abcdefghijk"]
    assert migrated_video["content_subtype"] is None


def test_discovery_preserves_platform_id_and_video_subtype(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "221.pdf").write_bytes(b"placeholder")

    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer._get_sqlite_conn() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, source_value, claude_ocr_text, status
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "abcdefghijk",
                "video",
                "abcdefghijk",
                None,
                "pending",
            ),
        )
        conn.commit()
    (syncer._transcript_dir / "abcdefghijk.txt").write_text(
        "cached transcript",
        encoding="utf-8",
    )

    video_rows = [
        {
            "platform_material_id": 901,
            "link": "https://www.youtube.com/watch?v=abcdefghijk",
            "platform_type": 1,
        },
        {
            "platform_material_id": 902,
            "link": "Zyxwvutsrqp",
            "platform_type": 2,
        },
        {
            "platform_material_id": 903,
            "link": "https://example.com/not-youtube",
            "platform_type": 1,
        },
    ]

    class FakeMatheClient:
        def __init__(self):
            self.closed = False

        def get_video_materials(self):
            return video_rows

        def close(self):
            self.closed = True

    fake_mathe_client = FakeMatheClient()
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: fake_mathe_client)

    syncer._init_data()
    syncer._init_data()

    entries = {entry["id"]: entry for entry in syncer.get_raw()}
    assert len(entries) == 3
    assert fake_mathe_client.closed is True

    lesson = entries["abcdefghijk"]
    assert lesson["platform_material_id"] == "901"
    assert lesson["type"] == "video"
    assert lesson["content_subtype"] == "video_lesson"
    assert lesson["id"] == "abcdefghijk"
    assert lesson["source_value"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert lesson["status"] == "completed"
    assert lesson["claude_ocr_text"] == "cached transcript"

    review = entries["Zyxwvutsrqp"]
    assert review["platform_material_id"] == "902"
    assert review["content_subtype"] == "video_review"
    assert review["id"] == "Zyxwvutsrqp"
    assert review["source_value"] == "Zyxwvutsrqp"
    assert review["status"] == "pending"

    document = entries["221.pdf"]
    assert document["platform_material_id"] == "221"
    assert document["type"] == "document"
    assert document["content_subtype"] == "pdf"
