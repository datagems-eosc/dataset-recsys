"""Tests for MathE sync catalog metadata and material discovery."""

import sqlite3

import pytest

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

    with syncer.catalog.connection() as conn:
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
    with syncer.catalog.connection() as conn:
        conn.execute(
            """
            UPDATE sync_entries
            SET content_subtype = 'video_unknown'
            WHERE id = 'abcdefghijk'
            """
        )
        conn.commit()

    # Re-running the migration must preserve state and remain idempotent.
    syncer.catalog = type(syncer.catalog)(syncer.db_path)
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
    with syncer.catalog.connection() as conn:
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

    syncer.reconcile_sources()
    syncer.reconcile_sources()

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


def test_discovery_reconciles_added_updated_and_removed_materials(
    tmp_path,
    monkeypatch,
):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "221.pdf").write_bytes(b"first")
    (pdf_dir / "222.pdf").write_bytes(b"second")

    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
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
    ]

    class FakeMatheClient:
        def get_video_materials(self):
            return video_rows

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", FakeMatheClient)
    syncer.reconcile_sources()

    removed_transcript = syncer._transcript_dir / "Zyxwvutsrqp.txt"
    removed_transcript.write_text("recoverable transcript", encoding="utf-8")
    (pdf_dir / "221.pdf").unlink()
    (pdf_dir / "223.pdf").write_bytes(b"third")
    video_rows[:] = [
        {
            "platform_material_id": 904,
            "link": "https://youtu.be/abcdefghijk",
            "platform_type": 2,
        },
        {
            "platform_material_id": 905,
            "link": "https://youtube.com/watch?v=abcdefghijk",
            "platform_type": 1,
        },
        {
            "platform_material_id": 906,
            "link": "newvideo123",
            "platform_type": 1,
        },
    ]

    syncer.reconcile_sources()

    entries = {entry["id"]: entry for entry in syncer.get_raw()}
    assert set(entries) == {"222.pdf", "223.pdf", "abcdefghijk", "newvideo123"}
    assert entries["abcdefghijk"]["platform_material_id"] == "904"
    assert entries["abcdefghijk"]["content_subtype"] == "video_review"
    assert entries["abcdefghijk"]["source_value"] == (
        "https://youtu.be/abcdefghijk"
    )
    assert entries["newvideo123"]["content_subtype"] == "video_lesson"
    assert removed_transcript.read_text(encoding="utf-8") == (
        "recoverable transcript"
    )


def test_video_registry_failure_preserves_cached_catalog(tmp_path, monkeypatch):
    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer.catalog.connection() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, source_value, claude_ocr_text, status,
                platform_material_id, content_subtype
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "abcdefghijk",
                "video",
                "https://youtu.be/abcdefghijk",
                "cached transcript",
                "completed",
                "901",
                "video_lesson",
            ),
        )
        conn.commit()

    class FailingMatheClient:
        def get_video_materials(self):
            raise RuntimeError("temporary PostgreSQL outage")

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", FailingMatheClient)

    syncer.reconcile_sources()

    assert [entry["id"] for entry in syncer.get_raw()] == ["abcdefghijk"]


def test_document_reconciliation_only_uses_available_source_directories(
    tmp_path,
    monkeypatch,
):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer.catalog.connection() as conn:
        conn.executemany(
            """
            INSERT INTO sync_entries (
                id, type, internal_pdf_path, claude_ocr_text, status,
                platform_material_id, content_subtype
            ) VALUES (?, 'document', ?, ?, 'completed', ?, ?)
            """,
            [
                ("221.pdf", "/old/221.pdf", "pdf text", "221", "pdf"),
                (
                    "222.docx",
                    "/old/222_docx.pdf",
                    "docx text",
                    "222",
                    "docx",
                ),
            ],
        )
        conn.commit()

    class EmptyMatheClient:
        def get_video_materials(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", EmptyMatheClient)

    syncer.reconcile_sources()

    entries = {entry["id"]: entry for entry in syncer.get_raw()}
    assert "221.pdf" not in entries
    assert entries["222.docx"]["claude_ocr_text"] == "docx text"


def test_reconciliation_preserves_completed_legacy_document_row(
    tmp_path,
    monkeypatch,
):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    source_pdf = pdf_dir / "221.pdf"
    source_pdf.write_bytes(b"current source")

    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer.catalog.connection() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, internal_pdf_path, claude_ocr_text, status,
                platform_material_id, content_subtype
            ) VALUES (?, 'document', ?, ?, 'completed', ?, 'pdf')
            """,
            ("./221.pdf", "./221.pdf", "completed OCR", "221"),
        )
        conn.commit()

    class EmptyMatheClient:
        def get_video_materials(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", EmptyMatheClient)

    syncer.reconcile_sources()

    entries = syncer.get_raw()
    assert len(entries) == 1
    assert entries[0]["id"] == "./221.pdf"
    assert entries[0]["internal_pdf_path"] == str(source_pdf)
    assert entries[0]["claude_ocr_text"] == "completed OCR"
    assert entries[0]["status"] == "completed"


def test_empty_video_transcript_is_requeued(tmp_path, monkeypatch):
    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer.catalog.connection() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, source_value, claude_ocr_text, status,
                platform_material_id, content_subtype
            ) VALUES (?, 'video', ?, '', 'completed', ?, ?)
            """,
            (
                "abcdefghijk",
                "https://youtu.be/abcdefghijk",
                "901",
                "video_lesson",
            ),
        )
        conn.commit()
    (syncer._transcript_dir / "abcdefghijk.txt").write_text(
        "\n",
        encoding="utf-8",
    )

    class FakeMatheClient:
        def get_video_materials(self):
            return [
                {
                    "platform_material_id": 901,
                    "link": "https://youtu.be/abcdefghijk",
                    "platform_type": 1,
                }
            ]

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", FakeMatheClient)

    syncer.reconcile_sources()

    [entry] = syncer.get_raw()
    assert entry["status"] == "pending"
    assert entry["claude_ocr_text"] is None


def test_source_reconciliation_propagates_catalog_failure(tmp_path, monkeypatch):
    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    def fail_catalog_read():
        raise RuntimeError("catalog read failed")

    monkeypatch.setattr(syncer.catalog, "list_entries", fail_catalog_read)

    with pytest.raises(RuntimeError, match="catalog read failed"):
        syncer.reconcile_sources()
