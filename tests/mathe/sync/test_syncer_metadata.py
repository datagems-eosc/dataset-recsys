"""Tests for MathE sync catalog metadata and material discovery."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from dataset_recsys.utils import mathe_sync_migrations
import dataset_recsys.utils.mathe_syncer as mathe_syncer_module
from dataset_recsys.storage import mathe_mirror_client


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


def test_syncer_passes_configured_schema_to_mathe_client(tmp_path, monkeypatch):
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


def test_mirror_client_uses_configured_schema_by_default(monkeypatch):
    monkeypatch.setenv("DATAGEMS_POSTGRES_SCHEMA", "mathe_dev")
    connection = MagicMock()
    monkeypatch.setattr(mathe_mirror_client.psycopg2, "connect", lambda **kwargs: connection)

    client = mathe_mirror_client.MatheMirrorClient()

    assert client.schema == "mathe_dev"
    statement = connection.cursor.return_value.__enter__.return_value.execute.call_args.args[0]
    assert statement == mathe_mirror_client.sql.SQL("SET search_path TO {};").format(
        mathe_mirror_client.sql.Identifier("mathe_dev")
    )


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
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    def fail_catalog_read():
        raise RuntimeError("catalog read failed")

    monkeypatch.setattr(syncer.catalog, "list_entries", fail_catalog_read)

    with pytest.raises(RuntimeError, match="catalog read failed"):
        syncer.reconcile_sources()


SOURCE_DIRECTORY_NAMES = ("pdfs", "docxs", "pptxs", "transcripts")


def _build_syncer(tmp_path, monkeypatch):
    return mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)


SYNCER_BUILDERS = pytest.mark.parametrize("build_syncer", [_build_syncer])


# The API builds the syncer at import time, so construction must not depend on
# the material directories being mounted.
@SYNCER_BUILDERS
def test_syncer_can_be_created_without_source_directories(
    build_syncer, tmp_path, monkeypatch
):
    syncer = build_syncer(tmp_path, monkeypatch)

    assert not (tmp_path / "pdfs").exists()
    assert syncer.get_raw() == []


@SYNCER_BUILDERS
def test_require_source_directories_lists_every_missing_directory(
    build_syncer, tmp_path, monkeypatch
):
    (tmp_path / "pdfs").mkdir()
    syncer = build_syncer(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError) as error:
        syncer.require_source_directories()

    message = str(error.value)
    assert str(tmp_path / "docxs") in message
    assert str(tmp_path / "pptxs") in message
    assert str(tmp_path / "pdfs") not in message
    assert str(tmp_path / "transcripts") not in message


@SYNCER_BUILDERS
def test_require_source_directories_passes_when_all_exist(
    build_syncer, tmp_path, monkeypatch
):
    for name in SOURCE_DIRECTORY_NAMES:
        (tmp_path / name).mkdir(exist_ok=True)
    syncer = build_syncer(tmp_path, monkeypatch)

    assert syncer.require_source_directories() is None


def _document_entry(tmp_path):
    pdf_path = tmp_path / "221.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    return {"id": "221.pdf", "internal_pdf_path": str(pdf_path)}


def test_syncer_uses_scayle_ocr_model(tmp_path, monkeypatch):
    monkeypatch.delenv("MATHE_OCR_BACKEND", raising=False)

    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert syncer.ocr_model == mathe_syncer_module.OCR_LLM


# MATHE_OCR_BACKEND used to select Bedrock. A value left in a deployment's
# configuration must not route OCR away from SCAYLE.
@pytest.mark.parametrize("value", ["bedrock", " Bedrock ", "tesseract"])
def test_retired_ocr_backend_setting_still_ocrs_with_scayle(
    value, tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("MATHE_OCR_BACKEND", value)
    calls = []

    def fake_ocr_file(path, **kwargs):
        calls.append(kwargs["llm"])
        return "scayle text"

    monkeypatch.setattr(mathe_syncer_module, "ocr_file", fake_ocr_file)

    with caplog.at_level("WARNING", logger=mathe_syncer_module.__name__):
        syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    entry = _document_entry(tmp_path)
    syncer._process_document_entry(entry)

    assert "MATHE_OCR_BACKEND" in caplog.text
    assert "always uses SCAYLE" in caplog.text
    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == "scayle text"
    assert entry["ocr_model"] == mathe_syncer_module.OCR_LLM
    assert calls == [mathe_syncer_module.OCR_LLM]


def test_scayle_setting_for_retired_ocr_backend_does_not_warn(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("MATHE_OCR_BACKEND", "scayle")

    with caplog.at_level("WARNING", logger=mathe_syncer_module.__name__):
        mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert "MATHE_OCR_BACKEND" not in caplog.text


def test_scayle_document_ocr_skips_text_layer_and_uses_math_prompt(
    tmp_path, monkeypatch
):
    calls = []

    def fake_ocr_file(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return "$x^2$"

    monkeypatch.setattr(mathe_syncer_module, "ocr_file", fake_ocr_file)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    entry = _document_entry(tmp_path)

    syncer._process_document_entry(entry)

    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == "$x^2$"
    assert calls == [
        {
            "path": entry["internal_pdf_path"],
            "llm": mathe_syncer_module.OCR_LLM,
            "prefer_text_layer": False,
            "prompt": mathe_syncer_module.SCAYLE_OCR_PROMPT,
        }
    ]


def test_scayle_document_ocr_marks_empty_text_as_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(mathe_syncer_module, "ocr_file", lambda path, **kwargs: "  \n")
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    entry = _document_entry(tmp_path)

    syncer._process_document_entry(entry)

    assert entry["status"] == "failed"
    assert entry["claude_ocr_text"] == "OCR Failed: SCAYLE OCR returned empty text"


def _insert_pending_entry(syncer, entry_id, entry_type, pdf_path=None):
    with syncer.catalog.connection() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (id, type, internal_pdf_path, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (entry_id, entry_type, pdf_path),
        )


def _entries_by_id(syncer):
    return {entry["id"]: entry for entry in syncer.get_raw()}


# Rows processed before the column existed have an unrecorded OCR model, so the
# migration must add the column without guessing a value for them.
def test_migration_adds_ocr_model_column_without_backfilling(tmp_path, monkeypatch):
    db_path = tmp_path / "syncer.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(LEGACY_SYNC_SCHEMA)
        conn.execute(
            """
            INSERT INTO sync_entries (id, type, claude_ocr_text, status)
            VALUES ('221.pdf', 'document', 'old text', 'completed')
            """
        )

    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    entry = _entries_by_id(syncer)["221.pdf"]
    assert "ocr_model" in entry
    assert entry["ocr_model"] is None
    assert entry["claude_ocr_text"] == "old text"
    assert entry["status"] == "completed"

    # Re-opening an already migrated catalog must not fail or change the row.
    reopened = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    assert _entries_by_id(reopened)["221.pdf"] == entry


def test_completed_document_records_the_ocr_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mathe_syncer_module, "ocr_file", lambda path, **kwargs: "scayle text"
    )
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    pdf_path = _document_entry(tmp_path)["internal_pdf_path"]
    _insert_pending_entry(syncer, "221.pdf", "document", pdf_path)

    syncer.process_pending_materials()

    entry = _entries_by_id(syncer)["221.pdf"]
    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == "scayle text"
    assert entry["ocr_model"] == mathe_syncer_module.OCR_LLM


def test_failed_document_ocr_does_not_record_a_model(tmp_path, monkeypatch):
    def failing_ocr_file(path, **kwargs):
        raise RuntimeError("SCAYLE unavailable")

    monkeypatch.setattr(mathe_syncer_module, "ocr_file", failing_ocr_file)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    pdf_path = _document_entry(tmp_path)["internal_pdf_path"]
    _insert_pending_entry(syncer, "221.pdf", "document", pdf_path)

    syncer.process_pending_materials()

    entry = _entries_by_id(syncer)["221.pdf"]
    assert entry["status"] == "failed"
    assert entry["claude_ocr_text"] == "OCR Failed: SCAYLE unavailable"
    assert entry["ocr_model"] is None


def test_video_transcript_does_not_record_an_ocr_model(tmp_path, monkeypatch):
    def fake_process_video_entry(self, entry, whisper_model):
        entry["claude_ocr_text"] = "transcript"
        entry["status"] = "completed"

    monkeypatch.setattr(mathe_syncer_module, "WhisperModel", lambda *a, **k: object())
    monkeypatch.setattr(
        mathe_syncer_module.MathE_Syncer,
        "_process_video_entry",
        fake_process_video_entry,
    )
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    _insert_pending_entry(syncer, "abcdefghijk", "video")

    syncer.process_pending_materials()

    entry = _entries_by_id(syncer)["abcdefghijk"]
    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == "transcript"
    assert entry["ocr_model"] is None


def test_empty_catalog_table_lists_the_ocr_model_column(tmp_path):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert "ocr_model" in syncer.get().columns


def test_materials_sharing_a_video_are_all_kept_with_one_transcript(
    tmp_path,
    monkeypatch,
    caplog,
):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    (syncer._transcript_dir / "abcdefghijk.txt").write_text(
        "shared transcript",
        encoding="utf-8",
    )
    video_rows = [
        {
            "platform_material_id": 598,
            "link": "https://www.youtube.com/watch?v=abcdefghijk",
            "platform_type": 1,
        },
        {
            "platform_material_id": 657,
            "link": "https://youtu.be/abcdefghijk",
            "platform_type": 2,
        },
        {
            "platform_material_id": 700,
            "link": "Zyxwvutsrqp",
            "platform_type": 1,
        },
        {
            "platform_material_id": 812,
            "link": "https://example.com/not-youtube",
            "platform_type": 1,
        },
    ]

    class FakeMatheClient:
        def get_video_materials(self):
            return video_rows

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", FakeMatheClient)
    with caplog.at_level("WARNING", logger=mathe_syncer_module.__name__):
        syncer.reconcile_sources()

    # One catalog entry, and so one transcription, per YouTube video.
    entries = {entry["id"]: entry for entry in syncer.get_raw()}
    assert set(entries) == {"abcdefghijk", "Zyxwvutsrqp"}
    assert entries["abcdefghijk"]["platform_material_id"] == "598"

    # Every MathE material is linked to its video with its own subtype.
    assert syncer.catalog.list_video_material_links() == [
        {"platform_material_id": "700", "video_id": "Zyxwvutsrqp", "content_subtype": "video_lesson"},
        {"platform_material_id": "598", "video_id": "abcdefghijk", "content_subtype": "video_lesson"},
        {"platform_material_id": "657", "video_id": "abcdefghijk", "content_subtype": "video_review"},
    ]

    # Both materials of the shared video are indexed with its transcript;
    # the pending video is not yet.
    assert syncer.get_completed_materials() == [
        {"id": "abcdefghijk", "type": "video", "platform_material_id": "598", "claude_ocr_text": "shared transcript"},
        {"id": "abcdefghijk", "type": "video", "platform_material_id": "657", "claude_ocr_text": "shared transcript"},
    ]

    # A skipped link is reported, not dropped silently.
    assert any("812" in record.getMessage() for record in caplog.records)


def test_video_links_follow_the_registry(tmp_path, monkeypatch):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    video_rows = [
        {"platform_material_id": 598, "link": "abcdefghijk", "platform_type": 1},
        {"platform_material_id": 657, "link": "abcdefghijk", "platform_type": 2},
    ]

    class FakeMatheClient:
        def get_video_materials(self):
            return video_rows

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", FakeMatheClient)
    syncer.reconcile_sources()
    video_rows[:] = [video_rows[1]]
    syncer.reconcile_sources()

    assert syncer.catalog.list_video_material_links() == [
        {"platform_material_id": "657", "video_id": "abcdefghijk", "content_subtype": "video_review"},
    ]
    assert syncer.get_raw()[0]["platform_material_id"] == "657"


def test_failed_registry_lookup_keeps_video_links(tmp_path, monkeypatch):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    class FakeMatheClient:
        def get_video_materials(self):
            return [
                {"platform_material_id": 598, "link": "abcdefghijk", "platform_type": 1},
                {"platform_material_id": 657, "link": "abcdefghijk", "platform_type": 2},
            ]

        def close(self):
            pass

    class FailingMatheClient:
        def get_video_materials(self):
            raise RuntimeError("registry unavailable")

        def close(self):
            pass

    monkeypatch.setattr(syncer, "_get_mathe_client", FakeMatheClient)
    syncer.reconcile_sources()
    monkeypatch.setattr(syncer, "_get_mathe_client", FailingMatheClient)
    syncer.reconcile_sources()

    assert [
        link["platform_material_id"]
        for link in syncer.catalog.list_video_material_links()
    ] == ["598", "657"]


def test_completed_video_without_links_uses_its_own_material_id(tmp_path):
    # A catalog written before links existed has no rows in the link table
    # until the next reconciliation.
    catalog = mathe_syncer_module.MathESyncCatalog(tmp_path / "syncer.db")
    with catalog.connection() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, claude_ocr_text, status, platform_material_id
            ) VALUES ('abcdefghijk', 'video', 'transcript', 'completed', '598')
            """
        )

    assert catalog.list_completed_materials() == [
        {"id": "abcdefghijk", "type": "video", "platform_material_id": "598", "claude_ocr_text": "transcript"},
    ]


class _RegistryClient:
    """A MathE registry with no videos and the given document rows."""

    def __init__(self, documents):
        self.documents = documents

    def get_video_materials(self):
        return []

    def get_document_materials(self):
        if isinstance(self.documents, Exception):
            raise self.documents
        return self.documents

    def close(self):
        pass


def test_registry_documents_without_a_source_file_are_recorded(
    tmp_path,
    monkeypatch,
    caplog,
):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "221.pdf").write_bytes(b"present")
    # On disk but not in the registry: still indexed, and reported.
    (pdf_dir / "230.pdf").write_bytes(b"unlisted")

    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    registry = _RegistryClient(
        [
            {"platform_material_id": 221, "file_ext": "pdf"},
            {"platform_material_id": 222, "file_ext": "pdf"},
            {"platform_material_id": 223, "file_ext": "docx"},
            {"platform_material_id": 224, "file_ext": "doc"},
            {"platform_material_id": 225, "file_ext": ""},
        ]
    )
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)

    with caplog.at_level("WARNING", logger=mathe_syncer_module.__name__):
        syncer.reconcile_sources()

    assert syncer.get_document_registry_gaps() == [
        {"platform_material_id": "222", "file_ext": "pdf", "reason": "missing_source_file"},
        {"platform_material_id": "223", "file_ext": "docx", "reason": "missing_source_file"},
        {"platform_material_id": "224", "file_ext": "doc", "reason": "unsupported_extension"},
        {"platform_material_id": "225", "file_ext": "", "reason": "unsupported_extension"},
        {"platform_material_id": "230", "file_ext": "pdf", "reason": "not_in_registry"},
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("missing_source_file" in m and "222, 223" in m for m in messages)
    assert any("unsupported_extension" in m and "224, 225" in m for m in messages)
    assert any("not_in_registry" in m and "230" in m for m in messages)
    # Discovery itself is unchanged.
    assert {entry["id"] for entry in syncer.get_raw()} == {"221.pdf", "230.pdf"}


def test_document_gaps_clear_once_the_file_is_mounted(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    registry = _RegistryClient([{"platform_material_id": 222, "file_ext": "pdf"}])
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)

    syncer.reconcile_sources()
    assert [gap["platform_material_id"] for gap in syncer.get_document_registry_gaps()] == ["222"]

    (pdf_dir / "222.pdf").write_bytes(b"now mounted")
    syncer.reconcile_sources()
    assert syncer.get_document_registry_gaps() == []


def test_failed_document_registry_lookup_keeps_previous_gaps(tmp_path, monkeypatch):
    (tmp_path / "pdfs").mkdir()
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    registry = _RegistryClient([{"platform_material_id": 222, "file_ext": "pdf"}])
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)
    syncer.reconcile_sources()

    registry.documents = RuntimeError("registry unavailable")
    syncer.reconcile_sources()

    assert [gap["platform_material_id"] for gap in syncer.get_document_registry_gaps()] == ["222"]


def test_catalogued_legacy_document_is_not_reported_as_a_gap(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "221.pdf").write_bytes(b"present")
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer.catalog.connection() as conn:
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, internal_pdf_path, claude_ocr_text, status,
                platform_material_id, content_subtype
            ) VALUES ('./221.pdf', 'document', './221.pdf', 'text', 'completed', '221', 'pdf')
            """
        )
    registry = _RegistryClient([{"platform_material_id": 221, "file_ext": "pdf"}])
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)

    syncer.reconcile_sources()

    assert syncer.get_document_registry_gaps() == []


def test_unregistered_material_in_several_formats_is_reported_once(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "271.pdf").write_bytes(b"pdf copy")
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer.catalog.connection() as conn:
        # A converted DOCX of the same material already in the catalog.
        conn.execute(
            """
            INSERT INTO sync_entries (
                id, type, internal_pdf_path, status,
                platform_material_id, content_subtype
            ) VALUES ('271.docx', 'document', '/tmp/271_docx.pdf', 'pending', '271', 'docx')
            """
        )
    registry = _RegistryClient([{"platform_material_id": 221, "file_ext": "pdf"}])
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)

    syncer.reconcile_sources()

    assert syncer.get_document_registry_gaps() == [
        {"platform_material_id": "221", "file_ext": "pdf", "reason": "missing_source_file"},
        {"platform_material_id": "271", "file_ext": "docx, pdf", "reason": "not_in_registry"},
    ]


def test_unregistered_file_stops_being_reported_once_registered(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "271.pdf").write_bytes(b"present")
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    registry = _RegistryClient([])
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)

    syncer.reconcile_sources()
    assert [gap["reason"] for gap in syncer.get_document_registry_gaps()] == ["not_in_registry"]

    registry.documents = [{"platform_material_id": 271, "file_ext": "pdf"}]
    syncer.reconcile_sources()
    assert syncer.get_document_registry_gaps() == []


@pytest.mark.parametrize(
    "link",
    [
        "abcdefghijk",
        " abcdefghijk ",
        "https://www.youtube.com/watch?v=abcdefghijk",
        "https://youtube.com/watch?feature=share&v=abcdefghijk&t=10",
        "https://m.youtube.com/watch?app=desktop&amp;v=abcdefghijk",
        "https://music.youtube.com/watch?v=abcdefghijk&list=x",
        "HTTPS://WWW.YOUTUBE.COM/WATCH?V=abcdefghijk",
        "https://youtu.be/abcdefghijk?si=xyz",
        "youtu.be/abcdefghijk",
        "www.youtube.com/v/abcdefghijk",
        "https://www.youtube.com/shorts/abcdefghijk",
        "https://www.youtube.com/live/abcdefghijk?feature=share",
        "https://www.youtube-nocookie.com/embed/abcdefghijk?rel=0",
        '<iframe src="https://www.youtube.com/embed/abcdefghijk" frameborder="0">',
    ],
)
def test_youtube_link_forms_resolve_to_the_video_id(tmp_path, monkeypatch, link):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert syncer._is_youtube_asset(link)
    assert syncer._extract_video_id(link) == "abcdefghijk"


@pytest.mark.parametrize(
    "link",
    [
        "https://www.youtube.com/watch?v=abcdefghijkl",
        "https://www.youtube.com/playlist?list=PLabcdefghijk",
        "https://www.youtube.com/channel/UCabcdefghij",
        "https://vimeo.com/123456789",
        "",
        # Incomplete values stored in MathE records.
        "Test",
        "Og9zGParQ",
        "0&p3e:27170",
        "efficients/",
        "duction",
        "ZZGA&t=521s",
        "abcdefghijkl&t",
        "a watch?v=abcdefghijk",
    ],
)
def test_links_naming_no_youtube_video_are_rejected(tmp_path, monkeypatch, link):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    if syncer._is_youtube_asset(link):
        with pytest.raises(ValueError):
            syncer._extract_video_id(link)


class _VideoRegistryClient:
    def __init__(self, videos):
        self.videos = videos

    def get_video_materials(self):
        if isinstance(self.videos, Exception):
            raise self.videos
        return self.videos

    def get_document_materials(self):
        return []

    def close(self):
        pass


def test_unindexed_video_materials_are_reported_with_a_reason(
    tmp_path, monkeypatch
):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    (syncer._transcript_dir / "doneVideo01.txt").write_text("transcript", encoding="utf-8")
    registry = _VideoRegistryClient(
        [
            {"platform_material_id": 487, "link": "https://vimeo.com/123", "platform_type": 1},
            {"platform_material_id": 515, "link": "removedVid1", "platform_type": 1},
            {"platform_material_id": 609, "link": "pendingVid1", "platform_type": 1},
            {"platform_material_id": 700, "link": "blockedVid1", "platform_type": 2},
            {"platform_material_id": 719, "link": "https://youtu.be/removedVid1", "platform_type": 2},
            {"platform_material_id": 800, "link": "doneVideo01", "platform_type": 1},
        ]
    )
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)
    syncer.reconcile_sources()
    syncer.catalog.update_processing_result(
        entry_id="removedVid1",
        status="failed",
        text=(
            "Transcription Failed: ERROR: [youtube] removedVid1: Video "
            "unavailable. This video has been removed by the uploader"
        ),
    )
    syncer.catalog.update_processing_result(
        entry_id="blockedVid1",
        status="failed",
        text="Transcription Failed: Sign in to confirm you're not a bot",
    )

    assert syncer.get_video_registry_gaps() == [
        {"platform_material_id": "487", "video_id": None, "link": "https://vimeo.com/123", "reason": "unrecognised_link"},
        {"platform_material_id": "515", "video_id": "removedVid1", "link": "removedVid1", "reason": "video_unavailable"},
        {"platform_material_id": "609", "video_id": "pendingVid1", "link": "pendingVid1", "reason": "pending"},
        {"platform_material_id": "700", "video_id": "blockedVid1", "link": "blockedVid1", "reason": "transcription_failed"},
        # Shares the removed video of 515.
        {"platform_material_id": "719", "video_id": "removedVid1", "link": "removedVid1", "reason": "video_unavailable"},
    ]


def test_link_gaps_follow_the_registry_and_survive_a_failed_lookup(
    tmp_path, monkeypatch
):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    registry = _VideoRegistryClient(
        [{"platform_material_id": 705, "link": "https://example.com/video", "platform_type": 1}]
    )
    monkeypatch.setattr(syncer, "_get_mathe_client", lambda: registry)
    syncer.reconcile_sources()
    assert [gap["platform_material_id"] for gap in syncer.get_video_registry_gaps()] == ["705"]

    registry.videos = RuntimeError("registry unavailable")
    syncer.reconcile_sources()
    assert [gap["platform_material_id"] for gap in syncer.get_video_registry_gaps()] == ["705"]

    registry.videos = [
        {"platform_material_id": 705, "link": "https://www.youtube.com/shorts/fixedVideo1", "platform_type": 1}
    ]
    syncer.reconcile_sources()
    assert syncer.get_video_registry_gaps() == [
        {"platform_material_id": "705", "video_id": "fixedVideo1", "link": "https://www.youtube.com/shorts/fixedVideo1", "reason": "pending"},
    ]


@pytest.mark.parametrize(
    ("link", "video_id"),
    [
        # Values stored in MathE records without the YouTube domain.
        ("odkaPgWPQGo&t", "odkaPgWPQGo"),
        ("watch?v=KQ5tyLxCgb8", "KQ5tyLxCgb8"),
        ("KQ5tyLxCgb8?t=10", "KQ5tyLxCgb8"),
        ("/watch?feature=share&v=KQ5tyLxCgb8&t=5s", "KQ5tyLxCgb8"),
    ],
)
def test_links_without_the_youtube_domain_resolve(tmp_path, monkeypatch, link, video_id):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert syncer._is_youtube_asset(link)
    assert syncer._extract_video_id(link) == video_id
