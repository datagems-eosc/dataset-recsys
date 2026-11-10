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


SOURCE_DIRECTORY_NAMES = ("pdfs", "docxs", "pptxs", "transcripts")


def _build_syncer(ocr_backend):
    def build(tmp_path, monkeypatch):
        _disable_bedrock(monkeypatch)
        return mathe_syncer_module.MathE_Syncer(
            base_dir=tmp_path,
            ocr_backend=ocr_backend,
        )

    return build


SYNCER_BUILDERS = pytest.mark.parametrize(
    "build_syncer",
    [_build_syncer("bedrock"), _build_syncer("scayle")],
    ids=["bedrock", "scayle"],
)


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


def _fail_if_bedrock_client_is_created(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("The Bedrock client must not be created")

    monkeypatch.setattr(mathe_syncer_module.boto3, "client", fail)


def _document_entry(tmp_path):
    pdf_path = tmp_path / "221.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    return {"id": "221.pdf", "internal_pdf_path": str(pdf_path)}


def test_syncer_uses_scayle_ocr_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MATHE_OCR_BACKEND", raising=False)
    _fail_if_bedrock_client_is_created(monkeypatch)

    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert syncer.ocr_backend == "scayle"
    assert syncer.ocr_model == mathe_syncer_module.OCR_LLM
    assert syncer.bedrock is None


def test_syncer_reads_ocr_backend_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MATHE_OCR_BACKEND", " Bedrock ")
    _disable_bedrock(monkeypatch)

    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    assert syncer.ocr_backend == "bedrock"
    assert syncer.ocr_model == mathe_syncer_module.BEDROCK_OCR_MODEL_ID
    assert syncer.bedrock is not None


def test_ocr_backend_argument_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MATHE_OCR_BACKEND", "bedrock")
    _fail_if_bedrock_client_is_created(monkeypatch)

    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="scayle")

    assert syncer.ocr_backend == "scayle"


def test_syncer_rejects_unknown_ocr_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MATHE_OCR_BACKEND", "tesseract")

    with pytest.raises(ValueError, match="Unsupported MathE OCR backend 'tesseract'"):
        mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)


def test_scayle_document_ocr_skips_text_layer_and_uses_math_prompt(
    tmp_path, monkeypatch
):
    calls = []

    def fake_ocr_file(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return "$x^2$"

    monkeypatch.setattr(mathe_syncer_module, "ocr_file", fake_ocr_file)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="scayle")
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
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="scayle")
    entry = _document_entry(tmp_path)

    syncer._process_document_entry(entry)

    assert entry["status"] == "failed"
    assert entry["claude_ocr_text"] == "OCR Failed: SCAYLE OCR returned empty text"


def test_bedrock_document_ocr_sends_whole_pdf_to_configured_model(
    tmp_path, monkeypatch
):
    requests = []

    class FakeBedrock:
        def converse(self, **kwargs):
            requests.append(kwargs)
            return {"output": {"message": {"content": [{"text": "$y^2$"}]}}}

    monkeypatch.setattr(
        mathe_syncer_module.boto3,
        "client",
        lambda *args, **kwargs: FakeBedrock(),
    )
    monkeypatch.setattr(
        mathe_syncer_module,
        "ocr_file",
        lambda *args, **kwargs: pytest.fail("SCAYLE must not be called"),
    )
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="bedrock")
    entry = _document_entry(tmp_path)

    syncer._process_document_entry(entry)

    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == "$y^2$"
    assert len(requests) == 1
    assert requests[0]["modelId"] == mathe_syncer_module.BEDROCK_OCR_MODEL_ID
    assert requests[0]["inferenceConfig"] == {"temperature": 0.0}
    content = requests[0]["messages"][0]["content"]
    assert content[0]["document"]["source"]["bytes"] == b"%PDF-1.4 test"
    assert content[1] == {"text": mathe_syncer_module.BEDROCK_OCR_PROMPT}


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

    _disable_bedrock(monkeypatch)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)

    entry = _entries_by_id(syncer)["221.pdf"]
    assert "ocr_model" in entry
    assert entry["ocr_model"] is None
    assert entry["claude_ocr_text"] == "old text"
    assert entry["status"] == "completed"

    # Re-opening an already migrated catalog must not fail or change the row.
    reopened = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    assert _entries_by_id(reopened)["221.pdf"] == entry


@pytest.mark.parametrize(
    ("ocr_backend", "expected_model"),
    [
        ("scayle", mathe_syncer_module.OCR_LLM),
        ("bedrock", mathe_syncer_module.BEDROCK_OCR_MODEL_ID),
    ],
)
def test_completed_document_records_the_ocr_model(
    ocr_backend, expected_model, tmp_path, monkeypatch
):
    class FakeBedrock:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "bedrock text"}]}}}

    monkeypatch.setattr(
        mathe_syncer_module.boto3, "client", lambda *args, **kwargs: FakeBedrock()
    )
    monkeypatch.setattr(
        mathe_syncer_module, "ocr_file", lambda path, **kwargs: "scayle text"
    )
    syncer = mathe_syncer_module.MathE_Syncer(
        base_dir=tmp_path, ocr_backend=ocr_backend
    )
    pdf_path = _document_entry(tmp_path)["internal_pdf_path"]
    _insert_pending_entry(syncer, "221.pdf", "document", pdf_path)

    syncer.process_pending_materials()

    entry = _entries_by_id(syncer)["221.pdf"]
    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == f"{ocr_backend} text"
    assert entry["ocr_model"] == expected_model


def test_failed_document_ocr_does_not_record_a_model(tmp_path, monkeypatch):
    def failing_ocr_file(path, **kwargs):
        raise RuntimeError("SCAYLE unavailable")

    monkeypatch.setattr(mathe_syncer_module, "ocr_file", failing_ocr_file)
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="scayle")
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
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="scayle")
    _insert_pending_entry(syncer, "abcdefghijk", "video")

    syncer.process_pending_materials()

    entry = _entries_by_id(syncer)["abcdefghijk"]
    assert entry["status"] == "completed"
    assert entry["claude_ocr_text"] == "transcript"
    assert entry["ocr_model"] is None


def test_empty_catalog_table_lists_the_ocr_model_column(tmp_path):
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path, ocr_backend="scayle")

    assert "ocr_model" in syncer.get().columns
