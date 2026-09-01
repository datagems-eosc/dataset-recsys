"""Tests for the MathE synchronization and index publication pipeline."""

import sys
import sqlite3
import types
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks

from dataset_recsys.workflows import mathe_sync_pipeline
from dataset_recsys.api.routes import mathe

import dataset_recsys.utils.mathe_syncer as mathe_syncer_module

class FakeSyncer:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.sync_called = False
        self.status = {}

    def sync_and_process(self):
        self.sync_called = True

    def get_sync_status(self):
        return dict(self.status)

    def save_sync_status(self, status):
        self.status = dict(status)

    def _get_sqlite_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self) -> pd.DataFrame:
        """Mirrors the production MathE_Syncer.get() method using SQLite."""
        with self._get_sqlite_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM sync_entries", conn)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "id",
                    "source_type",
                    "claude_ocr_text",
                    "status",
                    "material_id",
                    "pdf_path",
                ]
            )

        df["material_id"] = df["id"]
        df["source_type"] = df["id"].apply(lambda p: Path(p).suffix.lstrip('.').lower())
        df["pdf_path"] = df["internal_pdf_path"]

        return df.replace("", pd.NA)

class FakeRecommendationClient:
    stored = []

    def store_recommendations(self, application, data):
        self.__class__.stored.append({"application": application, "data": data})
        return len(data)


class FakeEmbeddingClient:
    TABLE_MATHE = "mathe_embeddings"
    stored = []
    deleted = []

    def store_embeddings(self, **kwargs):
        self.__class__.stored.append(kwargs)
        return len(kwargs["dataset_ids"])

    def delete_application(self, application, table):
        self.__class__.deleted.append({"application": application, "table": table})
        return 0


def test_run_mathe_pipeline_builds_separate_document_and_video_indexes(
    tmp_path,
    monkeypatch,
):
    # 1. Establish path for SQLite database instead of JSON
    db_path = tmp_path / "syncer.db"

    # 2. Build the database schema and populate it with mock rows
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE sync_entries (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source_value TEXT,
            internal_pdf_path TEXT,
            claude_ocr_text TEXT,
            status TEXT NOT NULL,
            platform_material_id TEXT,
            content_subtype TEXT
        )
    """)

    mock_rows = [
        (
            "./6.pdf",
            "document",
            None,
            "./6.pdf",
            "linear algebra matrices",
            "completed",
            "6",
            "pdf",
        ),
        (
            "./7.docx",
            "document",
            None,
            "/tmp/7_docx.pdf",
            "calculus derivatives",
            "completed",
            "7",
            "docx",
        ),
        (
            "abcdefghijk",
            "video",
            "https://www.youtube.com/watch?v=abcdefghijk",
            None,
            "lesson transcript",
            "completed",
            "901",
            "video_lesson",
        ),
        (
            "zyxwvutsrqp",
            "video",
            "https://www.youtube.com/watch?v=zyxwvutsrqp",
            None,
            "review transcript",
            "completed",
            "902",
            "video_review",
        ),
        (
            "./8.pdf",
            "document",
            None,
            "./8.pdf",
            "",
            "completed",
            "8",
            "pdf",
        ),
        (
            "./10.pdf",
            "document",
            None,
            "./10.pdf",
            "not ready",
            "pending",
            "10",
            "pdf",
        ),
    ]

    conn.executemany(
        "INSERT INTO sync_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        mock_rows
    )
    conn.commit()
    conn.close()

    # Initialize our SQLite-based FakeSyncer
    fake_syncer = FakeSyncer(db_path)

    encoded_with = {}

    def fake_encode_texts(texts, model_name):
        encoded_with["texts"] = texts
        encoded_with["model_name"] = model_name
        return np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]],
            dtype=float,
        )

    monkeypatch.setattr(mathe_sync_pipeline, "encode_texts", fake_encode_texts)
    monkeypatch.setattr(
        mathe_sync_pipeline,
        "_build_recommendation_client",
        lambda: FakeRecommendationClient(),
    )
    monkeypatch.setattr(
        mathe_sync_pipeline,
        "_build_embedding_client",
        lambda: FakeEmbeddingClient(),
    )

    FakeRecommendationClient.stored = []
    FakeEmbeddingClient.stored = []
    FakeEmbeddingClient.deleted = []
    summary = mathe_sync_pipeline.run_mathe_pipeline(fake_syncer)

    assert fake_syncer.sync_called is True
    assert summary == {
        "status": "completed",
        "processed_materials": 4,
        "embeddings_created": 4,
        "redis_keys_updated": 8,
        "application": "mathe",
        "collections": {
            "mathe": {
                "processed_materials": 4,
                "embeddings_stored": 4,
                "redis_keys_updated": 4,
            },
            "mathe_documents": {
                "processed_materials": 2,
                "embeddings_stored": 2,
                "redis_keys_updated": 2,
            },
            "mathe_videos": {
                "processed_materials": 2,
                "embeddings_stored": 2,
                "redis_keys_updated": 2,
            },
        },
    }
    assert fake_syncer.status["sync_status"] == "completed"
    assert fake_syncer.status["embeddings_created"] == 4
    assert encoded_with["model_name"] == "BAAI/bge-m3"
    assert encoded_with["texts"] == [
        "linear algebra matrices",
        "calculus derivatives",
        "lesson transcript",
        "review transcript",
    ]

    redis_by_application = {
        str(call["application"]): call["data"]
        for call in FakeRecommendationClient.stored
    }
    assert set(redis_by_application) == {
        "mathe",
        "mathe_documents",
        "mathe_videos",
    }
    assert set(redis_by_application["mathe"]) == {
        "6.pdf",
        "7.docx",
        "abcdefghijk.txt",
        "zyxwvutsrqp.txt",
    }
    assert set(redis_by_application["mathe_documents"]) == {"6", "7"}
    assert set(redis_by_application["mathe_videos"]) == {"901", "902"}
    assert {
        neighbor_id
        for neighbors in redis_by_application["mathe_documents"].values()
        for neighbor_id, _score in neighbors
    } <= {"6", "7"}
    assert {
        neighbor_id
        for neighbors in redis_by_application["mathe_videos"].values()
        for neighbor_id, _score in neighbors
    } <= {"901", "902"}

    embeddings_by_application = {
        str(call["application"]): call
        for call in FakeEmbeddingClient.stored
    }
    assert embeddings_by_application["mathe"]["dataset_ids"] == [
        "6.pdf",
        "7.docx",
        "abcdefghijk.txt",
        "zyxwvutsrqp.txt",
    ]
    assert embeddings_by_application["mathe_documents"]["dataset_ids"] == ["6", "7"]
    assert embeddings_by_application["mathe_videos"]["dataset_ids"] == ["901", "902"]
    assert all(
        call["table"] == "mathe_embeddings"
        for call in FakeEmbeddingClient.stored
    )
    assert FakeEmbeddingClient.deleted == []


def test_split_indexes_skip_unresolved_platform_ids_and_clear_empty_collection():
    materials = [
        {
            "sync_entry_id": "6.pdf",
            "platform_material_id": "6",
            "type": "document",
            "text": "document text",
        },
        {
            "sync_entry_id": "abcdefghijk",
            "platform_material_id": None,
            "type": "video",
            "text": "legacy transcript",
        },
    ]
    indices = mathe_sync_pipeline._build_collection_indices(materials)

    assert set(indices) == {
        mathe_sync_pipeline.MatheApplication.DOCUMENTS,
        mathe_sync_pipeline.MatheApplication.VIDEOS,
    }
    assert set(indices[mathe_sync_pipeline.MatheApplication.DOCUMENTS]) == {"6"}
    assert indices[mathe_sync_pipeline.MatheApplication.VIDEOS] == {}

    embedding_client = FakeEmbeddingClient()
    recommendation_client = FakeRecommendationClient()
    FakeEmbeddingClient.deleted = []
    FakeRecommendationClient.stored = []
    result = (
        mathe_sync_pipeline._replace_embedding_and_recommendation_collection(
            application=mathe_sync_pipeline.MatheApplication.VIDEOS,
            entity_indices={},
            materials=materials,
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
            embedding_client=embedding_client,
            recommendation_client=recommendation_client,
            run_id="stage-2-test",
        )
    )

    assert result == {
        "processed_materials": 0,
        "embeddings_stored": 0,
        "redis_keys_updated": 0,
    }
    assert FakeEmbeddingClient.deleted == [
        {"application": "mathe_videos", "table": "mathe_embeddings"}
    ]
    assert FakeRecommendationClient.stored == [
        {"application": "mathe_videos", "data": {}}
    ]


# The pipeline should skip recommendation storage when SQLite has no completed materials.
def test_run_mathe_pipeline_handles_missing_sync_database(tmp_path):
    fake_syncer = FakeSyncer(tmp_path / "missing.db")

    summary = mathe_sync_pipeline.run_mathe_pipeline(fake_syncer)

    assert fake_syncer.sync_called is True
    assert summary["status"] == "skipped"
    assert summary["processed_materials"] == 0
    assert summary["redis_keys_updated"] == 0
    assert fake_syncer.status["sync_status"] == "skipped"
    assert fake_syncer.status["embeddings_created"] == 0


# This test checks that the MathE /sync API endpoint is wired correctly.
# It verifies that when the endpoint is hit,
# it triggers the background task to run the sync pipeline,
# and returns the expected response.
def test_sync_endpoint_schedules_mathe_pipeline(monkeypatch):

    fake_logger = types.SimpleNamespace(
        bind=lambda **kwargs: fake_logger,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "structlog",
        types.SimpleNamespace(get_logger=lambda *args, **kwargs: fake_logger),
    )
    monkeypatch.setitem(
        sys.modules,
        "redis",
        types.SimpleNamespace(Redis=lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        mathe_syncer_module.boto3,
        "client",
        lambda *args, **kwargs: object(),
    )

    background_tasks = BackgroundTasks()
    response = asyncio.run(mathe.sync_data(background_tasks))

    assert response["status"] == "Accepted"
    assert background_tasks.tasks
    assert background_tasks.tasks[0].func is mathe.run_mathe_pipeline


def test_document_recommend_endpoint_uses_curricular_pool_ranker(monkeypatch):
    calls = {}
    fake_mathe_client = object()
    fake_embedding_client = object()

    async def fake_authorized_entity_ids(token):
        calls["token"] = token
        return {mathe.MATHE_DATASET_ID: "dg_ds-browse"}

    def fake_recommend_from_curricular_pool(**kwargs):
        calls["recommender"] = kwargs
        return ["818", "900"]

    monkeypatch.setattr(
        mathe.security,
        "get_authorized_entity_ids",
        fake_authorized_entity_ids,
    )
    monkeypatch.setattr(mathe, "get_mathe_client", lambda: fake_mathe_client)
    monkeypatch.setattr(mathe, "get_embedding_client", lambda: fake_embedding_client)
    monkeypatch.setattr(
        mathe,
        "recommend_from_curricular_pool",
        fake_recommend_from_curricular_pool,
    )

    response = asyncio.run(
        mathe.get_document_recommendations(
            request=mathe.MatheRecsRequest(
                question_id="272",
                question="Differentiate $y = (2x^3 - 5x)^5$.",
                n=2,
            ),
            claims={"sub": "user-1"},
            token="token-1",
        )
    )

    assert calls["token"] == "token-1"
    assert calls["recommender"] == {
        "question_id": 272,
        "question": "Differentiate $y = (2x^3 - 5x)^5$.",
        "k": 2,
        "mathe_mirror_client": fake_mathe_client,
        "embedding_client": fake_embedding_client,
    }
    assert response.question_id == "272"
    assert [rec.material_id for rec in response.recommendations] == ["818", "900"]


def test_legacy_and_explicit_document_routes_share_one_handler():
    routes_by_path = {route.path: route for route in mathe.router.routes}

    legacy_route = routes_by_path["/dataset-recsys/mathe/recommend"]
    document_route = routes_by_path["/dataset-recsys/mathe/recommend/documents"]

    assert legacy_route.endpoint is mathe.get_document_recommendations
    assert document_route.endpoint is mathe.get_document_recommendations
    assert legacy_route.deprecated is True
    assert document_route.deprecated is not True

# This test verifies that the /status endpoint of the MathE API correctly returns the synchronization metadata,
# including the status of the sync, timestamps, and counts of materials in different OCR states.
def test_status_endpoint_returns_sync_metadata_and_failed_material_ids(monkeypatch):
    fake_syncer = types.SimpleNamespace(
        is_running=False,
        get_sync_status=lambda: {
            "sync_status": "completed",
            "last_sync_started_at": "2026-05-13T10:24:18Z",
            "last_sync_completed_at": "2026-05-13T10:31:42Z",
            "embeddings_created": 461,
        },
        get=lambda: pd.DataFrame(
            [
                {"status": "completed", "material_id": "6.pdf"},
                {"status": "completed", "material_id": "7.pdf"},
                {"status": "pending", "material_id": "8.pdf"},
                {"status": "failed", "material_id": "70.pdf"},
            ]
        ),
    )
    monkeypatch.setattr(mathe, "syncer", fake_syncer)

    response = asyncio.run(mathe.get_status())

    assert response["sync_status"] == "completed"
    assert response["total_materials"] == 4
    assert response["ocr_completed_materials"] == 2
    assert response["ocr_pending_materials"] == 1
    assert response["ocr_failed_material_ids"] == ["70.pdf"]
    assert response["embeddings_created"] == 461

# It simulates a scenario where the last sync was marked as "running" but the heartbeat timestamp is old,
# indicating that the sync process may have stalled.
def test_status_endpoint_marks_old_running_sync_as_stale(monkeypatch):
    fake_syncer = types.SimpleNamespace(
        is_running=False,
        get_sync_status=lambda: {
            "sync_status": "running",
            "last_sync_started_at": "2000-01-01T10:24:18Z",
            "last_sync_heartbeat_at": "2000-01-01T10:25:18Z",
            "last_sync_completed_at": None,
            "embeddings_created": 12,
        },
        get=lambda: pd.DataFrame(
            [{"status": "pending", "material_id": "8.pdf"}]
        ),
    )
    monkeypatch.setattr(mathe, "syncer", fake_syncer)

    response = asyncio.run(mathe.get_status())

    assert response["sync_status"] == "stale"
    assert response["last_sync_heartbeat_at"] == "2000-01-01T10:25:18Z"
    assert response["ocr_pending_materials"] == 1
    assert response["embeddings_created"] == 12

# Completed and failed entries are terminal states for the current batch processor.
def test_batch_processing_skips_completed_and_failed_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mathe_syncer_module.boto3,
        "client",
        lambda *args, **kwargs: object(),
    )
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    with syncer._get_sqlite_conn() as conn:
        conn.executemany(
            """
            INSERT INTO sync_entries (
                id, type, internal_pdf_path, claude_ocr_text, status
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("6.pdf", "document", "6.pdf", "already done", "completed"),
                ("8.pdf", "document", "8.pdf", "OCR Failed: timeout", "failed"),
                ("9.pdf", "document", "9.pdf", None, "pending"),
            ],
        )
        conn.commit()

    processed = []

    def fake_process_document_entry(entry):
        processed.append(entry["id"])
        entry["claude_ocr_text"] = "new OCR text"
        entry["status"] = "completed"

    monkeypatch.setattr(syncer, "_process_document_entry", fake_process_document_entry)

    syncer.run_hybrid_batch_processing()

    assert processed == ["9.pdf"]
    with syncer._get_sqlite_conn() as conn:
        rows = {
            row["id"]: dict(row)
            for row in conn.execute(
                "SELECT id, status, claude_ocr_text FROM sync_entries"
            ).fetchall()
        }

    assert rows["6.pdf"]["claude_ocr_text"] == "already done"
    assert rows["8.pdf"]["status"] == "failed"
    assert rows["9.pdf"]["status"] == "completed"
    assert rows["9.pdf"]["claude_ocr_text"] == "new OCR text"


def test_batch_processing_routes_video_type_to_transcription(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mathe_syncer_module.boto3,
        "client",
        lambda *args, **kwargs: object(),
    )
    whisper_model = object()
    monkeypatch.setattr(
        mathe_syncer_module,
        "WhisperModel",
        lambda *args, **kwargs: whisper_model,
    )
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
                "https://www.youtube.com/watch?v=abcdefghijk",
                None,
                "pending",
            ),
        )
        conn.commit()

    processed = []

    def fake_process_video_entry(entry, model):
        processed.append((entry["id"], model))
        entry["claude_ocr_text"] = "new transcript"
        entry["status"] = "completed"

    monkeypatch.setattr(syncer, "_process_video_entry", fake_process_video_entry)
    syncer.run_hybrid_batch_processing()

    assert processed == [("abcdefghijk", whisper_model)]
    assert syncer.get_raw()[0]["type"] == "video"
    assert syncer.get_raw()[0]["claude_ocr_text"] == "new transcript"
