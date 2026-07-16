import json
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
    def __init__(self, json_file: Path):
        self.json_file = json_file
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
                columns=["id", "source_type", "claude_ocr_text", "status", "material_id", "pdf_path"]
            )
            
        df["material_id"] = df["id"]
        df["source_type"] = df["id"].apply(lambda p: Path(p).suffix.lstrip('.').lower())
        df["pdf_path"] = df["internal_pdf_path"]
        
        return df.replace("", pd.NA)

class FakeRecommendationClient:
    stored = None

    def store_recommendations(self, application, data):
        self.__class__.stored = {"application": application, "data": data}
        return len(data)


class FakeEmbeddingClient:
    TABLE_MATHE = "mathe_embeddings"
    stored = None

    def store_embeddings(self, **kwargs):
        self.__class__.stored = kwargs
        return len(kwargs["dataset_ids"])


def test_run_mathe_pipeline_builds_and_stores_recommendations(tmp_path, monkeypatch):
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
            status TEXT NOT NULL
        )
    """)
    
    mock_rows = [
        ("./6.pdf", "document", None, "./6.pdf", "linear algebra matrices", "completed"),
        ("./7.pdf", "document", None, "./7.pdf", "calculus derivatives", "completed"),
        ("./8.pdf", "document", None, "./8.pdf", "geometry triangles", "completed"),
        ("./9.pdf", "document", None, "./9.pdf", "", "completed"),
        ("./10.pdf", "document", None, "./10.pdf", "not ready", "pending")
    ]
    
    conn.executemany(
        "INSERT INTO sync_entries VALUES (?, ?, ?, ?, ?, ?)", 
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
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]],
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

    summary = mathe_sync_pipeline.run_mathe_pipeline(fake_syncer)

    assert fake_syncer.sync_called is True
    assert summary == {
        "status": "completed",
        "processed_materials": 3,
        "embeddings_created": 3,
        "redis_keys_updated": 3,
        "application": "mathe",
    }
    assert fake_syncer.status["sync_status"] == "completed"
    assert fake_syncer.status["embeddings_created"] == 3
    assert encoded_with["model_name"] == "BAAI/bge-m3"
    assert FakeRecommendationClient.stored["application"] == "mathe"
    assert set(FakeRecommendationClient.stored["data"]) == {"6.pdf", "7.pdf", "8.pdf"}
    assert FakeEmbeddingClient.stored["application"] == "mathe"
    assert FakeEmbeddingClient.stored["dataset_ids"] == ["6.pdf", "7.pdf", "8.pdf"]
    assert FakeEmbeddingClient.stored["table"] == "mathe_embeddings"


# This test ensures that if the data.json file is missing, 
# the pipeline handles it gracefully without attempting to process or store recommendations.
def test_run_mathe_pipeline_handles_missing_data_json(tmp_path):
    fake_syncer = FakeSyncer(tmp_path / "missing.json")

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


def test_recommend_endpoint_uses_curricular_pool_ranker(monkeypatch):
    calls = {}
    fake_mathe_client = object()
    fake_embedding_client = object()

    async def fake_authorized_entity_ids(token):
        calls["token"] = token
        return [mathe.MATHE_DATASET_ID]

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
        mathe.get_recommendations(
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

# This test ensures that the OCR step of the pipeline correctly identifies materials 
# that already have valid OCR text and skips reprocessing them, 
# while still processing materials that are pending or previously failed.
def test_ocr_skips_materials_that_already_have_text(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mathe_syncer_module.boto3,
        "client",
        lambda *args, **kwargs: object(),
    )
    syncer = mathe_syncer_module.MathE_Syncer(base_dir=tmp_path)
    syncer.data = [
        {
            "id": "./6.pdf",
            "claude_ocr_text": "already done",
            "status": "completed",
        },
        {
            "id": "./7.pdf",
            "claude_ocr_text": "valid text from older run",
            "status": "pending",
        },
        {
            "id": "./8.pdf",
            "claude_ocr_text": "OCR Failed: timeout",
            "status": "failed",
        },
        {
            "id": "./9.pdf",
            "claude_ocr_text": None,
            "status": "pending",
        },
    ]
    processed = []

    def fake_perform_claude_call(path):
        processed.append(path.name)
        return "new OCR text"

    monkeypatch.setattr(syncer, "_perform_claude_call", fake_perform_claude_call)

    syncer.run_batch_ocr()

    assert processed == ["9.pdf"]
    assert syncer.data[3]["status"] == "completed"
    assert syncer.data[3]["claude_ocr_text"] == "new OCR text"
