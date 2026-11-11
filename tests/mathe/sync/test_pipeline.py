"""Tests for the MathE synchronization and index publication pipeline."""

import sys
import sqlite3
import types
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import BackgroundTasks
from qdrant_client import QdrantClient

from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.storage.qdrant_client import QdrantStorageClient
from dataset_recsys.workflows import mathe_sync_pipeline
from dataset_recsys.api.routes import mathe

import dataset_recsys.utils.mathe_syncer as mathe_syncer_module

class FakeSyncer:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.is_running = False
        self.reconcile_called = False
        self.process_called = False
        self.status = {}

    def require_source_directories(self):
        pass

    def reconcile_sources(self):
        self.reconcile_called = True

    def process_pending_materials(self):
        self.process_called = True

    def get_sync_status(self):
        return dict(self.status)

    def save_sync_status(self, status):
        self.status = dict(status)

    def get_completed_materials(self):
        with self._get_sqlite_conn() as conn:
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

MATHE = QdrantStorageClient.COLLECTION_MATHE
PIPELINES = pytest.mark.parametrize(
    "pipeline_name",
    ["run_full_mathe_pipeline", "run_incremental_mathe_pipeline"],
)


@pytest.fixture
def qdrant(monkeypatch):
    """The real storage client on Qdrant's in-memory mode, used by the pipeline."""
    client = QdrantStorageClient(client=QdrantClient(":memory:"))
    monkeypatch.setattr(mathe_sync_pipeline, "_build_qdrant_client", lambda: client)
    return client


def _store_materials(qdrant, application, vectors):
    qdrant.store_embeddings(
        application=application,
        dataset_ids=list(vectors),
        embeddings=np.array(list(vectors.values()), dtype=float),
        embedding_inputs=[f"text {material_id}" for material_id in vectors],
        embedding_model="test-model",
        collection_name=MATHE,
    )


def _write_catalog(db_path, rows):
    """Create a sync catalog from (id, type, text, status, platform_id) rows."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_entries (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                claude_ocr_text TEXT,
                status TEXT NOT NULL,
                platform_material_id TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO sync_entries VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def _encode_from(vectors_by_text, encoded_texts):
    """Build an encode_texts stand-in that records what it was asked to embed."""

    def fake_encode_texts(texts, model_name):
        encoded_texts.extend(texts)
        return np.array([vectors_by_text[text] for text in texts], dtype=float)

    return fake_encode_texts


def test_full_pipeline_builds_separate_document_and_video_indexes(
    tmp_path,
    monkeypatch,
    qdrant,
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

    summary = mathe_sync_pipeline.run_full_mathe_pipeline(fake_syncer)

    assert fake_syncer.reconcile_called is True
    assert fake_syncer.process_called is True
    assert fake_syncer.is_running is False
    assert summary == {
        "status": "completed",
        "processed_materials": 4,
        "embeddings_created": 4,
        "qdrant_records_updated": 4,
        "collections": {
            "mathe_documents": {
                "processed_materials": 2,
                "embeddings_stored": 2,
                "qdrant_records_updated": 2,
            },
            "mathe_videos": {
                "processed_materials": 2,
                "embeddings_stored": 2,
                "qdrant_records_updated": 2,
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

    # Each namespace holds only the platform material IDs of its content kind.
    assert qdrant.list_entities("mathe_documents", MATHE) == ["6", "7"]
    assert qdrant.list_entities("mathe_videos", MATHE) == ["901", "902"]

    # Neighbours never cross the document/video boundary.
    assert qdrant.get_recommendations("mathe_documents", "6", collection_name=MATHE) == ["7"]
    assert qdrant.get_recommendations("mathe_documents", "7", collection_name=MATHE) == ["6"]
    assert qdrant.get_recommendations("mathe_videos", "901", collection_name=MATHE) == ["902"]
    assert qdrant.get_recommendations("mathe_videos", "902", collection_name=MATHE) == ["901"]
    # The retired mixed namespace is not published.
    assert qdrant.list_entities("mathe", MATHE) == []


def test_split_indexes_skip_unresolved_platform_ids_and_clear_empty_collection(qdrant):
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
        MatheApplication.DOCUMENTS,
        MatheApplication.VIDEOS,
    }
    assert set(indices[MatheApplication.DOCUMENTS]) == {"6"}
    assert indices[MatheApplication.VIDEOS] == {}

    # A video published by an earlier run must not stay once the namespace is empty.
    _store_materials(qdrant, MatheApplication.VIDEOS, {"901": [1.0, 0.0]})
    _store_materials(qdrant, MatheApplication.DOCUMENTS, {"6": [0.0, 1.0]})

    result = mathe_sync_pipeline._replace_embedding_and_recommendation_collection(
        application=MatheApplication.VIDEOS,
        entity_indices={},
        materials=materials,
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        qdrant_client=qdrant,
        run_id="stage-2-test",
    )

    assert result == {
        "processed_materials": 0,
        "embeddings_stored": 0,
        "qdrant_records_updated": 0,
    }
    assert qdrant.list_entities(MatheApplication.VIDEOS, MATHE) == []
    assert qdrant.list_entities(MatheApplication.DOCUMENTS, MATHE) == ["6"]


# A broken local catalog must fail safely instead of clearing published indexes.
@PIPELINES
def test_pipeline_handles_missing_sync_database(pipeline_name, tmp_path):
    fake_syncer = FakeSyncer(tmp_path / "missing.db")

    with pytest.raises(sqlite3.OperationalError):
        getattr(mathe_sync_pipeline, pipeline_name)(fake_syncer)

    assert fake_syncer.reconcile_called is True
    assert fake_syncer.process_called is True
    assert fake_syncer.is_running is False
    assert fake_syncer.status["sync_status"] == "failed"
    assert fake_syncer.status["embeddings_created"] == 0


@PIPELINES
def test_pipeline_does_not_start_a_duplicate_run(pipeline_name, tmp_path):
    fake_syncer = FakeSyncer(tmp_path / "unused.db")
    fake_syncer.is_running = True

    summary = getattr(mathe_sync_pipeline, pipeline_name)(fake_syncer)

    assert summary == {
        "status": "already_running",
        "processed_materials": 0,
        "embeddings_created": 0,
        "qdrant_records_updated": 0,
        "reason": "A MathE refresh pipeline is already running.",
    }
    assert fake_syncer.reconcile_called is False
    assert fake_syncer.process_called is False
    assert fake_syncer.status == {}
    assert fake_syncer.is_running is True


def test_full_pipeline_clears_collections_for_empty_catalog(tmp_path, qdrant):
    db_path = tmp_path / "syncer.db"
    _write_catalog(db_path, [])
    _store_materials(qdrant, MatheApplication.DOCUMENTS, {"6": [1.0, 0.0]})
    _store_materials(qdrant, MatheApplication.VIDEOS, {"901": [0.0, 1.0]})

    summary = mathe_sync_pipeline.run_full_mathe_pipeline(FakeSyncer(db_path))

    assert summary["status"] == "completed"
    assert summary["processed_materials"] == 0
    assert summary["embeddings_created"] == 0
    # A successful empty catalog is authoritative: nothing stays recommendable.
    for application in ("mathe_documents", "mathe_videos"):
        assert qdrant.list_entities(application, MATHE) == []


# An index built by an earlier version still holds the retired mixed "mathe"
# namespace. A full rebuild removes it and leaves the split namespaces intact.
def test_full_pipeline_deletes_the_retired_legacy_namespace(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(db_path, [("./6.pdf", "document", "matrices", "completed", "6")])
    monkeypatch.setattr(
        mathe_sync_pipeline, "encode_texts", _encode_from({"matrices": [1.0, 0.0]}, [])
    )
    _store_materials(qdrant, "mathe", {"6.pdf": [1.0, 0.0], "abcdefghijk.txt": [0.0, 1.0]})

    mathe_sync_pipeline.run_full_mathe_pipeline(FakeSyncer(db_path))

    assert qdrant.list_entities("mathe", MATHE) == []
    assert qdrant.list_entities("mathe_documents", MATHE) == ["6"]


# ---------------------------------------------------------------------------
# Incremental pipeline
# ---------------------------------------------------------------------------


def test_incremental_pipeline_indexes_new_materials_per_content_kind(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(
        db_path,
        [
            ("./6.pdf", "document", "matrices", "completed", "6"),
            ("./7.docx", "document", "derivatives", "completed", "7"),
            ("abcdefghijk", "video", "lesson", "completed", "901"),
            ("./8.pdf", "document", "pending text", "pending", "8"),
            ("zyxwvutsrqp", "video", "no platform id", "completed", None),
        ],
    )
    encoded_texts = []
    monkeypatch.setattr(
        mathe_sync_pipeline,
        "encode_texts",
        _encode_from(
            {"matrices": [1.0, 0.0], "derivatives": [0.6, 0.8], "lesson": [0.0, 1.0]},
            encoded_texts,
        ),
    )
    fake_syncer = FakeSyncer(db_path)

    summary = mathe_sync_pipeline.run_incremental_mathe_pipeline(fake_syncer)

    assert summary == {
        "status": "completed",
        "mode": "incremental",
        "processed_new_materials": 3,
        "removed_materials": 0,
        "embeddings_created": 3,
    }
    assert fake_syncer.reconcile_called is True
    assert fake_syncer.process_called is True
    assert fake_syncer.is_running is False
    assert fake_syncer.status["sync_status"] == "completed"
    assert fake_syncer.status["embeddings_created"] == 3
    # Pending entries and entries without a platform ID are not embedded.
    assert sorted(encoded_texts) == ["derivatives", "lesson", "matrices"]

    assert qdrant.list_entities("mathe_documents", MATHE) == ["6", "7"]
    assert qdrant.list_entities("mathe_videos", MATHE) == ["901"]
    assert qdrant.get_recommendations("mathe_documents", "6", collection_name=MATHE) == ["7"]
    assert qdrant.get_recommendations("mathe_documents", "7", collection_name=MATHE) == ["6"]
    # The only video has no other video to be recommended with.
    assert qdrant.get_recommendations("mathe_videos", "901", collection_name=MATHE) == []


def test_incremental_pipeline_embeds_only_materials_not_yet_indexed(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(
        db_path,
        [
            ("./6.pdf", "document", "matrices", "completed", "6"),
            ("./7.docx", "document", "derivatives", "completed", "7"),
        ],
    )
    vectors = {
        "matrices": [1.0, 0.0],
        "derivatives": [0.0, 1.0],
        "matrix inverses": [0.8, 0.6],
    }
    encoded_texts = []
    monkeypatch.setattr(
        mathe_sync_pipeline, "encode_texts", _encode_from(vectors, encoded_texts)
    )
    mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))
    encoded_texts.clear()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sync_entries VALUES (?, ?, ?, ?, ?)",
            ("./9.pdf", "document", "matrix inverses", "completed", "9"),
        )
    summary = mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))

    assert summary["processed_new_materials"] == 1
    assert summary["embeddings_created"] == 1
    assert encoded_texts == ["matrix inverses"]
    assert qdrant.list_entities("mathe_documents", MATHE) == ["6", "7", "9"]
    # The new material gets its neighbours, best first ...
    assert qdrant.get_recommendations_with_scores(
        "mathe_documents", "9", collection_name=MATHE
    ) == [("6", pytest.approx(0.8)), ("7", pytest.approx(0.6))]
    # ... and is inserted into each neighbour's list at the right rank.
    assert qdrant.get_recommendations("mathe_documents", "6", collection_name=MATHE) == ["9", "7"]
    assert qdrant.get_recommendations("mathe_documents", "7", collection_name=MATHE) == ["9", "6"]


def test_incremental_pipeline_with_nothing_new_changes_nothing(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(db_path, [("./6.pdf", "document", "matrices", "completed", "6")])
    encoded_texts = []
    monkeypatch.setattr(
        mathe_sync_pipeline,
        "encode_texts",
        _encode_from({"matrices": [1.0, 0.0]}, encoded_texts),
    )
    mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))
    encoded_texts.clear()

    summary = mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))

    assert summary["processed_new_materials"] == 0
    assert summary["removed_materials"] == 0
    assert summary["embeddings_created"] == 0
    assert encoded_texts == []
    assert qdrant.list_entities("mathe_documents", MATHE) == ["6"]


def _stored_rankings(qdrant, application):
    return {
        material_id: qdrant.get_recommendations_with_scores(
            application, material_id, collection_name=MATHE
        )
        for material_id in qdrant.list_entities(application, MATHE)
    }


# Adding materials one sync at a time must end in the same index as a single
# full rebuild of the same catalog: every material keeps its complete ranking.
def test_incremental_syncs_store_the_same_rankings_as_a_full_rebuild(
    tmp_path, monkeypatch
):
    import math

    # Unevenly spaced directions, so no two neighbours are exactly equally
    # similar to a material and the expected order is unambiguous.
    document_ids = [str(100 + step) for step in range(14)]
    angles = [5 * step + 1.7 * math.sqrt(step) for step in range(14)]
    vectors = {
        f"text {material_id}": [
            math.cos(math.radians(angle)),
            math.sin(math.radians(angle)),
        ]
        for material_id, angle in zip(document_ids, angles)
    }
    monkeypatch.setattr(mathe_sync_pipeline, "encode_texts", _encode_from(vectors, []))

    def run(pipeline, db_path, storage):
        monkeypatch.setattr(mathe_sync_pipeline, "_build_qdrant_client", lambda: storage)
        pipeline(FakeSyncer(db_path))

    incremental = QdrantStorageClient(client=QdrantClient(":memory:"))
    incremental_db = tmp_path / "incremental.db"
    _write_catalog(incremental_db, [])
    for material_id in document_ids:
        with sqlite3.connect(incremental_db) as conn:
            conn.execute(
                "INSERT INTO sync_entries VALUES (?, ?, ?, ?, ?)",
                (f"./{material_id}.pdf", "document", f"text {material_id}", "completed", material_id),
            )
        run(mathe_sync_pipeline.run_incremental_mathe_pipeline, incremental_db, incremental)

    full = QdrantStorageClient(client=QdrantClient(":memory:"))
    run(mathe_sync_pipeline.run_full_mathe_pipeline, incremental_db, full)

    incremental_rankings = _stored_rankings(incremental, "mathe_documents")
    full_rankings = _stored_rankings(full, "mathe_documents")
    assert set(incremental_rankings) == set(document_ids)
    for material_id in document_ids:
        # Thirteen neighbours each: more than the old cap of ten.
        assert len(full_rankings[material_id]) == 13
        assert [n for n, _ in incremental_rankings[material_id]] == [
            n for n, _ in full_rankings[material_id]
        ]
        assert [score for _, score in incremental_rankings[material_id]] == pytest.approx(
            [score for _, score in full_rankings[material_id]]
        )


# A material that left the catalog must stop being recommendable without
# waiting for a full rebuild, and no other material may keep pointing at it.
def test_incremental_pipeline_removes_materials_that_left_the_catalog(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(
        db_path,
        [
            ("./6.pdf", "document", "matrices", "completed", "6"),
            ("./7.docx", "document", "derivatives", "completed", "7"),
            ("abcdefghijk", "video", "lesson", "completed", "901"),
        ],
    )
    encoded_texts = []
    monkeypatch.setattr(
        mathe_sync_pipeline,
        "encode_texts",
        _encode_from(
            {"matrices": [1.0, 0.0], "derivatives": [0.6, 0.8], "lesson": [0.0, 1.0]},
            encoded_texts,
        ),
    )
    mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))
    assert qdrant.get_recommendations("mathe_documents", "6", collection_name=MATHE) == ["7"]
    encoded_texts.clear()

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM sync_entries WHERE id = './7.docx'")
    summary = mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))

    assert summary["removed_materials"] == 1
    assert summary["processed_new_materials"] == 0
    assert summary["embeddings_created"] == 0
    assert encoded_texts == []
    assert qdrant.list_entities("mathe_documents", MATHE) == ["6"]
    assert qdrant.get_recommendations("mathe_documents", "6", collection_name=MATHE) == []
    # The other content kind is untouched.
    assert qdrant.list_entities("mathe_videos", MATHE) == ["901"]


# A material that is no longer completed (for example a requeued transcript)
# is not in the completed catalog, so it leaves the index like a removed one.
def test_incremental_pipeline_removes_materials_that_are_no_longer_completed(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(db_path, [("abcdefghijk", "video", "lesson", "completed", "901")])
    monkeypatch.setattr(
        mathe_sync_pipeline, "encode_texts", _encode_from({"lesson": [0.0, 1.0]}, [])
    )
    mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE sync_entries SET status = 'pending', claude_ocr_text = NULL")
    summary = mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))

    assert summary["removed_materials"] == 1
    assert qdrant.list_entities("mathe_videos", MATHE) == []


# A stored material is not embedded again when its text changes; only a full
# rebuild does that.
def test_incremental_pipeline_does_not_reembed_a_stored_material(
    tmp_path, monkeypatch, qdrant
):
    db_path = tmp_path / "syncer.db"
    _write_catalog(db_path, [("./6.pdf", "document", "matrices", "completed", "6")])
    encoded_texts = []
    monkeypatch.setattr(
        mathe_sync_pipeline,
        "encode_texts",
        _encode_from({"matrices": [1.0, 0.0], "chain rule": [0.0, 1.0]}, encoded_texts),
    )
    mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))
    encoded_texts.clear()

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE sync_entries SET claude_ocr_text = 'chain rule'")
    summary = mathe_sync_pipeline.run_incremental_mathe_pipeline(FakeSyncer(db_path))

    assert summary["processed_new_materials"] == 0
    assert summary["removed_materials"] == 0
    assert encoded_texts == []

    mathe_sync_pipeline.run_full_mathe_pipeline(FakeSyncer(db_path))

    assert encoded_texts == ["chain rule"]


# These tests check that the MathE sync API endpoints are wired correctly:
# /sync schedules the incremental pipeline and /sync/rebuild the full one,
# each as a background task.
@pytest.mark.parametrize(
    ("endpoint_name", "pipeline_name"),
    [
        ("sync_data", "run_incremental_mathe_pipeline"),
        ("rebuild_catalog", "run_full_mathe_pipeline"),
    ],
)
def test_sync_endpoints_schedule_their_mathe_pipeline(
    endpoint_name, pipeline_name, monkeypatch
):

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
    response = asyncio.run(getattr(mathe, endpoint_name)(background_tasks))

    assert response["status"] == "Accepted"
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is getattr(mathe, pipeline_name)
    assert background_tasks.tasks[0].args == (mathe.syncer,)


def test_sync_routes_are_registered():
    routes_by_path = {route.path: route for route in mathe.router.routes}

    assert routes_by_path["/dataset-recsys/mathe/sync"].endpoint is mathe.sync_data
    assert (
        routes_by_path["/dataset-recsys/mathe/sync/rebuild"].endpoint
        is mathe.rebuild_catalog
    )


def test_document_recommend_endpoint_uses_curricular_pool_ranker(monkeypatch):
    calls = {}
    fake_mathe_client = object()
    fake_qdrant_client = object()

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
    monkeypatch.setattr(mathe, "get_qdrant_client", lambda: fake_qdrant_client)
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
        "qdrant_client": fake_qdrant_client,
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
    with syncer.catalog.connection() as conn:
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

    syncer.process_pending_materials()

    assert processed == ["9.pdf"]
    with syncer.catalog.connection() as conn:
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
    syncer.process_pending_materials()

    assert processed == [("abcdefghijk", whisper_model)]
    assert syncer.get_raw()[0]["type"] == "video"
    assert syncer.get_raw()[0]["claude_ocr_text"] == "new transcript"


class MissingDirectoriesSyncer(FakeSyncer):
    def require_source_directories(self):
        raise FileNotFoundError("MathE sync cannot start")


# A sync with unmounted material directories must stop before reconciliation,
# so it can never treat missing folders as removed materials.
@pytest.mark.parametrize(
    "pipeline_name",
    ["run_full_mathe_pipeline", "run_incremental_mathe_pipeline"],
)
def test_pipeline_fails_before_reconciliation_when_directories_are_missing(
    pipeline_name, tmp_path
):
    fake_syncer = MissingDirectoriesSyncer(tmp_path / "unused.db")
    run_pipeline = getattr(mathe_sync_pipeline, pipeline_name)

    with pytest.raises(FileNotFoundError):
        run_pipeline(fake_syncer)

    assert fake_syncer.reconcile_called is False
    assert fake_syncer.process_called is False
    assert fake_syncer.is_running is False
    assert fake_syncer.status["sync_status"] == "failed"


# A full MathE rebuild is authoritative: a material removed from the catalog
# must disappear from its collection, while the other content kind is untouched.
def test_full_rebuild_removes_materials_that_left_the_catalog():
    from qdrant_client import QdrantClient

    from dataset_recsys.mathe_recommenders.constants import MatheApplication
    from dataset_recsys.storage.qdrant_client import QdrantStorageClient

    qdrant = QdrantStorageClient(client=QdrantClient(":memory:"))
    mathe = qdrant.COLLECTION_MATHE

    def publish(application, material_ids, embeddings):
        materials = [
            {"sync_entry_id": material_id, "text": f"text {material_id}"}
            for material_id in material_ids
        ]
        return mathe_sync_pipeline._replace_embedding_and_recommendation_collection(
            application=application,
            entity_indices={mid: index for index, mid in enumerate(material_ids)},
            materials=materials,
            embeddings=np.array(embeddings, dtype=float),
            qdrant_client=qdrant,
            run_id="run",
        )

    publish(MatheApplication.VIDEOS, ["900", "901"], [[1.0, 0.0], [0.0, 1.0]])
    publish(
        MatheApplication.DOCUMENTS,
        ["221", "222", "223"],
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
    )
    assert qdrant.list_entities(MatheApplication.DOCUMENTS, mathe) == ["221", "222", "223"]

    summary = publish(MatheApplication.DOCUMENTS, ["221", "222"], [[1.0, 0.0], [0.6, 0.8]])

    assert summary["processed_materials"] == 2
    assert qdrant.list_entities(MatheApplication.DOCUMENTS, mathe) == ["221", "222"]
    assert qdrant.get_recommendations(
        MatheApplication.DOCUMENTS, "221", collection_name=mathe
    ) == ["222"]
    assert qdrant.list_entities(MatheApplication.VIDEOS, mathe) == ["900", "901"]
