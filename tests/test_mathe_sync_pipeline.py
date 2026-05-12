import json
import sys
import types
import asyncio
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks

from dataset_recsys.workflows import mathe_sync_pipeline
from dataset_recsys.api.routes import mathe

import dataset_recsys.utils.mathe_syncer as mathe_syncer_module

class FakeSyncer:
    def __init__(self, json_file: Path):
        self.json_file = json_file
        self.sync_called = False

    def sync_and_process(self):
        self.sync_called = True


class FakeRecommendationClient:
    stored = None

    def store_recommendations(self, application, data):
        self.__class__.stored = {"application": application, "data": data}
        return len(data)


def test_run_mathe_pipeline_builds_and_stores_recommendations(tmp_path, monkeypatch):
    data_file = tmp_path / "data.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "id": "./6.pdf",
                    "claude_ocr_text": "linear algebra matrices",
                    "status": "completed",
                },
                {
                    "id": "./7.pdf",
                    "claude_ocr_text": "calculus derivatives",
                    "status": "completed",
                },
                {
                    "id": "./8.pdf",
                    "claude_ocr_text": "geometry triangles",
                    "status": "completed",
                },
                {
                    "id": "./9.pdf",
                    "claude_ocr_text": "",
                    "status": "completed",
                },
                {
                    "id": "./10.pdf",
                    "claude_ocr_text": "not ready",
                    "status": "pending",
                },
            ]
        ),
        encoding="utf-8",
    )
    fake_syncer = FakeSyncer(data_file)

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

    summary = mathe_sync_pipeline.run_mathe_pipeline(fake_syncer)

    assert fake_syncer.sync_called is True
    assert summary["status"] == "completed"
    assert summary["processed_materials"] == 3
    assert summary["redis_keys_updated"] == 3
    assert summary["ocr_data_deleted"] is True
    assert not data_file.exists()
    assert encoded_with["model_name"] == "BAAI/bge-m3"
    assert FakeRecommendationClient.stored["application"] == "mathe"
    assert set(FakeRecommendationClient.stored["data"]) == {"6.pdf", "7.pdf", "8.pdf"}
    assert FakeRecommendationClient.stored["data"]["6.pdf"] == [("8.pdf", 0.8), ("7.pdf", 0.0)]
    assert FakeRecommendationClient.stored["data"]["7.pdf"] == [("8.pdf", 0.6), ("6.pdf", 0.0)]
    assert FakeRecommendationClient.stored["data"]["8.pdf"] == [("6.pdf", 0.8), ("7.pdf", 0.6)]


# This test ensures that if the data.json file is missing, 
# the pipeline handles it gracefully without attempting to process or store recommendations.
def test_run_mathe_pipeline_handles_missing_data_json(tmp_path):
    fake_syncer = FakeSyncer(tmp_path / "missing.json")

    summary = mathe_sync_pipeline.run_mathe_pipeline(fake_syncer)

    assert fake_syncer.sync_called is True
    assert summary["status"] == "skipped"
    assert summary["processed_materials"] == 0
    assert summary["redis_keys_updated"] == 0


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
