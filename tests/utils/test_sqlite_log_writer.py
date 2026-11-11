"""Tests for the six-month SQLite request log."""

import importlib
import sqlite3
import time

import pytest

from dataset_recsys.utils import sqlite_log_writer


@pytest.fixture
def log_db(tmp_path, monkeypatch):
    """Point the request log at a temporary file in a not-yet-existing folder."""
    db_path = tmp_path / "volume" / "logs" / "requests.db"
    monkeypatch.setattr(sqlite_log_writer, "DB_PATH", str(db_path))
    return db_path


def _rows(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM api_request_logs")]


def test_log_location_comes_from_the_environment(tmp_path, monkeypatch):
    configured = tmp_path / "persistent" / "requests.db"
    try:
        monkeypatch.setenv("REQUEST_LOG_DB_PATH", str(configured))
        assert importlib.reload(sqlite_log_writer).DB_PATH == str(configured)

        monkeypatch.delenv("REQUEST_LOG_DB_PATH")
        assert importlib.reload(sqlite_log_writer).DB_PATH == "api_request_logs.db"

        # An empty setting falls back to the default instead of an empty path.
        monkeypatch.setenv("REQUEST_LOG_DB_PATH", "")
        assert importlib.reload(sqlite_log_writer).DB_PATH == "api_request_logs.db"
    finally:
        monkeypatch.undo()
        importlib.reload(sqlite_log_writer)


def test_init_creates_missing_folders_and_is_repeatable(log_db):
    sqlite_log_writer.init_sqlite_db()
    sqlite_log_writer.init_sqlite_db()

    assert log_db.exists()
    assert _rows(log_db) == []


def test_request_is_written_with_its_details(log_db):
    sqlite_log_writer.init_sqlite_db()
    before = time.time()

    sqlite_log_writer.write_request_log_to_sqlite(
        user_id="user-1",
        action="get_recommendations",
        entity_id="dataset-1",
        requested_n=5,
        status_code=200,
        duration_ms=12.5,
    )

    rows = _rows(log_db)
    assert len(rows) == 1
    row = rows[0]
    assert row["user_id"] == "user-1"
    assert row["action"] == "get_recommendations"
    assert row["entity_id"] == "dataset-1"
    assert row["requested_n"] == 5
    assert row["status_code"] == 200
    assert row["duration_ms"] == 12.5
    assert before <= row["timestamp"] <= time.time()
    assert row["id"]


# Records must survive a restart: a second start-up on the same file keeps them.
def test_records_survive_a_restart_on_the_same_file(log_db):
    sqlite_log_writer.init_sqlite_db()
    sqlite_log_writer.write_request_log_to_sqlite("user-1", "get_recommendations", "a", 1, 200, 1.0)

    sqlite_log_writer.init_sqlite_db()
    sqlite_log_writer.write_request_log_to_sqlite("user-2", "get_recommendations", "b", 1, 200, 1.0)

    assert sorted(row["user_id"] for row in _rows(log_db)) == ["user-1", "user-2"]


def test_purge_removes_only_records_older_than_six_months(log_db):
    sqlite_log_writer.init_sqlite_db()
    now = time.time()
    six_months = sqlite_log_writer.SIX_MONTHS_SECONDS
    with sqlite3.connect(log_db) as conn:
        conn.executemany(
            "INSERT INTO api_request_logs (id, user_id, timestamp) VALUES (?, ?, ?)",
            [
                ("old", "user-old", now - six_months - 3600),
                ("recent", "user-recent", now - six_months + 3600),
                ("new", "user-new", now),
            ],
        )

    sqlite_log_writer.purge_expired_sqlite_logs()

    assert sorted(row["id"] for row in _rows(log_db)) == ["new", "recent"]


# Logging must never break a request: a write to an unusable location is
# reported in the service log and otherwise ignored.
def test_failed_write_does_not_raise(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sqlite_log_writer, "DB_PATH", str(tmp_path))

    with caplog.at_level("ERROR"):
        sqlite_log_writer.write_request_log_to_sqlite("user-1", "get_recommendations", "a", 1, 200, 1.0)

    assert "Failed to write log to SQLite" in caplog.text
