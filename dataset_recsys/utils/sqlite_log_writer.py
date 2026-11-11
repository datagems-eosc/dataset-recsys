"""Request log of the dataset recommendation endpoint, kept for six months.

The log is a SQLite file. Its location comes from the REQUEST_LOG_DB_PATH
environment variable and defaults to ``api_request_logs.db`` in the working
directory.

Deployment requirements for the records to be kept:

- Point REQUEST_LOG_DB_PATH at persistent storage. A file inside the container
  is lost whenever the container restarts or is redeployed.
- Use local or block storage, not a network file system: the file is opened in
  SQLite's WAL mode, which is not reliable over NFS or similar mounts.
- Run a single replica. Several replicas writing one SQLite file are not
  supported; a shared log for several replicas needs a database server.
"""
import os
import sqlite3
import time
import uuid
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "api_request_logs.db"
DB_PATH = os.getenv("REQUEST_LOG_DB_PATH") or DEFAULT_DB_PATH
SIX_MONTHS_SECONDS = 15_552_000 

def init_sqlite_db():
    """Initializes the SQLite database and creates the logs table/index."""
    # The configured location may be a directory on a freshly mounted volume.
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    # WAL mode enables concurrent reads and writes, crucial for web APIs
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_request_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                action TEXT,
                entity_id TEXT,
                requested_n INTEGER,
                status_code INTEGER,
                duration_ms REAL,
                timestamp REAL
            )
        """)
        # Index accelerates the 6-month purge queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_request_logs(timestamp);")
        conn.commit()

def write_request_log_to_sqlite(user_id: str | None, action: str, entity_id: str | None, requested_n: int | None, status_code: int, duration_ms: float):
    """Inserts a log entry into the SQLite database."""
    try:
        now = time.time()
        log_id = str(uuid.uuid4())
        
        # timeout=5.0 gives threads a moment to wait if the DB is briefly locked
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            conn.execute(
                """
                INSERT INTO api_request_logs 
                (id, user_id, action, entity_id, requested_n, status_code, duration_ms, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, user_id, action, entity_id, requested_n, status_code, duration_ms, now)
            )
    except Exception as e:
        logger.error(f"Failed to write log to SQLite: {e}")

def purge_expired_sqlite_logs():
    """Deletes SQLite logs older than 6 months."""
    try:
        six_months_ago = time.time() - SIX_MONTHS_SECONDS
        
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            cursor = conn.execute("DELETE FROM api_request_logs WHERE timestamp < ?", (six_months_ago,))
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"Purged {deleted_count} expired request logs from SQLite.")
                # Optional: Reclaim disk space after large deletions
                # conn.execute("VACUUM;") 
    except Exception as e:
        logger.error(f"Failed to purge expired SQLite logs: {e}")

def start_daily_purge_scheduler():
    """Starts a daemon thread that purges expired logs once every 24 hours."""
    # Ensure the database and tables exist before background operations start
    init_sqlite_db()
    
    def loop():
        while True:
            try:
                purge_expired_sqlite_logs()
            except Exception as e:
                logger.error(f"Scheduled daily purge failed: {e}")
            
            time.sleep(86400)

    threading.Thread(target=loop, daemon=True).start()