import json
import time
import uuid
import logging
import threading

logger = logging.getLogger(__name__)

REDIS_LOGS_KEY = "api_request_logs"
SIX_MONTHS_SECONDS = 15_552_000 

def write_request_log_to_redis(recs_client, user_id: str | None, action: str, entity_id: str | None, requested_n: int | None, status_code: int, duration_ms: float):
    """Pushes a log entry into a Redis Sorted Set."""
    try:
        now = time.time()
        log_payload = {
            "id": str(uuid.uuid4()),  # Uniqueness key
            "user_id": user_id,
            "action": action,
            "entity_id": entity_id,
            "requested_n": requested_n,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "timestamp": now
        }
        # Updated to use self.r from RecommendationClient
        r = recs_client.r  
        r.zadd(REDIS_LOGS_KEY, {json.dumps(log_payload): now})
    except Exception as e:
        logger.error(f"Failed to write log to Redis: {e}")

def purge_expired_redis_logs(recs_client):
    """Deletes Redis logs older than 6 months."""
    try:
        # Updated to use self.r from RecommendationClient
        r = recs_client.r
        six_months_ago = time.time() - SIX_MONTHS_SECONDS
        deleted_count = r.zremrangebyscore(REDIS_LOGS_KEY, 0, six_months_ago)
        if deleted_count > 0:
            logger.info(f"Purged {deleted_count} expired request logs from Redis.")
    except Exception as e:
        logger.error(f"Failed to purge expired Redis logs: {e}")

def start_daily_purge_scheduler(recs_client):
    """Starts a daemon thread that purges expired logs once every 24 hours."""
    def loop():
        while True:
            # Run the purge inside the safety of a try/except so the thread never dies
            try:
                purge_expired_redis_logs(recs_client)
            except Exception as e:
                logger.error(f"Scheduled daily purge failed: {e}")
            
            # Sleep for 24 hours (86,400 seconds)
            time.sleep(86400)

    # Daemon=True means this thread automatically shuts down when the FastAPI app stops
    threading.Thread(target=loop, daemon=True).start()