import os
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

class MatheMirrorClient:
    """
    PostgreSQL client for the MatheMirror platform, providing access to 
    educational content and assessment analytics.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.conn = psycopg2.connect(
            host=host or os.getenv("DATAGEMS_POSTGRES_HOST", "localhost"),
            port=port or os.getenv("DATAGEMS_POSTGRES_PORT", "5432"),
            dbname=dbname or os.getenv("DB_DS_NAME", "postgres"),
            user=user or os.getenv("DB_DS_USER", "ds_writer"),
            password=password or os.getenv("DB_DS_PASSWORD", "postgres"),
        )
        self.schema = "public"
        self.conn.autocommit = True
        
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")

    # -------------------------
    # CONTENT RETRIEVAL
    # -------------------------

    def get_material_id_by_question_id(self, question_id: int) -> Optional[str]:
        """
        Retrieves the most clicked PDF (material type 3) associated with the 
        topic of a specific question.
        """
        query = """
        SELECT 
            m.id
        FROM platform__sna__questions q
        JOIN material_top_sub mts ON q.topic = mts.platformtopicid
        JOIN platform_materials m ON mts.platformmaterialid = m.id
        WHERE q.id = %s 
          AND m.file_ext = 'pdf'
        ORDER BY m.clicks DESC
        LIMIT 1;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            result = cur.fetchone()
            # result will be a tuple like (465,) or None
            return str(result[0]) if result else None        

    # -------------------------
    # UTILITIES
    # -------------------------

    def check_connection(self) -> bool:
        """Verify the database connection is active."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()