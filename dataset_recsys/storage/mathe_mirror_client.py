import os
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from dataset_recsys.mathe_seed_scoring import score_pdf_seed_candidates


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

    def get_material_by_question_id(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the most clicked PDF (material type 3) associated with the 
        topic of a specific question.
        """
        query = """
        SELECT 
            m.id,
            m.title,
            m.author,
            m.description,
            m.link,
            m.clicks,
            m.file_name,
            m.file_ext
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
            return cur.fetchone()

    def get_question_metadata(self, question_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve topic, subtopic, and keywords for a question."""
        query = """
        SELECT
            q.id AS question_id,
            q.topic AS topic_id,
            q.subtopic AS subtopic_id,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform__sna__questions q
        LEFT JOIN platform_keyword_snaquestion qk
            ON q.id = qk.platformsnaquestionid
        LEFT JOIN platform__keywords k
            ON qk.platformkeywordid = k.id
        WHERE q.id = %s
        GROUP BY
            q.id,
            q.topic,
            q.subtopic;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            return cur.fetchone()

    def get_pdf_seed_candidates(
        self,
        question_id: int,
    ) -> list[Dict[str, Any]]:
        """
        Retrieve PDF seed candidates and their metadata based on question attributes.
        """
        query = """
        SELECT
            m.id AS material_id,
            mts.platformtopicid AS topic_id,
            mts.platformsubtopicid AS subtopic_id,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform__sna__questions q
        JOIN platform_materials m
            ON LOWER(COALESCE(m.file_ext, '')) = 'pdf'
        LEFT JOIN material_top_sub mts
            ON m.id = mts.platformmaterialid
        LEFT JOIN platform_material_keyword mk
            ON m.id = mk.platformmaterialid
        LEFT JOIN platform__keywords k
            ON mk.platformkeywordid = k.id
        WHERE q.id = %s
        AND (
            q.topic = mts.platformtopicid
            OR q.subtopic = mts.platformsubtopicid
            OR EXISTS (
                SELECT 1
                FROM platform_keyword_snaquestion qk
                JOIN platform_material_keyword mk_match
                    ON qk.platformkeywordid = mk_match.platformkeywordid
                WHERE qk.platformsnaquestionid = q.id
                AND mk_match.platformmaterialid = m.id
            )
        )
        GROUP BY
            m.id,
            mts.platformtopicid,
            mts.platformsubtopicid;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            return cur.fetchall()

    def recommend_pdf_seeds_for_question(
        self,
        question_id: int,
        k: int = 10,
    ) -> list[Dict[str, Any]]:
        """Return the top-k PDF seed materials for a question."""
        if k <= 0:
            return []

        question_metadata = self.get_question_metadata(question_id)
        if not question_metadata:
            return []

        seed_candidates = self.get_pdf_seed_candidates(question_id)
        scored_candidates = score_pdf_seed_candidates(
            dict(question_metadata),
            [dict(candidate) for candidate in seed_candidates],
        )

        return scored_candidates[:k]

    def get_pdf_material_details(
        self,
        material_ids: list[str | int],
    ) -> list[Dict[str, Any]]:
        """Retrieve metadata for PDF materials."""
        db_ids: list[int] = []
        for material_id in material_ids:
            normalized_id = str(material_id).removesuffix(".pdf")
            if normalized_id.isdigit():
                db_ids.append(int(normalized_id))

        if not db_ids:
            return []

        query = """
        SELECT
            m.id AS material_id,
            m.title,
            m.author,
            m.description,
            m.file_name,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT t.name), NULL) AS topics,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT s.name), NULL) AS subtopics,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform_materials m
        LEFT JOIN material_top_sub mts
            ON m.id = mts.platformmaterialid
        LEFT JOIN platform__topic t
            ON mts.platformtopicid = t.id
        LEFT JOIN platform__subtopic s
            ON mts.platformsubtopicid = s.id
        LEFT JOIN platform_material_keyword mk
            ON m.id = mk.platformmaterialid
        LEFT JOIN platform__keywords k
            ON mk.platformkeywordid = k.id
        WHERE m.id = ANY(%s)
        AND LOWER(COALESCE(m.file_ext, '')) = 'pdf'
        GROUP BY
            m.id,
            m.title,
            m.author,
            m.description,
            m.file_name;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (db_ids,))
            return cur.fetchall()

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
