import os
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


def material_id_to_redis_id(material_id: Any) -> str:
    """Convert a MathE DB material ID to the Redis/OCR entity ID."""
    return f"{str(material_id).strip().removesuffix('.pdf')}.pdf"


def redis_id_to_material_id(material_redis_id: Any) -> int | None:
    """Convert a Redis/OCR entity ID back to a MathE DB material ID."""
    material_id = str(material_redis_id).strip().removesuffix(".pdf")
    return int(material_id) if material_id.isdigit() else None


def _material_ids_from_redis_ids(material_redis_ids: list[str]) -> list[int]:
    """Return unique DB material IDs represented by current Redis IDs.

    Current production sync stores PDFs and Redis entities as
    `<platform_materials.id>.pdf`, even though Postgres `file_name` can be
    something else, e.g. `221 -> ChainRule.pdf -> Redis 221.pdf`.

    Future migration note:
    If synced PDFs/Redis keys are changed to use `platform_materials.file_name`,
    update the SELECT aliases below to `m.file_name AS material_redis_id` and
    change detail/metadata lookup methods to filter with `m.file_name = ANY(%s)`
    instead of parsing Redis IDs back to integer DB IDs.
    """
    material_ids = [
        material_id
        for material_id in (
            redis_id_to_material_id(material_redis_id)
            for material_redis_id in material_redis_ids
        )
        if material_id is not None
    ]
    return list(dict.fromkeys(material_ids))


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
            m.id AS material_id,
            m.id::text || '.pdf' AS material_redis_id,
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
            t.name AS topic,
            s.name AS subtopic,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform__sna__questions q
        LEFT JOIN platform__topic t
            ON q.topic = t.id
        LEFT JOIN platform__subtopic s
            ON q.subtopic = s.id
        LEFT JOIN platform_keyword_snaquestion qk
            ON q.id = qk.platformsnaquestionid
        LEFT JOIN platform__keywords k
            ON qk.platformkeywordid = k.id
        WHERE q.id = %s
        GROUP BY
            q.id,
            q.topic,
            q.subtopic,
            t.name,
            s.name;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            return cur.fetchone()

    def get_questions_by_topic_subtopics(
        self,
        topic_subtopics: list[tuple[str, str]],
    ) -> list[Dict[str, Any]]:
        """
        Retrieve MathE questions for topic/subtopic name pairs.

        Topic and subtopic names are matched case-insensitively.
        """
        normalized_pairs = [
            (topic.strip().lower(), subtopic.strip().lower())
            for topic, subtopic in topic_subtopics
            if topic and subtopic
        ]
        if not normalized_pairs:
            return []

        values_clause = ", ".join(["(%s, %s)"] * len(normalized_pairs))
        params = [
            value
            for topic_subtopic in normalized_pairs
            for value in topic_subtopic
        ]
        query = f"""
        SELECT
            q.id AS question_id,
            t.name AS topic_name,
            s.name AS subtopic_name,
            q.question
        FROM platform__sna__questions q
        JOIN platform__topic t ON q.topic = t.id
        JOIN platform__subtopic s ON q.subtopic = s.id
        JOIN (VALUES {values_clause}) AS target(topic_name, subtopic_name)
            ON LOWER(t.name) = target.topic_name
            AND LOWER(s.name) = target.subtopic_name
        ORDER BY
            LOWER(t.name),
            LOWER(s.name),
            q.id;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return list(cur.fetchall())

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
            m.id::text || '.pdf' AS material_redis_id,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformtopicid), NULL) AS topic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformsubtopicid), NULL) AS subtopic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform__sna__questions q
        JOIN platform_materials m
            ON m.file_ext = 'pdf'
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
            m.id;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            return cur.fetchall()

    def get_pdf_materials_for_question_topic_subtopic(
        self,
        question_id: int,
    ) -> list[Dict[str, Any]]:
        """Retrieve PDF materials in the same topic/subtopic as a question."""
        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.pdf' AS material_redis_id,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformtopicid), NULL) AS topic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformsubtopicid), NULL) AS subtopic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform__sna__questions q
        JOIN material_top_sub pool_mts
            ON q.topic = pool_mts.platformtopicid
            AND q.subtopic = pool_mts.platformsubtopicid
        JOIN platform_materials m
            ON pool_mts.platformmaterialid = m.id
            AND m.file_ext = 'pdf'
        LEFT JOIN material_top_sub mts
            ON m.id = mts.platformmaterialid
        LEFT JOIN platform_material_keyword mk
            ON m.id = mk.platformmaterialid
        LEFT JOIN platform__keywords k
            ON mk.platformkeywordid = k.id
        WHERE q.id = %s
        GROUP BY
            m.id
        ORDER BY
            m.id;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            return cur.fetchall()

    def get_pdf_material_details(
        self,
        material_redis_ids: list[str],
    ) -> list[Dict[str, Any]]:
        """Retrieve display metadata for PDF materials by Redis material ID.

        This is intentionally for API/debug presentation: title, author,
        description, file_name, and human-readable topic/subtopic/keyword names.
        Scoring uses get_pdf_material_metadata_by_redis_ids to avoid fetching
        display fields and topic/subtopic names it does not need.
        """
        material_ids = _material_ids_from_redis_ids(material_redis_ids)
        if not material_ids:
            return []

        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.pdf' AS material_redis_id,
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
        AND m.file_ext = 'pdf'
        GROUP BY
            m.id,
            m.file_name,
            m.title,
            m.author,
            m.description;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (material_ids,))
            return cur.fetchall()

    def get_pdf_material_metadata_by_redis_ids(
        self,
        material_redis_ids: list[str],
    ) -> list[Dict[str, Any]]:
        """Retrieve minimal PDF metadata needed for recommendation scoring."""
        material_ids = _material_ids_from_redis_ids(material_redis_ids)
        if not material_ids:
            return []

        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.pdf' AS material_redis_id,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformtopicid), NULL) AS topic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformsubtopicid), NULL) AS subtopic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform_materials m
        LEFT JOIN material_top_sub mts
            ON m.id = mts.platformmaterialid
        LEFT JOIN platform_material_keyword mk
            ON m.id = mk.platformmaterialid
        LEFT JOIN platform__keywords k
            ON mk.platformkeywordid = k.id
        WHERE m.id = ANY(%s)
        AND m.file_ext = 'pdf'
        GROUP BY
            m.id;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (material_ids,))
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
