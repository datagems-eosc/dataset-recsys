import os
from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


SUPPORTED_DOCUMENT_EXTENSIONS = ("pdf", "docx", "pptx")


def material_id_to_redis_id(material_id: Any, file_ext: str = "pdf") -> str:
    """Convert a MathE DB material ID to the Redis/OCR entity ID."""
    clean_id = str(material_id).strip()
    if any(clean_id.lower().endswith(f".{ext}") for ext in SUPPORTED_DOCUMENT_EXTENSIONS):
        return clean_id

    clean_ext = str(file_ext).strip().lower().lstrip(".") or "pdf"
    return f"{clean_id}.{clean_ext}"


def redis_id_to_material_id(material_redis_id: Any) -> int | None:
    """Convert a Redis/OCR entity ID back to a MathE DB material ID."""
    material_id = str(material_redis_id).strip()
    for extension in SUPPORTED_DOCUMENT_EXTENSIONS:
        suffix = f".{extension}"
        if material_id.lower().endswith(suffix):
            material_id = material_id[: -len(suffix)]
            break
    return int(material_id) if material_id.isdigit() else None


def _material_ids_from_redis_ids(material_redis_ids: list[str]) -> list[int]:
    """Return unique DB material IDs represented by current Redis IDs.

    Current production sync stores document Redis entities as
    `<platform_materials.id>.<file_ext>`, even though Postgres `file_name` can be
    something else, e.g. `221 -> ChainRule.pdf -> Redis 221.pdf`.

    Future migration note:
    If synced document Redis keys are changed to use `platform_materials.file_name`,
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

    def get_evaluation_benchmark_questions(self) -> list[Dict[str, Any]]:
        """
        Retrieve one benchmark question per topic/subtopic pool.

        Questions are selected from historical assessment activity. Within each
        topic/subtopic pool, the selected question is the one attempted by the
        largest number of distinct students; ties are broken by wrong-answer
        rate and then total attempts.
        """
        query = """
        WITH question_stats AS (
            SELECT
                q.id AS question_id,
                q.question,
                q.topic AS _topic_id,
                q.subtopic AS _subtopic_id,
                t.name AS topic_name,
                s.name AS subtopic_name,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords,
                COUNT(*) AS total_attempts,
                COUNT(DISTINCT a.student_id) AS distinct_students,
                1.0 - AVG(CASE WHEN a.answer = 1 THEN 1.0 ELSE 0.0 END) AS wrong_rate
            FROM assessment a
            JOIN platform__sna__questions q
                ON q.id = a.question_id
            LEFT JOIN platform__topic t
                ON q.topic = t.id
            LEFT JOIN platform__subtopic s
                ON q.subtopic = s.id
            LEFT JOIN platform_keyword_snaquestion qk
                ON q.id = qk.platformsnaquestionid
            LEFT JOIN platform__keywords k
                ON qk.platformkeywordid = k.id
            GROUP BY
                q.id,
                q.question,
                q.topic,
                q.subtopic,
                t.name,
                s.name
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY _topic_id, _subtopic_id
                    ORDER BY distinct_students DESC, wrong_rate DESC, total_attempts DESC
                ) AS rn
            FROM question_stats
        )
        SELECT
            question_id,
            question,
            topic_name,
            subtopic_name,
            keywords,
            total_attempts,
            distinct_students,
            wrong_rate
        FROM ranked
        WHERE rn = 1
        ORDER BY distinct_students DESC, wrong_rate DESC;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
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
            AND (
                q.subtopic = pool_mts.platformsubtopicid
                OR (q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL)
            )
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

    def get_document_materials_for_question_topic_subtopic(
        self,
        question_id: int,
    ) -> list[Dict[str, Any]]:
        """Retrieve document teaching materials in the same topic/subtopic as a question."""
        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformtopicid), NULL) AS topic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT mts.platformsubtopicid), NULL) AS subtopic_ids,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords
        FROM platform__sna__questions q
        JOIN material_top_sub pool_mts
            ON q.topic = pool_mts.platformtopicid
            AND (
                q.subtopic = pool_mts.platformsubtopicid
                OR (q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL)
            )
        JOIN platform_materials m
            ON pool_mts.platformmaterialid = m.id
            AND m.type = 3
            AND LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')
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

    def get_most_popular_document_material_for_question_topic_subtopic(
        self,
        question_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the most-clicked eligible document material for a question pool."""
        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id,
            m.title,
            LOWER(m.file_ext) AS file_ext,
            m.clicks
        FROM platform__sna__questions q
        JOIN material_top_sub pool_mts
            ON q.topic = pool_mts.platformtopicid
            AND (
                q.subtopic = pool_mts.platformsubtopicid
                OR (q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL)
            )
        JOIN platform_materials m
            ON pool_mts.platformmaterialid = m.id
            AND m.type = 3
            AND LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')
        WHERE q.id = %s
        ORDER BY
            m.clicks DESC,
            m.id
        LIMIT 1;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (question_id,))
            return cur.fetchone()

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

    def get_document_material_details(
        self,
        material_redis_ids: list[str],
    ) -> list[Dict[str, Any]]:
        """Retrieve display metadata for supported document materials by Redis material ID."""
        material_ids = _material_ids_from_redis_ids(material_redis_ids)
        if not material_ids:
            return []

        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id,
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
        AND m.type = 3
        AND LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')
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

    def get_document_material_metadata_by_redis_ids(
        self,
        material_redis_ids: list[str],
    ) -> list[Dict[str, Any]]:
        """Retrieve minimal supported document metadata needed for recommendation scoring."""
        material_ids = _material_ids_from_redis_ids(material_redis_ids)
        if not material_ids:
            return []

        query = """
        SELECT
            m.id AS material_id,
            m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id,
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
        AND m.type = 3
        AND LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')
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
