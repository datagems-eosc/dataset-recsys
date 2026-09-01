from dataset_recsys.storage import mathe_mirror_client as mathe_mirror_client_module
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


class FakeCursor:
    def __init__(self):
        self.executed_query = None
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        self.executed_query = query
        self.executed_params = params

    def fetchall(self):
        return [
            {
                "question_id": 272,
                "topic_name": "Differentiation",
                "subtopic_name": "Derivatives",
                "question": "Differentiate x^2.",
            }
        ]

    def fetchone(self):
        return {
            "material_id": 30,
            "title": "Product Rule",
            "file_ext": "pdf",
            "clicks": 2,
        }


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def cursor(self, cursor_factory=None):
        return self.cursor_instance


def test_client_uses_configured_schema(monkeypatch):
    connection = FakeConnection()
    connect_kwargs = {}

    def fake_connect(**kwargs):
        connect_kwargs.update(kwargs)
        return connection

    monkeypatch.setattr(
        mathe_mirror_client_module.psycopg2,
        "connect",
        fake_connect,
    )

    client = MatheMirrorClient(dbname="test_db", schema="mathe_dev")

    assert client.schema == "mathe_dev"
    assert connection.autocommit is True
    assert connection.cursor_instance.executed_query is not None
    assert connect_kwargs["dbname"] == "test_db"

    default_client = MatheMirrorClient(dbname="test_db")
    assert default_client.schema == "public"


def test_get_questions_by_topic_subtopics_filters_requested_pairs():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    questions = client.get_questions_by_topic_subtopics(
        [
            (" Differentiation ", " Derivatives "),
            (
                "Fundamental Mathematics",
                "Algebraic expressions, Equations, and Inequalities",
            ),
        ]
    )

    cursor = client.conn.cursor_instance
    assert questions[0]["question_id"] == 272
    assert "JOIN (VALUES (%s, %s), (%s, %s))" in cursor.executed_query
    assert "LOWER(t.name) = target.topic_name" in cursor.executed_query
    assert cursor.executed_params == [
        "differentiation",
        "derivatives",
        "fundamental mathematics",
        "algebraic expressions, equations, and inequalities",
    ]


def test_get_video_materials_selects_lesson_and_review_fields():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_video_materials()

    cursor = client.conn.cursor_instance
    assert "id AS platform_material_id" in cursor.executed_query
    assert "link" in cursor.executed_query
    assert "type AS platform_type" in cursor.executed_query
    assert "WHERE type IN (1, 2)" in cursor.executed_query
    assert "AND NULLIF(BTRIM(link), '') IS NOT NULL" in cursor.executed_query
    assert "ORDER BY id" in cursor.executed_query
    assert cursor.executed_params is None


def test_get_questions_by_topic_subtopics_returns_empty_for_no_pairs():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    assert client.get_questions_by_topic_subtopics([]) == []
    assert client.conn.cursor_instance.executed_query is None


def test_get_evaluation_benchmark_questions_uses_assessment_question_stats():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    benchmark_questions = client.get_evaluation_benchmark_questions()

    cursor = client.conn.cursor_instance
    assert benchmark_questions[0]["question_id"] == 272
    assert "WITH question_stats AS" in cursor.executed_query
    assert "q.question" in cursor.executed_query
    assert "q.topic AS _topic_id" in cursor.executed_query
    assert "q.subtopic AS _subtopic_id" in cursor.executed_query
    assert "SELECT\n            question_id,\n            question,\n            topic_name," in cursor.executed_query
    assert "t.name AS topic_name" in cursor.executed_query
    assert "s.name AS subtopic_name" in cursor.executed_query
    assert "ARRAY_REMOVE(ARRAY_AGG(DISTINCT k.name), NULL) AS keywords" in cursor.executed_query
    assert "LEFT JOIN platform_keyword_snaquestion qk" in cursor.executed_query
    assert "LEFT JOIN platform__keywords k" in cursor.executed_query
    assert "COUNT(*) AS total_attempts" in cursor.executed_query
    assert "COUNT(DISTINCT a.student_id) AS distinct_students" in cursor.executed_query
    assert "AS correct_rate" not in cursor.executed_query
    assert "1.0 - AVG(CASE WHEN a.answer = 1 THEN 1.0 ELSE 0.0 END) AS wrong_rate" in cursor.executed_query
    assert "PARTITION BY _topic_id, _subtopic_id" in cursor.executed_query
    assert "ORDER BY distinct_students DESC, wrong_rate DESC, total_attempts DESC" in cursor.executed_query
    assert "WHERE rn = 1" in cursor.executed_query
    assert cursor.executed_params is None


def test_get_document_materials_for_question_topic_subtopic_uses_hard_pool():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_document_materials_for_question_topic_subtopic(82)

    cursor = client.conn.cursor_instance
    assert "q.topic = pool_mts.platformtopicid" in cursor.executed_query
    assert "q.subtopic = pool_mts.platformsubtopicid" in cursor.executed_query
    assert "q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL" in cursor.executed_query
    assert "m.type = 3" in cursor.executed_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" in cursor.executed_query
    assert "m.id AS material_id" in cursor.executed_query
    assert "AS material_redis_id" not in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_get_videos_for_question_uses_video_only_hard_pool():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_videos_for_question(82)

    cursor = client.conn.cursor_instance
    assert "SELECT DISTINCT" in cursor.executed_query
    assert "m.id AS material_id" in cursor.executed_query
    assert "m.type AS platform_type" in cursor.executed_query
    assert "q.topic = pool_mts.platformtopicid" in cursor.executed_query
    assert "q.subtopic = pool_mts.platformsubtopicid" in cursor.executed_query
    assert "q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL" in cursor.executed_query
    assert "m.type IN (1, 2)" in cursor.executed_query
    assert "m.type = 3" not in cursor.executed_query
    assert "NULLIF(BTRIM(m.link), '') IS NOT NULL" in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_get_document_seed_candidates_use_platform_ids():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_document_seed_candidates(82)

    cursor = client.conn.cursor_instance
    assert "m.id AS material_id" in cursor.executed_query
    assert "AS material_redis_id" not in cursor.executed_query
    assert "eligible_material_ids AS" in cursor.executed_query
    assert "UNION" in cursor.executed_query
    assert "EXISTS" not in cursor.executed_query
    assert "m.type = 3" in cursor.executed_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_get_popular_document_for_question_limits_to_top_click():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    material = client.get_popular_document_for_question(82)

    cursor = client.conn.cursor_instance
    assert material["material_id"] == 30
    assert "m.id AS material_id" in cursor.executed_query
    assert "m.title" in cursor.executed_query
    assert "LOWER(m.file_ext) AS file_ext" in cursor.executed_query
    assert "m.clicks" in cursor.executed_query
    assert "q.topic = pool_mts.platformtopicid" in cursor.executed_query
    assert "q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL" in cursor.executed_query
    assert "m.type = 3" in cursor.executed_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" in cursor.executed_query
    assert "ORDER BY\n            m.clicks DESC,\n            m.id\n        LIMIT 1;" in cursor.executed_query
    assert "LIMIT 1" in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_document_details_and_metadata_by_ids_use_platform_ids():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_document_material_details_by_ids(["100", "invalid", "100"])
    details_query = client.conn.cursor_instance.executed_query
    assert "m.id AS material_id" in details_query
    assert "AS material_redis_id" not in details_query
    assert client.conn.cursor_instance.executed_params == ([100],)

    client.get_document_material_metadata_by_ids(["101"])
    metadata_query = client.conn.cursor_instance.executed_query
    assert "m.id AS material_id" in metadata_query
    assert "AS material_redis_id" not in metadata_query
    assert client.conn.cursor_instance.executed_params == ([101],)
