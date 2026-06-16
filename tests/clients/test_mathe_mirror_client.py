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
            "material_redis_id": "30.pdf",
            "title": "Product Rule",
            "file_ext": "pdf",
            "clicks": 2,
        }


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def cursor(self, cursor_factory=None):
        return self.cursor_instance


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
    assert "m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id" in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_get_most_popular_document_material_for_question_topic_subtopic_limits_to_top_click():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    material = client.get_most_popular_document_material_for_question_topic_subtopic(82)

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


def test_get_pdf_materials_for_question_topic_subtopic_stays_pdf_only():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_pdf_materials_for_question_topic_subtopic(82)

    cursor = client.conn.cursor_instance
    assert "q.topic = pool_mts.platformtopicid" in cursor.executed_query
    assert "q.subtopic = pool_mts.platformsubtopicid" in cursor.executed_query
    assert "q.subtopic IS NULL AND pool_mts.platformsubtopicid IS NULL" in cursor.executed_query
    assert "m.file_ext = 'pdf'" in cursor.executed_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" not in cursor.executed_query
    assert "m.id::text || '.pdf' AS material_redis_id" in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_document_details_and_metadata_use_document_extensions():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_document_material_details(["100.docx"])
    details_query = client.conn.cursor_instance.executed_query
    assert "m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id" in details_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" in details_query

    client.get_document_material_metadata_by_redis_ids(["100.pptx"])
    metadata_query = client.conn.cursor_instance.executed_query
    assert "m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id" in metadata_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" in metadata_query


def test_pdf_details_and_metadata_stay_pdf_only():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_pdf_material_details(["100.pdf"])
    details_query = client.conn.cursor_instance.executed_query
    assert "m.id::text || '.pdf' AS material_redis_id" in details_query
    assert "m.file_ext = 'pdf'" in details_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" not in details_query

    client.get_pdf_material_metadata_by_redis_ids(["100.pdf"])
    metadata_query = client.conn.cursor_instance.executed_query
    assert "m.id::text || '.pdf' AS material_redis_id" in metadata_query
    assert "m.file_ext = 'pdf'" in metadata_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" not in metadata_query
