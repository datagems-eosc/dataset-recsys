from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient


class FakeCursor:
    def __init__(self):
        self.executed_query = None
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params):
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


def test_get_document_materials_for_question_topic_subtopic_uses_hard_pool():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_document_materials_for_question_topic_subtopic(82)

    cursor = client.conn.cursor_instance
    assert "q.topic = pool_mts.platformtopicid" in cursor.executed_query
    assert "q.subtopic = pool_mts.platformsubtopicid" in cursor.executed_query
    assert "m.type = 3" in cursor.executed_query
    assert "LOWER(m.file_ext) IN ('pdf', 'docx', 'pptx')" in cursor.executed_query
    assert "m.id::text || '.' || LOWER(m.file_ext) AS material_redis_id" in cursor.executed_query
    assert cursor.executed_params == (82,)


def test_get_pdf_materials_for_question_topic_subtopic_stays_pdf_only():
    client = MatheMirrorClient.__new__(MatheMirrorClient)
    client.conn = FakeConnection()

    client.get_pdf_materials_for_question_topic_subtopic(82)

    cursor = client.conn.cursor_instance
    assert "q.topic = pool_mts.platformtopicid" in cursor.executed_query
    assert "q.subtopic = pool_mts.platformsubtopicid" in cursor.executed_query
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
