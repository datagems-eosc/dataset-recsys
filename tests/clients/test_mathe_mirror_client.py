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
