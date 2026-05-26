from dataset_recsys.storage.embedding_client import EmbeddingClient


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
        return []


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()

    def cursor(self):
        return self.cursor_instance


# Test cases for EmbeddingClient.find_similar to verify LIMIT clause behavior 
# based on top_k parameter either being None or set to a specific value. 

def test_find_similar_omits_limit_when_top_k_is_none():
    client = EmbeddingClient.__new__(EmbeddingClient)
    client.schema = "public"
    client.conn = FakeConnection()

    client.find_similar("mathe", [0.1, 0.2], top_k=None)

    cursor = client.conn.cursor_instance
    assert "LIMIT" not in cursor.executed_query
    assert cursor.executed_params == ([0.1, 0.2], "mathe", [0.1, 0.2])


def test_find_similar_includes_limit_when_top_k_is_set():
    client = EmbeddingClient.__new__(EmbeddingClient)
    client.schema = "public"
    client.conn = FakeConnection()

    client.find_similar("mathe", [0.1, 0.2], top_k=20)

    cursor = client.conn.cursor_instance
    assert "LIMIT %s" in cursor.executed_query
    assert cursor.executed_params == ([0.1, 0.2], "mathe", [0.1, 0.2], 20)


def test_find_similar_uses_material_id_for_mathe_table():
    client = EmbeddingClient.__new__(EmbeddingClient)
    client.schema = "public"
    client.conn = FakeConnection()

    client.find_similar(
        "mathe",
        [0.1, 0.2],
        top_k=5,
        table=client.TABLE_MATHE,
    )

    cursor = client.conn.cursor_instance
    assert "SELECT material_id" in cursor.executed_query
    assert "FROM public.mathe_embeddings" in cursor.executed_query
    assert cursor.executed_params == ([0.1, 0.2], "mathe", [0.1, 0.2], 5)


def test_find_similar_by_ids_filters_to_requested_mathe_materials():
    client = EmbeddingClient.__new__(EmbeddingClient)
    client.schema = "public"
    client.conn = FakeConnection()

    client.find_similar_by_ids(
        "mathe",
        [0.1, 0.2],
        entity_ids=["100.pdf", "101.pdf"],
        table=client.TABLE_MATHE,
    )

    cursor = client.conn.cursor_instance
    assert "SELECT material_id" in cursor.executed_query
    assert "FROM public.mathe_embeddings" in cursor.executed_query
    assert "material_id = ANY(%s)" in cursor.executed_query
    assert cursor.executed_params == (
        [0.1, 0.2],
        "mathe",
        ["100.pdf", "101.pdf"],
    )
