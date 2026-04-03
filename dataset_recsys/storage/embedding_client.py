import os
from typing import List, Any

import psycopg2
from psycopg2.extras import execute_values


class EmbeddingClient:
    """
    PostgreSQL + pgvector client for storing dataset embeddings.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.conn = psycopg2.connect(
            host=host or os.getenv("DB_HOST", "localhost"),
            port=port or os.getenv("DB_PORT", "5432"),
            dbname=dbname or os.getenv("DB_NAME", "postgres"),
            user=user or os.getenv("DB_USER", "postgres"),
            password=password or os.getenv("DB_PASSWORD", "postgres"),
        )
        self.schema = os.getenv("DB_SCHEMA", "public")
        self.conn.autocommit = True

        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            # cur.execute(f"""
            # CREATE TABLE IF NOT EXISTS {self.schema}.dataset_embeddings (
            #     application TEXT NOT NULL,
            #     dataset_id TEXT NOT NULL,
            #     embedding VECTOR(1536) NOT NULL,
            #     embedding_model TEXT NOT NULL,
            #     PRIMARY KEY (application, dataset_id)
            # );
            # """)

    # -------------------------
    # STORAGE
    # -------------------------

    def store_embeddings(
        self,
        application: str,
        dataset_ids: List[str],
        embeddings: Any,
        embedding_model: str,
    ) -> int:
        """
        Store embeddings in bulk (replaces existing ones for the application).
        """

        self.delete_application(application)

        rows = [
            (
                application,
                dataset_id,
                embedding.tolist(),  # numpy -> list
                embedding_model,
            )
            for dataset_id, embedding in zip(dataset_ids, embeddings)
        ]

        query = """
        INSERT INTO dataset_embeddings (application, dataset_id, embedding, embedding_model)
        VALUES %s
        ON CONFLICT (dataset_id)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model
        """

        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            execute_values(cur, query, rows)

        return len(rows)

    # -------------------------
    # QUERYING
    # -------------------------

    def find_similar(
        self,
        application: str,
        query_embedding: List[float],
        top_k: int = 10,
    ):
        query = """
        SELECT dataset_id, embedding <-> %s AS distance
        FROM dataset_embeddings
        WHERE application = %s
        ORDER BY embedding <-> %s
        LIMIT %s
        """

        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            cur.execute(query, (query_embedding, application, query_embedding, top_k))
            return cur.fetchall()

    def get_schema_overview(self) -> dict:
        """
        Return database schema: tables, columns, and types.
        """
        query = """
        SELECT
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position;
        """

        schema = {}

        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            cur.execute(query, (self.schema,))
            rows = cur.fetchall()

            for row in rows:
                table_schema, table_name, column_name, data_type, is_nullable, default = row

                if table_name not in schema:
                    schema[table_name] = []

                schema[table_name].append({
                    "column": column_name,
                    "type": data_type,
                    "nullable": is_nullable == "YES",
                    "default": default,
                })

        return schema

    # -------------------------
    # UTILITIES
    # -------------------------

    def delete_application(self, application: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM dataset_embeddings WHERE application = %s",
                (application,),
            )
            return cur.rowcount

    def check_connection(self) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False