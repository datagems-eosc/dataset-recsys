import os
from typing import List, Any
import psycopg2
from psycopg2.extras import execute_values


class EmbeddingClient:
    """
    PostgreSQL + pgvector client for storing dataset embeddings with enriched metadata.
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
            host=host or os.getenv("DATAGEMS_POSTGRES_HOST", "localhost"),
            port=port or os.getenv("DATAGEMS_POSTGRES_PORT", "5432"),
            dbname=dbname or os.getenv("DATAGEMS_POSTGRES_DBNAME", "postgres"),
            user=user or os.getenv("DATAGEMS_POSTGRES_USERNAME", "postgres"),
            password=password or os.getenv("DATAGEMS_POSTGRES_PASSWORD", "postgres"),
        )
        self.schema = os.getenv("DATAGEMS_POSTGRES_SCHEMA", "public")
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.dataset_embeddings (
                application TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                embedding VECTOR(768) NOT NULL,
                embedding_input TEXT,
                embedding_model TEXT NOT NULL,
                enrichment_llm TEXT,
                prompt_version TEXT,
                run_id TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (dataset_id)
            );
            """)

    # -------------------------
    # STORAGE
    # -------------------------

    def store_embeddings(
        self,
        application: str,
        dataset_ids: List[str],
        embeddings: Any,
        embedding_inputs: List[str],
        embedding_model: str,
        enrichment_llm: str | None = None,
        prompt_version: str | None = None,
        run_id: str | None = None,
    ) -> int:
        """
        Store embeddings in bulk with metadata (replaces existing entries for the application).
        """

        self.delete_application(application)

        rows = [
            (
                application,
                dataset_id,
                embedding.tolist(),  # numpy -> list
                embedding_input,
                embedding_model,
                enrichment_llm,
                prompt_version,
                run_id,
            )
            for dataset_id, embedding, embedding_input in zip(dataset_ids, embeddings, embedding_inputs)
        ]

        query = f"""
        INSERT INTO {self.schema}.dataset_embeddings (
            application,
            dataset_id,
            embedding,
            embedding_input,
            embedding_model,
            enrichment_llm,
            prompt_version,
            run_id
        )
        VALUES %s
        ON CONFLICT (dataset_id)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            embedding_input = EXCLUDED.embedding_input,
            embedding_model = EXCLUDED.embedding_model,
            enrichment_llm = EXCLUDED.enrichment_llm,
            prompt_version = EXCLUDED.prompt_version,
            run_id = EXCLUDED.run_id,
            created_at = NOW();
        """

        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            execute_values(cur, query, rows)

        return len(rows)

    def upsert_single_embedding(
        self, 
        application: str, 
        dataset_id: str, 
        embedding: List[float], 
        embedding_input: str,
        metadata: dict
    ) -> None:
        """Upsert a single vector and metadata."""
        query = f"""
        INSERT INTO {self.schema}.dataset_embeddings (
            application, dataset_id, embedding, embedding_input, 
            embedding_model, enrichment_llm, prompt_version, run_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dataset_id) 
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            embedding_input = EXCLUDED.embedding_input,
            created_at = NOW();
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (
                application, dataset_id, embedding, embedding_input,
                metadata.get("model"), metadata.get("llm"), 
                metadata.get("prompt"), metadata.get("run_id")
            ))

    # -------------------------
    # QUERYING
    # -------------------------

    def exists(self, dataset_id: str) -> bool:
        """Check if a dataset_id already has an embedding record."""
        query = f"SELECT 1 FROM {self.schema}.dataset_embeddings WHERE dataset_id = %s LIMIT 1;"
        with self.conn.cursor() as cur:
            cur.execute(query, (dataset_id,))
            return cur.fetchone() is not None

    def find_similar(
        self,
        application: str,
        query_embedding: List[float],
        top_k: int = 10,
    ):
        # Using <=> for cosine distance. Similarity = 1 - (A <=> B)
        query = f"""
        SELECT dataset_id, 1 - (embedding <=> %s) AS similarity
        FROM {self.schema}.dataset_embeddings
        WHERE application = %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """

        with self.conn.cursor() as cur:
            # Note: query_embedding must be a list or np.array
            cur.execute(query, (query_embedding, application, query_embedding, top_k))
            return cur.fetchall()

    # -------------------------
    # UTILITIES
    # -------------------------

    def delete_application(self, application: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.schema}.dataset_embeddings WHERE application = %s",
                (application,),
            )
            return cur.rowcount

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

    def check_connection(self) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False