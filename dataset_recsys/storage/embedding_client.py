import os
from typing import List, Any
import psycopg2
from psycopg2.extras import execute_values


class EmbeddingClient:
    """
    PostgreSQL + pgvector client for storing dataset and MathE embeddings.
    """
    TABLE_DATASET = "dataset_embeddings"
    TABLE_MATHE = "mathe_embeddings"

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
        
        self._init_db()

    def _init_db(self):
        """Initialize necessary tables and extensions."""
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema};")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_DATASET} (
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

            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_MATHE} (
                application TEXT NOT NULL,
                material_id TEXT NOT NULL,
                embedding VECTOR(1024) NOT NULL,
                embedding_input TEXT,
                embedding_model TEXT NOT NULL,
                enrichment_llm TEXT,
                prompt_version TEXT,
                run_id TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (material_id)
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
        table: str = "dataset_embeddings",        
        run_id: str | None = None,
        **kwargs
    ) -> int:
        """
        Store embeddings in bulk with metadata (replaces existing entries for the application).
        """
        id_column = "material_id" if table == self.TABLE_MATHE else "dataset_id"

        self.delete_application(application, table=table)

        rows = [
            (
                application,
                dataset_id,
                embedding.tolist(),
                embedding_input,
                embedding_model,
                kwargs.get("enrichment_llm", "none"),
                kwargs.get("prompt_version", "none"),
                run_id,
            )
            for dataset_id, embedding, embedding_input in zip(dataset_ids, embeddings, embedding_inputs)
        ]

        query = f"""
        INSERT INTO {self.schema}.{table} (
            application,
            {id_column},
            embedding,
            embedding_input,
            embedding_model,
            enrichment_llm,
            prompt_version,
            run_id
        )
        VALUES %s
        ON CONFLICT ({id_column})
        DO UPDATE SET
            -- TODO: Remove ''application = EXCLUDED.application'' after all
            -- consumers use the permanent split MathE application namespaces.
            application = EXCLUDED.application,
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
        top_k: int | None = 10,
        table: str = "dataset_embeddings"        
    ):
        id_column = "material_id" if table == self.TABLE_MATHE else "dataset_id"
        # Using <=> for cosine distance. Similarity = 1 - (A <=> B)
        limit_clause = "LIMIT %s" if top_k is not None else ""
        query = f"""
        SELECT {id_column}, 1 - (embedding <=> %s::vector) AS similarity
        FROM {self.schema}.{table}
        WHERE application = %s
        ORDER BY embedding <=> %s::vector
        {limit_clause}
        """
        params = (
            (query_embedding, application, query_embedding, top_k)
            if top_k is not None
            else (query_embedding, application, query_embedding)
        )

        with self.conn.cursor() as cur:
            # Note: query_embedding must be a list or np.array
            cur.execute(query, params)
            return cur.fetchall()

    def find_similar_by_ids(
        self,
        application: str,
        query_embedding: List[float],
        entity_ids: List[str],
        table: str = "dataset_embeddings",
    ):
        """
        Return query-vector similarities for the requested IDs that have stored
        embeddings. IDs missing from the embedding table are omitted from the
        result, so callers should keep an explicit default for missing scores.
        """
        if not entity_ids:
            return []

        id_column = "material_id" if table == self.TABLE_MATHE else "dataset_id"
        query = f"""
        SELECT {id_column}, 1 - (embedding <=> %s::vector) AS similarity
        FROM {self.schema}.{table}
        WHERE application = %s
          AND {id_column} = ANY(%s)
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (query_embedding, application, entity_ids))
            return cur.fetchall()

    def delete_single_embedding(self, dataset_id: str) -> int:
        """Delete a single embedding from the database."""
        query = f"DELETE FROM {self.schema}.dataset_embeddings WHERE dataset_id = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (dataset_id,))
            return cur.rowcount

    # -------------------------
    # UTILITIES
    # -------------------------

    def delete_application(self, application: str, table: str = "dataset_embeddings") -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.schema}.{table} WHERE application = %s", (application,))
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

    def close(self) -> None:
        if self.conn:
            self.conn.close()
