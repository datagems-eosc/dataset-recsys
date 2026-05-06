from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from time import perf_counter
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np

from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile, fetch_catalog
from dataset_recsys.utils.bedrock import enrich_batch
from dataset_recsys.embeddings import build_embedding_text, encode_texts
from dataset_recsys.retrieval import build_recommendations
from dataset_recsys.storage.recommendation_client import RecommendationClient
from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.utils.text_preprocessing import preprocess_catalog

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("data/gems_datasets_metadata/workflow_artifacts")


def _artifact_filename(name: str, extension: str = "json") -> str:
    """Build a compact artifact filename."""
    return f"{name}.{extension}"


def _save_json_artifact(run_dir: Path, filename: str, payload: Any, message_prefix: str) -> Path:
    """Save a JSON artifact inside the current workflow run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[TEST] {message_prefix}: {output_path}")
    return output_path


CatalogFetcher = Callable[[], list[DatasetProfile]]
CatalogEnricher = Callable[[list[DatasetProfile]], list[DatasetProfile]]
CatalogPreprocessor = Callable[[list[DatasetProfile]], list[DatasetProfile]]
EmbeddingGenerator = Callable[[list[DatasetProfile]], Any]
EmbeddingWriter = Callable[[Any], None]
# EmbeddingMetadataWriter = Callable[[list[DatasetProfile], Any], None]
RecommendationList = list[dict[str, Any]]
RecommendationComputer = Callable[[Any, list[DatasetProfile]], RecommendationList]
RecommendationWriter = Callable[[RecommendationList], None]


@dataclass(slots=True)
class BatchRebuildArtifacts:
    """Outputs and metadata produced by a full batch rebuild."""

    started_at: datetime
    finished_at: datetime
    raw_catalog_size: int
    processed_catalog_size: int
    recommendation_count: int
    duration_seconds: float
    embeddings: Any | None = None


@dataclass(slots=True)
class FullBatchRebuildWorkflow:
    """Orchestrates a full catalog rebuild.

    The workflow executes the following steps:
    1. Fetch dataset profiles from the catalog.
    2. Enrich profiles using an LLM to generate catalog summaries.
    3. Preprocess profiles by cleaning and normalizing text fields.
    4. Generate embeddings for all datasets.
    5. Compute recommendations based on the embeddings.
    6. Store the recommendations for serving.
    """

    fetch_catalog: CatalogFetcher
    enrich_catalog: CatalogEnricher
    preprocess_catalog: CatalogPreprocessor
    generate_embeddings: EmbeddingGenerator
    write_embeddings: EmbeddingWriter
    # write_embeddings_metadata: EmbeddingMetadataWriter
    compute_recommendations: RecommendationComputer
    write_recommendations: RecommendationWriter
    logger: logging.Logger = field(default_factory=lambda: logger)

    def run(self) -> BatchRebuildArtifacts:
        """Execute a full batch rebuild for the entire catalog."""
        started_at = datetime.now()
        started_perf = perf_counter()
        self.logger.info("Starting full batch rebuild")

        raw_catalog = self._fetch_catalog()
        enriched_catalog = self._enrich_catalog(raw_catalog)
        processed_catalog = self._preprocess_catalog(enriched_catalog)
        embeddings = self._generate_embeddings(processed_catalog)
        self.write_embeddings(embeddings)
        recommendations = self._compute_recommendations(embeddings, processed_catalog)
        self._write_recommendations(recommendations)

        finished_at = datetime.now()
        duration_seconds = perf_counter() - started_perf

        artifacts = BatchRebuildArtifacts(
            started_at=started_at,
            finished_at=finished_at,
            raw_catalog_size=len(raw_catalog),
            processed_catalog_size=len(processed_catalog),
            recommendation_count=len(recommendations),
            duration_seconds=duration_seconds,
            embeddings=embeddings,
        )

        self.logger.info(
            "Finished full batch rebuild in %.2fs (raw=%d, processed=%d, recos=%d)",
            artifacts.duration_seconds,
            artifacts.raw_catalog_size,
            artifacts.processed_catalog_size,
            artifacts.recommendation_count,
        )
        return artifacts

    def _fetch_catalog(self) -> list[DatasetProfile]:
        self.logger.info("Fetching full dataset catalog")
        catalog = self.fetch_catalog()
        self._validate_catalog(catalog)
        self.logger.info("Fetched %d datasets", len(catalog))
        return catalog

    def _enrich_catalog(self, catalog: list[DatasetProfile]) -> list[DatasetProfile]:
        self.logger.info("Enriching dataset catalog")
        enriched_catalog = self.enrich_catalog(catalog)
        self._validate_catalog(enriched_catalog)
        self.logger.info("Enriched %d datasets", len(enriched_catalog))
        return enriched_catalog

    def _preprocess_catalog(self, catalog: list[DatasetProfile]) -> list[DatasetProfile]:
        self.logger.info("Preprocessing dataset catalog")
        processed_catalog = self.preprocess_catalog(catalog)
        self._validate_catalog(processed_catalog)
        self.logger.info("Preprocessed %d datasets", len(processed_catalog))
        return processed_catalog

    def _generate_embeddings(self, processed_catalog: list[DatasetProfile]) -> Any:
        self.logger.info("Generating dataset embeddings")
        embeddings = self.generate_embeddings(processed_catalog)
        self.logger.info("Generated embeddings with shape %s", getattr(embeddings, "shape", None))
        return embeddings

    def _write_embeddings(self, embeddings: Any) -> None:
        self.logger.info("Writing embeddings to storage")
        self.write_embeddings(embeddings)
        self.logger.info("Embeddings written")

    def _compute_recommendations(
        self,
        embeddings: Any,
        processed_catalog: list[DatasetProfile],
    ) -> RecommendationList:
        self.logger.info("Computing recommendation lists for full catalog")
        recommendations = self.compute_recommendations(embeddings, processed_catalog)
        self._validate_recommendations(recommendations)
        self.logger.info("Computed %d recommendation lists", len(recommendations))
        return recommendations

    def _write_recommendations(
        self,
        recommendations: RecommendationList,
    ) -> None:
        self.logger.info("Writing recommendations to serving store")
        self.write_recommendations(recommendations)
        self.logger.info("Recommendations written")

    @staticmethod
    def _validate_catalog(catalog: list[DatasetProfile]) -> None:
        if not isinstance(catalog, list):
            raise TypeError("Catalog must be returned as a list of DatasetProfile objects.")

    @staticmethod
    def _validate_recommendations(recommendations: RecommendationList) -> None:
        if not isinstance(recommendations, list):
            raise TypeError(
                "Recommendations must be returned as a list of ranked recommendation entries."
            )


def _save_profiles_artifact(run_dir: Path, filename: str, profiles: list[DatasetProfile]) -> None:
    """Save profile artifacts locally for inspection and reproducibility."""
    payload = [asdict(profile) for profile in profiles]
    _save_json_artifact(run_dir, filename, payload, "Saved profiles to")

# TODO: align embedding_id format with the future DB schema once vector storage
# is introduced.
def _save_embedding_metadata_artifact(
    run_dir: Path,
    filename: str,
    profiles: list[DatasetProfile],
    embedding_texts: list[str],
    embedding_model: str,
    enrichment_llm: str | None = None,
    prompt_version: str | None = None,
) -> None:
    """Save embedding metadata locally for inspection and reproducibility."""
    metadata = [
        {
            "embedding_id": f"{embedding_model}:{profile.id}",
            "dataset_id": profile.id,
            "embedding_input": text,
            "embedding_model": embedding_model,
            "input_enrichment_llm": enrichment_llm,
            "input_prompt_version": prompt_version,
        }
        for profile, text in zip(profiles, embedding_texts, strict=False)
    ]
    _save_json_artifact(
        run_dir,
        filename,
        metadata,
        "Saved embedding metadata to",
    )

def _save_embeddings_artifact(
    run_dir: Path,
    filename: str,
    embeddings: Any,
) -> None:
    """Save embeddings locally as a .npy file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / filename
    np.save(output_path, embeddings)
    print(f"[TEST] Saved embeddings to: {output_path}")

# Function to run the full batch rebuild workflow with configurable parameters (scheduler/integration calls)
def run_full_batch_rebuild(
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    application: str = "ds2ds",
    enrichment_llm: str = "claude-sonnet-4-6",
    prompt_version: str = "catalog_summary_v1",
    embedding_model: str = "allenai/specter2_base",
) -> BatchRebuildArtifacts:
    """Run the full batch rebuild workflow for the dataset recommender."""
    recs_client = RecommendationClient(host=redis_host, port=redis_port, db=redis_db)
    embedding_client = EmbeddingClient()

    def generate_embeddings_step(catalog):
        embedding_texts = [build_embedding_text(profile) for profile in catalog]
        embeddings = encode_texts(embedding_texts, model_name=embedding_model)

        # Save local metadata artifact
        artifact_name = _artifact_filename("embeddings_metadata")
        _save_embedding_metadata_artifact(
            run_dir,
            artifact_name,
            catalog,
            embedding_texts,
            embedding_model,
            enrichment_llm=enrichment_llm,
            prompt_version=prompt_version,
        )

        # Save local embeddings (.npy)
        embeddings_filename = _artifact_filename("embeddings", extension="npy")
        _save_embeddings_artifact(run_dir, embeddings_filename, embeddings)

        return {
            "embeddings": embeddings,
            "texts": embedding_texts,
            "catalog": catalog,
        }
    
    def write_embeddings_step(payload):
        embeddings = payload["embeddings"]
        embedding_texts = payload["texts"]
        catalog = payload["catalog"]
        # Store embeddings + metadata in Postgres/vector DB
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # unique per workflow run
        dataset_ids = [p.id for p in catalog]
        embedding_client.store_embeddings(
            application=application,
            dataset_ids=dataset_ids,
            embeddings=embeddings,
            embedding_inputs=embedding_texts,
            embedding_model=embedding_model,
            enrichment_llm=enrichment_llm,
            prompt_version=prompt_version,
            run_id=run_id,
        )
        print(f"[TEST] Stored {len(dataset_ids)} embeddings with metadata (run_id={run_id})")        

    def compute_recommendations_step(payload, catalog):
        embeddings = payload["embeddings"]
        return build_recommendations(catalog, embeddings)

    def write_recommendations_step(recommendations):
        recs_client.store_recommendations(application=application, data=recommendations)

    workflow = FullBatchRebuildWorkflow(
        fetch_catalog=fetch_catalog,
        enrich_catalog=partial(
            enrich_batch,
            llm=enrichment_llm,
            prompt_version=prompt_version,
        ),
        preprocess_catalog=preprocess_catalog,
        generate_embeddings=generate_embeddings_step,
        write_embeddings=write_embeddings_step,
        compute_recommendations=compute_recommendations_step,
        write_recommendations=write_recommendations_step,
    )

    return workflow.run()


__all__ = ["BatchRebuildArtifacts", "FullBatchRebuildWorkflow"]


if __name__ == "__main__":
    
    enrichment_llm = "claude-sonnet-4-6"
    prompt_version = "catalog_summary_v1"
    embedding_model = "allenai/specter2_base"
    application = "ds2ds"

    # TODO: support a "load from existing artifacts" mode to skip Bedrock and
    # embedding recomputation during local debugging.
    TEST_MODE = False
    test_limit = 1 if TEST_MODE else None

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / f"{application}_{run_timestamp}"

    # Force local Redis for testing (avoid using any server/project credentials)
    redis_host, redis_port, redis_db = "localhost", 6379, 0
    recs_client = RecommendationClient(host=redis_host, port=redis_port, db=redis_db)
    embedding_client = EmbeddingClient(
        host="localhost",
        port=5433,
        dbname="postgres",
        user="postgres",
        password="postgres"
    )
    print("Redis OK:", recs_client.check_connection())
    print("Embedding DB OK:", embedding_client.check_connection())

    def fetch_catalog_step():
        catalog = fetch_catalog()
        limited_catalog = catalog[:test_limit]
        print(f"[TEST] Using {len(limited_catalog)} dataset(s) for this local run")
        return limited_catalog

    def preprocess_catalog_step(catalog):
        processed_catalog = preprocess_catalog(catalog)
        artifact_name = _artifact_filename("dataset_profiles")
        _save_profiles_artifact(run_dir, artifact_name, processed_catalog)
        return processed_catalog
    
    # .npy saving only for offline debugging/artifact inspection.
    def generate_embeddings_step(catalog):
        embedding_texts = [build_embedding_text(profile) for profile in catalog]
        embeddings = encode_texts(embedding_texts, model_name=embedding_model)
        artifact_name = _artifact_filename("embeddings_metadata")
        _save_embedding_metadata_artifact(
            run_dir,
            artifact_name,
            catalog,
            embedding_texts,
            embedding_model,
            enrichment_llm=enrichment_llm,
            prompt_version=prompt_version,
        )

        embeddings_filename = _artifact_filename("embeddings", extension="npy")
        _save_embeddings_artifact(run_dir, embeddings_filename, embeddings)
        return {
            "embeddings": embeddings,
            "texts": embedding_texts,
            "catalog": catalog,
        }   

    def write_embeddings_step(payload):
        embeddings = payload["embeddings"]
        embedding_texts = payload["texts"]
        catalog = payload["catalog"]
        # Store embeddings + metadata in Postgres/vector DB
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # unique per workflow run
        dataset_ids = [p.id for p in catalog]
        embedding_client.store_embeddings(
            application=application,
            dataset_ids=dataset_ids,
            embeddings=embeddings,
            embedding_inputs=embedding_texts,
            embedding_model=embedding_model,
            enrichment_llm=enrichment_llm,
            prompt_version=prompt_version,
            run_id=run_id,
        )
        print(f"[TEST] Stored {len(dataset_ids)} embeddings with metadata (run_id={run_id})")

    def compute_recommendations_step(payload, catalog):
        embeddings = payload["embeddings"]
        return build_recommendations(catalog, embeddings)

    def write_recommendations_step(recommendations):
        stored_entities = recs_client.store_recommendations(application=application, data=recommendations)
        print(
            f"[TEST] Recommendations written to Redis for application '{application}' "
            f"({stored_entities} entities stored)"
        )

    workflow = FullBatchRebuildWorkflow(
        fetch_catalog=fetch_catalog_step,
        enrich_catalog=partial(enrich_batch, llm=enrichment_llm, prompt_version=prompt_version),
        preprocess_catalog=preprocess_catalog_step,
        generate_embeddings=generate_embeddings_step,
        write_embeddings=write_embeddings_step,
        compute_recommendations=compute_recommendations_step,
        write_recommendations=write_recommendations_step,
    )

    artifacts = workflow.run()
    print("\n[TEST] Batch rebuild finished:")
    print(f"Datasets fetched: {artifacts.raw_catalog_size}")
    print("Fetch, enrichment, preprocessing, and embedding-generation steps completed successfully.")
    print(f"Enrichment LLM: {enrichment_llm}")
    print(f"Prompt version: {prompt_version}")
    print(f"Embedding model: {embedding_model}")
    print(f"Serving application: {application}")
    print(f"Test catalog limit: {test_limit}")
    print(f"Workflow artifacts directory: {run_dir}")
    print("Recommendations were written to Redis, replacing any existing recommendations for this application.")

    # --- TEST REDIS LOCALLY ---
    # 1) Start Redis (if not running):
    #    docker start redis-recsys
    #    OR (first time): docker run -d -p 6380:6379 --name redis-recsys redis:7
    #
    # 2) Check connection:
    #    redis-cli -p 6380 ping
    #    -> should return: PONG
    #
    # 3) Inspect stored recommendations:
    #    redis-cli -p 6380
    #    KEYS recs:*
    #    ZREVRANGE recs:ds2ds:<dataset_id> 0 9 WITHSCORES
