"""Evaluate the MathE document and video rankers against expert reference IDs.

Three ranking configurations are compared, for documents and for videos
separately:

    keyword_only    keyword Jaccard alone
    embedding_only  question-to-material cosine alone
    combined        both signals min-max normalised within the pool, equal weight

with the expected score of a random ordering of the pool for context.

Two stages, so the services are queried once and the analysis is offline and
reproducible:

fetch
    Needs the MathE mirror database, Qdrant and the embedding model. For each
    reference question it scores every document and video in the question's
    topic/subtopic pool with the deployed scoring code, and saves the raw
    signals (keyword Jaccard, question-to-material cosine) together with the
    parsed expert IDs to a JSON snapshot. Nothing is written to the services.

analyze
    Needs only the snapshot. Ranks each pool under the three configurations,
    computes metrics against the expert IDs, and writes a workbook.

Usage:
    poetry run python -m evaluation.mathe.evaluate_ranking_variants fetch
    poetry run python -m evaluation.mathe.evaluate_ranking_variants analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

# The script forks git after the tokenizer has run; without this,
# huggingface/tokenizers prints a fork warning for every subprocess.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from evaluation.mathe.ranking_variants import (
    CONFIGURATIONS,
    METRIC_NAMES,
    RANDOM,
    REFERENCE_CONFIGURATION,
    bootstrap_mean_ci,
    expected_random_metrics,
    list_metrics,
    wins_ties_losses,
)


HERE = Path(__file__).parent
DEFAULT_REFERENCE_WORKBOOK = (
    HERE / "mathe_provider_reference_submission" / "mathe_reference_alltopics.xlsx"
)
OUTPUT_DIR = HERE / "outputs"
DEFAULT_SNAPSHOT = OUTPUT_DIR / "mathe_pool_scores_snapshot.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "mathe_ranking_evaluation.xlsx"
SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_TIMING_REPEATS = 0

RANDOM_PERMUTATIONS = 2000
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261117
CI_METRICS = ("ndcg_at_5", "hit_at_5", "ndcg_at_10")
POPULATIONS = ("all_evaluated", "expert_items_embedded")


@dataclass(frozen=True)
class ContentKind:
    name: str
    sheet_prefix: str
    pool_field: str
    expert_field: str
    no_suitable_field: str


CONTENT_KINDS = (
    ContentKind("document", "doc", "documents", "expert_document_ids", "no_suitable_documents"),
    ContentKind("video", "video", "videos", "expert_video_ids", "no_suitable_videos"),
)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _snapshot_candidates(candidates: list[dict[str, Any]], similarity_field: str) -> list[dict]:
    return [
        {
            "material_id": str(candidate["material_id"]).strip(),
            "keyword_jaccard": float(candidate["keyword_jaccard"]),
            "similarity": float(candidate.get(similarity_field, 0.0)),
            "has_embedding": bool(candidate.get("has_embedding")),
        }
        for candidate in candidates
    ]


LATENCY_STAGES = (
    "pool_query_seconds",
    "keyword_scoring_seconds",
    "question_embedding_seconds",
    "vector_lookup_seconds",
    "ranking_seconds",
    "precomputed_lookup_seconds",
)


def measure_latency(
    question_id: int,
    question_text: str,
    kind_name: str,
    mathe_client: Any,
    qdrant_client: Any,
    encode: Callable[[str], list[float]],
    similarity_lookup: Callable[..., dict[str, float]],
    keyword_jaccard: Callable[[Any, Any], float],
    application: Any,
    repeats: int,
    question_hash: Callable[[str], str],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any] | None:
    """Time each stage of serving one question, as the deployed code runs it.

    Returns the median over ``repeats`` runs for every stage, or None when
    ``repeats`` is 0. The stages are what a live ranking needs: the pool and
    keyword query, Jaccard scoring, the question embedding, the Qdrant vector
    lookup and the sort. ``precomputed_lookup_seconds`` times reading the
    ranking the API serves when a precomputed one is fresh.
    ``precomputed_hash_matches`` says whether the stored ranking was built from
    this question text; if not, the API falls back to live scoring. (The API
    also checks the scoring version and sync generation, which only the
    server can see.) Nothing is written to any service.
    """
    if repeats <= 0:
        return None
    samples: dict[str, list[float]] = {stage: [] for stage in LATENCY_STAGES}
    pool_size = 0
    payload: dict[str, Any] | None = None
    for _ in range(repeats):
        start = clock()
        metadata = mathe_client.get_question_metadata(question_id) or {}
        if kind_name == "document":
            pool = mathe_client.get_document_materials_for_question_topic_subtopic(question_id)
        else:
            pool = mathe_client.get_videos_for_question(question_id)
        after_pool = clock()
        jaccards = {
            str(item["material_id"]).strip(): keyword_jaccard(
                metadata.get("keywords"), item.get("keywords")
            )
            for item in pool
        }
        after_keywords = clock()
        embedding = encode(question_text)
        after_embedding = clock()
        lookup_ids = list(jaccards)
        if kind_name == "video":
            # The video ranker also looks up the transcript point IDs.
            lookup_ids = [i for material_id in jaccards for i in (material_id, f"{material_id}.txt")]
        similarities = similarity_lookup(
            embedding, lookup_ids, qdrant_client, application=application
        )
        after_lookup = clock()
        sorted(
            jaccards,
            key=lambda material_id: (
                0.5 * jaccards[material_id]
                + 0.5 * max(float(similarities.get(material_id, 0.0)), 0.0)
            ),
            reverse=True,
        )
        after_ranking = clock()
        payload = qdrant_client.get_mathe_question_ranking(question_id, kind_name)
        after_precomputed = clock()

        pool_size = len(pool)
        for stage, seconds in zip(
            LATENCY_STAGES,
            (
                after_pool - start,
                after_keywords - after_pool,
                after_embedding - after_keywords,
                after_lookup - after_embedding,
                after_ranking - after_lookup,
                after_precomputed - after_ranking,
            ),
        ):
            samples[stage].append(seconds)
    return {
        **{stage: median(values) for stage, values in samples.items()},
        "pool_size": pool_size,
        "precomputed_ranking_found": bool(payload and payload.get(kind_name) is not None),
        "precomputed_hash_matches": bool(
            payload and payload.get("question_hash") == question_hash(question_text)
        ),
        "repeats": repeats,
    }


def fetch_snapshot(
    reference_workbook: Path,
    snapshot_path: Path,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
    environment_label: str = "",
) -> int:
    """Score every pool once with the production code and save the signals."""
    # Imported here so that `analyze` and the tests need none of these services.
    from evaluation.mathe.expert_reference import (
        _is_yes,
        _safe_text,
        load_reference_rows,
        parse_material_ids,
    )
    from dataset_recsys.mathe_recommenders.curricular_pool_ranker import (
        score_curricular_pool_candidates,
    )
    from dataset_recsys.mathe_recommenders.constants import MatheApplication
    from dataset_recsys.mathe_recommenders.question_embedding import (
        DEFAULT_MATHE_EMBEDDING_MODEL,
        encode_question,
        score_question_similarity_for_material_ids,
    )
    from dataset_recsys.mathe_recommenders.precomputed_rankings import _question_hash
    from dataset_recsys.mathe_recommenders.seed_scoring import compute_keyword_jaccard
    from dataset_recsys.mathe_recommenders.video_pool_ranker import (
        score_video_pool_candidates,
    )
    from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
    from dataset_recsys.storage.qdrant_client import QdrantStorageClient

    reference_rows = load_reference_rows(reference_workbook)
    mathe_client = MatheMirrorClient()
    qdrant_client = QdrantStorageClient()
    try:
        # Production embeds the canonical database text (precomputed rankings
        # are keyed by its hash), so use it rather than the workbook copy.
        database_text = {
            int(row["question_id"]): str(row["question"]).strip()
            for row in mathe_client.get_recommendation_questions()
        }

        # The first encode loads the model; time it once, apart from the
        # per-question latency, which is measured on a warm model.
        warmup_start = time.perf_counter()
        encode_question("warm-up", DEFAULT_MATHE_EMBEDDING_MODEL)
        model_warmup_seconds = time.perf_counter() - warmup_start

        def encode(text: str) -> list[float]:
            return encode_question(text, DEFAULT_MATHE_EMBEDDING_MODEL)

        applications = {
            "document": MatheApplication.DOCUMENTS,
            "video": MatheApplication.VIDEOS,
        }
        questions = []
        for reference in reference_rows:
            question_id = int(reference["question_id"])
            workbook_text = _safe_text(reference.get("question")).strip()
            text = database_text.get(question_id) or workbook_text
            embedding = encode_question(text, DEFAULT_MATHE_EMBEDDING_MODEL)
            documents = score_curricular_pool_candidates(
                question_id=question_id,
                question=text,
                mathe_mirror_client=mathe_client,
                qdrant_client=qdrant_client,
                question_embedding=embedding,
            )
            videos = score_video_pool_candidates(
                question_id=question_id,
                question=text,
                mathe_mirror_client=mathe_client,
                qdrant_client=qdrant_client,
                question_embedding=embedding,
            )
            questions.append(
                {
                    "question_id": question_id,
                    "topic_name": _safe_text(reference.get("topic_name")),
                    "subtopic_name": _safe_text(reference.get("subtopic_name")),
                    "question_text": text,
                    "question_text_source": (
                        "database" if question_id in database_text else "reference_workbook"
                    ),
                    "workbook_text_matches_database": (
                        database_text.get(question_id) == workbook_text
                    ),
                    "expert_document_ids": parse_material_ids(
                        reference.get("related_document_material_ids")
                    ),
                    "no_suitable_documents": _is_yes(reference.get("no_suitable_documents")),
                    "expert_video_ids": parse_material_ids(
                        reference.get("related_video_material_ids")
                    ),
                    "no_suitable_videos": _is_yes(reference.get("no_suitable_videos")),
                    "documents": _snapshot_candidates(
                        documents, "question_to_material_similarity"
                    ),
                    "videos": _snapshot_candidates(videos, "question_to_video_similarity"),
                    "latency": {
                        kind_name: measure_latency(
                            question_id,
                            text,
                            kind_name,
                            mathe_client,
                            qdrant_client,
                            encode,
                            score_question_similarity_for_material_ids,
                            compute_keyword_jaccard,
                            application,
                            timing_repeats,
                            _question_hash,
                        )
                        for kind_name, application in applications.items()
                    },
                }
            )
    finally:
        qdrant_client.close()
        mathe_client.close()

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_git_state(),
        "embedding_model": DEFAULT_MATHE_EMBEDDING_MODEL,
        "timing": {
            "environment_label": environment_label,
            "repeats": timing_repeats,
            "model_warmup_seconds": model_warmup_seconds,
            "client_platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "reference_workbook": str(reference_workbook),
        "reference_workbook_sha256": _sha256(reference_workbook),
        "questions": questions,
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return len(questions)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def evaluation_status(question: dict[str, Any], kind: ContentKind) -> str:
    if not question[kind.pool_field]:
        return f"no_{kind.name}_candidates"
    if not question[kind.expert_field]:
        return f"no_expert_{kind.name}_ground_truth"
    return "evaluated"


def has_unembedded_expert_item(question: dict[str, Any], kind: ContentKind) -> bool:
    """An expert item without an embedding cannot be ranked by the embedding
    signal, which biases the comparison against it."""
    experts = set(question[kind.expert_field])
    return any(
        c["material_id"] in experts and not c["has_embedding"]
        for c in question[kind.pool_field]
    )


def _results_for_population(
    population: str,
    question_ids: list[int],
    metrics: dict[str, dict[int, dict[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results_rows = []
    for name in (*CONFIGURATIONS, RANDOM):
        row: dict[str, Any] = {
            "population": population,
            "questions": len(question_ids),
            "configuration": name,
        }
        for metric in METRIC_NAMES:
            row[f"mean_{metric}"] = mean(metrics[name][q][metric] for q in question_ids)
        for metric in CI_METRICS:
            low, high = bootstrap_mean_ci(
                [metrics[name][q][metric] for q in question_ids], BOOTSTRAP_RESAMPLES, SEED
            )
            row[f"{metric}_95ci_low"] = low
            row[f"{metric}_95ci_high"] = high
        results_rows.append(row)

    comparison_rows = []
    for name in CONFIGURATIONS:
        if name == REFERENCE_CONFIGURATION:
            continue
        for metric in CI_METRICS:
            differences = [
                metrics[name][q][metric] - metrics[REFERENCE_CONFIGURATION][q][metric]
                for q in question_ids
            ]
            low, high = bootstrap_mean_ci(differences, BOOTSTRAP_RESAMPLES, SEED)
            wins, ties, losses = wins_ties_losses(differences)
            comparison_rows.append(
                {
                    "population": population,
                    "questions": len(question_ids),
                    "configuration": name,
                    "metric": metric,
                    "mean_difference_vs_keyword_only": mean(differences),
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
    return results_rows, comparison_rows


def analyze_content(
    questions: list[dict[str, Any]],
    kind: ContentKind,
) -> dict[str, list[dict[str, Any]]] | None:
    """Evaluate one content kind; None when no question can be evaluated.

    Results are reported for two populations: every evaluated question, and
    the questions whose expert items all have an embedding.
    """
    evaluated = [q for q in questions if evaluation_status(q, kind) == "evaluated"]
    if not evaluated:
        return None

    metrics: dict[str, dict[int, dict[str, float]]] = {
        name: {} for name in (*CONFIGURATIONS, RANDOM)
    }
    per_question_rows = []
    for question in evaluated:
        question_id = question["question_id"]
        pool = question[kind.pool_field]
        experts = question[kind.expert_field]
        for name, rank in CONFIGURATIONS.items():
            ranked = rank(pool)
            metrics[name][question_id] = list_metrics(ranked, experts)
            per_question_rows.append(
                {
                    "question_id": question_id,
                    "configuration": name,
                    **metrics[name][question_id],
                    "top_10_material_ids": "; ".join(ranked[:10]),
                    "top_10_expert_hits": "; ".join(m for m in ranked[:10] if m in set(experts)),
                }
            )
        metrics[RANDOM][question_id] = expected_random_metrics(
            pool, experts, RANDOM_PERMUTATIONS, SEED + question_id
        )

    populations = {
        "all_evaluated": [q["question_id"] for q in evaluated],
        "expert_items_embedded": [
            q["question_id"] for q in evaluated if not has_unembedded_expert_item(q, kind)
        ],
    }
    results_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for population in POPULATIONS:
        if populations[population]:
            results, comparisons = _results_for_population(
                population, populations[population], metrics
            )
            results_rows += results
            comparison_rows += comparisons

    coverage_rows = [
        {
            "question_id": q["question_id"],
            "pool_size": len(q[kind.pool_field]),
            "pool_materials_without_embedding": sum(
                not c["has_embedding"] for c in q[kind.pool_field]
            ),
            "expert_ids": "; ".join(q[kind.expert_field]),
            "expert_ids_without_embedding": "; ".join(
                c["material_id"]
                for c in q[kind.pool_field]
                if c["material_id"] in set(q[kind.expert_field]) and not c["has_embedding"]
            ),
            "expert_ids_outside_pool": "; ".join(
                m
                for m in q[kind.expert_field]
                if m not in {c["material_id"] for c in q[kind.pool_field]}
            ),
        }
        for q in evaluated
    ]
    return {
        "results": results_rows,
        "vs_keyword_only": comparison_rows,
        "per_question": per_question_rows,
        "coverage": coverage_rows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, -(-len(ordered) * percentile // 100) - 1))
    return ordered[int(index)]


LIVE_KEYWORD_ONLY_STAGES = ("pool_query_seconds", "keyword_scoring_seconds", "ranking_seconds")
LIVE_WITH_EMBEDDING_STAGES = (
    "pool_query_seconds",
    "keyword_scoring_seconds",
    "question_embedding_seconds",
    "vector_lookup_seconds",
    "ranking_seconds",
)
LATENCY_MEASURES = (
    ("live_with_embeddings_total", LIVE_WITH_EMBEDDING_STAGES,
     "Live scoring as the deployed fallback runs it; every variant except jaccard_only"),
    ("live_keyword_only_total", LIVE_KEYWORD_ONLY_STAGES,
     "Live scoring without embeddings; jaccard_only"),
    ("precomputed_lookup", ("precomputed_lookup_seconds",),
     "Reading the stored ranking, as the API serves it when fresh"),
    ("pool_query", ("pool_query_seconds",), "Stage"),
    ("keyword_scoring", ("keyword_scoring_seconds",), "Stage"),
    ("question_embedding", ("question_embedding_seconds",), "Stage (warm model)"),
    ("vector_lookup", ("vector_lookup_seconds",), "Stage"),
    ("ranking", ("ranking_seconds",), "Stage"),
)


def latency_sheets(
    questions: list[dict[str, Any]],
    kind: ContentKind,
) -> dict[str, list[dict[str, Any]]] | None:
    """Response-time sheets for one content kind; None without timings.

    Covers every reference question with a non-empty pool of this kind:
    latency does not need expert labels. Values are per-question medians over
    the timing repeats, measured from the machine that ran ``fetch``, so they
    include its network round trips to the database and Qdrant.
    """
    timed = [
        (q, q["latency"][kind.name])
        for q in questions
        if (q.get("latency") or {}).get(kind.name) and q[kind.pool_field]
    ]
    if not timed:
        return None

    per_question_rows = []
    totals: dict[str, list[float]] = {name: [] for name, _, _ in LATENCY_MEASURES}
    for question, timing in timed:
        row = {
            "question_id": question["question_id"],
            "pool_size": timing["pool_size"],
            "precomputed_ranking_found": timing["precomputed_ranking_found"],
            "precomputed_hash_matches": timing["precomputed_hash_matches"],
        }
        for name, stages, _ in LATENCY_MEASURES:
            seconds = sum(float(timing[stage]) for stage in stages)
            totals[name].append(seconds)
            row[f"{name}_ms"] = seconds * 1000
        per_question_rows.append(row)

    summary_rows = []
    for name, stages, description in LATENCY_MEASURES:
        values = [v * 1000 for v in totals[name]]
        summary_rows.append(
            {
                "measure": name,
                "description": description,
                "questions": len(values),
                "mean_ms": mean(values),
                "median_ms": median(values),
                "p95_ms": _percentile(values, 95),
                "max_ms": max(values),
            }
        )
    live = totals["live_with_embeddings_total"]
    precomputed = totals["precomputed_lookup"]
    speedups = [l / p for l, p in zip(live, precomputed) if p > 0]
    summary_rows += [
        {
            "measure": "median_speedup_precomputed_vs_live",
            "description": "Per-question live_with_embeddings_total / precomputed_lookup",
            "questions": len(speedups),
            "median_ms": None,
            "value": median(speedups) if speedups else None,
        },
        {
            "measure": "share_with_fresh_precomputed_ranking",
            "description": "Stored ranking exists and matches the question text "
            "(server-side version/generation checks not visible here)",
            "questions": len(timed),
            "value": mean(
                float(t["precomputed_ranking_found"] and t["precomputed_hash_matches"])
                for _, t in timed
            ),
        },
    ]
    return {"latency_summary": summary_rows, "latency_per_question": per_question_rows}


def excluded_questions(questions: list[dict[str, Any]], kind: ContentKind) -> list[dict]:
    return [
        {
            "question_id": q["question_id"],
            "exclusion_reason": evaluation_status(q, kind),
            "topic_name": q["topic_name"],
            "subtopic_name": q["subtopic_name"],
            "pool_size": len(q[kind.pool_field]),
            "no_suitable_flag": q[kind.no_suitable_field],
        }
        for q in questions
        if evaluation_status(q, kind) != "evaluated"
    ]


def analyze_snapshot(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return the workbook sheets, per content kind, as lists of row dicts."""
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema {snapshot.get('schema_version')!r}; "
            f"expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    questions = sorted(snapshot["questions"], key=lambda q: q["question_id"])
    content_sheets: dict[str, list[dict[str, Any]]] = {}
    population_rows = []
    for kind in CONTENT_KINDS:
        sheets = analyze_content(questions, kind)
        excluded = excluded_questions(questions, kind)
        evaluated = [q for q in questions if evaluation_status(q, kind) == "evaluated"]
        population_rows += [
            {"field": f"{kind.name}_evaluated_questions", "value": len(evaluated)},
            {"field": f"{kind.name}_excluded_questions", "value": len(excluded)},
            {
                "field": f"{kind.name}_questions_with_expert_items_without_embedding",
                "value": ", ".join(
                    str(q["question_id"]) for q in evaluated if has_unembedded_expert_item(q, kind)
                ),
            },
        ]
        if sheets is None:
            continue
        for sheet_name, rows in sheets.items():
            content_sheets[f"{kind.sheet_prefix}_{sheet_name}"] = rows
        content_sheets[f"{kind.sheet_prefix}_excluded_questions"] = excluded
        for sheet_name, rows in (latency_sheets(questions, kind) or {}).items():
            content_sheets[f"{kind.sheet_prefix}_{sheet_name}"] = rows
    if not content_sheets:
        raise ValueError("The snapshot has no question with expert IDs and a pool")

    timing = snapshot.get("timing") or {}
    run_info_rows = [
        {"field": "snapshot_created_at", "value": snapshot.get("created_at")},
        {"field": "snapshot_git_commit", "value": snapshot.get("git_commit")},
        {"field": "snapshot_git_dirty", "value": snapshot.get("git_dirty")},
        {"field": "environment", "value": timing.get("environment_label")},
        {"field": "embedding_model", "value": snapshot.get("embedding_model")},
        {"field": "reference_workbook", "value": snapshot.get("reference_workbook")},
        {"field": "reference_workbook_sha256", "value": snapshot.get("reference_workbook_sha256")},
        {"field": "reference_questions", "value": len(questions)},
        *population_rows,
        {
            "field": "questions_text_differs_from_workbook",
            "value": ", ".join(
                str(q["question_id"])
                for q in questions
                if not q.get("workbook_text_matches_database", True)
            ),
        },
        {"field": "keyword_only", "value": "Keyword Jaccard alone; materials with Jaccard 0 left out."},
        {
            "field": "embedding_only",
            "value": "Question-to-material cosine alone; materials with no positive cosine "
            "(including no embedding) left out.",
        },
        {
            "field": "combined",
            "value": "Jaccard and max(cos, 0) min-max normalised within the pool, then averaged; "
            "materials with neither signal left out.",
        },
        {"field": "random", "value": f"Mean over {RANDOM_PERMUTATIONS} seeded random orderings of the pool."},
        {
            "field": "populations",
            "value": "all_evaluated: every question with expert IDs and a pool. "
            "expert_items_embedded: the same without questions where an expert item has no embedding.",
        },
        {
            "field": "relevance",
            "value": "Binary: expert IDs of the content kind are relevant, every other pool "
            "material is not. Ideal DCG uses all expert IDs; precision@5 divides by 5.",
        },
        {
            "field": "confidence_intervals",
            "value": f"Percentile bootstrap over questions, {BOOTSTRAP_RESAMPLES} resamples, seed {SEED}.",
        },
        {"field": "timing_repeats_per_question", "value": timing.get("repeats")},
        {"field": "model_warmup_seconds", "value": timing.get("model_warmup_seconds")},
        {"field": "timing_client_platform", "value": timing.get("client_platform")},
    ]
    return {"run_info": run_info_rows, **content_sheets}


def write_workbook(sheets: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Score the pools once (needs services).")
    fetch.add_argument("--reference-workbook", type=Path, default=DEFAULT_REFERENCE_WORKBOOK)
    fetch.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    fetch.add_argument("--env-file", default=".env")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument(
        "--timing-repeats",
        type=int,
        default=DEFAULT_TIMING_REPEATS,
        help="Runs per question for the latency medians (default 0: no timing; try 5).",
    )
    fetch.add_argument(
        "--environment-label",
        default="",
        help='Where the services were reached, e.g. "dev via VPN".',
    )

    analyze = subparsers.add_parser("analyze", help="Evaluate the configurations from a snapshot.")
    analyze.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    analyze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    analyze.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "fetch":
        if args.snapshot.exists() and not args.overwrite:
            parser.error(f"{args.snapshot} exists; pass --overwrite to replace it")
        from dotenv import load_dotenv

        load_dotenv(args.env_file)
        count = fetch_snapshot(
            args.reference_workbook,
            args.snapshot,
            timing_repeats=args.timing_repeats,
            environment_label=args.environment_label,
        )
        print(f"Wrote pool scores for {count} questions to {args.snapshot}")
    else:
        if args.output.exists() and not args.overwrite:
            parser.error(f"{args.output} exists; pass --overwrite to replace it")
        snapshot = json.loads(args.snapshot.read_text())
        sheets = analyze_snapshot(snapshot)
        write_workbook(sheets, args.output)
        print(f"Wrote {len(sheets)} sheets to {args.output}")


if __name__ == "__main__":
    main()
