"""DataFinder benchmark of the dataset-to-dataset recommender: S1 vs S2.

    S1  service pipeline without the summary step:
        build_raw_embedding_text (title. description. tasks) -> specter2_base
    S2  service pipeline with a SCAYLE summary:
        catalog_summary_v1 on qwen3.6 -> LightTextPreprocessor
        -> "<title>. <summary>" -> specter2_base

Both rank all 5,548 corpus datasets by cosine similarity (rank_similar_entities)
and are scored on every query dataset with at least one co-citation link
(DataFinder.get_links_from_queries). See evaluation/datafinder/README.md.

Steps (run from the repository root):

    # 1. Summaries: SCAYLE over the VPN, resumable, safe to stop and rerun
    poetry run python -m evaluation.datafinder.run_benchmark summarize --workers 8

    # 2. Embeddings, rankings, metrics (S1 always; S2 once summaries exist)
    poetry run python -m evaluation.datafinder.run_benchmark evaluate

Nothing is written to Qdrant, Redis, PostgreSQL or any DataGEMS service.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# evaluate forks git after tokenizing; avoid the huggingface/tokenizers fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data.datafinder import DataFinder
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile
from dataset_recsys.utils.text_preprocessing import (
    LightTextPreprocessor,
    preprocess_profiles_field,
)
from evaluation.datafinder.benchmark import (
    MAX_CUTOFF,
    build_profiles,
    evaluate_rankings,
    most_used_datasets,
    paired_bootstrap,
    popularity_rankings,
    random_rankings,
    restrict_links,
    restrict_to_tail,
    summarize,
    summarize_by_bucket,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_CSV = ROOT / "data" / "datafinder" / "results.csv"

EMBEDDING_MODEL = "allenai/specter2_base"
SUMMARY_LLM = "qwen3.6"
PROMPT_VERSION = "catalog_summary_v1"
EMBED_CHUNK = 64
BOOTSTRAP_SEED = 42
PRIMARY_METRICS = ("recall@10", "ndcg@10")
# Long-tail settings: drop the most-used datasets (by corpus usage count)
# as candidates, linked datasets and query datasets.
TAIL_SETTINGS = {"tail_excl_top1pct": 0.01, "tail_excl_top5pct": 0.05}


def summaries_path() -> Path:
    return OUTPUT_DIR / f"summaries_scayle_{SUMMARY_LLM}_{PROMPT_VERSION}.jsonl"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_benchmark() -> tuple[list[DatasetProfile], dict[str, set[str]], DataFinder]:
    """Corpus profiles (descriptions cleaned) and co-citation links."""
    datafinder = DataFinder()
    corpus = datafinder.get()["corpus"]
    profiles = build_profiles(corpus)
    # Same cleaning of the raw description as notebooks/ds2ds_demo.ipynb:
    # DataFinder descriptions carry markdown (**bold**, links).
    profiles = preprocess_profiles_field(profiles, field_name="description")
    links = restrict_links(datafinder.get_links_from_queries(), [p.id for p in profiles])
    return profiles, links, datafinder


def raw_data_hashes(datafinder: DataFinder) -> dict[str, str]:
    data_dir = Path(datafinder.dataset_folder) / "data"
    hashes = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


# --------------------------------------------------------------------------- #
# Summaries (SCAYLE)
# --------------------------------------------------------------------------- #
def read_summaries(path: Path) -> dict[str, dict]:
    """Latest record per dataset ID; a later success overrides an earlier failure."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            previous = records.get(record["id"])
            if previous and previous.get("summary") and not record.get("summary"):
                continue
            records[record["id"]] = record
    return records


def cmd_summarize(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # scayle.py reads SCAYLE_* at import time
    from dataset_recsys.utils import scayle

    profiles, _, _ = load_benchmark()
    path = summaries_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {i for i, r in read_summaries(path).items() if r.get("summary")}
    todo = [p for p in profiles if p.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} summaries already done, {len(todo)} to request "
          f"({SUMMARY_LLM}, {PROMPT_VERSION}, workers={args.workers})", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    started = time.perf_counter()
    counts = {"ok": 0, "failed": 0}

    def summarize_one(profile: DatasetProfile) -> dict:
        t0 = time.perf_counter()
        try:
            enriched = scayle.enrich_data_profile(
                profile, llm=SUMMARY_LLM, prompt_version=PROMPT_VERSION
            )
            summary = (enriched.catalog_summary or "").strip()
            error = "" if summary else "empty response"
        except Exception as exc:  # recorded and retried on the next run
            summary, error = "", f"{type(exc).__name__}: {exc}"[:500]
        return {
            "id": profile.id,
            "summary": summary,
            "error": error,
            "llm": SUMMARY_LLM,
            "prompt_version": PROMPT_VERSION,
            "seconds": round(time.perf_counter() - t0, 2),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # scayle.py prints one line per request (and per retry); silence it so the
    # progress line stays readable. Errors are kept in the JSONL file.
    progress = sys.__stdout__
    with contextlib.redirect_stdout(open(os.devnull, "w")), \
            ThreadPoolExecutor(max_workers=args.workers) as pool, \
            path.open("a", encoding="utf-8") as out:
        futures = [pool.submit(summarize_one, p) for p in todo]
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            counts["ok" if record["summary"] else "failed"] += 1
            if n % 25 == 0 or n == len(todo):
                elapsed = time.perf_counter() - started
                rate = n / elapsed
                eta_min = (len(todo) - n) / rate / 60 if rate else float("nan")
                print(f"[{n}/{len(todo)}] ok={counts['ok']} failed={counts['failed']} "
                      f"{rate:.2f}/s, ~{eta_min:.0f} min left", file=progress, flush=True)
    if counts["failed"]:
        print(f"{counts['failed']} failed; rerun the same command to retry them.")


# --------------------------------------------------------------------------- #
# Embeddings and rankings
# --------------------------------------------------------------------------- #
def embed(texts: list[str], cache_name: str) -> np.ndarray:
    """Service encoder (encode_texts) in chunks; cached by text content."""
    from dataset_recsys.embeddings import encode_texts

    digest = hashlib.sha256("\x1f".join([EMBEDDING_MODEL, *texts]).encode()).hexdigest()[:16]
    cache = OUTPUT_DIR / "cache" / f"{cache_name}_{digest}.npy"
    if cache.exists():
        print(f"{cache_name}: using cached embeddings {cache.name}")
        return np.load(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    t0 = time.perf_counter()
    for start in range(0, len(texts), EMBED_CHUNK):
        chunks.append(encode_texts(texts[start:start + EMBED_CHUNK], model_name=EMBEDDING_MODEL))
        done = min(start + EMBED_CHUNK, len(texts))
        if done % (EMBED_CHUNK * 10) == 0 or done == len(texts):
            print(f"{cache_name}: embedded {done}/{len(texts)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    embeddings = np.vstack(chunks)
    np.save(cache, embeddings)
    return embeddings


def rank(ids: list[str], embeddings: np.ndarray, query_ids: set[str], top_k: int) -> dict[str, list[str]]:
    from dataset_recsys.retrieval import rank_similar_entities

    ranked = rank_similar_entities(ids, embeddings, top_k=top_k)
    return {q: [i for i, _ in ranked[q]] for q in query_ids}


def token_lengths(texts: list[str]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def reference_rows() -> pd.DataFrame:
    """BM25 and SPECTER rows of results.csv, quoted as they stand (not rerun)."""
    ref = pd.read_csv(RESULTS_CSV)
    ref = ref[ref["Model"].isin(["BM25", "SPECTER (pre-trained)"])]
    rows = []
    for _, r in ref.iterrows():
        row = {"system": f"results.csv: {r['Model']}", "n_queries": None}
        for k in (10, 20, 50):
            row[f"recall@{k}"] = r[f"Recall@{k}"]
            row[f"tndcg@{k}"] = r[f"NDCG@{k}"]  # results.csv NDCG is tndcg_at_n
        rows.append(row)
    return pd.DataFrame(rows)


def git_state() -> dict[str, str | bool]:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def cmd_evaluate(args: argparse.Namespace) -> None:
    import torch
    import transformers

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from dataset_recsys.embeddings import build_embedding_text, build_raw_embedding_text

    profiles, links, datafinder = load_benchmark()
    ids = [p.id for p in profiles]
    query_ids = set(links)
    print(f"Corpus {len(ids)} datasets, {len(links)} query datasets, "
          f"{sum(len(v) for v in links.values()) // 2} linked pairs")

    texts: dict[str, list[str]] = {"S1": [build_raw_embedding_text(p) for p in profiles]}

    summaries = read_summaries(summaries_path())
    with_summary = {i for i, r in summaries.items() if r.get("summary")}
    missing = [i for i in ids if i not in with_summary]
    if summaries and len(missing) <= args.max_missing:
        cleaner = LightTextPreprocessor()
        s2_profiles = [
            replace(p, catalog_summary=cleaner.clean(summaries[p.id]["summary"])) if p.id in with_summary else p
            for p in profiles
        ]
        # A dataset without a summary falls back to its raw-metadata text.
        texts["S2"] = [
            build_embedding_text(p) if p.id in with_summary else build_raw_embedding_text(p)
            for p in s2_profiles
        ]
        print(f"S2: {len(with_summary)} summaries, {len(missing)} missing (raw-text fallback)")
    else:
        print(f"S2 skipped: {len(with_summary)}/{len(ids)} summaries, {len(missing)} missing "
              f"(allowed: {args.max_missing}). Rerun `summarize` to fill them, or raise "
              f"--max-missing to embed the missing ones from raw metadata.")

    corpus = datafinder.get()["corpus"]
    popularity = dict(zip(corpus["id"].astype(str).str.strip(), corpus["popularity"]))
    heads = {name: most_used_datasets(popularity, frac) for name, frac in TAIL_SETTINGS.items()}
    # Deep enough that MAX_CUTOFF items remain after removing the largest head.
    depth = MAX_CUTOFF + max((len(h) for h in heads.values()), default=0)

    deep: dict[str, dict[str, list[str]]] = {
        # Chance level: a seeded uniform shuffle of the candidates.
        "Random": random_rankings(ids, query_ids, top_k=depth, seed=BOOTSTRAP_SEED),
        # Non-personalised: the most-used datasets (corpus `popularity`, Papers
        # with Code usage counts) for everyone. Usage counts come from papers,
        # the same kind of evidence as the links, so this baseline is optimistic.
        "Popularity": popularity_rankings(popularity, query_ids, top_k=depth),
    }
    lengths = {}
    for system, system_texts in texts.items():
        deep[system] = rank(ids, embed(system_texts, system), query_ids, top_k=depth)
        lengths[system] = token_lengths(system_texts)

    per_query = []
    populations = {}
    for setting, head in {"full": set(), **heads}.items():
        for system, rankings in deep.items():
            r, l = restrict_to_tail(rankings, links, head, top_k=MAX_CUTOFF)
            per_query.append(evaluate_rankings(system, r, l).assign(setting=setting))
        populations[setting] = {"removed_most_used": len(head), "query_datasets": len(l),
                                "linked_pairs": sum(len(v) for v in l.values()) // 2}
    per_query_df = pd.concat(per_query, ignore_index=True)
    per_query_df.insert(0, "setting", per_query_df.pop("setting"))
    per_query_df.to_csv(OUTPUT_DIR / "per_query.csv", index=False)

    metrics = summarize(per_query_df)
    metrics = pd.concat([metrics, reference_rows().assign(setting="full")], ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    buckets = summarize_by_bucket(per_query_df[per_query_df.setting == "full"], list(PRIMARY_METRICS))
    buckets.to_csv(OUTPUT_DIR / "metrics_by_link_count.csv", index=False)

    comparison = []
    if "S2" in texts:
        for setting in populations:
            subset = per_query_df[per_query_df.setting == setting]
            for m in (*PRIMARY_METRICS, "recall@50", "ndcg@50"):
                comparison.append({"setting": setting,
                                   **asdict(paired_bootstrap(subset, "S1", "S2", m, seed=BOOTSTRAP_SEED))})
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))

    truncation = {
        s: {"texts": len(v), "over_512_tokens": int(sum(n > 512 for n in v)),
            "median_tokens": float(np.median(v))}
        for s, v in lengths.items()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "raw_data_sha256": raw_data_hashes(datafinder),
        "population": {
            "corpus_datasets": len(ids),
            "query_datasets": len(links),
            "linked_pairs": sum(len(v) for v in links.values()) // 2,
            "labels": "co-citation in DataFinder paper positives (train+test), get_links_from_queries",
            "settings": populations,
        },
        "embedding": {"model": EMBEDDING_MODEL, "pooling": "mean", "max_length": 512,
                      "torch": torch.__version__, "transformers": transformers.__version__},
        "summaries": {"provider": "SCAYLE", "llm": SUMMARY_LLM, "prompt_version": PROMPT_VERSION,
                      "file": summaries_path().name, "with_summary": len(with_summary),
                      "missing_raw_fallback": len(missing) if "S2" in texts else None},
        "ranking": {"scorer": "cosine, rank_similar_entities", "top_k": MAX_CUTOFF,
                    "ranked_depth_before_tail_filter": depth, "self_excluded": True},
        "token_lengths": truncation,
        "bootstrap": {"resamples": 10_000, "seed": BOOTSTRAP_SEED},
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pd.set_option("display.width", 200)
    cols = ["setting", "system", "n_queries", "recall@10", "ndcg@10", "tndcg@10", "recall@50", "ndcg@50", "hit@10", "rr@50"]
    print("\n" + metrics[[c for c in cols if c in metrics]].to_string(index=False, float_format="%.4f"))
    print("\n" + buckets.to_string(index=False, float_format="%.4f"))
    for c in comparison:
        print(f"[{c['setting']}] S2-S1 {c['metric']}: {c['mean_difference']:+.4f} "
              f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] W/T/L {c['wins']}/{c['ties']}/{c['losses']}")
    print(f"\nOutputs in {OUTPUT_DIR}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarize", help="Request SCAYLE summaries (resumable).")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=0, help="Only request this many (for a trial run).")
    s.set_defaults(func=cmd_summarize)
    e = sub.add_parser("evaluate", help="Embed, rank and score S1 (and S2 if summaries exist).")
    e.add_argument("--max-missing", type=int, default=0,
                   help="Run S2 only if at most this many datasets lack a summary; "
                        "those fall back to raw metadata text (default 0).")
    e.set_defaults(func=cmd_evaluate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
