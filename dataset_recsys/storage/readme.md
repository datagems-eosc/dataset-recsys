# Storage

This folder contains code that **stores and retrieves recommendation results and embeddings used by the system**.

## Rule of thumb

Put code here if it **handles where and how data is stored or retrieved**.

Examples:

* The Qdrant client that stores embeddings and recommendation rankings
* The read-only client for the MathE mirror database in PostgreSQL
* The local MathE synchronization catalog stored in SQLite
* Utilities that manage stored recommendations

## MathE synchronization catalog

`mathe_sync_catalog.py` owns the local `sync_entries` SQLite table. It stores
document OCR and video transcript processing state, applies transitional schema
migrations, reconciles the catalog with the currently available document and
video sources, and exposes completed entries to the MathE refresh pipeline.

Filesystem discovery, document conversion, video transcription, and workflow
orchestration remain outside the storage layer.


## Qdrant storage (embeddings and recommendations)

`qdrant_client.py` contains `QdrantStorageClient`, the only component that
talks to Qdrant. It stores one point per entity. A point holds the entity's
embedding and, in its payload, the entity's stored ranking of related entities.

### Collections and applications

| Collection | Application | Entity | Vector size |
|---|---|---|---|
| `dataset_embeddings` | `ds2ds` | DataGEMS dataset | 768 (`allenai/specter2_base`) |
| `mathe_embeddings` | `mathe_documents` | MathE document | 1024 (`BAAI/bge-m3`) |
| `mathe_embeddings` | `mathe_videos` | MathE video | 1024 (`BAAI/bge-m3`) |

An application is a namespace inside a collection. Every search and every
listing filters on it, so documents and videos share a collection without ever
appearing in each other's results.

### Point structure

The point ID is a UUID derived deterministically from the entity ID, so the
same entity always maps to the same point.

```json
{
  "id": "<uuid5 of the entity ID>",
  "vector": [0.012, -0.044, "..."],
  "payload": {
    "application": "mathe_documents",
    "dataset_id": "221",
    "embedding_input": "<the text that was embedded>",
    "embedding_model": "BAAI/bge-m3",
    "enrichment_llm": "none",
    "prompt_version": "none",
    "run_id": "2026-11-11T20:00:00Z",
    "recommendations": {
      "mathe_documents": [
        {"id": "222", "score": 0.83},
        {"id": "305", "score": 0.79}
      ]
    }
  }
}
```

`dataset_id` holds the entity ID for every application: a DataGEMS dataset ID,
or a MathE platform material ID. `enrichment_llm` and `prompt_version` are set
for datasets, whose embedded text includes a model-written summary.

### Behaviour worth knowing

- **Reads on an empty index do not fail.** Before the first rebuild there are
  no collections. Reads then report "nothing stored": status `NOT_FOUND`, empty
  lists, `False` for existence.
- **`store_embeddings(..., replace=True)` makes a rebuild authoritative.** After
  writing the new embeddings it deletes the application's entities that are not
  in the new set. Without `replace`, it only adds or overwrites. An empty batch
  never deletes anything; use `delete_application` to clear on purpose.
- **Storing an embedding resets that entity's recommendations.** Writing a
  point replaces its payload, so rankings are stored after embeddings.
- **A vector size mismatch is refused.** If embeddings of a different size are
  sent to an existing collection, the client raises an error and writes
  nothing. Changing the embedding model means dropping the collection
  deliberately and running a full rebuild.
- **`find_similar(top_k=None)` returns the complete ranking** of an application,
  computed exactly. With an integer `top_k` the search is approximate.
- **`get_recommendations` raises `KeyError` for an unknown entity;
  `get_recommendations_with_scores` returns an empty list.** The first is used
  where an unknown entity is an error, the second where a material without
  stored neighbours simply has none.

### Main operations

| Operation | Method |
|---|---|
| Store embeddings in bulk | `store_embeddings` |
| Store one embedding | `upsert_single_embedding` |
| Store rankings in bulk | `store_recommendations` |
| Replace one entity's ranking | `update_single_entity_recs` |
| Insert an entity into another entity's ranking | `update_neighbor_recs` |
| Remove an entity and every reference to it | `remove_dataset` |
| Rank entities against a query vector | `find_similar` |
| Score a given set of entities against a query vector | `find_similar_by_ids` |
| Read an entity's ranking | `get_recommendations`, `get_recommendations_with_scores` |
| Check what is stored | `exists`, `check_existence_batch`, `get_entity_status`, `list_entities`, `count_entities` |

### Command-line helper

`manage_recommendations.py` lists, reads, ingests and deletes stored
recommendations. Run it as a module from the repository root:

```bash
python -m dataset_recsys.storage.manage_recommendations list-entities ds2ds
python -m dataset_recsys.storage.manage_recommendations get ds2ds <dataset_id> --limit 5
```

It connects to the Qdrant instance given by `QDRANT_HOST` and `QDRANT_PORT`.

### Tests

`tests/clients/test_qdrant_client.py` runs the client against Qdrant's
in-memory mode, so it needs no server.

## MathE mirror database (PostgreSQL)

`mathe_mirror_client.py` reads question and material metadata (topic, subtopic,
keywords, material type) from the MathE mirror database. The recommender only
reads from it.
