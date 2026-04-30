# Storage

This folder contains code that **stores and retrieves recommendation results and embeddings used by the system**.

## Rule of thumb

Put code here if it **handles where and how data is stored or retrieved**.

Examples:

* Clients that read/write recommendations from Redis
* Clients that store/query embeddings in PostgreSQL + pgvector
* Utilities that manage keys, indexes, schemas, or lookup operations


## Redis Storage (Recommendations)

Recommendations are stored in Redis using an **application → entity → recommended entities** structure.

### Key patterns

1. **Recommendation set for an entity**

Redis key:

```
recs:{application}:{entity_id}
```

Redis value:

A **Redis Set** containing the IDs of recommended entities.

Example:

```
recs:mathe:6.pdf
```

Value:

```
{7.pdf, 9.pdf, 221.pdf, 52.pdf, ...}
```

Meaning:

```
For application "mathe", entity "6.pdf" has recommendations [7.pdf, 9.pdf, 221.pdf, ...]
```

---

2. **Application index of stored entities**

Redis key:

```
recs:index:{application}
```

Redis value:

A **Redis Set** containing all entity IDs for which recommendations exist in that application.

Example:

```
recs:index:mathe
```

Value:

```
{6.pdf, 65.pdf, 221.pdf, ...}
```

Meaning:

```
All entities in the "mathe" application that have stored recommendations
```

This index is used to:

- list all entities for an application
- delete all recommendations for an application
- perform reverse lookups

---

### Example full structure

Example Redis keys for the **MathE recommender**:

```
recs:mathe:ds2ds
recs:mathe:6.pdf
recs:mathe:65.pdf
recs:mathe:221.pdf
```

Example Redis keys for the **Datagems recommender**:

```
recs:index:ds2ds
recs:ds2ds:meteo_era5land
recs:ds2ds:wikipedia
```

---

### Conceptual model

The system implements an **entity-to-entity recommender**:

```
application
    └── entity_id
            └── recommended_entity_ids
```

Examples:

MathE materials:

```
6.pdf → {7.pdf, 9.pdf, 221.pdf}
```

Datagems:

```
meteo_era5land → {weather_stations_climpact, wikipedia}
```

This design allows the same storage layer to support multiple applications (e.g., MathE materials, datasets, or other entity collections).

---

# PostgreSQL + pgvector Storage (Embeddings)

Embeddings are stored in PostgreSQL using the **pgvector extension**, enriched with metadata for reproducibility and traceability.

---

## Table schema

Table:

```
{schema}.dataset_embeddings
```

Columns:

| Column          | Type         | Description                               |
| --------------- | ------------ | ----------------------------------------- |
| application     | TEXT         | Logical grouping (e.g. ds2ds, mathe)      |
| dataset_id      | TEXT (PK)    | Unique identifier of the dataset          |
| embedding       | VECTOR(1536) | Embedding vector                          |
| embedding_input | TEXT         | Input text used to generate the embedding |
| embedding_model | TEXT         | Model used to generate embeddings         |
| enrichment_llm  | TEXT         | LLM used for enrichment (optional)        |
| prompt_version  | TEXT         | Prompt version used (optional)            |
| run_id          | TEXT         | Workflow run identifier                   |
| created_at      | TIMESTAMP    | Last update timestamp                     |

Primary key:

```
PRIMARY KEY (dataset_id)
```

---

## Conceptual model

```
application
    └── dataset_id
            ├── embedding (vector)
            ├── embedding_input (text)
            ├── embedding_model
            ├── enrichment_llm
            ├── prompt_version
            └── run_id
```

---

## Storage behavior

* Embeddings are written **in bulk**
* Existing embeddings for an application are **deleted before insert**
* Upserts are handled via:

```
ON CONFLICT (dataset_id)
```

* Metadata is always updated together with the embedding

---

## Similarity search

Similarity queries use pgvector distance:

```
embedding <-> query_embedding
```

Example:

```
SELECT dataset_id, embedding <-> %s AS distance
FROM dataset_embeddings
WHERE application = %s
ORDER BY embedding <-> %s
LIMIT %s;
```