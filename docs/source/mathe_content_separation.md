# MathE Document and Video Separation

## Contract

MathE document and video recommendations must never be blended into the same
recommendation list.

The target logical collections are:

```text
mathe_documents
mathe_videos
```

Collection names are defined centrally by the string-compatible
`MatheApplication` enum in
`dataset_recsys/mathe_recommenders/constants.py`. The same file maps MathE's
numeric video types to the dynamic subtype values received during discovery.

The existing `mathe` collection remains active during the staged migration.

## API Contract

The existing endpoint remains document-only and backward compatible:

```text
POST /dataset-recsys/mathe/recommend
```

The target explicit endpoints are:

```text
POST /dataset-recsys/mathe/recommend/documents
POST /dataset-recsys/mathe/recommend/videos
```

The current endpoint will become an alias of the document endpoint. A mixed
document-and-video endpoint is intentionally out of scope.

## Identity Contract

`platform_materials.id` is the canonical API identity for both documents and
videos. Source identifiers such as a filename or YouTube ID remain separate
attributes and must not replace the MathE material ID in API responses.

Video lesson and video review are subtypes of the `video` content kind. They may
share a video recommendation list, but neither subtype may appear in a document
recommendation list.

Legacy `type: audio` rows are migrated to `type: video` without changing their
text or status. Their subtype remains null until PostgreSQL discovery resolves
them to lesson or review. Audio is only an intermediate artifact: the pipeline
downloads a video's audio, transcribes it, and embeds the resulting transcript.

## Stage 1: Sync Catalog Classification

Stage 1 adds explicit identity and classification fields to `syncer.db`:

```text
platform_material_id
content_subtype
```

The migration is additive and backfills existing rows without removing OCR or
transcript state. Video discovery now reads the MathE material ID, link, and
type, allowing the catalog to retain the MathE material ID separately from the
YouTube ID. The compatibility functions live in `mathe_sync_migrations.py` and
can be deleted, together with their single syncer call, after every deployed
catalog has been upgraded.

The existing `id` column remains the processing and legacy-index compatibility
key, while `type` consistently contains `document` or `video`.
`platform_material_id` is the canonical identity for the split indexes and
later document and video endpoints.

## Stage 2: Separate Index Generation

The refresh pipeline encodes each completed OCR document or video transcript
once and reuses the vector when publishing these collections:

```text
mathe            legacy processing IDs; retained during migration
mathe_documents  document platform_material_id values only
mathe_videos     video platform_material_id values only
```

Nearest-neighbor recommendations are computed independently for each
collection. Document IDs can therefore never appear in a `mathe_videos` list,
and video IDs can never appear in a `mathe_documents` list. Empty split
collections are cleared in pgvector and Redis so results from an earlier run do
not remain stale.

A completed legacy video with no `platform_material_id` remains available in
the transitional `mathe` collection but is not published to `mathe_videos`.
The next successful PostgreSQL discovery pass supplies its MathE ID and subtype.

The legacy collection construction and processing-ID normalization live in
`mathe_index_migrations.py`. Once all consumers use a split namespace, that
module and its single pipeline hook can be deleted without changing permanent
document/video index generation.

## Stage 3: Explicit Document API

The document recommender is available at:

```text
POST /dataset-recsys/mathe/recommend/documents
```

The existing `/dataset-recsys/mathe/recommend` route is a deprecated alias of
the same handler, so existing clients continue receiving the same document-only
response. Both routes use canonical `platform_materials.id` values and score
question similarity against the `mathe_documents` embedding namespace.

The request-time curricular pool remains restricted to MathE material type `3`
and the supported PDF, DOCX, and PPTX formats.

## Stage 4: Explicit Video API

The video recommender is available at:

```text
POST /dataset-recsys/mathe/recommend/videos
```

It builds a hard curricular pool containing only MathE video lessons and
reviews (`platform_materials.type` values `1` and `2`) assigned to the same
topic and subtopic as the question. It embeds the question and scores only the
platform material IDs in that pool against the `mathe_videos` collection.

Videos without a published `mathe_videos` embedding are omitted because there
is no transcript-based score with which to rank them. The response contains
canonical `platform_materials.id` values. The document endpoints continue to
query only `mathe_documents`, so neither endpoint can return the other content
kind.

Before enabling the endpoint in an environment, run the Stage 2 refresh so the
environment contains `mathe_videos` embeddings. If the collection is empty,
the endpoint returns no recommendations rather than falling back to documents
or the mixed legacy collection.

## Post-Migration Cleanup

Code-level `TODO` markers identify the remaining compatibility work. Cleanup is
complete only after both deployed data and API consumers have migrated:

- delete `mathe_sync_migrations.py` and its syncer hook after every SQLite
  catalog has the Stage 1 fields and backfill;
- delete `mathe_index_migrations.py` and its pipeline hook after no consumer
  reads the mixed `mathe` collection;
- require non-null `platform_material_id` values in the split-index loader;
- remove the legacy `MatheApplication.LEGACY` value, pipeline summary value,
  and transitional embedding application reassignment;
- remove `/dataset-recsys/mathe/recommend` only after every API client uses
  `/dataset-recsys/mathe/recommend/documents`.
