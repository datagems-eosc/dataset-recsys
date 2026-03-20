
# Storage

This folder contains code that **stores and retrieves recommendation results used by the system**.

## Rule of thumb

Put code here if it **handles where and how recommendations are stored or retrieved**.

Examples:
- Scripts that ingest recommendation JSON files into Redis
- Clients that read recommendations from Redis
- Utilities that manage recommendation keys, indexes, or lookup operations

## Redis Storage Schema

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

Example Redis keys for the **MathE materials recommender**:

```
recs:index:mathe
recs:mathe:6.pdf
recs:mathe:65.pdf
recs:mathe:221.pdf
```

Example Redis keys for the **dataset portal recommender**:

```
recs:index:portal
recs:portal:meteo_era5land
recs:portal:wikipedia
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

Dataset portal:

```
meteo_era5land → {weather_stations_climpact, wikipedia}
```

This design allows the same storage layer to support multiple applications (e.g., MathE materials, dataset portal, or other entity collections).