# Analytical Patterns

This folder contains the **logic and data models for handling graph-based recommendation requests**.

It enables the API to process **Analytical Patterns (AP)**—JSON-based graph structures. The service extracts recommendation parameters from these graphs, executes the query, and transforms the graph by injecting results directly into the structure.

---

## Key Components

### `ap_handling.py`
The core engine for graph transformation. It handles the lifecycle of an AP request by navigating the nodes and edges of the provided graph.
* **Graph Parsing:** Locates the `DatasetRecommender_Operator` and extracts the `n` parameter.
* **Seed Identification:** Traces the incoming `input` edge to identify the seed dataset ID.
* **Graph Mutation:** Cleanly removes old recommendation nodes and edges before injecting new results.
* **Result Injection:** Dynamically generates new nodes (**sc:Dataset**) and links them to the operator via `output` edges with a `rank` property.

### `models.py`
Defines the Pydantic schemas used for internal data validation and API documentation.
* **`RecsRequest`**: Standardizes the input parameters (`entity_id`, `n`) extracted from the AP.
* **`RecsResponse`**: Structure for the final recommendation payload (recommended dataset IDs).
* **Validation**: Enforces constraints such as limits on the number of recommendations ($n \le 20$).
