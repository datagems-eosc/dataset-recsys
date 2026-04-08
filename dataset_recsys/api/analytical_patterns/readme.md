# Analytical Patterns

This folder contains the **logic and data models for handling graph-based recommendation requests**.

It enables the API to process **Analytical Patterns (AP)**—JSON-based graph structures representing complex research workflows. The service extracts recommendation parameters from these graphs, executes the query, and transforms the graph by injecting results directly into the structure.

---

## Key Components

### `ap_handling.py`
The core engine for graph transformation. It handles the lifecycle of an AP request by navigating the nodes and edges of the provided graph.
* **Graph Parsing:** Locates the `DatasetRecommender_Operator` and extracts the `application` and `n` parameters.
* **Conditional Logic:** * **MathE:** Extracts the `entity_id` directly from the operator's properties.
    * **Portal:** Traces the incoming `input` edge to identify the seed dataset ID.
* **Graph Mutation:** Cleanly removes old recommendation nodes and edges before injecting new results.
* **Result Injection:** Dynamically generates new nodes (**sc:Dataset** for Portal or **cr:FileObject** for MathE) and links them to the operator via `output` edges with a `rank` property.

### `models.py`
Defines the Pydantic schemas used for internal data validation and API documentation.
* **`RecsRequest`**: Standardizes the input parameters (`application`, `entity_id`, `n`) extracted from the AP.
* **`RecsResponse`**: Structure for the final recommendation payload, including metadata like `query_time`.
* **Validation**: Enforces constraints such as supported application names (**portal**, **mathe**) and limits on the number of recommendations ($n \le 20$).

---

## Rule of thumb

Put code here if it **deals with the translation, parsing, or structural transformation of Graph-based Analytical Patterns**.