# Logging & Exceptions

This folder contains **observability and error-handling utilities** that provide structured visibility into the API's runtime behavior.

It standardizes how the system communicates failures to clients and how it traces requests across distributed services using structured logging and correlation IDs.

Typical logging and exception tasks include:

* **Defining standardized error schemas** for consistent API responses
* **Intercepting HTTP traffic** to log performance metrics and status codes
* **Managing Correlation IDs** to trace requests across microservices
* **Formatting logs into JSON** for ingestion by log management systems (e.g., ELK, Datadog)
* **Handling downstream dependency failures** with specialized exception classes

---

## Key Components

### `exceptions.py`
Standardizes the "shape" of error messages returned by the API. 
* **Custom Exceptions:** Defines `FailedDependencyException` for 424 errors when external services (like OIDC or Gateways) fail.
* **Pydantic Models:** Provides response schemas (`ErrorResponse`, `ValidationErrorResponse`) to ensure the frontend receives predictable error structures.
* **Contextual Metadata:** Captures correlation IDs and downstream payloads within exceptions to simplify debugging.

### `logging_config.py`
Configures `structlog` and FastAPI middleware for high-fidelity observability.
* **Correlation Middleware:** Injects an `x-tracking-correlation` header into every request/response cycle to link logs together.
* **Traffic Logging:** Automatically captures request paths, methods, processing times, and status codes.
* **Structured Formatting:** Implements custom formatters (like `datagems_log_formatter_cf2`) to ensure logs meet organizational JSON standards.

---

## Rule of thumb

Put code here if it **standardizes how the system reports its internal state or failures to the outside world**.