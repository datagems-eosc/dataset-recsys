# Security

This folder contains **security and configuration scripts that manage authentication and authorization** for the API.

It ensures that only verified users and systems can interact with the recommendation engine by validating JSON Web Tokens (JWT) against an OpenID Connect (OIDC) provider.

Typical security and config steps include:

* **Fetching OIDC metadata** from the identity provider
* **Caching public keys (JWKS)** for efficient token validation
* **Decoding and validating JWTs** to verify user identity
* **Enforcing role-based access control (RBAC)** on API endpoints
* **Resolving entity-level permissions** via external gateway services
* **Managing environment variables** and application settings

---

## Key Components

### `security.py`
The core logic for protecting the API. It defines FastAPI dependencies that can be injected into routes to ensure requests are both authenticated and authorized.
* **Token Validation:** Uses `jose` and `httpx` to validate tokens against the configured OIDC issuer.
* **Role Checking:** Provides `require_role()` to restrict access based on user claims.
* **Permission Resolution:** Includes `get_authorized_entity_ids()` to fetch fine-grained data access rights from the DataGEMS Gateway.

### `config.py`
A centralized configuration hub powered by `pydantic-settings`.
* **Environment Management:** Automatically loads variables from `.env` files.
* **OIDC Settings:** Defines the issuer URL, audience, and dynamically constructs the OIDC configuration endpoint.
* **Type Safety:** Ensures all required secrets (like `IdpClientSecret`) are present at runtime.

---

## Rule of thumb

Put code here if it **deals with identity, and/or access management settings**.