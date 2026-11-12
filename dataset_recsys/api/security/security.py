from typing import List, Optional, Set, Dict

import httpx
import structlog
from dataset_recsys.api.security.config import settings
from dataset_recsys.api.logging.exceptions import (
    FailedDependencyException,
)
from dataset_recsys.api.logging.logging_config import (
    get_correlation_id,
)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

logger = structlog.get_logger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
_oidc_config = None
_jwks_keys = None


async def get_oidc_config():
    global _oidc_config
    if _oidc_config is None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(settings.OIDC_CONFIG_URL)
                response.raise_for_status()
                _oidc_config = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Failed to fetch OIDC configuration due to HTTP error",
                url=str(e.request.url),
                status_code=e.response.status_code,
                response=e.response.text,
            )
            try:
                payload = e.response.json()
            except Exception:
                payload = e.response.text
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=e.response.status_code,
                correlation_id=get_correlation_id(),
                payload=payload,
                detail="Authentication service returned an error.",
            )
        except httpx.RequestError as e:
            logger.error(
                "Failed to fetch OIDC configuration due to network error",
                url=settings.OIDC_CONFIG_URL,
                error=str(e),
            )
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=503,
                correlation_id=get_correlation_id(),
                payload={"error": f"Network error: {type(e).__name__}"},
                detail="Authentication service is unavailable.",
            )
    return _oidc_config


async def get_jwks_keys():
    """Fetches and caches the JSON Web Key Set (JWKS) containing public keys."""
    global _jwks_keys
    if _jwks_keys is None:
        oidc_config = await get_oidc_config()
        jwks_uri = oidc_config.get("jwks_uri")
        if not jwks_uri:
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=500,
                detail="jwks_uri not found in OIDC config.",
                correlation_id=get_correlation_id(),
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(jwks_uri)
                response.raise_for_status()
                _jwks_keys = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Failed to fetch JWKS keys due to HTTP error",
                url=str(e.request.url),
                status_code=e.response.status_code,
                response=e.response.text,
            )
            try:
                payload = e.response.json()
            except Exception:
                payload = e.response.text
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=e.response.status_code,
                correlation_id=get_correlation_id(),
                payload=payload,
                detail="Could not fetch public keys for token validation.",
            )
        except httpx.RequestError as e:
            logger.error(
                "Failed to fetch JWKS keys due to network error",
                url=jwks_uri,
                error=str(e),
            )
            raise FailedDependencyException(
                source="OIDCProvider",
                status_code=503,
                correlation_id=get_correlation_id(),
                payload={"error": f"Network error: {type(e).__name__}"},
                detail="Could not fetch public keys for token validation.",
            )
    return _jwks_keys


async def get_current_user_claims(token: str = Depends(oauth2_scheme)) -> dict:
    """
    A FastAPI dependency that validates the JWT and returns its claims.
    This will be applied to protected endpoints.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        keys = await get_jwks_keys()
        payload = jwt.decode(
            token,
            keys,
            algorithms=["RS256"],
            audience=settings.OIDC_AUDIENCE,
            issuer=settings.OIDC_ISSUER_URL,
        )
        return payload
    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e))
        raise credentials_exception
    except Exception as e:
        logger.error(
            "An unexpected error occurred during token validation", error=str(e)
        )
        raise credentials_exception


# Realm roles that authorize any action on any dataset. They bypass both the
# per-endpoint role gate below and the per-dataset context-grant check.
PRIVILEGED_DATASET_ROLES = {"dg_admin", "dg_system", "dg_dataset-curator"}


def has_privileged_dataset_role(claims: dict) -> bool:
    """True if the caller holds a realm role that bypasses per-dataset grant checks."""
    user_roles = set(claims.get("realm_access", {}).get("roles", []))
    return bool(user_roles & PRIVILEGED_DATASET_ROLES)


def require_role(required_roles: List[str]):
    """
    A FastAPI dependency that checks if the user has at least one of the required
    roles, or any privileged role (admin / system / curator).
    """

    def role_checker(claims: dict = Depends(get_current_user_claims)) -> dict:
        user_roles = set(claims.get("realm_access", {}).get("roles", []))

        # Accept privileged roles immediately before checking required roles
        if user_roles & PRIVILEGED_DATASET_ROLES:
            logger.info(
                "Authorization granted via privileged role.",
                UserId=claims.get("sub"),
                ClientId=claims.get("clientid"),
            )
            return claims

        if not user_roles.intersection(required_roles):
            accepted_roles = sorted(set(required_roles) | PRIVILEGED_DATASET_ROLES)
            log_context = {
                "required_roles": accepted_roles,
                "UserId": claims.get("sub"),
                "user_roles": list(user_roles),
            }
            client_id = claims.get("clientid")
            if client_id:
                log_context["ClientId"] = client_id

            logger.warning(
                "Authorization failed: User missing any of the required roles",
                **log_context,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have any of the required roles: {', '.join(accepted_roles)}.",
            )

        return claims

    return role_checker


async def get_authorized_entity_ids(
    claims: dict, token: str, action: str = "dg_ds-browse"
) -> Optional[Dict[str, str]]:
    """
    Calls the DataGEMS Gateway to get the entity IDs a user can access.
    Keeps the original return type: dict[str, str] (mapping targetId -> role).
    
    Returns None if the caller holds a privileged realm role, signalling
    that downstream filtering can be entirely bypassed.
    """
    if has_privileged_dataset_role(claims):
        logger.info(
            "Entity listing authorized via privileged role; skipping context-grant filter.",
            action=action,
            UserId=claims.get("sub"),
        )
        return None

    log = logger.bind()

    try:
        api_url = f"{settings.GATEWAY_API_URL}/api/principal/context-grants/query"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"roles": [action]}

        log = log.bind(gateway_url=api_url, gateway_payload=payload)

        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            context_grants = response.json()
            # Retain the original dictionary return mapping targetId -> role
            entity_ids = {
                grant["targetId"]: grant["role"]
                for grant in context_grants
                if grant.get("targetType") == 0  # 0 indicates an entity
                and "targetId" in grant
                and "role" in grant
            }

            log.info(
                "Successfully fetched user permissions from Gateway.",
                total_grants=len(context_grants),
                entity_grants=len(entity_ids),
            )
            return entity_ids

    except httpx.HTTPStatusError as e:
        log.error(
            "Gateway returned an error when fetching entity permissions.",
            status_code=e.response.status_code,
            response=e.response.text,
        )
        return {}
    except Exception as e:
        log.error(
            "An unexpected error occurred while fetching entity permissions.",
            error=str(e),
        )
        return {}