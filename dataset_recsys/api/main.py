import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dataset_recsys.api.logging.exceptions import (
    ErrorResponse,
    FailedDependencyException,
    FailedDependencyMessage,
    FailedDependencyResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
)
from dataset_recsys.api.logging.logging_config import (
    correlation_id_middleware,
    request_response_logging_middleware,
    setup_logging,
)
from dataset_recsys.api.routes.datagems import router as datagems_router
from dataset_recsys.api.routes.health import router as health_router
from dataset_recsys.api.routes.mathe import router as mathe_router
from dataset_recsys.api.routes.admin import router as admin_router

setup_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    openapi_url="/dataset-recsys/openapi.json",
    docs_url="/dataset-recsys/docs",
    redoc_url="/dataset-recsys/redoc",
)

# --- Middleware ---
app.middleware("http")(request_response_logging_middleware)
app.middleware("http")(correlation_id_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(datagems_router)
app.include_router(mathe_router)
app.include_router(admin_router)
app.include_router(health_router)


# --- Exception Handlers ---
@app.exception_handler(FailedDependencyException)
async def failed_dependency_exception_handler(
    request: Request, exc: FailedDependencyException
):
    response_content = FailedDependencyResponse(
        code=104,
        error="error communicating with underpinning service",
        message=FailedDependencyMessage(
            statusCode=exc.downstream_status_code,
            source=exc.source,
            correlationId=exc.correlation_id,
            payload=exc.downstream_payload,
        ),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content.model_dump(exclude_none=True),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=exc.status_code, error=exc.detail).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        ValidationErrorDetail(
            Key=".".join(map(str, err.get("loc", []))), Value=[err.get("msg", "")]
        )
        for err in exc.errors()
    ]
    response_content = ValidationErrorResponse(
        code=102, error="Validation Error", message=details
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_content.model_dump(),
    )


# ===========================
# LOCAL DEVELOPMENT GUIDE
# ===========================

# 1. Ensure Kubernetes access is configured
# If you previously set a wrong KUBECONFIG, unset it:
# unset KUBECONFIG
#
# Check available contexts:
# kubectl config get-contexts
# kubectl config current-context

# 2. Make sure you are connected to the required VPN/network
# (cluster is on private IP, e.g., 172.x.x.x)

# 3. Find the current Redis pod (names CHANGE over time!)
# kubectl get pods -n athenarc | grep dataset-recsys-redis
#
# Example output:
# dataset-recsys-redis-66d8f4c4b4-p9bc2

# 4. Port-forward Redis (use CURRENT pod name)
# kubectl port-forward pod/<REDIS_POD_NAME> -n athenarc 6380:6379
#
# Example:
# kubectl port-forward pod/dataset-recsys-redis-66d8f4c4b4-p9bc2 -n athenarc 6380:6379

# (Optional) Prefer service if available (more stable than pod names):
# kubectl get svc -n athenarc | grep redis
# kubectl port-forward svc/dataset-recsys-redis -n athenarc 6380:6379

# 5. Test Redis connection
# redis-cli -p 6380
# PING  -> should return PONG

# ===========================
# RUN THE API
# ===========================

# Activate environment
# conda activate <env_name>

# Run the API
# OIDC_AUDIENCE=dataset-recsys-api REDIS_HOST=localhost REDIS_PORT=6380 uvicorn dataset_recsys.api.main:app --reload

# Open docs:
# http://127.0.0.1:8000/dataset-recsys/docs
# http://127.0.0.1:8000/dataset-recsys/redoc

# ===========================
# TEST ENDPOINTS
# ===========================

# Health check
# curl -X GET "http://127.0.0.1:8000/dataset-recsys/health" -v

#
# Example datagems' dataset id for testing:
# 07382b91-5bc5-42f9-8391-33adc2460c19
#
# To obtain a Bearer token (valid for a few minutes), run:
# TOKEN=$(curl --location "$DATAGEMS_AUTH_URL" \
#   --header 'Content-Type: application/x-www-form-urlencoded' \
#   --data-urlencode "grant_type=password" \
#   --data-urlencode "client_id=$DATAGEMS_CLIENT_ID" \
#   --data-urlencode "username=$DATAGEMS_USER" \
#   --data-urlencode "password=$DATAGEMS_PASSWORD" \
#   --data-urlencode "scope=$DATAGEMS_SCOPE" | jq -r '.access_token')
#
# Then call the API with:
# curl -X POST "http://127.0.0.1:8000/dataset-recsys/recommend?entity_id=07382b91-5bc5-42f9-8391-33adc2460c19&n=5" \
#      -H "Authorization: Bearer $TOKEN"

# curl -X POST "http://127.0.0.1:8000/dataset-recsys/recommend/ap" \
#   -H "Authorization: Bearer $TOKEN" \
#   -H "Content-Type: application/json" \
#   -d @dataset_recsys/api/api_docs/ap_request_example.json

# MathE recommendations
# curl -X POST "http://127.0.0.1:8000/dataset-recsys/mathe/recommend?entity_id=6.pdf&n=5"
