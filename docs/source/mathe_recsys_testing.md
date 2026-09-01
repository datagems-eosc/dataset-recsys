# MathE RecSys API Testing

Testing the deployed or local MathE API endpoint for a single question ID.

## Local API

Start Redis port-forwarding:

```bash
kubectl port-forward svc/<REDIS_SERVICE_NAME> -n <NAMESPACE> 6380:6379
```

Start the local API from the repository root:

```bash
poetry run python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); os.environ['REDIS_HOST']='localhost'; os.environ['REDIS_PORT']='6380'; os.environ['OIDC_AUDIENCE']='dataset-recsys-api'; import uvicorn; uvicorn.run('dataset_recsys.api.main:app', host='127.0.0.1', port=8001, reload=True)"
```

Create a token from the repository root:

```bash
TOKEN=$(poetry run python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os, requests; r=requests.post(os.environ['DATAGEMS_AUTH_URL'], data={'grant_type':'password','client_id':os.environ['DATAGEMS_CLIENT_ID'],'username':os.environ['DATAGEMS_USER'],'password':os.environ['DATAGEMS_PASSWORD'],'scope':os.environ['DATAGEMS_SCOPE']}); print(r.json().get('access_token',''))")
```

Call the local API:

```bash
curl -s -X POST "http://127.0.0.1:8001/dataset-recsys/mathe/recommend/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": "82",
    "question": "Differentiate y = (2x^3 - 5x)^5.",
    "n": 10
  }' | jq
```

Call the separate video endpoint with the same request shape:

```bash
curl -s -X POST "http://127.0.0.1:8001/dataset-recsys/mathe/recommend/videos" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": "82",
    "question": "Differentiate y = (2x^3 - 5x)^5.",
    "n": 10
  }' | jq
```

## Deployed API

Use the deployed API base URL for the environment you want to test:

```bash
API_BASE_URL="<DATASET_RECSYS_API_BASE_URL>"

curl -s -X POST "$API_BASE_URL/dataset-recsys/mathe/recommend/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": "82",
    "question": "Differentiate y = (2x^3 - 5x)^5.",
    "n": 10
  }' | jq
```

The response returns MathE database material IDs:

```json
{
  "question_id": "82",
  "recommendations": [
    {"material_id": "221"}
  ]
}
```

`/dataset-recsys/mathe/recommend` remains available as a deprecated alias for
clients that have not yet moved to the explicit document path.

The document and video endpoints use different embedding namespaces and cannot
return mixed recommendation lists.
