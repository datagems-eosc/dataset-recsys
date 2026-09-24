from qdrant_client import QdrantStorageClient, models

# Connect to Qdrant via local port-forward
client = QdrantStorageClient(host="localhost", port=6333)
collection_name = "dataset_embeddings"

# 1. Check collection stats & total points
collection_info = client.get_collection(collection_name=collection_name)
print(f"Status: {collection_info.status}")
print(f"Total Points (Vectors): {collection_info.points_count}")

# 2. Retrieve sample points to check payloads & recommendations
records, _ = client.scroll(
    collection_name=collection_name,
    limit=2,
    with_payload=True,
    with_vectors=False,
)

print("\n--- Sample Point Payloads ---")
for record in records:
    print(f"Point ID: {record.id}")
    print(f"Dataset ID: {record.payload.get('dataset_id')}")
    print(f"Application field: {record.payload.get('application')}")
    
    # Updated: Handle nested recommendations payload
    recs_payload = record.payload.get("recommendations", {})
    if isinstance(recs_payload, dict):
        print("Recommendations by application:")
        for app, rec_list in recs_payload.items():
            print(f"  - [{app}] Count: {len(rec_list)} | Sample: {rec_list[:2]}")
    else:  # Fallback for flat lists or recommendations_<app> style
        print(f"Stored Recs Count: {len(recs_payload)}")
        print(f"Sample Recs: {recs_payload[:2]}\n")

# 3. Test filtered vector retrieval / search
# Get vector size dynamically
vectors_config = collection_info.config.params.vectors
if hasattr(vectors_config, "size"):
    vector_size = vectors_config.size
elif isinstance(vectors_config, dict) and "size" in vectors_config:
    vector_size = vectors_config["size"]
else:
    # Handles named vectors setup by taking the first vector dimension
    vector_size = list(vectors_config.values())[0].size

search_results = client.query_points(
    collection_name=collection_name,
    query=[0.0] * vector_size,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="application",
                match=models.MatchValue(value="ds2ds")
            )
        ]
    ),
    limit=3,
).points

print("\n--- Test Search Query ---")
print(f"Returned {len(search_results)} results for 'ds2ds' application filter.")