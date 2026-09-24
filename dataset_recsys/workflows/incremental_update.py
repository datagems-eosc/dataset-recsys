from dataset_recsys.storage.qdrant_client import QdrantStorageClient
from datetime import datetime, timezone
from dataset_recsys.utils.bedrock import enrich_batch
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_access_token() -> str:
    """
    Retrieves an OAuth2 access token using the Password Grant flow.
    """
    payload = {
        "grant_type": "password",
        "client_id": os.getenv("DATAGEMS_CLIENT_ID"),
        "username": os.getenv("DATAGEMS_USER"),
        "password": os.getenv("DATAGEMS_PASSWORD"),
        "scope": os.getenv("DATAGEMS_SCOPE", "openid profile email"),
    }
    
    response = requests.post(
        os.getenv("DATAGEMS_AUTH_URL"), 
        data=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"Failed to retrieve token: {response.status_code} - {response.text}")
        response.raise_for_status()
        
    return response.json()["access_token"]

async def process_incremental_update(
    dataset_profile,
    application: str,
    enrichment_llm: str = "claude-sonnet-4-6",
    prompt_version: str = "catalog_summary_v1",
    embedding_model: str = "allenai/specter2_base",
    qdrant_client: QdrantStorageClient | None = None,
) -> bool:
    if qdrant_client is None:
        qdrant_client = QdrantStorageClient()

    # 1. Existence Check
    if qdrant_client.exists(dataset_profile.id):
        return False

    # 2. LLM Enrichment
    enriched_list = enrich_batch([dataset_profile])
    enriched_profile = enriched_list[0]

    # 3. Embedding Generation
    from dataset_recsys.embeddings import build_embedding_text, encode_texts

    text_input = build_embedding_text(enriched_profile)
    vector = encode_texts([text_input], model_name=embedding_model)[0].tolist()

    # 4. Storage in Qdrant Vector DB
    qdrant_client.upsert_single_embedding(
        application=application,
        dataset_id=dataset_profile.id,
        embedding=vector,
        embedding_input=text_input,
        metadata={
            "llm": enrichment_llm,
            "prompt": prompt_version,
            "model": embedding_model,
            "run_id": f"inc_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        },
    )

    # 5. NEW DATASET RECS (Outbound)
    # Perform vector similarity search in Qdrant
    neighbors = qdrant_client.find_similar(
        application=application, 
        query_vector=vector, 
        top_k=100
    )

    outbound_recs = {
        res["dataset_id"]: float(res["score"])
        for res in neighbors
        if res["dataset_id"] != enriched_profile.id
    }
    qdrant_client.update_single_entity_recs(
        application=application, 
        entity_id=enriched_profile.id, 
        recommendations=outbound_recs
    )

    # 6. NEIGHBOR UPDATES (Inbound)
    # Inject the new dataset into each neighbor's Qdrant recommendation payload
    for neighbor_id, similarity_score in outbound_recs.items():
        qdrant_client.update_neighbor_recs(
            application=application,
            neighbor_id=neighbor_id,
            new_entity_id=enriched_profile.id,
            score=similarity_score,
            limit=None,
        )

    return True

if __name__ == "__main__":
    # Example usage for testing
    enrichment_llm = "claude-sonnet-4-6"
    prompt_version = "catalog_summary_v1"
    embedding_model = "allenai/specter2_base"
    application = "ds2ds"
    
    from dataset_recsys.ingestion.moma_dataset import MomaDataset
    moma = MomaDataset(get_access_token())
    moma.get_from_external("07382b91-5bc5-42f9-8391-33adc2460c19")
    profile = moma.to_dataset_profile()

    qdrant_client = QdrantStorageClient(host="localhost", port=6333)
    print("Qdrant DB Connection OK:", qdrant_client.check_connection())

    import asyncio
    asyncio.run(process_incremental_update(profile, application=application, enrichment_llm=enrichment_llm, prompt_version=prompt_version, embedding_model=embedding_model, qdrant_client=qdrant_client))
