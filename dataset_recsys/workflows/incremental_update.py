from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient
from datetime import datetime
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

async def process_incremental_update(dataset_profile, application: str, enrichment_llm: str = "claude-sonnet-4-6", prompt_version: str = "catalog_summary_v1", embedding_model: str = "allenai/specter2_base", recs_client: RecommendationClient | None = None, emb_client: EmbeddingClient | None = None) -> bool:

    # 1. Existence Check
    if emb_client.exists(dataset_profile.id):
        return False

    # 2. LLM Enrichment
    enriched_list = enrich_batch([dataset_profile])
    enriched_profile = enriched_list[0]

    # 3. Embedding Generation
    from dataset_recsys.embeddings import build_embedding_text, encode_texts
    text_input = build_embedding_text(enriched_profile)
    vector = encode_texts([text_input], model_name=embedding_model)[0].tolist()
    
    # 4. Storage in Vector DB
    emb_client.upsert_single_embedding(
        application=application,
        dataset_id=dataset_profile.id,
        embedding=vector,
        embedding_input=text_input,
        metadata={
            "llm": enrichment_llm,
            "prompt": prompt_version,
            "model": embedding_model,
            "run_id": f"inc_{datetime.now().strftime('%Y%m%d')}"
        }
    )
    
    # 5. NEW DATASET RECS (Outbound)
    # Store the full ranked neighbor list; API endpoints decide how many to return.
    neighbors = emb_client.find_similar(application, vector, top_k=None)

    outbound_recs = {
        row[0]: float(row[1]) 
        for row in neighbors if row[0] != enriched_profile.id
    }
    recs_client.update_single_entity_recs(application, enriched_profile.id, outbound_recs)

    # 6. NEIGHBOR UPDATES (Inbound)
    # Recompute recommendations ONLY for the neighbors affected by the new dataset
    for neighbor_id, similarity_score in outbound_recs.items():
        # Inject the new dataset into the neighbor's Redis ZSET
        # Redis ZADD will automatically place it in the correct rank
        recs_client.update_neighbor_recs(
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
    redis_host, redis_port, redis_db = "localhost", 6379, 0
    recs_client = RecommendationClient(host=redis_host, port=redis_port, db=redis_db)
    embedding_client = EmbeddingClient(
        host="localhost",
        port=5433,
        dbname="postgres",
        user="postgres",
        password="postgres"
    )
    print("Redis OK:", recs_client.check_connection())
    print("Embedding DB OK:", embedding_client.check_connection())    
    import asyncio
    asyncio.run(process_incremental_update(profile, application=application, enrichment_llm=enrichment_llm, prompt_version=prompt_version, embedding_model=embedding_model, recs_client=recs_client, emb_client=embedding_client))
