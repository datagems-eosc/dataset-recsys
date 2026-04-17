from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient
from datetime import datetime
from dataset_recsys.utils.bedrock import enrich_batch

async def process_incremental_update(dataset_profile, application: str, enrichment_llm: str = "claude-sonnet-4-6", prompt_version: str = "catalog_summary_v1", embedding_model: str = "allenai/specter2_base"):
    emb_client = EmbeddingClient()

    if emb_client.exists(dataset_profile.id):
        print(f"Dataset {dataset_profile.id} already exists in catalog. Skipping incremental flow.")
        return False # Indicate no update was performed
    
    print(f"Processing incremental update for dataset {dataset_profile.id} in application '{application}'")
    
    # 1. Enrich (LLM step - generate catalog summary)
    enriched_list = enrich_batch([dataset_profile], enrichment_llm, prompt_version)
    enriched_profile = enriched_list[0] if enriched_list else dataset_profile
    
    # 2. Generate Embedding
    from dataset_recsys.embeddings import build_embedding_text, encode_texts
    text_input = build_embedding_text(enriched_profile)
    vector = encode_texts([text_input], embedding_model)[0].tolist()
    
    # 3. Update Vector DB
    emb_client = EmbeddingClient()
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
    
    # 4. Outbound Recs (What does this new dataset recommend?)
    neighbors = emb_client.find_similar(application, vector, top_k=20)
    recs_client = RecommendationClient()

    # TODO: see what we should do with the scores (distances)
    
    # Map for Redis: {neighbor_id: score} -> using 1 - distance as a simple score
    # new_recs = {res[0]: (1.0 - res[1]) for res in neighbors if res[0] != dataset_profile.id}
    # recs_client.update_single_entity_recs(application, dataset_profile.id, new_recs)
    
    # # 5. Update existing neighbors (Inbound)
    # # If the new dataset is close to them, it should appear in their lists
    # for neighbor_id, distance in neighbors:
    #     if neighbor_id == dataset_profile.id: continue
        
    #     score = 1.0 - distance
    #     recs_client.update_neighbor_recs(
    #         application=application,
    #         neighbor_id=neighbor_id,
    #         new_entity_id=dataset_profile.id,
    #         score=score
    #     )
    
    return True

if __name__ == "__main__":
    # Example usage for testing
    from dataset_recsys.ingestion.moma_dataset import MomaDataset
    moma = MomaDataset()
    moma.get_from_external("some-dataset-id")
    profile = moma.to_dataset_profile()
    import asyncio
    asyncio.run(process_incremental_update(profile, application="portal"))