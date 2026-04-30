from dataset_recsys.storage.embedding_client import EmbeddingClient
from dataset_recsys.storage.recommendation_client import RecommendationClient
import structlog

logger = structlog.get_logger(__name__)

async def dataset_removal(
    entity_id: str, 
    application: str, 
    recs_client: RecommendationClient, 
    emb_client: EmbeddingClient
) -> bool:
    """
    Orchestrates the removal of a dataset from the entire system.
    """
    # 1. Check existence
    if not emb_client.exists(entity_id):
        return False

    logger.info(f"Starting incremental removal for: {entity_id}")

    referring_ids = recs_client.find_entities_recommending(application, entity_id)
    for neighbor_id in referring_ids:
        if neighbor_id != entity_id:
            logger.info(f"Removing reference to {entity_id} from {neighbor_id}")
            recs_client.remove_from_neighbor_recs(application, neighbor_id, entity_id)    

    # 3. Remove from Redis (Outbound and Index)
    recs_client.remove_single_entity_recs(application, entity_id)

    # 4. Remove from Vector DB
    emb_client.delete_single_embedding(entity_id)

    logger.info(f"Successfully removed {entity_id} from all storage layers.")

    return True

if __name__ == "__main__":
    # Example usage for testing
    application = "ds2ds"
    entity_id = "b573d56a-6e74-4b7b-bbce-e1c4ea847572"

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
    asyncio.run(dataset_removal(entity_id, application=application, recs_client=recs_client, emb_client=embedding_client))