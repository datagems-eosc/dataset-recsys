from dataset_recsys.storage.qdrant_client import QdrantStorageClient
import structlog

logger = structlog.get_logger(__name__)

async def dataset_removal(
    entity_id: str,
    application: str,
    qdrant_client: QdrantStorageClient | None = None,
) -> bool:
    """
    Orchestrates the removal of a dataset from the Qdrant storage layer.
    """
    if qdrant_client is None:
        qdrant_client = QdrantStorageClient()

    # 1. Check existence
    if not qdrant_client.exists(entity_id):
        return False

    logger.info(f"Starting incremental removal for: {entity_id}")

    # 2. Clean inbound references from neighbors
    referring_ids = qdrant_client.find_entities_recommending(application, entity_id)
    for neighbor_id in referring_ids:
        if neighbor_id != entity_id:
            logger.info(f"Removing reference to {entity_id} from {neighbor_id}")
            qdrant_client.remove_from_neighbor_recs(application, neighbor_id, entity_id)

    # 3. Remove entity's own recommendation list
    qdrant_client.remove_single_entity_recs(application, entity_id)

    # 4. Delete vector embedding point from Qdrant
    qdrant_client.delete_single_embedding(entity_id)

    logger.info(f"Successfully removed {entity_id} from Qdrant storage.")

    return True

if __name__ == "__main__":
    # Example usage for testing
    application = "ds2ds"
    entity_id = "b573d56a-6e74-4b7b-bbce-e1c4ea847572"

    qdrant_client = QdrantStorageClient(host="localhost", port=6333)
    print("Qdrant DB Connection OK:", qdrant_client.check_connection())

    import asyncio
    asyncio.run(dataset_removal(entity_id, application=application, qdrant_client=qdrant_client))