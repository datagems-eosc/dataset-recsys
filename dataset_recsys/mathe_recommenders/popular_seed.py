from dataset_recsys.mathe_recommenders.constants import MatheApplication
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.qdrant_client import QdrantStorageClient


def recommend_from_popular_seed(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    qdrant_client: QdrantStorageClient,
) -> list[str]:
    """Previous MathE flow using the most-clicked document as its seed."""
    material = mathe_mirror_client.get_popular_document_for_question(question_id)
    material_id = str(material["material_id"]).strip() if material else None

    if not material_id:
        return []

    return qdrant_client.get_recommendations(
        application=MatheApplication.DOCUMENTS,
        entity_id=material_id,
        limit=k,
        collection_name=qdrant_client.COLLECTION_MATHE,
    )