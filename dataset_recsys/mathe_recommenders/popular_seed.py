from dataset_recsys.mathe_recommenders.metadata_ocr import (
    MATHE_APPLICATION,
    seed_redis_id,
)
from dataset_recsys.storage.mathe_mirror_client import MatheMirrorClient
from dataset_recsys.storage.recommendation_client import RecommendationClient


def recommend_from_popular_seed(
    question_id: int,
    k: int,
    mathe_mirror_client: MatheMirrorClient,
    recommendation_client: RecommendationClient,
) -> list[str]:
    """Previous MathE flow: most-clicked topic material as Redis seed."""
    material = mathe_mirror_client.get_material_by_question_id(question_id)
    material_id = seed_redis_id(material) if material else None

    if not material_id:
        return []

    return recommendation_client.get_recommendations(
        application=MATHE_APPLICATION,
        entity_id=material_id,
        limit=k,
    )

