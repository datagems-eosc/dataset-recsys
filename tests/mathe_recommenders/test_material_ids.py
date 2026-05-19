from dataset_recsys.storage.mathe_mirror_client import (
    material_id_to_redis_id,
    redis_id_to_material_id,
)


def test_mathe_material_redis_id_maps_to_db_material_id():
    assert material_id_to_redis_id(221) == "221.pdf"
    assert material_id_to_redis_id("221.pdf") == "221.pdf"
    assert redis_id_to_material_id("221.pdf") == 221
    assert redis_id_to_material_id("ChainRule.pdf") is None
