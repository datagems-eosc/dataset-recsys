from dataset_recsys.mathe_recommenders.constants import (
    MatheApplication,
    VIDEO_TYPE_TO_SUBTYPE,
)


def test_mathe_collection_names_are_explicit_and_distinct():
    assert MatheApplication.LEGACY == "mathe"
    assert MatheApplication.DOCUMENTS == "mathe_documents"
    assert MatheApplication.VIDEOS == "mathe_videos"
    assert VIDEO_TYPE_TO_SUBTYPE == {1: "video_lesson", 2: "video_review"}
    assert len(set(MatheApplication)) == 3
