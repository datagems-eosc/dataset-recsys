import asyncio
from threading import Event

from dataset_recsys.api.routes import mathe


def test_video_recommend_endpoint_uses_video_pool_ranker(monkeypatch):
    calls = {}
    fake_mathe_client = object()
    fake_embedding_client = object()

    async def fake_authorized_entity_ids(token):
        calls["token"] = token
        return {mathe.MATHE_DATASET_ID: "dg_ds-browse"}

    def fake_recommend_videos_for_question(**kwargs):
        calls["recommender"] = kwargs
        return ["901", "902"]

    def fail_document_recommender(**kwargs):
        raise AssertionError("document recommender must not serve the video route")

    monkeypatch.setattr(
        mathe.security,
        "get_authorized_entity_ids",
        fake_authorized_entity_ids,
    )
    monkeypatch.setattr(mathe, "get_mathe_client", lambda: fake_mathe_client)
    monkeypatch.setattr(mathe, "get_embedding_client", lambda: fake_embedding_client)
    monkeypatch.setattr(
        mathe,
        "recommend_videos_for_question",
        fake_recommend_videos_for_question,
    )
    monkeypatch.setattr(
        mathe,
        "recommend_from_curricular_pool",
        fail_document_recommender,
    )

    response = asyncio.run(
        mathe.get_video_recommendations(
            request=mathe.MatheRecsRequest(
                question_id="272",
                question="Explain the chain rule.",
                n=2,
            ),
            claims={"sub": "user-1"},
            token="token-1",
        )
    )

    assert calls["token"] == "token-1"
    assert calls["recommender"] == {
        "question_id": 272,
        "question": "Explain the chain rule.",
        "k": 2,
        "mathe_mirror_client": fake_mathe_client,
        "embedding_client": fake_embedding_client,
    }
    assert response.question_id == "272"
    assert [recommendation.material_id for recommendation in response.recommendations] == [
        "901",
        "902",
    ]


def test_video_route_is_separate_from_document_routes():
    routes_by_path = {route.path: route for route in mathe.router.routes}

    video_route = routes_by_path["/dataset-recsys/mathe/recommend/videos"]
    document_route = routes_by_path["/dataset-recsys/mathe/recommend/documents"]
    legacy_route = routes_by_path["/dataset-recsys/mathe/recommend"]

    assert video_route.endpoint is mathe.get_video_recommendations
    assert document_route.endpoint is mathe.get_document_recommendations
    assert legacy_route.endpoint is mathe.get_document_recommendations
    assert video_route.endpoint is not document_route.endpoint


def test_recommendation_ranking_does_not_block_event_loop(monkeypatch):
    recommender_started = Event()
    release_recommender = Event()

    async def fake_authorized_entity_ids(token):
        return {mathe.MATHE_DATASET_ID: "dg_ds-browse"}

    def blocking_recommender(**kwargs):
        recommender_started.set()
        assert release_recommender.wait(timeout=2)
        return ["901"]

    monkeypatch.setattr(
        mathe.security,
        "get_authorized_entity_ids",
        fake_authorized_entity_ids,
    )
    monkeypatch.setattr(mathe, "get_mathe_client", lambda: object())
    monkeypatch.setattr(mathe, "get_embedding_client", lambda: object())

    async def run_request():
        task = asyncio.create_task(
            mathe._get_content_recommendations(
                request=mathe.MatheRecsRequest(
                    question_id="272",
                    question="Explain the chain rule.",
                    n=1,
                ),
                claims={"sub": "user-1"},
                token="token-1",
                content_type="video",
                recommender=blocking_recommender,
            )
        )

        for _ in range(100):
            if recommender_started.is_set():
                break
            await asyncio.sleep(0.01)

        assert recommender_started.is_set()
        release_recommender.set()
        return await task

    response = asyncio.run(run_request())

    assert [item.material_id for item in response.recommendations] == ["901"]
