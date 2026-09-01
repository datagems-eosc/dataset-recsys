from dataset_recsys.storage.recommendation_client import RecommendationClient


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def zadd(self, *args):
        self.calls.append(("zadd", args))

    def zremrangebyrank(self, *args):
        self.calls.append(("zremrangebyrank", args))

    def execute(self):
        self.calls.append(("execute", ()))


class FakeRedis:
    def __init__(self):
        self.pipeline_instance = FakePipeline()
        self.calls = []

    def pipeline(self):
        return self.pipeline_instance

    def sismember(self, *_args):
        return True

    def zrevrange(self, *args, **kwargs):
        self.calls.append(("zrevrange", args, kwargs))
        if kwargs.get("withscores"):
            return [("8.pdf", 0.8), ("7.pdf", 0.7)]
        return ["8.pdf", "7.pdf"]


def test_normalize_recommendations_accepts_scored_pairs():
    client = RecommendationClient.__new__(RecommendationClient)

    normalized = client._normalize_recommendations(
        [("8.pdf", 0.8), ("7.pdf", 0.0)]
    )

    assert normalized == {"8.pdf": 0.8, "7.pdf": 0.0}


# Backward compatibility: if only ordered IDs are provided,
# assign descending rank-derived scores so Redis preserves the order.
def test_normalize_recommendations_still_accepts_ranked_id_lists():
    client = RecommendationClient.__new__(RecommendationClient)

    normalized = client._normalize_recommendations(["8.pdf", "7.pdf"])

    assert normalized == {"8.pdf": 2.0, "7.pdf": 1.0}


def test_normalize_recommendations_still_accepts_score_maps():
    client = RecommendationClient.__new__(RecommendationClient)

    normalized = client._normalize_recommendations({"8.pdf": 0.8, "7.pdf": 0.0})

    assert normalized == {"8.pdf": 0.8, "7.pdf": 0.0}


def test_normalize_recommendations_accepts_empty_inputs():
    client = RecommendationClient.__new__(RecommendationClient)

    assert client._normalize_recommendations([]) == {}
    assert client._normalize_recommendations({}) == {}


# This test ensures that when the limit is None, the method does not attempt to trim the sorted set.
def test_update_neighbor_recs_does_not_trim_when_limit_is_none():
    client = RecommendationClient.__new__(RecommendationClient)
    client.r = FakeRedis()

    client.update_neighbor_recs(
        application="mathe",
        neighbor_id="6.pdf",
        new_entity_id="8.pdf",
        score=0.8,
        limit=None,
    )

    calls = client.r.pipeline_instance.calls
    assert ("zadd", ("recs:mathe:6.pdf", {"8.pdf": 0.8})) in calls
    assert not any(name == "zremrangebyrank" for name, _args in calls)


# This test ensures that when a limit is provided, the method attempts to trim the sorted set 
# to maintain only the top N recommendations, which involves a call to zremrangebyrank.
def test_get_recommendations_can_limit_redis_range():
    client = RecommendationClient.__new__(RecommendationClient)
    client.r = FakeRedis()

    recommendations = client.get_recommendations(
        application="mathe",
        entity_id="6.pdf",
        limit=2,
    )

    assert recommendations == ["8.pdf", "7.pdf"]
    assert client.r.calls == [("zrevrange", ("recs:mathe:6.pdf", 0, 1), {})]


# This test ensures that recommendaitons with scores are correctly retrieved from Redis 
# when the withscores option is used, and that the correct Redis command is issued.
def test_get_recommendations_with_scores_returns_redis_scores():
    client = RecommendationClient.__new__(RecommendationClient)
    client.r = FakeRedis()

    recommendations = client.get_recommendations_with_scores(
        application="mathe",
        entity_id="6.pdf",
        limit=2,
    )

    assert recommendations == [("8.pdf", 0.8), ("7.pdf", 0.7)]
    assert client.r.calls == [
        (
            "zrevrange",
            ("recs:mathe:6.pdf", 0, 1),
            {"withscores": True},
        )
    ]
