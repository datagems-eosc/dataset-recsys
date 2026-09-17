from concurrent.futures import ThreadPoolExecutor
from threading import Event

from dataset_recsys import embeddings


def test_sentence_transformer_disables_meta_tensor_loading(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))
            self.max_seq_length = None

    embeddings._load_sentence_transformer_model_cached.cache_clear()
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    try:
        model = embeddings._load_sentence_transformer_model("BAAI/bge-m3")
    finally:
        embeddings._load_sentence_transformer_model_cached.cache_clear()

    assert isinstance(model, FakeSentenceTransformer)
    assert calls == [
        (
            "BAAI/bge-m3",
            {
                "device": embeddings.DEVICE,
                "model_kwargs": {"low_cpu_mem_usage": False},
            },
        )
    ]
    assert model.max_seq_length == 8192


def test_sentence_transformer_is_loaded_once_for_concurrent_requests(monkeypatch):
    first_load_started = Event()
    release_first_load = Event()
    created_models = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            created_models.append(self)
            first_load_started.set()
            assert release_first_load.wait(timeout=2)
            self.max_seq_length = None

    embeddings._load_sentence_transformer_model_cached.cache_clear()
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                embeddings._load_sentence_transformer_model,
                "BAAI/bge-m3",
            )
            assert first_load_started.wait(timeout=2)
            second = executor.submit(
                embeddings._load_sentence_transformer_model,
                "BAAI/bge-m3",
            )
            release_first_load.set()

            first_model = first.result(timeout=2)
            second_model = second.result(timeout=2)
    finally:
        release_first_load.set()
        embeddings._load_sentence_transformer_model_cached.cache_clear()

    assert first_model is second_model
    assert created_models == [first_model]
