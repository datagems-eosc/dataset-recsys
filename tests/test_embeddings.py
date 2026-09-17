from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np

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


def test_sentence_transformer_encoding_is_serialized(monkeypatch):
    first_encoding_started = Event()
    release_first_encoding = Event()
    active_lock = Lock()
    active_encodings = 0
    max_active_encodings = 0

    class FakeTokenizer:
        def __call__(self, text, truncation=False):
            return {"input_ids": [1]}

    class FakeSentenceTransformer:
        tokenizer = FakeTokenizer()
        max_seq_length = 8192

        def encode(self, texts, **kwargs):
            nonlocal active_encodings, max_active_encodings
            with active_lock:
                active_encodings += 1
                max_active_encodings = max(max_active_encodings, active_encodings)
                is_first_encoding = active_encodings == 1 and not first_encoding_started.is_set()

            if is_first_encoding:
                first_encoding_started.set()
                assert release_first_encoding.wait(timeout=2)

            try:
                return np.ones((len(texts), 2))
            finally:
                with active_lock:
                    active_encodings -= 1

    model = FakeSentenceTransformer()
    monkeypatch.setattr(
        embeddings,
        "_load_sentence_transformer_model",
        lambda model_name: model,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            embeddings._encode_sentence_transformer_texts,
            ["first"],
            "BAAI/bge-m3",
        )
        assert first_encoding_started.wait(timeout=2)
        second = executor.submit(
            embeddings._encode_sentence_transformer_texts,
            ["second"],
            "BAAI/bge-m3",
        )
        release_first_encoding.set()

        first.result(timeout=2)
        second.result(timeout=2)

    assert max_active_encodings == 1
