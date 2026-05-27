from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from chonky import ParagraphSplitter
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from sentence_transformers import SentenceTransformer
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_CONFIG = {
    "allenai/specter2_base": {
        "backend": "transformers",
        "max_length": 512,
    },
    "BAAI/bge-m3": {
        "backend": "sentence_transformers",
        "max_length": 8192,
    }
}


def _model_path(model_name: str) -> str:
    path = Path(model_name).expanduser()
    return str(path) if path.exists() else model_name


def _embedding_model_config(model_name: str) -> dict:
    config = EMBEDDING_MODEL_CONFIG.get(model_name)
    if config is not None:
        return config
    if Path(model_name).expanduser().exists():
        return EMBEDDING_MODEL_CONFIG["BAAI/bge-m3"]
    raise ValueError(f"Unsupported embedding model: {model_name}")


def get_default_max_length(model_name: str) -> int:
    config = _embedding_model_config(model_name)
    return config["max_length"]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_embedding_model(model_name: str) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    return tokenizer, model


def _token_count(text: str, tokenizer) -> int:
    return len(tokenizer(text, truncation=False)["input_ids"])


def _encode_with_optional_chunking(
    text: str,
    model: SentenceTransformer,
    max_length: int,
) -> np.ndarray:
    tokenizer = model.tokenizer
    if _token_count(text, tokenizer) <= max_length:
        return model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    splitter = ParagraphSplitter(device=DEVICE)
    chunks = [str(chunk) for chunk in splitter(text) if str(chunk).strip()]
    if not chunks:
        return model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    chunk_embeddings = model.encode(
        chunks,
        batch_size=8,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.mean(chunk_embeddings, axis=0)


def _encode_sentence_transformer_texts(
    texts: list[str],
    model_name: str,
) -> np.ndarray:
    model = _load_sentence_transformer_model(model_name)
    max_length = get_default_max_length(model_name)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length

    if not texts:
        return np.empty((0, 0))

    tokenizer = model.tokenizer
    token_counts = [_token_count(text, tokenizer) for text in texts]
    embeddings: list[np.ndarray | None] = [None] * len(texts)

    short_indices = [
        index for index, token_count in enumerate(token_counts)
        if token_count <= max_length
    ]
    if short_indices:
        short_embeddings = model.encode(
            [texts[index] for index in short_indices],
            batch_size=8,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for index, embedding in zip(short_indices, short_embeddings, strict=False):
            embeddings[index] = embedding

    for index, token_count in enumerate(token_counts):
        if token_count > max_length:
            embeddings[index] = _encode_with_optional_chunking(
                texts[index],
                model=model,
                max_length=max_length,
            )

    return np.stack([embedding for embedding in embeddings if embedding is not None])


@lru_cache(maxsize=2)
def _load_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(_model_path(model_name), device=DEVICE)
    max_length = get_default_max_length(model_name)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length
    return model


def encode_texts(
    texts: list[str],
    model_name: str,
) -> np.ndarray:
    config = _embedding_model_config(model_name)
    if config.get("backend") == "sentence_transformers":
        return _encode_sentence_transformer_texts(texts, model_name=model_name)

    tokenizer, model = load_embedding_model(model_name)
    max_length = get_default_max_length(model_name)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs: BaseModelOutput = model(**inputs)

    embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return embeddings.cpu().numpy()


def build_embedding_text(profile: DatasetProfile) -> str:
    """Build embedding input text from dataset title and generated catalog summary."""
    title = (profile.title or "").strip()
    catalog_summary = (profile.catalog_summary or "").strip()

    if title and catalog_summary:
        return f"{title}. {catalog_summary}"
    return title or catalog_summary


def build_raw_embedding_text(profile: DatasetProfile) -> str:
    """Build embedding input directly from raw dataset metadata fields."""
    parts = []

    if profile.title:
        parts.append(profile.title)
    if profile.headline:
        parts.append(profile.headline)
    if profile.description:
        parts.append(profile.description)
    if profile.keywords:
        parts.append(profile.keywords)
    if profile.field_of_science:
        parts.append(profile.field_of_science)

    return ". ".join(part.strip() for part in parts if part and str(part).strip())


__all__ = [
    "EMBEDDING_MODEL_CONFIG",
    "build_embedding_text",
    "build_raw_embedding_text",
    "encode_texts",
    "load_embedding_model",
]
