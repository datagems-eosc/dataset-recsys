from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from dataset_recsys.ingestion.fetch_gems_datasets import DatasetProfile


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_CONFIG = {
    "allenai/specter2_base": {
        "max_length": 512,
    }
}


def get_default_max_length(model_name: str) -> int:
    config = EMBEDDING_MODEL_CONFIG.get(model_name)
    if config is None:
        raise ValueError(f"Unsupported embedding model: {model_name}")
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


def encode_texts(
    texts: list[str],
    model_name: str,
) -> np.ndarray:
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