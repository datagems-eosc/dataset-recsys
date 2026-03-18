"""
Portal Dataset Recommendation MVP Pipeline

Goal
----
This script builds the offline preprocessing pipeline for dataset recommendations
based on metadata extracted from the DataGEMS portal.

Scope (current stage)
---------------------
Input:
- Dataset metadata fetched from the portal

Processing:
1. Run ingestion to extract dataset profiles from the portal response
2. Enrich the profiles with LLM-generated descriptions and store them
3. Clean text inputs for embedding generation
4. Generate embeddings
5. Candidate retrieval based on embedding similarity
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.dataset_recsys.ingestion.fetch_gems_datasets import run_ingestion
from src.dataset_recsys.utils.bedrock import enrich_batch
from src.dataset_recsys.utils.text_preprocessing import LightTextPreprocessor, build_embedding_text

LLM_NAME = "claude-sonnet-4-6"
DATA_PATH = Path("data/gems_datasets_metadata/moma/datagems_dataset_profiles.json")
ENRICHED_OUTPUT_PATH = Path(
    f"data/gems_datasets_metadata/moma/datagems_dataset_profiles_enriched_{LLM_NAME}.json"
)
EMBEDDINGS_OUTPUT_PATH = Path(
    f"data/gems_datasets_metadata/moma/datagems_dataset_embeddings_{LLM_NAME}.npy"
)
EMBEDDINGS_METADATA_PATH = Path(
    f"data/gems_datasets_metadata/moma/datagems_dataset_embeddings_{LLM_NAME}.json"
)
RECOMMENDATIONS_OUTPUT_PATH = Path(
    f"data/gems_datasets_metadata/moma/datagems_dataset_recommendations_{LLM_NAME}.json"
)
TOP_K = 10
EMBEDDING_MODEL_NAME = "allenai/specter2_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

def save_json(data, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    return tokenizer, model

def encode_texts(texts, model_name: str = EMBEDDING_MODEL_NAME, max_length: int = MAX_LENGTH):
    tokenizer, model = load_embedding_model(model_name)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return embeddings.cpu().numpy()


# --- Candidate retrieval functions ---

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_recommendations(profiles, embeddings: np.ndarray, top_k: int = TOP_K):
    similarity = cosine_similarity_matrix(embeddings)
    recommendations = []

    for i, profile in enumerate(profiles):
        ranked_indices = np.argsort(similarity[i])[::-1]
        ranked_indices = [j for j in ranked_indices if j != i][:top_k]

        recommendations.append(
            {
                "id": profile.get("id"),
                "title": profile.get("title"),
                "recommendations": [
                    {
                        "id": profiles[j].get("id"),
                        "title": profiles[j].get("title"),
                        "score": float(similarity[i, j]),
                    }
                    for j in ranked_indices
                ],
            }
        )

    return recommendations

if __name__ == "__main__":
    # Step 1: Run ingestion to extract dataset profiles from the portal response
    #  _, profiles = run_ingestion()
    #  profiles = load_profiles(DATA_PATH)

    # Step 2: Enrich profiles with LLM-generated description
    #  enriched_profiles = enrich_batch(profiles, llm=LLM_NAME)
    #  save_profiles(enriched_profiles, output_path=ENRICHED_OUTPUT_PATH)
    enriched_profiles = load_json(ENRICHED_OUTPUT_PATH)

    # Step 3: Clean text inputs for embedding generation
    cleaner = LightTextPreprocessor()
    model_inputs = [build_embedding_text(profile, cleaner) for profile in enriched_profiles]

    print("\nSample model inputs:\n")
    for i, text in enumerate(model_inputs[:2], start=1):
        print(f"[{i}]\n{text}\n")

    # Step 4: Generate embeddings
    embeddings = encode_texts(model_inputs)
    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)

    embedding_metadata = [
        {
            "id": profile.get("id"),
            "title": profile.get("title"),
            "embedding_text": text,
        }
        for profile, text in zip(enriched_profiles, model_inputs)
    ]
    save_json(embedding_metadata, EMBEDDINGS_METADATA_PATH)

    # Step 5: Candidate retrieval based on embedding similarity
    recommendations = build_recommendations(enriched_profiles, embeddings)
    save_json(recommendations, RECOMMENDATIONS_OUTPUT_PATH)

    print(f"Saved embeddings with shape {embeddings.shape} to {EMBEDDINGS_OUTPUT_PATH}")
    print(f"Saved embedding metadata to {EMBEDDINGS_METADATA_PATH}")
    print(f"Saved recommendations to {RECOMMENDATIONS_OUTPUT_PATH}")

    