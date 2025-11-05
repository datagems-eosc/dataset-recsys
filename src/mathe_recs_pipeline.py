from huggingface_hub import snapshot_download
from pathlib import Path
import json, random
import os

from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from chonky import ParagraphSplitter

import torch

#TODO: What about material in the form of videos?

def download_mathe_assets():
    """
    Downloads MathE assets from Hugging Face (including PDFs, OCR, and indexes)
    and returns the local path to the downloaded 'mathe' directory.
    """
    local_dir = snapshot_download(
        repo_id="DARELab/cross-dataset-assets",
        repo_type="dataset",
        local_dir_use_symlinks=False,
        allow_patterns=["mathe/**"],   # downloads PDFs + OCR + indexes
    )
    print("Assets stored under:", local_dir)
    return Path(local_dir) / "mathe"

def load_mathe_ocr_data(base):
    """
    Loads and returns the MathE OCR JSON data from the specified base directory.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary represents one MathE material with:
            - "id": The relative path to the source PDF file (e.g., "materials/56.pdf").
            - "contents": The OCR-extracted text from that PDF.
    """
    ocr_path = base / "data.json"
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data), "materials found")
    return data

def compute_token_stats(data, tokenizer):
    """
    Computes token statistics for the given data using the provided tokenizer.
    Prints summary statistics and returns the IDs of materials whose tokenized
    length exceeds the model's maximum token length.
    """
    lengths = []
    for d in data:
        n_tokens = len(tokenizer(d["contents"], truncation=False)["input_ids"])
        lengths.append(n_tokens)

    print(f"\nToken length stats:")
    print(f"  Avg: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Max: {np.max(lengths)}")

    # Check which ones exceed the token limit (e.g., 8192 for BGE-M3)
    max_length = getattr(tokenizer.model_max_length, "real", tokenizer.model_max_length)

    over_limit_ids = [d["id"] for d, n in zip(data, lengths) if n > max_length]
    print(f"\n{len(over_limit_ids)} materials exceed {max_length} tokens.")
    return over_limit_ids

if __name__ == "__main__":

    base = download_mathe_assets()
    data = load_mathe_ocr_data(base)

    # Step 1: Compute embeddings for all materials

    # Does any material exceed token limit?
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    over_limit_ids = compute_token_stats(data, tokenizer)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0 / 1, adjust as needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model = SentenceTransformer("BAAI/bge-m3").to(device)
    print("\nStarting embedding computation for all materials...")
    embeddings = {}
    max_length = getattr(tokenizer.model_max_length, "real", tokenizer.model_max_length)
    splitter = ParagraphSplitter(device=device)
    for idx, d in enumerate(data, 1):
        material_id = Path(d["id"]).name
        text = d["contents"]
        if d["id"] not in over_limit_ids:
            emb = model.encode(text)
            embeddings[material_id] = emb
            print(f"Embedded material {idx}/{len(data)} (ID: {material_id}) directly.")
        else:
            print(f"Over token limit → chunking {material_id} semantically...")
            chunks = list(splitter(text))
            chunk_embs = [model.encode(chunk) for chunk in chunks]
            emb = np.mean(chunk_embs, axis=0)
            embeddings[material_id] = emb
            print(f"  Averaged embeddings from {len(chunks)} chunks.")

    print("\nEmbedding computation completed.")
    print(f"Total materials embedded: {len(embeddings)}")

    mathe_path = Path("data/mathe")
    mathe_path.mkdir(parents=True, exist_ok=True)

    # Convert embeddings to numpy array and save
    material_ids = list(embeddings.keys())
    embeddings_array = np.stack([embeddings[mid] for mid in material_ids])
    np.save(mathe_path / "mathe_embeddings.npy", embeddings_array)

    # Create index mapping and save
    index_mapping = {i: material_id for i, material_id in enumerate(material_ids)}
    with open(mathe_path / "mathe_embedding_index.json", "w", encoding="utf-8") as f:
        json.dump(index_mapping, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to mathe_embeddings.npy with shape {embeddings_array.shape}")
    print(f"Saved index mapping to mathe_embedding_index.json")

    # Step 2: Generate top-n recommendations

    n = 20  # n for top-n recommendations
    mathe_path = Path("data/mathe")
    loaded_embeddings = np.load(mathe_path / "mathe_embeddings.npy")
    with open(mathe_path / "mathe_embedding_index.json", "r", encoding="utf-8") as f:
        loaded_index_mapping = json.load(f)
        
    # Invert the mapping to get index -> material_id
    idx_to_material_id = {int(k): v for k, v in loaded_index_mapping.items()}
    num_materials = loaded_embeddings.shape[0]

    # Normalize embeddings for cosine similarity
    normed_embs = loaded_embeddings / np.linalg.norm(loaded_embeddings, axis=1, keepdims=True)
    sim_matrix = np.dot(normed_embs, normed_embs.T)

    topn_recommendations = {
        idx_to_material_id[i]: [
            idx_to_material_id[j] for j in np.argsort(sim_matrix[i])[::-1] if j != i
        ][:n]
        for i in range(num_materials)
    }

    with open(mathe_path / f"mathe_top{n}_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(topn_recommendations, f, ensure_ascii=False, indent=2)

    print(f"Top-{n} similarity recommendations generated for {num_materials} materials.")
    print(f"Saved recommendations to {mathe_path / f'mathe_top{n}_recommendations.json'}")

    # # Pick a random material to inspect
    # sample = random.choice(data)
    # pdf_rel = sample["id"].lstrip("./")     # e.g., "materials/6.pdf"
    # pdf_path = base / pdf_rel

    # print("\nSelected material:", pdf_rel)
    # print(f"PDF path: {pdf_path}")

    # print("\n--- OCR Extracted Text (first 1000 chars) ---")
    # print(sample["contents"][:1000])

    # # Display sample recommendations
    # recs = json.load(open(mathe_path / f"mathe_top{n}_recommendations.json", "r", encoding="utf-8"))
    # query_doc = "56.pdf"

    # print("\nQuery document path:")
    # print(base / query_doc)

    # print("\nTop 3 recommended document paths:")
    # for rec in recs[query_doc][:3]:
    #     print(base / "materials" / rec)