import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from darelabdb.utils_datasets.datafinder import DataFinder
from darelabdb.recs_metrics.item_item import recall_at_n, tndcg_at_n
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel

from datasets import Dataset

SEED = 42

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()


    def clean(self, text):
        text = text.lower()

        # Remove markdown artifacts
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # markdown links [links](url)
        text = re.sub(r"\[(.*?)\]", r"\1", text)  # bare brackets

        # Remove punctuation but preserve hyphenated words and IDs
        # This preserves tokens like "sysu-mm01-c" or "image-net1k"
        text = re.sub(r"[^\w\- ]", " ", text)

        # Remove standalone numbers (tokens that are only digits)
        text = re.sub(r"\b\d+\b", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize and lemmatize
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stopwords and len(t) > 2
        ]

        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean(doc) for doc in X]
    
# Data preparation
df = DataFinder()
data = df.get()
corpus = data["corpus"]

corpus["text"] = corpus.apply(
    lambda row: " ".join(filter(None, [
        f"Title: {row['title']}." if pd.notna(row['title']) else None,
        f"Description: {row['description']}." if pd.notna(row['description']) else None,
        f"Tasks: {'; '.join(row['tasks'])}." if isinstance(row['tasks'], list) else None,
        f"Modalities: {row['modalities']}." if pd.notna(row['modalities']) else None
    ])),
    axis=1
)

cleaner = TextPreprocessor()
corpus["text"] = cleaner.transform(corpus["text"])

# Dataset ID mappings
id_to_idx = {id_: i for i, id_ in enumerate(corpus["id"])}
idx_to_id = {i: id_ for id_, i in id_to_idx.items()}

ground_truth_links = df.get_links_from_queries()

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models
from torch.utils.data import DataLoader
import random

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

def prepare_training_data(corpus, ground_truth_links, num_negatives=1):
    """
    Prepare training data for fine-tuning.
    Returns a list of InputExample objects with positive and negative pairs.
    """
    id_to_text = dict(zip(corpus["id"], corpus["text"]))
    all_ids = list(id_to_text.keys())
    examples = []

    for anchor_id, positive_ids in ground_truth_links.items():
        anchor_text = id_to_text.get(anchor_id, None)
        if anchor_text is None:
            continue
        for pos_id in positive_ids:
            pos_text = id_to_text.get(pos_id, None)
            if pos_text is None:
                continue
            # Positive pair
            examples.append(InputExample(texts=[anchor_text, pos_text], label=1.0))
            # Negative samples
            negatives = []
            while len(negatives) < num_negatives:
                neg_id = random.choice(all_ids)
                if neg_id != anchor_id and neg_id not in positive_ids:
                    neg_text = id_to_text.get(neg_id, None)
                    if neg_text is not None:
                        negatives.append(neg_text)
            for neg_text in negatives:
                examples.append(InputExample(texts=[anchor_text, neg_text], label=0.0))
    return examples

def convert_to_hf_dataset(input_examples):
    data = {
        "text1": [ex.texts[0] for ex in input_examples],
        "text2": [ex.texts[1] for ex in input_examples],
        "label": [ex.label for ex in input_examples]
    }
    return Dataset.from_dict(data)

num_epochs = 3  # Maximum number of epochs

# Prepare training and validation sets
all_train_examples = prepare_training_data(corpus, ground_truth_links, num_negatives=1)
split_idx = int(len(all_train_examples) * 0.9)
train_examples = all_train_examples[:split_idx]
val_examples = all_train_examples[split_idx:]

train_dataset = convert_to_hf_dataset(train_examples)
val_dataset = convert_to_hf_dataset(val_examples)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name='val-eval')

#
# Fine-tuning SciBERT
sbert_model_name = "allenai/scibert_scivocab_uncased"
word_embedding_model = models.Transformer(sbert_model_name, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_loss = losses.CosineSimilarityLoss(sbert_model)

training_args_scibert = SentenceTransformerTrainingArguments(
    output_dir="fine_tuned_scibert_model",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    logging_dir="./logs_scibert"
)

trainer_scibert = SentenceTransformerTrainer(
    model=sbert_model,
    args=training_args_scibert,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=train_loss,
    evaluator=val_evaluator,
)
trainer_scibert.train()
trainer_scibert.save_model("fine_tuned_scibert_model")

#
# Fine-tuning SPECTER
specter_model_name = "sentence-transformers/allenai-specter"
specter_model = SentenceTransformer(specter_model_name)

train_examples_specter = prepare_training_data(corpus, ground_truth_links, num_negatives=1)
train_dataset_specter = convert_to_hf_dataset(train_examples_specter[:split_idx])
val_dataset_specter = convert_to_hf_dataset(train_examples_specter[split_idx:])
val_evaluator_specter = EmbeddingSimilarityEvaluator.from_input_examples(train_examples_specter[split_idx:], name='val-eval-specter')

train_loss_specter = losses.CosineSimilarityLoss(specter_model)

training_args_specter = SentenceTransformerTrainingArguments(
    output_dir="fine_tuned_specter_model",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    logging_dir="./logs_specter"
)

trainer_specter = SentenceTransformerTrainer(
    model=specter_model,
    args=training_args_specter,
    train_dataset=train_dataset_specter,
    eval_dataset=val_dataset_specter,
    loss=train_loss_specter,
    evaluator=val_evaluator_specter,
)
trainer_specter.train()
trainer_specter.save_model("fine_tuned_specter_model")

# Load fine-tuned models
sbert_model_ft = SentenceTransformer("fine_tuned_scibert_model")
specter_model_ft = SentenceTransformer("fine_tuned_specter_model")

# Compute embeddings and similarity matrices for fine-tuned models
embeddings_scibert_ft = sbert_model_ft.encode(corpus["text"].tolist(), batch_size=16, show_progress_bar=True)
similarity_matrix_scibert_ft = cosine_similarity(embeddings_scibert_ft)

embeddings_specter_ft = specter_model_ft.encode(corpus["text"].tolist(), batch_size=16, show_progress_bar=True)
similarity_matrix_specter_ft = cosine_similarity(embeddings_specter_ft)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu if available



tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model.eval()  # inference mode
model.to(device)

embeddings_scibert = []
with torch.no_grad():
    for text in tqdm(corpus["text"].tolist(), desc="Encoding with SciBERT"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        input_mask = inputs['attention_mask']
        masked_embeddings = token_embeddings * input_mask.unsqueeze(-1)
        mean_embedding = masked_embeddings.sum(1) / input_mask.sum(1).unsqueeze(-1)
        embeddings_scibert.append(mean_embedding.squeeze().cpu().numpy())

embeddings_scibert = np.array(embeddings_scibert)
similarity_matrix_scibert = cosine_similarity(embeddings_scibert)



tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoModel.from_pretrained("allenai/specter2_base")
model.eval()
model.to(device)

embeddings_specter = []
with torch.no_grad():
    for text in tqdm(corpus["text"].tolist(), desc="Encoding with SPECTER"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        input_mask = inputs['attention_mask']
        masked_embeddings = token_embeddings * input_mask.unsqueeze(-1)
        mean_embedding = masked_embeddings.sum(1) / input_mask.sum(1).unsqueeze(-1)
        embeddings_specter.append(mean_embedding.squeeze().cpu().numpy())

embeddings_specter = np.array(embeddings_specter)
similarity_matrix_specter = cosine_similarity(embeddings_specter)



results_all = {model: {} for model in [
    "SciBERT (base)", "SPECTER (base)",
    "SciBERT (fine-tuned)", "SPECTER (fine-tuned)"
]}

for n in [10, 20, 50]:

    # SciBERT
    predictions_scibert = {
        idx_to_id[i]: [idx_to_id[j] for j in similarity_matrix_scibert[i].argsort()[::-1] if j != i][:n]
        for i in range(len(corpus))
    }
    recall_scibert = recall_at_n(predictions_scibert, ground_truth_links, n=n)
    ndcg_scibert = tndcg_at_n(predictions_scibert, ground_truth_links, n=n)

    # SPECTER
    predictions_specter = {
        idx_to_id[i]: [idx_to_id[j] for j in similarity_matrix_specter[i].argsort()[::-1] if j != i][:n]
        for i in range(len(corpus))
    }
    recall_specter = recall_at_n(predictions_specter, ground_truth_links, n=n)
    ndcg_specter = tndcg_at_n(predictions_specter, ground_truth_links, n=n)

    # SciBERT (fine-tuned)
    predictions_scibert_ft = {
        idx_to_id[i]: [idx_to_id[j] for j in similarity_matrix_scibert_ft[i].argsort()[::-1] if j != i][:n]
        for i in range(len(corpus))
    }
    recall_scibert_ft = recall_at_n(predictions_scibert_ft, ground_truth_links, n=n)
    ndcg_scibert_ft = tndcg_at_n(predictions_scibert_ft, ground_truth_links, n=n)

    # SPECTER (fine-tuned)
    predictions_specter_ft = {
        idx_to_id[i]: [idx_to_id[j] for j in similarity_matrix_specter_ft[i].argsort()[::-1] if j != i][:n]
        for i in range(len(corpus))
    }
    recall_specter_ft = recall_at_n(predictions_specter_ft, ground_truth_links, n=n)
    ndcg_specter_ft = tndcg_at_n(predictions_specter_ft, ground_truth_links, n=n)

    for model, recall, ndcg in [
        ("SciBERT (base)", recall_scibert, ndcg_scibert),
        ("SPECTER (base)", recall_specter, ndcg_specter),
        ("SciBERT (fine-tuned)", recall_scibert_ft, ndcg_scibert_ft),
        ("SPECTER (fine-tuned)", recall_specter_ft, ndcg_specter_ft)
    ]:
        results_all[model][f"Recall@{n}"] = recall
        results_all[model][f"NDCG@{n}"] = ndcg

results_df = pd.DataFrame.from_dict(results_all, orient="index")
os.makedirs("recommender_results", exist_ok=True)
results_df.to_csv("recommender_results/results.csv", index_label="Model")