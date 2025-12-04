from typing import Optional, Dict
import pandas as pd
from collections import defaultdict
from itertools import combinations
import os
import json
from urllib.request import urlretrieve
from pathlib import Path

GITHUB_BASE_URL = "https://raw.githubusercontent.com/viswavi/datafinder/main/data/"
REQUIRED_FILES = [
    "train_data.jsonl",
    "test_data.jsonl",
    "dataset_search_collection.jsonl",
]

class DataFinder:
    """
    Usage:
        from datafinder import DataFinder
        datafinder = DataFinder()
        data = datafinder.get()
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.data = None
        self.cache_dir = cache_dir if cache_dir is not None else str(Path.home()) + "/.cache/darelabdb/datasets/"
        self.dataset_folder = os.path.join(self.cache_dir, "datafinder/")

    def _init_data(self):
        """
        Downloads the DataFinder files from GitHub and loads them into memory.
        """
        data_dir = os.path.join(self.dataset_folder, "data/")
        os.makedirs(data_dir, exist_ok=True)

        # Download each required file if not already present
        for filename in REQUIRED_FILES:
            dest_path = os.path.join(data_dir, filename)
            if not os.path.exists(dest_path):
                url = GITHUB_BASE_URL + filename
                urlretrieve(url, dest_path)

        # Load the files
        with open(os.path.join(data_dir, "train_data.jsonl")) as f:
            train = [json.loads(line) for line in f]
        with open(os.path.join(data_dir, "test_data.jsonl")) as f:
            test = [json.loads(line) for line in f]
        with open(os.path.join(data_dir, "dataset_search_collection.jsonl")) as f:
            corpus = [json.loads(line) for line in f]

        self.data = {
            "train": pd.DataFrame(train).replace("", pd.NA),
            "test": pd.DataFrame(test).replace("", pd.NA),
            "corpus": pd.DataFrame(corpus).replace("", pd.NA),
        }

    def get_info(self) -> Dict[str, str]:
        """
        Returns a dictionary containing information about the dataset.
        """
        return {
            "name": "DataFinder",
            "description": (
                "DataFinder is a dataset designed for scientific dataset recommendation. "
                "Each instance is a query derived from a scientific paper abstract, paired with one or more relevant datasets. "
                "The dataset includes train and test splits with annotated positives and a corpus of over 7,000 dataset entries. "
                "Corpus entries include title, textual description, modality, field of study, and usage frequency."
            ),
            "source": "https://github.com/viswavi/datafinder",
            "formats": ["jsonl"],
            "dataset_folder": (
                self.dataset_folder if self.data is not None else "NOT_YET_LOADED"
            ),
        }
    
    def get(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Returns the main splits (train queries, test queries) and dataset corpus.

        - "train": A DataFrame containing scientific paper queries used for training.
            Includes the fields:
                - title, authors, abstract, year: Metadata describing the paper
                - query: A natural language query derived from the paper
                - keyphrase_query: A keyword-based version of the query
                - positives: List of relevant dataset IDs
                - negatives: List of irrelevant dataset IDs

        - "test": A DataFrame containing evaluation queries.
            Includes the fields:
                - query, keyphrase_query: Descriptions of the query
                - positives: List of relevant dataset IDs (renamed from "documents")

        - "corpus": A DataFrame of all candidate datasets.
            Includes the fields:
                - id: Canonical dataset identifier
                - title: Title of the paper the dataset was proposed in
                - description: Free-text description of the dataset (renamed from "contents")
                - year: Year of the paper the dataset was proposed in
                - tasks: Extracted tasks (if available) from structured_info
                - modalities: Extracted modalities (if available) from structured_info
                - popularity: Estimated usage frequency (times used)

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with "train", "test", and "corpus" keys.
        """
        if self.data is None:
            self._init_data()
        train = self.data["train"]
        train_filtered = train[[
            "title", "authors", "abstract", "year",
            "query", "keyphrase_query", "positives", "negatives"
        ]]

        test = self.data["test"]
        test_filtered = test[[
            "query", "keyphrase_query", "documents"
            ]].rename(columns={"documents": "positives"})

        # Keep only queries with non-empty positive lists
        test_filtered = test_filtered[test_filtered["positives"].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )]

        # Filter and enrich corpus
        corpus = self.data["corpus"]
        corpus_filtered = corpus[["id", "title", "contents", "year", "structured_info"]].copy()
        corpus_filtered = corpus_filtered.rename(columns={"contents": "description"})
        structured_parts = corpus_filtered["structured_info"].apply(self._extract_structured_parts)
        corpus_filtered = pd.concat([corpus_filtered.drop(columns=["structured_info"]), structured_parts], axis=1)

        corpus_filtered = corpus_filtered.replace("", pd.NA)
        corpus_filtered = corpus_filtered.dropna(subset=["description", "tasks"], how="any") # exclude datasets without description or tasks

        # Deduplicate: keep the oldest entry per ID
        corpus_filtered = corpus_filtered.sort_values(by="year", ascending=True, na_position="last")
        corpus_filtered = corpus_filtered.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

        # Build set of valid dataset IDs from corpus
        valid_ids = set(corpus_filtered["id"])

        # Filter train positives and negatives to only include valid dataset IDs
        train_filtered.loc[:, "positives"] = train_filtered["positives"].apply(lambda lst: [d for d in lst if d in valid_ids])
        train_filtered.loc[:, "negatives"] = train_filtered["negatives"].apply(lambda lst: [d for d in lst if d in valid_ids])

        # Filter test positives to only include valid dataset IDs
        test_filtered["positives"] = test_filtered["positives"].apply(lambda lst: [d for d in lst if d in valid_ids])

        # Exclude train and test rows with no remaining positives
        train_filtered = train_filtered[train_filtered["positives"].apply(lambda x: len(x) > 0)]
        test_filtered = test_filtered[test_filtered["positives"].apply(lambda x: len(x) > 0)]

        return {
            "train": pd.DataFrame(train_filtered),
            "test": pd.DataFrame(test_filtered),
            "corpus": pd.DataFrame(corpus_filtered),
        }

    def _extract_structured_parts(self, info: str) -> pd.Series:
        """
        Extracts tasks, modalities, and popularity from the 'structured_info' text blob.

        Returns:
            pd.Series: A Series with columns ['tasks', 'modalities', 'popularity'].
        """
        tasks = modalities = popularity = pd.NA

        task_prefix = "this dataset can be used to study the task of"
        for line in info.split("\n"):
            line = line.strip().rstrip(".")

            if line.lower().startswith(task_prefix):
                raw_task_string = line[len(task_prefix):].strip()
                tasks = [t.strip() for t in raw_task_string.replace(" and ", ",").split(",") if t.strip()]
            elif "having been used" in line and "times" in line:
                try:
                    popularity = int(line.split("having been used")[1].split("times")[0].strip())
                except Exception:
                    pass
            elif modalities is pd.NA:
                modalities = line

        return pd.Series([tasks, modalities, popularity], index=["tasks", "modalities", "popularity"])

    def get_raw(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the full raw contents of train, test, and corpus.

        - The "corpus" contains metadata entries for thousands of research datasets. 
        Each entry includes:
            - `id`: Canonical name of the dataset (umbrella concept).
            - `variants`: A list of specific names or versions associated with that dataset. 
                    These may include:  - alternative spellings/naming  - benchmark-specific tracks (e.g., 2017 challenge)  
                    - versioned releases (v1, v2, etc.)  - modality-specific variants
            - `title`: Title of the paper that introduced or described the dataset.
            - `contents`: Free-text summary describing the dataset and its intended purpose.
            - `structured_info`: Semi-structured string including tasks, modalities, and usage frequency.
            - `year`: Publication or release year of the dataset.
            - `date`: Exact date (when available) of dataset release or publication.

        - The "train" and "test" splits contain scientific paper metadata along with annotations:
            - `query`: A short user-style query derived from the paper (e.g., a description of a task).
            - `keyphrase_query`: A keyword-based version of the query.
            - `title`, `abstract`, `authors`, etc.: Metadata describing the scientific paper.
            - `positives`: List of relevant datasets (by `id`) from the corpus.
            - `negatives`: Optional list of non-relevant datasets (by `id`).

        Returns:
            Dict[str, pd.DataFrame]: Full unfiltered contents for "train", "test", and "corpus".
        """
        if self.data is None:
            self._init_data()
        return self.data

    def get_links_from_queries(self) -> Dict[str, set]:
        """
        Returns a dictionary of dataset-to-dataset links based on co-occurrence
        in the 'positives' field of train and test splits (clean data).
        Each dataset ID maps to a set of related dataset IDs.
        Relationships are symmetric: if d1 is related to d2, then d2 is also related to d1.
        """
        if self.data is None:
            self._init_data()

        data = self.get()
        splits = [data["train"], data["test"]]

        related = defaultdict(set)

        for split in splits:
            for _, row in split.iterrows():
                positives = set(row.get("positives", []))
                if len(positives) > 1:
                    for d1, d2 in combinations(positives, 2):
                        related[d1].add(d2)
                        related[d2].add(d1)

        return dict(related)

    def get_links_from_tasks(self) -> Dict[str, set]:
        """
        Returns a dictionary of dataset-to-dataset links based on shared tasks
        in the corpus (clean data).
        Each dataset ID maps to a set of related dataset IDs.
        Relationships are symmetric: if d1 and d2 share a task, both will reference each other.
        """
        if self.data is None:
            self._init_data()

        data = self.get()
        corpus = data["corpus"]

        related = defaultdict(set)

        for _, row in corpus.iterrows():
            dataset_id = row["id"]
            tasks = row["tasks"]
            for task in set(tasks):  # eliminate duplicate tasks
                co_datasets = corpus[corpus["tasks"].apply(lambda t: task in t if isinstance(t, list) else False)]["id"]
                for other_id in co_datasets:
                    if other_id != dataset_id:
                        related[dataset_id].add(other_id)
                        related[other_id].add(dataset_id)

        return dict(related)
    
    def check_dataset_id(self, dataset_id: str) -> bool:
        """
        Checks whether a dataset ID exists in the corpus.

        Args:
            dataset_id (str): The dataset ID to check.

        Returns:
            bool: True if the ID exists, False otherwise.
        """
        if self.data is None:
            self._init_data()
        return dataset_id in set(self.data["corpus"]["id"])