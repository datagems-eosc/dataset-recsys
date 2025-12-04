
# DataGEMS RecSys 

Implementation of the DataGEMS recommendation services, designed to support dataset discovery across the DataGEMS platform. The goal of these services is to reduce the manual effort required to locate relevant datasets by identifying related resources. 

In addition to powering dataset discovery on the platform, the recommendation services will also be demonstrated within the four DataGEMS use cases (i.e., higher education, lifelong learning, language, and weather), where domain-specific recommenders are instantiated and showcased. In these use cases, the type of recommendations varies depending on the application. For example, Higher Education may recommend educational materials, Lifelong Learning may recommend skills, and the Language use case may recommend linguistic resources. The recommender service adjusts its inputs and retrieval strategy according to the needs of each domain.

The dataset-to-dataset recommendation pipeline consists of four components:

1. **Metadata ingestion layer**: the textual information associated with each dataset.
2. **Representation module**: transforms this information into numerical embeddings.
3. **Candidate generator**: retrieves a pool of potentially related datasets.
4. **Re-ranking module**: orders the retrieved candidates according to their relevance.

## 📂 Data

### `data/datafinder/`

This folder includes the utilities for downloading and preparing the **DataFinder** benchmark dataset for scientific dataset recommendation ([Viswanathan et al., 2023](https://arxiv.org/pdf/2305.16636)). DataFinder consists of natural language queries (derived from paper abstracts) paired with relevant research datasets. To load the dataset:
```python
from data.datafinder import DataFinder
df = DataFinder().get()
```

### `data/gems_datasets_metadata/`

This folder contains the **dataset profiles** for all datasets ingested into the DataGEMS platform.  
Each JSON file corresponds to one dataset and includes key descriptive metadata such as:

- dataset title  
- description  
- thematic domain  
- publisher / source  
- license

### `data/mathe/`

This folder includes the utilities for downloading and preparing the MathE educational materials used in the Higher Education use case. These materials originate from the MathE platform. The dataset includes PDF files and their OCR-extracted textual content. To load the dataset:
```python
from data.mathe import MathE
df = MathE().get()
```

