
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

---

## Deploy Changes

### 1. Prepare Your Code

Clone both repositories:
- [This repo - Recsys API](https://github.com/datagems-eosc/dataset-recsys/tree/main)
- [Deployment repo - Recsys Backend](https://github.com/datagems-eosc/dataset-recsys-deployment-dev)

After making your changes, commit and tag:
```bash
git add .
git commit -m "your commit message"
git push
git tag vx.y.z
git push origin vx.y.z
```

### 2. Build Image

Monitor the image build progress in the [GitHub Actions tab](https://github.com/datagems-eosc/dataset-recsys/actions). Once it is deployed, move on to step 3.

### 3. Deployment Repo

Navigate to the deployment repo:
```bash
cd dataset-recsys-discovery-deployment-dev/
```

Ensure you are in the correct branch:
```bash
git checkout main
```

Pull the latest commit:
```bash
git pull
```

### 4. Update with latest API image


For simple changes that don't require configuration updates, edit the following file:
```
deployment_api/dataset-recsys-api.deployment.yaml
```

Change this line (around line 31):
```yaml
image: ghcr.io/datagems-eosc/dataset-recsys-api:{tag}
```

Update `{tag}` to the version you used in **Step 1**

Apply the changes:
```bash
kubectl apply -f dataset-recsys-api.deployment.yaml
```

### 5. Monitor Deployment

Watch the pod status:
```bash
kubectl get pods -n athenarc -w
```

When the new pod is running successfully, the old one will terminate automatically.
