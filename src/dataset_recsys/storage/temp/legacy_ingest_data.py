import os
from dataset_recsys.storage.recommendation_client import RecommendationClient # Assuming your class is in this file

def run_ingestion(file_path: str, dataset_id: str):
    # 1. Setup environment for local port-forwarding
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6380"
    os.environ["REDIS_DB"] = "0"

    client = RecommendationClient()

    # 2. Verify connection
    if not client.check_connection():
        print("Sub-optimal: Could not connect to Redis. Is the port-forward active?")
        return

    print(f"🚀 Starting ingestion for dataset: {dataset_id}...")
    
    try:
        # 3. Perform ingestion
        # This will create keys like recommendations:dataset_id:item_id
        result = client.ingest_dataset(file_path, dataset_id=dataset_id)
        print(f"✅ Success: {result}")
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")

if __name__ == "__main__":
    # Update these paths/IDs as needed
    src_base = os.path.dirname(os.path.abspath(__file__))
    repo_base = os.path.dirname(src_base)
    DATA_FILE = os.path.join(repo_base, "data", "mathe", "mathe_top20_recommendations.json")
    NEW_DATASET_ID = "9b25bc46-8bd3-4f7f-94b4-52dbc38c130f"
    
    run_ingestion(DATA_FILE, NEW_DATASET_ID)