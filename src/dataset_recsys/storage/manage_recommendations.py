"""
CLI utility to manage recommendation data stored in Redis.

Supported actions:
- ingest recommendation results for one application
- delete all stored recommendations for one application
- list stored entity IDs for one application
"""
import os
import argparse
from typing import Optional
from dataset_recsys.storage.recommendation_client import RecommendationClient


def get_client() -> Optional[RecommendationClient]:
    """Create a Redis client using the local test configuration."""
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["REDIS_PORT"] = "6380"
    os.environ["REDIS_DB"] = "0"

    client = RecommendationClient()

    if not client.check_connection():
        print("Sub-optimal: Could not connect to Redis. Is the port-forward active?")
        return None

    return client

def ingest_recommendations(file_path: str, application: str):
    client = get_client()
    if client is None:
        return

    print(f"Starting ingestion for application: {application}...")
    
    try:
        result = client.ingest_dataset(file_path, application=application)
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        

def delete_application(application: str):
    client = get_client()
    if client is None:
        return

    print(f"Deleting recommendations for application: {application}...")

    try:
        result = client.delete_application(application)
        print(f"✅ Deleted {result} recommendation keys for application '{application}'.")
    except Exception as e:
        print(f"❌ Delete failed: {e}")


def list_entities(application: str):
    client = get_client()
    if client is None:
        return

    entity_ids = client.list_entities(application)
    print(f"Entities currently stored for application '{application}': {entity_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage application recommendations stored in Redis")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest recommendation JSON file for one application")
    ingest_parser.add_argument("file", help="Path to recommendation JSON file")
    ingest_parser.add_argument("application", help="Application name")

    delete_parser = subparsers.add_parser("delete-application", help="Delete all recommendations for one application")
    delete_parser.add_argument("application", help="Application name")

    list_parser = subparsers.add_parser("list-entities", help="List entity IDs stored for one application")
    list_parser.add_argument("application", help="Application name")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_recommendations(args.file, args.application)
    elif args.command == "delete-application":
        delete_application(args.application)
    elif args.command == "list-entities":
        list_entities(args.application)
    else:
        parser.print_help()

# python src/dataset_recsys/storage/manage_recommendations.py ingest data/mathe/mathe_top20_recommendations.json mathe
# python src/dataset_recsys/storage/manage_recommendations.py delete-application mathe
# python src/dataset_recsys/storage/manage_recommendations.py list-entities mathe