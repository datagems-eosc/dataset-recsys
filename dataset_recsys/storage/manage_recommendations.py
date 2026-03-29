"""
CLI utility to manage recommendation data stored in Redis.

Supported actions:
- ingest recommendation results for one application
- delete all stored recommendations for one application
- list stored entity IDs for one application
"""
import json
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

def remove_dataset(application: str, entity_id: str):
    client = get_client()
    if client is None:
        return

    print(f"Removing dataset '{entity_id}' from application '{application}'...")

    try:
        deleted = client.remove_dataset(application, entity_id)
        print(f"✅ Removed dataset '{entity_id}' (deleted keys: {deleted}).")
    except Exception as e:
        print(f"❌ Remove dataset failed: {e}")

def get_recommendations(application: str, entity_id: str):
    client = get_client()
    if client is None:
        return

    print(f"🔍 Fetching recommendations for '{entity_id}' in application '{application}'...")
    recs = client.get_recommendations(application, entity_id)
    
    if not recs:
        print(f"⚠️ No recommendations found for entity '{entity_id}'.")
    else:
        print(f"✅ Found {len(recs)} recommendations:")
        # Print as a clean JSON list for easy reading
        print(json.dumps(recs, indent=2))

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

    get_parser = subparsers.add_parser("get", help="Get recommendations for a specific entity")
    get_parser.add_argument("application", help="Application name")
    get_parser.add_argument("entity_id", help="The ID to lookup (e.g. 6.pdf or a UUID)")

    delete_parser = subparsers.add_parser("delete-application", help="Delete all recommendations for one application")
    delete_parser.add_argument("application", help="Application name")

    list_parser = subparsers.add_parser("list-entities", help="List entity IDs stored for one application")
    list_parser.add_argument("application", help="Application name")

    remove_parser = subparsers.add_parser("remove-dataset", help="Remove a dataset and its references from one application")
    remove_parser.add_argument("application", help="Application name")
    remove_parser.add_argument("entity_id", help="Dataset ID to remove")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_recommendations(args.file, args.application)
    elif args.command == "get":
        get_recommendations(args.application, args.entity_id)
    elif args.command == "delete-application":
        delete_application(args.application)
    elif args.command == "list-entities":
        list_entities(args.application)
    elif args.command == "remove-dataset":
        remove_dataset(args.application, args.entity_id)
    else:
        parser.print_help()

# python dataset_recsys/storage/manage_recommendations.py ingest data/mathe/mathe_top20_recommendations.json ds2ds_mathe
# python dataset_recsys/storage/manage_recommendations.py ingest data/gems_datasets_metadata/moma/datagems_dataset_recommendations_claude-sonnet-4-6.json ds2ds
# python dataset_recsys/storage/manage_recommendations.py delete-application ds2ds_mathe
# python dataset_recsys/storage/manage_recommendations.py delete-application ds2ds
# python dataset_recsys/storage/manage_recommendations.py list-entities ds2ds_mathe
# python dataset_recsys/storage/manage_recommendations.py list-entities ds2ds
# python dataset_recsys/storage/manage_recommendations.py get ds2ds_mathe 6.pdf
# python dataset_recsys/storage/manage_recommendations.py get ds2ds 07382b91-5bc5-42f9-8391-33adc2460c19
# python dataset_recsys/storage/manage_recommendations.py remove-dataset ds2ds_mathe 7.pdf