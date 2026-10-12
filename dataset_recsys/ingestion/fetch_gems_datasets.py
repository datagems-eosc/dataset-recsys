"""
Fetch all datasets from DataGEMS DMM API and extract the sc:Dataset nodes
into a flat list (optionally with projection).

This script calls: GET https://app.datagems.eu/dmm/api/v1/dataset/search
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import os
from dotenv import load_dotenv

DMM_BASE_URL = os.getenv("DMM_BASE_URL", "https://datagems-dev.scayle.es/dmm")  # Default to a placeholder if not set
BASE_URL = DMM_BASE_URL + "/api/v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "gems_datasets_metadata" / "moma"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For local testing we do load environment variables from a .env file, but in production these should be set in the environment directly (e.g. via Kubernetes secrets or similar).
load_dotenv()

def get_access_token() -> str:
    """
    Retrieves an OAuth2 access token using the Password Grant flow.
    """
    payload = {
        "grant_type": "password",
        "client_id": os.getenv("DATAGEMS_CLIENT_ID"),
        "username": os.getenv("DATAGEMS_USER"),
        "password": os.getenv("DATAGEMS_PASSWORD"),
        "scope": os.getenv("DATAGEMS_SCOPE", "openid profile email"),
    }
    
    response = requests.post(
        os.getenv("DATAGEMS_AUTH_URL"), 
        data=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"Failed to retrieve token: {response.status_code} - {response.text}")
        response.raise_for_status()
        
    return response.json()["access_token"]

@dataclass
class DatasetProfile:
    # Minimal text profile for embedding generation.
    id: str
    title: str = ""
    headline: str = ""
    description: str = ""
    keywords: str = ""
    field_of_science: str = ""
    catalog_summary: str = ""

def _is_dataset_node(node: Dict[str, Any]) -> bool:
    return "sc:Dataset" in (node.get("labels") or [])

def fetch_search(
    token: str,
    projection_fields: Optional[List[str]] = None,
    status: Optional[List[str]] = None,
    offset: int = 0,
    count: int = 100,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Calls /dataset/search and returns the complete JSON payload.
    """
    url = f"{BASE_URL}/dataset/search"

    params: List[tuple[str, str]] = [
        ("offset", str(offset)),
        ("count", str(count)),
    ]

    if projection_fields:
        for f in projection_fields:
            params.append(("properties", f))

    if status:
        for s in status:
            params.append(("status", s))

    resp = requests.get(
        url,
        params=params if params else None,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()

def summarize_dataset_property_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect sc:Dataset nodes and report which metadata fields exist and how often they appear. 
    This just summarizes what is present in the raw API response.
    """
    graphs = payload.get("datasets", [])
    dataset_nodes: List[Dict[str, Any]] = []

    # Gather all sc:Dataset nodes from the response
    for g in graphs:
        for n in g.get("nodes", []) or []:
            if _is_dataset_node(n):
                dataset_nodes.append(n)

    total = len(dataset_nodes)

    # Count how many times each property key appears
    key_counts: Dict[str, int] = {}
    for n in dataset_nodes:
        props = n.get("properties") or {}
        for k in props.keys():
            key_counts[k] = key_counts.get(k, 0) + 1

    distinct_keys = sorted(key_counts.keys())

    # Compute coverage for each key (how many datasets have this key vs total)
    coverage: Dict[str, Dict[str, Any]] = {}
    for k in distinct_keys:
        present = key_counts[k]
        missing = total - present
        coverage[k] = {
            "present": present,
            "missing": missing,
            "coverage": round(present / total, 4) if total else 0.0,
        }

    return {
        "dataset_nodes_total": total,
        "distinct_property_keys": distinct_keys,
        "property_key_coverage": coverage,
    }

def extract_dataset_profiles(payload: Dict[str, Any]) -> List[DatasetProfile]:
    """
    Extract minimal text-focused dataset profiles from sc:Dataset nodes.
    These are intended for building embedding input text.
    """
    graphs = payload.get("datasets", [])
    out: List[DatasetProfile] = []

    for g in graphs:
        nodes = g.get("nodes", [])
        for n in nodes:
            if not _is_dataset_node(n):
                continue
            ds_id = n.get("id") or (n.get("properties") or {}).get("id")
            if not ds_id:
                continue
            props = n.get("properties") or {}

            title = (props.get("name") or "").strip()
            headline_raw = (props.get("headline") or "").strip()
            headline = "" if not headline_raw or headline_raw.lower() == title.lower() else headline_raw

            kw = props.get("keywords")
            if isinstance(kw, list):
                keywords = ", ".join(str(x).strip() for x in kw if str(x).strip())
            elif kw is None:
                keywords = ""
            else:
                keywords = str(kw).strip()

            fos = props.get("fieldOfScience")
            if isinstance(fos, list):
                field_of_science = ", ".join(str(x).strip() for x in fos if str(x).strip())
            elif fos is None:
                field_of_science = ""
            else:
                field_of_science = str(fos).strip()

            out.append(
                DatasetProfile(
                    id=ds_id,
                    title=title,
                    headline=headline,
                    description=(props.get("description") or "").strip(),
                    keywords=keywords,
                    field_of_science=field_of_science,
                )
            )

    return out

def run_ingestion(token: str) -> tuple[Dict[str, Any], List[DatasetProfile]]:
    """
    Fetch all DataGEMS data without field projection and return full payload and profiles.
    """
    all_datasets: List[Dict[str, Any]] = []
    offset = 0
    count = 100

    try:
        while True:
            # Do NOT pass projection_fields so the API returns all node properties and graph details
            page_data = fetch_search(token, projection_fields=None, offset=offset, count=count)
            graphs = page_data.get("datasets", [])
            if not graphs:
                break
            all_datasets.extend(graphs)

            total = page_data.get("total")
            offset += len(graphs)

            if total is not None and offset >= total:
                break
            if len(graphs) < count:
                break

        # Save the complete un-filtered dataset graph array
        raw_data = {"datasets": all_datasets}
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        raise
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}", file=sys.stderr)
        raise

    print(f"Fetched dataset graphs: {len(raw_data.get('datasets', []))}")

    profiles = extract_dataset_profiles(raw_data)
    return raw_data, profiles

def fetch_catalog() -> List[DatasetProfile]:
    _, profiles = run_ingestion(get_access_token())
    return profiles 

if __name__ == "__main__":
    raw_data, profiles = run_ingestion(get_access_token())

    # Write complete un-truncated raw response payload to JSON
    raw_path = OUTPUT_DIR / "datagems_datasets_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote complete raw dump: {raw_path}")

    profiles_path = OUTPUT_DIR / "datagems_dataset_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump([p.__dict__ for p in profiles], f, ensure_ascii=False, indent=2)
    print(f"Wrote minimal text profiles: {profiles_path}")

__all__ = [
    "DatasetProfile",
    "extract_dataset_profiles",
    "fetch_catalog",
    "fetch_search",
    "run_ingestion",
    "summarize_dataset_property_fields",
]