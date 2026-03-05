"""
Fetch all datasets from DataGEMS DMM API and extract the sc:Dataset nodes
into a flat list (optionally with projection).

This script calls: GET https://datagems-dev.scayle.es/dmm/api/v1/dataset/search
  with optional `properties=...` projection params.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://datagems-dev.scayle.es/dmm/api/v1"

@dataclass
class DatasetProfile:
    # Minimal text profile for embedding generation.
    id: str
    title: str = ""
    headline: str = ""
    description: str = ""
    keywords: str = ""
    field_of_science: str = ""

def _is_dataset_node(node: Dict[str, Any]) -> bool:
    return "sc:Dataset" in (node.get("labels") or [])

def fetch_search(
    projection_fields: Optional[List[str]] = None, # if requesting specific fields to return, e.g. ["name", "description"]
    status: Optional[List[str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Calls /dataset/search and returns the JSON payload.

    Structure of the response:

    datasets
    ├─ graph_1
    │   ├─ nodes
    │   │   ├─ sc:Dataset        ← the dataset entity
    │   │   ├─ cr:FileObject     ← files belonging to the dataset
    │   │   ├─ cr:RecordSet      ← dataset schema / table
    │   │   ├─ cr:Field          ← column definitions
    │   │   └─ ?                 ← other node types
    │   └─ edges                 ← relations between those nodes
    ├─ graph_2
    │   └─ ...
    """
    url = f"{BASE_URL}/dataset/search"

    # The DataGEMS API allows requesting only specific dataset properties using
    # repeated 'properties' parameters. For example:
    # .../dataset/search?properties=name&properties=description
    params: List[tuple[str, str]] = []
    if projection_fields:
        for f in projection_fields:
            params.append(("properties", f))

    if status:
        # Add dataset status filters to the request.
        # The API allows filtering datasets by their dg:status value (e.g. "loaded", "ready").
        # We append one "status" parameter per value so the final URL may look like:
        # .../dataset/search?status=loaded&status=ready
        for s in status:
            params.append(("status", s))

    resp = requests.get(
        url,
        params=params if params else None,
        headers={"Accept": "application/json"},
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
            if not ds_id: # if dataset has no ID, skip it since we can't reference it
                continue
            props = n.get("properties") or {}

            title = (props.get("name") or "").strip()
            headline_raw = (props.get("dg:headline") or "").strip()
            # If headline is missing or identical to title, we set it to empty string to avoid redundancy in the profile text.
            headline = "" if not headline_raw or headline_raw.lower() == title.lower() else headline_raw

            kw = props.get("dg:keywords")
            if isinstance(kw, list):
                keywords = ", ".join(str(x).strip() for x in kw if str(x).strip())
            elif kw is None:
                keywords = ""
            else:
                keywords = str(kw).strip()

            fos = props.get("dg:fieldOfScience")
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

def main() -> int:
    try:
        payload = fetch_search()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return 2
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}", file=sys.stderr)
        return 2

    print(f"Fetched dataset graphs: {len(payload.get('datasets', []))}")

    raw_path = "datagems_datasets_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote raw dump: {raw_path}")

    # Summarize which dataset metadata fields exist and how often they appear.
    fields_summary = summarize_dataset_property_fields(payload)
    fields_path = "datagems_dataset_fields_summary.json"
    with open(fields_path, "w", encoding="utf-8") as f:
        json.dump(fields_summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote dataset fields summary: {fields_path}")

    profiles = extract_dataset_profiles(payload)

    profiles_path = "datagems_dataset_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump([p.__dict__ for p in profiles], f, ensure_ascii=False, indent=2)
    print(f"Wrote minimal text profiles: {profiles_path}")

if __name__ == "__main__":
    main()