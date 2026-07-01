"""Handling Analytical Patterns."""

from datetime import datetime
from typing import Dict, Union

import copy

from fastapi import HTTPException, status
from networkx import edges

from dataset_recsys.api.analytical_patterns.models import RecsRequest, RecsResponse
import uuid

OPTIONAL_SEARCH_ARGS = ["n"]

def get_node_from_label(analytical_pattern: Dict, node_label: str) -> Union[Dict, None]:
    """Returns the first node from the analytical pattern that has the given label.

    Args:
        analytical_pattern (Dict): The analytical pattern to search in
        node_label (str): The label to search for

    Returns:
        Union[Dict, None]: The node containing the label, or None if it wasn't found.
    """
    target_node = None
    for node in analytical_pattern["nodes"]:
        if node_label in node["labels"]:
            target_node = node
            break
    return target_node


def remove_node(analytical_pattern: Dict, remove_node_id: str) -> Dict:
    """Remove node with given ID from analytical pattern.
    Creates a new copy of the analytical pattern without all nodes with given ID
    and all edges that start or end at given ID.

    Args:
        analytical_pattern (Dict): The analytical pattern from which the node
            will be removed.
        remove_node_id (str): The node ID to remove from the analytical pattern.

    Returns:
        Dict: A copy of the given analytical pattern without the given node.
    """
    new_ap = {"nodes": [], "edges": []}

    for node in analytical_pattern["nodes"]:
        if node["id"] != remove_node_id:
            new_ap["nodes"].append(node)

    for edge in analytical_pattern["edges"]:
        if edge["from"] != remove_node_id and edge["to"] != remove_node_id:
            new_ap["edges"].append(edge)

    analytical_pattern = new_ap
    return analytical_pattern


def parse_recommendation_request_ap(analytical_pattern: Dict) -> RecsRequest:
    """
    Parses a DataGEMS AP JSON to extract recommendation parameters.
    Raises 422 if the graph structure is invalid.
    """
    # Access the nested ap key from the JSON body
    # ap = analytical_pattern.get("ap", {})
    ap_graph = analytical_pattern.get("ap", analytical_pattern)
    
    nodes = ap_graph.get("nodes", [])
    edges = ap_graph.get("edges", [])

    operator_node = next(
        (n for n in nodes if "DatasetRecommender_Operator" in n["labels"]),
        None,
    )
    if not operator_node:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: Operator 'DatasetRecommender_Operator' not found.",
        )

    operator_id = operator_node["id"]
    props = operator_node.get("properties", {})
    n_value = props.get("n", 10)

    input_edge = next(
        (e for e in edges if e["to"] == operator_id and "input" in e["labels"]),
        None,
    )
    if not input_edge:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: No input dataset linked to the Recommender Operator. Expected edge direction: seed dataset --input--> Operator.",
        )
    entity_id = input_edge["from"]

    return RecsRequest(
        entity_id=entity_id,
        n=n_value,
    )

def create_recommendation_response_ap(
    analytical_pattern: Dict,
    search_response: RecsResponse
) -> Dict:
    """
    Updates a DataGEMS AP with dataset recommendations.

    Recommended datasets are added as `sc:Dataset` nodes and linked to the
    recommender operator through ranked `output` edges.
    """
    is_wrapped = "ap" in analytical_pattern
    ap_graph = copy.deepcopy(analytical_pattern.get("ap", analytical_pattern))

    ap_node = get_node_from_label(ap_graph, "Analytical_Pattern")
    if ap_node:
        if "properties" not in ap_node:
            ap_node["properties"] = {}
        
        # Generate ISO format timestamp with 'Z' suffix indicating UTC
        # e.g., '2026-06-29T10:47:17.829Z'
        current_time_iso = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        ap_node["properties"]["endTime"] = current_time_iso

    # Locate operator
    operator_node = get_node_from_label(ap_graph, "DatasetRecommender_Operator")
    if not operator_node:
        # If it was there during parsing but gone now, something went very wrong during processing
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: DatasetRecommender_Operator node disappeared during processing."
        )
    
    op_id = operator_node["id"]

    # Cleanup old output nodes
    old_output_nodes = [
        e["to"]
        for e in ap_graph["edges"]
        if e["from"] == op_id and "output" in e["labels"]
    ]
    for node_id in old_output_nodes:
        ap_graph = remove_node(ap_graph, node_id)
        
    current_time_iso = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    operator_node["properties"]["endTime"] = current_time_iso
    
    # Inject recommendations
    for rank, rec in enumerate(search_response.recommendations, start=1):
        node_id = rec.entity_id
        new_node = {
            "id": node_id,
            "labels": ["sc:Dataset"],
            "properties": {
                "name": f"Recommended Entity {rank}"
            }
        }

        ap_graph["nodes"].append(new_node)

        # Add output edge
        ap_graph["edges"].append({
            "from": op_id,
            "to": node_id,
            "labels": ["output"],
            "properties": {"rank": rank}
        })

    return {"ap": ap_graph} if is_wrapped else ap_graph

### Template based handling of request and response metadata for APs that follow a fixed structure and only require parameter extraction and response injection without graph transformations.
def parse_template_request_metadata(request_body: dict) -> RecsRequest:
    """
    Extracts runtime parameters from the metadata structure of a template AP request.
    Raises 422 if metadata or execution fields are malformed.
    """
    metadata = request_body.get("metadata", {})
    execution_type = metadata.get("execution_type", "REQUEST")
    
    if execution_type != "REQUEST":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid execution_type '{execution_type}' for a template request. Expected 'REQUEST'."
        )
        
    parameters = metadata.get("parameters", {})
    inputs = parameters.get("inputs", {})
    
    seed = inputs.get("seed")
    n_value = inputs.get("n", 2)  # Default fallback mirroring template defaults
    
    if not seed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed Metadata: 'inputs.seed' is missing from request parameters."
        )
        
    return RecsRequest(
        entity_id=str(seed),
        n=int(n_value)
    )


def create_template_response_metadata(request_body: dict, search_response: RecsResponse) -> dict:
    """
    Copies the original fixed template structure and overrides the metadata
    block to reflect a successful execution response.
    """
    # Create a deep copy of the request payload to preserve the fixed "ap" template graph
    import copy
    response_payload = copy.deepcopy(request_body)
    
    metadata = response_payload.get("metadata", {})
    parameters = metadata.get("parameters", {})
    
    # 1. Elevate execution type and generate a tracking execution UUID
    metadata["execution_type"] = "RESPONSE"
    metadata["uuid"] = str(uuid.uuid4())
    metadata["status"] = "SUCCESS"
    metadata["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # 2. Extract recommended entity strings from the RecsResponse object
    rec_ids = [rec.entity_id for rec in search_response.recommendations]
    
    # 3. Populate outputs inside parameters
    parameters["outputs"] = {
        "recommendations": rec_ids
    }
    
    # Re-assign clean runtime references back to payload structural root
    metadata["parameters"] = parameters
    response_payload["metadata"] = metadata
    
    return response_payload