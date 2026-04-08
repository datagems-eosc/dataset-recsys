"""Handling Analytical Patterns."""

from typing import Dict, Union, List
import uuid
from fastapi import HTTPException, status # Add these imports
from dataset_recsys.api.analytical_patterns.models import RecsRequest, RecsResponse

OPTIONAL_SEARCH_ARGS = ["n"]  # Extendable list of optional arguments for recommendations
MATHE_DATASET_ID = "b551f361-3f61-4ccf-a001-7c28d065c30d"

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
    Parses the AP JSON to extract recommendation parameters.
    Raises 422 if the graph structure is invalid.
    """
    # Access the nested 'ap' key from the JSON
    # ap_data = analytical_pattern.get("ap", {})
    nodes = analytical_pattern.get("nodes", [])
    edges = analytical_pattern.get("edges", [])

    # 1. Locate the Operator Node
    operator_node = next(
        (n for n in nodes if "DatasetRecommender_Operator" in n["labels"]), 
        None
    )
    if not operator_node:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: Operator 'DatasetRecommender_Operator' not found."
        )

    operator_id = operator_node["id"]
    props = operator_node.get("properties", {})
    
    # 2. Extract 'n' and 'iid' from operator properties
    # In your JSON, seed_entity_id is the 'iid' for the RecsRequest
    n_value = props.get("n", 10) # check for 'n' in properties, if not found default to 10
    application = props.get("application")
    if not application:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: Operator missing 'application' in properties."
        )
    
    if application == "mathe":
        if not props.get("entity_id"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Malformed AP Graph: For 'mathe' application, 'entity_id' must be provided in operator properties."
            )

        return RecsRequest(
            application=application,
            entity_id=props.get("entity_id"),
            n=n_value
        )

    # 3. Extract 'dataset_id' from the incoming 'input' edge
    # This finds the node ID of the sc:Dataset connected to the operator
    input_edge = next(
        (e for e in edges if e["to"] == operator_id and "input" in e["labels"]),
        None
    )
    
    if not input_edge:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: No input dataset linked to the Recommender Operator."
        )
    
    # The 'from' field in the edge is the UUID of the sc:Dataset node
    entity_id = input_edge["from"]

    # 4. Instantiate the Pydantic Model
    return RecsRequest(
        application=application,
        entity_id=entity_id,
        n=n_value
    )

def create_recommendation_response_ap(
    analytical_pattern: Dict,
    search_response: RecsResponse
) -> Dict:
    """
    Updates the Analytical Pattern with recommendations.

    For 'mathe':
        - Generates cr:FileObject nodes with unique IDs (UUIDs)

    For other applications:
        - Generates sc:Dataset nodes
    """
    # Locate operator
    operator_node = get_node_from_label(analytical_pattern, "DatasetRecommender_Operator")
    if not operator_node:
        # If it was there during parsing but gone now, something is very wrong
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Malformed AP Graph: DatasetRecommender_Operator node disappeared during processing."
        )
    
    op_id = operator_node["id"]
    props = operator_node.get("properties", {})
    application = props.get("application")
    
    # Cleanup old output nodes
    old_output_nodes = [e["to"] for e in analytical_pattern["edges"] 
                        if e["from"] == op_id and "output" in e["labels"]]
    for node_id in old_output_nodes:
        analytical_pattern = remove_node(analytical_pattern, node_id)
    
    # Inject recommendations
    for rank, rec in enumerate(search_response.recommendations, start=1):
        if application == "mathe":
            node_id = str(uuid.uuid4())  # unique graph ID
            new_node = {
                "id": node_id,
                "labels": ["cr:FileObject"],
                "properties": {
                    "fileName": rec.id,  # keep original file name as reference
                    "rank": rank
                }
            }
        else:
            node_id = rec.id
            new_node = {
                "id": node_id,
                "labels": ["sc:Dataset"],
                "properties": {
                    "name": f"Recommended Entity {rank}"
                }
            }
        
        analytical_pattern["nodes"].append(new_node)

        # Add output edge
        analytical_pattern["edges"].append({
            "from": op_id,
            "to": node_id,
            "labels": ["output"],
            "properties": {"rank": rank}
        })
    
    # Mathe-specific cleanup
    if application == "mathe":
        # Remove input edges
        analytical_pattern["edges"] = [
            e for e in analytical_pattern["edges"] if "input" not in e["labels"]
        ]
        # Remove any sc:Dataset nodes (seed dataset or others)
        analytical_pattern["nodes"] = [
            n for n in analytical_pattern["nodes"] if "sc:Dataset" not in n["labels"]
            or n["id"] == op_id  # keep operator node if it was mislabeled
        ]
    else:
        # Step 1: find input edges
        input_edges = [
            e for e in analytical_pattern["edges"]
            if "input" in e["labels"]
        ]

        # Step 2: collect dataset IDs used as input
        input_node_ids = {e["from"] for e in input_edges}

        # Step 3: remove input edges
        analytical_pattern["edges"] = [
            e for e in analytical_pattern["edges"]
            if "input" not in e["labels"]
        ]

        # Step 4: remove input dataset nodes
        analytical_pattern["nodes"] = [
            n for n in analytical_pattern["nodes"]
            if n["id"] not in input_node_ids
        ]
    
    return analytical_pattern