"""Handling Analytical Patterns."""

from typing import Dict, Union, List

from src.services.models import SearchRequest, SearchResponse

OPTIONAL_SEARCH_ARGS = ["n"]  # Extendable list of optional arguments for recommendations

def get_node_from_label(analytical_pattern: Dict, node_label: str) -> Union[Dict, None]:
    """Returns the first node from the analytical pattern that has the given label.

    Args:
        analytical_pattern (Dict): The analytical pattern to search in
        node_label (str): The label to search for

    Returns:
        Union[Dict, None]: The node containing the label, or None if it wasn't found.
    """
    target_node = None
    for node in analytical_pattern["ap"]["nodes"]:
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

    for node in analytical_pattern["ap"]["nodes"]:
        if node["id"] != remove_node_id:
            new_ap["nodes"].append(node)

    for edge in analytical_pattern["ap"]["edges"]:
        if edge["from"] != remove_node_id and edge["to"] != remove_node_id:
            new_ap["edges"].append(edge)

    analytical_pattern["ap"] = new_ap
    return analytical_pattern


def parse_recommendation_request_ap(analytical_pattern: Dict) -> SearchRequest:
    """
    Parses the AP JSON to extract recommendation parameters.
    Expected JSON structure: analytical_pattern["ap"]["nodes"]
    """
    # Access the nested 'ap' key from the JSON
    ap_data = analytical_pattern.get("ap", {})
    nodes = ap_data.get("nodes", [])
    edges = ap_data.get("edges", [])

    # 1. Locate the Operator Node
    operator_node = next(
        (n for n in nodes if "DatasetRecommender_Operator" in n["labels"]), 
        None
    )
    if not operator_node:
        raise ValueError("Operator 'DatasetRecommender_Operator' not found.")

    operator_id = operator_node["id"]
    props = operator_node.get("properties", {})
    
    # 2. Extract 'n' and 'iid' from operator properties
    # In your JSON, seed_item_id is the 'iid' for the SearchRequest
    n_value = props.get("n", 10) # check for 'n' in properties, if not found default to 10

    # 3. Extract 'dataset_id' from the incoming 'input' edge
    # This finds the node ID of the sc:Dataset connected to the operator
    input_edge = next(
        (e for e in edges if e["to"] == operator_id and "input" in e["labels"]),
        None
    )
    
    if not input_edge:
        raise ValueError("No input dataset linked to the Recommender Operator.")
    
    # The 'from' field in the edge is the UUID of the sc:Dataset node
    source_dataset_id = input_edge["from"]

    # 4. Instantiate the Pydantic Model
    # dataset_id: from the graph topology
    # iid: from the operator properties
    # n: from the operator properties
    return SearchRequest(
        dataset_id=source_dataset_id,
        n=n_value
    )

def create_recommendation_response_ap(
    analytical_pattern: Dict, 
    search_response: SearchResponse
) -> Dict:
    """
    Updates the Analytical Pattern with item-to-item recommendations.
    Maps SearchResponse.recommendations to sc:Dataset nodes.
    """
    # 1. Locate the Operator
    operator_node = get_node_from_label(analytical_pattern, "DatasetRecommender_Operator")
    if not operator_node:
        raise ValueError("DatasetRecommender_Operator node not found.")
    
    op_id = operator_node["id"]
    
    # 2. Cleanup: Remove old output results to avoid duplication
    edges = analytical_pattern["ap"]["edges"]
    nodes = analytical_pattern["ap"]["nodes"]
    
    # Find all node IDs currently pointed to by an 'output' edge from this operator
    nodes_to_remove = [e["to"] for e in edges if e["from"] == op_id and "output" in e["labels"]]
    
    for node_id in nodes_to_remove:
        analytical_pattern = remove_node(analytical_pattern, node_id)

    # 3. Inject Recommendations
    # search_response.recommendations is a List[API_SearchResult]
    for rank, rec in enumerate(search_response.recommendations, start=1):
        item_id = rec.item_id

        # Create the Item Node (sc:Dataset)
        new_item_node = {
            "id": item_id,
            "labels": ["sc:Dataset"],
            "properties": {
                "item_id": item_id,
                "name": f"Recommended Item {rank}"
            }
        }
        analytical_pattern["ap"]["nodes"].append(new_item_node)

        # Edge: Operator -> Output -> Item (with Rank)
        analytical_pattern["ap"]["edges"].append({
            "from": op_id,
            "to": item_id,
            "labels": ["output"],
            "properties": {"rank": rank}
        })

    # 4. Attach Metadata
    analytical_pattern["metadata"] = search_response.model_dump()

    return analytical_pattern