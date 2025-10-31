import numpy as np

def recall_at_n(predictions: dict, ground_truth: dict, n=10):
    """
    Compute the average Recall@n for a set of predictions against ground truth.

    Recall@n measures the proportion of relevant items (from the ground truth)
    that are present in the top-n predicted items for each item.

    Parameters
    ----------
    predictions : dict
        A dictionary mapping item IDs to an ordered list of predicted related item IDs.
    ground_truth : dict
        A dictionary mapping item IDs to a set of true related item IDs.
    n : int, optional
        The number of top predictions to consider for each item (default is 10).

    Returns
    -------
    float
        The average recall@n across all items with non-empty ground truth.

    Notes
    -----
    - Items with no ground truth related items are skipped in the calculation.
    - If none of the items have ground truth, the function will raise a ZeroDivisionError.
    """
    recalls = []
    for item_id, predicted in predictions.items():
        true_related = ground_truth.get(item_id, set())
        if not true_related:
            continue
        top_n = list(predicted)[:n]
        hits = set(top_n) & true_related
        recalls.append(len(hits) / len(true_related))
    return sum(recalls) / len(recalls)

def tndcg_at_n(predictions: dict, ground_truth: dict, n=10):
    """
    Compute the average truncated Normalized Discounted Cumulative Gain (NDCG@n) for a set of predictions.

    NDCG@n evaluates the ranking quality of the top-n predicted items for each item,
    taking into account the positions of relevant items. Higher NDCG indicates better ranking.

    Parameters
    ----------
    predictions : dict
        A dictionary mapping item IDs to an ordered list (or iterable) of predicted related item IDs.
    ground_truth : dict
        A dictionary mapping item IDs to a set of true related item IDs.
    n : int, optional
        The number of top predictions to consider for each item (default is 10).

    Returns
    -------
    float
        The average NDCG@n across all items with non-empty ground truth.

    Notes
    -----
    - Items with no ground truth related items are skipped in the calculation.
    - The ideal DCG (IDCG) is computed by sorting the relevance scores for the top-n predictions in descending order.
    - If there are no relevant items in the top-n, NDCG is set to 0 for that item.
    """
    def dcg(relevance_scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    ndcgs = []
    for dataset_id, predicted in predictions.items():
        true_related = ground_truth.get(dataset_id, set())
        if not true_related:
            continue
        top_n = list(predicted)[:n]
        relevance_scores = [1 if pid in true_related else 0 for pid in top_n]
        ideal_scores = sorted(relevance_scores, reverse=True)
        dcg_val = dcg(relevance_scores)
        idcg_val = dcg(ideal_scores)
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
        ndcgs.append(ndcg)
    return sum(ndcgs) / len(ndcgs)