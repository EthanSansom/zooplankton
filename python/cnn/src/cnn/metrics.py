from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def classification_metrics(
    true: List[str],
    pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    Compute flat classification metrics from string label lists.
    Compatible with both FlatModel and LCPNModel predictions.

    Precision, recall and F1 use macro-averaging (equal weight per class).

    Args:
        true:   True class names
        pred:   Predicted class names
        labels: Ordered list of class names. If None, inferred from data.
                Pass explicitly to ensure consistent ordering across models.

    Returns:
        {
            "accuracy":  float,
            "precision": float (macro),
            "recall":    float (macro),
            "f1":        float (macro),
            "report":    str (sklearn classification report),
            "confusion": np.ndarray (confusion matrix),
        }
    """
    return {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "f1": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
        "report": classification_report(true, pred, labels=labels, zero_division=0),
        "confusion": confusion_matrix(true, pred, labels=labels),
    }


def hierarchical_metrics(
    true_paths: List[List[str]],
    pred_paths: List[List[str]],
) -> Dict:
    """
    Compute hierarchical precision (hier_precision), recall (hier_recall), and
    F-score (hier_fscore). Compatible with LCPNModel only, since path information
    is required.

    Let pred_i be the set of predicted hierarchical labels for sample i
    (e.g. {root, round, Q}) and true_i be the set of true hierarchical
    labels (e.g. {root, round, O}), then:
    - hier_precision = sum(|pred_i & true_i|) / sum(|pred_i|)
    - hier_recall = sum(|pred_i & true_i|) / sum(|true_i|)

    Reference: Kiritchenko et al. (2006), as cited in HiClass documentation.
    See: https://hiclass.readthedocs.io/en/latest/algorithms/metrics.html

    Args:
        true_paths: True root-to-node paths, one per sample
        pred_paths: Predicted root-to-leaf paths, one per sample

    Returns:
        {
            "hier_precision": float,
            "hier_recall": float,
            "hier_fscore": float
        }
    """

    # We're dealing with sets here. Consider the following hierarchical prediction:
    # Pred: [letter, round, Q]
    # True: [letter, round, O]
    #
    # - intersection: {letter, round, Q} & {letter, round, O} = {letter, round}
    # - intersection size (AKA cardinality): |{letter, round}| = 2
    # - pred_size = |{letter, round, Q}| = 3, true size = |{letter, round, O}| = 3
    #
    # Note that if every leaf in the hierarchy has the same distance from the
    # root (e.g. [letter, round, C] and [letter, angular, V] form length-3 paths)
    # then `hier_precision` and `hier_recall` are equivilant.
    total_intersection_size = 0
    total_pred_size = 0
    total_true_size = 0
    for pred, true in zip(pred_paths, true_paths):
        pred_set = set(pred)
        true_set = set(true)

        total_intersection_size += len(pred_set & true_set)
        total_pred_size += len(pred_set)
        total_true_size += len(true_set)

    hier_precision = (
        total_intersection_size / total_pred_size if total_pred_size > 0 else 0.0
    )
    hier_recall = (
        total_intersection_size / total_true_size if total_true_size > 0 else 0.0
    )
    hier_fscore = (
        2 * hier_precision * hier_recall / (hier_precision + hier_recall)
        if (hier_precision + hier_recall) > 0
        else 0.0
    )

    return {
        "hier_precision": hier_precision,
        "hier_recall": hier_recall,
        "hier_fscore": hier_fscore,
    }


def flat_predictions_to_names(
    predictions: List[int],
    index_to_name: Dict[int, str],
) -> List[str]:
    """
    Convert FlatModel integer predictions to string class names,
    for use with classification_metrics().

    Args:
        predictions:   Integer class indices from FlatModel.test()
        index_to_name: Mapping from integer index to class name.

    Returns:
        List of class name strings
    """
    return [index_to_name[p] for p in predictions]


def print_metrics(metrics: Dict, header: Optional[str] = None) -> None:
    """
    Print a metrics dict in a readable format.

    Args:
        metrics: Dict from classification_metrics() or hierarchical_metrics()
        header:  Optional header string printed above metrics
    """
    if header:
        print(header)

    for key, value in metrics.items():
        if key == "report":
            print(f"\n{value}")
        elif key == "confusion":
            print("  confusion: (use plot_confusion_matrix() to visualise)")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
