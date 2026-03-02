from typing import Dict, List, Optional, Callable, TypedDict

import torch

MetricFn = Callable[[List[int], List[int]], float]
MetricsFns = Dict[str, MetricFn]
MetricsDict = Dict[str, float]


class StatisticsDict(TypedDict):
    loss: float
    n_correct: int
    n_total: int
    predictions: Optional[List[int]]
    labels: Optional[List[int]]
    n_batches: int


def init_statistics(collect_predictions: bool = False) -> StatisticsDict:
    """Initiate a statistics dictionary to record training/evaluation statistics"""

    return {
        "loss": 0.0,
        "n_correct": 0,
        "n_total": 0,
        "predictions": [] if collect_predictions else None,
        "labels": [] if collect_predictions else None,
        "n_batches": 0,
    }


def update_statistics(
    stats: StatisticsDict, loss: float, predictions: torch.Tensor, labels: torch.Tensor
) -> None:
    """Update statistics dictionary with batch results (in-place)"""

    stats["loss"] += loss
    stats["n_batches"] += 1

    batch_size = labels.size(0)
    stats["n_total"] += batch_size
    stats["n_correct"] += (predictions == labels).sum().item()

    if stats["predictions"] is not None:
        stats["predictions"].extend(predictions.cpu().numpy())
        stats["labels"].extend(labels.cpu().numpy())


def calculate_metrics(
    stats: StatisticsDict, metrics_fns: Optional[MetricsFns] = None
) -> MetricsDict:
    """Calculate final metrics from accumulated statistics"""

    if stats["n_total"] == 0:
        raise ValueError("No samples processed (n_total = 0)")

    results = {
        "loss": stats["loss"] / stats["n_batches"],
        "accuracy": 100 * stats["n_correct"] / stats["n_total"],
    }

    if metrics_fns is not None:
        if stats["predictions"] is None or stats["labels"] is None:
            raise ValueError(
                "metrics_fns provided but predictions not collected. "
                "Set collect_predictions=True in init_statistics()"
            )

        predictions = stats["predictions"]
        labels = stats["labels"]

        if len(predictions) != len(labels):
            raise ValueError(
                f"Length mismatch: predictions ({len(predictions)}) "
                f"vs labels ({len(labels)})"
            )

        if len(predictions) != stats["n_total"]:
            raise ValueError(
                f"predictions length ({len(predictions)}) does not match "
                f"n_total ({stats['n_total']})"
            )

        # Attempt to apply the metrics functions. Catching errors since these
        # are arbitrary-ish functions.
        for name, metric_fn in metrics_fns.items():
            try:
                results[name] = metric_fn(labels, predictions)
            except Exception as e:
                raise ValueError(f"Error computing metric '{name}': {e}") from e

    return results


def print_metrics(metrics: MetricsDict, header: str = "Metrics:") -> None:
    """Print metrics dictionary"""
    print(header)
    for name, value in metrics.items():
        print(f"  {name.replace('_', ' ').title()}: {value:.4f}")
