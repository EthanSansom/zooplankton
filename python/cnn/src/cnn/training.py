from typing import Tuple, Optional

from tqdm import tqdm
import torch

from .utils import set_seed
from .metrics import (
    MetricsDict,
    MetricsFns,
    update_statistics,
    init_statistics,
    calculate_metrics,
    print_metrics,
)

# This is loosely based on the implementation in:
# https://www.leoniemonigatti.com/blog/pytorch-image-classification.html
# https://github.com/Tony-Y/pytorch_warmup/blob/master/examples/emnist/main.py


def train_model(
    model, loader, criterion, optimizer, cfg, metrics_fns: Optional[MetricsFns] = None
) -> MetricsDict:
    """Train for one epoch"""
    model.train()
    stats = init_statistics(collect_predictions=(metrics_fns is not None))

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(cfg.metadata.device), labels.to(cfg.metadata.device)

        # Update the model
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update statistics
        _, predicted = torch.max(outputs.data, 1)
        update_statistics(stats, loss.item(), predicted, labels)

    epoch_metrics = calculate_metrics(stats, metrics_fns)
    return epoch_metrics


def validate_model(
    model,
    loader,
    criterion,
    cfg,
    metrics_fns: Optional[MetricsFns] = None,
) -> MetricsDict:
    """Validate a model"""
    metrics, _, _ = evaluate_model(
        model,
        loader,
        criterion,
        cfg,
        desc="Validating",
        metrics_fns=metrics_fns,
        collect_predictions=False,
    )
    return metrics


def test_model(
    model,
    loader,
    criterion,
    cfg,
    metrics_fns: Optional[MetricsFns] = None,
) -> Tuple[MetricsDict, Optional[list], Optional[list]]:
    """Test a model"""
    return evaluate_model(
        model,
        loader,
        criterion,
        cfg,
        desc="Training",
        metrics_fns=metrics_fns,
        collect_predictions=True,
    )


def evaluate_model(
    model,
    loader,
    criterion,
    cfg,
    desc="Evaluating",
    metrics_fns: Optional[MetricsFns] = None,
    collect_predictions=False,
) -> Tuple[MetricsDict, Optional[list], Optional[list]]:
    """Evaluate for one epoch"""
    model.eval()
    needs_predictions = metrics_fns is not None
    stats = init_statistics(collect_predictions or needs_predictions)

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = (
                inputs.to(cfg.metadata.device),
                labels.to(cfg.metadata.device),
            )

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            update_statistics(stats, loss.item(), predicted, labels)

    epoch_metrics = calculate_metrics(stats, metrics_fns)
    return epoch_metrics, stats["predictions"], stats["labels"]


def fit_model(
    model,
    optimizer,
    criterion,
    scheduler,
    cfg,
    train_loader,
    valid_loader=None,
    train_metrics_fns: Optional[MetricsFns] = None,
    valid_metrics_fns: Optional[MetricsFns] = None,
):
    """Train model with optional validation"""
    if valid_metrics_fns is None:
        valid_metrics_fns = train_metrics_fns

    # TODO: Dump the metrics to some results file for later reference!
    history = {"train_metrics": [], "valid_metrics": []}

    n_epochs = cfg.train.epochs
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        set_seed(cfg.train.seed + epoch)

        train_metrics = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            cfg,
            metrics_fns=train_metrics_fns,
        )
        scheduler.step()

        history["train_metrics"].append(train_metrics)
        print_metrics(train_metrics, header="Training Metrics:")

        if valid_loader:
            valid_metrics = validate_model(
                model,
                valid_loader,
                criterion,
                cfg,
                metrics_fns=valid_metrics_fns,
            )
            history["valid_metrics"].append(valid_metrics)
            print_metrics(valid_metrics, header="Validation Metrics:")

    return model, history
