import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

import timm
import numpy as np

from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, List, Optional, Callable, TypedDict, Tuple
from pathlib import Path
import tomllib
import random
import os

# This is loosely based on the implementation in:
# https://www.leoniemonigatti.com/blog/pytorch-image-classification.html
# https://github.com/Tony-Y/pytorch_warmup/blob/master/examples/emnist/main.py

# Type Aliases -----------------------------------------------------------------

# During training and evaluation, the caller may provide a dictionary of
# metrics functions, e.g. {"accuracy" : sklearn.metrics::accuracy_score },
# which take a list of labels and predictions as it's input.
MetricFn = Callable[[List[int], List[int]], float]
MetricsFns = Dict[str, MetricFn]
MetricsDict = Dict[str, float]


# Running statistics collected during training and evaluation
class StatsDict(TypedDict):
    loss: float
    n_correct: int
    n_total: int
    predictions: Optional[List[int]]
    labels: Optional[List[int]]
    n_batches: int


# Constants --------------------------------------------------------------------

EMNIST_N_CLASSES = 62
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_DATA_DIR = SCRIPT_DIR / "00_raw_data"
CONFIG_PATH = SCRIPT_DIR / "config.toml"

# Config -----------------------------------------------------------------------

with open(CONFIG_PATH, "rb") as f:
    cfg_dict = tomllib.load(f)

cfg = SimpleNamespace(
    **{
        k: SimpleNamespace(**v) if isinstance(v, dict) else v
        for k, v in cfg_dict.items()
    }
)

cfg.system = SimpleNamespace(
    device=torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
)


# TODO Replace with formal logging
def print_lvl(*args, level=1, **kwargs):
    """Print if level >= cfg.dev.print_level"""
    if level >= cfg.dev.print_level:
        print(*args, **kwargs)


# Print configuration
print_lvl(f"Using device: {cfg.system.device}")


def set_seed(seed=123):
    # Seed python, numpy, and torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Metal Performance Shaders (MPS) on Mac
    # See: https://developer.apple.com/metal/pytorch/
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # CUDA-specific seeding
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Apply seed
set_seed(cfg.train.seed)

# Helpers ----------------------------------------------------------------------

# TODO We'll want to look into moving functions into a "package" of some sort
# - Look into python-y way to do this
# - Eventually we'll want to add better function documentation (e.g. docstring)


def init_stats(collect_predictions: bool = False) -> StatsDict:
    """Initiate a statistics dictionary to record training/evaluation statistics"""

    return {
        "loss": 0.0,
        "n_correct": 0,
        "n_total": 0,
        "predictions": [] if collect_predictions else None,
        "labels": [] if collect_predictions else None,
        "n_batches": 0,
    }


def update_stats(
    stats: StatsDict, loss: float, predictions: torch.Tensor, labels: torch.Tensor
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
    stats: StatsDict, metrics_fns: Optional[MetricsFns] = None
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
                "Set collect_predictions=True in init_stats()"
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


def train_epoch(
    model, loader, criterion, optimizer, cfg, metrics_fns: Optional[MetricsFns] = None
) -> MetricsDict:
    """Train for one epoch"""
    model.train()
    stats = init_stats(collect_predictions=(metrics_fns is not None))

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(cfg.system.device), labels.to(cfg.system.device)

        # Update the model
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update statistics
        _, predicted = torch.max(outputs.data, 1)
        update_stats(stats, loss.item(), predicted, labels)

    epoch_metrics = calculate_metrics(stats, metrics_fns)
    return epoch_metrics


def evaluate(
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
    stats = init_stats(
        collect_predictions=collect_predictions or (metrics_fns is not None)
    )

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(cfg.system.device), labels.to(cfg.system.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            update_stats(stats, loss.item(), predicted, labels)

    epoch_metrics = calculate_metrics(stats, metrics_fns)
    return epoch_metrics, stats["predictions"], stats["labels"]


def fit(
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

    history = {"train_metrics": [], "valid_metrics": []}

    n_epochs = cfg.train.epochs
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        set_seed(cfg.train.seed + epoch)

        # Train for one epoch and update the learning rate scheduler
        train_metrics = train_epoch(
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
            # evaluate() returns predictions and labels as well as metrics
            valid_metrics, _, _ = evaluate(
                model,
                valid_loader,
                criterion,
                cfg,
                desc="Validating",
                metrics_fns=valid_metrics_fns,
            )
            history["valid_metrics"].append(valid_metrics)
            print_metrics(valid_metrics, header="Validation Metrics:")

    return model, history


# Load -------------------------------------------------------------------------

# EMNIST images are rotated, flipping them for easier human readability
# https://stackoverflow.com/questions/48532761/letters-in-emnist-training-dataset-are-rotated-and-little-vague
transform = transforms.Compose(
    [
        lambda img: torchvision.transforms.functional.rotate(img, -90),
        lambda img: torchvision.transforms.functional.hflip(img),
        torchvision.transforms.ToTensor(),
        # Known EMNIST mean and SD, see:
        # https://github.com/Tony-Y/pytorch_warmup/blob/master/examples/emnist/main.py
        transforms.Normalize((0.1751,), (0.3332,)),
    ]
)

train_data = datasets.EMNIST(
    root=RAW_DATA_DIR,
    split="byclass",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.EMNIST(
    root=RAW_DATA_DIR,
    split="byclass",
    train=False,
    download=True,
    transform=transform,
)

print_lvl(f"Train samples: {len(train_data)}")
print_lvl(f"Test samples: {len(test_data)}")
print_lvl(f"Classes: {len(train_data.classes)}")

# Subset the images for faster training
if cfg.dev.mode == "dev":
    n_train = len(train_data)
    n_train_subset = int(len(train_data) * cfg.dev.data_frac)
    train_indices = np.random.choice(n_train, n_train_subset, replace=False)
    train_data = Subset(train_data, train_indices)
    print_lvl(f"Subset train data: N = {n_train} -> {n_train_subset}")

    n_test = len(test_data)
    n_test_subset = int(n_test * cfg.dev.data_frac)
    test_indices = np.random.choice(n_test, n_test_subset, replace=False)
    test_data = Subset(test_data, test_indices)
    print_lvl(f"Subset test data: N = {n_test} -> {n_test_subset}")

# Split ------------------------------------------------------------------------

# Split into training and validation sets
valid_size = int(len(train_data) * cfg.train.valid_frac)
train_size = len(train_data) - valid_size

train_data, valid_data = random_split(
    train_data,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(cfg.train.seed),
)
print_lvl(f"Train/Validate split: {len(train_data)}/{len(valid_data)}")

train_loader = DataLoader(
    train_data,
    batch_size=cfg.train.batch_size,
    shuffle=True,
    num_workers=cfg.train.num_workers,
)
valid_loader = DataLoader(
    valid_data,
    batch_size=cfg.train.batch_size,
    shuffle=False,
    num_workers=cfg.train.num_workers,
)
test_loader = DataLoader(
    test_data,
    batch_size=cfg.train.batch_size,
    shuffle=False,
    num_workers=cfg.train.num_workers,
)

# Train ------------------------------------------------------------------------

model = timm.create_model(
    cfg.model.backbone,
    pretrained=True,
    num_classes=EMNIST_N_CLASSES,
    # EMNIST images are greyscale (1 channel), not RGB (3 channels)
    in_chans=1,
).to(cfg.system.device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.train.learning_rate,
    weight_decay=0,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.learning_rate_min
)

fit(model, optimizer, criterion, scheduler, cfg, train_loader, valid_loader)
