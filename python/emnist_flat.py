import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path
import tomllib
import random
import os

# NOTE: Based on:
# https://www.leoniemonigatti.com/blog/pytorch-image-classification.html

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

# cfg.root_dir = "images"
# cfg.image_size = 256
# cfg.batch_size = 32
# cfg.n_classes = 2
# cfg.backbone = 'resnet18'
# cfg.learning_rate = 1e-4
# cfg.lr_min = 1e-5
# cfg.epochs = 5
# cfg.n_folds = 3


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


def train_epoch(model, loader, criterion, optimizer, scheduler, cfg):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    n_correct = 0
    n_total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(cfg.system.device), labels.to(cfg.system.device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        scheduler.step()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * n_correct / n_total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, cfg, desc="Evaluating", metrics=None):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(cfg.system.device), labels.to(cfg.system.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Allowing `metrics` to be nothing, a single call, or a dictionary of calls
    if metrics is None:
        metrics = {"accuracy": accuracy_score}
    elif callable(metrics):
        metrics = {"metric": metrics}
    elif not isinstance(metrics, dict):
        raise ValueError("metrics must be None, callable, or dict")

    epoch_loss = running_loss / len(loader)  # Mean loss
    epoch_metrics = {
        name: metric_fn(all_labels, all_predictions)
        for name, metric_fn in metrics.items()
    }

    return epoch_loss, epoch_metrics, all_predictions, all_labels


# Load -------------------------------------------------------------------------

# EMNIST images are rotated, flipping them for easier human readability
# https://stackoverflow.com/questions/48532761/letters-in-emnist-training-dataset-are-rotated-and-little-vague
transform = transforms.Compose(
    [
        lambda img: torchvision.transforms.functional.rotate(img, -90),
        lambda img: torchvision.transforms.functional.hflip(img),
        torchvision.transforms.ToTensor(),
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

# Plot a 3 x 3 grid of test images post-transform
if cfg.dev.mode == "dev":
    indices = np.random.choice(len(test_data), size=9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        image, label = test_data[idx]

        # Convert tensor to numpy
        img = image.squeeze().numpy()

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# Split ------------------------------------------------------------------------

# Split into training and validation sets
val_size = int(len(test_data) * cfg.train.val_frac)
train_size = len(train_data) - val_size

train_data, val_data = random_split(
    train_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(cfg.train.seed),
)
print_lvl(f"Train/Validate split: {len(train_data)}/{len(val_data)}")

train_loader = DataLoader(
    train_data,
    batch_size=cfg.train.batch_size,
    shuffle=True,
    num_workers=cfg.train.num_workers,
)
val_loader = DataLoader(
    val_data,
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

# TODO: Training
# model = timm.create_model(
#     cfg.backbone,
#     pretrained = True,
#     num_classes = 62,
#     in_chans=1
# )
# model = model.to(cfg.device)
