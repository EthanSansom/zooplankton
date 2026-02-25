import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from types import SimpleNamespace
from pathlib import Path
import random
import os

# NOTE: Based on:
# https://www.leoniemonigatti.com/blog/pytorch-image-classification.html

# Constants --------------------------------------------------------------------

EMNIST_N_CLASSES = 62

# Config -----------------------------------------------------------------------

cfg = SimpleNamespace(**{})

# Misc
cfg.mode = "dev"  # dev | train
cfg.seed = 123
cfg.raw_data_dir = Path("00_raw_data")

# Torch/CNN Settings
cfg.batch_size = 64
cfg.num_workers = 2
cfg.val_fraction = 0.2  # Fraction of train using for validation
cfg.backbone = "resnet18"

# Development settings
cfg.dev_subset = 0.20  # Fraction of images to subset in dev (faster training)
cfg.print_level = 1  # Minimum level of message printed by `print_lvl()`

cfg.device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
    """Print if level >= cfg.print_level"""
    if level >= cfg.print_level:
        print(*args, **kwargs)


# Print configuration
print_lvl(f"Using device: {cfg.device}")


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
set_seed(cfg.seed)

# Helpers ----------------------------------------------------------------------

# TODO We'll want to look into moving functions into a "package" of some sort
# - Look into python-y way to do this
# - Eventually we'll want to add better function documentation (e.g. docstring)

# def train_epoch(model, loader, criterion, optimizer, scheduler, cfg):
#     """Train for one epoch"""
#     model.train()
#     running_loss = 0.0
#     n_correct = 0
#     n_total = 0

#     for inputs, labels in tqdm(loader, desc="Training"):
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         with torch.set_grad_enabled(True):
#             predictions = model(inputs)

#             loss = criterion(predictions, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         # Update the learning rate using the learning-rate scheduler
#         scheduler.step()

#     epoch_loss = running_loss / len(loader)
#     epoch_acc = 100 * correct / total

#     return epoch_loss, epoch_acc


# Load -------------------------------------------------------------------------

# EMNIST images are rotated
# https://stackoverflow.com/questions/48532761/letters-in-emnist-training-dataset-are-rotated-and-little-vague
transform = transforms.Compose(
    [
        lambda img: torchvision.transforms.functional.rotate(img, -90),
        lambda img: torchvision.transforms.functional.hflip(img),
        torchvision.transforms.ToTensor(),
    ]
)

train_data = datasets.EMNIST(
    root=cfg.raw_data_dir,
    split="byclass",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.EMNIST(
    root=cfg.raw_data_dir,
    split="byclass",
    train=False,
    download=True,
    transform=transform,
)

print_lvl(f"Train samples: {len(train_data)}")
print_lvl(f"Test samples: {len(test_data)}")
print_lvl(f"Classes: {len(train_data.classes)}")

# Subset the images for faster training
if cfg.mode == "dev":
    n_train = len(train_data)
    n_train_subset = int(len(train_data) * cfg.dev_subset)
    train_indices = np.random.choice(n_train, n_train_subset, replace=False)
    train_data = Subset(train_data, train_indices)
    print_lvl(f"Subset train data: N = {n_train} -> {n_train_subset}")

    n_test = len(test_data)
    n_test_subset = int(n_test * cfg.dev_subset)
    test_indices = np.random.choice(n_test, n_test_subset, replace=False)
    test_data = Subset(test_data, test_indices)
    print_lvl(f"Subset test data: N = {n_test} -> {n_test_subset}")

# Plot a 3 x 3 grid of test images post-transform
if cfg.mode == "dev":
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
val_size = int(len(test_data) * cfg.val_fraction)
train_size = len(train_data) - val_size

train_data, val_data = random_split(
    train_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(cfg.seed),
)
print_lvl(f"Train/Validate split: {len(train_data)}/{len(val_data)}")

train_loader = DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
)
val_loader = DataLoader(
    val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
)
test_loader = DataLoader(
    test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
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
