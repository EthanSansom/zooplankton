from cnn.config import Config
from cnn.hierarchy import Hierarchy
from cnn.models.flat import FlatModel
from cnn.models.lcpn import LCPNModel
from cnn.data import LCPNDataset, LCPNCollator
from cnn.metrics import (
    classification_metrics,
    hierarchical_metrics,
    flat_predictions_to_names,
    print_metrics,
)
from cnn.utils import set_seed, split

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

# User Settings ----------------------------------------------------------------

FLAT_CONFIG_FILE = "demo_flat.toml"
LCPN_CONFIG_FILE = "demo_lcpn.toml"
FLAT_MODEL_NAME = "demo_flat"
LCPN_MODEL_NAME = "demo_lcpn"
N_CLASSES = 62

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
DATA_DIR = BASE_DIR / "00_raw_data"
SAVE_DIR = BASE_DIR / "01_results"

flat_cfg = Config(BASE_DIR / "00_configs" / FLAT_CONFIG_FILE)
lcpn_cfg = Config(BASE_DIR / "00_configs" / LCPN_CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

assert flat_cfg.train.seed == lcpn_cfg.train.seed
assert flat_cfg.train.epochs == lcpn_cfg.train.epochs
assert flat_cfg.validate.fraction == lcpn_cfg.validate.fraction
assert flat_cfg.data.fraction == lcpn_cfg.data.fraction
assert flat_cfg.data.batch_size == lcpn_cfg.data.batch_size
assert flat_cfg.model.backbone == lcpn_cfg.model.backbone
assert flat_cfg.model.pretrained == lcpn_cfg.model.pretrained
assert flat_cfg.model.in_chans == lcpn_cfg.model.in_chans

set_seed(flat_cfg.train.seed)

# Class mappings ---------------------------------------------------------------

# fmt: off
node_index_to_name = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E",
    15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
    20: "K", 21: "L", 22: "M", 23: "N", 24: "O",
    25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
    30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z",
    36: "a", 37: "b", 38: "c", 39: "d", 40: "e",
    41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
    46: "k", 47: "l", 48: "m", 49: "n", 50: "o",
    51: "p", 52: "q", 53: "r", 54: "s", 55: "t",
    56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z",
}
# fmt: on

class_names = list(node_index_to_name.values())

# Data -------------------------------------------------------------------------

transform = transforms.Compose(
    [
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ]
)

full_train = datasets.EMNIST(
    root=DATA_DIR, split="byclass", train=True, download=True, transform=transform
)
full_test = datasets.EMNIST(
    root=DATA_DIR, split="byclass", train=False, download=True, transform=transform
)

if flat_cfg.data.fraction < 1:

    def subset(data, what, fraction=flat_cfg.data.fraction):
        n, n_subset = len(data), int(len(data) * fraction)
        print(f"Subset {what}: N = {n} -> {n_subset}")
        return Subset(data, np.random.choice(n, n_subset, replace=False))

    full_train = subset(full_train, "training data")
    full_test = subset(full_test, "test data")

# Train/valid split for flat ---------------------------------------------------

valid_size = int(len(full_train) * flat_cfg.validate.fraction)
train_size = len(full_train) - valid_size
flat_train, flat_valid = split(full_train, [train_size, valid_size], flat_cfg)

flat_train_loader = DataLoader(
    flat_train,
    batch_size=flat_cfg.data.batch_size,
    shuffle=True,
    num_workers=flat_cfg.data.num_workers,
)
flat_valid_loader = DataLoader(
    flat_valid,
    batch_size=flat_cfg.data.batch_size,
    shuffle=False,
    num_workers=flat_cfg.data.num_workers,
)
flat_test_loader = DataLoader(
    full_test,
    batch_size=flat_cfg.data.batch_size,
    shuffle=False,
    num_workers=flat_cfg.data.num_workers,
)

# Train/valid split for LCPN ---------------------------------------------------

lcpn_train = LCPNDataset(full_train, hierarchy, node_index_to_name)
lcpn_test = LCPNDataset(full_test, hierarchy, node_index_to_name)

valid_size = int(len(lcpn_train) * lcpn_cfg.validate.fraction)
train_size = len(lcpn_train) - valid_size
lcpn_train, lcpn_valid = split(lcpn_train, [train_size, valid_size], lcpn_cfg)

collator = LCPNCollator(hierarchy)

lcpn_train_loader = DataLoader(
    lcpn_train,
    batch_size=lcpn_cfg.data.batch_size,
    shuffle=True,
    num_workers=lcpn_cfg.data.num_workers,
    collate_fn=collator,
)
lcpn_valid_loader = DataLoader(
    lcpn_valid,
    batch_size=lcpn_cfg.data.batch_size,
    shuffle=False,
    num_workers=lcpn_cfg.data.num_workers,
    collate_fn=collator,
)
lcpn_test_loader = DataLoader(
    lcpn_test,
    batch_size=lcpn_cfg.data.batch_size,
    shuffle=False,
    num_workers=lcpn_cfg.data.num_workers,
    collate_fn=collator,
)

# Train flat model -------------------------------------------------------------

print("=" * 60)
print("Training FlatModel")
print("=" * 60)

flat_model = FlatModel(
    FLAT_MODEL_NAME, SAVE_DIR, n_classes=N_CLASSES, config=flat_cfg
).to(flat_cfg.metadata.device)
print(flat_model)

flat_model.fit(flat_train_loader, flat_valid_loader)

# Train LCPN model -------------------------------------------------------------

print("\n" + "=" * 60)
print("Training LCPNModel")
print("=" * 60)

lcpn_model = LCPNModel(
    LCPN_MODEL_NAME, SAVE_DIR, hierarchy=hierarchy, config=lcpn_cfg
).to(lcpn_cfg.metadata.device)
print(lcpn_model)

lcpn_model.fit(lcpn_train_loader, lcpn_valid_loader, collator)

# Test flat model --------------------------------------------------------------

print("\n" + "=" * 60)
print("Testing FlatModel")
print("=" * 60)

flat_metrics, flat_preds, flat_true = flat_model.test(flat_test_loader)

flat_pred_names = flat_predictions_to_names(flat_preds, node_index_to_name)
flat_true_names = flat_predictions_to_names(flat_true, node_index_to_name)
flat_cls_metrics = classification_metrics(
    flat_true_names, flat_pred_names, labels=class_names
)

print_metrics(flat_cls_metrics, header="\nFlat classification metrics:")

# Test LCPN model --------------------------------------------------------------

print("\n" + "=" * 60)
print("Testing LCPNModel")
print("=" * 60)

lcpn_metrics, lcpn_preds, lcpn_true = lcpn_model.test(lcpn_test_loader, collator)
lcpn_cls_metrics = classification_metrics(lcpn_true, lcpn_preds, labels=class_names)

print_metrics(lcpn_cls_metrics, header="\nLCPN flat classification metrics:")

# Hierarchical metrics require full paths — run predict_greedy over test set
print("\nCollecting paths for hierarchical metrics...")
lcpn_model.eval()
all_pred_paths, all_true_paths = [], []

with torch.no_grad():
    for inputs, labels in lcpn_test_loader:
        inputs = inputs.to(lcpn_cfg.metadata.device)
        _, pred_paths = lcpn_model.predict_greedy(inputs)
        true_paths = collator.uncollate_label_paths(labels)
        all_pred_paths.extend(pred_paths)
        all_true_paths.extend(true_paths)

lcpn_hier_metrics = hierarchical_metrics(all_true_paths, all_pred_paths)
print_metrics(lcpn_hier_metrics, header="\nLCPN hierarchical metrics:")

# Summary ----------------------------------------------------------------------

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"\n{'Metric':<25} {'Flat':>10} {'LCPN':>10}")
print("-" * 47)
print(
    f"  {'accuracy':<23} {flat_cls_metrics['accuracy']:>10.4f} {lcpn_cls_metrics['accuracy']:>10.4f}"
)
print(
    f"  {'precision (macro)':<23} {flat_cls_metrics['precision']:>10.4f} {lcpn_cls_metrics['precision']:>10.4f}"
)
print(
    f"  {'recall (macro)':<23} {flat_cls_metrics['recall']:>10.4f} {lcpn_cls_metrics['recall']:>10.4f}"
)
print(
    f"  {'f1 (macro)':<23} {flat_cls_metrics['f1']:>10.4f} {lcpn_cls_metrics['f1']:>10.4f}"
)
print(
    f"  {'hier_precision':<23} {'—':>10} {lcpn_hier_metrics['hier_precision']:>10.4f}"
)
print(f"  {'hier_recall':<23} {'—':>10} {lcpn_hier_metrics['hier_recall']:>10.4f}")
print(f"  {'hier_fscore':<23} {'—':>10} {lcpn_hier_metrics['hier_fscore']:>10.4f}")
