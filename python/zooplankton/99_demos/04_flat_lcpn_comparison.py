from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset, LCPNCollator, LCPNDataset
from cnn.hierarchy import Hierarchy
from cnn.metrics import (
    classification_metrics,
    flat_predictions_to_names,
    hierarchical_metrics,
    print_metrics,
)
from cnn.models.flat import FlatModel
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

# User settings ----------------------------------------------------------------

FLAT_CONFIG_FILE = "demo_flat.toml"
LCPN_CONFIG_FILE = "demo_lcpn.toml"
FLAT_MODEL_NAME = "demo_flat"
LCPN_MODEL_NAME = "demo_lcpn"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
SAVE_DIR = BASE_DIR / "01_results"

flat_cfg = Config(BASE_DIR / "00_configs" / FLAT_CONFIG_FILE)
lcpn_cfg = Config(BASE_DIR / "00_configs" / LCPN_CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

assert flat_cfg.train.seed == lcpn_cfg.train.seed
assert flat_cfg.train.epochs == lcpn_cfg.train.epochs
assert flat_cfg.validate.fraction == lcpn_cfg.validate.fraction
assert flat_cfg.data.batch_size == lcpn_cfg.data.batch_size
assert flat_cfg.model.backbone == lcpn_cfg.model.backbone
assert flat_cfg.model.pretrained == lcpn_cfg.model.pretrained
assert flat_cfg.model.in_chans == lcpn_cfg.model.in_chans

set_seed(flat_cfg.train.seed)

# Class mappings ---------------------------------------------------------------

# fmt: off
class_to_index = {
    "bosminidae":       0,  "eubosmina":        0,  "daphnia":          1,
    "rotifer":          2,  "trichocerca":      2,  "conochilus":       2,  "kellicottia":  2,
    "nauplius_copepod": 3,  "cyclopoid":        4,  "harpacticoid":     5,  "calanoid":     6,
    "exoskeleton":      7,  "fiber_hairlike":   8,  "fiber_squiggly":   8,  "plant_matter": 9,
    "cladocera":        10, "copepoda":         11,
}

node_index_to_name = {
    0: "bosmina",   1: "daphnia",      2: "rotifer",   3: "nauplius",
    4: "cyclopoid", 5: "harpacticoid", 6: "calanoid",  7: "exoskeleton",
    8: "fiber",     9: "plant_matter", 10: "cladocera", 11: "copepoda",
}
# fmt: on

N_CLASSES = len(set(class_to_index.values()))
class_names = list(node_index_to_name.values())

# Data -------------------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((flat_cfg.data.image_size, flat_cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((flat_cfg.data.mean,), (flat_cfg.data.std,)),
    ]
)

full_dataset = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=class_to_index,
    class_to_nmax=flat_cfg.data.class_nmax,
)

print(full_dataset)

# Train/valid/test split -------------------------------------------------------

test_size = int(len(full_dataset) * flat_cfg.data.test_fraction)
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = split(full_dataset, [train_size, test_size], flat_cfg)

valid_size = int(train_size * flat_cfg.validate.fraction)
train_size = train_size - valid_size
train_dataset, valid_dataset = split(train_dataset, [train_size, valid_size], flat_cfg)

print(f"\nSplit: train={train_size:,}  valid={valid_size:,}  test={test_size:,}")

# Flat loaders -----------------------------------------------------------------

flat_train_loader = DataLoader(
    train_dataset,
    batch_size=flat_cfg.data.batch_size,
    shuffle=True,
    num_workers=flat_cfg.data.num_workers,
)
flat_valid_loader = DataLoader(
    valid_dataset,
    batch_size=flat_cfg.data.batch_size,
    shuffle=False,
    num_workers=flat_cfg.data.num_workers,
)
flat_test_loader = DataLoader(
    test_dataset,
    batch_size=flat_cfg.data.batch_size,
    shuffle=False,
    num_workers=flat_cfg.data.num_workers,
)

# LCPN loaders -----------------------------------------------------------------

collator = LCPNCollator(hierarchy)

lcpn_train_loader = DataLoader(
    LCPNDataset(train_dataset, hierarchy, node_index_to_name),
    batch_size=lcpn_cfg.data.batch_size,
    shuffle=True,
    num_workers=lcpn_cfg.data.num_workers,
    collate_fn=collator,
)
lcpn_valid_loader = DataLoader(
    LCPNDataset(valid_dataset, hierarchy, node_index_to_name),
    batch_size=lcpn_cfg.data.batch_size,
    shuffle=False,
    num_workers=lcpn_cfg.data.num_workers,
    collate_fn=collator,
)
lcpn_test_loader = DataLoader(
    LCPNDataset(test_dataset, hierarchy, node_index_to_name),
    batch_size=lcpn_cfg.data.batch_size,
    shuffle=False,
    num_workers=lcpn_cfg.data.num_workers,
    collate_fn=collator,
)

# Train flat model -------------------------------------------------------------

print("Training FlatModel...")

flat_model = FlatModel(
    FLAT_MODEL_NAME, SAVE_DIR, n_classes=N_CLASSES, config=flat_cfg
).to(flat_cfg.metadata.device)
print(flat_model)

flat_model.fit(flat_train_loader, flat_valid_loader)

# Train LCPN model -------------------------------------------------------------

print("Training LCPNModel...")

lcpn_model = LCPNModel(
    LCPN_MODEL_NAME, SAVE_DIR, hierarchy=hierarchy, config=lcpn_cfg
).to(lcpn_cfg.metadata.device)
print(lcpn_model)

lcpn_model.fit(lcpn_train_loader, lcpn_valid_loader, collator)

# Test flat model --------------------------------------------------------------

print("Testing FlatModel...")

flat_metrics, flat_preds, flat_true = flat_model.test(flat_test_loader)

flat_pred_names = flat_predictions_to_names(flat_preds, node_index_to_name)
flat_true_names = flat_predictions_to_names(flat_true, node_index_to_name)
flat_cls_metrics = classification_metrics(
    flat_true_names, flat_pred_names, labels=class_names
)

print_metrics(flat_cls_metrics, header="\nFlat classification metrics:")

# Test LCPN model --------------------------------------------------------------

print("Testing LCPNModel...")

lcpn_metrics, lcpn_preds, lcpn_true = lcpn_model.test(lcpn_test_loader, collator)
lcpn_cls_metrics = classification_metrics(lcpn_true, lcpn_preds, labels=class_names)

print_metrics(lcpn_cls_metrics, header="\nLCPN flat classification metrics:")

# Hierarchical metrics ---------------------------------------------------------

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

print("\nSUMMARY")

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
