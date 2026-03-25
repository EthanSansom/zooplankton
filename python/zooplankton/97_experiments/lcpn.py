import json
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset, LCPNDataset, LCPNCollator
from cnn.hierarchy import Hierarchy
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

from cnn.metrics import classification_metrics, hierarchical_metrics, print_metrics

# User settings ----------------------------------------------------------------

HIERARCHY_FILE = "taxonomic_extended_2026_01_26.json"  # "taxonomic_2026_01_26.json"
CONFIG_FILE = "lcpn_2026_03_23.toml"
MODEL_NAME = "lcpn"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / HIERARCHY_FILE)

print("\nHiearchy:")
hierarchy.print_hierarchy()

set_seed(cfg.train.seed)

print("\nConfig:")
print(cfg)

# Class mappings ---------------------------------------------------------------

# Maps per-class directories under `zooplankton/00_raw_data` to a node index.
# Samples with the same index, from different directories, are merged into one
# class for training and testing purposes.
#
# Partially labelled classes (cladocera, copepoda) are included with their
# own index. These classes correspond to a parent node on the hierarchy, while
# fully labelled classes correspond to a leaf node.

# fmt: off
class_to_index = {
    # Parent node: "yes_zooplankton"
    "cladocera": 0,  # Partially labelled images

    # Parent node: "cladocera"
    "bosminidae": 1, "eubosmina": 1,  # Merged into: "bosmina"
    "daphnia": 2,

    # Parent node: "yes_zooplankton"
    "rotifer": 3,
    "trichocerca": 3, "conochilus": 3, "kellicottia": 3,  # Merged into: "rotifer"

    # Parent node: "yes_zooplankton"
    "copepoda": 4,  # Partially labelled images

    # Parent node: "copepoda"
    "nauplius_copepod": 5,
    "cyclopoid": 6,
    "harpacticoid": 7,
    "calanoid": 8,

    # Parent node: "not_zooplankton"
    "exoskeleton": 9,
    "fiber_hairlike": 10, "fiber_squiggly": 10,  # Merged into: "fiber"
    "plant_matter": 11,
    "bubbles": 12,
}

node_index_to_name = {
    # Internal nodes (parent: yes_zooplankton)
    0: "cladocera", 4: "copepoda",

    # Leaf nodes (parent: cladocera)
    1: "bosmina", 2: "daphnia",

    # Leaf nodes (parent: yes_zooplankton)
    3: "rotifer",

    # Leaf nodes (parent: copepoda)
    5: "nauplius", 6: "cyclopoid", 7: "harpacticoid", 8: "calanoid",

    # Leaf nodes (parent: not_zooplankton)
    9: "exoskeleton", 10: "fiber", 11: "plant_matter", 12: "bubbles",
}
# fmt: on

# Data -------------------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
    ]
)

image_dataset = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=class_to_index,
    class_to_nmax=cfg.data.class_nmax,
)

print("\nImageDataset:")
print(image_dataset)

print("\nClasses:")
image_dataset.print_classes(node_index_to_name)

lcpn_dataset = LCPNDataset(
    base_dataset=image_dataset,
    hierarchy=hierarchy,
    node_index_to_name=node_index_to_name,
)

collator = LCPNCollator(hierarchy)

# Split ------------------------------------------------------------------------

sample_size = len(lcpn_dataset)
test_size = int(sample_size * cfg.test.fraction)
valid_size = int(sample_size * cfg.validate.fraction)
train_size = sample_size - valid_size - test_size
train_data, valid_data, test_data = split(
    lcpn_dataset, [train_size, valid_size, test_size], cfg
)

print("\nTrain/Validation/Test Split:")
print(f"  train_data: N = {train_size:,} ({(train_size / sample_size):.2f}%)")
print(f"  valid_data: N = {valid_size:,} ({(valid_size / sample_size):.2f}%)")
print(f"  test_data:  N = {test_size:,} ({(test_size / sample_size):.2f}%)")

train_loader = DataLoader(
    train_data,
    batch_size=cfg.data.batch_size,
    shuffle=True,
    num_workers=cfg.data.num_workers,
    collate_fn=collator,
)
valid_loader = DataLoader(
    valid_data,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=cfg.data.num_workers,
    collate_fn=collator,
)
test_loader = DataLoader(
    test_data,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=cfg.data.num_workers,
    collate_fn=collator,
)

# Train ------------------------------------------------------------------------

model = LCPNModel(
    MODEL_NAME,
    SAVE_DIR,
    hierarchy=hierarchy,
    config=cfg,
).to(cfg.metadata.device)

print("\nModel:")
print(model)

print("\nTraining model...")
history = model.fit(train_loader, valid_loader, collator)

# Save -------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save(timestamp=True, overwrite=False)

# Test -------------------------------------------------------------------------

print("\nTesting model...")
model.eval()
test_metrics, preds, true, pred_paths, true_paths = model.test(test_loader, collator)

print("\nTest Metrics:")
print_metrics(test_metrics)
print_metrics(classification_metrics(true, preds))
print_metrics(hierarchical_metrics(true_paths, pred_paths))

# Save test results ------------------------------------------------------------

test_results = {
    "metrics": {k: v for k, v in test_metrics.items() if k != "confusion"},
    "classification_report": classification_metrics(true, preds)["report"],
    "predictions": preds,
    "true": true,
    "pred_paths": pred_paths,
    "true_paths": true_paths,
}

with open(save_dir / "test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print(f"Test results saved to {save_dir / 'test_results.json'}")
