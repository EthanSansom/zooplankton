import json
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset, LCPNDataset, LCPNCollator
from cnn.hierarchy import Hierarchy
from cnn.label_map import LabelMap
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

from cnn.metrics import classification_metrics, hierarchical_metrics, print_metrics

# User settings ----------------------------------------------------------------

HIERARCHY_FILE = "taxonomic_2026_01_26.json"
LABEL_MAP_FILE = "nodes_2026_01_26.json"
CONFIG_FILE = "lcpn_2026_03_23.toml"
MODEL_NAME = "lcpn"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
LABEL_MAPS_DIR = BASE_DIR / "00_label_maps"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / HIERARCHY_FILE)
label_map = LabelMap(LABEL_MAPS_DIR / LABEL_MAP_FILE)

print("\nHiearchy:")
hierarchy.print_hierarchy()

print("\nConfig:")
print(cfg)

set_seed(cfg.train.seed)

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
    class_to_index=label_map.class_to_index,
    class_to_nmax=cfg.data.class_nmax,
)

print("\nImageDataset:")
print(image_dataset)

print("\nClasses:")
image_dataset.print_classes(label_map.index_to_label)

lcpn_dataset = LCPNDataset(
    base_dataset=image_dataset,
    hierarchy=hierarchy,
    node_index_to_name=label_map.index_to_label,
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
