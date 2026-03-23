import json
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset
from cnn.models.flat import FlatModel
from cnn.utils import set_seed, split

from cnn.metrics import classification_metrics, flat_predictions_to_names, print_metrics

# User settings ----------------------------------------------------------------

CONFIG_FILE = "flat_2026_03_23.toml"
MODEL_NAME = "flat"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)

set_seed(cfg.train.seed)

# Class mappings ---------------------------------------------------------------

# Maps per-class directories under `zooplankton/01_raw_data` to a class index.
# Samples with the same index, from different directories, are merged into one
# class for training and testing purposes.
#
# Partially labelled classes, e.g. "cladocera" or "zooplankton", are excluded
# from the training and testing of the FlatModel.

# fmt: off
class_to_index = {
    # Parent node: "cladocera"
    "bosminidae": 0, "eubosmina": 0, # Merged into: "bosmina"
    "daphnia": 1,

    # Parent node: "yes_zooplankton"
    "rotifer": 2,
    "trichocerca": 2, "conochilus": 2, "kellicottia": 2, # Merged into: "rotifer"
    
    # Parent node: "copepoda"
    "nauplius_copepod": 3,
    "cyclopoid": 4,
    "harpacticoid": 5,
    "calanoid": 6,

    # Parent node: "not_zooplankton"
    "exoskeleton": 7,
    "fiber_hairlike": 8, "fiber_squiggly": 8, # Merged into: "fiber"
    "plant_matter": 9,
    "bubbles": 10,
}

index_to_class_name = {
    0: "bosmina", 1: "daphnia", 2: "rotifer", 3: "nauplius",
    4: "cyclopoid", 5: "harpacticoid", 6: "calanoid", 7: "exoskeleton",
    8: "fiber", 9: "plant_matter", 10: "bubbles",
}
# fmt: on

N_CLASSES = len(set(class_to_index.values()))

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
image_dataset.print_classes(index_to_class_name)

# Split ------------------------------------------------------------------------

sample_size = len(image_dataset)
test_size = int(sample_size * cfg.test.fraction)
valid_size = int(sample_size * cfg.validate.fraction)
train_size = sample_size - valid_size - test_size
train_data, valid_data, test_data = split(
    image_dataset, [train_size, valid_size, test_size], cfg
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
)
valid_loader = DataLoader(
    valid_data,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=cfg.data.num_workers,
)
test_loader = DataLoader(
    test_data,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=cfg.data.num_workers,
)

# Train ------------------------------------------------------------------------

model = FlatModel(MODEL_NAME, SAVE_DIR, n_classes=N_CLASSES, config=cfg).to(
    cfg.metadata.device
)

print("\nModel:")
print(model)

print("\nTraining model...")
history = model.fit(train_loader, valid_loader)

# Save -------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save(timestamp=True, overwrite=False)

# Test -------------------------------------------------------------------------

print("\nTesting model...")
model.eval()
test_metrics, preds, true = model.test(test_loader)
pred_names = flat_predictions_to_names(preds, index_to_class_name)
true_names = flat_predictions_to_names(true, index_to_class_name)

print("\nTest Metrics:")
print_metrics(test_metrics)
print_metrics(classification_metrics(true_names, pred_names))

# Save test results ------------------------------------------------------------

test_results = {
    "metrics": {k: v for k, v in test_metrics.items() if k != "confusion"},
    "classification_report": classification_metrics(true_names, pred_names)["report"],
    "predictions": pred_names,
    "true": true_names,
}

with open(save_dir / "test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print(f"Test results saved to {save_dir / 'test_results.json'}")
