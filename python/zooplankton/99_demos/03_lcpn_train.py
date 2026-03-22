from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from cnn.config import Config
from cnn.hierarchy import Hierarchy
from cnn.models.lcpn import LCPNModel
from cnn.data import ImageDataset, LCPNDataset, LCPNCollator
from cnn.utils import set_seed, split

# User Settings ----------------------------------------------------------------

CONFIG_FILE = "demo_lcpn.toml"
MODEL_NAME = "demo_lcpn"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

set_seed(cfg.train.seed)

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

# Data -------------------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((cfg.data.mean,), (cfg.data.std,)),
    ]
)

image_dataset = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=class_to_index,
    class_to_nmax=cfg.data.class_nmax,
)

print(image_dataset)

train_data = LCPNDataset(image_dataset, hierarchy, node_index_to_name)
test_data = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=class_to_index,
)
test_data = LCPNDataset(test_data, hierarchy, node_index_to_name)

# Split ------------------------------------------------------------------------

valid_size = int(len(train_data) * cfg.validate.fraction)
train_size = len(train_data) - valid_size
train_data, valid_data = split(train_data, [train_size, valid_size], cfg)

collator = LCPNCollator(hierarchy)

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

# Train ------------------------------------------------------------------------

model = LCPNModel(MODEL_NAME, SAVE_DIR, hierarchy=hierarchy, config=cfg).to(
    cfg.metadata.device
)

print("\nModel:")
print(model)

history = model.fit(train_loader, valid_loader, collator)

# Save -------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save(timestamp=False, overwrite=True)

# Load -------------------------------------------------------------------------

print("\nLoading model...")
loaded_model = LCPNModel.load(save_dir).to(cfg.metadata.device)

print(f"Loaded: {loaded_model}")
print(f"Epochs completed: {loaded_model.history['epochs_completed']}")
print(f"Duration: {loaded_model.history['duration_seconds']:.1f}s")

# Verify weights loaded correctly
print("\nVerifying loaded model...")
loaded_metrics, _, _ = loaded_model.evaluate(valid_loader, collator)
original_metrics = model.history["valid"][-1]

print(f"  Original final val accuracy: {original_metrics['accuracy']:.4f}")
print(f"  Loaded   final val accuracy: {loaded_metrics['accuracy']:.4f}")
print(
    f"  {'OK - weights match.' if abs(loaded_metrics['accuracy'] - original_metrics['accuracy']) < 1e-4 else 'WARNING - accuracy mismatch.'}"
)
