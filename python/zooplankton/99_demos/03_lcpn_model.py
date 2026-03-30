from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset, LCPNCollator, LCPNDataset
from cnn.hierarchy import Hierarchy
from cnn.label_map import LabelMap
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

# User settings ----------------------------------------------------------------

CONFIG_FILE = "demo_lcpn.toml"
HIERARCHY_FILE = "demo_taxonomic.json"
LABEL_MAP_FILE = "demo_lcpn.json"
MODEL_NAME = "demo_lcpn"

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

set_seed(cfg.train.seed)

print("\nLabels:")
print(label_map)

print("\nHierarchy:")
hierarchy.print_hierarchy()

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
    class_to_index=label_map.class_to_index,
    class_to_nmax=cfg.data.class_nmax,
)

print("\nImageDataset:")
print(image_dataset)

train_data = LCPNDataset(image_dataset, hierarchy, label_map.index_to_label)
test_data = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=label_map.class_to_index,
)
test_data = LCPNDataset(test_data, hierarchy, label_map.index_to_label)

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

print("\nTraining model...")
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
