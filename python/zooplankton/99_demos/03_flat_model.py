from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.config import Config
from cnn.data import ImageDataset
from cnn.label_map import LabelMap
from cnn.models.flat import FlatModel
from cnn.utils import set_seed, split

# User settings ----------------------------------------------------------------

CONFIG_FILE = "demo_flat.toml"
LABEL_MAP_FILE = "demo_flat.json"
MODEL_NAME = "demo_flat"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
LABEL_MAPS_DIR = BASE_DIR / "00_label_maps"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
label_map = LabelMap(LABEL_MAPS_DIR / LABEL_MAP_FILE)

set_seed(cfg.train.seed)

print("\nLabel map:")
print(label_map)

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

# Split ------------------------------------------------------------------------

valid_size = int(len(image_dataset) * cfg.validate.fraction)
train_size = len(image_dataset) - valid_size
train_data, valid_data = split(image_dataset, [train_size, valid_size], cfg)

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

# Train ------------------------------------------------------------------------

model = FlatModel(MODEL_NAME, SAVE_DIR, n_classes=label_map.n_classes(), config=cfg).to(
    cfg.metadata.device
)

print("\nModel:")
print(model)

print("\nTraining model...")
history = model.fit(train_loader, valid_loader)

# Save -------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save(timestamp=False, overwrite=True)

# Load -------------------------------------------------------------------------

print("\nLoading model...")
loaded_model = FlatModel.load(save_dir).to(cfg.metadata.device)

print(f"Loaded: {loaded_model}")
print(f"Epochs completed: {loaded_model.history['epochs_completed']}")
print(f"Duration: {loaded_model.history['duration_seconds']:.1f}s")

# Verify weights loaded correctly
print("\nVerifying loaded model...")
loaded_metrics, _, _ = loaded_model.evaluate(valid_loader)
original_metrics = model.history["valid"][-1]

print(f"  Original final val accuracy: {original_metrics['accuracy']:.4f}")
print(f"  Loaded   final val accuracy: {loaded_metrics['accuracy']:.4f}")
print(
    f"  {'OK - weights match.' if abs(loaded_metrics['accuracy'] - original_metrics['accuracy']) < 1e-4 else 'WARNING - accuracy mismatch.'}"
)
