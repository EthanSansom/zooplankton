from cnn.config import Config
from cnn.models.flat import FlatModel
from cnn.utils import set_seed, split

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

# User Settings ----------------------------------------------------------------

CONFIG_FILE = "demo_flat.toml"
MODEL_NAME = "demo_flat"
N_CLASSES = 62

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)

set_seed(cfg.train.seed)

# Data -------------------------------------------------------------------------

transform = transforms.Compose(
    [
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ]
)

train_data = datasets.EMNIST(
    root=DATA_DIR,
    split="byclass",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.EMNIST(
    root=DATA_DIR,
    split="byclass",
    train=False,
    download=True,
    transform=transform,
)

if cfg.data.fraction < 1:

    def subset(data, what, fraction=cfg.data.fraction):
        n, n_subset = len(data), int(len(data) * fraction)
        print(f"Subset {what}: N = {n} -> {n_subset}")
        return Subset(data, np.random.choice(n, n_subset, replace=False))

    train_data = subset(train_data, "training data")
    test_data = subset(test_data, "testing data")

# Split ------------------------------------------------------------------------

valid_size = int(len(train_data) * cfg.validate.fraction)
train_size = len(train_data) - valid_size
train_data, valid_data = split(train_data, [train_size, valid_size], cfg)

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

model = FlatModel(MODEL_NAME, SAVE_DIR, n_classes=N_CLASSES, config=cfg).to(
    cfg.metadata.device
)

print("\nModel:")
print(model)

history = model.fit(train_loader, valid_loader)

print("\nModel history:")
print(history)

# Save -------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save()

# Load -------------------------------------------------------------------------

print("\nLoading model...")
loaded_model = FlatModel.load(save_dir)
loaded_model = loaded_model.to(cfg.metadata.device)

print(f"Loaded: {loaded_model}")
print(f"Epochs completed: {loaded_model.history['epochs_completed']}")
print(f"Duration: {loaded_model.history['duration_seconds']:.1f}s")
print(f"Backbone: {loaded_model.model_metadata['backbone']}")

# Verify weights loaded correctly by comparing validation accuracy
print("\nVerifying loaded model...")
loaded_metrics, _, _ = loaded_model.evaluate(valid_loader)
original_metrics = model.history["valid"][-1]

print(f"  Original final val accuracy: {original_metrics['accuracy']:.4f}")
print(f"  Loaded   final val accuracy: {loaded_metrics['accuracy']:.4f}")
print(
    f"  {'OK - weights match.' if abs(loaded_metrics['accuracy'] - original_metrics['accuracy']) < 1e-4 else 'WARNING - accuracy mismatch, weights may not have loaded correctly.'}"
)
