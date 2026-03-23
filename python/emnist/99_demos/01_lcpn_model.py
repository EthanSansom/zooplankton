from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cnn.config import Config
from cnn.data import LCPNCollator, LCPNDataset
from cnn.hierarchy import Hierarchy
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

# User settings ----------------------------------------------------------------

CONFIG_FILE = "demo_lcpn.toml"
MODEL_NAME = "demo_lcpn"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
DATA_DIR = BASE_DIR / "00_raw_data"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

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

# fmt: off
node_index_to_name = {
     0: "0",  1: "1",  2: "2",  3: "3",  4: "4",  5: "5",  6: "6",  7: "7",  8: "8",  9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
    20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
    30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z",
    36: "a", 37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
    46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s", 55: "t",
    56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z",
}
# fmt: on

train_data = LCPNDataset(train_data, hierarchy, node_index_to_name)
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
print("\nModel history:")
print(history)

# Save ------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save(timestamp=False, overwrite=True)

# Load ------------------------------------------------------------------------

print("\nLoading model...")
loaded_model = LCPNModel.load(save_dir)
loaded_model = loaded_model.to(cfg.metadata.device)

print(f"Loaded: {loaded_model}")
print(f"Epochs completed: {loaded_model.history['epochs_completed']}")
print(f"Duration: {loaded_model.history['duration_seconds']:.1f}s")
print(f"Backbone: {loaded_model.model_metadata['backbone']}")

# Verify weights loaded correctly by comparing validation accuracy
print("\nVerifying loaded model...")
loaded_metrics, _, _ = loaded_model.evaluate(valid_loader, collator)
original_metrics = model.history["valid"][-1]

print(f"  Original final val accuracy: {original_metrics['accuracy']:.4f}")
print(f"  Loaded   final val accuracy: {loaded_metrics['accuracy']:.4f}")
print(
    f"  {'OK - weights match.' if abs(loaded_metrics['accuracy'] - original_metrics['accuracy']) < 1e-4 else 'WARNING - accuracy mismatch, weights may not have loaded correctly.'}"
)
