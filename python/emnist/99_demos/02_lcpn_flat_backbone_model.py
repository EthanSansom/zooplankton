from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cnn.config import Config
from cnn.data import LCPNCollator, LCPNDataset
from cnn.hierarchy import Hierarchy
from cnn.models.flat import FlatModel
from cnn.models.lcpn import LCPNModel
from cnn.utils import set_seed, split

# User settings ----------------------------------------------------------------

CONFIG_FILE = "demo_lcpn_flat_backbone.toml"
MODEL_NAME = "demo_lcpn_flat_backbone"

# Configuration ----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"
DATA_DIR = BASE_DIR / "00_raw_data"
SAVE_DIR = BASE_DIR / "01_results"

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")
FLAT_MODEL_DIR = Path(cfg.model.backbone_model)

if not FLAT_MODEL_DIR.exists():
    raise ValueError(
        f"Directory {FLAT_MODEL_DIR} doesn't exist. "
        "Run '99_demos/01_flat_train.py' to train a `FlatModel` backend."
    )

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

# Verify flat model weights ----------------------------------------------------

print("\nVerifying flat model weights...")
flat_model = FlatModel.load(FLAT_MODEL_DIR).to(cfg.metadata.device)
flat_backbone_state = {
    k: v for k, v in flat_model.state_dict().items() if k.startswith("backbone.")
}

# Model ------------------------------------------------------------------------

model = LCPNModel(MODEL_NAME, SAVE_DIR, hierarchy=hierarchy, config=cfg).to(
    cfg.metadata.device
)

print("\nModel:")
print(model)

# Verify pre-trained weights loaded correctly ----------------------------------

print("\nVerifying backbone weights match flat model...")
lcpn_backbone_state = {
    k: v for k, v in model.state_dict().items() if k.startswith("backbone.")
}

all_match = all(
    torch.allclose(flat_backbone_state[k], lcpn_backbone_state[k])
    for k in lcpn_backbone_state
)
print(
    f"  {'OK - backbone weights match flat model.' if all_match else 'WARNING - backbone weights do not match flat model.'}"
)

# Verify backbone is frozen ----------------------------------------------------

print("\nVerifying backbone is frozen...")
assert model.backbone_is_frozen(), "ERROR - backbone is not frozen."
print("  OK - backbone is frozen.")

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.backbone.parameters())
n_heads = sum(p.numel() for p in model.heads.parameters())
print(f"  Frozen parameters:   {n_frozen:,} (backbone)")
print(f"  Trainable parameters: {n_trainable:,} (heads only, expected {n_heads:,})")
assert n_trainable == n_heads, "ERROR - trainable parameters include backbone weights."
print("  OK - only head parameters are trainable.")

# Train ------------------------------------------------------------------------

history = model.fit(train_loader, valid_loader, collator)

# Verify backbone weights unchanged after training -----------------------------

print("\nVerifying backbone weights unchanged after training...")
lcpn_backbone_state_after = {
    k: v for k, v in model.state_dict().items() if k.startswith("backbone.")
}
still_match = all(
    torch.allclose(flat_backbone_state[k], lcpn_backbone_state_after[k])
    for k in lcpn_backbone_state_after
)
print(
    f"  {'OK - backbone weights unchanged by training.' if still_match else 'WARNING - backbone weights changed during training, freeze may not have worked.'}"
)

# Save -------------------------------------------------------------------------

print("\nSaving model...")
save_dir = model.save(timestamp=False, overwrite=True)

# Load -------------------------------------------------------------------------

print("\nLoading model...")
loaded_model = LCPNModel.load(save_dir)
loaded_model = loaded_model.to(cfg.metadata.device)

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
    f"  {'OK - weights match.' if abs(loaded_metrics['accuracy'] - original_metrics['accuracy']) < 1e-4 else 'WARNING - accuracy mismatch, weights may not have loaded correctly.'}"
)
