from cnn.config import Config
from cnn.training import fit_model
from cnn.utils import set_seed, split

# TODO: update() method for the Config class, can only update existing keys
# TODO: Create a `loader.py` script under /cnn
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torchvision import datasets, transforms
import timm
import numpy as np

from pathlib import Path

# Setup ------------------------------------------------------------------------

CONFIG_FILE = "flat.toml"
BASE_DIR = Path(__file__).parent.resolve()
N_CLASSES = 62

cfg = Config(BASE_DIR / "00_configs" / CONFIG_FILE)
set_seed(cfg.train.seed)

# Load -------------------------------------------------------------------------

# EMNIST images are rotated, flipping them for easier human readability
# https://stackoverflow.com/questions/48532761/letters-in-emnist-training-dataset-are-rotated-and-little-vague
transform = transforms.Compose(
    [
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
        # Known EMNIST mean and SD, see:
        # https://github.com/Tony-Y/pytorch_warmup/blob/master/examples/emnist/main.py
        transforms.Normalize((0.1751,), (0.3332,)),
    ]
)

train_data = datasets.EMNIST(
    root=BASE_DIR / "00_raw_data",
    split="byclass",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.EMNIST(
    root=BASE_DIR / "00_raw_data",
    split="byclass",
    train=False,
    download=True,
    transform=transform,
)

# Subset the images for faster training
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
test_loader = DataLoader(
    test_data,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=cfg.data.num_workers,
)

# Train ------------------------------------------------------------------------

model = timm.create_model(
    cfg.model.backbone,
    pretrained=True,
    num_classes=N_CLASSES,
    in_chans=1,  # EMNIST images are greyscale (1 channel), not RGB (3 channels)
).to(cfg.metadata.device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.optimizer.learning_rate,
    weight_decay=0,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.train.epochs, eta_min=cfg.scheduler.learning_rate_min
)

fit_model(model, optimizer, criterion, scheduler, cfg, train_loader, valid_loader)

cfg.save(BASE_DIR / "01_results" / CONFIG_FILE)
