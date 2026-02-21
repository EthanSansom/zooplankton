import torchvision
from pathlib import Path

RAW_DATA_DIR = Path("00_raw_data")

train_data = torchvision.datasets.EMNIST(
    root=RAW_DATA_DIR, split="byclass", train=True, download=True
)
test_data = torchvision.datasets.EMNIST(
    root=RAW_DATA_DIR, split="byclass", train=False, download=True
)
