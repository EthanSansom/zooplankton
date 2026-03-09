from cnn.hierarchy import Hierarchy
from cnn.data import LCPNDataset, LCPNCollator

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from pathlib import Path

# Setup ------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
HIERARCHIES_DIR = SCRIPT_DIR.parent / "00_hierarchies"
DATA_DIR = SCRIPT_DIR.parent / "00_raw_data"

hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

# Map EMNIST indices to class labels: 0-9 (digits), 10-35 (A-Z), 36-61 (a-z)
leaf_index_to_name = {
    # Digits 0-9
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    # Uppercase A-Z
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    # Lowercase a-z
    36: "a",
    37: "b",
    38: "c",
    39: "d",
    40: "e",
    41: "f",
    42: "g",
    43: "h",
    44: "i",
    45: "j",
    46: "k",
    47: "l",
    48: "m",
    49: "n",
    50: "o",
    51: "p",
    52: "q",
    53: "r",
    54: "s",
    55: "t",
    56: "u",
    57: "v",
    58: "w",
    59: "x",
    60: "y",
    61: "z",
}

# Demo -------------------------------------------------------------------------

# Flipping for human readability
transform = transforms.Compose(
    [
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
    ]
)

## Load Base Dataset
print("LOADING EMNIST DATA")

base_dataset = datasets.EMNIST(
    root=DATA_DIR, split="byclass", train=True, download=True, transform=transform
)

print(f"Base dataset size: {len(base_dataset)}")
print()

## Convert to hierarchical dataset
print("CREATING HIERARCHICAL LCPN DATASET")

hierarchical_dataset = LCPNDataset(
    base_dataset=base_dataset,
    hierarchy=hierarchy,
    leaf_index_to_name=leaf_index_to_name,
)

print(f"Hierarchical dataset size: {len(hierarchical_dataset)}")
print()

## Display hierarchical info for a single sample
print("SINGLE SAMPLE TEST")

image, labels = hierarchical_dataset[0]
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print("\nHierarchical labels:")
for node, label in labels.items():
    print(f"  {node:20s}: {label}")

base_image, base_label = base_dataset[0]
leaf_name = leaf_index_to_name[base_label]
print(f"\nLeaf class: '{leaf_name}' (index {base_label})")
print(f"Path: {' → '.join(hierarchy.get_path_to_root(leaf_name))}")
print()

# Plot the image
plt.figure(figsize=(4, 4))
plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"Sample Image: '{leaf_name}'\nPath: {' → '.join(hierarchy.get_path_to_root(leaf_name))}"
)
plt.axis("off")
plt.tight_layout()
plt.show()
print()

## Create dataloader with collator
print("DATALOADER TEST")

dataloader = DataLoader(
    hierarchical_dataset, batch_size=8, shuffle=True, collate_fn=LCPNCollator(hierarchy)
)

# Get one batch
images, batch_labels = next(iter(dataloader))

print(f"Batch images shape: {images.shape}")
print("\nBatch labels:")
for node_name, labels_tensor in batch_labels.items():
    print(f"  {node_name:20s}: {labels_tensor}")
    print(f"    {'Shape:':<20s} {labels_tensor.shape}")
    print(f"    {'Unique values:':<20s} {torch.unique(labels_tensor).tolist()}")
print()

## Validate the collator
print("COLLATOR VALIDATION")

expected_nodes = set(hierarchy.get_parent_nodes())
actual_nodes = set(batch_labels.keys())

print(f"Expected parent nodes: {len(expected_nodes)}")
print(f"Actual nodes in batch: {len(actual_nodes)}")
print(f"All nodes present: {expected_nodes == actual_nodes}")

if expected_nodes != actual_nodes:
    missing = expected_nodes - actual_nodes
    extra = actual_nodes - expected_nodes
    if missing:
        print(f"  Missing: {missing}")
    if extra:
        print(f"  Extra: {extra}")
else:
    print("Collator includes all parent nodes!")
print()

print("OFF-PATH NODES CHECK")

for node_name, labels_tensor in batch_labels.items():
    n_off_path = (labels_tensor == -1).sum().item()
    n_on_path = (labels_tensor != -1).sum().item()
    print(f"{node_name:20s}: {n_on_path} on-path, {n_off_path} off-path")
print()

## Show hierarchical paths for each image
print("BATCH SAMPLE PATHS")

# Show the first four images in the batch
for i in range(min(4, images.shape[0])):
    # Find which sample this is in the original dataset
    print(f"Sample {i}:")
    print("  Labels:")
    for node_name, labels_tensor in batch_labels.items():
        # The tensor has -1 sentinels for off-path nodes not visited on
        # the way from the root to a leaf (e.g. the image's base class).
        label_val = labels_tensor[i].item()
        if label_val != -1:
            children = hierarchy.parent_to_children[node_name]
            child_name = children[label_val]
            print(f"    {node_name:20s} -> {child_name} (index {label_val})")
        else:
            print(f"    {node_name:20s} -> (off-path)")
    print()
