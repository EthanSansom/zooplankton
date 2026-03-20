from cnn.hierarchy import Hierarchy
from cnn.models.hierarchical import LCPNModel
from cnn.data import LCPNDataset, LCPNCollator

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# Setup ------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
HIERARCHIES_DIR = SCRIPT_DIR.parent / "00_hierarchies"
DATA_DIR = SCRIPT_DIR.parent / "00_raw_data"

# Demo -------------------------------------------------------------------------

print("LCPN MODEL TEST")

# 1. LOAD HIERARCHY
print("\n1. Loading hierarchy...")
hierarchy = Hierarchy(HIERARCHIES_DIR / "morphological.json")

print(f"   Parent nodes: {len(hierarchy.get_parent_nodes())}")
print(f"   Leaf nodes:   {len(hierarchy.get_leaf_nodes())}")
print(f"   Levels:       {hierarchy.max_level + 1}")  # +1 as 0-indexed

# 2. CREATE MODEL
print("\n2. Creating LCPN model...")
model = LCPNModel(hierarchy=hierarchy, backbone="resnet18", pretrained=True, in_chans=1)

print(model)

# Parameter counts
params = model.get_num_parameters()
print(f"\n   Backbone parameters: {params['backbone']:,}")
print(f"   Heads parameters:    {params['heads']:,}")
print(f"   Total parameters:    {params['total']:,}")

# 3. TEST FORWARD PASS (RANDOM INPUT)
print("\n3. Testing forward pass with random input...")
batch_size = 4
random_images = torch.randn(batch_size, 1, 28, 28)

outputs = model(random_images)

print(f"   Input shape: {random_images.shape}")
print(f"   Number of output heads: {len(outputs)}")
print("\n   Output shapes:")
for node_name, logits in outputs.items():
    print(f"      {node_name:20s}: {logits.shape}")

# 4. TEST GREEDY PREDICTION (RANDOM INPUT)
print("\n4. Testing greedy prediction...")
predictions, paths = model.predict_greedy(random_images)

print(f"   Number of predictions: {len(predictions)}")
print("\n   Predictions:")
for i, (pred, path) in enumerate(zip(predictions, paths)):
    print(f"      Image {i}: {pred:5s}  (path: {' -> '.join(path)})")

print("\n5. Testing probability prediction...")
probs_structured = model.predict_probabilities(random_images)

print(f"   Number of parent nodes: {len(probs_structured['parents'])}")
print(f"   Number of leaf nodes: {len(probs_structured['leaves'])}")

print("\n   Parent node probabilities (first sample):")
for parent in list(probs_structured["parents"].keys()):
    print(f"      {parent:20s}: {probs_structured['parents'][parent][0]}")

print("\n   Leaf node probabilities (top 5 for first sample):")
leaf_probs = [
    (leaf, prob[0].item()) for leaf, prob in probs_structured["leaves"].items()
]
leaf_probs.sort(key=lambda x: x[1], reverse=True)
for leaf, prob in leaf_probs[:5]:
    print(f"      {leaf:5s}: {prob:.6f}")

# Verify probabilities sum to 1
total_leaf_prob = sum(prob[0].item() for prob in probs_structured["leaves"].values())
print(f"\n   Sum of all leaf probabilities: {total_leaf_prob:.6f}")
print(
    f"   {'Valid probability distribution!' if abs(total_leaf_prob - 1.0) < 1e-5 else 'ERROR: Does not sum to 1!'}"
)

# 6. TEST WITH REAL EMNIST DATA
print("\n6. Testing with real EMNIST data...")

# Create mapping
leaf_index_to_name = {
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

# Load EMNIST
transform = transforms.Compose([transforms.ToTensor()])

base_dataset = datasets.EMNIST(
    root=DATA_DIR, split="byclass", train=True, download=True, transform=transform
)

# Wrap with hierarchical labels
hierarchical_dataset = LCPNDataset(
    base_dataset=base_dataset,
    hierarchy=hierarchy,
    leaf_index_to_name=leaf_index_to_name,
)

# Create dataloader
collator = LCPNCollator(hierarchy)
dataloader = DataLoader(
    hierarchical_dataset, batch_size=8, shuffle=True, collate_fn=collator
)

# Get one batch
images, labels = next(iter(dataloader))

print(f"   Loaded batch: {images.shape}")

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(images)

print("   Forward pass successful!")
print("\n   Output shapes:")
for node_name, logits in list(outputs.items()):
    print(f"      {node_name:20s}: {logits.shape}")

# Greedy predictions on real data
predictions, paths = model.predict_greedy(images)

print("\n   Greedy predictions vs ground truth (first 4 samples):")
for i in range(min(4, len(predictions))):
    # Unbatch to get true path
    true_path = collator.unbatch(labels, i)
    true_leaf = true_path[-1]

    # Compare
    match = "CORRECT" if predictions[i] == true_leaf else "WRONG"
    print(f"      {match} Pred: {predictions[i]:5s} | True: {true_leaf:5s}")
    print(f"         True path: {' -> '.join(true_path)}")
    print(f"         Pred path: {' -> '.join(paths[i])}")

# Probability predictions on real data
probs_structured = model.predict_probabilities(images)

print("\n   Probability predictions (first sample):")
print("      Top 5 leaf probabilities:")
leaf_probs = [
    (leaf, prob[0].item()) for leaf, prob in probs_structured["leaves"].items()
]
leaf_probs.sort(key=lambda x: x[1], reverse=True)
for leaf, prob in leaf_probs[:5]:
    print(f"         {leaf:5s}: {prob:.6f}")
