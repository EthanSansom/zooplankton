from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.data import ImageDataset, LCPNCollator, LCPNDataset
from cnn.hierarchy import Hierarchy

# Setup ------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "00_raw_data"
HIERARCHIES_DIR = BASE_DIR / "00_hierarchies"

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

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# ImageDataset -----------------------------------------------------------------

print("\nIMAGEDATASET")

image_dataset = ImageDataset(
    root=DATA_DIR,
    transform=transform,
    class_to_index=class_to_index,
)

print(f"\n{image_dataset}")

# Verify only requested classes are loaded
loaded_classes = set(image_dataset.class_to_index.keys())
expected_classes = set(class_to_index.keys())
assert loaded_classes == expected_classes, (
    f"Loaded classes don't match expected.\n"
    f"  Extra:   {loaded_classes - expected_classes}\n"
    f"  Missing: {expected_classes - loaded_classes}"
)
print(f"\nOK - only requested classes loaded ({len(loaded_classes)} directories)")

# Verify unwanted directories are excluded
all_dirs = {p.name for p in DATA_DIR.iterdir() if p.is_dir()}
excluded = all_dirs - expected_classes
print(f"OK - excluded {len(excluded)} directories: {sorted(excluded)}")

# Class distribution
print("\nClass distribution:")
counts = Counter(image_dataset.labels)
print(f"  {'Index':<8} {'Name':<20} {'Directories':<35} {'Count':>8}")
print(f"  {'-' * 75}")
for idx, name in node_index_to_name.items():
    dirs = [d for d, i in class_to_index.items() if i == idx]
    count = counts.get(idx, 0)
    print(f"  {idx:<8} {name:<20} {', '.join(dirs):<35} {count:>8,}")

# Verify grouped classes share the same index
assert class_to_index["bosminidae"] == class_to_index["eubosmina"], (
    "bosminidae and eubosmina should share index"
)
assert class_to_index["fiber_hairlike"] == class_to_index["fiber_squiggly"], (
    "fiber_hairlike and fiber_squiggly should share index"
)
print("\nOK - grouped classes share correct indices")

# Hierarchy --------------------------------------------------------------------

print("\nHIERARCHY")

hierarchy = Hierarchy(HIERARCHIES_DIR / "demo_morphological.json")
hierarchy.print_hierarchy()

print(f"\n  Nodes:   {len(hierarchy.nodes)}")
print(f"  Leaves:  {len(hierarchy.leaves)}")
print(f"  Parents: {len(hierarchy.parents)}")
print(f"  Levels:  {hierarchy.max_level + 1}")

# Verify all node_index_to_name values exist in hierarchy
print("\nVerifying node_index_to_name against hierarchy...")
for idx, name in node_index_to_name.items():
    assert name in hierarchy.nodes, (
        f"Node '{name}' (index {idx}) not found in hierarchy."
    )
print(f"OK - all {len(node_index_to_name)} nodes found in hierarchy")

# Identify which are leaves and which are internal
leaf_indices = {
    idx for idx, name in node_index_to_name.items() if hierarchy.node_is_leaf(name)
}
partial_indices = {
    idx for idx, name in node_index_to_name.items() if hierarchy.node_is_parent(name)
}
print(f"\nFully labelled (leaf) indices:     {sorted(leaf_indices)}")
print(f"Partially labelled (parent) indices: {sorted(partial_indices)}")

# LCPNDataset ------------------------------------------------------------------

print("\nLCPNDATASET")

lcpn_dataset = LCPNDataset(
    base_dataset=image_dataset,
    hierarchy=hierarchy,
    node_index_to_name=node_index_to_name,
)

print(f"\nDataset size: {len(lcpn_dataset)}")

# Inspect a fully labelled sample (e.g. calanoid -> index 6 -> leaf)
print("\nInspecting fully labelled sample (calanoid)...")
calanoid_indices = [i for (i, label) in enumerate(image_dataset.labels) if label == 6]
assert calanoid_indices, "No calanoid samples found"
image, labels = lcpn_dataset[calanoid_indices[0]]

print(f"  Image shape: {image.shape}")
print(f"  Labels dict: {labels}")
expected_path = hierarchy.get_path_to_root("calanoid")
print(f"  Expected path: {expected_path}")
for i in range(len(expected_path) - 1):
    parent = expected_path[i]
    child = expected_path[i + 1]
    expected_idx = hierarchy.get_child_index(parent, child)
    assert labels[parent] == expected_idx, (
        f"Label mismatch at '{parent}': expected {expected_idx}, got {labels[parent]}"
    )
print("  OK - all labels on path are correct")

# Inspect a partially labelled sample (e.g. copepoda -> index 11 -> internal node)
print("\nInspecting partially labelled sample (copepoda)...")
copepoda_indices = [i for (i, label) in enumerate(image_dataset.labels) if label == 11]
assert copepoda_indices, "No copepoda samples found"
image, labels = lcpn_dataset[copepoda_indices[0]]

print(f"  Labels dict: {labels}")
expected_path = hierarchy.get_path_to_root("copepoda")
print(f"  Expected path (truncated): {expected_path}")
assert "copepoda" not in labels, (
    "Partially labelled sample should not have labels below its node"
)
for i in range(len(expected_path) - 1):
    parent = expected_path[i]
    child = expected_path[i + 1]
    expected_idx = hierarchy.get_child_index(parent, child)
    assert labels[parent] == expected_idx, (
        f"Label mismatch at '{parent}': expected {expected_idx}, got {labels[parent]}"
    )
print("  OK - labels terminate at copepoda, nothing below")

# LCPNCollator -----------------------------------------------------------------

print("\nLCPNCOLLATOR")

collator = LCPNCollator(hierarchy)
loader = DataLoader(lcpn_dataset, batch_size=16, shuffle=True, collate_fn=collator)
images, batch_labels = next(iter(loader))

print(f"\nBatch image shape: {images.shape}")
print(f"Batch label keys:  {list(batch_labels.keys())}")
print(f"Expected keys:     {hierarchy.get_parent_nodes()}")
assert set(batch_labels.keys()) == set(hierarchy.get_parent_nodes()), (
    "Batch label keys don't match hierarchy parent nodes"
)
print("OK - batch label keys match hierarchy parent nodes")

for node, tensor in batch_labels.items():
    assert tensor.shape[0] == 16, (
        f"Expected batch size 16 for node '{node}', got {tensor.shape[0]}"
    )
print("OK - all label tensors have correct batch size (16)")

# Verify sentinel masking
print("\nSentinel (-1) distribution per node:")
for node, tensor in batch_labels.items():
    n_active = (tensor != -1).sum().item()
    n_masked = (tensor == -1).sum().item()
    print(f"  {node:<20} active={n_active:>3}  masked={n_masked:>3}")

# uncollate() round-trip -------------------------------------------------------

print("\nCOLLATE -> UNCOLLATE CONVERSION")

paths = collator.uncollate_label_paths(batch_labels)
leaves, is_leaf = collator.uncollate_label_leaves(batch_labels)

print("\nSample paths and leaf status (first 8):")
for i in range(min(8, len(paths))):
    status = "leaf" if is_leaf[i] else "partial"
    print(f"  [{i}] {' -> '.join(paths[i]):<50} ({status})")

n_leaf = sum(is_leaf)
n_partial = len(is_leaf) - n_leaf
print(f"\nIn this batch: {n_leaf} fully labelled, {n_partial} partially labelled")

# Verify partial samples have internal nodes as their terminal label
for i, (path, leaf) in enumerate(zip(paths, is_leaf)):
    terminal = path[-1]
    if not leaf:
        assert hierarchy.node_is_parent(terminal), (
            f"Sample {i} is marked partial but terminal node '{terminal}' is a leaf"
        )
print("OK - all partial samples terminate at internal nodes")
print("OK - all leaf samples terminate at leaf nodes")
