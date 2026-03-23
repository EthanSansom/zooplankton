from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from cnn.data import ImageDataset

# Setup ------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "00_raw_data"

# Load dataset -----------------------------------------------------------------

print("\nLoading ImageDataset...")

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

dataset = ImageDataset(root=DATA_DIR, transform=transform)

print(f"\n{dataset}")
print("\nClass to index mapping:")
for class_name, index in dataset.class_to_index.items():
    print(f"  {index:>3}  {class_name}")

# Class distribution -----------------------------------------------------------

print("\nCLASS DISTRIBUTION")

counts = Counter(dataset.labels)
total = len(dataset)

print(f"\n{'Class':<25} {'Count':>8} {'Proportion':>12}")
print("-" * 47)
for class_name, index in dataset.class_to_index.items():
    count = counts[index]
    prop = count / total
    print(f"  {class_name:<23} {count:>8,} {prop:>11.1%}")
print("-" * 47)
print(f"  {'TOTAL':<23} {total:>8,}")

# Sample items -----------------------------------------------------------------

print("\nSAMPLE ITEMS")

for i in [0, 1, 2]:
    image, label = dataset[i]
    class_name = [k for k, v in dataset.class_to_index.items() if v == label][0]
    print(f"\n  Sample {i}:")
    print(f"    Image shape: {image.shape}")
    print(f"    Image dtype: {image.dtype}")
    print(f"    Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"    Label:       {label} ({class_name})")
    print(f"    Path:        {dataset.image_paths[i]}")

# DataLoader -------------------------------------------------------------------

print("\nDATALOADER BATCH")

loader = DataLoader(dataset, batch_size=8, shuffle=True)
images, labels = next(iter(loader))

print(f"\n  Batch image shape: {images.shape}")
print(f"  Batch label shape: {labels.shape}")
print(f"  Labels in batch:   {labels.tolist()}")
