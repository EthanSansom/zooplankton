from cnn.data import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter

# Setup ------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "00_raw_data"

# 1. Load dataset --------------------------------------------------------------

print("=" * 60)
print("1. Loading ImageDataset")
print("=" * 60)

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

# 2. Class distribution --------------------------------------------------------

print("\n" + "=" * 60)
print("2. Class distribution")
print("=" * 60)

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

# 3. Sample items --------------------------------------------------------------

print("\n" + "=" * 60)
print("3. Sample items")
print("=" * 60)

for i in [0, 1, 2]:
    image, label = dataset[i]
    class_name = [k for k, v in dataset.class_to_index.items() if v == label][0]
    print(f"\n  Sample {i}:")
    print(f"    Image shape: {image.shape}")
    print(f"    Image dtype: {image.dtype}")
    print(f"    Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"    Label:       {label} ({class_name})")
    print(f"    Path:        {dataset.image_paths[i]}")

# 4. DataLoader ----------------------------------------------------------------

print("\n" + "=" * 60)
print("4. DataLoader batch")
print("=" * 60)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
images, labels = next(iter(loader))

print(f"\n  Batch image shape: {images.shape}")
print(f"  Batch label shape: {labels.shape}")
print(f"  Labels in batch:   {labels.tolist()}")

# 5. LCPNDataset compatibility -------------------------------------------------

print("\n" + "=" * 60)
print("5. leaf_index_to_name (for LCPNDataset)")
print("=" * 60)

leaf_index_to_name = {v: k for k, v in dataset.class_to_index.items()}
print(f"\n  leaf_index_to_name has {len(leaf_index_to_name)} entries")
print("  Example entries:")
for index in list(leaf_index_to_name.keys())[:5]:
    print(f"    {index}: '{leaf_index_to_name[index]}'")
