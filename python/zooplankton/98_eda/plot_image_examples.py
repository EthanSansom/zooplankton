from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
DATA_DIR = BASE_DIR / "00_raw_data"
CLASS_NMAX = 10_000

class_to_dirs = {
    "Bosmina": ["bosminidae", "eubosmina"],
    "Daphnia": ["daphnia"],
    "Rotifer": ["rotifer", "trichocerca", "conochilus", "kellicottia"],
    "Nauplius": ["nauplius_copepod"],
    "Cyclopoid": ["cyclopoid"],
    "Harpacticoid": ["harpacticoid"],
    "Calanoid": ["calanoid"],
    "Exoskeleton": ["exoskeleton"],
    "Fiber": ["fiber_hairlike", "fiber_squiggly"],
    "Plant Matter": ["plant_matter"],
    "Bubbles": ["bubbles"],
}

# Load one image per class and count samples -----------------------------------

class_images = {}
class_counts = {}

for class_name, dirs in class_to_dirs.items():
    total = 0
    for dir_name in dirs:
        candidates = sorted((DATA_DIR / dir_name).glob("*.tif"))
        total += len(candidates)
        if class_name not in class_images and candidates:
            class_images[class_name] = Image.open(candidates[0]).convert("L")
    class_counts[class_name] = total

# Plot: sample images ----------------------------------------------------------

n_classes = len(class_images)
n_cols = 4
n_rows = (n_classes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
axes = axes.flatten()

for i, (class_name, image) in enumerate(class_images.items()):
    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(class_name, fontsize=10)
    axes[i].axis("off")

for i in range(len(class_images), len(axes)):
    axes[i].axis("off")

plt.suptitle("Sample Image per Class", fontsize=12)
plt.tight_layout()
plt.savefig(BASE_DIR / "01_results" / "class_samples.png", dpi=150)
plt.show()

# Plot: class counts -----------------------------------------------------------

labels = list(class_counts.keys())
counts_capped = [min(c, CLASS_NMAX) for c in class_counts.values()]
counts_raw = list(class_counts.values())

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, counts_capped)

for bar, raw_count in zip(bars, counts_raw):
    label = f"{CLASS_NMAX:,}+" if raw_count > CLASS_NMAX else f"{raw_count:,}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.set_ylim(0, CLASS_NMAX * 1.1)
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")
ax.set_title("Class Sample Counts (Capped at 10,000)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(BASE_DIR / "01_results" / "class_counts.png", dpi=150)
plt.show()
