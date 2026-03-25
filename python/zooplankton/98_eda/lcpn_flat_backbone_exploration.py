import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
RESULTS_PATH = BASE_DIR / "01_results/lcpn_20260323_221528/test_results.json"
DATA_DIR = BASE_DIR / "00_raw_data"

INSPECT_TRUE = "harpacticoid"
INSPECT_PRED = "cyclopoid"
N_SAMPLES = 16

# Load results -----------------------------------------------------------------

with open(RESULTS_PATH) as f:
    test_results = json.load(f)

preds = test_results["predictions"]
true = test_results["true"]

# Confusion matrix -------------------------------------------------------------

labels = sorted(set(true))
cm = confusion_matrix(true, preds, labels=labels)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Normalized): LCPN Flat Backbone Frozen")

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_normalized[i, j] > 0.5 else "black",
            fontsize=7,
        )

plt.tight_layout()
plt.savefig(RESULTS_PATH.parent / "confusion_matrix.png", dpi=150)
plt.show()
