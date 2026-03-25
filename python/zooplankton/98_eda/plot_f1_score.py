import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
RESULTS_DIR = BASE_DIR / "01_results"
ZOOPLANKTON_CLASSES = [
    "bosmina",
    "calanoid",
    "cyclopoid",
    "daphnia",
    "harpacticoid",
    "nauplius",
    "rotifer",
]

models = {
    "flat": RESULTS_DIR / "flat_20260323_162915",
    "lcpn": RESULTS_DIR / "lcpn_20260323_171541",
    "lcpn_extended_hierarchy": RESULTS_DIR / "lcpn_20260323_221528",
}

model_to_name = {
    "flat": "Flat Classifier",
    "lcpn": "LCPN Classifier",
    "lcpn_extended_hierarchy": "LCPN Classifier (Extended Hierarchy)",
}

# Load results -----------------------------------------------------------------

f1_scores = {}
for model_id, results_dir in models.items():
    with open(results_dir / "test_results.json") as f:
        test_results = json.load(f)

    preds = test_results["predictions"]
    true = test_results["true"]

    f1_scores[model_id] = f1_score(
        true, preds, labels=ZOOPLANKTON_CLASSES, average=None
    )

# Plot -------------------------------------------------------------------------

# Plot -------------------------------------------------------------------------

n_classes = len(ZOOPLANKTON_CLASSES)
n_models = len(model_to_name)
bar_width = 0.25
x = np.arange(n_classes)

fig, ax = plt.subplots(figsize=(12, 5))

for i, (model_id, name) in enumerate(model_to_name.items()):
    offset = (i - n_models / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, f1_scores[model_id], width=bar_width, label=name)

    for bar, value in zip(bars, f1_scores[model_id]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.03,
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=7,
            color="white",
        )

ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in ZOOPLANKTON_CLASSES])
ax.set_xlabel("Class")
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score: Zooplankton Leaf Classes")
ax.set_ylim(0, 1.0)
ax.legend(loc="lower right")

fig.text(
    0.5,
    -0.02,
    "F1 scores computed on the held-out test set. Only Zooplankton leaf classes shown.",
    ha="center",
    fontsize=9,
    style="italic",
)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "f1_bar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
