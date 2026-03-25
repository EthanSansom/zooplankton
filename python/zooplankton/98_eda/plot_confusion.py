import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
DATA_DIR = BASE_DIR / "00_raw_data"
RESULTS_DIR = BASE_DIR / "01_results"

models = {
    "flat": RESULTS_DIR / "flat_20260323_162915",
    "lcpn": RESULTS_DIR / "lcpn_20260323_171541",
    "lcpn_flat_backbone_frozen": RESULTS_DIR
    / "lcpn_flat_backbone_frozen_20260323_192816",
    "lcpn_flat_backbone_unfrozen": RESULTS_DIR
    / "lcpn_flat_backbone_unfrozen_20260323_181503",
    "lcpn_extended_hierarchy": RESULTS_DIR / "lcpn_20260323_221528",
}

model_to_name = {
    "flat": "Flat Classifier",
    "lcpn": "LCPN Classifier",
    "lcpn_flat_backbone_frozen": "LCPN Classifier (Frozen Flat Backbone)",
    "lcpn_flat_backbone_unfrozen": "LCPN Classifier (Unfrozen Flat Backbone)",
    "lcpn_extended_hierarchy": "LCPN Classifier (Extended Hierarchy)",
}

# EDA loop ---------------------------------------------------------------------

for model_id, results_dir in models.items():
    print(f"\nMODEL: {model_id}")

    results_path = results_dir / "test_results.json"
    with open(results_path) as f:
        test_results = json.load(f)

    preds = test_results["predictions"]
    true = test_results["true"]

    # Confusion matrix ---------------------------------------------------------

    labels = sorted(set(true))
    labels_display = [label.capitalize() for label in labels]
    cm = confusion_matrix(true, preds, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels_display, rotation=45, ha="right")
    ax.set_yticklabels(labels_display)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Normalized): {model_to_name[model_id]}")
    fig.text(
        0.5,
        -0.02,
        "Values represent the proportion of true-class samples predicted as each class (row-normalized). "
        "Diagonal entries are per-class recall.",
        ha="center",
        fontsize=9,
        style="italic",
        wrap=True,
    )

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
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.show()
