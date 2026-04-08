import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
DATA_DIR = BASE_DIR / "00_raw_data"
RESULTS_DIR = BASE_DIR / "01_results"

models = {
    "flat": RESULTS_DIR / "flat_20260407_191102",
    "lcpn": RESULTS_DIR / "lcpn_20260407_211606",
    "lcpn_extended_hierarchy": RESULTS_DIR / "lcpn_extended_20260407_232625",
}

model_to_name = {
    "flat": "Flat Classifier",
    "lcpn": "LCPN Classifier",
    "lcpn_extended_hierarchy": "LCPN Classifier (Extended Hierarchy)",
}

COPEPODA_CLASSES = ["nauplius", "cyclopoid", "harpacticoid", "calanoid"]

# Helpers ----------------------------------------------------------------------


def plot_confusion_matrix(ax, cm_normalized, labels, title, fontsize_cells=7):
    labels_display = [label.replace("_", " ").capitalize() for label in labels]
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels_display, rotation=45, ha="right")
    ax.set_yticklabels(labels_display)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
                fontsize=fontsize_cells,
            )


CAPTION = (
    "Values represent the proportion of true-class samples predicted as each class "
    "(row-normalized). Diagonal entries are per-class recall."
)

# EDA loop ---------------------------------------------------------------------

for model_id, results_dir in models.items():
    print(f"\nMODEL: {model_id}")

    results_path = results_dir / "test_results.json"
    with open(results_path) as f:
        test_results = json.load(f)

    preds = test_results["predictions"]
    true = test_results["true"]

    # Full confusion matrix ----------------------------------------------------

    labels = sorted(set(true))
    cm = confusion_matrix(true, preds, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_confusion_matrix(
        ax,
        cm_normalized,
        labels,
        f"Confusion Matrix (Normalized): {model_to_name[model_id]}",
    )
    fig.text(0.5, -0.02, CAPTION, ha="center", fontsize=9, style="italic", wrap=True)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Copepoda subset confusion matrix -----------------------------------------

    # Filter to samples whose true label is a copepoda class. Predicted labels
    # may fall outside this set (e.g. the model predicts "rotifer" for a true
    # "calanoid"), so we use COPEPODA_CLASSES as the fixed label set, which
    # causes out-of-set predictions to be tallied in the off-diagonal columns
    # only and keeps the row sums correct for row-normalisation.
    copepoda_mask = [t in COPEPODA_CLASSES for t in true]
    true_copa = [t for t, m in zip(true, copepoda_mask) if m]
    preds_copa = [p for p, m in zip(preds, copepoda_mask) if m]

    if not true_copa:
        print(
            f"  No copepoda samples found in test set for {model_id}, skipping subset plot."
        )
        continue

    copa_labels = sorted(COPEPODA_CLASSES)
    cm_copa = confusion_matrix(true_copa, preds_copa, labels=copa_labels)
    cm_copa_normalized = cm_copa.astype(float) / cm_copa.sum(axis=1, keepdims=True)

    n_copa = len(true_copa)
    n_out_of_set = sum(p not in COPEPODA_CLASSES for p in preds_copa)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(
        ax,
        cm_copa_normalized,
        copa_labels,
        f"Copepoda Confusion Matrix (Normalized): {model_to_name[model_id]}",
        fontsize_cells=9,
    )
    ax.set_xlabel("Predicted\n")
    fig.text(0.5, -0.02, CAPTION, ha="center", fontsize=9, style="italic", wrap=True)
    plt.tight_layout()
    plt.savefig(
        results_dir / "confusion_matrix_copepoda.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
