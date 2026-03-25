import json
from pathlib import Path

import matplotlib.pyplot as plt

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
RESULTS_DIR = BASE_DIR / "01_results"

models = {
    "flat": RESULTS_DIR / "flat_20260323_162915",
    "lcpn_extended_hierarchy": RESULTS_DIR / "lcpn_20260323_221528",
}

model_to_name = {
    "flat": "Flat Classifier",
    "lcpn_extended_hierarchy": "LCPN Classifier (Extended Hierarchy)",
}

# Load histories ---------------------------------------------------------------

max_epochs = 0
histories = {}
for model_id, results_dir in models.items():
    with open(results_dir / "history.json") as f:
        histories[model_id] = json.load(f)
        if max_epochs < histories[model_id]["epochs_completed"]:
            max_epochs = histories[model_id]["epochs_completed"]

# Plot -------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

fig, ax = plt.subplots(figsize=(10, 5))

for model_id, history in histories.items():
    acc = [e["accuracy"] for e in history["valid"]]
    n_epochs = len(acc)
    epochs = range(1, n_epochs + 1)
    (line,) = ax.plot(epochs, acc, label=model_to_name[model_id])

    # Dashed horizontal line at best accuracy, extending to max_epochs
    best_acc = max(acc)
    best_epoch = acc.index(best_acc) + 1
    ax.hlines(
        best_acc,
        xmin=1,
        xmax=max_epochs,
        color=line.get_color(),
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.text(
        max_epochs + 0.1,
        best_acc,
        f"{best_acc:.3f}",
        va="center",
        fontsize=8,
        color=line.get_color(),
        alpha=0.6,
    )

ax.set_xticks(range(1, max_epochs + 1))
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Flat vs. LCPN (Extended Hierarchy): Validation Accuracy")
fig.text(
    0.5,
    -0.02,
    "Dashed horizontal lines and corresponding labels indicate the best validation accuracy achieved by each model.",
    ha="center",
    fontsize=9,
    style="italic",
)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "flat_vs_lcpn_extended_validation_accuracy.png", dpi=150)
plt.show()
