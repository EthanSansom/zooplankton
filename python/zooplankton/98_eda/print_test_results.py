import json
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
RESULTS_DIR = BASE_DIR / "01_results"

models = {
    "flat": RESULTS_DIR / "flat_20260323_162915",
    "lcpn": RESULTS_DIR / "lcpn_20260323_171541",
    # "lcpn_extended_hierarchy": RESULTS_DIR / "lcpn_20260323_221528",
}

model_to_name = {
    "flat": "Flat Classifier",
    "lcpn": "LCPN Classifier",
    # "lcpn_extended_hierarchy": "LCPN Classifier (Extended Hierarchy)",
}


# Load results -----------------------------------------------------------------

rows = []
for model_id, results_dir in models.items():
    with open(results_dir / "test_results.json") as f:
        test_results = json.load(f)
    with open(results_dir / "history.json") as f:
        history = json.load(f)

    preds = test_results["predictions"]
    true = test_results["true"]
    epochs = history["epochs_completed"]
    duration = history["duration_seconds"]

    rows.append(
        {
            "model": model_to_name[model_id],
            "accuracy": accuracy_score(true, preds),
            "precision": precision_score(true, preds, average="macro", zero_division=0),
            "recall": recall_score(true, preds, average="macro", zero_division=0),
            "f1": f1_score(true, preds, average="macro", zero_division=0),
            "epochs": epochs,
            "duration_min": duration / 60,
            "duration_per_epoch": duration / epochs / 60,
        }
    )

# Print table ------------------------------------------------------------------

for row in rows:
    print(f"{row['model']}")
    print(f"  Accuracy:          {row['accuracy']:.4f}")
    print(f"  Precision:         {row['precision']:.4f}")
    print(f"  Recall:            {row['recall']:.4f}")
    print(f"  F1:                {row['f1']:.4f}")
    print(f"  Epochs:            {row['epochs']}")
    print(f"  Duration:          {row['duration_min']:.1f} min")
    print(f"  Per epoch:         {row['duration_per_epoch']:.2f} min")
    print()
