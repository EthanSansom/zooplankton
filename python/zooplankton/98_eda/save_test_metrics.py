import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
RESULTS_DIR = BASE_DIR / "01_results"
OUTPUT_DIR = RESULTS_DIR

models = {
    "flat": RESULTS_DIR / "flat_20260407_191102",
    "lcpn": RESULTS_DIR / "lcpn_20260407_211606",
    "lcpn_extended_hierarchy": RESULTS_DIR / "lcpn_extended_20260407_232625",
}

model_to_name = {
    "flat": "Flat",
    "lcpn": "LCPN",
    "lcpn_extended_hierarchy": "LCPN (Extended)",
}

ALL_CLASSES = [
    "bosmina",
    "calanoid",
    "cyclopoid",
    "daphnia",
    "harpacticoid",
    "nauplius",
    "rotifer",
    "exoskeleton",
    "fiber",
    "plant_matter",
    "bubbles",
]

# Helpers ----------------------------------------------------------------------


def compute_metrics(true, preds, average="macro", labels=None):
    kwargs = dict(average=average, zero_division=0, labels=labels)

    # accuracy_score() doesn't have an option to subset labels
    def accuracy_score_with_labels(true, preds, labels=None):
        if labels is not None:
            mask = [i for (i, label) in enumerate(true) if label in labels]
            true = [true[i] for i in mask]
            preds = [preds[i] for i in mask]
        return accuracy_score(true, preds)

    return {
        "accuracy": round(accuracy_score_with_labels(true, preds, labels), 4),
        "f1": round(f1_score(true, preds, **kwargs), 4),
        "precision": round(precision_score(true, preds, **kwargs), 4),
        "recall": round(recall_score(true, preds, **kwargs), 4),
    }


def extract_level(sequences, level):
    return [
        seq[min(level, len(seq) - 1)] if isinstance(seq, list) else seq
        for seq in sequences
    ]


def extract_leaf(sequences):
    return [seq[-1] if isinstance(seq, list) else seq for seq in sequences]


# Load data --------------------------------------------------------------------

data = {}
for model_id, results_dir in models.items():
    with open(results_dir / "test_results.json") as f:
        test_results = json.load(f)
    with open(results_dir / "history.json") as f:
        history = json.load(f)
    data[model_id] = {
        "test_results": test_results,
        "history": history,
    }

# Table 1: Leaf-level overall metrics -----------------------------------------

leaf_rows = []
for model_id, model_data in data.items():
    preds = model_data["test_results"]["predictions"]
    true = model_data["test_results"]["true"]
    metrics = compute_metrics(true, preds)
    leaf_rows.append({"model": model_to_name[model_id], **metrics})

pd.DataFrame(leaf_rows).to_csv(OUTPUT_DIR / "metrics_leaf_overall.csv", index=False)
print("Saved: metrics_leaf_overall.csv")

# Table 2: Level 1 and Level 2 overall metrics --------------------------------

level_rows = []
for model_id, model_data in data.items():
    if model_id == "flat":
        continue
    preds_paths = model_data["test_results"]["pred_paths"]
    true_paths = model_data["test_results"]["true_paths"]
    for level_idx, level_name in enumerate(["Level 1", "Level 2"], start=1):
        true = extract_level(true_paths, level_idx)
        preds = extract_level(preds_paths, level_idx)
        metrics = compute_metrics(true, preds)
        level_rows.append(
            {
                "model": model_to_name[model_id],
                "level": level_name,
                **metrics,
            }
        )

pd.DataFrame(level_rows).to_csv(OUTPUT_DIR / "metrics_by_level.csv", index=False)
print("Saved: metrics_by_level.csv")

# Table 3: Per-class leaf-level metrics ---------------------------------------

per_class_rows = []
for model_id, model_data in data.items():
    preds = model_data["test_results"]["predictions"]
    true = model_data["test_results"]["true"]
    for image_class in ALL_CLASSES:
        metrics = compute_metrics(true, preds, average="macro", labels=[image_class])
        per_class_rows.append(
            {
                "model": model_to_name[model_id],
                "class": image_class,
                **metrics,
            }
        )

pd.DataFrame(per_class_rows).to_csv(
    OUTPUT_DIR / "metrics_leaf_per_class.csv", index=False
)
print("Saved: metrics_leaf_per_class.csv")

# Table 4: Training information -----------------------------------------------

training_rows = []
for model_id, model_data in data.items():
    history = model_data["history"]
    epochs = history["epochs_completed"]
    duration = history["duration_seconds"]
    training_rows.append(
        {
            "model": model_to_name[model_id],
            "epochs_completed": epochs,
            "duration_min": round(duration / 60, 1),
            "duration_per_epoch_min": round(duration / epochs / 60, 2),
        }
    )

pd.DataFrame(training_rows).to_csv(OUTPUT_DIR / "training_info.csv", index=False)
print("Saved: training_info.csv")
