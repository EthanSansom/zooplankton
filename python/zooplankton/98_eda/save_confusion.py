import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

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

# Helpers ----------------------------------------------------------------------


def normalized_confusion_to_df(true, preds, model_name):
    labels = sorted(set(true))
    cm = confusion_matrix(true, preds, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append(
                {
                    "model": model_name,
                    "true": true_label,
                    "predicted": pred_label,
                    "proportion": round(cm_normalized[i, j], 4),
                }
            )
    return pd.DataFrame(rows)


def extract_level(sequences, level):
    return [
        seq[min(level, len(seq) - 1)] if isinstance(seq, list) else seq
        for seq in sequences
    ]


# Load data --------------------------------------------------------------------

data = {}
for model_id, results_dir in models.items():
    with open(results_dir / "test_results.json") as f:
        data[model_id] = json.load(f)

# Table 1: Normalized leaf-level confusion matrix for all models ---------------

leaf_frames = []
for model_id, test_results in data.items():
    df = normalized_confusion_to_df(
        test_results["true"],
        test_results["predictions"],
        model_to_name[model_id],
    )
    leaf_frames.append(df)

pd.concat(leaf_frames).to_csv(OUTPUT_DIR / "confusion_leaf.csv", index=False)
print("Saved: confusion_leaf.csv")

# Tables 2, 3: Normalized Level-1 and Level-2 confusion for LCPN models --------

lcpn_models = {k: v for k, v in models.items() if k != "flat"}

for level_idx, level_name in enumerate(["Level 1", "Level 2"], start=1):
    level_frames = []
    for model_id in lcpn_models:
        test_results = data[model_id]
        true = extract_level(test_results["true_paths"], level_idx)
        preds = extract_level(test_results["pred_paths"], level_idx)
        df = normalized_confusion_to_df(true, preds, model_to_name[model_id])
        level_frames.append(df)

    filename = f"confusion_level_{level_idx}.csv"
    pd.concat(level_frames).to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Saved: {filename}")
