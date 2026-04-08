import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configuration ----------------------------------------------------------------

BASE_DIR = Path("python/zooplankton")
RESULTS_DIR = BASE_DIR / "01_results"

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

OUTPUT_IMAGE = RESULTS_DIR / "metrics_table.png"

# Helpers ----------------------------------------------------------------------


def compute_metrics(true_labels, pred_labels):
    """Return (accuracy, f1_macro, precision_macro, recall_macro)."""
    avg = "macro"
    zero_division = dict(zero_division=0)
    return {
        "Accuracy": round(accuracy_score(true_labels, pred_labels), 4),
        "F1": round(
            f1_score(true_labels, pred_labels, average=avg, **zero_division), 4
        ),
        "Precision": round(
            precision_score(true_labels, pred_labels, average=avg, **zero_division), 4
        ),
        "Recall": round(
            recall_score(true_labels, pred_labels, average=avg, **zero_division), 4
        ),
    }


def extract_level(sequences, level):
    """
    Pull the label at the nth level from a hierarchical label.
    """
    out = []
    for seq in sequences:
        if isinstance(seq, list):
            idx = min(level, len(seq) - 1)
            out.append(seq[idx])
        else:
            out.append(seq)
    return out


def extract_leaf(sequences):
    """Always return the last element of each sequence (deepest prediction)."""
    out = []
    for seq in sequences:
        if isinstance(seq, list):
            out.append(seq[-1])
        else:
            out.append(seq)
    return out


# Load and calculate -----------------------------------------------------------

METRICS = ["Accuracy", "F1", "Precision", "Recall"]
LEVELS = ["Level 1", "Level 2", "Leaf"]

rows = []  # each row: (model_name, level, metric_dict)

for model_id, results_dir in models.items():
    with open(results_dir / "test_results.json") as f:
        data = json.load(f)

    is_flat = model_id == "flat"

    # Flat predictions/true used for leaf metrics for all models, as these exclude
    # predications on partially labelled samples.
    preds_leaf = data["predictions"]
    true_leaf = data["true"]

    # Hierarchical paths for level 1 / level 2 metrics
    if not is_flat:
        preds_paths = data["pred_paths"]
        true_paths = data["true_paths"]

    for level_name in LEVELS:
        # Flat models only have leaf predictions
        if is_flat and level_name in ("Level 1", "Level 2"):
            rows.append((model_to_name[model_id], level_name, None))
            continue

        if level_name == "Level 1":
            t = extract_level(true_paths, 1)
            p = extract_level(preds_paths, 1)
        elif level_name == "Level 2":
            t = extract_level(true_paths, 2)
            p = extract_level(preds_paths, 2)
        else:  # Leaf level always uses flat keys.
            t = true_leaf
            p = preds_leaf

        rows.append((model_to_name[model_id], level_name, compute_metrics(t, p)))

# Table ------------------------------------------------------------------------

col_headers = ["Model", "Level"] + METRICS
table_data = []

for model_name, level_name, metrics in rows:
    if metrics is None:
        row = [model_name, level_name] + ["N/A"] * len(METRICS)
    else:
        row = [model_name, level_name] + [f"{metrics[m]:.4f}" for m in METRICS]
    table_data.append(row)

n_rows = len(table_data)
n_cols = len(col_headers)

fig_h = 0.45 * (n_rows + 2) + 0.8
fig, ax = plt.subplots(figsize=(11, fig_h))
ax.axis("off")

HEADER_COLOR = "#2C3E50"
MODEL_COLORS = {
    "Flat": "#EBF5FB",
    "LCPN": "#EAFAF1",
    "LCPN (Extended)": "#FEF9E7",
}
NA_COLOR = "#F2F3F4"
TEXT_DARK = "#1C2833"
TEXT_LIGHT = "white"
BORDER_COLOR = "#ABB2B9"

col_widths = [0.18, 0.10, 0.18, 0.18, 0.18, 0.18]

tbl = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for col_idx in range(n_cols):
    cell = tbl[0, col_idx]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color=TEXT_LIGHT, fontweight="bold")
    cell.set_edgecolor(BORDER_COLOR)

for row_idx, (model_name, level_name, _) in enumerate(rows, start=1):
    base_color = MODEL_COLORS.get(model_name, "white")
    for col_idx in range(n_cols):
        cell = tbl[row_idx, col_idx]
        cell_val = table_data[row_idx - 1][col_idx]

        if cell_val == "N/A":
            cell.set_facecolor(NA_COLOR)
            cell.set_text_props(color="#AAB7B8", style="italic")
        else:
            cell.set_facecolor(base_color)
            cell.set_text_props(color=TEXT_DARK)

        if col_idx == 0:
            cell.set_text_props(fontweight="bold", color=TEXT_DARK)
        if col_idx == 1:
            cell.set_text_props(fontstyle="italic", color="#5D6D7E")

        cell.set_edgecolor(BORDER_COLOR)

legend_patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
ax.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=3,
    fontsize=9,
    frameon=False,
)

ax.set_title(
    "Model Comparison: Hierarchical Classification Metrics",
    fontsize=13,
    fontweight="bold",
    color=TEXT_DARK,
    pad=14,
)
fig.text(
    0.5,
    0.01,
    "Metrics computed macro-averaged over all classes on the held-out test set.  "
    "Level 1 = root label · Level 2 = sub-class label · Leaf = deepest label.  "
    "N/A = not applicable for flat classifier.",
    ha="center",
    fontsize=8,
    color="#717D7E",
    style="italic",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTPUT_IMAGE}")
plt.show()
