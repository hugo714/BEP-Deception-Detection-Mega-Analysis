"""
generate_heatmap.py
Produces publication-ready LODO heatmaps for a deception-detection thesis.

Outputs (same directory as script):
  lodo_heatmap.png     — AUC values
  lodo_heatmap_f1.png  — Macro-F1 values
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

TFIDF_PATH   = os.path.join(RESULTS_DIR, "tfidf_results.csv")
FASTTEXT_PATH = os.path.join(RESULTS_DIR, "fasttext_results.csv")
SBERT_PATH   = os.path.join(RESULTS_DIR, "sbert_results.csv")

# --------------------------------------------------------------------------- #
# Leakage-affected domains (asterisk appended to row label)
# --------------------------------------------------------------------------- #
LEAKAGE_DOMAINS = {
    # Ott et al. (2 domains)
    "deceptive opinion spam",
    "negative deceptive opinion spam",
    # Zeng et al. 2022 (5 subsets)
    "zeng et al 2022 fake news",
    "zeng et al 2022 job scams",
    "zeng et al 2022 phishing",
    "zeng et al 2022 political statements",
    "zeng et al 2022 product reviews",
    # Li et al. 2015 (3 subsets)
    "li et al 2015 doctor",
    "li et al 2015 hotel",
    "li et al 2015 restaurant",
}

# --------------------------------------------------------------------------- #
# Font setup
# --------------------------------------------------------------------------- #
matplotlib.rcParams["font.family"] = "serif"
# Try Times New Roman first, fall back gracefully
for font in ["Times New Roman", "DejaVu Serif", "Georgia"]:
    matplotlib.rcParams["font.serif"] = [font]
    break


# --------------------------------------------------------------------------- #
# Helper: build pivot table
# --------------------------------------------------------------------------- #
def build_pivot(metric: str) -> pd.DataFrame:
    """Return a DataFrame with domains as rows, representations as columns."""
    frames = {}
    for label, path in [
        ("TF-IDF",   TFIDF_PATH),
        ("FastText",  FASTTEXT_PATH),
        ("SBERT",     SBERT_PATH),
    ]:
        df = pd.read_csv(path)
        # Normalise column name variants
        col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
        metric_col = col_map.get(metric.lower().replace(" ", "_"), metric)
        frames[label] = df.set_index("dataset")[metric_col]

    pivot = pd.DataFrame(frames)[["TF-IDF", "FastText", "SBERT"]]

    # Sort rows by mean across representations, descending
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

    return pivot


# --------------------------------------------------------------------------- #
# Helper: row labels with asterisk for leakage domains
# --------------------------------------------------------------------------- #
def make_row_labels(index: pd.Index) -> list[str]:
    labels = []
    for name in index:
        suffix = " *" if name in LEAKAGE_DOMAINS else ""
        # Title-case for readability
        labels.append(name.title() + suffix)
    return labels


# --------------------------------------------------------------------------- #
# Core plotting function
# --------------------------------------------------------------------------- #
def plot_heatmap(
    pivot: pd.DataFrame,
    metric_label: str,
    out_path: str,
    center: float = 0.5,
    cmap: str = "RdYlGn",
) -> None:
    row_labels = make_row_labels(pivot.index)
    annot_data = pivot.round(3).astype(str)  # text shown inside cells

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.heatmap(
        pivot,
        ax=ax,
        annot=annot_data,
        fmt="",
        cmap=cmap,
        center=center,
        vmin=0.3,
        vmax=0.85,
        linewidths=0,
        linecolor="none",
        cbar_kws={
            "label": metric_label,
            "shrink": 0.6,
            "pad": 0.02,
        },
        annot_kws={"size": 8, "family": "serif"},
    )

    # Axis labels & ticks
    ax.set_yticklabels(row_labels, rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11, fontweight="bold")
    ax.set_xlabel("Representation", fontsize=12, labelpad=8)
    ax.set_ylabel("Domain", fontsize=12, labelpad=8)
    ax.set_title(
        f"LODO Evaluation — {metric_label} by Domain and Representation",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Gridlines off
    ax.grid(False)

    # Colour-bar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(metric_label, fontsize=10)

    # Legend note
    note = "* indicates domain has sibling dataset(s) in training corpus"
    fig.text(
        0.5, -0.01,
        note,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        transform=fig.transFigure,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # AUC heatmap
    pivot_auc = build_pivot("roc_auc")
    plot_heatmap(
        pivot_auc,
        metric_label="AUC",
        out_path=os.path.join(SCRIPT_DIR, "lodo_heatmap.png"),
    )

    # Macro-F1 heatmap
    pivot_f1 = build_pivot("f1_macro")
    plot_heatmap(
        pivot_f1,
        metric_label="Macro F1",
        out_path=os.path.join(SCRIPT_DIR, "lodo_heatmap_f1.png"),
    )

    print("Done.")
