"""
sensitivity_rq2.py
------------------
Sensitivity analysis for RQ2: domain dissimilarity correlations
restricted to the 15 leakage-free LODO folds only.

OLS/Ridge regression is intentionally omitted — n = 15 is too small
for stable regression estimates (one regressor per two observations).

Correlations computed:
  Primary:   each feature vs transfer_difficulty (1 - avg_auc across 3 reps)
  Secondary: each feature vs per-representation AUC
             (auc_tfidf, auc_fasttext, auc_sbert)

Input file (read-only, not modified):
  results/rq2_features.csv   — produced by scripts/rq2_domain_dissimilarity.py;
                                contains all 25 folds, pre-computed features,
                                and per-representation AUC values.

  Note: auc_tfidf in rq2_features.csv uses the balanced TF-IDF variant
  (tfidf_results_balanced.csv), consistent with the original RQ2 analysis.

Output:
  results/sensitivity_rq2_results.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "results" / "rq2_features.csv"
OUTPUT_CSV   = ROOT / "results" / "sensitivity_rq2_results.csv"

# Discovered from rq2_features.csv — exact column names in that file
FEATURE_COLS = [
    "vocab_jaccard",
    "vocab_coverage",
    "js_divergence",
    "tfidf_cosine_dist",
    "sbert_cosine_dist",
    "mean_text_length",
    "pct_deceptive",
]

# Human-readable labels for the printed table
FEATURE_LABELS = {
    "vocab_jaccard"     : "Vocabulary Jaccard similarity",
    "vocab_coverage"    : "Vocabulary coverage",
    "js_divergence"     : "Jensen-Shannon distance",
    "tfidf_cosine_dist" : "TF-IDF centroid distance",
    "sbert_cosine_dist" : "SBERT centroid distance",
    "mean_text_length"  : "Mean document length",
    "pct_deceptive"     : "Proportion deceptive",
}

# 10 folds with near-sibling datasets in training — excluded from analysis
LEAKAGE_FOLDS = {
    "trec 2005 email spam",
    "deceptive opinion spam",
    "nazario trec 2007 email spam",
    "zeng et al 2022 phishing",
    "negative deceptive opinion spam",
    "li et al 2015 restaurant",
    "li et al 2015 hotel",
    "fake job postings",
    "zeng et al 2022 fake news",
    "zeng et al 2022 job scams",
}

# ---------------------------------------------------------------------------
# Data loading and leakage-fold validation
# ---------------------------------------------------------------------------

def load_and_filter() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the pre-computed feature matrix, validate leakage-fold names,
    and return (free_df, full_df) where free_df contains only the 15
    leakage-free folds.
    """
    df = pd.read_csv(FEATURES_CSV)
    all_datasets = set(df["dataset"])

    # Validate that every leakage-fold name resolves to a known dataset
    print("=" * 65)
    print("LEAKAGE-FOLD NAME CHECK")
    print("=" * 65)
    mismatches = []
    for name in sorted(LEAKAGE_FOLDS):
        status = "OK " if name in all_datasets else "MISMATCH"
        print(f"  {status}  {name!r}")
        if name not in all_datasets:
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            f"The following leakage-fold names were not found in "
            f"rq2_features.csv: {mismatches}. Adjust LEAKAGE_FOLDS."
        )
    print()

    # Filter to leakage-free folds
    free = df[~df["dataset"].isin(LEAKAGE_FOLDS)].reset_index(drop=True)
    assert len(free) == len(df) - len(LEAKAGE_FOLDS)
    return free, df


# ---------------------------------------------------------------------------
# Significance star helper
# ---------------------------------------------------------------------------

def sig_star(p: float) -> str:
    """Return significance star string per APA convention."""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r and two-tailed p-value for each feature against:
      - transfer_difficulty (primary outcome: 1 - avg_auc)
      - auc_tfidf, auc_fasttext, auc_sbert (secondary, per-representation)

    Rows with any NaN in the feature or outcome are dropped pairwise.
    Returns one row per feature.
    """
    per_rep_cols = ["auc_tfidf", "auc_fasttext", "auc_sbert"]
    rows = []

    for feat in FEATURE_COLS:
        row = {"Feature": FEATURE_LABELS[feat]}

        # Primary: correlation with transfer difficulty
        sub = df[[feat, "transfer_difficulty"]].dropna()
        r, p = pearsonr(sub[feat], sub["transfer_difficulty"])
        row["r"]    = round(r, 4)
        row["p"]    = round(p, 4)
        row["sig"]  = sig_star(p)
        row["n"]    = len(sub)

        # Secondary: per-representation AUC correlations
        for col in per_rep_cols:
            if col not in df.columns:
                row[f"r_{col.replace('auc_', '')}"] = float("nan")
                continue
            sub2 = df[[feat, col]].dropna()
            r2, _ = pearsonr(sub2[feat], sub2[col])
            row[f"r_{col.replace('auc_', '')}"] = round(r2, 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    free, full = load_and_filter()

    # ── Print retained folds for verification ────────────────────────────
    print("=" * 65)
    print(f"RETAINED FOLDS  (n = {len(free)}, leakage-free)")
    print("=" * 65)
    verify_cols = ["dataset", "transfer_difficulty", "auc_tfidf",
                   "auc_fasttext", "auc_sbert"]
    print(free[verify_cols].round(4).to_string(index=False))
    print()

    # ── Feature summary for the retained folds ───────────────────────────
    print("=" * 65)
    print("FEATURE SUMMARY — 15 leakage-free folds")
    print("=" * 65)
    print(free[["dataset"] + FEATURE_COLS].round(4).to_string(index=False))
    print()

    # ── Pearson correlations ─────────────────────────────────────────────
    corr = compute_correlations(free)

    print("=" * 65)
    print("PEARSON CORRELATIONS WITH TRANSFER DIFFICULTY")
    print(f"(n = {len(free)}, * p < .05, ** p < .01)")
    print("=" * 65)
    print(corr.to_string(index=False))
    print()

    # Highlight any significant results
    sig_rows = corr[corr["sig"] != ""]
    print(f"Significant correlations (p < .05): "
          f"{len(sig_rows)} of {len(corr)}")
    if not sig_rows.empty:
        print(sig_rows[["Feature", "r", "p", "sig"]].to_string(index=False))
    print()

    # ── Comparison: full 25 folds vs restricted 15 folds ────────────────
    print("=" * 65)
    print("COMPARISON: full 25 folds vs 15 leakage-free folds")
    print("=" * 65)
    corr_full     = compute_correlations(full)
    corr_full_sub = corr_full[["Feature", "r", "p"]].rename(
        columns={"r": "r_full25", "p": "p_full25"}
    )
    corr_free_sub = corr[["Feature", "r", "p"]].rename(
        columns={"r": "r_free15", "p": "p_free15"}
    )
    comparison = corr_full_sub.merge(corr_free_sub, on="Feature")
    comparison["Δr"] = (comparison["r_free15"] - comparison["r_full25"]).round(4)
    print(comparison.to_string(index=False))
    print()

    # ── Save ─────────────────────────────────────────────────────────────
    corr.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved → {OUTPUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
