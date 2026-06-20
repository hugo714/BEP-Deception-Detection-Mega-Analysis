"""
sensitivity_rq1.py
------------------
Sensitivity analysis for RQ1: pairwise representation comparisons
restricted to the 15 leakage-free LODO folds only.

Methodology mirrors the main RQ1 analysis but excludes the 10 folds where
near-identical sibling datasets were present in the training corpus (leakage).

Tests:
  - Three pairwise two-sided paired permutation t-tests
    (10,000 sign-flip permutations, seed=42):
      TF-IDF vs FastText | TF-IDF vs SBERT | FastText vs SBERT
  - Run separately for ROC-AUC and macro-F1 (6 tests total)
  - Benjamini-Hochberg FDR correction (q=0.05) within each metric family
  - Matched-pair Cohen's d = mean(diffs) / sd(diffs)  [ddof=1]

Input files (read-only, not modified):
  results/tfidf_results.csv
  results/fasttext_results.csv
  results/sbert_results.csv

Output:
  results/sensitivity_rq1_results.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# Discovered from results/tfidf_results.csv — all 10 map exactly, verified below
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

N_PERM = 10_000
SEED   = 42
Q_FDR  = 0.05

OUTPUT_CSV = ROOT / "results" / "sensitivity_rq1_results.csv"

# ---------------------------------------------------------------------------
# Data loading and leakage-fold validation
# ---------------------------------------------------------------------------

def load_and_filter():
    """
    Load per-fold AUC and macro-F1 from each representation's results file,
    merge on dataset name, then filter to the 15 leakage-free folds.
    Prints a mismatch report if any leakage-fold name is not found.
    """
    tfidf    = pd.read_csv(ROOT / "results" / "tfidf_results.csv"
                           )[["dataset", "roc_auc", "f1_macro"]
                            ].rename(columns={"roc_auc": "auc_tfidf",
                                              "f1_macro": "f1_tfidf"})

    fasttext = pd.read_csv(ROOT / "results" / "fasttext_results.csv"
                           )[["dataset", "roc_auc", "f1_macro"]
                            ].rename(columns={"roc_auc": "auc_fasttext",
                                              "f1_macro": "f1_fasttext"})

    sbert    = pd.read_csv(ROOT / "results" / "sbert_results.csv"
                           )[["dataset", "roc_auc", "f1_macro"]
                            ].rename(columns={"roc_auc": "auc_sbert",
                                              "f1_macro": "f1_sbert"})

    # Merge all three on dataset name
    merged = tfidf.merge(fasttext, on="dataset").merge(sbert, on="dataset")
    all_datasets = set(merged["dataset"])

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
            f"The following leakage-fold names were not found in results files: "
            f"{mismatches}. Adjust LEAKAGE_FOLDS to match exact dataset strings."
        )
    print()

    # Apply filter: keep only leakage-free folds
    free = merged[~merged["dataset"].isin(LEAKAGE_FOLDS)].reset_index(drop=True)

    excluded = merged[merged["dataset"].isin(LEAKAGE_FOLDS)]
    assert len(free) + len(excluded) == len(merged), "Filter row count mismatch"
    return free, merged

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def paired_permutation_test(a: np.ndarray, b: np.ndarray,
                             n_perm: int = N_PERM,
                             seed: int   = SEED) -> tuple[float, float]:
    """
    Two-sided paired sign-flip permutation test.

    Under H0 the sign of each difference is exchangeable; we flip signs
    randomly and count how often the permuted |t| meets or exceeds the
    observed |t|.

    Returns (t_observed, p_value).
    """
    rng   = np.random.default_rng(seed)
    diffs = a - b
    n     = len(diffs)
    se    = diffs.std(ddof=1) / np.sqrt(n)
    t_obs = diffs.mean() / se

    extreme = 0
    for _ in range(n_perm):
        signs      = rng.choice([-1.0, 1.0], size=n)
        pd_        = diffs * signs
        t_perm     = pd_.mean() / (pd_.std(ddof=1) / np.sqrt(n))
        if abs(t_perm) >= abs(t_obs):
            extreme += 1

    p = extreme / n_perm
    return float(t_obs), float(p)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Matched-pair Cohen's d = mean(diffs) / sd(diffs).
    Uses ddof=1 (sample standard deviation).
    """
    diffs = a - b
    return float(diffs.mean() / diffs.std(ddof=1))


def apply_bh(p_values: list[float], q: float = Q_FDR) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns array of BH-adjusted p-values in the same order.
    """
    _, p_adj, _, _ = multipletests(p_values, alpha=q, method="fdr_bh")
    return p_adj

# ---------------------------------------------------------------------------
# Run pairwise tests for one metric
# ---------------------------------------------------------------------------

def run_metric(df: pd.DataFrame,
               col_a: str, col_b: str, col_c: str,
               metric_label: str) -> list[dict]:
    """
    Run three pairwise permutation tests for one metric (AUC or F1).
    Returns a list of result dicts (one per pair) before BH correction.
    """
    pairs = [
        ("TF-IDF",   "FastText", col_a, col_b),
        ("TF-IDF",   "SBERT",    col_a, col_c),
        ("FastText",  "SBERT",   col_b, col_c),
    ]
    rows = []
    for rep_a, rep_b, ca, cb in pairs:
        a = df[ca].values.astype(float)
        b = df[cb].values.astype(float)

        t_stat, p_raw = paired_permutation_test(a, b)
        d             = cohens_d_paired(a, b)
        median_delta  = float(np.median(a - b))

        rows.append({
            "Pair"       : f"{rep_a} vs {rep_b}",
            "Metric"     : metric_label,
            "Median_Δ"   : round(median_delta, 4),
            "Cohen_d"    : round(d, 4),
            "t"          : round(t_stat, 4),
            "p_raw"      : round(p_raw, 4),
        })
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    free, merged = load_and_filter()

    # ── Verification: print retained fold names and per-rep AUC ──────────
    print("=" * 65)
    print(f"RETAINED FOLDS  (n = {len(free)}, leakage-free)")
    print("=" * 65)
    verify_cols = ["dataset", "auc_tfidf", "auc_fasttext", "auc_sbert"]
    print(free[verify_cols].to_string(index=False))
    print()

    # ── Run tests for AUC and F1 ─────────────────────────────────────────
    auc_rows = run_metric(free,
                          "auc_tfidf", "auc_fasttext", "auc_sbert",
                          "ROC-AUC")
    f1_rows  = run_metric(free,
                          "f1_tfidf", "f1_fasttext", "f1_sbert",
                          "macro-F1")

    # ── Apply BH correction within each metric family ────────────────────
    auc_p_bh = apply_bh([r["p_raw"] for r in auc_rows])
    f1_p_bh  = apply_bh([r["p_raw"] for r in f1_rows])

    for row, p_bh in zip(auc_rows, auc_p_bh):
        row["p_BH"] = round(float(p_bh), 4)
    for row, p_bh in zip(f1_rows, f1_p_bh):
        row["p_BH"] = round(float(p_bh), 4)

    all_rows = auc_rows + f1_rows
    results  = pd.DataFrame(all_rows, columns=[
        "Pair", "Metric", "Median_Δ", "Cohen_d", "t", "p_raw", "p_BH"
    ])

    # ── Print results table ───────────────────────────────────────────────
    print("=" * 65)
    print("PAIRWISE PERMUTATION TESTS — 15 LEAKAGE-FREE FOLDS")
    print(f"(10,000 sign-flip permutations, seed={SEED}, BH q={Q_FDR})")
    print("=" * 65)
    print(results.to_string(index=False))
    print()

    # Significance summary
    print("Significant after BH correction (p_BH < 0.05):")
    sig = results[results["p_BH"] < Q_FDR]
    if sig.empty:
        print("  None")
    else:
        print(sig[["Pair", "Metric", "Cohen_d", "p_BH"]].to_string(index=False))
    print()

    # ── Save CSV ─────────────────────────────────────────────────────────
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved → {OUTPUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
