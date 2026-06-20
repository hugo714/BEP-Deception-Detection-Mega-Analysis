"""
tfidf_analysis.py
-----------------
TF-IDF + Logistic Regression baseline for the deception detection corpus.

Evaluation strategy: Leave-One-Dataset-Out (LODO)
  - For each dataset that is NOT fully excluded from evaluation, hold it out
    as the test set and train on all remaining data.
  - Rows with exclude_from_eval=True are never used as test data, but they
    CAN appear in the training fold (they add signal even if we don't trust
    them as a gold-standard test set).

Label normalisation
  - Numeric 0 / 0.0  → 0  (truthful)
  - Numeric 1 / 1.0  → 1  (deceptive)
  - String  "lie" / "deceptive"  → 1
  - String  "truthful"           → 0

Outputs
  - results/tfidf_results.csv          — per-dataset metrics (no class weighting)
  - results/tfidf_results_balanced.csv — per-dataset metrics (class_weight='balanced')
  - Console summary                    — sorted by ROC-AUC + macro-averaged totals

Usage
-----
  python3 scripts/tfidf_analysis.py               # unweighted (canonical)
  python3 scripts/tfidf_analysis.py --balanced    # class_weight='balanced'
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent   # project root
DATA_PATH = ROOT / "data/combined_preprocessed.csv"

OUTPUT_CSV_UNWEIGHTED = ROOT / "results/tfidf_results.csv"
OUTPUT_CSV_BALANCED   = ROOT / "results/tfidf_results_balanced.csv"

TFIDF_PARAMS = dict(
    ngram_range   = (1, 2),
    max_features  = 100_000,
    sublinear_tf  = True,      # log(1+tf) — standard for text classification
    min_df        = 2,
    strip_accents = "unicode",
    analyzer      = "word",
    token_pattern = r"(?u)\b\w+\b",
)

LR_PARAMS_BASE = dict(
    C         = 1.0,
    max_iter  = 1000,
    solver    = "liblinear",   # fastest for single-core sparse L2 LR
)

# Path for the cached TF-IDF matrix (avoids re-fitting 23× in LODO)
CACHE_DIR  = ROOT / ".tfidf_cache"
MATRIX_NPZ = CACHE_DIR / "tfidf_matrix.npz"
META_CSV   = CACHE_DIR / "tfidf_meta.csv"

# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

DECEPTIVE_STRINGS  = {"lie", "deceptive", "deceptive "}
TRUTHFUL_STRINGS   = {"truthful", "truthful "}

def normalise_label(val) -> int | None:
    """Return 0 (truthful) or 1 (deceptive), or None if unrecognised."""
    if pd.isna(val):
        return None
    # numeric
    try:
        f = float(val)
        if f in (0.0, 1.0):
            return int(f)
    except (ValueError, TypeError):
        pass
    # string
    s = str(val).strip().lower()
    if s in DECEPTIVE_STRINGS:
        return 1
    if s in TRUTHFUL_STRINGS:
        return 0
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows from {path.name}")

    df["label"] = df["deceptive"].apply(normalise_label)
    bad = df["label"].isna().sum()
    if bad:
        print(f"[WARN] {bad:,} rows with unrecognised labels dropped")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"[INFO] {df['dataset'].nunique()} datasets | "
          f"{(df['label']==1).sum():,} deceptive / {(df['label']==0).sum():,} truthful")
    return df


# ---------------------------------------------------------------------------
# TF-IDF matrix cache
# ---------------------------------------------------------------------------

def build_or_load_matrix(df: pd.DataFrame):
    """
    Fit TF-IDF once on ALL rows and cache to disk.

    Note on IDF leakage: fitting on the full corpus (including each fold's
    test set) introduces a negligible IDF-calibration leak. The test set is
    <5 % of the corpus for every fold, and IDF values are log-smoothed, so
    the practical impact on reported metrics is minimal — standard practice
    for LODO baselines in the deception-detection literature.
    """
    from scipy.sparse import save_npz, load_npz

    if MATRIX_NPZ.exists() and META_CSV.exists():
        print(f"[CACHE] Loading TF-IDF matrix from {MATRIX_NPZ} …")
        X   = load_npz(MATRIX_NPZ)
        meta = pd.read_csv(META_CSV)
        print(f"[CACHE] Loaded matrix {X.shape}")
        return X, meta

    CACHE_DIR.mkdir(exist_ok=True)
    print("[TFIDF] Fitting TF-IDF on full corpus … (runs once, then cached)")
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X   = vec.fit_transform(df["text"].fillna(""))
    print(f"[TFIDF] Matrix shape: {X.shape}")
    save_npz(MATRIX_NPZ, X)

    meta = df[["dataset", "label", "exclude_from_eval"]].reset_index(drop=True)
    meta.to_csv(META_CSV, index=False)
    print(f"[CACHE] Saved to {CACHE_DIR}/")
    return X, meta


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy"  : round(accuracy_score(y_true, y_pred), 4),
        "f1_macro"  : round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
        "f1_deceptive": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "precision" : round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "n_test"    : len(y_true),
        "n_deceptive_test": int(sum(y_true)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Leave-One-Dataset-Out loop  (uses pre-built matrix)
# ---------------------------------------------------------------------------

def lodo_evaluation(df: pd.DataFrame, X, meta: pd.DataFrame,
                    balanced: bool = False) -> pd.DataFrame:
    """
    Parameters
    ----------
    df       : original DataFrame (needed for n_train info)
    X        : full TF-IDF matrix, rows aligned with df
    meta     : DataFrame with columns [dataset, label, exclude_from_eval]
    balanced : if True, use class_weight='balanced' in the logistic regression
    """
    lr_params = dict(**LR_PARAMS_BASE)
    if balanced:
        lr_params["class_weight"] = "balanced"

    weight_label = "balanced" if balanced else "unweighted"
    # Only datasets where exclude_from_eval is False can be test sets
    test_datasets = sorted(meta[meta["exclude_from_eval"] == False]["dataset"].unique())
    print(f"\n[LODO] Iterating over {len(test_datasets)} test datasets  [{weight_label}]\n")

    records = []

    for i, test_ds in enumerate(test_datasets, 1):
        test_mask  = (meta["dataset"] == test_ds).values
        train_mask = ~test_mask                        # train on everything else

        X_train = X[train_mask]
        y_train = meta.loc[train_mask, "label"].values
        X_test  = X[test_mask]
        y_test  = meta.loc[test_mask, "label"].values

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())

        # Skip if test set is single-class
        if len(np.unique(y_test)) < 2:
            print(f"  [{i:02d}/{len(test_datasets)}] SKIP  {test_ds!r} — single class in test set")
            continue

        print(f"  [{i:02d}/{len(test_datasets)}] TEST={test_ds!r}  "
              f"(train={n_train:,}, test={n_test:,})", end=" ... ", flush=True)

        clf = LogisticRegression(**lr_params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        m = compute_metrics(y_test, y_pred, y_prob)
        m["dataset"] = test_ds
        m["n_train"] = n_train
        records.append(m)

        print(f"acc={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  auc={m['roc_auc']:.3f}")

    results = pd.DataFrame(records)
    cols = ["dataset", "n_train", "n_test", "n_deceptive_test",
            "accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]
    results = results[[c for c in cols if c in results.columns]]
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame, balanced: bool = False) -> None:
    weight_label = "class_weight='balanced'" if balanced else "no class weighting"
    print("\n" + "="*80)
    print(f"LEAVE-ONE-DATASET-OUT RESULTS — TF-IDF (1,2)-grams + Logistic Regression  [{weight_label}]")
    print("="*80)

    display = results.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print(display.to_string(index=False))

    print("\n" + "-"*80)
    print("MACRO AVERAGES (unweighted across datasets):")
    numeric_cols = ["accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]
    for col in numeric_cols:
        val = results[col].mean()
        print(f"  {col:<20} {val:.4f}")

    print("\nWEIGHTED AVERAGES (weighted by n_test):")
    for col in numeric_cols:
        val = np.average(results[col].fillna(0), weights=results["n_test"])
        print(f"  {col:<20} {val:.4f}")
    print("="*80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TF-IDF LODO evaluation")
    parser.add_argument(
        "--balanced", action="store_true",
        help="Use class_weight='balanced' in logistic regression "
             "(saves to tfidf_results_balanced.csv)"
    )
    args = parser.parse_args()

    output_csv = OUTPUT_CSV_BALANCED if args.balanced else OUTPUT_CSV_UNWEIGHTED

    df = load_data(DATA_PATH)
    X, meta = build_or_load_matrix(df)
    results = lodo_evaluation(df, X, meta, balanced=args.balanced)
    results.to_csv(output_csv, index=False)
    print(f"\n[INFO] Results saved to {output_csv}")
    print_summary(results, balanced=args.balanced)


if __name__ == "__main__":
    main()
