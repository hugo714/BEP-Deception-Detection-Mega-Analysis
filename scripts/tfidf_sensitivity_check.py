"""
tfidf_sensitivity_check.py
--------------------------
Sensitivity check: full-corpus TF-IDF fitting (Variant A, current approach)
vs fold-local refitting (Variant B) for three selected LODO folds.

Selected folds (from tfidf_results_balanced.csv):
  - Highest AUC : deceptive opinion spam       (AUC 0.9777)
  - Mid-range   : cross cultural deception      (AUC 0.6446, closest to 0.65)
  - Lowest AUC  : boulder lies and truth        (AUC 0.4718)

Run with --stage 1..5 in sequence, or --all to run everything in one go.
Each stage saves intermediate results to .sensitivity_cache/ so work is not lost.

Usage
-----
  python3 scripts/tfidf_sensitivity_check.py --stage 1   # build full-corpus vocab
  python3 scripts/tfidf_sensitivity_check.py --stage 2   # fold: deceptive opinion spam
  python3 scripts/tfidf_sensitivity_check.py --stage 3   # fold: cross cultural deception
  python3 scripts/tfidf_sensitivity_check.py --stage 4   # fold: boulder lies and truth
  python3 scripts/tfidf_sensitivity_check.py --stage 5   # compile & print final table
  python3 scripts/tfidf_sensitivity_check.py --all       # all stages in sequence
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/combined_preprocessed.csv"
CACHE_DIR = ROOT / ".tfidf_cache"
MATRIX_NPZ = CACHE_DIR / "tfidf_matrix.npz"
META_CSV   = CACHE_DIR / "tfidf_meta.csv"

SENS_CACHE = ROOT / ".sensitivity_cache"
OUTPUT_CSV = ROOT / "results/tfidf_sensitivity_check.csv"

TFIDF_PARAMS = dict(
    ngram_range   = (1, 2),
    max_features  = 100_000,
    sublinear_tf  = True,
    min_df        = 2,
    strip_accents = "unicode",
    analyzer      = "word",
    token_pattern = r"(?u)\b\w+\b",
)

LR_PARAMS = dict(
    C            = 1.0,
    max_iter     = 1000,
    solver       = "liblinear",
    class_weight = "balanced",
)

TARGET_FOLDS = [
    "deceptive opinion spam",    # highest AUC
    "cross cultural deception",  # mid-range
    "boulder lies and truth",    # lowest AUC
]

# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

DECEPTIVE_STRINGS = {"lie", "deceptive", "deceptive "}
TRUTHFUL_STRINGS  = {"truthful", "truthful "}

def normalise_label(val):
    if pd.isna(val):
        return None
    try:
        f = float(val)
        if f in (0.0, 1.0):
            return int(f)
    except (ValueError, TypeError):
        pass
    s = str(val).strip().lower()
    if s in DECEPTIVE_STRINGS:
        return 1
    if s in TRUTHFUL_STRINGS:
        return 0
    return None

def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["label"] = df["deceptive"].apply(normalise_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"]  = df["text"].fillna("")
    df = df.reset_index(drop=True)
    print(f"[INFO] Loaded {len(df):,} rows, {df['dataset'].nunique()} datasets")
    return df

def load_full_matrix():
    X    = load_npz(MATRIX_NPZ)
    meta = pd.read_csv(META_CSV)
    print(f"[CACHE] Loaded full-corpus matrix {X.shape}")
    return X, meta

def compute_metrics(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return round(float(auc), 4), round(float(f1), 4)

# ---------------------------------------------------------------------------
# Stage 1: Build & cache full-corpus vocabulary (for overlap comparison)
# ---------------------------------------------------------------------------

def stage1_build_vocab():
    """Refit TF-IDF on full corpus to capture vocabulary, save vocab set."""
    SENS_CACHE.mkdir(exist_ok=True)
    vocab_path = SENS_CACHE / "vocab_full.json"
    if vocab_path.exists():
        print(f"[STAGE 1] Vocab already cached at {vocab_path} — skipping refit")
        return

    df = load_data()
    print("[STAGE 1] Fitting TF-IDF on full corpus to capture vocabulary …")
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    vec.fit(df["text"].values)
    vocab = list(vec.vocabulary_.keys())
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"[STAGE 1] Saved {len(vocab):,}-term vocabulary to {vocab_path}")


# ---------------------------------------------------------------------------
# Stage 2–4: Run one fold
# ---------------------------------------------------------------------------

def run_fold_stage(fold_name: str, stage_num: int):
    out_path = SENS_CACHE / f"fold_{stage_num}.json"
    if out_path.exists():
        print(f"[STAGE {stage_num}] Already done — {out_path}")
        return

    vocab_path = SENS_CACHE / "vocab_full.json"
    if not vocab_path.exists():
        raise RuntimeError("Run --stage 1 first to build the full-corpus vocabulary.")

    with open(vocab_path) as f:
        vocab_A = set(json.load(f))

    df      = load_data()
    X_full, meta_full = load_full_matrix()

    if len(df) != X_full.shape[0]:
        raise ValueError(
            f"Row mismatch: df={len(df)}, matrix={X_full.shape[0]}. "
            "Delete .tfidf_cache/ and re-run tfidf_analysis.py."
        )

    test_mask  = (meta_full["dataset"] == fold_name).values
    train_mask = ~test_mask
    y_train = meta_full.loc[train_mask, "label"].values
    y_test  = meta_full.loc[test_mask,  "label"].values
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())

    print(f"\n[STAGE {stage_num}] Fold: {fold_name!r}  "
          f"train={n_train:,}  test={n_test:,}  "
          f"({100*n_test/(n_train+n_test):.1f}% of corpus)")

    # Variant A — full-corpus matrix (pre-built)
    print(f"[STAGE {stage_num}] Variant A: using cached full-corpus matrix …")
    X_train_A = X_full[train_mask]
    X_test_A  = X_full[test_mask]
    clf_A = LogisticRegression(**LR_PARAMS)
    clf_A.fit(X_train_A, y_train)
    y_pred_A = clf_A.predict(X_test_A)
    y_prob_A = clf_A.predict_proba(X_test_A)[:, 1]
    auc_A, f1_A = compute_metrics(y_test, y_pred_A, y_prob_A)
    print(f"[STAGE {stage_num}] Variant A  AUC={auc_A:.4f}  F1-macro={f1_A:.4f}")

    # Variant B — fold-local refit
    print(f"[STAGE {stage_num}] Variant B: refitting TF-IDF on training corpus only …")
    texts_train = df.loc[train_mask, "text"].values
    texts_test  = df.loc[test_mask,  "text"].values
    vec_B = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_B = vec_B.fit_transform(texts_train)
    X_test_B  = vec_B.transform(texts_test)
    vocab_B   = set(vec_B.vocabulary_.keys())
    clf_B = LogisticRegression(**LR_PARAMS)
    clf_B.fit(X_train_B, y_train)
    y_pred_B = clf_B.predict(X_test_B)
    y_prob_B = clf_B.predict_proba(X_test_B)[:, 1]
    auc_B, f1_B = compute_metrics(y_test, y_pred_B, y_prob_B)
    print(f"[STAGE {stage_num}] Variant B  AUC={auc_B:.4f}  F1-macro={f1_B:.4f}")

    # Vocab overlap: share of Variant B's vocab found in Variant A's vocab
    overlap_pct = round(len(vocab_B & vocab_A) / len(vocab_B) * 100, 2)
    print(f"[STAGE {stage_num}] Vocab overlap: {overlap_pct:.2f}%  "
          f"(|B|={len(vocab_B):,}, |A∩B|={len(vocab_B & vocab_A):,})")

    result = {
        "fold_domain"       : fold_name,
        "n_train"           : n_train,
        "n_test"            : n_test,
        "auc_full_corpus"   : auc_A,
        "auc_fold_local"    : auc_B,
        "auc_diff"          : round(abs(auc_A - auc_B), 4),
        "f1_full_corpus"    : f1_A,
        "f1_fold_local"     : f1_B,
        "f1_diff"           : round(abs(f1_A - f1_B), 4),
        "vocab_overlap_pct" : overlap_pct,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[STAGE {stage_num}] Saved to {out_path}")


# ---------------------------------------------------------------------------
# Stage 5: Compile & print final table
# ---------------------------------------------------------------------------

def stage5_compile():
    records = []
    for i, fold in zip([2, 3, 4], TARGET_FOLDS):
        path = SENS_CACHE / f"fold_{i}.json"
        if not path.exists():
            raise RuntimeError(f"Missing {path} — run stage {i} first.")
        with open(path) as f:
            records.append(json.load(f))

    cols = [
        "fold_domain",
        "auc_full_corpus", "auc_fold_local", "auc_diff",
        "f1_full_corpus",  "f1_fold_local",  "f1_diff",
        "vocab_overlap_pct",
    ]
    results = pd.DataFrame(records)[cols]
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Results saved to {OUTPUT_CSV}\n")

    print("="*95)
    print("TF-IDF SENSITIVITY CHECK — Full-corpus (Variant A) vs Fold-local (Variant B) vectoriser")
    print("="*95)
    print(results.to_string(index=False))

    max_auc_diff      = results["auc_diff"].max()
    min_vocab_overlap = results["vocab_overlap_pct"].min()

    print(f"\n{'─'*95}")
    print(f"Max AUC difference : {max_auc_diff:.4f}    (threshold: < 0.01)")
    print(f"Min vocab overlap  : {min_vocab_overlap:.2f}%  (threshold: > 99.0%)")
    print()

    if max_auc_diff < 0.01 and min_vocab_overlap > 99.0:
        print("VERDICT: Negligible impact confirmed — full-corpus fitting is defensible.")
        print(f"  All AUC diffs < 0.01 (max={max_auc_diff:.4f}), "
              f"vocab overlap > 99% (min={min_vocab_overlap:.2f}%).")
    else:
        reasons = []
        if max_auc_diff >= 0.01:
            reasons.append(f"AUC diff reached {max_auc_diff:.4f} (≥ 0.01 threshold)")
        if min_vocab_overlap <= 99.0:
            reasons.append(f"vocab overlap dropped to {min_vocab_overlap:.2f}% (≤ 99% threshold)")
        print("VERDICT: Non-negligible impact detected — report exact differences as a limitation.")
        print("  Reasons: " + "; ".join(reasons) + ".")
    print("="*95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a single stage (1–5)")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages in sequence")
    args = parser.parse_args()

    if args.all or args.stage == 1:
        stage1_build_vocab()
    if args.all or args.stage == 2:
        run_fold_stage(TARGET_FOLDS[0], 2)   # deceptive opinion spam
    if args.all or args.stage == 3:
        run_fold_stage(TARGET_FOLDS[1], 3)   # cross cultural deception
    if args.all or args.stage == 4:
        run_fold_stage(TARGET_FOLDS[2], 4)   # boulder lies and truth
    if args.all or args.stage == 5:
        stage5_compile()


if __name__ == "__main__":
    main()
