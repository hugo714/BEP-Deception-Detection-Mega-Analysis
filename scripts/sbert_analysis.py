"""
sbert_analysis.py
-----------------
Sentence-BERT + Logistic Regression LODO baseline for the deception
detection corpus.

Quick-start
-----------
  pip install sentence-transformers scikit-learn pandas numpy
  python3 scripts/sbert_analysis.py

Evaluation strategy
-------------------
Leave-One-Dataset-Out (LODO): each eval-eligible dataset is held out once
as the test set; the model trains on all remaining data (subject to the
job-domain toggle below).

Toggle
------
Set EXCLUDE_JOB_FROM_TRAIN = True to remove 'fake job postings' and
'zeng et al 2022 job scams' from every training fold while still
evaluating on them as test sets.  The output CSV is named accordingly so
both runs can coexist.
"""

# =============================================================================
#  TOGGLE — edit here, then run
# =============================================================================

EXCLUDE_JOB_FROM_TRAIN = False   # True → ablation (no-job), False → full corpus

# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

# ---------------------------------------------------------------------------
# Config — paths, model, classifier
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data/combined_preprocessed.csv"

# SBERT model — swap to "all-MiniLM-L6-v2" for a 5× speed-up at some cost
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 64          # reduce if GPU OOM (32 is safe on 8 GB VRAM)
VECTOR_DIM = 768         # 768 for mpnet, 384 for MiniLM

JOB_DATASETS = {
    "fake job postings",
    "zeng et al 2022 job scams",
}

LR_PARAMS = dict(
    C            = 1.0,
    max_iter     = 1000,
    solver       = "liblinear",
    class_weight = "balanced",   # consistent with TF-IDF and FastText baselines
)

# Cache: one directory per model so swapping models doesn't corrupt the cache
_model_slug = MODEL_NAME.replace("/", "_").replace("-", "_")
CACHE_DIR  = ROOT / f".sbert_cache_{_model_slug}"
EMBED_NPY  = CACHE_DIR / "doc_embeddings.npy"
META_CSV   = CACHE_DIR / "meta.csv"

# Output file name reflects the toggle so both runs coexist
_suffix    = "_no_job" if EXCLUDE_JOB_FROM_TRAIN else ""
OUTPUT_CSV = ROOT / f"results/sbert_results{_suffix}.csv"

# ---------------------------------------------------------------------------
# Label normalisation (identical to tfidf/fasttext scripts)
# ---------------------------------------------------------------------------

def normalise_label(val) -> "int | None":
    if pd.isna(val):
        return None
    try:
        f = float(val)
        if f in (0.0, 1.0):
            return int(f)
    except (ValueError, TypeError):
        pass
    s = str(val).strip().lower()
    if s in {"lie", "deceptive"}:
        return 1
    if s in {"truthful"}:
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

    n_ds = df["dataset"].nunique()
    n_dec = (df["label"] == 1).sum()
    n_tru = (df["label"] == 0).sum()
    print(f"[INFO] {n_ds} datasets | {n_dec:,} deceptive / {n_tru:,} truthful")

    mode = "ABLATION (job datasets excluded from training)" if EXCLUDE_JOB_FROM_TRAIN else "FULL CORPUS"
    print(f"[INFO] Mode: {mode}")
    return df

# ---------------------------------------------------------------------------
# SBERT embeddings — compute once, cache to disk
# ---------------------------------------------------------------------------

def build_or_load_embeddings(df: pd.DataFrame) -> tuple:
    """
    Encode all documents with SBERT and cache the result.

    SBERT embeddings are computed independently per document — they do not
    depend on the train/test split — so the same cache is valid for both
    toggle modes.  Switching EXCLUDE_JOB_FROM_TRAIN does not require a
    new cache.
    """
    if EMBED_NPY.exists() and META_CSV.exists():
        print(f"[CACHE] Loading embeddings from {EMBED_NPY} …")
        X    = np.load(EMBED_NPY)
        meta = pd.read_csv(META_CSV)
        print(f"[CACHE] Loaded {X.shape[0]:,} × {X.shape[1]}-dim matrix")
        return X, meta

    print(f"[SBERT] Loading model '{MODEL_NAME}' …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    texts = df["text"].fillna("").tolist()
    print(f"[SBERT] Encoding {len(texts):,} documents  "
          f"(batch_size={BATCH_SIZE}, dim={VECTOR_DIM}) …")
    X = model.encode(
        texts,
        batch_size      = BATCH_SIZE,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = False,
    )
    print(f"[SBERT] Done. Matrix shape: {X.shape}")

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(EMBED_NPY, X)
    meta = df[["dataset", "label", "exclude_from_eval"]].reset_index(drop=True)
    meta.to_csv(META_CSV, index=False)
    print(f"[CACHE] Saved to {CACHE_DIR}/")
    return X, meta

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy"        : round(accuracy_score(y_true, y_pred), 4),
        "f1_macro"        : round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_deceptive"    : round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "precision"       : round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall"          : round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "n_test"          : len(y_true),
        "n_deceptive_test": int(sum(y_true)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        m["roc_auc"] = float("nan")
    return m

# ---------------------------------------------------------------------------
# Leave-One-Dataset-Out loop
# ---------------------------------------------------------------------------

def lodo_evaluation(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """
    For each eval-eligible dataset:
      test  = that dataset's rows
      train = all other rows, minus JOB_DATASETS if EXCLUDE_JOB_FROM_TRAIN
    """
    test_datasets = sorted(
        meta[meta["exclude_from_eval"] == False]["dataset"].unique()
    )

    # Build the persistent exclusion mask (empty if toggle is off)
    if EXCLUDE_JOB_FROM_TRAIN:
        excl_mask = meta["dataset"].isin(JOB_DATASETS).values
        print(f"\n[LODO] {len(test_datasets)} test folds  |  "
              f"excluded from training: {', '.join(sorted(JOB_DATASETS))}\n")
    else:
        excl_mask = np.zeros(len(meta), dtype=bool)
        print(f"\n[LODO] {len(test_datasets)} test folds  |  full corpus mode\n")

    records = []

    for i, test_ds in enumerate(test_datasets, 1):
        test_mask  = (meta["dataset"] == test_ds).values
        train_mask = ~test_mask & ~excl_mask

        X_train = X[train_mask];  y_train = meta.loc[train_mask, "label"].values
        X_test  = X[test_mask];   y_test  = meta.loc[test_mask,  "label"].values

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())

        if len(np.unique(y_test)) < 2:
            print(f"  [{i:02d}/{len(test_datasets)}] SKIP  {test_ds!r} — single class in test")
            continue

        tag = " [EXCL-FROM-TRAIN]" if (EXCLUDE_JOB_FROM_TRAIN and test_ds in JOB_DATASETS) else ""
        print(f"  [{i:02d}/{len(test_datasets)}] TEST={test_ds!r}{tag}  "
              f"(train={n_train:,}, test={n_test:,})", end=" ... ", flush=True)

        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        m = compute_metrics(y_test, y_pred, y_prob)
        m["dataset"] = test_ds
        m["n_train"] = n_train
        records.append(m)

        print(f"acc={m['accuracy']:.3f}  f1={m['f1_macro']:.3f}  auc={m['roc_auc']:.3f}")

    cols = ["dataset", "n_train", "n_test", "n_deceptive_test",
            "accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]
    results = pd.DataFrame(records)
    return results[[c for c in cols if c in results.columns]]

# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame) -> None:
    mode = "no-job ablation" if EXCLUDE_JOB_FROM_TRAIN else "full corpus"
    print("\n" + "=" * 80)
    print(f"LODO RESULTS — Sentence-BERT ({MODEL_NAME}) + LR  [{mode}]")
    print("=" * 80)
    print(results.sort_values("f1_macro", ascending=False).to_string(index=False))

    numeric = ["accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]
    print("\n--- MACRO AVERAGES (unweighted) ---")
    for c in numeric:
        print(f"  {c:<20} {results[c].mean():.4f}")

    print("\n--- WEIGHTED AVERAGES (by n_test) ---")
    for c in numeric:
        v = np.average(results[c].fillna(0), weights=results["n_test"])
        print(f"  {c:<20} {v:.4f}")
    print("=" * 80)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df        = load_data(DATA_PATH)
    X, meta   = build_or_load_embeddings(df)
    results   = lodo_evaluation(X, meta)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Results saved → {OUTPUT_CSV.relative_to(ROOT)}")
    print_summary(results)


if __name__ == "__main__":
    main()
