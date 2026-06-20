"""
fasttext_analysis.py
--------------------
FastText (Common Crawl 300d pre-trained vectors) + Logistic Regression baseline
for the deception detection corpus.

Before running, download the pre-trained vectors:
  https://fasttext.cc/docs/en/english-vectors.html
  → crawl-300d-2M.vec.gz  (~1.5 GB)
  Place it at: data/crawl-300d-2M.vec.gz

Evaluation strategy: Leave-One-Dataset-Out (LODO)
  Same protocol as tfidf_analysis.py:
  - Each reliable dataset is held out once as the test set.
  - exclude_from_eval=True rows are excluded from test folds but included
    in training folds.

Document embedding
  Each document is represented as the mean of the FastText word vectors for
  all tokens that appear in the pre-trained vocabulary. Out-of-vocabulary (OOV)
  tokens are skipped; documents where all tokens are OOV receive a zero vector.

Outputs
  - results/fasttext_results.csv  — per-dataset metrics (same schema as tfidf_results.csv)

Usage
-----
  python3 scripts/fasttext_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import gzip
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent.parent   # project root
DATA_PATH    = ROOT / "data/combined_preprocessed.csv"
VECTORS_PATH = ROOT / "data/crawl-300d-2M.vec"
OUTPUT_CSV   = ROOT / "results/fasttext_results.csv"
CACHE_DIR    = ROOT / ".fasttext_cache"
EMBED_NPY    = CACHE_DIR / "doc_embeddings.npy"
META_CSV     = CACHE_DIR / "fasttext_meta.csv"

VECTOR_DIM   = 300
MAX_VOCAB    = 500_000       # load only the top-N vectors (sorted by frequency in .vec file)

LR_PARAMS = dict(
    C            = 1.0,
    max_iter     = 1000,
    solver       = "liblinear",
    class_weight = "balanced",
)

TOKEN_RE = re.compile(r"(?u)\b\w+\b")

# ---------------------------------------------------------------------------
# Label normalisation (identical to tfidf_analysis.py)
# ---------------------------------------------------------------------------

def normalise_label(val) -> int | None:
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
# Vector loading
# ---------------------------------------------------------------------------

def load_vectors(path: Path, max_vocab: int = MAX_VOCAB) -> dict:
    """
    Load word vectors from a .vec or .vec.gz file (word2vec text format).
    Returns a dict {word: np.ndarray of shape (300,)}.

    Only the first max_vocab entries are loaded (the file is sorted by
    descending frequency, so this keeps the most useful vocabulary).
    """
    print(f"[VECTORS] Loading up to {max_vocab:,} vectors from {path.name} …")
    vectors = {}
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
        total_in_file, dim = int(header[0]), int(header[1])
        print(f"[VECTORS] File contains {total_in_file:,} vectors of dim {dim}")

        for i, line in enumerate(f):
            if i >= max_vocab:
                break
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                vectors[word] = vec

    print(f"[VECTORS] Loaded {len(vectors):,} vectors")
    return vectors


# ---------------------------------------------------------------------------
# Document embedding
# ---------------------------------------------------------------------------

def embed_documents(texts: pd.Series, vectors: dict, dim: int = VECTOR_DIM) -> np.ndarray:
    """
    Convert each document to a 300-dim vector by averaging its word vectors.
    OOV tokens are skipped; fully-OOV documents receive a zero vector.
    """
    print(f"[EMBED] Computing embeddings for {len(texts):,} documents …")
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    oov_docs   = 0

    for i, text in enumerate(texts):
        if pd.isna(text):
            oov_docs += 1
            continue
        tokens   = TOKEN_RE.findall(text.lower())
        vecs     = [vectors[t] for t in tokens if t in vectors]
        if vecs:
            embeddings[i] = np.mean(vecs, axis=0)
        else:
            oov_docs += 1

        if (i + 1) % 10_000 == 0:
            print(f"  … {i+1:,} / {len(texts):,}")

    oov_pct = 100 * oov_docs / len(texts)
    print(f"[EMBED] Done. Fully-OOV documents: {oov_docs:,} ({oov_pct:.1f}%)")
    return embeddings


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def build_or_load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if EMBED_NPY.exists() and META_CSV.exists():
        print(f"[CACHE] Loading embeddings from {EMBED_NPY} …")
        X    = np.load(EMBED_NPY)
        meta = pd.read_csv(META_CSV)
        print(f"[CACHE] Loaded matrix {X.shape}")
        return X, meta

    CACHE_DIR.mkdir(exist_ok=True)

    if not VECTORS_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Pre-trained vectors not found at {VECTORS_PATH}\n"
            "Download crawl-300d-2M.vec from:\n"
            "  https://fasttext.cc/docs/en/english-vectors.html\n"
            "and place it at data/crawl-300d-2M.vec"
        )

    vectors = load_vectors(VECTORS_PATH, max_vocab=MAX_VOCAB)
    X       = embed_documents(df["text"], vectors, dim=VECTOR_DIM)
    del vectors  # free ~600 MB

    np.save(EMBED_NPY, X)
    meta = df[["dataset", "label", "exclude_from_eval"]].reset_index(drop=True)
    meta.to_csv(META_CSV, index=False)
    print(f"[CACHE] Saved embeddings to {CACHE_DIR}/")
    return X, meta


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy"        : round(accuracy_score(y_true, y_pred), 4),
        "f1_macro"        : round(f1_score(y_true, y_pred, average="macro",  zero_division=0), 4),
        "f1_deceptive"    : round(f1_score(y_true, y_pred, pos_label=1,      zero_division=0), 4),
        "precision"       : round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall"          : round(recall_score(y_true, y_pred, pos_label=1,   zero_division=0), 4),
        "n_test"          : len(y_true),
        "n_deceptive_test": int(sum(y_true)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        m["roc_auc"] = float("nan")
    return m


# ---------------------------------------------------------------------------
# LODO loop
# ---------------------------------------------------------------------------

def lodo_evaluation(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    test_datasets = sorted(meta[meta["exclude_from_eval"] == False]["dataset"].unique())
    print(f"\n[LODO] Iterating over {len(test_datasets)} test datasets\n")

    records = []
    for i, ds in enumerate(test_datasets, 1):
        tmask = (meta["dataset"] == ds).values
        rmask = ~tmask

        X_train = X[rmask];  y_train = meta.loc[rmask, "label"].values
        X_test  = X[tmask];  y_test  = meta.loc[tmask, "label"].values

        if len(np.unique(y_test)) < 2:
            print(f"  [{i:02d}/{len(test_datasets)}] SKIP  {ds!r} — single class in test")
            continue

        print(f"  [{i:02d}/{len(test_datasets)}] TEST={ds!r}  "
              f"(train={rmask.sum():,}, test={tmask.sum():,})", end=" ... ", flush=True)

        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        m = compute_metrics(y_test, y_pred, y_prob)
        m["dataset"] = ds
        m["n_train"] = int(rmask.sum())
        records.append(m)
        print(f"acc={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  auc={m['roc_auc']:.3f}")

    results = pd.DataFrame(records)
    cols = ["dataset", "n_train", "n_test", "n_deceptive_test",
            "accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]
    return results[[c for c in cols if c in results.columns]]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("LEAVE-ONE-DATASET-OUT — FastText CC-300d + Logistic Regression")
    print("="*80)
    print(results.sort_values("f1_macro", ascending=False).to_string(index=False))

    print("\n--- MACRO AVG (unweighted) ---")
    for c in ["accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]:
        print(f"  {c:<20} {results[c].mean():.4f}")

    print("\n--- WEIGHTED AVG (by n_test) ---")
    for c in ["accuracy", "f1_macro", "f1_deceptive", "precision", "recall", "roc_auc"]:
        v = np.average(results[c].fillna(0), weights=results["n_test"])
        print(f"  {c:<20} {v:.4f}")
    print("="*80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load corpus
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows")
    df["label"] = df["deceptive"].apply(normalise_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Build or load embeddings
    X, meta = build_or_load_embeddings(df)

    # LODO evaluation
    results = lodo_evaluation(X, meta)
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] Results saved to {OUTPUT_CSV}")
    print_summary(results)


if __name__ == "__main__":
    main()
