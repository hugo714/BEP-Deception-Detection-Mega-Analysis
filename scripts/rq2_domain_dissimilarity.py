"""
rq2_domain_dissimilarity.py
---------------------------
Answers RQ2: How does domain dissimilarity relate to performance
degradation, and which domain characteristics are most predictive of
transfer difficulty?

For each of the 25 LODO test folds this script computes seven dissimilarity
features between the test domain and its training corpus, then regresses
them against transfer difficulty (1 - AUC).

Dissimilarity features
----------------------
  vocab_jaccard       : |test_vocab ∩ train_vocab| / |test_vocab ∪ train_vocab|
  vocab_coverage      : |test_vocab ∩ train_vocab| / |test_vocab|
                        (proportion of test words seen during training)
  js_divergence       : Jensen-Shannon divergence of unigram distributions
  tfidf_cosine_dist   : cosine distance between TF-IDF domain centroids
  sbert_cosine_dist   : cosine distance between SBERT domain centroids
                        (Panda & Levitan 2023, metric 4 — proper implementation)
  mean_text_length    : mean token count of documents in test domain
  pct_deceptive       : proportion of deceptive labels in test domain
                        (proxy for class imbalance severity)

Relationship to Panda & Levitan (2023)
---------------------------------------
  vocab_jaccard     ≈ 1 − their metric 1 (vocabulary intersection distance)
  js_divergence     ≈ their metric 2     (JS instead of KL, more stable)
  sbert_cosine_dist = their metric 4     (SBERT centroid cosine distance)
  vocab_coverage    is our addition       (coverage asymmetry)
  tfidf_cosine_dist is our addition       (lexical centroid baseline)
  Metrics 3 (LR weights) and 5 (embedding distribution) are omitted.

Outcome
-------
  transfer_difficulty = 1 − avg_auc   (avg across available models)
  Per-model AUC columns are also kept for sensitivity analysis.

Analysis
--------
  1. Pearson correlations (r, p-value) for each feature vs outcome
  2. OLS linear regression on standardised features (LOO-CV R²)
  3. All results written to results/rq2_*.csv

Usage
-----
  python3 scripts/rq2_domain_dissimilarity.py

  The script gracefully skips any model whose result CSV is missing,
  so you can run it before all model results are available.
  SBERT centroid distances require the SBERT embedding cache to exist at
  .sbert_cache_all_mpnet_base_v2/; if absent the feature is filled with NaN.
"""

import re
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/combined_preprocessed.csv"

# Result CSVs for each model (full-corpus runs, not no-job ablation)
RESULT_FILES = {
    "tfidf"   : ROOT / "results/tfidf_results_balanced.csv",
    "fasttext": ROOT / "results/fasttext_results.csv",
    "sbert"   : ROOT / "results/sbert_results.csv",
}

# TF-IDF cache from tfidf_analysis.py — reused to avoid re-fitting
TFIDF_CACHE_DIR = ROOT / ".tfidf_cache"
TFIDF_MATRIX    = TFIDF_CACHE_DIR / "tfidf_matrix.npz"
TFIDF_META      = TFIDF_CACHE_DIR / "tfidf_meta.csv"

# SBERT cache from sbert_analysis.py (all-mpnet-base-v2)
SBERT_CACHE_DIR = ROOT / ".sbert_cache_all_mpnet_base_v2"
SBERT_EMBED_NPY = SBERT_CACHE_DIR / "doc_embeddings.npy"
SBERT_META_CSV  = SBERT_CACHE_DIR / "meta.csv"

OUTPUT_FEATURES   = ROOT / "results/rq2_features.csv"
OUTPUT_REGRESSION = ROOT / "results/rq2_regression.csv"

# Tokeniser — identical to tfidf_analysis.py
TOKEN_RE = re.compile(r"(?u)\b\w+\b")

def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower()) if pd.notna(text) else []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"[INFO] Corpus: {len(df):,} rows, {df['dataset'].nunique()} datasets")
    return df


def load_results() -> pd.DataFrame:
    """
    Load per-dataset AUC from each available model result file.
    Returns a DataFrame with columns [dataset, auc_<model>, ...].
    """
    merged = None
    for model, path in RESULT_FILES.items():
        if not path.exists():
            print(f"[WARN] {path.name} not found — skipping {model}")
            continue
        res = pd.read_csv(path)[["dataset", "roc_auc"]].rename(
            columns={"roc_auc": f"auc_{model}"}
        )
        print(f"[INFO] Loaded {model} results ({len(res)} rows)")
        merged = res if merged is None else merged.merge(res, on="dataset", how="outer")

    if merged is None:
        raise FileNotFoundError("No model result files found.")

    # Average AUC across available models
    auc_cols = [c for c in merged.columns if c.startswith("auc_")]
    merged["avg_auc"]            = merged[auc_cols].mean(axis=1)
    merged["transfer_difficulty"] = 1 - merged["avg_auc"]
    print(f"[INFO] AUC columns: {auc_cols}")
    return merged


# ---------------------------------------------------------------------------
# TF-IDF matrix (reuse cache or refit)
# ---------------------------------------------------------------------------

def get_tfidf_matrix(df: pd.DataFrame):
    """
    Load the cached TF-IDF matrix if available, otherwise refit.
    Returns (X_sparse, meta_df) aligned row-for-row with df.
    """
    from scipy.sparse import load_npz, save_npz

    if TFIDF_MATRIX.exists() and TFIDF_META.exists():
        print("[TFIDF] Loading cached matrix …")
        X    = load_npz(TFIDF_MATRIX)
        meta = pd.read_csv(TFIDF_META)
        if len(meta) == len(df):
            print(f"[TFIDF] Cache hit — shape {X.shape}")
            return X, meta
        print("[TFIDF] Cache row count mismatch — refitting")

    print("[TFIDF] Fitting TF-IDF on full corpus …")
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=100_000, sublinear_tf=True,
        min_df=2, strip_accents="unicode", analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    X    = vec.fit_transform(df["text"].fillna(""))
    meta = df[["dataset"]].reset_index(drop=True)
    TFIDF_CACHE_DIR.mkdir(exist_ok=True)
    save_npz(TFIDF_MATRIX, X)
    meta.to_csv(TFIDF_META, index=False)
    print(f"[TFIDF] Done — shape {X.shape}")
    return X, meta


def get_sbert_embeddings(df: pd.DataFrame):
    """
    Load the cached SBERT embeddings produced by sbert_analysis.py.
    Returns (X_dense, aligned_index) where aligned_index[i] is the row
    in df that corresponds to X_dense[i].

    If the cache does not exist or its row count does not match df,
    returns (None, None) and sbert_cosine_dist will be NaN for all folds.

    The SBERT meta.csv stores dataset / label / exclude_from_eval columns.
    We re-align by dataset name — both df and the cache should have been
    produced from the same combined_preprocessed.csv, so the dataset
    membership is identical; row order may differ only if the CSV was
    re-sorted.  We therefore align on positional index (same as df) which
    is the safe assumption when nothing has been shuffled.
    """
    if not (SBERT_EMBED_NPY.exists() and SBERT_META_CSV.exists()):
        print("[SBERT] Cache not found — sbert_cosine_dist will be NaN")
        return None, None

    print(f"[SBERT] Loading embeddings from {SBERT_EMBED_NPY} …")
    X    = np.load(SBERT_EMBED_NPY)
    meta = pd.read_csv(SBERT_META_CSV)

    if len(meta) != len(df):
        print(f"[SBERT] Row count mismatch ({len(meta)} vs {len(df)}) — sbert_cosine_dist will be NaN")
        return None, None

    print(f"[SBERT] Loaded {X.shape[0]:,} × {X.shape[1]}-dim matrix")
    return X, meta


# ---------------------------------------------------------------------------
# Per-fold dissimilarity features
# ---------------------------------------------------------------------------

def vocab_features(test_texts: list[str], train_texts: list[str]) -> dict:
    """Jaccard overlap and vocabulary coverage."""
    test_vocab  = set(w for t in test_texts  for w in tokenise(t))
    train_vocab = set(w for t in train_texts for w in tokenise(t))

    intersection = test_vocab & train_vocab
    union        = test_vocab | train_vocab

    jaccard   = len(intersection) / len(union)        if union        else 0.0
    coverage  = len(intersection) / len(test_vocab)   if test_vocab   else 0.0
    return {"vocab_jaccard": jaccard, "vocab_coverage": coverage}


def js_divergence_feature(test_texts: list[str], train_texts: list[str]) -> dict:
    """
    Jensen-Shannon divergence between unigram distributions.
    Uses +1 Laplace smoothing over the shared vocabulary.
    Returns JS distance (√JS divergence, range [0, 1]).
    """
    test_counts  = Counter(w for t in test_texts  for w in tokenise(t))
    train_counts = Counter(w for t in train_texts for w in tokenise(t))
    vocab = sorted(test_counts.keys() | train_counts.keys())

    # Laplace-smoothed frequency vectors
    p = np.array([test_counts.get(w, 0)  + 1 for w in vocab], dtype=float)
    q = np.array([train_counts.get(w, 0) + 1 for w in vocab], dtype=float)
    p /= p.sum();  q /= q.sum()

    js = float(jensenshannon(p, q))   # scipy returns JS distance
    return {"js_divergence": js}


def tfidf_cosine_feature(test_idx: np.ndarray, train_idx: np.ndarray,
                          X_tfidf) -> dict:
    """
    Cosine distance between the TF-IDF centroid of the test domain
    and the TF-IDF centroid of the training corpus.
    """
    test_centroid  = np.asarray(X_tfidf[test_idx].mean(axis=0)).ravel()
    train_centroid = np.asarray(X_tfidf[train_idx].mean(axis=0)).ravel()

    # Guard against zero vectors
    if test_centroid.sum() == 0 or train_centroid.sum() == 0:
        return {"tfidf_cosine_dist": float("nan")}
    return {"tfidf_cosine_dist": float(cosine_dist(test_centroid, train_centroid))}


def sbert_cosine_feature(test_idx: np.ndarray, train_idx: np.ndarray,
                          X_sbert) -> dict:
    """
    Scaled cosine distance between the SBERT centroid of the test domain
    and the SBERT centroid of the training corpus.

    Implements Panda & Levitan (2023) metric 4 exactly:
        distance = (1 - cos(S_test, S_train)) / 2
    The /2 maps the range to [0, 1] for embeddings that may be negatively
    correlated (cos in [-1, 1]). Their metric was the strongest single
    predictor (r = -0.519 with AUC).

    Returns NaN if X_sbert is None (cache unavailable).
    """
    if X_sbert is None:
        return {"sbert_cosine_dist": float("nan")}

    test_centroid  = X_sbert[test_idx].mean(axis=0)
    train_centroid = X_sbert[train_idx].mean(axis=0)

    norm_test  = np.linalg.norm(test_centroid)
    norm_train = np.linalg.norm(train_centroid)
    if norm_test == 0 or norm_train == 0:
        return {"sbert_cosine_dist": float("nan")}

    cos_sim = np.dot(test_centroid, train_centroid) / (norm_test * norm_train)
    return {"sbert_cosine_dist": float((1 - cos_sim) / 2)}


def domain_covariates(test_df: pd.DataFrame) -> dict:
    """Text length and class imbalance of the test domain."""
    lengths = test_df["text"].fillna("").apply(lambda t: len(tokenise(t)))
    pct_dec = test_df["deceptive"].apply(
        lambda v: 1 if str(v).strip().lower() in {"1", "1.0", "lie", "deceptive"}
        else 0
    ).mean()
    return {
        "mean_text_length": float(lengths.mean()),
        "pct_deceptive"   : float(pct_dec),
        "n_test"          : len(test_df),
    }


# ---------------------------------------------------------------------------
# Build full feature matrix (one row per test domain)
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, X_tfidf,
                          tfidf_meta: pd.DataFrame,
                          X_sbert=None) -> pd.DataFrame:
    """
    Iterate over all eval-eligible test domains and compute dissimilarity
    features for each LODO fold.

    X_sbert is the dense SBERT embedding matrix (rows aligned with df).
    Pass None if the cache is unavailable; sbert_cosine_dist will be NaN.
    """
    test_datasets = sorted(
        df[df["exclude_from_eval"] == False]["dataset"].unique()
    )
    sbert_avail = X_sbert is not None
    print(f"\n[FEATURES] Computing dissimilarity for {len(test_datasets)} folds …")
    print(f"[FEATURES] SBERT centroid distance: {'enabled' if sbert_avail else 'DISABLED (cache missing)'}\n")

    rows = []
    for i, test_ds in enumerate(test_datasets, 1):
        print(f"  [{i:02d}/{len(test_datasets)}] {test_ds!r}", end=" … ", flush=True)

        test_mask  = (df["dataset"] == test_ds).values
        train_mask = ~test_mask

        test_texts  = df.loc[test_mask,  "text"].tolist()
        train_texts = df.loc[train_mask, "text"].tolist()

        # Index arrays for TF-IDF and SBERT matrices (aligned with df)
        test_idx  = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        row = {"dataset": test_ds}
        row.update(vocab_features(test_texts, train_texts))
        row.update(js_divergence_feature(test_texts, train_texts))
        row.update(tfidf_cosine_feature(test_idx, train_idx, X_tfidf))
        row.update(sbert_cosine_feature(test_idx, train_idx, X_sbert))
        row.update(domain_covariates(df[test_mask].reset_index(drop=True)))

        rows.append(row)
        sbert_str = f"  sbert_cos={row['sbert_cosine_dist']:.3f}" if sbert_avail else ""
        print(
            f"jaccard={row['vocab_jaccard']:.3f}  "
            f"js={row['js_divergence']:.3f}  "
            f"tfidf_cos={row['tfidf_cosine_dist']:.3f}"
            f"{sbert_str}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regression & correlation analysis
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "vocab_jaccard",
    "vocab_coverage",
    "js_divergence",
    "tfidf_cosine_dist",
    "sbert_cosine_dist",   # Panda & Levitan (2023) metric 4
    "mean_text_length",
    "pct_deceptive",
]


def pearson_table(features_df: pd.DataFrame,
                  results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson r and two-tailed p-value for each feature vs each outcome.
    Outcomes: transfer_difficulty + per-model AUC columns.
    """
    outcome_cols = ["transfer_difficulty"] + \
                   [c for c in results_df.columns if c.startswith("auc_")]
    merged = features_df.merge(results_df, on="dataset")
    merged = merged.dropna(subset=FEATURE_COLS + outcome_cols)

    records = []
    for feat in FEATURE_COLS:
        for outcome in outcome_cols:
            r, p = pearsonr(merged[feat], merged[outcome])
            records.append({
                "feature": feat,
                "outcome": outcome,
                "pearson_r": round(r, 4),
                "p_value"  : round(p, 4),
                "sig"      : "***" if p < 0.001 else
                             "**"  if p < 0.01  else
                             "*"   if p < 0.05  else
                             "."   if p < 0.10  else "",
            })
    return pd.DataFrame(records)


def regression_analysis(features_df: pd.DataFrame,
                         results_df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS regression of transfer_difficulty on standardised features.
    Reports:
      - Standardised coefficients (β)
      - In-sample R²
      - LOO-CV R² (unbiased estimate for n=25)

    Uses Ridge regression (α=0.1) to handle potential multicollinearity
    while staying close to OLS.  Results are reported for both OLS and
    Ridge so the sensitivity can be assessed.
    """
    merged = features_df.merge(results_df[["dataset", "transfer_difficulty"]], on="dataset")
    merged = merged.dropna(subset=FEATURE_COLS + ["transfer_difficulty"])

    X_raw = merged[FEATURE_COLS].values
    y     = merged["transfer_difficulty"].values

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    records = []

    for label, model in [("OLS", LinearRegression()), ("Ridge(α=0.1)", Ridge(alpha=0.1))]:
        model.fit(X, y)
        y_pred_in = model.predict(X)
        r2_in     = r2_score(y, y_pred_in)

        loo   = LeaveOneOut()
        y_loo = cross_val_predict(model, X, y, cv=loo)
        r2_loo = r2_score(y, y_loo)

        for feat, coef in zip(FEATURE_COLS, model.coef_):
            records.append({
                "model"      : label,
                "feature"    : feat,
                "beta_std"   : round(coef, 4),
                "r2_insample": round(r2_in,  4),
                "r2_loo"     : round(r2_loo, 4),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_correlations(corr_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PEARSON CORRELATIONS WITH TRANSFER DIFFICULTY (1 − avg_AUC)")
    print("=" * 70)
    td = corr_df[corr_df["outcome"] == "transfer_difficulty"].copy()
    td = td.sort_values("pearson_r", key=abs, ascending=False)
    print(td[["feature", "pearson_r", "p_value", "sig"]].to_string(index=False))

    auc_outcomes = corr_df[corr_df["outcome"] != "transfer_difficulty"]["outcome"].unique()
    for outcome in sorted(auc_outcomes):
        sub = corr_df[corr_df["outcome"] == outcome].copy()
        sub = sub.sort_values("pearson_r", key=abs, ascending=False)
        print(f"\n  {outcome}:")
        print(sub[["feature", "pearson_r", "p_value", "sig"]].to_string(index=False))


def print_regression(reg_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION — transfer_difficulty ~ standardised features")
    print("=" * 70)
    for model_label in reg_df["model"].unique():
        sub = reg_df[reg_df["model"] == model_label]
        r2_in  = sub["r2_insample"].iloc[0]
        r2_loo = sub["r2_loo"].iloc[0]
        print(f"\n  {model_label}   R²={r2_in:.3f}   LOO-CV R²={r2_loo:.3f}")
        print("  " + "-" * 40)
        for _, row in sub.iterrows():
            bar_len = int(abs(row["beta_std"]) * 20)
            bar     = ("+" if row["beta_std"] > 0 else "−") * bar_len
            print(f"  {row['feature']:<22} β={row['beta_std']:+.4f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df      = load_corpus()
    results = load_results()

    X_tfidf, tfidf_meta = get_tfidf_matrix(df)
    X_sbert, sbert_meta = get_sbert_embeddings(df)

    features = build_feature_matrix(df, X_tfidf, tfidf_meta, X_sbert)

    # Merge features with AUC results
    combined = features.merge(results, on="dataset")
    combined.to_csv(OUTPUT_FEATURES, index=False)
    print(f"\n[INFO] Feature matrix saved → {OUTPUT_FEATURES.relative_to(ROOT)}")

    corr_df = pearson_table(features, results)
    reg_df  = regression_analysis(features, results)
    reg_df.to_csv(OUTPUT_REGRESSION, index=False)
    print(f"[INFO] Regression results saved → {OUTPUT_REGRESSION.relative_to(ROOT)}")

    print_correlations(corr_df)
    print_regression(reg_df)

    # Quick feature summary
    print("=" * 70)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 70)
    display_cols = [c for c in FEATURE_COLS if c in features.columns]
    print(features[display_cols + ["dataset"]].set_index("dataset")
          .sort_values("js_divergence", ascending=False).to_string())


if __name__ == "__main__":
    main()
