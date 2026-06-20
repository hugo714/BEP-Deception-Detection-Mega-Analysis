"""
generate_preprocessed_stats.py
--------------------------------
Generates corpus_stats_preprocessed.json from the preprocessed deception corpus.

Extends generate_corpus_stats.py with three additions specific to the
preprocessed file:

  1. exclude_from_eval awareness — stats are reported for the full corpus,
     the eval-eligible subset (exclude_from_eval=False), and the excluded
     subset (exclude_from_eval=True) separately.

  2. Mixed-type label normalisation — handles all label encodings present
     in the preprocessed file: int 0/1, float 0.0/1.0, string "0"/"1",
     "lie"/"truthful", "deceptive".

  3. LODO evaluation summary — counts test-eligible datasets, reports the
     min/max/mean test-set size, and flags any datasets that are single-class
     (which would be skipped in evaluation).

Usage
-----
    python3 scripts/generate_preprocessed_stats.py
    python3 scripts/generate_preprocessed_stats.py --input data/combined_preprocessed.csv
    python3 scripts/generate_preprocessed_stats.py --output stats/my_stats.json
    python3 scripts/generate_preprocessed_stats.py --skip-pairwise
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

DECEPTIVE_STRINGS = {"1", "1.0", "lie", "deceptive"}
TRUTHFUL_STRINGS  = {"0", "0.0", "truthful"}

def normalise_label(val) -> int | None:
    """Map any observed label encoding to 0 (truthful) or 1 (deceptive)."""
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


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows from {path.name}")

    required = {"text", "deceptive", "dataset", "exclude_from_eval"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns: {missing}")

    df["label"] = df["deceptive"].apply(normalise_label)
    n_failed = df["label"].isna().sum()
    if n_failed:
        print(f"[WARN] {n_failed:,} rows with unrecognised labels — dropped")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["exclude_from_eval"] = df["exclude_from_eval"].astype(bool)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def token_counts(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).str.split().apply(len).values

def text_length_stats(toks: np.ndarray) -> dict:
    return {
        "mean":   round(float(np.mean(toks)), 1),
        "median": round(float(np.median(toks)), 1),
        "std":    round(float(np.std(toks)), 1),
        "min":    int(np.min(toks)),
        "max":    int(np.max(toks)),
        "p25":    round(float(np.percentile(toks, 25)), 1),
        "p75":    round(float(np.percentile(toks, 75)), 1),
    }

def class_balance_tag(frac: float) -> str:
    if frac < 0.20 or frac > 0.80:
        return "severely_imbalanced"
    if frac < 0.35 or frac > 0.65:
        return "imbalanced"
    return "balanced"


# ---------------------------------------------------------------------------
# Corpus-level summary
# ---------------------------------------------------------------------------

def corpus_summary(df: pd.DataFrame) -> dict:
    n         = len(df)
    n_dec     = int((df["label"] == 1).sum())
    n_tru     = int((df["label"] == 0).sum())
    n_excl    = int(df["exclude_from_eval"].sum())
    n_eval    = n - n_excl
    toks      = token_counts(df["text"])

    return {
        "n_total":              n,
        "n_deceptive":          n_dec,
        "n_truthful":           n_tru,
        "frac_deceptive":       round(n_dec / n, 4),
        "n_eval_eligible":      n_eval,
        "n_excluded_from_eval": n_excl,
        "n_datasets_total":     int(df["dataset"].nunique()),
        "n_datasets_eval":      int(df[~df["exclude_from_eval"]]["dataset"].nunique()),
        "n_datasets_excluded":  int(df[df["exclude_from_eval"]]["dataset"].nunique()),
        "text_length":          text_length_stats(toks),
    }


# ---------------------------------------------------------------------------
# Per-dataset stats
# ---------------------------------------------------------------------------

def per_dataset_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for ds, grp in df.groupby("dataset"):
        n        = len(grp)
        n_dec    = int((grp["label"] == 1).sum())
        n_tru    = int((grp["label"] == 0).sum())
        frac     = round(n_dec / n, 4) if n > 0 else None
        toks     = token_counts(grp["text"])
        excl     = bool(grp["exclude_from_eval"].iloc[0])
        n_classes = grp["label"].nunique()

        stats[ds] = {
            "n_total":            n,
            "n_deceptive":        n_dec,
            "n_truthful":         n_tru,
            "frac_deceptive":     frac,
            "class_balance":      class_balance_tag(frac) if frac is not None else None,
            "n_classes":          n_classes,
            "single_class":       n_classes < 2,
            "exclude_from_eval":  excl,
            "text_length":        text_length_stats(toks),
            "raw_label_values":   sorted(df[df["dataset"]==ds]["deceptive"]
                                         .astype(str).unique().tolist()),
        }
    return stats


# ---------------------------------------------------------------------------
# Eval-eligible subset summary
# ---------------------------------------------------------------------------

def eval_subset_summary(df: pd.DataFrame) -> dict:
    eval_df  = df[~df["exclude_from_eval"]]
    excl_df  = df[df["exclude_from_eval"]]

    ds_sizes = eval_df.groupby("dataset").size()
    skipped  = [ds for ds, grp in eval_df.groupby("dataset")
                if grp["label"].nunique() < 2]

    return {
        "eval_eligible": {
            "n_total":       int(len(eval_df)),
            "n_deceptive":   int((eval_df["label"] == 1).sum()),
            "n_truthful":    int((eval_df["label"] == 0).sum()),
            "frac_deceptive": round((eval_df["label"] == 1).mean(), 4),
            "n_datasets":    int(eval_df["dataset"].nunique()),
            "lodo_test_set_sizes": {
                "min":    int(ds_sizes.min()),
                "max":    int(ds_sizes.max()),
                "mean":   round(float(ds_sizes.mean()), 1),
                "median": round(float(ds_sizes.median()), 1),
            },
            "single_class_datasets": skipped,
            "n_single_class_skipped": len(skipped),
        },
        "excluded_from_eval": {
            "n_total":      int(len(excl_df)),
            "n_deceptive":  int((excl_df["label"] == 1).sum()),
            "n_truthful":   int((excl_df["label"] == 0).sum()),
            "datasets":     sorted(excl_df["dataset"].unique().tolist()),
            "note": (
                "These datasets are included in training folds but never "
                "used as test sets due to reliability concerns."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Pairwise domain dissimilarity (eval-eligible datasets only)
# ---------------------------------------------------------------------------

def pairwise_dissimilarity(df: pd.DataFrame, vocab_size: int = 10_000) -> dict:
    eval_df = df[~df["exclude_from_eval"]]
    print(f"[INFO] Computing pairwise JS divergence across "
          f"{eval_df['dataset'].nunique()} eval datasets ...")

    vec = CountVectorizer(max_features=vocab_size, dtype=np.float32)
    vec.fit(eval_df["text"].fillna("").astype(str))

    domains   = sorted(eval_df["dataset"].unique())
    dists     = {}
    for ds in domains:
        texts  = eval_df[eval_df["dataset"] == ds]["text"].fillna("").astype(str)
        X      = vec.transform(texts)
        counts = np.asarray(X.sum(axis=0)).flatten().astype(float)
        counts += 1e-9
        dists[ds] = counts / counts.sum()

    pairs = {}
    for i, src in enumerate(domains):
        for tgt in domains[i + 1:]:
            js = float(jensenshannon(dists[src], dists[tgt], base=2))
            pairs[f"{src} -> {tgt}"] = round(js, 6)

    # Sort by divergence descending for easy reading
    pairs = dict(sorted(pairs.items(), key=lambda x: -x[1]))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate corpus_stats_preprocessed.json"
    )
    parser.add_argument(
        "--input", default="data/combined_preprocessed.csv",
        help="Path to preprocessed corpus CSV (default: data/combined_preprocessed.csv)"
    )
    parser.add_argument(
        "--output", default="stats/corpus_stats_preprocessed.json",
        help="Output JSON path (default: stats/corpus_stats_preprocessed.json)"
    )
    parser.add_argument(
        "--skip-pairwise", action="store_true",
        help="Skip pairwise JS divergence computation (faster)"
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    df = load(path)

    print("[INFO] Computing corpus summary ...")
    summary = corpus_summary(df)

    print("[INFO] Computing per-dataset stats ...")
    ds_stats = per_dataset_stats(df)

    print("[INFO] Computing eval/excluded subset summary ...")
    eval_summary = eval_subset_summary(df)

    pair_diss = {}
    if not args.skip_pairwise:
        pair_diss = pairwise_dissimilarity(df)

    output = {
        "generated_from":          str(path.resolve()),
        "corpus_summary":          summary,
        "eval_subset_summary":     eval_summary,
        "per_dataset_stats":       ds_stats,
        "pairwise_js_divergence":  pair_diss,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Written to: {out_path.resolve()}")
    print(f"       Total rows   : {summary['n_total']:,}")
    print(f"       Eval datasets: {summary['n_datasets_eval']}")
    print(f"       Excluded     : {summary['n_datasets_excluded']} datasets "
          f"({summary['n_excluded_from_eval']:,} rows — training only)")
    print(f"       Deceptive    : {summary['frac_deceptive']:.1%} of corpus")
    skipped = eval_summary["eval_eligible"]["single_class_datasets"]
    if skipped:
        print(f"       Skipped (single-class test set): {skipped}")


if __name__ == "__main__":
    main()
