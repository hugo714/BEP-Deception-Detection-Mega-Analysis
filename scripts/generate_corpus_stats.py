"""
generate_corpus_stats.py
------------------------
Generates corpus_stats.json from the unified deception corpus.

Expected schema: text | deceptive | dataset
Labels: 1 = deceptive, 0 = truthful  (column: 'deceptive')
Domain: 'dataset' column

Usage:
    python3 scripts/generate_corpus_stats.py --input data/combined.csv
    python3 scripts/generate_corpus_stats.py --input data/combined.csv --output stats/my_stats.json
    python3 scripts/generate_corpus_stats.py --input data/combined.csv --skip-pairwise
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(p)
    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, low_memory=False)
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(p, lines=(suffix == ".jsonl"))
    else:
        sys.exit(f"[ERROR] Unsupported format: {suffix}")

    # Normalise column names to internal conventions
    df = df.rename(columns={"deceptive": "label", "dataset": "domain"})

    # Normalise label values to int 0/1 regardless of source encoding
    deceptive_strings = {"1", "1.0", "lie", "deceptive", "true"}
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda v: 1 if v in deceptive_strings else 0)
    )

    print(f"[INFO] Loaded {len(df):,} rows from {p.name}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    required = {"text", "label", "domain"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing required columns (after renaming): {missing}\n"
                 f"        Found columns: {list(df.columns)}\n"
                 f"        Expected original columns: text, deceptive, dataset")
    # Warn if dataset_id column is absent (optional in this format)
    if "dataset_id" not in df.columns:
        print("[INFO] No 'dataset_id' column found — per-dataset stats will use 'domain' as the key.")
        df["dataset_id"] = df["domain"]
    print("[INFO] Schema validation passed.")


# ---------------------------------------------------------------------------
# Token count (whitespace split — no NLTK dependency required)
# ---------------------------------------------------------------------------

def token_counts(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.split().apply(len).values


# ---------------------------------------------------------------------------
# Per-domain stats
# ---------------------------------------------------------------------------

def domain_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for domain, grp in df.groupby("domain"):
        n = len(grp)
        n_deceptive = int((grp["label"] == 1).sum())
        n_truthful  = int((grp["label"] == 0).sum())
        frac_deceptive = round(n_deceptive / n, 4) if n > 0 else None

        toks = token_counts(grp["text"])
        datasets = sorted(grp["dataset_id"].unique().tolist())

        stats[domain] = {
            "n_total":        n,
            "n_deceptive":    n_deceptive,
            "n_truthful":     n_truthful,
            "frac_deceptive": frac_deceptive,
            "class_balance":  "balanced" if 0.4 <= frac_deceptive <= 0.6 else "imbalanced",
            "text_length": {
                "mean":   round(float(np.mean(toks)), 2),
                "median": round(float(np.median(toks)), 2),
                "std":    round(float(np.std(toks)), 2),
                "min":    int(np.min(toks)),
                "max":    int(np.max(toks)),
                "p25":    round(float(np.percentile(toks, 25)), 2),
                "p75":    round(float(np.percentile(toks, 75)), 2),
            },
            "datasets":       datasets,
            "n_datasets":     len(datasets),
        }
    return stats


# ---------------------------------------------------------------------------
# Per-dataset stats
# ---------------------------------------------------------------------------

def dataset_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for ds_id, grp in df.groupby("dataset_id"):
        n = len(grp)
        n_deceptive = int((grp["label"] == 1).sum())
        n_truthful  = int((grp["label"] == 0).sum())
        frac_deceptive = round(n_deceptive / n, 4) if n > 0 else None
        domains = sorted(grp["domain"].unique().tolist())

        toks = token_counts(grp["text"])

        stats[ds_id] = {
            "n_total":        n,
            "n_deceptive":    n_deceptive,
            "n_truthful":     n_truthful,
            "frac_deceptive": frac_deceptive,
            "text_length": {
                "mean":   round(float(np.mean(toks)), 2),
                "median": round(float(np.median(toks)), 2),
                "std":    round(float(np.std(toks)), 2),
            },
            "domains": domains,
        }
    return stats


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def duplicate_report(df: pd.DataFrame) -> dict:
    """
    Flags exact-text duplicates that span multiple domains — these would
    contaminate the LODO evaluation if the same text ends up in train and test.
    """
    dup_mask = df.duplicated(subset=["text"], keep=False)
    dupes = df[dup_mask][["text", "domain", "dataset_id", "label"]].copy()

    # Cross-domain duplicates: same text, different domains
    cross_domain = (
        dupes.groupby("text")["domain"]
        .nunique()
        .reset_index()
        .query("domain > 1")
    )
    n_cross = len(cross_domain)

    # Within-domain duplicates
    within_domain = (
        dupes.groupby("text")["domain"]
        .nunique()
        .reset_index()
        .query("domain == 1")
    )
    n_within = len(within_domain)

    # Label conflicts: same text but different labels
    label_conflicts = (
        dupes.groupby("text")["label"]
        .nunique()
        .reset_index()
        .query("label > 1")
    )
    n_conflicts = len(label_conflicts)

    report = {
        "total_exact_duplicates":     int(dup_mask.sum()),
        "cross_domain_duplicates":    n_cross,
        "within_domain_duplicates":   n_within,
        "label_conflicts":            n_conflicts,
        "warning": (
            "Cross-domain duplicates will contaminate LODO evaluation — "
            "deduplicate before training."
            if n_cross > 0 else None
        ),
    }

    # List affected domain pairs for cross-domain dupes (top 20)
    if n_cross > 0:
        affected = (
            dupes[dupes["text"].isin(cross_domain["text"])]
            .groupby("text")["domain"]
            .apply(lambda x: sorted(x.unique().tolist()))
            .reset_index()
        )
        pairs = defaultdict(int)
        for domains_list in affected["domain"]:
            for i in range(len(domains_list)):
                for j in range(i + 1, len(domains_list)):
                    pairs[(domains_list[i], domains_list[j])] += 1
        report["cross_domain_duplicate_pairs"] = {
            f"{a} <-> {b}": cnt
            for (a, b), cnt in sorted(pairs.items(), key=lambda x: -x[1])[:20]
        }

    return report


# ---------------------------------------------------------------------------
# Corpus-level summary
# ---------------------------------------------------------------------------

def corpus_summary(df: pd.DataFrame, domain_stats_dict: dict) -> dict:
    n_domains  = df["domain"].nunique()
    n_datasets = df["dataset_id"].nunique()
    n_total    = len(df)
    n_deceptive = int((df["label"] == 1).sum())
    n_truthful  = int((df["label"] == 0).sum())

    toks = token_counts(df["text"])

    # Class balance across domains
    frac_list = [d["frac_deceptive"] for d in domain_stats_dict.values()
                 if d["frac_deceptive"] is not None]
    balanced_domains   = sum(1 for f in frac_list if 0.4 <= f <= 0.6)
    imbalanced_domains = n_domains - balanced_domains

    # Domain sizes (for imbalance warning)
    domain_sizes = sorted(
        [(dom, d["n_total"]) for dom, d in domain_stats_dict.items()],
        key=lambda x: x[1]
    )

    return {
        "n_total":                n_total,
        "n_deceptive":            n_deceptive,
        "n_truthful":             n_truthful,
        "frac_deceptive":         round(n_deceptive / n_total, 4),
        "n_domains":              n_domains,
        "n_datasets":             n_datasets,
        "balanced_domains":       balanced_domains,
        "imbalanced_domains":     imbalanced_domains,
        "text_length_corpus": {
            "mean":   round(float(np.mean(toks)), 2),
            "median": round(float(np.median(toks)), 2),
            "std":    round(float(np.std(toks)), 2),
            "min":    int(np.min(toks)),
            "max":    int(np.max(toks)),
        },
        "smallest_domain": {"name": domain_sizes[0][0],  "n": domain_sizes[0][1]},
        "largest_domain":  {"name": domain_sizes[-1][0], "n": domain_sizes[-1][1]},
    }


# ---------------------------------------------------------------------------
# Domain-pair dissimilarity (needed later for RQ2, cheap to compute now)
# ---------------------------------------------------------------------------

def domain_pair_dissimilarity(df: pd.DataFrame, vocab_size: int = 10_000) -> dict:
    """
    Computes pairwise JS divergence and vocabulary Jaccard between domains.
    Uses top-`vocab_size` unigrams from the full corpus as the shared vocabulary.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    print("[INFO] Computing domain-pair dissimilarity metrics (this may take a moment)...")

    # Build a shared vocabulary from the full corpus
    vec = CountVectorizer(
        max_features=vocab_size,
        stop_words=None,   # keep all tokens — consistent with thesis preprocessing
        dtype=np.float32,
    )
    vec.fit(df["text"].astype(str))
    vocab = set(vec.get_feature_names_out())

    domains = sorted(df["domain"].unique())
    domain_texts = {d: df[df["domain"] == d]["text"].astype(str) for d in domains}

    # Per-domain: unigram distribution + vocabulary set
    domain_dists  = {}
    domain_vocabs = {}
    for d, texts in domain_texts.items():
        X = vec.transform(texts)
        counts = np.asarray(X.sum(axis=0)).flatten()
        counts += 1e-9   # Laplace smoothing to avoid zero-prob issues
        domain_dists[d]  = counts / counts.sum()
        domain_vocabs[d] = set(vec.get_feature_names_out()[counts > 1e-8])

    # Pairwise metrics
    pairs = {}
    for i, src in enumerate(domains):
        for tgt in domains[i + 1:]:
            js  = float(jensenshannon(domain_dists[src], domain_dists[tgt], base=2))
            v_src = domain_vocabs[src]
            v_tgt = domain_vocabs[tgt]
            jaccard = len(v_src & v_tgt) / len(v_src | v_tgt) if (v_src | v_tgt) else 0.0
            pairs[f"{src} -> {tgt}"] = {
                "js_divergence":      round(js, 6),
                "vocab_jaccard":      round(jaccard, 6),
                "vocab_overlap":      round(1 - jaccard, 6),   # dissimilarity
            }

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate corpus_stats.json")
    parser.add_argument("--input",  required=True,            help="Path to merged corpus file")
    parser.add_argument("--output", default="stats/corpus_stats.json", help="Output JSON path")
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Skip domain-pair dissimilarity (faster, skip for quick checks)")
    args = parser.parse_args()

    df = load_corpus(args.input)
    validate_schema(df)

    print("[INFO] Computing per-domain stats...")
    dom_stats = domain_stats(df)

    print("[INFO] Computing per-dataset stats...")
    ds_stats = dataset_stats(df)

    print("[INFO] Checking for duplicates...")
    dup_report = duplicate_report(df)
    if dup_report["cross_domain_duplicates"] > 0:
        print(f"[WARNING] {dup_report['cross_domain_duplicates']} cross-domain duplicates found! "
              f"See 'duplicate_report' in output for details.")

    print("[INFO] Computing corpus-level summary...")
    summary = corpus_summary(df, dom_stats)

    pair_diss = {}
    if not args.skip_pairwise:
        try:
            pair_diss = domain_pair_dissimilarity(df)
        except ImportError:
            print("[WARNING] scikit-learn not found — skipping pairwise dissimilarity. "
                  "Install with: pip install scikit-learn")

    output = {
        "corpus_summary":            summary,
        "domain_stats":              dom_stats,
        "dataset_stats":             ds_stats,
        "duplicate_report":          dup_report,
        "domain_pair_dissimilarity": pair_diss,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] corpus_stats.json written to: {out_path.resolve()}")
    print(f"       Domains:  {summary['n_domains']}")
    print(f"       Datasets: {summary['n_datasets']}")
    print(f"       Samples:  {summary['n_total']:,}")
    print(f"       Deceptive fraction: {summary['frac_deceptive']:.1%}")
    if dup_report["cross_domain_duplicates"] > 0:
        print(f"\n[!] ACTION REQUIRED: {dup_report['cross_domain_duplicates']} "
              f"cross-domain duplicates must be resolved before LODO training.")


if __name__ == "__main__":
    main()
