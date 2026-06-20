"""
preprocess_corpus.py
--------------------
Reads the merged corpus, applies two preprocessing steps, and writes the
result to a new file.

Steps
-----
1. Deduplicate — exact-text duplicates are removed (first occurrence kept).

2. Flag unreliable domains — adds an `exclude_from_eval` column (bool) so
   downstream code can filter without hardcoding domain names again.

3. Per-domain sample cap — each domain is capped at DOMAIN_CAP rows.
   Rows are drawn proportionally from each class so the original
   deceptive/truthful ratio is preserved.

Usage
-----
    python3 scripts/preprocess_corpus.py --input data/combined.csv
    python3 scripts/preprocess_corpus.py --input data/combined.csv --output data/combined_preprocessed.csv
    python3 scripts/preprocess_corpus.py --input data/combined.csv --cap 5000
    python3 scripts/preprocess_corpus.py --input data/combined.csv --cap 0     # disable cap
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOMAIN_CAP = 5_000  # default per-domain sample cap; override with --cap

EXCLUDE_DOMAINS: set[str] = {"pops", "real-life trial"}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(p)
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",", low_memory=False)
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(p, lines=(suffix == ".jsonl"))
    else:
        sys.exit(f"[ERROR] Unsupported format: {suffix}")
    print(f"[INFO] Loaded {len(df):,} rows from {p.name}")
    return df


def save(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(p, index=False)
    elif suffix in (".csv", ".tsv"):
        df.to_csv(p, sep="\t" if suffix == ".tsv" else ",", index=False)
    else:
        df.to_csv(p, index=False)
    print(f"[INFO] Wrote {len(df):,} rows to {p.name}")


# ---------------------------------------------------------------------------
# Step 1 — deduplicate
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[DEDUP] Removed {removed:,} duplicate rows ({len(df):,} remain)")
    else:
        print("[DEDUP] No duplicates found.")
    return df


# ---------------------------------------------------------------------------
# Step 2 — flag unreliable domains
# ---------------------------------------------------------------------------

def flag_unreliable(df: pd.DataFrame) -> pd.DataFrame:
    domain_col = "dataset" if "dataset" in df.columns else "domain"
    df = df.copy()
    df["exclude_from_eval"] = df[domain_col].isin(EXCLUDE_DOMAINS)
    for domain in EXCLUDE_DOMAINS:
        n = (df[domain_col] == domain).sum()
        if n == 0:
            print(f"[WARNING] Flagged domain {domain!r} not found in the corpus.")
        else:
            print(f"[FLAG] {domain!r} ({n:,} rows) → exclude_from_eval=True")
    return df


# ---------------------------------------------------------------------------
# Step 2 — per-domain cap with proportional class scaling
# ---------------------------------------------------------------------------

def cap_domains(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    domain_col = "dataset" if "dataset" in df.columns else "domain"
    label_col  = "deceptive" if "deceptive" in df.columns else "label"

    parts = []
    removed_total = 0

    for domain, group in df.groupby(domain_col, sort=False):
        n = len(group)
        if n <= cap:
            parts.append(group)
            continue

        # Sample each class proportionally so class balance is preserved
        sampled_classes = []
        for label_val, cls_group in group.groupby(label_col, sort=False):
            n_cls     = len(cls_group)
            n_keep    = round(cap * n_cls / n)   # proportional share of cap
            n_keep    = max(1, min(n_keep, n_cls))
            sampled_classes.append(
                cls_group.sample(n=n_keep, random_state=42)
            )

        sampled = pd.concat(sampled_classes)
        removed = n - len(sampled)
        removed_total += removed
        print(f"[CAP] {domain!r}: {n:,} → {len(sampled):,} rows ({removed:,} removed)")
        parts.append(sampled)

    result = pd.concat(parts, ignore_index=True)
    if removed_total:
        print(f"[CAP] Total removed by domain cap: {removed_total:,} rows")
    else:
        print(f"[CAP] No domain exceeded the {cap:,}-row cap.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess merged deception corpus")
    parser.add_argument("--input",  required=True,
                        help="Path to merged corpus file (csv/tsv/parquet/json/jsonl)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <input-stem>_preprocessed.<ext>)")
    parser.add_argument("--cap", type=int, default=DOMAIN_CAP, metavar="N",
                        help=f"Max samples per domain (default: {DOMAIN_CAP:,}; 0 = disabled)")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.with_stem(p.stem + "_preprocessed"))

    df = load(args.input)
    print(f"[INFO] {df['dataset'].nunique() if 'dataset' in df.columns else df['domain'].nunique()} domains, "
          f"{len(df):,} rows before preprocessing")

    print("\n--- Step 1: deduplicate ---")
    df = deduplicate(df)

    print("\n--- Step 2: flag unreliable domains ---")
    df = flag_unreliable(df)

    if args.cap > 0:
        print(f"\n--- Step 3: apply {args.cap:,}-row per-domain cap (proportional) ---")
        df = cap_domains(df, args.cap)
    else:
        print("\n[INFO] Domain cap disabled (--cap 0).")

    save(df, args.output)
    domain_col = "dataset" if "dataset" in df.columns else "domain"
    print(f"\n[DONE] {df[domain_col].nunique()} domains, {len(df):,} rows after preprocessing")
    print(f"       exclude_from_eval=True rows: {df['exclude_from_eval'].sum():,}")


if __name__ == "__main__":
    main()
