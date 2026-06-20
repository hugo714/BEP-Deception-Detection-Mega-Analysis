# BEP — Deception Detection Mega-Analysis

Bachelor End Project (BEP) at Eindhoven University of Technology and Tilburg University. Cross-domain deception detection: A Comparison of Text Representation Methods in Cross-Domain Deception Detection.

## Abstract

**Background.** Automated deception detection generalises poorly across text domains, and growing
evidence suggests there is no universal linguistic fingerprint for deception. It remains unclear
whether this cross-domain gap is driven by the choice of text representation, by structural properties
of the domains themselves, or by both.
**Methods.** Three text representations: TF-IDF, FastText, and Sentence-BERT were evaluated
under a leave-one-domain-out (LODO) protocol across 25 deception domains drawn from a unified
corpus of 71,512 documents. For each fold, seven domain-level features were related to transfer
difficulty (1 − AUC) via Pearson correlation and regression.
**Results. **On the full sample, TF-IDF outperformed both dense representations on ROC-AUC
(vs. FastText p = .002, d = +0.69; vs. SBERT p = .005, d = +0.66), but this advantage did not
survive a sensitivity analysis restricted to the 15 low-transfer folds, indicating it was concentrated
in domains whose siblings appear in training. FastText and SBERT were not reliably distinguished.
SBERT centroid distance was the strongest univariate predictor of transfer difficulty on the full
sample (r = +.48, p = .016), but this correlation reversed sign on the low-transfer folds (r = −.46).
No feature predicted difficulty once domain leakage was controlled for, and the joint regression
failed to generalise (LOO-CV R2 = −.47).
**Conclusions.** Frozen general-purpose sentence encoders should not be treated as a default upgrade
over lexical baselines, and no domain-dissimilarity measure can yet be recommended as a label-free
indicator of transfer difficulty. The dominant factor is whether a target domain carries a lexically
stable deceptive signal at all. Establishing this on genuinely novel domains will require larger,
leakage-free evaluation sets.

---

## Repository structure

```
.
├── data/               # NOT tracked — see Data section below
├── figures/            # Output figures (heatmaps, LODO diagram)
├── results/            # CSV outputs from all analysis scripts
├── scripts/            # All analysis and preprocessing scripts
│   ├── preprocess_corpus.py          # Deduplication, flagging, per-domain cap
│   ├── generate_corpus_stats.py      # Corpus statistics → stats/corpus_stats.json
│   ├── generate_preprocessed_stats.py
│   ├── tfidf_analysis.py             # TF-IDF + LR baseline (LODO)
│   ├── tfidf_sensitivity_check.py
│   ├── fasttext_analysis.py          # FastText embeddings (LODO)
│   ├── sbert_analysis.py             # SBERT embeddings (LODO)
│   ├── rq2_domain_dissimilarity.py   # RQ2 dissimilarity features + regression
│   ├── moe_analysis.py               # Margin of Error Analysis for 
│   ├── generate_heatmap.py           # Heatmap figures
│   ├── sensitivity_rq1.py            # RQ1 sensitivity analysis
│   └── sensitivity_rq2.py            # RQ2 sensitivity analysis
├── stats/              # Corpus statistics (JSON)
└── .gitignore
```

---

## Data

The corpus is not committed to this repository due to size (raw: ~534 MB, preprocessed: ~87 MB).

The corpus is not publicly distributed. To obtain the raw or preprocessed data, email the author at [hugo@hugonooij.eu](mailto:hugo@hugonooij.eu).

The FastText analysis additionally requires the [fastText CommonCrawl vectors](https://fasttext.cc/docs/en/english-vectors.html) (`crawl-300d-2M.vec`, ~4.3 GB) placed at `data/crawl-300d-2M.vec`.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the analysis

All scripts read from `data/combined_preprocessed.csv` and write to `results/` unless stated otherwise. Run from the repository root.

### RQ1 — Cross-domain generalisation

```bash
# TF-IDF
python3 scripts/tfidf_analysis.py

# FastText embeddings
python3 scripts/fasttext_analysis.py

# SBERT embeddings 
python3 scripts/sbert_analysis.py

```

### RQ2 — Domain dissimilarity

```bash
python3 scripts/rq2_domain_dissimilarity.py
```

Reads LODO results from `results/` and computes dissimilarity features (vocab Jaccard, JS divergence, TF-IDF centroid cosine distance, SBERT centroid cosine distance, etc.) before fitting a regression.


---

## Key results

Results CSVs live in `results/`. The main files are:

| File | Contents |
|---|---|
| `tfidf_results.csv` | Per-dataset LODO metrics for TF-IDF baseline |
| `fasttext_results.csv` | Per-dataset LODO metrics for FastText |
| `sbert_results.csv` | Per-dataset LODO metrics for SBERT |
| `rq2_correlations.csv` | Dissimilarity feature correlations with transfer difficulty |
| `rq2_regression.csv` | Regression coefficients and significance |
| `sensitivity_rq1_results.csv` | RQ1 sensitivity analysis |
| `sensitivity_rq2_results.csv` | RQ2 sensitivity analysis |

---

## Author

Hugo Nooij (2053535) - Bachelor End Project, 2025–2026
