"""
moe_analysis.py
---------------
Computes margin-of-error tiers for each domain in corpus_stats.json.

Usage
-----
  python3 scripts/moe_analysis.py
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # project root

with open(ROOT / "stats/corpus_stats.json") as f:
    stats = json.load(f)

rows = []
for domain, d in stats["domain_stats"].items():
    n_total = d["n_total"]
    frac_dec = d["frac_deceptive"]
    n_minority = int(min(frac_dec, 1 - frac_dec) * n_total)

    # Margin of error at F1=0.70, 95% CI
    f1 = 0.70
    moe = 1.96 * (f1 * (1 - f1) / n_minority) ** 0.5 if n_minority > 0 else 999

    rows.append({
        "domain":     domain,
        "n_total":    n_total,
        "n_minority": n_minority,
        "moe_95":     round(moe, 3),
        "tier":       "reliable" if n_minority >= 100
                      else "marginal" if n_minority >= 50
                      else "unreliable"
    })

df = pd.DataFrame(rows).sort_values("n_minority")
print(df.to_string(index=False))
df.to_csv(ROOT / "results/moe_analysis.csv", index=False)
