# -*- coding: utf-8 -*-
"""Recompute the Section 8.1 scaling-law fit and the same-ticker gap
distribution with STANCE-AWARE realized excursions.

Long-only target (what the paper reports):  (max High since posting)/entry - 1
Stance-aware target (what the label means):
    bullish / neutral -> (max High since posting)/entry - 1
    bearish           -> 1 - (min Low since posting)/entry

Guards mirror build_top210.py: quoted price accepted only within 15% of the
post-date close; realized bound must respect the compounded daily price limit
(+10%/day up, -10%/day down).
"""
import json
import math
from itertools import combinations
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

import numpy as np
from scipy.stats import spearmanr, pearsonr, norm

from finarg3_sm.analysis.collection_sample import build_rows

DATA = PROJECT_ROOT / "data"
S_FIT = 0.187  # reliability-model scale fitted in the paper

rows = build_rows()

print(f"sample: n={len(rows)}  (paper reports 502)")
n_bear = sum(1 for r in rows if r["stance"] < 0)
print(f"  bearish {n_bear} ({100*n_bear/len(rows):.1f}%), "
      f"neutral {sum(1 for r in rows if r['stance']==0)}, "
      f"bullish {sum(1 for r in rows if r['stance']>0)}")

x = np.array([r["x"] for r in rows])


def fit(name, y, mask=None):
    xx, yy = (x, np.array(y)) if mask is None else (x[mask], np.array(y)[mask])
    rho = spearmanr(xx, yy).statistic
    pr = pearsonr(xx, yy)
    slope = float((xx @ yy) / (xx @ xx))
    print(f"  {name:<34} n={len(xx):4d}  Spearman rho={rho:+.3f}  "
          f"Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.1e})  slope={slope:.2f}")
    return rho


print("\n=== Scaling law: sigma*sqrt(N) vs realized excursion ===")
y_long = [r["long"] for r in rows]
y_sa = [r["stance_aware"] for r in rows]
fit("long-only target (as published)", y_long)
fit("STANCE-AWARE target", y_sa)
bear = np.array([r["stance"] < 0 for r in rows])
fit("  bearish subset, long-only", y_long, bear)
fit("  bearish subset, stance-aware", y_sa, bear)
fit("  non-bearish subset (unchanged)", y_long, ~bear)
excl = np.array([r["stance"] != 0 for r in rows])
fit("excluding neutral-stance posts", y_sa, excl)
print(f"  driftless prediction for slope: sqrt(2/pi) = {math.sqrt(2/math.pi):.2f}")

# --- same-ticker gap distribution ---
print("\n=== Same-ticker |gap| distribution (feeds the Phi(gap/s)=0.55 claim) ===")
by_t = {}
for r in rows:
    by_t.setdefault(r["ticker"], []).append(r)


def gaps(key, pair_filter=None):
    g = []
    for t, group in by_t.items():
        for a, b in combinations(group, 2):
            if pair_filter and not pair_filter(a, b):
                continue
            g.append(abs(a[key] - b[key]))
    return np.array(g)


for label, key, filt in [
    ("long-only, all same-ticker pairs", "long", None),
    ("stance-aware, all same-ticker pairs", "stance_aware", None),
    ("stance-aware, SAME-stance pairs", "stance_aware",
     lambda a, b: a["stance"] == b["stance"]),
    ("stance-aware, OPPOSITE-stance pairs", "stance_aware",
     lambda a, b: a["stance"] * b["stance"] < 0),
]:
    g = gaps(key, filt)
    med = float(np.median(g))
    print(f"  {label:<38} n={len(g):6d}  median={100*med:.1f} pts  "
          f"under-2pts={100*np.mean(g < 0.02):.0f}%  -> Phi(gap/{S_FIT})={norm.cdf(med/S_FIT):.3f}")

# all cross-ticker pairs, not a per-ticker subsample: capping posts per
# ticker biases the gap distribution toward whichever posts sort first.
tickers = sorted(by_t)
cross = np.array([abs(a["stance_aware"] - b["stance_aware"])
                  for i in range(len(tickers))
                  for j in range(i + 1, len(tickers))
                  for a in by_t[tickers[i]] for b in by_t[tickers[j]]])
print(f"  cross-ticker (stance-aware, all pairs)      n={len(cross):6d}  "
      f"median={100*np.median(cross):.1f} pts  "
      f"under-2pts={100*np.mean(cross < 0.02):.0f}%")

out = {"n": len(rows), "n_bearish": int(n_bear),
       "rho_long_only": float(spearmanr(x, y_long).statistic),
       "rho_stance_aware": float(spearmanr(x, y_sa).statistic),
       "median_gap_same_ticker_long_only": float(np.median(gaps("long"))),
       "median_gap_same_ticker_stance_aware": float(np.median(gaps("stance_aware")))}
(DATA / "stance_aware_scaling.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\nsaved: data/stance_aware_scaling.json")
