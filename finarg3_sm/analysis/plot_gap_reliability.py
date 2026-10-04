# -*- coding: utf-8 -*-
"""Clean development gap-reliability figure (paper/fig_gap_reliability_clean.*).

Circles: ensemble development accuracy in six equal-count |MPP gap| bins and
the maximum-likelihood fit Phi(|Delta|/s), both from data/reliability_fit.json.
Nothing from the July proxy reconstruction is drawn: the paper deliberately
does not substitute proxy gaps into the development curve.
"""
import json
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = PROJECT_ROOT
fit = json.loads((HERE / "data" / "reliability_fit.json").read_text(encoding="utf-8"))
s = fit["s_hat"]

fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = np.linspace(0, 0.35, 400)
ax.plot(x, norm.cdf(x / s), color="#1f77b4", lw=1.6, label=rf"$\Phi(|\Delta|/s)$, $s={s:.3f}$")
ax.scatter(fit["bin_x"], fit["bin_y"], s=28, facecolors="none", edgecolors="#1f77b4",
           lw=1.3, zorder=3, label="development accuracy, six gap bins")
ax.axhline(0.5, color="#999999", lw=0.7, ls=":")
ax.set_xlabel(r"absolute labeled MPP gap $|\Delta|$", fontsize=8)
ax.set_ylabel("pairwise accuracy", fontsize=8)
ax.set_xlim(0, 0.35)
ax.set_ylim(0.45, 1.0)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
out = HERE / "paper"
fig.savefig(out / "fig_gap_reliability_clean.pdf")
fig.savefig(out / "fig_gap_reliability_clean.png", dpi=300)
print(f"s={s:.4f}; bins x={np.round(fit['bin_x'],3).tolist()} y={np.round(fit['bin_y'],3).tolist()}")
print("wrote paper/fig_gap_reliability_clean.pdf/.png")
