# -*- coding: utf-8 -*-
"""Section 8.4: cross-period decomposition on the July-2026 collection.

Question: on 2026 data, does the text model carry signal about realized MPP
beyond what asset movement capacity already explains?

For every post with a stance-aware realized MPP, a capacity covariate and a
transferred text-model score:

    y = stance-aware realized MPP
    c = sigma * sqrt(N)                     (capacity)
    t = text-model score                    (trained on the 91 2019-2020 posts)

we report rho(c,y), rho(t,y), rho(t,c) and the partial Spearman rho(t,y|c),
computed on ranks as

    rho(t,y|c) = (r_ty - r_tc r_cy) / sqrt((1 - r_tc^2)(1 - r_cy^2))

with a t-test on df = n - 3. Rank-based throughout, to match the rank
correlations reported elsewhere in the paper.

STANCE IS A CONFOUND AND MUST BE CONTROLLED. Stance determines which formula
defines y, and in this crash-and-rebound month the two formulas have different
group means (bearish posts realize less). Any score that tracks stance -- an
LLM upside rating does so at rho=0.72 -- therefore correlates with y through
group means alone. The final block partials out stance alongside capacity;
only what survives there is evidence of per-post information.

The long-only target is reported alongside for continuity with the published
numbers, which were computed before the label's stance semantics were settled.
"""
import json
from pathlib import Path

import numpy as np
from numpy.linalg import lstsq
from scipy.stats import rankdata, spearmanr, t as tdist

from collection_sample import build_rows

DATA = Path(__file__).parent / "data"
N_BOOT = 2000
SEED = 20260822


def partial_spearman(a, b, ctrl):
    """Partial Spearman correlation of a and b controlling for ctrl."""
    ra, rb, rc = (rankdata(v) for v in (a, b, ctrl))
    r_ab = np.corrcoef(ra, rb)[0, 1]
    r_ac = np.corrcoef(ra, rc)[0, 1]
    r_bc = np.corrcoef(rb, rc)[0, 1]
    denom = ((1 - r_ac ** 2) * (1 - r_bc ** 2)) ** 0.5
    if denom == 0:
        return float("nan"), float("nan")
    rho = (r_ab - r_ac * r_bc) / denom
    n = len(ra)
    if n <= 3 or abs(rho) >= 1:
        return rho, float("nan")
    tstat = rho * ((n - 3) / (1 - rho ** 2)) ** 0.5
    return rho, 2 * tdist.sf(abs(tstat), n - 3)


def boot_ci(fn, arrays, rng):
    n = len(arrays[0])
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    rows = [r for r in build_rows()
            if r["enc"] is not None and r["lr"] is not None]
    n = len(rows)
    n_bear = sum(1 for r in rows if r["stance"] < 0)
    print(f"sample: n={n} posts with (stance-aware MPP, capacity, text score); "
          f"{n_bear} bearish ({100*n_bear/n:.0f}%)")

    y_sa = np.array([r["stance_aware"] for r in rows])
    y_lo = np.array([r["long"] for r in rows])
    c = np.array([r["x"] for r in rows])
    st = np.array([r["stance"] for r in rows])

    # enc and lr are trained on the 2019-2020 labeled posts and transferred
    # unchanged. The 0.4/0.3 weights are those of the submitted blend,
    # renormalized because that blend's third term is an LLM judgment made on
    # the 2026 post itself rather than a transferred model.
    enc = np.array([r["enc"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    blend = (0.4 * enc + 0.3 * lr) / 0.7
    llm = np.array([r["llm_upside"] if r["llm_upside"] is not None else np.nan
                    for r in rows])

    rng = np.random.default_rng(SEED)
    out = {"n": n, "n_bearish": n_bear}

    print("\n=== Capacity vs realized MPP ===")
    for name, y, key in (("stance-aware target", y_sa, "stance"),
                         ("long-only target (published)", y_lo, "long")):
        rho = spearmanr(c, y).statistic
        lo, hi = boot_ci(lambda a, b: spearmanr(a, b).statistic, (c, y), rng)
        print(f"  rho(c, y)  {name:<30} {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        out[f"rho_c_y_{key}"] = float(rho)

    print("\n=== Text score vs realized MPP, controlling for capacity ===")
    print(f"  {'text score':<34} {'rho(t,y)':>9} {'rho(t,c)':>9} "
          f"{'rho(t,y|c)':>11} {'p':>8}   95% CI of partial")
    variants = (("encoder (MacBERT durpft)", enc, "encoder"),
                ("lexicon LR", lr, "lr"),
                ("blended text model", blend, "blend"),
                ("LLM upside (2026, not transferred)", llm, "llm"))
    for label, tv, key in variants:
        ok = ~np.isnan(tv)
        tvv, yv, cv = tv[ok], y_sa[ok], c[ok]
        r_ty = spearmanr(tvv, yv).statistic
        r_tc = spearmanr(tvv, cv).statistic
        pr, pval = partial_spearman(tvv, yv, cv)
        lo, hi = boot_ci(lambda a, b, d: partial_spearman(a, b, d)[0],
                         (tvv, yv, cv), rng)
        print(f"  {label:<34} {r_ty:+9.3f} {r_tc:+9.3f} {pr:+11.3f} "
              f"{pval:8.3f}   [{lo:+.3f}, {hi:+.3f}]")
        out[f"rho_t_y_{key}"] = float(r_ty)
        out[f"partial_c_{key}"] = float(pr)
        out[f"partial_c_p_{key}"] = float(pval)

    print("\n=== Stance as a confound ===")
    print(f"  rho(LLM upside, stance) = {spearmanr(llm, st).statistic:+.3f}   "
          f"mean y: bullish {y_sa[st > 0].mean():+.3f}, "
          f"neutral {y_sa[st == 0].mean():+.3f}, bearish {y_sa[st < 0].mean():+.3f}")
    X = np.column_stack([np.ones(n), rankdata(c),
                         (st == 1).astype(float), (st == -1).astype(float)])

    def resid(v):
        b, *_ = lstsq(X, rankdata(v), rcond=None)
        return rankdata(v) - X @ b

    ry = resid(y_sa)
    for label, tv, key in variants:
        if np.isnan(tv).any():
            tv = np.where(np.isnan(tv), np.nanmedian(tv), tv)
        rho = spearmanr(resid(tv), ry).statistic
        ts = rho * ((n - 4) / (1 - rho ** 2)) ** 0.5
        p = 2 * tdist.sf(abs(ts), n - 4)
        print(f"  rho({label:<34}, y | capacity, stance) = {rho:+.3f} (p={p:.4f})")
        out[f"partial_c_stance_{key}"] = float(rho)
        out[f"partial_c_stance_p_{key}"] = float(p)

    print("\n=== Within-stance (group means removed by construction) ===")
    for s, name in ((1.0, "bullish"), (0.0, "neutral"), (-1.0, "bearish")):
        m = st == s
        if m.sum() < 30:
            continue
        for label, tv in (("blended text model", blend), ("LLM upside", llm)):
            pr, p = partial_spearman(tv[m], y_sa[m], c[m])
            print(f"  {name:<8} {label:<20} n={m.sum():3d}  "
                  f"rho={spearmanr(tv[m], y_sa[m]).statistic:+.3f}  "
                  f"partial|c={pr:+.3f} (p={p:.3f})")

    print("\n=== Long-only target, for continuity with the published numbers ===")
    r_lo = spearmanr(blend, y_lo).statistic
    pr_lo, p_lo = partial_spearman(blend, y_lo, c)
    print(f"  blended text model: rho(t,y)={r_lo:+.3f}, "
          f"rho(t,y|c)={pr_lo:+.3f} (p={p_lo:.3f})")
    print("  paper reports: -0.02 raw, +0.01 after removing sigma*sqrt(N)")
    out["rho_t_y_long_only_blend"] = float(r_lo)
    out["partial_c_long_only_blend"] = float(pr_lo)

    (DATA / "crossperiod_partial.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\nsaved: data/crossperiod_partial.json")


if __name__ == "__main__":
    main()
