# -*- coding: utf-8 -*-
"""Decomposition: does the transferred text model add predictive information
beyond genuinely ex-ante asset capacity?

Sample  : the W=20 eligible sample (n=502) written by exante_capacity.py.
Outcome : TFE (unchanged).
Market  : capacity_pre_20 = sigma_pre_20 * sqrt(N)   (strictly pre-post).
Text    : the manuscript's July cross-period score, unchanged:
          blend = (0.4*enc + 0.3*lr) / 0.7   (analyze_crossperiod.py), where
          enc/lr are the frozen 2019-2020 models applied to the July posts
          (data/ranking_scores.json).  Nothing is retrained or re-chosen.

Partial rank correlation (sections 2C/2D): rank the three continuous
variables; regress BOTH the target pair by OLS on [1, rank(control),
bullish, bearish]; Pearson-correlate the two residual vectors.

Decision rule for "incremental text signal", fixed before results:
  CLEAR : partial rho(text | capacity, stance) cluster CI excludes 0  AND
          equal-weight (capacity+text) concordance minus capacity-only
          concordance has a cluster CI excluding 0.
  WEAK  : partial point estimate > 0 but at least one of the two CIs
          includes 0.
  NONE  : partial point estimate <= 0.
"""
import sys
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = PROJECT_ROOT
from finarg3_sm.analysis.collection_sample import build_rows                      # noqa: E402
from finarg3_sm.analysis.exante_capacity import (B_CORR, B_PAIR, SEED, _rank,     # noqa: E402
                             cluster_boot, concordance)

OUT = HERE / "results" / "exante_capacity"
AUDIT = OUT / "post_level_exante_capacity.csv"


def stance_dummies(st):
    return (st > 0).astype(float), (st < 0).astype(float)


def partial_rank(a, b, ctrl, st):
    """Pearson of residuals of rank(a), rank(b) after OLS on [1, rank(ctrl), bull, bear]."""
    bull, bear = stance_dummies(st)
    X = np.column_stack([np.ones(len(a)), _rank(ctrl), bull, bear])
    ra, rb = _rank(a), _rank(b)
    ba, *_ = np.linalg.lstsq(X, ra, rcond=None)
    bb, *_ = np.linalg.lstsq(X, rb, rcond=None)
    return float(np.corrcoef(ra - X @ ba, rb - X @ bb)[0, 1])


def ols_fit(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, yhat, float(r2)


def std_coef(b, cols, y):
    return [float(bi * np.std(c, ddof=1) / np.std(y, ddof=1)) for bi, c in zip(b[1:], cols)]


def model_cols(name, cap, txt, st):
    bull, bear = stance_dummies(st)
    rc, rt = _rank(cap), _rank(txt)
    if name == "M1":
        return [rc, bull, bear], ["rank(capacity_pre_20)", "bullish", "bearish"]
    if name == "M2":
        return [rt, bull, bear], ["rank(text)", "bullish", "bearish"]
    if name == "M3":
        return [rc, rt, bull, bear], ["rank(capacity_pre_20)", "rank(text)", "bullish", "bearish"]
    if name == "M3x":
        return ([rc, rt, bull, bear, rc * bull, rc * bear],
                ["rank(capacity_pre_20)", "rank(text)", "bullish", "bearish",
                 "rank(capacity)*bullish", "rank(capacity)*bearish"])
    raise ValueError(name)


def model_concordance(name):
    """In-sample: fit on the (bootstrap) sample, score it, concordance with TFE."""
    def f(cap, txt, st, tfe):
        cols, _ = model_cols(name, cap, txt, st)
        _, yhat, _ = ols_fit(_rank(tfe), cols)
        return concordance(yhat, tfe)[0]
    return f


def z(v):
    return (v - v.mean()) / v.std(ddof=1)


def main():
    df = pd.read_csv(AUDIT, float_precision="round_trip", dtype={"post_date": str})
    df = df[df["sigma_pre_20"].notna()].reset_index(drop=True)
    ref = {r["pid"]: r for r in build_rows()}
    assert set(df["post_id"]) == set(ref) and len(df) == 502, "W=20 sample must be the 502 posts"
    enc = np.array([ref[p]["enc"] for p in df["post_id"]], dtype=float)
    lr = np.array([ref[p]["lr"] for p in df["post_id"]], dtype=float)
    assert not (np.isnan(enc).any() or np.isnan(lr).any()), "missing text scores"
    assert all(ref[p]["stance_aware"] == t for p, t in zip(df["post_id"], df["TFE"]))
    df["text_score"] = (0.4 * enc + 0.3 * lr) / 0.7

    tfe = df["TFE"].to_numpy(); cap = df["capacity_pre_20"].to_numpy()
    txt = df["text_score"].to_numpy(); st = df["stance"].to_numpy(dtype=float)
    tk = df["ticker"].to_numpy()
    rows = []

    def add(spec, est, boot, n=len(df), extra=None):
        med, lo, hi, nb = boot
        rows.append({"specification": spec, "n_posts": n, "n_tickers": len(set(tk)),
                     "estimate": est, "cluster_boot_median": med,
                     "cluster_boot_ci_low": lo, "cluster_boot_ci_high": hi,
                     "boot_B": nb if extra is None else extra.get("B", nb),
                     **({k: v for k, v in (extra or {}).items() if k != "B"})})

    sp = lambda a, b: spearmanr(a, b).statistic
    add("A_spearman_text_TFE", sp(txt, tfe), cluster_boot(sp, (txt, tfe), tk, B_CORR),
        extra={"scipy_p": spearmanr(txt, tfe).pvalue})
    add("B_spearman_capacity_pre20_TFE", sp(cap, tfe), cluster_boot(sp, (cap, tfe), tk, B_CORR),
        extra={"scipy_p": spearmanr(cap, tfe).pvalue})
    add("C_partial_text_TFE_given_capacity_stance", partial_rank(txt, tfe, cap, st),
        cluster_boot(partial_rank, (txt, tfe, cap, st), tk, B_CORR))
    add("D_partial_capacity_TFE_given_text_stance", partial_rank(cap, tfe, txt, st),
        cluster_boot(partial_rank, (cap, tfe, txt, st), tk, B_CORR))
    add("spearman_text_capacity", sp(txt, cap), cluster_boot(sp, (txt, cap), tk, B_CORR))

    # ---- minimal rank/linear models: coefficients, R2, in-sample concordance
    fits = {}
    for name in ("M1", "M2", "M3", "M3x"):
        cols, labels = model_cols(name, cap, txt, st)
        b, yhat, r2 = ols_fit(_rank(tfe), cols)
        fits[name] = {"r2": r2, "std_coef": dict(zip(labels, std_coef(b, cols, _rank(tfe))))}
        if name != "M3x":
            acc, npairs = concordance(yhat, tfe)
            add(f"{name}_concordance_in_sample", acc,
                cluster_boot(model_concordance(name), (cap, txt, st, tfe), tk, B_PAIR),
                extra={"r2": r2, "n_pairs": npairs, "B": B_PAIR})

    # ---- no-training scores
    zc, zt = z(_rank(cap)), z(_rank(txt))
    eq = zc + zt
    c1 = lambda s, y: concordance(s, y)[0]
    for spec, s in (("concordance_capacity_only", cap), ("concordance_text_only", txt),
                    ("concordance_equal_weight_capacity_plus_text", eq)):
        acc, npairs = concordance(s, tfe)
        add(spec, acc, cluster_boot(c1, (s, tfe), tk, B_PAIR), extra={"n_pairs": npairs, "B": B_PAIR})

    def eq_minus_cap(c, t, y):          # rebuild z-scores inside each replicate
        return concordance(z(_rank(c)) + z(_rank(t)), y)[0] - concordance(c, y)[0]
    d0 = eq_minus_cap(cap, txt, tfe)
    dboot = cluster_boot(eq_minus_cap, (cap, txt, tfe), tk, B_PAIR)
    add("delta_concordance_equal_weight_minus_capacity_only", d0, dboot, extra={"B": B_PAIR})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "text_capacity_decomposition.csv", index=False)

    # ---- verdicts (rule fixed in the module docstring)
    S = out.set_index("specification")
    C = S.loc["C_partial_text_TFE_given_capacity_stance"]
    Dd = S.loc["delta_concordance_equal_weight_minus_capacity_only"]
    Dcap = S.loc["D_partial_capacity_TFE_given_text_stance"]
    if C["estimate"] <= 0:
        verdict = "NO INCREMENTAL SIGNAL"
    elif C["cluster_boot_ci_low"] > 0 and Dd["cluster_boot_ci_low"] > 0:
        verdict = "CLEAR INCREMENTAL SIGNAL"
    else:
        verdict = "WEAK INCREMENTAL SIGNAL"
    cap_holds = Dcap["cluster_boot_ci_low"] > 0

    def L(spec, lab="rho"):
        r = S.loc[spec]
        return (f"{lab}={r['estimate']:+.3f}, cluster 95% CI "
                f"[{r['cluster_boot_ci_low']:+.3f}, {r['cluster_boot_ci_high']:+.3f}]")

    M = ["# Text vs ex-ante capacity decomposition\n",
         f"Sample: the W=20 eligible sample of the ex-ante capacity experiment, n={len(df)} posts / "
         f"{len(set(tk))} tickers; post ids asserted identical to `collection_sample.build_rows()`.",
         "Text score: the manuscript's July cross-period score, blend=(0.4·enc+0.3·lr)/0.7 of the "
         "frozen 2019–2020 encoder and lexicon-LR models (`data/ranking_scores.json`); not retrained, "
         "not re-selected. Capacity: sigma_pre_20·sqrt(N), strictly pre-post. Outcome: TFE, unchanged.",
         f"Uncertainty: ticker-cluster bootstrap, seed {SEED}, B={B_CORR} (correlations) / {B_PAIR} "
         "(concordance); median and 2.5/97.5 percentiles.\n",
         "## Post-level correlations",
         "- A. " + L("A_spearman_text_TFE"),
         "- B. " + L("B_spearman_capacity_pre20_TFE"),
         "- C. partial rho(text, TFE | capacity_pre_20, stance): " + L("C_partial_text_TFE_given_capacity_stance"),
         "- D. partial rho(capacity_pre_20, TFE | text, stance): " + L("D_partial_capacity_TFE_given_text_stance"),
         "- rho(text, capacity_pre_20): " + L("spearman_text_capacity"),
         "- Partial implementation: rank TFE, text and capacity; regress both targets by OLS on "
         "[1, rank(control), bullish, bearish]; Pearson correlation of the two residual vectors.\n",
         "## Minimal rank/linear models (descriptive, in-sample)"]
    for name, desc in (("M1", "capacity only"), ("M2", "text only"), ("M3", "combined"),
                       ("M3x", "exploratory: combined + capacity×stance")):
        f = fits[name]
        coefs = ", ".join(f"{k}={v:+.3f}" for k, v in f["std_coef"].items())
        M.append(f"- {name} ({desc}): R²={f['r2']:.3f}; standardized coefficients: {coefs}")
    M += ["\n## Pairwise concordance (unique post pairs with unequal TFE; capacity ties 0.5)",
          "- M1 capacity-only model: " + L("M1_concordance_in_sample", "acc"),
          "- M2 text-only model: " + L("M2_concordance_in_sample", "acc"),
          "- M3 learned combined model: " + L("M3_concordance_in_sample", "acc") +
          " (weights fitted on the same July outcomes; in-sample)",
          "- capacity score alone (no fitting): " + L("concordance_capacity_only", "acc"),
          "- text score alone (no fitting): " + L("concordance_text_only", "acc"),
          "- **equal-weight z(rank capacity)+z(rank text) (no fitting): " + L("concordance_equal_weight_capacity_plus_text", "acc") + "**",
          "- equal-weight minus capacity-only: " + L("delta_concordance_equal_weight_minus_capacity_only", "delta") + "\n",
          "## Verdict (rule fixed before results; see module docstring)",
          f"- Incremental text signal beyond ex-ante capacity: **{verdict}**",
          f"- Capacity remains predictive after controlling for text and stance: "
          f"**{'YES' if cap_holds else 'NO'}** (partial rho " + L("D_partial_capacity_TFE_given_text_stance") + ")",
          "\nNo manuscript edits were made."]
    (OUT / "TEXT_CAPACITY_DECOMPOSITION.md").write_text("\n".join(M), encoding="utf-8")
    print("\n".join(M))


if __name__ == "__main__":
    main()
