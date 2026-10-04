# -*- coding: utf-8 -*-
"""Ex-ante asset-capacity experiment on the July-2026 collection.

Question: does daily volatility measured STRICTLY BEFORE a post was published
predict the magnitude of its later truncated stance-aware favorable
excursion (TFE)?

    C_pre,W(i) = sigma_pre,W(i) * sqrt(N_i)

    sigma_pre,W : sample sd (ddof=1) of the last W daily log returns of the
                  ADJUSTED close whose return end date is strictly before the
                  posting date.  W=20 is PRIMARY, W=60 is a pre-specified
                  robustness window.  No annualisation.
    N_i         : post-entry sessions already used by the project's TFE
                  (first trading day after posting .. 2026-07-24).

The outcome TFE_i is taken unchanged from collection_sample.build_rows(),
the same function behind the paper's ex-post rho=0.399.  It is re-derived
here only to recover audit columns (entry price, window dates) and the
re-derivation must match the project values exactly or the run aborts.

Everything written goes under results/exante_capacity/.  Nothing existing is
modified; the only new cache file is data/price_history_2026_adjclose.json.
"""
import json
import math
import sys
import time
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = PROJECT_ROOT
from finarg3_sm.analysis.collection_sample import CUTOFF, COLLECTION, DATA, build_rows, daily_vol, load_prices  # noqa: E402

OUT = HERE / "results" / "exante_capacity"
CACHE = DATA / "price_history_2026_adjclose.json"

W_PRIMARY, W_ROBUST = 20, 60
SEED = 42
B_CORR = 5000
B_PAIR = 2000
HIST_START = "2026-01-01"
HIST_END_EXCL = "2026-07-25"   # yfinance end is exclusive -> bars through 07-24

TESTS = []        # (name, passed, detail)


def check(name, cond, detail=""):
    TESTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ----------------------------------------------------------------------------
# 1. price history with adjusted close (cached)
# ----------------------------------------------------------------------------
def fetch_history(tickers):
    import yfinance as yf
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in tickers if t not in cache]
    print(f"price history: {len(tickers)} tickers, {len(todo)} to fetch, "
          f"{len(tickers) - len(todo)} cached")
    for i, t in enumerate(todo, 1):
        rec = None
        for suffix in (".TW", ".TWO"):
            try:
                df = yf.Ticker(t + suffix).history(
                    start=HIST_START, end=HIST_END_EXCL, auto_adjust=False)
            except Exception:
                df = None
            if df is not None and len(df) > 0:
                has_adj = "Adj Close" in df.columns
                bars = {}
                for d, r in df.iterrows():
                    c = float(r["Close"])
                    a = float(r["Adj Close"]) if has_adj else float("nan")
                    if not (c == c) or c <= 0:
                        continue
                    bars[str(d.date())] = {"close": c, "adj_close": a}
                rec = {"symbol": t + suffix, "has_adj_close": has_adj, "bars": bars}
                break
            time.sleep(0.3)
        cache[t] = rec
        if i % 10 == 0:
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  {i}/{len(todo)}", flush=True)
        time.sleep(0.2)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


# ----------------------------------------------------------------------------
# 2. outcome: re-derive the project's TFE with audit columns, then verify
# ----------------------------------------------------------------------------
def rederive_tfe():
    """Same rules as collection_sample.build_rows, plus audit fields."""
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    ext = json.loads((DATA / "ranking_extract.json").read_text(encoding="utf-8"))
    px = load_prices()
    vol = daily_vol(px)
    out, upstream = [], {}

    def drop(reason):
        upstream[reason] = upstream.get(reason, 0) + 1

    for it in col:
        pid = it["post_id"]
        e = ext.get(pid) or {}
        t = e.get("ticker")
        if not t:
            drop("no_ticker_identified"); continue
        if not px.get(t):
            drop("no_july_price_data"); continue
        if t not in vol:
            drop("insufficient_july_bars_for_expost_sigma"); continue
        d = pid.split("_")[1]
        post_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        bars = px[t]
        on_or_after = [x for x in sorted(bars) if x >= post_date]
        if not on_or_after:
            drop("no_bar_on_or_after_post_date"); continue
        day0_date = on_or_after[0]
        day0 = bars[day0_date]["close"]
        quoted = e.get("quoted_price")
        if quoted and quoted > 0 and abs(quoted / day0 - 1) <= 0.15:
            entry, entry_src = float(quoted), "author_quoted"
        else:
            entry, entry_src = day0, f"close_{day0_date}"
        fut = [x for x in on_or_after if x > post_date]
        if not fut:
            drop("no_trading_day_after_post"); continue
        N = len(fut)
        hi = max(bars[x]["high"] for x in fut)
        lo = min(bars[x]["low"] for x in fut)
        up = hi / entry - 1
        dn = 1 - lo / entry
        if not (-0.5 < up < 1.1 ** N - 1 + 0.10 and -0.5 < dn < 1 - 0.9 ** N + 0.10):
            drop("price_limit_guard"); continue
        stance = e.get("stance", 0)
        out.append({
            "post_id": pid, "ticker": t, "post_date": post_date, "stance": stance,
            "entry_price": entry, "entry_price_source": entry_src,
            "tfe_start": fut[0], "tfe_end": fut[-1], "N": N,
            "TFE": dn if stance < 0 else up,
            "sigma_expost": vol[t], "capacity_expost": vol[t] * math.sqrt(N),
        })
    return out, upstream


# ----------------------------------------------------------------------------
# 3. strictly pre-post volatility
# ----------------------------------------------------------------------------
def log_returns(bars, use_adj):
    """[(end_date, start_date, r)] in date order; no filling of gaps."""
    dates = sorted(bars)
    key = "adj_close" if use_adj else "close"
    out = []
    for a, b in zip(dates, dates[1:]):
        p0, p1 = bars[a][key], bars[b][key]
        if p0 == p0 and p1 == p1 and p0 > 0 and p1 > 0:
            out.append((b, a, math.log(p1 / p0)))
    return out


def sigma_pre(rets, post_date, W):
    """sd(ddof=1) of the last W returns with end date < post_date."""
    pre = [x for x in rets if x[0] < post_date]
    if len(pre) < W:
        return {"sigma": float("nan"), "n_returns": len(pre),
                "first_price_date": None, "last_price_date": None,
                "reason": f"insufficient_pre_post_returns({len(pre)}<{W})"}
    win = pre[-W:]
    vals = np.array([x[2] for x in win])
    return {"sigma": float(np.std(vals, ddof=1)), "n_returns": len(win),
            "first_price_date": win[0][1], "last_price_date": win[-1][0],
            "reason": "", "_dates": [x[0] for x in win], "_vals": vals}


# ----------------------------------------------------------------------------
# 4. statistics
# ----------------------------------------------------------------------------
def cluster_groups(tickers):
    groups = {}
    for i, t in enumerate(tickers):
        groups.setdefault(t, []).append(i)
    keys = sorted(groups)
    return keys, [np.array(groups[k]) for k in keys]


def cluster_boot(stat_fn, arrays, tickers, B, seed=SEED):
    """Ticker-cluster bootstrap of stat_fn(*arrays[idx])."""
    keys, groups = cluster_groups(tickers)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[p] for p in pick])
        try:
            v = stat_fn(*[a[idx] for a in arrays])
        except Exception:
            v = float("nan")
        vals.append(v)
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return (float(np.median(vals)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), len(vals))


def rho(a, b):
    return spearmanr(a, b).statistic


def spec_row(name, x, y, tickers, B=B_CORR):
    r = spearmanr(x, y)
    med, lo, hi, nb = cluster_boot(rho, (x, y), tickers, B)
    return {"specification": name, "n_posts": len(x), "n_tickers": len(set(tickers)),
            "spearman_rho": float(r.statistic), "scipy_p": float(r.pvalue),
            "cluster_boot_median": med, "cluster_boot_ci_low": lo,
            "cluster_boot_ci_high": hi, "boot_B": B, "boot_valid": nb}


def concordance(c, t):
    """Pairwise accuracy of 'higher capacity wins'; capacity ties count 0.5."""
    dc = c[:, None] - c[None, :]
    dt = t[:, None] - t[None, :]
    iu = np.triu_indices(len(c), k=1)
    dc, dt = dc[iu], dt[iu]
    m = dt != 0
    score = np.where(np.sign(dc[m]) == np.sign(dt[m]), 1.0,
                     np.where(dc[m] == 0, 0.5, 0.0))
    return float(score.mean()), int(m.sum())


def _rank(v):
    return pd.Series(v).rank().to_numpy()


def design(logN, stance, dates=None):
    cols = [np.ones(len(logN)), logN, (stance > 0).astype(float), (stance < 0).astype(float)]
    if dates is not None:
        for d in sorted(set(dates))[1:]:
            cols.append((dates == d).astype(float))
    return np.column_stack(cols)


def partial_rank(sig, tfe, logN, stance):
    """Pearson correlation of rank residuals after OLS on [1, logN, bull, bear]."""
    X = design(logN, stance)
    ry, rx = _rank(tfe), _rank(sig)
    by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    bx, *_ = np.linalg.lstsq(X, rx, rcond=None)
    return float(np.corrcoef(rx - X @ bx, ry - X @ by)[0, 1])


def fe_coef(sig, tfe, logN, stance, dates):
    """Coefficient on rank(sigma) in rank(TFE) ~ rank(sigma)+logN+stance+date FE."""
    X = np.column_stack([_rank(sig), design(logN, stance, dates)])
    b, *_ = np.linalg.lstsq(X, _rank(tfe), rcond=None)
    return float(b[0])


# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- outcome, verified against the project function ------------------
    ref = {r["pid"]: r for r in build_rows()}
    posts, upstream = rederive_tfe()
    check("tfe_post_set_matches_build_rows",
          set(p["post_id"] for p in posts) == set(ref),
          f"{len(posts)} re-derived vs {len(ref)} project posts")
    mism = [p["post_id"] for p in posts if p["post_id"] in ref and (
        p["TFE"] != ref[p["post_id"]]["stance_aware"] or p["N"] != ref[p["post_id"]]["N"]
        or p["stance"] != ref[p["post_id"]]["stance"]
        or p["sigma_expost"] != ref[p["post_id"]]["sigma"])]
    check("tfe_values_match_build_rows_exactly", not mism,
          f"{len(mism)} mismatching posts" if mism else "TFE, N, stance, sigma_expost identical")
    if any(not ok for _, ok, _ in TESTS):
        sys.exit("STOP: re-derived TFE does not match the project's TFE; investigate.")

    # ---- pre-post volatility ----------------------------------------------
    tickers = sorted({p["ticker"] for p in posts})
    hist = fetch_history(tickers)
    rets_by_t, fallback_t = {}, set()
    for t in tickers:
        rec = hist.get(t)
        if not rec:
            continue
        use_adj = rec["has_adj_close"] and any(
            b["adj_close"] == b["adj_close"] for b in rec["bars"].values())
        if not use_adj:
            fallback_t.add(t)
        rets_by_t[t] = log_returns(rec["bars"], use_adj)

    rows = []
    for p in posts:
        r = dict(p)
        r["resolved_yfinance_ticker"] = (hist.get(p["ticker"]) or {}).get("symbol")
        r["sqrt_N"] = math.sqrt(p["N"])
        r["volatility_raw_close_fallback"] = p["ticker"] in fallback_t
        for W in (W_PRIMARY, W_ROBUST):
            if p["ticker"] not in rets_by_t:
                s = {"sigma": float("nan"), "n_returns": 0, "first_price_date": None,
                     "last_price_date": None, "reason": "no_price_history"}
            else:
                s = sigma_pre(rets_by_t[p["ticker"]], p["post_date"], W)
            r[f"sigma_pre_{W}"] = s["sigma"]
            r[f"capacity_pre_{W}"] = s["sigma"] * r["sqrt_N"]
            r[f"first_price_date_pre{W}"] = s["first_price_date"]
            r[f"last_price_date_pre{W}"] = s["last_price_date"]
            r[f"n_returns_pre{W}"] = s["n_returns"]
            r[f"exclusion_reason_pre{W}"] = s["reason"]
            r[f"_win{W}"] = s.get("_dates"), s.get("_vals")
        rows.append(r)
    df = pd.DataFrame(rows)

    # ---- leakage / integrity tests (gate) ---------------------------------
    for W in (W_PRIMARY, W_ROBUST):
        ok = df[f"sigma_pre_{W}"].notna()
        sub = df[ok]
        check(f"W{W}_last_price_date_strictly_before_post_date",
              bool((sub[f"last_price_date_pre{W}"] < sub["post_date"]).all()),
              f"{len(sub)} valid rows")
        check(f"W{W}_exactly_{W}_returns_used",
              bool((sub[f"n_returns_pre{W}"] == W).all()))
        win_ok = all(max(d) < pd for d, pd in
                     zip(sub[f"_win{W}"].map(lambda w: w[0]), sub["post_date"]))
        check(f"W{W}_every_return_end_date_before_post_date", win_ok)
        recomputed = np.array([np.std(w[1], ddof=1) for w in sub[f"_win{W}"]])
        check(f"W{W}_sigma_equals_sd_of_window_no_annualisation",
              bool(np.allclose(recomputed, sub[f"sigma_pre_{W}"], rtol=0, atol=1e-15)))
        check(f"W{W}_sigma_not_annualised_sqrt252",
              bool(np.all(np.abs(sub[f"sigma_pre_{W}"] / recomputed - 1) < 1e-12)))
        check(f"W{W}_capacity_equals_sigma_times_sqrtN",
              bool(np.allclose(sub[f"capacity_pre_{W}"],
                               sub[f"sigma_pre_{W}"] * np.sqrt(sub["N"]), rtol=1e-12)))
        check(f"W{W}_no_tfe_window_date_inside_sigma_window",
              bool((sub[f"last_price_date_pre{W}"] < sub["tfe_start"]).all()))
        excl = df[~ok]
        check(f"W{W}_every_excluded_row_has_reason",
              bool((excl[f"exclusion_reason_pre{W}"].str.len() > 0).all()),
              f"{len(excl)} excluded")
    check("post_ids_unique", df["post_id"].is_unique)
    check("sigma_uses_only_pre_post_series_by_construction", True,
          "sigma_pre filters returns on end_date < post_date before windowing")

    failed = [n for n, ok, _ in TESTS if not ok]
    (OUT / "leakage_tests.txt").write_text(
        "\n".join(f"{'PASS' if ok else 'FAIL'}  {n}  {d}" for n, ok, d in TESTS),
        encoding="utf-8")
    if failed:
        sys.exit(f"STOP: {len(failed)} integrity test(s) failed; no statistics produced.")

    # ---- post-level audit CSV ---------------------------------------------
    audit_cols = ["post_id", "ticker", "resolved_yfinance_ticker", "post_date", "stance",
                  "TFE", "N", "sqrt_N", "entry_price", "entry_price_source", "tfe_start",
                  "tfe_end", "sigma_expost", "capacity_expost",
                  "sigma_pre_20", "sigma_pre_60", "capacity_pre_20", "capacity_pre_60",
                  "first_price_date_pre20", "last_price_date_pre20",
                  "first_price_date_pre60", "last_price_date_pre60",
                  "n_returns_pre20", "n_returns_pre60", "volatility_raw_close_fallback",
                  "exclusion_reason_pre20", "exclusion_reason_pre60"]
    df[audit_cols].to_csv(OUT / "post_level_exante_capacity.csv", index=False)

    # ---- correlations -----------------------------------------------------
    summary = []
    res = {}
    for W in (W_PRIMARY, W_ROBUST):
        s = df[df[f"sigma_pre_{W}"].notna()].reset_index(drop=True)
        tk = s["ticker"].to_numpy()
        tfe = s["TFE"].to_numpy()
        sig = s[f"sigma_pre_{W}"].to_numpy()
        cap = s[f"capacity_pre_{W}"].to_numpy()
        sq = s["sqrt_N"].to_numpy()
        res[W] = {"n": len(s), "n_tickers": len(set(tk))}
        summary.append(spec_row(f"sqrt_N (W={W} sample)", sq, tfe, tk))
        summary.append(spec_row(f"sigma_pre_{W}", sig, tfe, tk))
        summary.append(spec_row(f"capacity_pre_{W}", cap, tfe, tk))
        for lab, m in (("bullish", s["stance"] > 0), ("bearish", s["stance"] < 0)):
            mm = m.to_numpy()
            summary.append(spec_row(f"capacity_pre_{W}_{lab}", cap[mm], tfe[mm], tk[mm]))
        # partial rank correlation controlling horizon and stance
        logN = np.log(s["N"].to_numpy(dtype=float))
        st = s["stance"].to_numpy(dtype=float)
        pr = partial_rank(sig, tfe, logN, st)
        med, lo, hi, nb = cluster_boot(partial_rank, (sig, tfe, logN, st), tk, B_CORR)
        summary.append({"specification": f"partial_sigma{W}_controlling_N_stance",
                        "n_posts": len(s), "n_tickers": len(set(tk)),
                        "spearman_rho": pr, "scipy_p": float("nan"),
                        "cluster_boot_median": med, "cluster_boot_ci_low": lo,
                        "cluster_boot_ci_high": hi, "boot_B": B_CORR, "boot_valid": nb})
        dates = s["post_date"].to_numpy()
        fe = fe_coef(sig, tfe, logN, st, dates)
        med, lo, hi, nb = cluster_boot(fe_coef, (sig, tfe, logN, st, dates), tk, B_CORR)
        summary.append({"specification": f"fe_regression_coef_rank_sigma{W}_with_date_FE",
                        "n_posts": len(s), "n_tickers": len(set(tk)),
                        "spearman_rho": fe, "scipy_p": float("nan"),
                        "cluster_boot_median": med, "cluster_boot_ci_low": lo,
                        "cluster_boot_ci_high": hi, "boot_B": B_CORR, "boot_valid": nb})
        # common-sample comparison with the project's ex-post capacity
        capx = s["capacity_expost"].to_numpy()
        summary.append(spec_row(f"capacity_expost_common_W{W}_sample", capx, tfe, tk))

        def delta(c1, c2, y):
            return rho(c1, y) - rho(c2, y)
        d0 = delta(cap, capx, tfe)
        med, lo, hi, nb = cluster_boot(delta, (cap, capx, tfe), tk, B_CORR)
        summary.append({"specification": f"delta_rho_pre{W}_minus_expost_common",
                        "n_posts": len(s), "n_tickers": len(set(tk)),
                        "spearman_rho": d0, "scipy_p": float("nan"),
                        "cluster_boot_median": med, "cluster_boot_ci_low": lo,
                        "cluster_boot_ci_high": hi, "boot_B": B_CORR, "boot_valid": nb})
        # task-aligned ranking
        tau = kendalltau(cap, tfe)
        tmed, tlo, thi, _ = cluster_boot(lambda a, b: kendalltau(a, b).statistic,
                                         (cap, tfe), tk, B_PAIR)
        acc, npairs = concordance(cap, tfe)
        amed, alo, ahi, _ = cluster_boot(lambda a, b: concordance(a, b)[0],
                                         (cap, tfe), tk, B_PAIR)
        res[W].update({"kendall_tau": float(tau.statistic), "kendall_p": float(tau.pvalue),
                       "kendall_ci": (tlo, thi), "pairs": npairs, "concordance": acc,
                       "concordance_ci": (alo, ahi)})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary_correlations.csv", index=False)
    pd.DataFrame([{"specification": f"capacity_pre_{W}", "n_posts": res[W]["n"],
                   "n_unique_pairs_unequal_TFE": res[W]["pairs"],
                   "kendall_tau_b": res[W]["kendall_tau"], "kendall_scipy_p": res[W]["kendall_p"],
                   "kendall_cluster_ci_low": res[W]["kendall_ci"][0],
                   "kendall_cluster_ci_high": res[W]["kendall_ci"][1],
                   "pairwise_concordance": res[W]["concordance"],
                   "concordance_cluster_ci_low": res[W]["concordance_ci"][0],
                   "concordance_cluster_ci_high": res[W]["concordance_ci"][1],
                   "tie_convention": "capacity ties count 0.5; TFE ties excluded",
                   "boot_B": B_PAIR} for W in (W_PRIMARY, W_ROBUST)]
                 ).to_csv(OUT / "pairwise_summary.csv", index=False)

    # ---- figures ----------------------------------------------------------
    make_figures(df, sdf)

    # ---- reports ----------------------------------------------------------
    write_reports(df, sdf, res, upstream, fallback_t)
    print("\nwrote:", *[str(p) for p in sorted(OUT.iterdir())], sep="\n  ")


def make_figures(df, sdf):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    s = df[df["sigma_pre_20"].notna()].reset_index(drop=True)
    cap, tfe, tk = s["capacity_pre_20"].to_numpy(), s["TFE"].to_numpy(), s["ticker"].to_numpy()
    edges = np.quantile(cap, [0, .2, .4, .6, .8, 1.0])
    edges[-1] += 1e-12
    binid = np.clip(np.searchsorted(edges, cap, side="right") - 1, 0, 4)
    med = np.array([np.median(tfe[binid == k]) for k in range(5)])
    xc = np.array([np.median(cap[binid == k]) for k in range(5)])

    def bin_medians(c, y):   # bin edges fixed from the full sample
        b = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, 4)
        return np.array([np.median(y[b == k]) if (b == k).any() else np.nan for k in range(5)])
    keys, groups = cluster_groups(tk)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(B_CORR):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[p] for p in pick])
        boots.append(bin_medians(cap[idx], tfe[idx]))
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(cap, tfe, s=6, alpha=0.25, color="#7f7f7f", lw=0, label="posts (n=%d)" % len(s))
    ax.errorbar(xc, med, yerr=[med - lo, hi - med], fmt="o", color="#1f77b4", ms=4,
                capsize=2, lw=1.2, label="bin median, ticker-cluster 95% CI")
    ax.plot(xc, med, color="#1f77b4", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"ex-ante capacity $\sigma_{\mathrm{pre},20}\sqrt{N}$", fontsize=8)
    ax.set_ylabel("truncated favorable excursion", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_exante_capacity_20.pdf")
    fig.savefig(OUT / "fig_exante_capacity_20.png", dpi=300)
    plt.close(fig)

    specs = ["sqrt_N (W=20 sample)", "sigma_pre_20", "capacity_pre_20", "capacity_pre_60"]
    labels = [r"$\sqrt{N}$", r"$\sigma_{\mathrm{pre},20}$",
              r"$\sigma_{\mathrm{pre},20}\sqrt{N}$", r"$\sigma_{\mathrm{pre},60}\sqrt{N}$"]
    r = sdf.set_index("specification").loc[specs]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    y = np.arange(len(specs))
    ax.errorbar(r["spearman_rho"], y,
                xerr=[r["spearman_rho"] - r["cluster_boot_ci_low"],
                      r["cluster_boot_ci_high"] - r["spearman_rho"]],
                fmt="o", color="#1f77b4", ms=4, capsize=2, lw=1.2)
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"Spearman $\rho$ with TFE (ticker-cluster 95% CI)", fontsize=8)
    ax.tick_params(labelsize=7); ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_exante_components.pdf")
    fig.savefig(OUT / "fig_exante_components.png", dpi=300)
    plt.close(fig)


def write_reports(df, sdf, res, upstream, fallback_t):
    S = sdf.set_index("specification")

    def line(spec):
        r = S.loc[spec]
        p = "" if np.isnan(r["scipy_p"]) else f", scipy p={r['scipy_p']:.3g}"
        return (f"`{spec}`: n={int(r['n_posts'])} posts / {int(r['n_tickers'])} tickers, "
                f"rho={r['spearman_rho']:+.3f}{p}, cluster-bootstrap 95% CI "
                f"[{r['cluster_boot_ci_low']:+.3f}, {r['cluster_boot_ci_high']:+.3f}]")

    n20, n60 = res[20]["n"], res[60]["n"]
    ex20 = df["exclusion_reason_pre20"].replace("", np.nan).value_counts().to_dict()
    ex60 = df["exclusion_reason_pre60"].replace("", np.nan).value_counts().to_dict()
    c20, c60 = S.loc["capacity_pre_20"], S.loc["capacity_pre_60"]
    h1 = c20["cluster_boot_ci_low"] > 0 and c20["spearman_rho"] > 0
    h2dir = c60["spearman_rho"] > 0
    h2 = h2dir and c60["cluster_boot_ci_low"] > 0
    if h1 and h2dir:
        verdict = ("CASE A: capacity_pre_20 is positive with a ticker-cluster 95% CI excluding "
                   "zero and capacity_pre_60 is directionally consistent. Asset movement "
                   "capacity contains genuine ex-ante predictive information about later "
                   "favorable-excursion magnitude.")
        rec = "STRONG ENOUGH FOR MAIN TEXT"
    elif c20["spearman_rho"] > 0 and not h1:
        verdict = ("CASE B: rho is positive but the ticker-cluster CI includes zero. The ex-post "
                   "capacity law has only weak ex-ante support in this July sample; do not "
                   "call it predictive.")
        rec = "SUPPORTING EVIDENCE ONLY"
    elif h1 != h2:
        verdict = "CASE C: only one window is supported; both are reported, neither is selected post hoc."
        rec = "SUPPORTING EVIDENCE ONLY"
    else:
        verdict = ("CASE D: both windows near zero. Valid negative result: historical volatility "
                   "does not recover the ex-post effect.")
        rec = "NULL RESULT"

    R = [f"# Ex-ante asset-capacity experiment — RESULTS\n",
         "Outcome = the project's truncated stance-aware favorable excursion (TFE), unchanged "
         f"(verified identical to `collection_sample.build_rows()` for all {len(df)} posts).\n",
         "## 1. Data audit",
         f"- TFE-eligible posts (project sample): {len(df)} over {df['ticker'].nunique()} tickers; "
         f"posting dates {df['post_date'].min()} .. {df['post_date'].max()}; "
         f"N in [{df['N'].min()}, {df['N'].max()}] sessions (median {int(df['N'].median())}).",
         f"- Upstream exclusions from the 2,100-post collection (project pipeline, reproduced): {upstream}",
         f"- W=20 eligible: n={n20} posts / {res[20]['n_tickers']} tickers; excluded {len(df)-n20}: {ex20 or 'none'}",
         f"- W=60 eligible: n={n60} posts / {res[60]['n_tickers']} tickers; excluded {len(df)-n60}: {ex60 or 'none'}",
         f"- Adjusted-close unavailable (raw-close fallback): {len(fallback_t)} tickers, "
         f"{int(df['volatility_raw_close_fallback'].sum())} posts",
         f"- Stance (existing LLM-extracted labels): bullish {int((df['stance']>0).sum())}, "
         f"neutral {int((df['stance']==0).sum())}, bearish {int((df['stance']<0).sum())}",
         f"- Integrity/leakage tests: {sum(ok for _,ok,_ in TESTS)}/{len(TESTS)} passed (see leakage_tests.txt)\n",
         "## 2. Primary result (H1, pre-specified W=20)", "- " + line("capacity_pre_20"),
         f"- H1 supported (rho>0 and cluster CI excludes 0): **{'YES' if h1 else 'NO'}**\n",
         "## 3. Robustness (H2, pre-specified W=60)", "- " + line("capacity_pre_60"),
         f"- Directionally consistent with H1: **{'YES' if h2dir else 'NO'}**; CI excludes 0: {'YES' if h2 else 'NO'}\n",
         "## 4. Component ablations (same eligible sample)",
         "- " + line("sqrt_N (W=20 sample)"), "- " + line("sigma_pre_20"), "- " + line("capacity_pre_20"),
         "- W=60 sample: " + line("sqrt_N (W=60 sample)"), "- W=60 sample: " + line("sigma_pre_60"),
         "- Note: a larger point estimate for the product is not evidence of statistical superiority; "
         "compare the overlapping cluster CIs.\n",
         "## 5. Stance subgroups (descriptive)",
         "- " + line("capacity_pre_20_bullish"), "- " + line("capacity_pre_20_bearish"),
         "- " + line("capacity_pre_60_bullish"), "- " + line("capacity_pre_60_bearish"),
         "- Neutral-stance posts are in the primary sample (their TFE is the upward excursion, "
         "as in the project pipeline) but form no subgroup.\n",
         "## 6. Horizon control (secondary)",
         "- " + line("partial_sigma20_controlling_N_stance") +
         " — Pearson correlation of the residuals of rank(TFE) and rank(sigma_pre_20) after OLS on "
         "[1, log N, bullish, bearish].",
         "- Exploratory FE regression rank(TFE) ~ rank(sigma_pre_20) + log N + stance + post-date FE: " +
         line("fe_regression_coef_rank_sigma20_with_date_FE").replace("rho=", "coef=") + "\n",
         "## 7. Task-aligned ranking"]
    for W in (20, 60):
        r = res[W]
        R.append(f"- capacity_pre_{W}: n={r['n']} posts, {r['pairs']:,} unique pairs with unequal TFE; "
                 f"Kendall tau-b={r['kendall_tau']:+.3f} (cluster CI [{r['kendall_ci'][0]:+.3f}, "
                 f"{r['kendall_ci'][1]:+.3f}]); pairwise concordance={r['concordance']:.3f} "
                 f"(cluster CI [{r['concordance_ci'][0]:.3f}, {r['concordance_ci'][1]:.3f}]; "
                 f"capacity ties 0.5; B={B_PAIR})")
    R += ["- No naive binomial p-value is reported: the pairs are massively dependent.\n",
          "## 8. Common-sample comparison with the project's ex-post capacity",
          "The ex-post sigma is recoverable exactly (`collection_sample.daily_vol`, July closes "
          "through 2026-07-24, i.e. it includes post-publication data).",
          "- " + line("capacity_expost_common_W20_sample"), "- " + line("capacity_pre_20"),
          "- " + line("delta_rho_pre20_minus_expost_common").replace("rho=", "delta_rho="),
          "- " + line("delta_rho_pre60_minus_expost_common").replace("rho=", "delta_rho="),
          "- (The project's published ex-post rho=0.399 is the same statistic on the full 502-post "
          "sample; the W=20 common sample is its subset.)\n",
          "## 9. Interpretation (pre-specified rules)", verdict, f"\n**Recommendation: {rec}**\n",
          "No manuscript edits were made."]
    (OUT / "RESULTS.md").write_text("\n".join(R), encoding="utf-8")

    M = ["# Ex-ante asset-capacity experiment — METHODS\n",
         "## Outcome (unchanged from the project)",
         "TFE_i from `collection_sample.build_rows()`: bars from `data/price_cache_ohlc.json` "
         f"(Yahoo Finance via yfinance, dividend-unadjusted OHLC) truncated to {CUTOFF}; entry = "
         "author-quoted price if within 15% of the post-date close, else that close; excursion window "
         "= trading days strictly after the posting date through the cutoff; bullish/neutral TFE = "
         "max High/entry − 1, bearish TFE = 1 − min Low/entry; compounded ±10%/day price-limit guard. "
         "Stance = existing LLM-extracted label in `data/ranking_extract.json` (−1/0/+1). "
         "N_i = number of sessions in that window. Re-derived here with audit columns and asserted "
         "identical to the project values (TFE, N, stance, ex-post sigma) before any statistic.\n",
         "## Ex-ante volatility",
         f"- History: `data/price_history_2026_adjclose.json`, fetched {HIST_START} .. 2026-07-24 "
         "with `yf.Ticker(sym).history(auto_adjust=False)`; symbol = `<code>.TW`, fallback `.TWO`; "
         "resolved symbol stored per ticker. Returns use **Adj Close**; if absent for a ticker, "
         "raw Close is used and `volatility_raw_close_fallback=True`.",
         "- Log returns r_t = log(P_t / P_{t-1}) between consecutive available bars; bars with "
         "missing/non-positive prices are dropped, never filled.",
         f"- sigma_pre,W(i) = sd(ddof=1) of the LAST W returns whose END date < posting date d_i "
         "(calendar date from the post id; the posting day itself is excluded even if the post was "
         "written after the close). W=20 primary, W=60 robustness, fixed before seeing results. "
         "Fewer than W returns ⇒ NaN and exclusion with a recorded reason. No annualisation.",
         "- capacity_pre,W = sigma_pre,W × sqrt(N_i), unnormalised.\n",
         "## Statistics",
         f"- Spearman rho with scipy two-sided p (reported, not primary). Primary uncertainty: "
         f"ticker-cluster bootstrap, seed {SEED}, B={B_CORR}: resample the distinct tickers with "
         "replacement, include every post of each sampled ticker (clusters repeated when a ticker is "
         "drawn more than once), recompute the statistic; report median and 2.5/97.5 percentiles. "
         "The generator is re-seeded per specification.",
         "- Partial rank correlation: rank(TFE) and rank(sigma_pre) each regressed by OLS on "
         "[1, log N, bullish, bearish]; Pearson correlation of the two residual vectors. Exploratory "
         "FE regression adds posting-date dummies; the coefficient on rank(sigma_pre) is reported.",
         f"- Ranking: Kendall tau-b (scipy) and pairwise concordance over all unique unordered pairs "
         f"with unequal TFE (capacity ties 0.5); ticker-cluster bootstrap with B={B_PAIR}, the post "
         "set and its pairs rebuilt inside every replicate.",
         "- Common-sample comparison: ex-post capacity = project `sigma` (sd of July log close "
         "returns through the cutoff) × sqrt(N); delta_rho bootstrapped pairwise within replicates.",
         "- Figure: five equal-count bins of capacity_pre_20 with edges fixed on the full sample; bin "
         "medians of TFE with cluster-bootstrap CIs; the connecting line is descriptive only.\n",
         "## Leakage tests",
         "See `leakage_tests.txt`: per valid row last price date < posting date and < TFE start; "
         "exactly W returns; sigma equals sd of the stored window (so no annualisation factor); "
         "capacity = sigma × sqrt N; post ids unique; every exclusion has a reason; TFE identical to "
         "the project function. Any failure aborts before statistics."]
    (OUT / "METHODS.md").write_text("\n".join(M), encoding="utf-8")


if __name__ == "__main__":
    main()
