# -*- coding: utf-8 -*-
"""Independent re-verification of the ex-ante capacity outputs.

Runs against the written CSV and the cached price history, recomputing every
volatility window from scratch, so it does not trust the in-script checks.

    python -m pytest tests/test_exante_capacity.py -q
or  python tests/test_exante_capacity.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "exante_capacity"
CSV = OUT / "post_level_exante_capacity.csv"
CACHE = HERE / "data" / "price_history_2026_adjclose.json"


def _df():
    # round_trip: pandas' default fast float parser is not guaranteed to
    # reproduce the written repr exactly, which would fail the equality checks
    return pd.read_csv(CSV, float_precision="round_trip",
                       dtype={"post_date": str, "last_price_date_pre20": str,
                              "last_price_date_pre60": str, "tfe_start": str})


def _returns(rec):
    key = "adj_close" if rec["has_adj_close"] else "close"
    dates = sorted(rec["bars"])
    out = []
    for a, b in zip(dates, dates[1:]):
        p0, p1 = rec["bars"][a][key], rec["bars"][b][key]
        if p0 == p0 and p1 == p1 and p0 > 0 and p1 > 0:
            out.append((b, math.log(p1 / p0)))
    return out


def test_post_ids_unique():
    assert _df()["post_id"].is_unique


def test_tfe_matches_project_function():
    from collection_sample import build_rows
    ref = {r["pid"]: r for r in build_rows()}
    df = _df()
    assert set(df["post_id"]) == set(ref)
    for _, r in df.iterrows():
        assert r["TFE"] == ref[r["post_id"]]["stance_aware"]
        assert r["N"] == ref[r["post_id"]]["N"]


def test_no_leakage_and_window_size():
    df = _df()
    for W in (20, 60):
        v = df[df[f"sigma_pre_{W}"].notna()]
        assert (v[f"last_price_date_pre{W}"] < v["post_date"]).all()
        assert (v[f"last_price_date_pre{W}"] < v["tfe_start"]).all()
        assert (v[f"n_returns_pre{W}"] == W).all()
        ex = df[df[f"sigma_pre_{W}"].isna()]
        # astype(str): with zero exclusions the column is an empty float
        # series and the .str accessor would raise
        assert ex[f"exclusion_reason_pre{W}"].fillna("").astype(str).str.len().gt(0).all()


def test_sigma_recomputed_from_cache_no_annualisation():
    df = _df()
    hist = json.loads(CACHE.read_text(encoding="utf-8"))
    rets = {t: _returns(rec) for t, rec in hist.items() if rec}
    for W in (20, 60):
        v = df[df[f"sigma_pre_{W}"].notna()]
        for _, r in v.iterrows():
            pre = [x for x in rets[r["ticker"]] if x[0] < r["post_date"]]
            win = pre[-W:]
            assert len(win) == W
            assert max(d for d, _ in win) < r["post_date"]
            sd = float(np.std([x for _, x in win], ddof=1))
            assert math.isclose(sd, r[f"sigma_pre_{W}"], rel_tol=0, abs_tol=1e-15)
            assert not math.isclose(sd * math.sqrt(252), r[f"sigma_pre_{W}"], rel_tol=1e-6)
            assert math.isclose(r[f"capacity_pre_{W}"], sd * math.sqrt(r["N"]), rel_tol=1e-12)


def test_future_data_not_used():
    """The sigma window must end before the first TFE session for every row."""
    df = _df()
    for W in (20, 60):
        v = df[df[f"sigma_pre_{W}"].notna()]
        assert (v[f"last_price_date_pre{W}"] < v["tfe_start"]).all()
        assert (v[f"first_price_date_pre{W}"] < v[f"last_price_date_pre{W}"]).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("all tests passed")
