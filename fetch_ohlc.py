# -*- coding: utf-8 -*-
"""Fetch July-2026 daily OHLC bars for every ticker identified in the ranking
collection, writing data/price_cache_ohlc.json for the Section 8 analyses.

Run AFTER extract_ranking.py (which produces data/ranking_extract.json).
Taiwanese listings are queried as <code>.TW with a .TWO fallback. Bars with
missing or non-positive prices are dropped; tickers with fewer than five valid
bars are stored as null. Prices are dividend-unadjusted, matching the label
construction. The downstream scripts truncate to 2026-07-24 themselves, so a
re-fetch on a later date changes nothing.
"""
import json
import math
import time
from pathlib import Path

import yfinance as yf

DATA = Path(__file__).parent / "data"
OUT = DATA / "price_cache_ohlc.json"

ext = json.loads((DATA / "ranking_extract.json").read_text(encoding="utf-8"))
tickers = sorted({v["ticker"] for v in ext.values() if v and v.get("ticker")})
print(f"tickers to fetch: {len(tickers)}", flush=True)

cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
for i, t in enumerate(tickers, 1):
    if t in cache:
        continue
    got = None
    for suffix in (".TW", ".TWO"):
        try:
            df = yf.Ticker(t + suffix).history(start="2026-07-01", auto_adjust=False)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            bars = {}
            for d, r in df.iterrows():
                vals = [float(r["High"]), float(r["Low"]), float(r["Close"])]
                if any(math.isnan(v) or v <= 0 for v in vals):
                    continue
                bars[str(d.date())] = {"high": vals[0], "low": vals[1], "close": vals[2]}
            if len(bars) >= 5:
                got = bars
            break
        time.sleep(0.3)
    cache[t] = got
    if i % 10 == 0:
        OUT.write_text(json.dumps(cache), encoding="utf-8")
        print(f"  {i}/{len(tickers)}", flush=True)
    time.sleep(0.2)

OUT.write_text(json.dumps(cache), encoding="utf-8")
ok = sum(1 for v in cache.values() if v)
print(f"done: {ok}/{len(cache)} tickers have data -> {OUT.name}")
