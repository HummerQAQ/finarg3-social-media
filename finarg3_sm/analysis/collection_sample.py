# -*- coding: utf-8 -*-
"""Single source of truth for the July-2026 collection analysis sample.

Builds, for every collection post with an identified ticker and usable price
data, the three quantities the cross-period analyses need:

    y_long   long-only realized excursion   (max High since posting)/entry - 1
    y_stance stance-aware realized MPP      bullish/neutral -> as above
                                            bearish        -> 1 - (min Low)/entry
    c        capacity covariate             sigma * sqrt(N)

Guards mirror build_top210.py: a quoted price is accepted only within 15% of
the post-date close, and a realized bound must respect the compounded daily
price limit (+/-10% per day).

Bars are truncated to CUTOFF, the submission-time horizon of the original
price cache. Without this the analysis silently measures excursions over the
extra month of price action present in any later re-fetch.
"""
import json
import math
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

import numpy as np

HERE = PROJECT_ROOT
DATA = HERE / "data"
COLLECTION = HERE / "realtime" / "data" / "FinArg3_Social_Media_Post_Ranking_2100_Collection.json"
CUTOFF = "2026-07-24"


def load_prices():
    px = json.loads((DATA / "price_cache_ohlc.json").read_text(encoding="utf-8"))
    px = {t: ({d: b for d, b in bars.items() if d <= CUTOFF} if bars else None)
          for t, bars in px.items()}
    return {t: (b if b and len(b) >= 5 else None) for t, b in px.items()}


def daily_vol(px):
    vol = {}
    for t, bars in px.items():
        if not bars:
            continue
        closes = [bars[d]["close"] for d in sorted(bars)]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
                if closes[i - 1] > 0 and closes[i] > 0]
        if len(rets) >= 5:
            vol[t] = float(np.std(rets, ddof=1))
    return vol


def build_rows():
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    ext = json.loads((DATA / "ranking_extract.json").read_text(encoding="utf-8"))
    sco = json.loads((DATA / "ranking_scores.json").read_text(encoding="utf-8"))
    px = load_prices()
    vol = daily_vol(px)

    rows = []
    for it in col:
        pid = it["post_id"]
        e = ext.get(pid) or {}
        t = e.get("ticker")
        if not t or not px.get(t) or t not in vol:
            continue
        d = pid.split("_")[1]
        post_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        bars = px[t]
        fut = [x for x in sorted(bars) if x >= post_date]
        if not fut:
            continue
        day0 = bars[fut[0]]["close"]
        entry = e.get("quoted_price")
        if not entry or entry <= 0 or abs(entry / day0 - 1) > 0.15:
            entry = day0
        # The excursion window starts the trading day AFTER the post. Entry is
        # the post-date close, so including the post day would let a post "exit"
        # at a high or low that may have printed before it was written.
        #
        # The originally published Section 8.1 fit did include the posting day
        # in both the excursion and N, while still requiring at least one later
        # day (which is where n=502 comes from). That convention reproduces the
        # published numbers closely -- rho 0.498 vs 0.495, Pearson 0.435 vs
        # 0.434 -- and is what build_top210.py used for the submitted ranking.
        # Excluding the posting day, as we do here, costs the long-only fit
        # rho 0.498 -> 0.442 and is the defensible convention.
        fut = [x for x in fut if x > post_date]
        if not fut:
            continue
        N = len(fut)
        up = max(bars[x]["high"] for x in fut) / entry - 1
        dn = 1 - min(bars[x]["low"] for x in fut) / entry
        if not (-0.5 < up < 1.1 ** N - 1 + 0.10 and -0.5 < dn < 1 - 0.9 ** N + 0.10):
            continue
        stance = e.get("stance", 0)
        s = sco.get(pid) or {}
        rows.append({
            "pid": pid, "ticker": t, "stance": stance, "N": N,
            "sigma": vol[t], "x": vol[t] * math.sqrt(N),
            "long": up, "stance_aware": dn if stance < 0 else up,
            # text-model scores transferred from the 2019-2020 training posts
            "enc": s.get("enc"), "lr": s.get("lr"),
            # LLM judgment made on the 2026 post itself (NOT transferred)
            "llm_upside": e.get("upside_score"),
        })
    return rows
