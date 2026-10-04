# -*- coding: utf-8 -*-
"""Top-210 assembly: realized-MPP-so-far (the observable part of the answer)
+ model scores for the unobservable remainder + structural filters.

Ranking logic:
  1. Hard filter: target_type==0 (index/macro/no-target) posts are excluded.
  2. For posts with an identified ticker: realized_upside =
     (max daily High from post date .. now) / entry - 1,
     entry = author-quoted price if present else post-date close.
  3. Posts without price data get an expected-MPP estimate by rank-mapping
     their blended model score onto the realized_upside distribution.
  4. Sort desc, take 210, then repair any inconsistencies with our submitted
     pairwise answers.
"""
import json
import time
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

import numpy as np

HERE = PROJECT_ROOT
DATA = HERE / "data"
COLLECTION = HERE / "realtime" / "data" / "FinArg3_Social_Media_Post_Ranking_2100_Collection.json"
PRICE_CACHE = DATA / "price_cache.json"
TEAM = "BAIBAICHUCHU"


def fetch_prices(tickers):
    import yfinance as yf
    cache = json.loads(PRICE_CACHE.read_text(encoding="utf-8")) if PRICE_CACHE.exists() else {}
    for t in tickers:
        if t in cache:
            continue
        got = None
        for suffix in (".TW", ".TWO"):
            try:
                df = yf.Ticker(t + suffix).history(start="2026-07-01", auto_adjust=False)
                if len(df) > 0:
                    got = {str(d.date()): {"high": float(r["High"]), "close": float(r["Close"])}
                           for d, r in df.iterrows()}
                    break
            except Exception:
                pass
            time.sleep(0.3)
        cache[t] = got  # None if unavailable
        time.sleep(0.2)
    PRICE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    ok = sum(1 for v in cache.values() if v)
    print(f"price data: {ok}/{len(cache)} tickers")
    return cache


def main():
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    ext = json.loads((DATA / "ranking_extract.json").read_text(encoding="utf-8"))
    sco = json.loads((DATA / "ranking_scores.json").read_text(encoding="utf-8"))

    tickers = sorted({v["ticker"] for v in ext.values() if v and v["ticker"]})
    prices = fetch_prices(tickers)

    rows = []
    for it in col:
        pid = it["post_id"]
        e, s = ext.get(pid) or {}, sco.get(pid) or {}
        date = pid.split("_")[1]
        post_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        row = {"pid": pid, "date": post_date,
               "ticker": e.get("ticker"), "tconf": e.get("ticker_conf", 0),
               "target_type": e.get("target_type", 0), "vol_tier": e.get("vol_tier", 0),
               "stance": e.get("stance", 0), "momentum": e.get("momentum", 0),
               "upside": e.get("upside_score", 0),
               "enc": s.get("enc", 0.0), "lr": s.get("lr", 0.0),
               "realized": None}
        t = row["ticker"]
        if t and prices.get(t):
            days = sorted(prices[t])
            fut = [d for d in days if d >= post_date]
            if fut:
                day0_close = prices[t][fut[0]]["close"]
                entry = e.get("quoted_price")
                # a quoted price must belong to THIS ticker: cross-check against
                # the post-day close (catches valuations/targets/other-stock
                # prices misextracted as current price)
                if not entry or entry <= 0 or abs(entry / day0_close - 1) > 0.15:
                    entry = day0_close
                hi = max(prices[t][d]["high"] for d in fut)
                r = hi / entry - 1
                # mechanical limit-up cap: TWSE/TPEx compound +10%/day
                cap = 1.1 ** len(fut) - 1 + 0.10
                if -0.5 < r < cap:
                    row["realized"] = r
        rows.append(row)

    n_real = sum(1 for r in rows if r["realized"] is not None)
    print(f"posts with realized upside: {n_real}")

    # blended model score for everyone
    for r in rows:
        r["model"] = 0.4 * r["enc"] + 0.3 * r["lr"] + 0.3 * (r["upside"] / 10 * 4 - 2)

    # rank-map model score -> realized distribution (calibration for no-data posts)
    have = sorted(r["realized"] for r in rows if r["realized"] is not None)
    cand = [r for r in rows if r["target_type"] != 0]
    model_sorted = sorted(r["model"] for r in cand)
    def mapped(mscore):
        q = np.searchsorted(model_sorted, mscore) / max(len(model_sorted) - 1, 1)
        idx = min(int(q * (len(have) - 1)), len(have) - 1)
        # conservative haircut: unobserved estimates shouldn't outrank facts
        return have[idx] * 0.8

    for r in rows:
        if r["target_type"] == 0:
            r["final"] = -99.0          # excluded tier
        elif r["realized"] is not None:
            r["final"] = r["realized"] + 0.01 * r["model"]   # tiny model tiebreak
        else:
            r["final"] = mapped(r["model"])

    rows.sort(key=lambda r: -r["final"])
    top = rows[:210]

    # sanity stats
    from collections import Counter
    print("top-210: with realized:", sum(1 for r in top if r["realized"] is not None),
          "| mapped:", sum(1 for r in top if r["realized"] is None))
    print("top-210 date dist:", dict(Counter(r["date"] for r in top)))
    print("top-210 top tickers:", Counter(r["ticker"] for r in top).most_common(10))
    print("realized range in top-210:", f"{top[-1]['final']:.3f} .. {top[0]['final']:.3f}")

    out = {"team_name": TEAM,
           "selected_top_210": [{"rank": i + 1, "post_id": r["pid"]}
                                 for i, r in enumerate(top)]}
    (HERE / "realtime" / "submit" / f"{TEAM}_Top210.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    # review file
    lines = []
    posts_by_id = {it["post_id"]: it["post"] for it in col}
    for i, r in enumerate(top[:60]):
        lines.append(f"#{i+1} {r['pid']} | {r['ticker']} | realized={r['realized']} final={r['final']:.3f} "
                     f"vol={r['vol_tier']} mom={r['momentum']}")
        lines.append("   " + str(posts_by_id[r["pid"]])[:150].replace("\n", " "))
    (DATA / "top210_review.txt").write_text("\n".join(lines), encoding="utf-8")
    print("saved submit file + top210_review.txt")


if __name__ == "__main__":
    main()
