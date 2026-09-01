# -*- coding: utf-8 -*-
"""External resolution of the MPP label's stance semantics.

The released labels are sign-degenerate: a long-only and a stance-aware
definition both give MPP >= 0 >= ML on every row, so no in-corpus check
separates them. External prices do. Writing O for the entry price and H, L for
the highest high and lowest low of the evaluation window,

    long-only    : H = O(1 + MPP),   L = O(1 + ML)
    stance-aware : L = O(1 - MPP),   H = O(1 - ML)      (bearish post)

We do not know the organizers' window length or price convention, so we do not
attempt to reconstruct the window exactly. Two weaker tests suffice:

  (1) ATTAINABILITY (window-free). Each reading requires the market to have
      printed a specific price level. Letting the entry range over everything
      the stock traded at when the post was written, a reading is REFUTED if
      the level it needs never occurred in the following 18 months -- no
      choice of window can rescue it.

  (2) BEST-FIT SCAN. For each window length, take the entry that reproduces
      the recorded MPP exactly, keep it only if the stock actually traded
      there, and report how far the resulting ML is from the recorded one.

Both anchors are training posts whose posting date is stated in the text.
Evidence is written to data/stance_anchors.json.
"""
import json
from pathlib import Path

import yfinance as yf

DATA = Path(__file__).parent / "data"

ANCHORS = [
    {
        "pid": 48, "name": "Macronix", "symbol": "2337.TW",
        # posting date and short entry price are stated verbatim in the post text
        "post_date": "2019-04-30", "quoted_entry": 23.8,
        "mpp": 0.1285, "ml": -0.3688,
    },
    {
        "pid": 51, "name": "Beyond Meat", "symbol": "BYND",
        # post references the prior two sessions -> next session after 6/10 peak
        "post_date": "2019-06-12", "quoted_entry": None,
        "mpp": 0.0000, "ml": -0.1681,
    },
]


def get_bars(symbol, start, end):
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    return [{"date": str(d.date()), "open": float(r["Open"]),
             "high": float(r["High"]), "low": float(r["Low"]),
             "close": float(r["Close"])} for d, r in df.iterrows()]


results = []
for a in ANCHORS:
    hist = get_bars(a["symbol"], a["post_date"], "2020-12-31")
    post_bar, fut = hist[0], hist[1:]
    mpp, ml = a["mpp"], a["ml"]
    e_lo = min(post_bar["low"], fut[0]["low"])
    e_hi = max(post_bar["high"], fut[0]["high"])
    ext = {"max_high": max(b["high"] for b in fut),
           "min_low": min(b["low"] for b in fut),
           "max_high_date": max(fut, key=lambda b: b["high"])["date"],
           "min_low_date": min(fut, key=lambda b: b["low"])["date"]}

    # (1) attainability, entry free within the range traded at posting time
    need = {
        "long_only": {
            "high_level": (e_lo * (1 + mpp), e_hi * (1 + mpp)),
            "low_level": (e_lo * (1 + ml), e_hi * (1 + ml))},
        "stance_aware": {
            "low_level": (e_lo * (1 - mpp), e_hi * (1 - mpp)),
            "high_level": (e_lo * (1 - ml), e_hi * (1 - ml))},
    }
    attain = {}
    for reading, lv in need.items():
        hi_ok = ext["max_high"] >= lv["high_level"][0]   # some entry works
        lo_ok = ext["min_low"] <= lv["low_level"][1]
        attain[reading] = {
            "high_level_required": lv["high_level"], "high_attained": hi_ok,
            "low_level_required": lv["low_level"], "low_attained": lo_ok,
            "refuted": not (hi_ok and lo_ok)}

    # (2) best-fit scan: entry pinned by MPP, judged by the ML it implies
    best = {}
    for reading in ("long_only", "stance_aware"):
        cands = []
        for T in range(2, min(len(fut), 250) + 1):
            seg = fut[:T]
            H = max(b["high"] for b in seg)
            L = min(b["low"] for b in seg)
            entry = H / (1 + mpp) if reading == "long_only" else L / (1 - mpp)
            if not (e_lo * 0.97 <= entry <= e_hi * 1.03):
                continue
            implied_ml = (L / entry - 1) if reading == "long_only" else (1 - H / entry)
            cands.append({"T": T, "entry": entry, "implied_ml": implied_ml,
                          "ml_error": abs(implied_ml - ml)})
        best[reading] = (min(cands, key=lambda c: c["ml_error"]) if cands
                         else {"note": "no window admits an entry the stock traded at"})

    r = {"pid": a["pid"], "name": a["name"], "symbol": a["symbol"],
         "post_date": a["post_date"], "post_date_bar": post_bar,
         "next_session_bar": fut[0], "quoted_entry": a["quoted_entry"],
         "mpp": mpp, "ml": ml,
         "price_range_at_posting": {"low": e_lo, "high": e_hi},
         "realized_18m": ext, "attainability": attain, "best_fit": best,
         "data_source": "Yahoo Finance via yfinance, auto_adjust=False, "
                        "daily OHLC, retrieved 2026-08-22"}
    results.append(r)

    print(f"\n=== {a['name']} ({a['symbol']}), posted {a['post_date']} ===")
    print(f"  labels MPP {mpp:+.4f} / ML {ml:+.4f}; traded {e_lo:.2f}-{e_hi:.2f} "
          f"when written" + (f"; author quotes {a['quoted_entry']}" if a['quoted_entry'] else ""))
    print(f"  realized over 18 months: high {ext['max_high']:.2f} "
          f"({ext['max_high_date']}), low {ext['min_low']:.2f} ({ext['min_low_date']})")
    for reading in ("long_only", "stance_aware"):
        at = attain[reading]
        print(f"  {reading}:")
        print(f"    needs a high of {at['high_level_required'][0]:.2f}-"
              f"{at['high_level_required'][1]:.2f} -> "
              f"{'attained' if at['high_attained'] else 'NEVER ATTAINED'}; "
              f"a low of {at['low_level_required'][0]:.2f}-"
              f"{at['low_level_required'][1]:.2f} -> "
              f"{'attained' if at['low_attained'] else 'NEVER ATTAINED'}")
        print(f"    verdict: {'REFUTED' if at['refuted'] else 'consistent'}")
        b = best[reading]
        if "T" in b:
            print(f"    best fit: T={b['T']}, entry {b['entry']:.2f}, "
                  f"implied ML {b['implied_ml']:+.4f} vs recorded {ml:+.4f} "
                  f"(error {b['ml_error']:.4f})")
        else:
            print(f"    best fit: {b['note']}")

(DATA / "stance_anchors.json").write_text(
    json.dumps(results, indent=1), encoding="utf-8")
print("\nsaved: data/stance_anchors.json")
