# -*- coding: utf-8 -*-
"""Top-210 task: mass extraction over the 2,100-post collection.

Per post the LLM extracts: primary ticker (4-digit TW code), the price the
author quotes as current (best entry estimate), plus the core capacity
attributes (target_type / stance / vol_tier / upside_score).

Output: data/ranking_extract.json  {post_id: {...}}
"""
import json
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

DATA = Path(__file__).parent / "data"
COLLECTION = Path(__file__).parent / "realtime" / "data" / "FinArg3_Social_Media_Post_Ranking_2100_Collection.json"
CACHE_FILE = DATA / "ranking_cache.json"

SYSTEM_PROMPT = """你是台股資深交易員。你會看到一篇 2026 年 7 月的台股社群貼文,請萃取以下欄位（JSON）：

- ticker: 貼文「主要討論標的」的台股 4 位數代號（字串,如 "2330"）。用你對台股的知識把公司名對應到代號（台積電=2330、鴻海=2317、聯電=2303、長榮=2603、群創=3481、力積電=6770、南亞科=2408、華邦電=2344、緯創=3231、緯穎=6669、國巨=2327、合晶=6182、環球晶=6488、日月光投控=3711、聯發科=2454、廣達=2382、0050=0050、正2=00631L、0056=0056...）。若是 ETF 用 ETF 代號。**若貼文是大盤/指數/總經文或無法辨識單一主要標的,填 null**
- ticker_confidence: 0-1,你對代號辨識的把握
- quoted_price: 貼文中作者引用的「當下價格」（數字,如 107.5）。沒有明確引用填 null
- target_type: 2=台股個股, 1=ETF/槓桿ETF/海外, 0=大盤指數/期貨/總經/無標的
- stance: 短期方向 1=看多 0=中性 -1=看空
- vol_tier: 2=投機中小型/題材股/興櫃, 1=一般個股, 0=大型牛皮股/金控/電信/一般ETF（槓桿ETF算1）
- momentum: 貼文透露的當下動能 1=正在漲/剛反彈 0=不明 -1=正在跌/破底
- upside_score: 0-10,未來一兩個月做多此標的的最大上漲潛力

只輸出JSON。"""

_lock = threading.Lock()


def load_cache():
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


def save_cache(c):
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_FILE)


def parse(text):
    try:
        o = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    out = {}
    t = o.get("ticker")
    out["ticker"] = str(t) if t not in (None, "", "null") else None
    try:
        out["ticker_conf"] = float(o.get("ticker_confidence", 0))
    except (TypeError, ValueError):
        out["ticker_conf"] = 0.0
    try:
        out["quoted_price"] = float(o.get("quoted_price")) if o.get("quoted_price") not in (None, "", "null") else None
    except (TypeError, ValueError):
        out["quoted_price"] = None
    for k, lo, hi in (("target_type", 0, 2), ("stance", -1, 1), ("vol_tier", 0, 2),
                      ("momentum", -1, 1), ("upside_score", 0, 10)):
        try:
            v = float(o.get(k, 0))
        except (TypeError, ValueError):
            v = 0.0
        out[k] = min(max(v, lo), hi)
    return out


def main():
    client = OpenAI()
    cache = load_cache()
    posts = json.loads(COLLECTION.read_text(encoding="utf-8"))
    print(f"extracting {len(posts)} posts ...")

    def one(it):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(it["post"])[:900]}]
        key = hashlib.sha256(json.dumps(["gpt-5-mini", msgs], ensure_ascii=False).encode()).hexdigest()
        with _lock:
            raw = cache.get(key)
        if raw is None:
            kwargs = dict(model="gpt-5-mini", messages=msgs,
                          response_format={"type": "json_object"})
            for attempt in range(6):
                try:
                    raw = client.chat.completions.create(**kwargs).choices[0].message.content
                    break
                except Exception as e:
                    if "response_format" in str(e) and "response_format" in kwargs:
                        kwargs.pop("response_format"); continue
                    if attempt == 5:
                        return it["post_id"], None
                    import time
                    time.sleep(min(2 ** attempt, 20))
            with _lock:
                cache[key] = raw
                if len(cache) % 100 == 0:
                    save_cache(cache)
        return it["post_id"], parse(raw)

    results = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for i, (pid, attrs) in enumerate(ex.map(one, posts)):
            results[pid] = attrs
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/2100")
    save_cache(cache)
    failed = [k for k, v in results.items() if v is None]
    print(f"done; failed: {len(failed)}")
    (DATA / "ranking_extract.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [v for v in results.values() if v]
    with_ticker = sum(1 for v in ok if v["ticker"])
    print(f"posts with identified ticker: {with_ticker}/{len(ok)}")


if __name__ == "__main__":
    main()
