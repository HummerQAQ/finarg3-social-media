# -*- coding: utf-8 -*-
"""Track 3: LLM attribute extraction (pointwise, per post).

The LLM reads one post and outputs structured attributes aligned with the
long-only MPP label semantics (momentum leakage, catalysts, chips, insider
hints, volatility tier...). A downstream logistic model learns the weights
from labels — no prior misalignment possible.

Pointwise design: one call per unique post, fully cached, reusable verbatim
for the real-time evaluation week.

Usage:
  python -m finarg3_sm.judging.extract_attributes --model gpt-5-mini
"""
import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

from openai import OpenAI

DATA = PROJECT_ROOT / "data"
CACHE_FILE = DATA / "attr_cache.json"
MAX_POST_CHARS = 900

SYSTEM_PROMPT = """你是一位深諳台股散戶生態的資深交易員。你會看到一篇台灣股市社群貼文，\
請萃取以下屬性，用來預測「發文當下買進該標的後、未來一兩個月內的最大可能獲利(MPP)」。\
（MPP一律以做多計算，與貼文立場無關。）

請輸出JSON，欄位定義如下：
- target_type: 貼文主要討論的標的類型。2=台股個股, 1=海外個股或其他證券, \
0=大盤指數/期貨/總經（例：討論台指期、道瓊、費半、外資期權未平倉、非農數據者為0）
- stance: 貼文對標的「短期方向」的立場（若短期與長期看法不同，以短期為準；\
例如「短多因子外資狂買+長線看壞」算看多）。1=看多, 0=中性/不明, -1=看空
- momentum: 貼文透露的當下價格動能（抱怨賣早了/氣死/被洗掉=正在漲；哀嚎套牢/又跳水=正在跌）。\
1=正在漲或剛突破, 0=不明或盤整, -1=正在跌或破底
- narrative: 貼文的主要投資敘事，從以下選一：\
"turnaround"(困境反轉:業績翻身/轉型/由虧轉盈), "growth"(成長:營收EPS高增長), \
"value"(低估:低本益比/高殖利率/低於淨值), "momentum"(動能:法人主力買超/突破/強勢), \
"mean_reversion"(錯殺反彈:超跌/利空出盡/抄底), "speculative"(投機題材:當紅話題/作夢/內線暗示), \
"bearish"(看空論述), "macro"(大盤總經分析), "other"
- catalyst: 明確催化事件的時間距離。2=數日內(明天財報/漲停回補/開獎), \
1=一個月內(當月除權息/財報季/審核), 0=無或超過一個月
- chips: 知情者籌碼訊號。1=主力吃貨/外資投信連買/老闆增持/借券回補, 0=不明, \
-1=出貨/融資暴增/籌碼凌亂
- insider_hint: 內線暗示（作夢文、「不方便說」、老手宣稱之前帶大家賺）。1=有, 0=無
- vol_tier: 標的波動屬性。2=投機中小型股/當紅題材股/生技, 1=一般個股, \
0=牛皮大型股/金控/電信/特別股/定存股/指數
- causal_chain: 因果鏈完整度。2=有具體機制的完整因果鏈(如:建案交屋→認列營收→EPS→股價), \
1=有理由但機制模糊, 0=只有結論沒有理由
- specificity: 具體性。2=有具體數字與可驗證事實, 1=部分具體, 0=空泛
- regret: 懊悔/踏空/停損抱怨文。1=是, 0=否
- upside_score: 綜合判斷未來一兩個月做多此標的的最大上漲潛力，0到10的整數。\
提醒：大盤指數文即使看對，漲幅也遠小於個股，upside_score應偏低；\
看空論述若紮實，做多的MPP低，分數也應偏低。

只輸出JSON，不要其他文字。"""

_lock = threading.Lock()


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_FILE)


FIELDS = {
    "target_type": (0, 2), "stance": (-1, 1), "momentum": (-1, 1),
    "catalyst": (0, 2), "chips": (-1, 1), "insider_hint": (0, 1),
    "vol_tier": (0, 2), "causal_chain": (0, 2), "specificity": (0, 2),
    "regret": (0, 1), "upside_score": (0, 10),
}

NARRATIVES = ["turnaround", "growth", "value", "momentum", "mean_reversion",
              "speculative", "bearish", "macro", "other"]


def parse_attrs(text):
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    out = {}
    for k, (lo, hi) in FIELDS.items():
        try:
            v = float(obj.get(k, 0))
        except (TypeError, ValueError):
            v = 0.0
        out[k] = min(max(v, lo), hi)
    nar = str(obj.get("narrative", "other")).strip().lower()
    out["narrative"] = nar if nar in NARRATIVES else "other"
    return out


def extract_one(client, model, text, cache):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text[:MAX_POST_CHARS]},
    ]
    key = hashlib.sha256(
        json.dumps([model, messages], ensure_ascii=False).encode()
    ).hexdigest()
    with _lock:
        if key in cache:
            return parse_attrs(cache[key])
    kwargs = dict(model=model, messages=messages,
                  response_format={"type": "json_object"})
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            break
        except Exception as e:
            if "response_format" in str(e) and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            if attempt == 3:
                raise
            import time
            time.sleep(2 ** attempt)
    with _lock:
        cache[key] = raw
        if len(cache) % 25 == 0:
            save_cache(cache)
    return parse_attrs(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY environment variable first.")
    client = OpenAI()
    cache = load_cache()

    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test_pairs.json").read_text(encoding="utf-8"))

    jobs = [(f"train:{p['pid']}", p["text"]) for p in posts]
    for t in test:
        jobs.append((f"test:{t['index']}:1", t["post1"]))
        jobs.append((f"test:{t['index']}:2", t["post2"]))
    print(f"extracting attributes for {len(jobs)} posts with {args.model} ...")

    results = {}
    def work(job):
        key, text = job
        return key, extract_one(client, args.model, text, cache)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for key, attrs in ex.map(work, jobs):
            results[key] = attrs
    save_cache(cache)

    failed = [k for k, v in results.items() if v is None]
    if failed:
        print(f"WARNING: {len(failed)} posts failed to parse: {failed[:5]}")
    out = DATA / f"attrs_v2_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("saved:", out.name)


if __name__ == "__main__":
    main()
