# -*- coding: utf-8 -*-
"""Track 1: LLM-as-judge for pairwise MPP comparison.

Design:
  - Rubric prompt in Traditional Chinese, grounded in the signals that the
    feature analysis validated (fundamental evidence +, concrete timeframes +,
    emotional/question markers -).
  - Every pair is judged in BOTH orderings; probabilities are averaged, which
    cancels position bias.
  - Optional multi-vote self-consistency (--votes N).
  - API responses cached on disk; complete cache hits avoid new calls.

Usage:
  set OPENAI_API_KEY=sk-...
  python -m finarg3_sm.judging.llm_judge --mode cv --model gpt-5-mini --sample 30
  python -m finarg3_sm.judging.llm_judge --mode test --model gpt-5-mini
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT

from openai import OpenAI

DATA = PROJECT_ROOT / "data"
CACHE_FILE = DATA / "llm_cache.json"
N_FOLDS = 5
MAX_POST_CHARS = 800

SYSTEM_PROMPT = """你是一位深諳台股散戶生態與市場動能的資深交易員。

你會看到兩篇台灣股市社群貼文（貼文A與貼文B）。你的任務：判斷「在發文當下買進\
哪一篇貼文討論的股票」，之後一段時間內的最大可能獲利（MPP）較高。

【關鍵規則】MPP的定義是「發文時買進、在未來窗口內的最高價賣出」的報酬率——\
一律做多、與貼文立場無關。所以你要預測的是「哪檔股票接下來比較會漲」，\
而不是「哪篇文章寫得比較好」。

判斷步驟與要點：
0. 【先做標的定位】判斷每篇貼文討論的是：台股個股／族群、大盤指數／期指／總經、\
還是海外市場。大盤期指文（外資期權未平倉、融資水位、島狀晨星、FED…）即使分析\
格式完整，MPP天生遠低於個股文——指數一個月只動幾個百分點，個股能動幾十個百分點。\
完整的盤勢籌碼文格式不等於個股alpha。
1. 貼文會洩漏當下的價格動能與市場情緒，優先讀取這些線索：抱怨「賣掉就漲、氣死」\
＝股票正在漲（高MPP訊號）；哀嚎「跌得真慘、套牢」＝正在跌（低MPP訊號）。
2. 【事後解釋≠前瞻訊號】描述「已經發生」的行情（已跳空、已反彈、外資已連買多日、\
型態已完成）的完整解釋文，剩餘上漲空間常已有限。同理，人盡皆知的近期催化劑\
（明天公布財報、下週開獎）常已被市場定價，甚至帶來失望——催化劑越明顯越不是優勢。
3. 【風控語氣≠看空】看好基本面與轉折、但提醒「本益比偏高、短線觀望、不建議追價」\
的文章，是有風險意識的看多（risk-aware bullish），常是高品質高MPP訊號，\
不要因保守措辭而低估。
4. 【看空文的相對評估】MPP一律做多計算：看空且理由紮實→股價傾向跌→MPP低。\
但比較是相對的——對手若是大盤文或訊號更弱的標的，看空文仍可能勝出。\
散戶的情緒性看空、停損投降文常出現在底部，反而可能是反轉訊號。
5. 【精煉摘要有交易含量】「底部放量 多頭排列 外資投信連買」這類壓縮短句是完整的\
價量籌碼訊號，勿因簡短而扣分；盤口語言（買賣差比、明天掛漲停買）同樣有資訊。\
借券回補（軋空）、主力吃貨、無量惜售都是有效籌碼訊號。
6. 【梗文要看背後】看似玩梗的文若背後有事件／估值／股利暗示（「關稅是美麗的誤會」\
＝錯殺修復；「發8.8就是發發的意思」＝股利暗示）常是高MPP；\
純口號式喊單（「一直XX一直爽」）沒有支撐則否。PTT「作夢文」（夢到財報亮眼、\
不方便多說）代表內線暗示，配合高波動題材股經常大漲。
7. 高波動的中小型題材股，MPP天生高於牛皮大型股、金控、電信、特別股（波動大、\
上檔空間大）。文章長度與專業度本身不是訊號——長篇分析若標的無波動潛力，MPP照樣低。
8. 貼文附上對帳單、財務圖表連結（imgur等），代表作者有證據意識，屬正面訊號。
9. 兩篇都看多時，比較誰的標的波動潛力大、動能更強、訊號更早期；兩篇都空泛時，\
比較誰透露的當下動能較強。

請輸出JSON格式：{"analysis": "比較兩檔標的的上漲潛力（50字內）", "better": "A" 或 "B", "confidence": 0.5到1.0之間的數字}"""

# Few-shot exemplars are real pairs from the official training release
# (labels = true MPP ordering), chosen to teach the counterintuitive cases
# the v1 rubric got wrong. Because the post texts belong to the organizers'
# restricted release, they are NOT distributed with this repository: the
# rationales, labels, and confidences below are ours, while the paired post
# texts must be supplied in data/fewshot_config.json as
#   {"posts": [{"a": "<post text>", "b": "<post text>"}, ...5 items...],
#    "exclude_snippets": ["<distinctive substring per exemplar post>", ...]}
# See README.md ("Few-shot exemplars") for how to obtain or reconstruct it.
FEWSHOT_META = [
    {
        # insider-teaser dream post beats earnest construction DD
        "better": "B",
        "analysis": "B是典型作夢文，暗示內線消息，玉晶光屬高波動蘋概題材股，爆發力強；A雖詳盡但營建股牛皮、催化劑在季報後才發酵。",
        "confidence": 0.7,
    },
    {
        # bearish-but-correct analysis loses to contrarian dividend hype
        "better": "B",
        "analysis": "A實質看空且股價正下跌，做多MPP必低；B雖粗糙但有除權息催化劑、散戶恐慌沈澱籌碼的反轉邏輯。",
        "confidence": 0.7,
    },
    {
        # bullish specifics + near catalyst beats bearish slogans
        "better": "A",
        "analysis": "A看多且有獲利上修、主力吃貨、殖利率保護，動能與催化劑俱足；B全面看空，做多MPP低。",
        "confidence": 0.85,
    },
    {
        # squeeze/chips signal beats long macro essay on quiet stock
        "better": "B",
        "analysis": "B透露守穩長紅、空單回補的軋空結構，短線動能明確；A是十年期宏觀故事，無近期催化劑。",
        "confidence": 0.65,
    },
    {
        # risk-aware bullish stock analysis beats complete-looking index post
        "better": "A",
        "analysis": "A是個股的風控式看多：業績翻倍+摸底翻紅，保守措辭不減其品質；B是大盤籌碼文且描述已發生的行情，指數波動小、訊號偏晚，MPP天生低。",
        "confidence": 0.8,
    },
]

_FEWSHOT_CFG = DATA / "fewshot_config.json"
if _FEWSHOT_CFG.exists():
    _cfg = json.loads(_FEWSHOT_CFG.read_text(encoding="utf-8"))
    FEWSHOT = [{**meta, **posts} for meta, posts in zip(FEWSHOT_META, _cfg["posts"])]
    FEWSHOT_SNIPPETS = list(_cfg["exclude_snippets"])
else:
    FEWSHOT = []
    FEWSHOT_SNIPPETS = []

_cache_lock = threading.Lock()


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_FILE)


def build_messages(post_a, post_b, refs_text=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOT:
        msgs.append({
            "role": "user",
            "content": f"貼文A：\n{ex['a']}\n\n貼文B：\n{ex['b']}",
        })
        msgs.append({
            "role": "assistant",
            "content": json.dumps(
                {"analysis": ex["analysis"], "better": ex["better"],
                 "confidence": ex["confidence"]},
                ensure_ascii=False),
        })
    content = f"貼文A：\n{post_a[:MAX_POST_CHARS]}\n\n貼文B：\n{post_b[:MAX_POST_CHARS]}"
    if refs_text:
        content += (
            "\n\n【歷史參照】以下是與兩篇貼文「寫法相似」的歷史貼文與其後續實際MPP，"
            "供校準參考（相似寫法不保證相同結果，仍以貼文本身訊號為主）：\n" + refs_text
        )
    msgs.append({"role": "user", "content": content})
    return msgs


class KnnIndex:
    """Cosine retrieval over frozen e5 embeddings of the labeled posts."""

    def __init__(self, posts):
        import numpy as np
        z = np.load(DATA / "emb_multilingual-e5-large.npz", allow_pickle=True)
        emb = {k: v for k, v in zip(z["keys"], z["vecs"])}
        self.np = np
        self.posts = posts                      # pid -> post dict
        self.pids = sorted(posts)
        self.mat = np.vstack([emb[f"train:{pid}"] for pid in self.pids])
        self.emb = emb

    def refs_for(self, query_key, exclude_fold=None, exclude_pid=None, k=2):
        sims = self.mat @ self.emb[query_key]
        order = self.np.argsort(-sims)
        out = []
        for i in order:
            pid = self.pids[i]
            p = self.posts[pid]
            if exclude_pid is not None and pid == exclude_pid:
                continue
            if exclude_fold is not None and p["fold"] == exclude_fold:
                continue
            out.append(f"「{p['text'][:120]}」→ 後續MPP {p['mpp']:+.2f}")
            if len(out) == k:
                break
        return out

    def pair_refs(self, key_a, key_b, exclude_fold=None,
                  exclude_pid_a=None, exclude_pid_b=None):
        ra = self.refs_for(key_a, exclude_fold, exclude_pid_a)
        rb = self.refs_for(key_b, exclude_fold, exclude_pid_b)
        lines = ["與貼文A相似："] + ra + ["與貼文B相似："] + rb
        return "\n".join(lines)


def call_llm(client, model, messages, cache, vote_id=0, effort=None):
    key = hashlib.sha256(
        json.dumps([model, vote_id, effort, messages], ensure_ascii=False).encode()
    ).hexdigest()
    with _cache_lock:
        if key in cache:
            return cache[key]
    kwargs = dict(model=model, messages=messages,
                  response_format={"type": "json_object"})
    if effort:
        kwargs["reasoning_effort"] = effort
    for attempt in range(7):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            break
        except Exception as e:  # drop unsupported params once, else backoff
            msg = str(e)
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            if attempt == 6:
                raise
            import time
            # long enough to ride out transient DNS/network blips
            time.sleep(min(2 ** attempt, 30))
    with _cache_lock:
        cache[key] = text
        if len(cache) % 50 == 0:
            save_cache(cache)
    return text


def parse_verdict(text):
    """Return P(A better) in [0,1], or None on parse failure."""
    try:
        obj = json.loads(text)
        better = str(obj.get("better", "")).strip().upper()
        conf = float(obj.get("confidence", 0.75))
        conf = min(max(conf, 0.5), 1.0)
    except (json.JSONDecodeError, TypeError, ValueError):
        m = re.search(r'"better"\s*:\s*"?([AB])', text or "")
        if not m:
            return None
        better, conf = m.group(1), 0.75
    if better == "A":
        return conf
    if better == "B":
        return 1.0 - conf
    return None


def judge_pair(client, model, post1, post2, cache, votes=1,
               refs=None, refs_swapped=None, effort=None):
    """Return P(post1 better), averaging both orderings x votes."""
    probs = []
    for order in ("12", "21"):
        a, b = (post1, post2) if order == "12" else (post2, post1)
        r = refs if order == "12" else refs_swapped
        msgs = build_messages(a, b, refs_text=r)
        for v in range(votes):
            p_a = parse_verdict(call_llm(client, model, msgs, cache, vote_id=v, effort=effort))
            if p_a is None:
                continue
            probs.append(p_a if order == "12" else 1.0 - p_a)
    return sum(probs) / len(probs) if probs else 0.5


def mode_cv(client, args, cache):
    posts = {p["pid"]: p for p in json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))}
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    fewshot_pids = {
        pid for pid, p in posts.items()
        if any(s in p["text"] for s in FEWSHOT_SNIPPETS)
    }
    eval_pairs = [
        p for p in pairs
        if p["a_fold"] == p["b_fold"] and abs(p["dmpp"]) >= args.min_dmpp
        and p["a_pid"] not in fewshot_pids and p["b_pid"] not in fewshot_pids
    ]
    print(f"excluded {len(fewshot_pids)} few-shot posts from CV evaluation")
    rng = random.Random(123)
    sampled = []
    for fold in range(N_FOLDS):
        fp = [p for p in eval_pairs if p["a_fold"] == fold]
        rng.shuffle(fp)
        sampled += fp[: args.sample]
    print(f"judging {len(sampled)} CV pairs with {args.model} (votes={args.votes}, knn={args.knn}) ...")
    knn = KnnIndex(posts) if args.knn else None

    def work(p):
        try:
            refs = refs_sw = None
            if knn:
                # leakage guard: retrieve only from posts OUTSIDE the pair's fold
                ka, kb = f"train:{p['a_pid']}", f"train:{p['b_pid']}"
                refs = knn.pair_refs(ka, kb, exclude_fold=p["a_fold"])
                refs_sw = knn.pair_refs(kb, ka, exclude_fold=p["a_fold"])
            prob = judge_pair(client, args.model, posts[p["a_pid"]]["text"],
                              posts[p["b_pid"]]["text"], cache, votes=args.votes,
                              refs=refs, refs_swapped=refs_sw, effort=args.effort)
        except Exception as e:
            print(f"  pair ({p['a_pid']},{p['b_pid']}) failed: {type(e).__name__}")
            prob = None
        return p, prob

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for p, prob in ex.map(work, sampled):
            if prob is not None:
                results.append((p, prob))
    save_cache(cache)
    if len(results) < len(sampled):
        print(f"WARNING: {len(sampled)-len(results)} pairs failed and were skipped")

    correct = sum(1 for p, prob in results if (prob > 0.5) == (p["label"] == 1))
    acc = correct / len(results)
    print(f"CV accuracy: {acc:.4f} on {len(results)} pairs")
    by_fold = {}
    for p, prob in results:
        by_fold.setdefault(p["a_fold"], []).append((prob > 0.5) == (p["label"] == 1))
    for f in sorted(by_fold):
        v = by_fold[f]
        print(f"  fold {f}: {sum(v)/len(v):.3f} ({len(v)} pairs)")
    # store per-pair probabilities for later ensembling
    out = DATA / f"judge_cv_{args.model.replace('/','_')}{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(
        [{"a_pid": p["a_pid"], "b_pid": p["b_pid"], "label": p["label"],
          "fold": p["a_fold"], "p1": prob} for p, prob in results],
        indent=1), encoding="utf-8")
    print("saved:", out.name)


def mode_test(client, args, cache):
    test = json.loads((DATA / "test_pairs.json").read_text(encoding="utf-8"))
    print(f"judging {len(test)} test pairs with {args.model} (votes={args.votes}, knn={args.knn}) ...")
    knn = None
    if args.knn:
        posts = {p["pid"]: p for p in json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))}
        knn = KnnIndex(posts)

    def work(t):
        try:
            refs = refs_sw = None
            if knn:
                k1, k2 = f"test:{t['index']}:1", f"test:{t['index']}:2"
                refs = knn.pair_refs(k1, k2)
                refs_sw = knn.pair_refs(k2, k1)
            prob = judge_pair(client, args.model, t["post1"], t["post2"], cache,
                              votes=args.votes, refs=refs, refs_swapped=refs_sw, effort=args.effort)
        except Exception as e:
            print(f"  pair {t['index']} failed: {type(e).__name__}")
            prob = None
        return t["index"], prob

    probs, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, prob in ex.map(work, test):
            if prob is None:
                failed.append(idx)
            else:
                probs[idx] = prob
    # test predictions must be complete: retry failures serially
    for t in test:
        if t["index"] in failed:
            probs[t["index"]] = judge_pair(client, args.model, t["post1"],
                                           t["post2"], cache, votes=args.votes)
    save_cache(cache)

    out = DATA / f"judge_test_{args.model.replace('/','_')}{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(probs, indent=1), encoding="utf-8")
    n1 = sum(1 for v in probs.values() if v > 0.5)
    print(f"predictions: Post 1 chosen {n1}/{len(probs)} times")
    print("saved:", out.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "test"], required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--sample", type=int, default=30, help="CV pairs per fold")
    ap.add_argument("--min_dmpp", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. v2")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high"],
                    help="reasoning effort for gpt-5 family models")
    ap.add_argument("--knn", action="store_true",
                    help="append retrieved similar posts with realized MPP")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY environment variable first.")
    if not FEWSHOT:
        raise SystemExit(
            "data/fewshot_config.json not found. The five few-shot exemplar "
            "posts come from the restricted official training release and are "
            "not distributed here; see README.md ('Few-shot exemplars').")
    client = OpenAI()
    cache = load_cache()
    if args.mode == "cv":
        mode_cv(client, args, cache)
    else:
        mode_test(client, args, cache)


if __name__ == "__main__":
    main()
