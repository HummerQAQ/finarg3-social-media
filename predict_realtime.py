# -*- coding: utf-8 -*-
"""Real-time week daily inference — one command, pairs in, predictions out.

Implements exactly the Run-1 logic (consistency-filtered majority vote of the
three frozen tracks) with graceful degradation:
  - judge order-inconsistent on a pair  -> judge abstains (LR+encoder decide)
  - API unreachable / --no-api          -> pure local blend for all pairs

Usage:
  python predict_realtime.py --input day1_pairs.json
  # output: day1_pairs.predictions.json + day1_pairs.audit.json
"""
import argparse
import json
import sys
import unicodedata
import re
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from features import feature_vector
import llm_judge

HERE = Path(__file__).parent
DATA = HERE / "data"
CKPT_GLOB = "final_durpft_s*"
MAX_LEN = 256


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def load_input(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for i, it in enumerate(raw):
        key = it.get("index", it.get("id", i))
        items.append({"key": key, "post1": normalize(it["post1"]),
                      "post2": normalize(it["post2"]), "orig": it})
    return items


def lr_track(posts, pairs, items):
    texts = {p["pid"]: p["text"] for p in posts}
    pids = sorted(texts)
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                            max_features=20000, sublinear_tf=True)
    tfidf.fit([texts[pid] for pid in pids])
    fvs = np.vstack([feature_vector(texts[pid]) for pid in pids])
    mu, sd = fvs.mean(0), fvs.std(0) + 1e-9

    def vec(t):
        return sparse.hstack([tfidf.transform([t]),
                              sparse.csr_matrix((np.array(feature_vector(t)) - mu) / sd)])

    vtr = {pid: vec(texts[pid]) for pid in pids}
    X, y, w = [], [], []
    for p in pairs:
        X.append(vtr[p["a_pid"]] - vtr[p["b_pid"]]); y.append(p["label"]); w.append(abs(p["dmpp"]))
        X.append(vtr[p["b_pid"]] - vtr[p["a_pid"]]); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sparse.vstack(X), y, sample_weight=w)
    return {it["key"]: float(clf.predict_proba(vec(it["post1"]) - vec(it["post2"]))[0, 1])
            for it in items}


class Scorer(torch.nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ckpt)
        hidden = self.encoder.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(hidden, 1))
        self.head.load_state_dict(torch.load(Path(ckpt) / "head.pt", map_location="cpu"))

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


@torch.no_grad()
def encoder_track(posts, items):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted((HERE / "models").glob(CKPT_GLOB))
    if not ckpts:
        raise SystemExit("no final encoder checkpoints — run train_encoder.py --final --save_model")
    anchor_texts = [p["text"] for p in posts]              # stable z-norm anchor
    new_texts = [it[k] for it in items for k in ("post1", "post2")]
    probs_parts = []
    for ck in ckpts:
        tok = AutoTokenizer.from_pretrained(ck)
        model = Scorer(str(ck)).to(device).eval()
        def score(texts):
            out = []
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i+32], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to(device)
                out += model(enc["input_ids"], enc["attention_mask"]).float().cpu().tolist()
            return np.array(out)
        anchor = score(anchor_texts)
        mu, sd = anchor.mean(), anchor.std() + 1e-9
        s = (score(new_texts) - mu) / sd
        probs_parts.append(s)
        del model
        torch.cuda.empty_cache()
    s = np.mean(probs_parts, axis=0)
    out = {}
    for j, it in enumerate(items):
        d = s[2 * j] - s[2 * j + 1]
        out[it["key"]] = float(1 / (1 + np.exp(-d)))
    print(f"encoder track: {len(ckpts)} checkpoints")
    return out


def judge_track(items, model_name, workers):
    """Returns (prob_map, consistency_map); on total failure returns ({}, {})."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI
    client = OpenAI()
    cache = llm_judge.load_cache()
    probs, cons = {}, {}

    def one(it):
        msgs_ab = llm_judge.build_messages(it["post1"], it["post2"])
        msgs_ba = llm_judge.build_messages(it["post2"], it["post1"])
        p_ab = llm_judge.parse_verdict(llm_judge.call_llm(client, model_name, msgs_ab, cache))
        p_ba = llm_judge.parse_verdict(llm_judge.call_llm(client, model_name, msgs_ba, cache))
        if p_ab is None or p_ba is None:
            return it["key"], None, None
        prob = (p_ab + (1 - p_ba)) / 2
        consistent = (p_ab > 0.5) == ((1 - p_ba) > 0.5)
        return it["key"], prob, consistent

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, prob, consistent in ex.map(one, items):
            if prob is not None:
                probs[key] = prob
                cons[key] = consistent
    llm_judge.save_cache(cache)
    return probs, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--judge_model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-api", action="store_true", help="local tracks only")
    args = ap.parse_args()

    items = load_input(args.input)
    print(f"loaded {len(items)} pairs from {args.input}")
    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))

    p_lr = lr_track(posts, pairs, items)
    p_enc = encoder_track(posts, items)
    p_j, cons = ({}, {})
    if not args.no_api:
        try:
            p_j, cons = judge_track(items, args.judge_model, args.workers)
            print(f"judge track: {len(p_j)}/{len(items)} judged, "
                  f"{sum(cons.values())} order-consistent")
        except Exception as e:
            print(f"WARNING: judge track failed entirely ({type(e).__name__}) — local fallback")

    out_items, audit = [], []
    for it in items:
        k = it["key"]
        lr, en = p_lr[k], p_enc[k]
        if k in p_j and cons.get(k, False):
            votes = int(lr > .5) + int(en > .5) + int(p_j[k] > .5)
            pred1 = votes >= 2
            mode = "majority3"
        else:
            pred1 = (lr + en) / 2 > 0.5
            mode = "local_blend" if k not in p_j else "judge_abstain"
        o = dict(it["orig"])
        o["prediction"] = "Post 1" if pred1 else "Post 2"
        out_items.append(o)
        audit.append({"key": k, "p_lr": round(lr, 4), "p_enc": round(en, 4),
                      "p_judge": round(p_j.get(k, -1), 4), "mode": mode,
                      "prediction": o["prediction"]})

    inp = Path(args.input)
    out_f = inp.with_suffix(".predictions.json")
    out_f.write_text(json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")
    inp.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    n1 = sum(1 for o in out_items if o["prediction"] == "Post 1")
    print(f"done: Post 1 = {n1}/{len(out_items)} -> {out_f}")


if __name__ == "__main__":
    main()
