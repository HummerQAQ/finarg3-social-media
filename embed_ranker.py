# -*- coding: utf-8 -*-
"""T2: frozen-embedding ranker — 4th ensemble member candidate.

Frozen multilingual sentence embeddings + logistic regression on pair diffs.
With n=91 posts, frozen features + tiny head is the right bias-variance point,
and the contrastive-pretrained embedding space is architecturally unrelated to
MacBERT (MLM), so errors should decorrelate from the encoder track.

Usage:
  python embed_ranker.py --model intfloat/multilingual-e5-large
  python embed_ranker.py --model BAAI/bge-m3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

DATA = Path(__file__).parent / "data"
N_FOLDS = 5


def embed_texts(model_name, texts, batch=16, max_len=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    prefix = "query: " if "e5" in model_name.lower() else ""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [prefix + t for t in texts[i:i + batch]]
            enc = tok(chunk, truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            if "bge" in model_name.lower():
                v = h[:, 0]                     # CLS pooling
            else:
                mask = enc["attention_mask"].unsqueeze(-1)
                v = (h * mask).sum(1) / mask.sum(1)  # mean pooling
            v = torch.nn.functional.normalize(v, dim=-1)
            out.append(v.float().cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test_pairs.json").read_text(encoding="utf-8"))

    cache_f = DATA / f"emb_{tag}.npz"
    keys = [f"train:{p['pid']}" for p in posts]
    texts = [p["text"] for p in posts]
    for t in test:
        keys += [f"test:{t['index']}:1", f"test:{t['index']}:2"]
        texts += [t["post1"], t["post2"]]
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        emb = {k: v for k, v in zip(z["keys"], z["vecs"])}
        print(f"loaded cached embeddings: {len(emb)}")
    else:
        vecs = embed_texts(args.model, texts)
        emb = dict(zip(keys, vecs))
        np.savez(cache_f, keys=np.array(keys), vecs=np.vstack(list(emb.values())))
        print(f"embedded {len(emb)} posts with {args.model}")

    ev = {p["pid"]: emb[f"train:{p['pid']}"] for p in posts}

    # grouped 5-fold CV, OOF scores saved for stacking
    oof = {}
    accs = []
    for fold in range(N_FOLDS):
        train = [p for p in pairs if p["a_fold"] != fold and p["b_fold"] != fold]
        evalp = [p for p in pairs if p["a_fold"] == fold and p["b_fold"] == fold]
        X, y, w = [], [], []
        for p in train:
            d = ev[p["a_pid"]] - ev[p["b_pid"]]
            X.append(d); y.append(p["label"]); w.append(abs(p["dmpp"]))
            X.append(-d); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
        clf = LogisticRegression(C=args.C, max_iter=3000).fit(
            np.vstack(X), y, sample_weight=w)
        # per-post score = w·emb (linear model => decompose pair prob)
        coef = clf.coef_[0]
        fold_pids = {p["pid"] for p in posts if p["fold"] == fold}
        for pid in fold_pids:
            oof[pid] = float(ev[pid] @ coef)
        ok = [
            ((ev[p["a_pid"]] @ coef) > (ev[p["b_pid"]] @ coef)) == (p["label"] == 1)
            for p in evalp
        ]
        accs.append((float(np.mean(ok)), len(evalp)))

    total = sum(n for _, n in accs)
    wavg = sum(a * n for a, n in accs) / total
    print(f"{tag} CV acc = {wavg:.4f} | folds: " + " ".join(f"{a:.3f}" for a, _ in accs))

    # full-train model for test scoring
    X, y, w = [], [], []
    for p in pairs:
        d = ev[p["a_pid"]] - ev[p["b_pid"]]
        X.append(d); y.append(p["label"]); w.append(abs(p["dmpp"]))
        X.append(-d); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
    clf = LogisticRegression(C=args.C, max_iter=3000).fit(np.vstack(X), y, sample_weight=w)
    coef = clf.coef_[0]
    test_scores = {}
    for t in test:
        for s in ("1", "2"):
            k = f"test:{t['index']}:{s}"
            test_scores[k] = float(emb[k] @ coef)

    out = DATA / f"embrank_{tag}.json"
    out.write_text(json.dumps({
        "model": args.model, "cv_acc": wavg,
        "per_fold": [a for a, _ in accs],
        "oof_scores": {str(k): v for k, v in oof.items()},
        "test_scores": test_scores,
    }, indent=1), encoding="utf-8")
    print("saved:", out.name)


if __name__ == "__main__":
    main()
