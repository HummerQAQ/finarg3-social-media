# -*- coding: utf-8 -*-
"""Top-210 task: local-track scores for all 2,100 collection posts.

Encoder: 5 durpft checkpoints, z-normed against the 91 training posts.
LR: pair model decomposed to per-post score w·f(x).
Output: data/ranking_scores.json {post_id: {"enc": z, "lr": z}}
"""
import json
import sys
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from features import feature_vector
from predict_realtime import Scorer, normalize

HERE = Path(__file__).parent
DATA = HERE / "data"
COLLECTION = HERE / "realtime" / "data" / "FinArg3_Social_Media_Post_Ranking_2100_Collection.json"


def main():
    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    col = json.loads(COLLECTION.read_text(encoding="utf-8"))
    texts = [normalize(it["post"]) for it in col]
    pids = [it["post_id"] for it in col]

    # --- LR per-post scores ---
    tr_texts = {p["pid"]: p["text"] for p in posts}
    tr_pids = sorted(tr_texts)
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                            max_features=20000, sublinear_tf=True)
    tfidf.fit([tr_texts[p] for p in tr_pids])
    fvs = np.vstack([feature_vector(tr_texts[p]) for p in tr_pids])
    mu, sd = fvs.mean(0), fvs.std(0) + 1e-9

    def vec(t):
        return sparse.hstack([tfidf.transform([t]),
                              sparse.csr_matrix((np.array(feature_vector(t)) - mu) / sd)])

    vtr = {p: vec(tr_texts[p]) for p in tr_pids}
    X, y, w = [], [], []
    for p in pairs:
        X.append(vtr[p["a_pid"]] - vtr[p["b_pid"]]); y.append(p["label"]); w.append(abs(p["dmpp"]))
        X.append(vtr[p["b_pid"]] - vtr[p["a_pid"]]); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sparse.vstack(X), y, sample_weight=w)
    coef = sparse.csr_matrix(clf.coef_[0])
    lr_anchor = np.array([float(coef.multiply(vtr[p]).sum()) for p in tr_pids])
    lmu, lsd = lr_anchor.mean(), lr_anchor.std() + 1e-9
    lr_scores = [(float(coef.multiply(vec(t)).sum()) - lmu) / lsd for t in texts]
    print("LR scores done")

    # --- encoder ensemble scores ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anchor_texts = [tr_texts[p] for p in tr_pids]
    enc_parts = []
    for ck in sorted((HERE / "models").glob("final_durpft_s*")):
        tok = AutoTokenizer.from_pretrained(ck)
        model = Scorer(str(ck)).to(device).eval()

        @torch.no_grad()
        def score(ts):
            out = []
            for i in range(0, len(ts), 32):
                enc = tok(ts[i:i+32], truncation=True, max_length=256,
                          padding=True, return_tensors="pt").to(device)
                out += model(enc["input_ids"], enc["attention_mask"]).float().cpu().tolist()
            return np.array(out)

        a = score(anchor_texts)
        amu, asd = a.mean(), a.std() + 1e-9
        enc_parts.append((score(texts) - amu) / asd)
        del model
        torch.cuda.empty_cache()
        print(f"encoder {ck.name} done")
    enc_scores = np.mean(enc_parts, axis=0)

    out = {pid: {"enc": float(e), "lr": float(l)}
           for pid, e, l in zip(pids, enc_scores, lr_scores)}
    (DATA / "ranking_scores.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved ranking_scores.json for {len(out)} posts")


if __name__ == "__main__":
    main()
