# -*- coding: utf-8 -*-
"""Baseline 1: handcrafted features + char n-gram TF-IDF, pairwise logistic
regression, evaluated with the grouped 5-fold CV protocol.

Pairwise input = feature_vector(a) - feature_vector(b); label 1 iff a has
higher MPP. Both orderings of every pair are included (antisymmetric model).
"""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from features import FEATURE_NAMES, feature_vector

DATA = Path(__file__).parent / "data"
N_FOLDS = 5


def load():
    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    return {p["pid"]: p for p in posts}, pairs


def run_cv(use_tfidf=True, use_feats=True, min_dmpp=0.02, C=0.1):
    posts, pairs = load()
    pairs = [p for p in pairs if abs(p["dmpp"]) >= min_dmpp]
    accs = []
    for fold in range(N_FOLDS):
        train = [p for p in pairs if p["a_fold"] != fold and p["b_fold"] != fold]
        evalp = [p for p in pairs if p["a_fold"] == fold and p["b_fold"] == fold]
        if not evalp:
            continue

        train_pids = sorted({q for p in train for q in (p["a_pid"], p["b_pid"])})
        texts = {pid: posts[pid]["text"] for pid in posts}

        blocks = []
        if use_tfidf:
            tfidf = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 3), min_df=2, max_features=20000,
                sublinear_tf=True,
            )
            tfidf.fit([texts[pid] for pid in train_pids])
            tf_vecs = {pid: tfidf.transform([texts[pid]]) for pid in texts}
            blocks.append(("tfidf", tf_vecs))
        if use_feats:
            fv = {pid: np.array(feature_vector(texts[pid])) for pid in texts}
            # standardize on train posts
            mat = np.vstack([fv[pid] for pid in train_pids])
            mu, sd = mat.mean(0), mat.std(0) + 1e-9
            fv = {pid: sparse.csr_matrix((v - mu) / sd) for pid, v in fv.items()}
            blocks.append(("feats", fv))

        def pair_x(a_pid, b_pid):
            parts = [vecs[a_pid] - vecs[b_pid] for _, vecs in blocks]
            return sparse.hstack(parts) if len(parts) > 1 else parts[0]

        X_tr, y_tr, w_tr = [], [], []
        for p in train:
            # both orderings -> antisymmetric decision function
            X_tr.append(pair_x(p["a_pid"], p["b_pid"])); y_tr.append(p["label"])
            X_tr.append(pair_x(p["b_pid"], p["a_pid"])); y_tr.append(1 - p["label"])
            w = abs(p["dmpp"])  # weight noisy small-gap pairs less
            w_tr += [w, w]
        X_tr = sparse.vstack(X_tr)

        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)

        X_ev = sparse.vstack([pair_x(p["a_pid"], p["b_pid"]) for p in evalp])
        pred = clf.predict_proba(X_ev)[:, 1] > 0.5
        acc = float(np.mean(pred == np.array([p["label"] for p in evalp])))
        accs.append((fold, acc, len(evalp)))

    total = sum(n for _, _, n in accs)
    wavg = sum(a * n for _, a, n in accs) / total
    return accs, wavg


if __name__ == "__main__":
    for cfg in [
        dict(use_tfidf=False, use_feats=True),
        dict(use_tfidf=True, use_feats=False),
        dict(use_tfidf=True, use_feats=True),
    ]:
        accs, wavg = run_cv(**cfg)
        name = "+".join(k.replace("use_", "") for k, v in cfg.items() if v)
        print(f"{name:15s} folds: " + " ".join(f"{a:.3f}" for _, a, _ in accs) + f"  | weighted avg = {wavg:.4f}")
