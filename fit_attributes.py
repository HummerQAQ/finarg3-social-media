# -*- coding: utf-8 -*-
"""Fit learned weights on LLM-extracted attributes and evaluate under the
grouped 5-fold CV protocol. Compares:
  A) attributes only
  B) attributes + lexicon features
  C) attributes + lexicon features + char TF-IDF
and shows learned attribute weights for interpretability.

Usage: python fit_attributes.py --attrs data/attrs_gpt-5-mini.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from extract_attributes import FIELDS, NARRATIVES
from features import feature_vector

DATA = Path(__file__).parent / "data"
ATTR_NAMES = list(FIELDS.keys()) + [f"nar_{n}" for n in NARRATIVES]


def attr_vec(attrs):
    base = [attrs[k] for k in FIELDS]
    nar = attrs.get("narrative", "other")
    base += [1.0 if nar == n else 0.0 for n in NARRATIVES]
    return np.array(base, dtype=float)


def run_cv(posts, pairs, attr_by_pid, use_feats=False, use_tfidf=False, C=0.1):
    texts = {pid: p["text"] for pid, p in posts.items()}
    accs, all_coef = [], []
    for fold in range(5):
        train = [p for p in pairs if p["a_fold"] != fold and p["b_fold"] != fold]
        evalp = [p for p in pairs if p["a_fold"] == fold and p["b_fold"] == fold]
        train_pids = sorted({q for p in train for q in (p["a_pid"], p["b_pid"])})

        blocks = {}
        av = {pid: attr_vec(attr_by_pid[pid]) for pid in posts}
        mat = np.vstack([av[pid] for pid in train_pids])
        mu, sd = mat.mean(0), mat.std(0) + 1e-9
        blocks["attrs"] = {pid: sparse.csr_matrix((v - mu) / sd) for pid, v in av.items()}
        if use_feats:
            fv = {pid: np.array(feature_vector(texts[pid])) for pid in posts}
            mat = np.vstack([fv[pid] for pid in train_pids])
            mu, sd = mat.mean(0), mat.std(0) + 1e-9
            blocks["feats"] = {pid: sparse.csr_matrix((v - mu) / sd) for pid, v in fv.items()}
        if use_tfidf:
            tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                                    max_features=20000, sublinear_tf=True)
            tfidf.fit([texts[pid] for pid in train_pids])
            blocks["tfidf"] = {pid: tfidf.transform([texts[pid]]) for pid in posts}

        def px(a, b):
            parts = [vecs[a] - vecs[b] for vecs in blocks.values()]
            return sparse.hstack(parts) if len(parts) > 1 else parts[0]

        X, y, w = [], [], []
        for p in train:
            X.append(px(p["a_pid"], p["b_pid"])); y.append(p["label"]); w.append(abs(p["dmpp"]))
            X.append(px(p["b_pid"], p["a_pid"])); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
        clf = LogisticRegression(C=C, max_iter=2000).fit(sparse.vstack(X), y, sample_weight=w)
        if not use_feats and not use_tfidf:
            all_coef.append(clf.coef_[0][: len(ATTR_NAMES)])

        Xe = sparse.vstack([px(p["a_pid"], p["b_pid"]) for p in evalp])
        pred = clf.predict_proba(Xe)[:, 1] > 0.5
        acc = float(np.mean(pred == np.array([p["label"] for p in evalp])))
        accs.append((acc, len(evalp)))

    total = sum(n for _, n in accs)
    wavg = sum(a * n for a, n in accs) / total
    return wavg, [a for a, _ in accs], (np.mean(all_coef, axis=0) if all_coef else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="data/attrs_v2_gpt-5-mini.json")
    args = ap.parse_args()

    posts = {p["pid"]: p for p in json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))}
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    raw = json.loads(Path(args.attrs).read_text(encoding="utf-8"))
    attr_by_pid = {}
    for pid in posts:
        a = raw.get(f"train:{pid}")
        if a is None:
            a = {k: 0.0 for k in FIELDS} | {"narrative": "other"}
        attr_by_pid[pid] = a

    for name, kw in [
        ("attrs only", {}),
        ("attrs+feats", dict(use_feats=True)),
        ("attrs+feats+tfidf", dict(use_feats=True, use_tfidf=True)),
    ]:
        wavg, folds, coef = run_cv(posts, pairs, attr_by_pid, **kw)
        print(f"{name:20s} CV acc = {wavg:.4f} | folds: " + " ".join(f"{a:.3f}" for a in folds))
        if coef is not None:
            order = np.argsort(-np.abs(coef))
            print("  learned attribute weights (desc |w|):")
            for i in order:
                print(f"    {coef[i]:+.3f}  {ATTR_NAMES[i]}")


if __name__ == "__main__":
    main()
