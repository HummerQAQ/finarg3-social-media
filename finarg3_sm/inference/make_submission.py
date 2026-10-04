# -*- coding: utf-8 -*-
"""Produce the three official submission runs for the Social Media Subtask.

Run 1: majority vote of LR + encoder-final ensemble + LLM judge  (best CV ~0.733)
Run 2: LR + encoder blend, local models only                     (no API dependence)
Run 3: majority vote of LR + encoder + attribute model           (judge-free, CV ~0.727)

Each run is written as a copy of the ORIGINAL organizer test file with the
required "prediction": "Post 1" / "Post 2" key added per instance.

Usage: python -m finarg3_sm.inference.make_submission
"""
import json
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT, required_input_path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from finarg3_sm.preprocessing.features import feature_vector
from finarg3_sm.training.fit_attributes import attr_vec

HERE = PROJECT_ROOT
DATA = HERE / "data"
JUDGE_TEST = DATA / "judge_test_gpt-5-mini_v3.json"
ATTRS_FILE = DATA / "attrs_v2_gpt-5-mini.json"
OUT = HERE / "submissions"

W_ENC_RUN2 = 0.6  # from CV grid: LR+enc blend peaked around w_enc 0.5-0.7


def lr_test_probs(posts, pairs, test):
    """Train the LR track on ALL pairs, return P(post1 better) per test index."""
    texts = {p["pid"]: p["text"] for p in posts}
    all_pids = sorted(texts)
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                            max_features=20000, sublinear_tf=True)
    tfidf.fit([texts[pid] for pid in all_pids])
    fvs = {pid: np.array(feature_vector(texts[pid])) for pid in all_pids}
    mat = np.vstack([fvs[pid] for pid in all_pids])
    mu, sd = mat.mean(0), mat.std(0) + 1e-9

    def vec(text):
        tf = tfidf.transform([text])
        fv = sparse.csr_matrix((np.array(feature_vector(text)) - mu) / sd)
        return sparse.hstack([tf, fv])

    vtr = {pid: vec(texts[pid]) for pid in all_pids}
    X, y, w = [], [], []
    for p in pairs:
        X.append(vtr[p["a_pid"]] - vtr[p["b_pid"]]); y.append(p["label"]); w.append(abs(p["dmpp"]))
        X.append(vtr[p["b_pid"]] - vtr[p["a_pid"]]); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sparse.vstack(X), y, sample_weight=w)

    probs = {}
    for t in test:
        x = vec(t["post1"]) - vec(t["post2"])
        probs[t["index"]] = float(clf.predict_proba(x)[0, 1])
    return probs


def encoder_test_probs(test):
    """Average z-normalized scores across the pre-finetuned final encoder runs.
    (durpft ensemble beats the base-MacBERT ensemble on CV: 0.681 vs 0.668,
    and is better on short pairs — the test length distribution.)"""
    runs = sorted(DATA.glob("encoder_final_durpft_*.json"))
    if not runs:
        raise SystemExit("no encoder_final_*.json found — run train_encoder.py --final first")
    print(f"encoder final runs: {len(runs)}")
    norm = []
    for r in runs:
        s = json.loads(r.read_text(encoding="utf-8"))["scores"]
        v = np.array(list(s.values())); mu, sd = v.mean(), v.std() + 1e-9
        norm.append({k: (x - mu) / sd for k, x in s.items()})
    avg = {k: float(np.mean([s[k] for s in norm])) for k in norm[0]}
    probs = {}
    for t in test:
        d = avg[f"test:{t['index']}:1"] - avg[f"test:{t['index']}:2"]
        probs[t["index"]] = float(1 / (1 + np.exp(-d)))
    return probs


def attr_test_probs(posts, pairs, test):
    """Learned-weight attribute model trained on ALL pairs -> test probs."""
    raw = json.loads(ATTRS_FILE.read_text(encoding="utf-8"))
    av = {f"train:{p['pid']}": attr_vec(raw[f"train:{p['pid']}"]) for p in posts}
    for t in test:
        for s in ("1", "2"):
            k = f"test:{t['index']}:{s}"
            av[k] = attr_vec(raw[k])
    train_keys = [f"train:{p['pid']}" for p in posts]
    mat = np.vstack([av[k] for k in train_keys])
    mu, sd = mat.mean(0), mat.std(0) + 1e-9
    nav = {k: (v - mu) / sd for k, v in av.items()}

    X, y, w = [], [], []
    for p in pairs:
        a, b = f"train:{p['a_pid']}", f"train:{p['b_pid']}"
        X.append(nav[a] - nav[b]); y.append(p["label"]); w.append(abs(p["dmpp"]))
        X.append(nav[b] - nav[a]); y.append(1 - p["label"]); w.append(abs(p["dmpp"]))
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(np.vstack(X), y, sample_weight=w)
    probs = {}
    for t in test:
        x = (nav[f"test:{t['index']}:1"] - nav[f"test:{t['index']}:2"]).reshape(1, -1)
        probs[t["index"]] = float(clf.predict_proba(x)[0, 1])
    return probs


def main():
    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test_pairs.json").read_text(encoding="utf-8"))
    orig = json.loads(required_input_path("FINARG3_ORIGINAL_TEST").read_text(encoding="utf-8"))

    p_lr = lr_test_probs(posts, pairs, test)
    p_enc = encoder_test_probs(test)
    p_judge = None
    if JUDGE_TEST.exists():
        p_judge = {int(k): v for k, v in json.loads(JUDGE_TEST.read_text(encoding="utf-8")).items()}
    else:
        print(f"WARNING: {JUDGE_TEST.name} missing — run 1 will be skipped")
    p_attr = attr_test_probs(posts, pairs, test) if ATTRS_FILE.exists() else None

    runs = {}
    runs["run2_local_blend"] = {
        i: (1 - W_ENC_RUN2) * p_lr[i] + W_ENC_RUN2 * p_enc[i] for i in p_lr
    }
    if p_judge:
        # order-consistency filter: when the judge's two orderings disagree it
        # is coin-flipping (CV: 0.53 acc on inconsistent pairs) — abstain and
        # let the two local tracks decide
        cons_file = DATA / "judge_test_v3_consistency.json"
        cons = ({int(k): v for k, v in json.loads(cons_file.read_text(encoding="utf-8")).items()}
                if cons_file.exists() else {})
        runs["run1_majority_vote"] = {
            i: (np.mean([p_lr[i] > 0.5, p_enc[i] > 0.5, p_judge[i] > 0.5])
                if cons.get(i, True)
                else (p_lr[i] + p_enc[i]) / 2)
            for i in p_lr
        }
    if p_attr:
        runs["run3_attr_majority"] = {
            i: np.mean([p_lr[i] > 0.5, p_enc[i] > 0.5, p_attr[i] > 0.5]) for i in p_lr
        }

    OUT.mkdir(exist_ok=True)
    for name, probs in runs.items():
        items = []
        for item in orig:
            out = dict(item)
            out["prediction"] = "Post 1" if probs[item["index"]] > 0.5 else "Post 2"
            items.append(out)
        d = OUT / name
        d.mkdir(exist_ok=True)
        f = d / "Social_Media_Pairwise_Test_predictions.json"
        f.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        n1 = sum(1 for it in items if it["prediction"] == "Post 1")
        print(f"{name}: Post 1 = {n1}/{len(items)} -> {f}")

    # cross-run agreement report
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = runs[names[i]], runs[names[j]]
            agree = np.mean([(a[k] > 0.5) == (b[k] > 0.5) for k in a])
            print(f"agreement {names[i]} vs {names[j]}: {agree:.2%}")


if __name__ == "__main__":
    main()
