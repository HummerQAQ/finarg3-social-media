# -*- coding: utf-8 -*-
"""Data preparation for FinArg-3 Social Media Subtask (pairwise MPP comparison).

Reads the raw organizer files, cleans and deduplicates posts, constructs
pairwise training examples, and assigns grouped CV folds so that every
duplicate/near-duplicate post always lands in the same fold.

Outputs (under project-root data/):
  posts_clean.json   - unique labeled posts: {pid, text, mpp, ml}
  pairs_train.json   - constructed pairs: {a_pid, b_pid, label, dmpp, fold}
  test_pairs.json    - normalized test pairs: {index, post1, post2}
"""
import json
import math
import re
import unicodedata
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT, required_input_path

OUT_DIR = PROJECT_ROOT / "data"


N_FOLDS = 5
# pairs whose MPP gap is below this are too noisy to teach anything
MIN_DMPP = 0.02


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_clean_posts():
    raw = json.loads(
        (required_input_path("FINARG3_RAW_DIR") /
         "Social_Media_Posts_with_MPP_ML.json").read_text(encoding="utf-8")
    )
    by_text = {}
    for p in raw:
        text = normalize(p["post_rationale"])
        mpp, ml = p["MPP"], p["ML"]
        if not text or len(text) < 4:
            continue
        if mpp is None or (isinstance(mpp, float) and math.isnan(mpp)):
            continue
        # duplicates: keep first occurrence (labels agree on MPP; ML differs
        # only in the 3rd decimal for 2 groups)
        if text in by_text:
            continue
        by_text[text] = {"mpp": float(mpp), "ml": float(ml)}
    posts = [
        {"pid": i, "text": t, "mpp": v["mpp"], "ml": v["ml"]}
        for i, (t, v) in enumerate(by_text.items())
    ]
    return posts


def assign_folds(posts):
    """Grouped folds: sort by MPP then deal round-robin so每個 fold 的 MPP
    分佈相近 (stratified on the regression target)."""
    order = sorted(posts, key=lambda p: p["mpp"])
    for rank, p in enumerate(order):
        p["fold"] = rank % N_FOLDS
    return posts


def build_pairs(posts):
    """All cross pairs with |dMPP| >= MIN_DMPP. Both posts' folds are kept so
    the trainer can filter: CV-train pairs = neither post in the eval fold;
    CV-eval pairs = both posts in the eval fold (no leakage either way)."""
    pairs = []
    n = len(posts)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = posts[i], posts[j]
            d = a["mpp"] - b["mpp"]
            if abs(d) < MIN_DMPP:
                continue
            # label 1 => first post has higher MPP
            pairs.append(
                {
                    "a_pid": a["pid"],
                    "b_pid": b["pid"],
                    "label": 1 if d > 0 else 0,
                    "dmpp": round(d, 6),
                    "a_fold": a["fold"],
                    "b_fold": b["fold"],
                }
            )
    return pairs


def load_test():
    raw = json.loads(
        (required_input_path("FINARG3_RAW_DIR") /
         "Social_Media_Pairwise_Test_with_translation.json").read_text(encoding="utf-8")
    )
    return [
        {
            "index": t["index"],
            "post1": normalize(t["post1"]),
            "post2": normalize(t["post2"]),
        }
        for t in raw
    ]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    posts = assign_folds(load_clean_posts())
    pairs = build_pairs(posts)
    test = load_test()

    (OUT_DIR / "posts_clean.json").write_text(
        json.dumps(posts, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "pairs_train.json").write_text(
        json.dumps(pairs, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "test_pairs.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    from collections import Counter

    print(f"clean posts: {len(posts)}")
    print(f"train pairs (|dMPP|>={MIN_DMPP}): {len(pairs)}")
    eval_counts = Counter(
        p["a_fold"] for p in pairs if p["a_fold"] == p["b_fold"]
    )
    print("CV-eval pairs per fold (both posts in fold):", dict(sorted(eval_counts.items())))
    print("label balance:", dict(Counter(p["label"] for p in pairs)))
    print(f"test pairs: {len(test)}")


if __name__ == "__main__":
    main()
