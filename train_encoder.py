# -*- coding: utf-8 -*-
"""Track 2: Chinese encoder scoring model with a pairwise ranking objective.

A single encoder maps one post to a scalar quality score s(x). Training loss
is BCEWithLogits(s(a) - s(b), label), which is antisymmetric by construction
(no Post1/Post2 position bias possible). Evaluation follows the grouped
5-fold protocol: train pairs = neither post in the eval fold; eval pairs =
both posts in the eval fold.

Usage:
  python train_encoder.py --model hfl/chinese-macbert-base --epochs 4
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

DATA = Path(__file__).parent / "data"
N_FOLDS = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, pairs, texts):
        self.pairs = pairs
        self.texts = texts

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        return (
            self.texts[p["a_pid"]],
            self.texts[p["b_pid"]],
            float(p["label"]),
            abs(p["dmpp"]),
            float(p.get("train_w", 1.0)),
        )


class Scorer(nn.Module):
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


def make_collate(tokenizer, max_len):
    def collate(batch):
        a, b, y, d, w = zip(*batch)
        enc_a = tokenizer(list(a), truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt")
        enc_b = tokenizer(list(b), truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt")
        return enc_a, enc_b, torch.tensor(y), torch.tensor(d), torch.tensor(w)
    return collate


@torch.no_grad()
def score_posts(model, tokenizer, texts_by_pid, max_len, device, bs=32):
    model.eval()
    pids = list(texts_by_pid.keys())
    scores = {}
    for i in range(0, len(pids), bs):
        chunk = pids[i : i + bs]
        enc = tokenizer([texts_by_pid[p] for p in chunk], truncation=True,
                        max_length=max_len, padding=True, return_tensors="pt").to(device)
        s = model(enc["input_ids"], enc["attention_mask"])
        for pid, v in zip(chunk, s.float().cpu().tolist()):
            scores[pid] = v
    return scores


def run_fold(fold, posts, pairs, args, device):
    texts = {p["pid"]: p["text"] for p in posts}
    train_pairs = [p for p in pairs if p["a_fold"] != fold and p["b_fold"] != fold]
    eval_pairs = [p for p in pairs if p["a_fold"] == fold and p["b_fold"] == fold]
    train_pairs = [dict(p) for p in train_pairs]
    for p in train_pairs:
        p["train_w"] = 1.0
        if args.weight_by_dmpp:
            p["train_w"] *= abs(p["dmpp"])
    if args.degree_norm:
        from collections import Counter
        deg = Counter()
        for p in train_pairs:
            deg[p["a_pid"]] += 1; deg[p["b_pid"]] += 1
        for p in train_pairs:
            p["train_w"] /= (deg[p["a_pid"]] * deg[p["b_pid"]]) ** 0.5

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Scorer(args.model, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(PairDataset(train_pairs, texts), batch_size=args.batch,
                        shuffle=True, collate_fn=make_collate(tokenizer, args.max_len))
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    scaler_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    best_acc, best_scores = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        for enc_a, enc_b, y, d, w in loader:
            enc_a, enc_b = enc_a.to(device), enc_b.to(device)
            y, d, w = y.to(device), d.to(device), w.to(device)
            with scaler_ctx:
                sa = model(enc_a["input_ids"], enc_a["attention_mask"])
                sb = model(enc_b["input_ids"], enc_b["attention_mask"])
                if args.soft_tau > 0:
                    # soft target: sigmoid(signed dMPP / tau) — near-ties get
                    # targets near 0.5 instead of confident hard labels
                    target = torch.sigmoid((d * (2 * y - 1)) / args.soft_tau)
                else:
                    target = y
                losses = loss_fn(sa - sb, target)
                losses = losses * (w / w.mean())
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        scores = score_posts(model, tokenizer, texts, args.max_len, device)
        correct = sum(
            1 for p in eval_pairs
            if (scores[p["a_pid"]] > scores[p["b_pid"]]) == (p["label"] == 1)
        )
        acc = correct / len(eval_pairs)
        print(f"  fold {fold} epoch {epoch+1}: eval acc {acc:.4f} ({len(eval_pairs)} pairs)")
        if acc >= best_acc:
            best_acc, best_scores = acc, scores
    return best_acc, len(eval_pairs), best_scores


def run_final(posts, pairs, args, device):
    """Train on ALL pairs (no folds), then score train + test posts."""
    texts = {p["pid"]: p["text"] for p in posts}
    pairs = [dict(p) for p in pairs]
    for p in pairs:
        p["train_w"] = abs(p["dmpp"]) if args.weight_by_dmpp else 1.0
    if args.degree_norm:
        from collections import Counter
        deg = Counter()
        for p in pairs:
            deg[p["a_pid"]] += 1; deg[p["b_pid"]] += 1
        for p in pairs:
            p["train_w"] /= (deg[p["a_pid"]] * deg[p["b_pid"]]) ** 0.5
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Scorer(args.model, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(PairDataset(pairs, texts), batch_size=args.batch,
                        shuffle=True, collate_fn=make_collate(tokenizer, args.max_len))
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(args.epochs):
        model.train()
        for enc_a, enc_b, y, d, w in loader:
            enc_a, enc_b = enc_a.to(device), enc_b.to(device)
            y, d, w = y.to(device), d.to(device), w.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sa = model(enc_a["input_ids"], enc_a["attention_mask"])
                sb = model(enc_b["input_ids"], enc_b["attention_mask"])
                if args.soft_tau > 0:
                    target = torch.sigmoid((d * (2 * y - 1)) / args.soft_tau)
                else:
                    target = y
                losses = loss_fn(sa - sb, target)
                losses = losses * (w / w.mean())
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f"  final epoch {epoch+1} done")

    test = json.loads((DATA / "test_pairs.json").read_text(encoding="utf-8"))
    all_texts = {f"train:{p['pid']}": p["text"] for p in posts}
    for t in test:
        all_texts[f"test:{t['index']}:1"] = t["post1"]
        all_texts[f"test:{t['index']}:2"] = t["post2"]
    scores = score_posts(model, tokenizer, all_texts, args.max_len, device)
    out = DATA / f"encoder_final_{args.tag or args.model.split('/')[-1]}_s{args.seed}.json"
    out.write_text(json.dumps({"model": args.model, "seed": args.seed,
                               "args": vars(args), "scores": scores}, indent=1),
                   encoding="utf-8")
    print("saved:", out.name)
    if args.save_model:
        ckpt = Path(__file__).parent / "models" / f"final_{args.tag or 'enc'}_s{args.seed}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save(model.head.state_dict(), ckpt / "head.pt")
        print("checkpoint saved:", ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--min_dmpp", type=float, default=0.02)
    ap.add_argument("--weight_by_dmpp", action="store_true")
    ap.add_argument("--soft_tau", type=float, default=0.0,
                    help=">0: soft pair targets sigmoid(dMPP/tau) instead of hard labels")
    ap.add_argument("--degree_norm", action="store_true",
                    help="down-weight pairs of high-degree posts by 1/sqrt(deg_a*deg_b)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--final", action="store_true",
                    help="train on all pairs and score test posts")
    ap.add_argument("--save_model", action="store_true",
                    help="with --final: save encoder+head checkpoint for real-time inference")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    posts = json.loads((DATA / "posts_clean.json").read_text(encoding="utf-8"))
    pairs = json.loads((DATA / "pairs_train.json").read_text(encoding="utf-8"))
    pairs = [p for p in pairs if abs(p["dmpp"]) >= args.min_dmpp]

    if args.final:
        run_final(posts, pairs, args, device)
        return

    results, oof_scores = [], {}
    for fold in range(N_FOLDS):
        acc, n, scores = run_fold(fold, posts, pairs, args, device)
        results.append((acc, n))
        # out-of-fold scores: keep only posts belonging to this fold
        fold_pids = {p["pid"] for p in posts if p["fold"] == fold}
        oof_scores.update({pid: s for pid, s in scores.items() if pid in fold_pids})

    total = sum(n for _, n in results)
    wavg = sum(a * n for a, n in results) / total
    print(f"\n{args.model} seed={args.seed} weighted CV acc = {wavg:.4f}")
    print("per-fold:", " ".join(f"{a:.3f}" for a, _ in results))

    out = DATA / f"encoder_cv_{args.tag or args.model.split('/')[-1]}_s{args.seed}.json"
    out.write_text(json.dumps({
        "model": args.model, "seed": args.seed, "cv_acc": wavg,
        "per_fold": [a for a, _ in results],
        "args": vars(args), "oof_scores": {str(k): v for k, v in oof_scores.items()},
    }, indent=1), encoding="utf-8")
    print("saved:", out.name)


if __name__ == "__main__":
    main()
