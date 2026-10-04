# -*- coding: utf-8 -*-
"""Phase-2 pre-finetuning: FinArg-1 SM argument-relation pairs.

Starts from the duration-adapted encoder (models/macbert_dur) and continues
on 6.5k [post_a, post_b, relation] cross-encoded pairs — structurally the
same input geometry as our ranking task (read two posts, judge relation).

Output: models/macbert_dur_rel/
"""
import json
import random
from pathlib import Path

from finarg3_sm.paths import PROJECT_ROOT, required_input_path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INIT = PROJECT_ROOT / "models" / "macbert_dur"
OUT = PROJECT_ROOT / "models" / "macbert_dur_rel"
MAX_LEN, BATCH, EPOCHS, LR, SEED = 256, 24, 2, 2e-5, 42


class RelDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b, y = self.items[i]
        return a, b, int(y)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda"
    sm = required_input_path("FINARG_EXTERNAL_DATA_DIR") / "FinArg-1" / "Social Media"
    train = json.loads((sm / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((sm / "dev.json").read_text(encoding="utf-8"))
    print(f"relation corpus: {len(train)} train / {len(dev)} dev")

    tokenizer = AutoTokenizer.from_pretrained(INIT)
    model = AutoModelForSequenceClassification.from_pretrained(
        INIT, num_labels=3).to(device)

    def collate(batch):
        a, b, ys = zip(*batch)
        enc = tokenizer(list(a), list(b), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        return enc, torch.tensor(ys)

    loader = DataLoader(RelDataset(train), batch_size=BATCH, shuffle=True,
                        collate_fn=collate)
    dev_loader = DataLoader(RelDataset(dev), batch_size=64, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=len(loader) * EPOCHS, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for enc, y in loader:
            enc, y = enc.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(**enc).logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for enc, y in dev_loader:
                preds += model(**enc.to(device)).logits.argmax(-1).cpu().tolist()
                gold += y.tolist()
        from sklearn.metrics import f1_score, accuracy_score
        print(f"epoch {epoch+1}: dev acc {accuracy_score(gold, preds):.4f} "
              f"macro-F1 {f1_score(gold, preds, average='macro'):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print("saved encoder to", OUT)


if __name__ == "__main__":
    main()
