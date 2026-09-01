# -*- coding: utf-8 -*-
"""Auxiliary-task pre-finetuning on FinArg-2 SM duration labels.

Organizers' recipe (Chiu et al. WWW-2025; Chen et al. CIKM-2024): train the
encoder on a related supervised task from the same domain (8.7k Taiwan
stock-forum posts, validity-duration classification), then fine-tune our
MPP ranking task from the adapted weights. Also fixes the length-distribution
gap: FinArg-2 posts are short (median 54 chars) like our test posts.

Output: models/macbert_dur/  (base encoder + tokenizer, loadable via
train_encoder.py --model models/macbert_dur)
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXT = Path(r"C:\Users\Hummer\Desktop\NTCIR19\external_data")
OUT = Path(__file__).parent / "models" / "macbert_dur"
MODEL = "hfl/chinese-macbert-base"
LABELS = {"Within 1 week": 0, "Longer than 1 week": 1, "Unsure": 2}
MAX_LEN, BATCH, EPOCHS, LR, SEED = 192, 32, 2, 2e-5, 42


class DurDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        return x["text"], LABELS[x["Label_duration"]]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda"
    sm = EXT / "FinArg-2" / "Social Media"
    train = json.loads((sm / "IDED_Train.json").read_text(encoding="utf-8"))
    dev = json.loads((sm / "IDED_Dev.json").read_text(encoding="utf-8"))
    # test_ans is additional labeled data — fold it into training
    train += json.loads((sm / "IDED_Test_ANS.json").read_text(encoding="utf-8"))
    print(f"pre-finetune corpus: {len(train)} train / {len(dev)} dev")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3).to(device)

    def collate(batch):
        texts, ys = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        return enc, torch.tensor(ys)

    loader = DataLoader(DurDataset(train), batch_size=BATCH, shuffle=True,
                        collate_fn=collate)
    dev_loader = DataLoader(DurDataset(dev), batch_size=64, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=len(loader) * EPOCHS, pct_start=0.1)
    # class weights against imbalance (roughly inverse-frequency, dampened)
    counts = np.array([1340 + 316, 4329 + 1287, 463 + 149], dtype=float)
    w = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    for epoch in range(EPOCHS):
        model.train()
        for enc, y in loader:
            enc, y = enc.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        # dev macro-F1
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for enc, y in dev_loader:
                enc = enc.to(device)
                preds += model(**enc).logits.argmax(-1).cpu().tolist()
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
