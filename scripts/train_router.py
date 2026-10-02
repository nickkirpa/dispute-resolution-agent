"""Fine-tune a transformer router: complaint text -> intent (77 Banking77 intents + synthetic not_received) -> DisputeType.

Trains on data/router/train.jsonl, selects the best epoch by validation macro-F1, reports on the untouched test split.
Runs on Apple Silicon (MPS), CUDA or CPU.

    uv sync --extra router
    uv run python scripts/train_router.py --limit 512 --epochs 1        # smoke test (~1 min)
    uv run python scripts/train_router.py                               # full run: ModernBERT-base, 4 epochs

Outputs: models/router/ (weights + tokenizer + label maps) and eval/results/router-<model>.json (metrics).
Why intents, not the 6 dispute types directly: 84% of rows are "other"; 78 fine-grained classes give the model a
richer signal, results are comparable with published Banking77 numbers, and intents map to dispute types exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "router"


def read(split: str, limit: int | None, rng: random.Random) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    if limit and len(rows) > limit:
        rows = rng.sample(rows, limit)
    return rows


def device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def report(y_true: list[str], y_pred: list[str]) -> dict:
    return {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="subsample every split (smoke tests)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "router")
    args = ap.parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    train, val, test = (read(s, args.limit, rng) for s in ("train", "val", "test"))
    intents = sorted({r["intent"] for r in train})
    label2id = {l: i for i, l in enumerate(intents)}
    intent_to_type = {r["intent"]: r["dispute_type"] for r in train + val + test}
    print(f"device={device_name()} model={args.model} classes={len(intents)} train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(rows: list[dict]) -> Dataset:
        rows = [r for r in rows if r["intent"] in label2id]  # (limit mode can drop rare intents from train)
        ds = Dataset.from_list([{"text": r["text"], "label": label2id[r["intent"]]} for r in rows])
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True, remove_columns=["text"])

    ds_train, ds_val = to_ds(train), to_ds(val)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(intents), id2label=dict(enumerate(intents)), label2id=label2id,
        attn_implementation="sdpa")  # flash-attention kernels are CUDA-only

    def compute_metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        gold_t = [intent_to_type[intents[i]] for i in p.label_ids]
        pred_t = [intent_to_type[intents[i]] for i in pred]
        return {"intent_acc": accuracy_score(p.label_ids, pred),
                "intent_macro_f1": f1_score(p.label_ids, pred, average="macro", zero_division=0),
                "type_macro_f1": f1_score(gold_t, pred_t, average="macro", zero_division=0)}

    total_steps = int(np.ceil(len(ds_train) / args.batch) * args.epochs)
    warmup = max(1, int(0.1 * total_steps))  # 10% linear warmup (transformers 5 dropped warmup_ratio)
    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        warmup_steps=warmup, weight_decay=0.01, per_device_train_batch_size=args.batch, per_device_eval_batch_size=128,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="intent_macro_f1", greater_is_better=True, logging_steps=50, report_to=[],
        seed=args.seed, dataloader_num_workers=0, torch_compile=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    t0 = time.perf_counter()
    trainer.train()
    train_min = (time.perf_counter() - t0) / 60

    # ---- test: predictions on the untouched split, reported overall / Banking77-only / synthetic-only
    model = trainer.model.eval()
    dev = next(model.parameters()).device
    preds, probs = [], []
    t0 = time.perf_counter()
    for i in range(0, len(test), 128):
        batch = tok([r["text"] for r in test[i:i + 128]], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**batch).logits.float(), dim=-1).cpu()
        probs += p.max(dim=-1).values.tolist()
        preds += [intents[j] for j in p.argmax(dim=-1).tolist()]
    batch_ms = (time.perf_counter() - t0) * 1000 / len(test)
    t0 = time.perf_counter()
    for r in test[:50]:  # single-message latency, what the agent sees per complaint
        with torch.no_grad():
            model(**tok(r["text"], truncation=True, max_length=args.max_len, return_tensors="pt").to(dev))
    single_ms = (time.perf_counter() - t0) * 1000 / 50

    def subset(src: str | None):
        idx = [i for i, r in enumerate(test) if src is None or r["source"] == src]
        return idx

    results = {}
    for name, src in (("all", None), ("banking77", "banking77"), ("synthetic", "synthetic")):
        idx = subset(src)
        if not idx:
            continue
        results[name] = {
            "intent": report([test[i]["intent"] for i in idx], [preds[i] for i in idx]),
            "dispute_type": report([test[i]["dispute_type"] for i in idx], [intent_to_type[preds[i]] for i in idx]),
        }
    conf = np.array(probs)
    correct = np.array([intent_to_type[p] == r["dispute_type"] for p, r in zip(preds, test)])
    coverage = {f">={t}": {"share_of_cases": float((conf >= t).mean()), "type_accuracy": float(correct[conf >= t].mean()) if (conf >= t).any() else None}
                for t in (0.5, 0.7, 0.8, 0.9, 0.95)}

    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "intent_to_type.json").write_text(json.dumps(intent_to_type, indent=1))
    summary = {"model": args.model, "device": device_name(), "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
               "max_len": args.max_len, "limit": args.limit, "train_minutes": round(train_min, 2),
               "latency_ms": {"single_message": round(single_ms, 2), "batched_per_message": round(batch_ms, 3)},
               "best_val": trainer.state.best_metric, "test": results, "confidence_coverage": coverage,
               "log_history": trainer.state.log_history}
    out = ROOT / "eval" / "results" / f"router-{args.model.split('/')[-1]}{'-smoke' if args.limit else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\ntrained in {train_min:.1f} min | best val intent macro-F1 {trainer.state.best_metric:.4f}")
    for name, r in results.items():
        print(f"test/{name:<9} intent acc {r['intent']['accuracy']:.4f} F1 {r['intent']['macro_f1']:.4f} | "
              f"dispute type acc {r['dispute_type']['accuracy']:.4f} F1 {r['dispute_type']['macro_f1']:.4f}  (n={r['intent']['n']})")
    print(f"latency: {single_ms:.1f} ms/message single, {batch_ms:.2f} ms/message batched")
    print("confidence threshold -> share of cases router decides alone, type accuracy on those:")
    for t, v in coverage.items():
        print(f"  {t}: {v['share_of_cases']:.3f} of cases, accuracy {v['type_accuracy']}")
    print(f"Report: {out.relative_to(ROOT)} | model: {args.out}")


if __name__ == "__main__":
    main()
