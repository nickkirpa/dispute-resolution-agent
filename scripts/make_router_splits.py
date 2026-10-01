"""Build router train / val / test splits (run before training the router).

- Banking77 train -> stratified train/val by the ORIGINAL 77 intents (finer than DisputeType, so every intent keeps
  its share in both splits; stratifying only on the 6 dispute types could leave rare intents out of val).
- Banking77 test stays the official, untouched test split (comparable with published Banking77 results).
- Synthetic not_received (human-rejected items dropped) -> 70/15/15 across train/val/test, tagged source=synthetic
  so metrics can be reported with and without it.

    uv run python scripts/make_router_splits.py --val-frac 0.1 --seed 13
    -> data/router/{train,val,test}.jsonl  + printed class distribution
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratified_split(rows: list[dict], key: str, frac: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keep, held = [], []
    for k in sorted(groups):  # sorted -> deterministic for a given seed
        g = groups[k][:]
        rng.shuffle(g)
        n_held = max(1, round(len(g) * frac))
        held += g[:n_held]
        keep += g[n_held:]
    return keep, held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for f in ("router_train.jsonl", "router_test.jsonl"):
        if not (DATA / f).exists():
            raise SystemExit(f"missing data/{f}: run scripts/download_banking77.py first")
    b_train = [r | {"source": "banking77"} for r in read(DATA / "router_train.jsonl")]
    b_test = [r | {"source": "banking77"} for r in read(DATA / "router_test.jsonl")]
    train, val = stratified_split(b_train, "intent", args.val_frac, rng)

    syn_path = DATA / "synthetic" / "not_received.jsonl"
    syn = [r for r in read(syn_path) if r.get("review_status") != "rejected"] if syn_path.exists() else []
    syn = [{"text": r["text"], "intent": "synthetic_not_received", "dispute_type": "not_received", "source": "synthetic"} for r in syn]
    rng.shuffle(syn)
    n = len(syn)
    syn_train, syn_val, syn_test = syn[: int(n * 0.7)], syn[int(n * 0.7): int(n * 0.85)], syn[int(n * 0.85):]

    # Banking77 contains a few exact duplicate texts across its own splits: drop them from train/val so the test set
    # never contains a message the model saw while training or tuning.
    norm = lambda r: r["text"].strip().lower()  # noqa: E731
    test_texts = {norm(r) for r in b_test + syn_test}
    before = len(train) + len(val)
    train = [r for r in train if norm(r) not in test_texts]
    val_seen = {norm(r) for r in train}
    val = [r for r in val if norm(r) not in test_texts and norm(r) not in val_seen]
    print(f"dropped {before - len(train) - len(val)} train/val rows duplicated in a later split")

    splits = {"train": train + syn_train, "val": val + syn_val, "test": b_test + syn_test}
    out = DATA / "router"
    out.mkdir(exist_ok=True)
    texts = {k: {r["text"].strip().lower() for r in v} for k, v in splits.items()}
    for k, v in splits.items():
        rng.shuffle(v)
        (out / f"{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
        dist = Counter(r["dispute_type"] for r in v)
        print(f"{k:>5}: {len(v):>5} rows  {dict(sorted(dist.items()))}")
    leaks = {f"{a}&{b}": len(texts[a] & texts[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    print(f"exact-duplicate texts across splits: {leaks}")
    intents_missing = set(r["intent"] for r in b_train) - set(r["intent"] for r in val)
    print(f"intents missing from val: {sorted(intents_missing) or 'none'}  (synthetic not_received: {len(syn)} after review filter)")


if __name__ == "__main__":
    main()
