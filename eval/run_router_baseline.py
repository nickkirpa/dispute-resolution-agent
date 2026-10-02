"""Zero-shot LLM baseline for the router: the agent's own `classify` prompt on the router test split.

Same messages, same 6 dispute types as the fine-tuned router, so accuracy / macro-F1 / latency / cost compare directly.

    uv run python eval/run_router_baseline.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
load_dotenv(ROOT / ".env")

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.llm_brain import make_llm_brain  # noqa: E402
from dispute_agent.state import Claim  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data" / "router" / "test.jsonl").read_text().splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    settings = Settings()

    def classify(r: dict) -> dict:
        brain = make_llm_brain(settings)
        t0 = time.perf_counter()
        try:
            pred = brain.classify(r["text"], Claim()).dispute_type.value
        except Exception as e:  # count failures as wrong, keep going
            pred = f"error:{type(e).__name__}"
        return {"pred": pred, "latency_s": time.perf_counter() - t0, "cost": brain.usage.cost_usd}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        out = list(pool.map(classify, rows))
    wall = time.perf_counter() - t0

    results = {}
    for name, src in (("all", None), ("banking77", "banking77"), ("synthetic", "synthetic")):
        idx = [i for i, r in enumerate(rows) if src is None or r["source"] == src]
        y, p = [rows[i]["dispute_type"] for i in idx], [out[i]["pred"] for i in idx]
        results[name] = {"n": len(idx), "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", zero_division=0)}
    cost = sum(o["cost"] for o in out)
    lat = sorted(o["latency_s"] for o in out)
    summary = {"model": settings.model, "results": results, "cost_usd": round(cost, 4), "cost_per_1k_usd": round(cost / len(rows) * 1000, 4),
               "latency_s": {"mean": round(sum(lat) / len(lat), 3), "p50": round(lat[len(lat) // 2], 3), "p95": round(lat[int(len(lat) * .95)], 3)},
               "errors": sum(o["pred"].startswith("error:") for o in out), "wall_minutes": round(wall / 60, 2)}
    path = ROOT / "eval" / "results" / f"router-baseline-{settings.model.replace('/', '_')}.json"
    path.write_text(json.dumps(summary | {"predictions": [o["pred"] for o in out]}, indent=2))
    for name, r in results.items():
        print(f"LLM {settings.model} test/{name:<9} dispute type acc {r['accuracy']:.4f} F1 {r['macro_f1']:.4f}  (n={r['n']})")
    print(f"cost ${cost:.3f} (${summary['cost_per_1k_usd']}/1k msgs) | latency mean {summary['latency_s']['mean']}s p95 {summary['latency_s']['p95']}s | errors {summary['errors']}")
    print(f"Report: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
