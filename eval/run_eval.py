"""Run the agent over the golden set and report metrics.

    uv run python eval/run_eval.py --brain rules
    uv run python eval/run_eval.py --brain llm --model claude-opus-5-5   # needs ANTHROPIC_API_KEY

Writes a JSON report to eval/results/ and prints a summary + per-case failures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from dispute_agent.brain import RuleBrain  # noqa: E402
from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.graph import Deps, build_graph, run_case  # noqa: E402
from dispute_agent.state import CaseState  # noqa: E402
from dispute_agent.tools import KnowledgeBase  # noqa: E402
from dispute_agent.tools import Ledger  # noqa: E402
from eval.fixtures import build_fixture_ledger  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    """Cases a human rejected in scripts/review.py are excluded."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("review_status") != "rejected"]


def make_brain(kind: str, settings: Settings, ledger):
    if kind == "rules":
        return RuleBrain(known_merchants=ledger.merchants())
    from dispute_agent.llm_brain import make_llm_brain

    return make_llm_brain(settings)


def build_ledger(golden: list[dict]) -> Ledger:
    """Generated golden sets carry their own ledger rows per case; the hand-written seed set uses eval/fixtures.py."""
    if not any("ledger" in g for g in golden):
        return build_fixture_ledger()
    led = Ledger(":memory:")
    for g in golden:
        led.load(**g["ledger"])
    return led


def run_one(g: dict, brain_kind: str, settings: Settings, ledger: Ledger, kb: KnowledgeBase) -> dict:
    ledger = ledger.fork()
    brain = make_brain(brain_kind, settings, ledger)  # fresh brain per case -> per-case usage
    app = build_graph(Deps(brain=brain, ledger=ledger, kb=kb, settings=settings, human_in_loop=False))
    case = CaseState(case_id=g["case_id"], customer_id=g["customer_id"], narrative=g["narrative"], as_of=g["as_of"])
    t0 = time.perf_counter()
    error = None
    try:
        out = run_case(app, case)
    except Exception as e:  # a crash is a failed case, not a crashed eval
        out, error = {}, f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    decision = out.get("decision")
    cited = set(out.get("cited_clauses", []))
    expected_clauses = set(g["expected_clauses"])
    usage = out.get("usage") or brain.usage
    return {
        "case_id": g["case_id"],
        "scenario": g.get("scenario"),
        "expected_decision": g["expected_decision"],
        "decision": decision.value if decision else None,
        "decision_ok": bool(decision) and decision.value == g["expected_decision"],
        "type_ok": bool(out.get("dispute_type")) and out["dispute_type"].value == g["expected_type"],
        "refund_ok": abs(out.get("refund_amount", 0.0) - g["expected_refund"]) < 0.01,
        "citation_recall": len(cited & expected_clauses) / len(expected_clauses),
        "unsafe_refund": bool(decision) and decision.value == "refund" and g["expected_decision"] != "refund",
        "guard_reason": out.get("human_reason", "") if out.get("human_reason", "").startswith("guard:") else "",
        "self_check_errors": out.get("self_check_errors", []),
        "steps": out.get("steps", 0),
        "llm_calls": usage.llm_calls,
        "cost_usd": usage.cost_usd,
        "latency_s": round(latency, 3),
        "trace": out.get("trace", []),
        "error": error,
    }


def evaluate(brain_kind: str = "rules", model: str | None = None, golden_path: Path = ROOT / "eval" / "golden.jsonl",
             workers: int = 1) -> dict:
    settings = Settings() if model is None else Settings(model=model)
    golden = load_golden(golden_path)
    ledger = build_ledger(golden)
    kb = KnowledgeBase(settings.kb_dir, settings.kb_version)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda g: run_one(g, brain_kind, settings, ledger, kb), golden))

    n = len(rows)
    summary = {
        "brain": brain_kind if brain_kind == "rules" else f"llm:{settings.model}",
        "golden": golden_path.name,
        "kb_version": kb.version,
        "n_cases": n,
        "decision_accuracy": sum(r["decision_ok"] for r in rows) / n,
        "type_accuracy": sum(r["type_ok"] for r in rows) / n,
        "refund_amount_accuracy": sum(r["refund_ok"] for r in rows) / n,
        "citation_recall": sum(r["citation_recall"] for r in rows) / n,
        "unsafe_refund_rate": sum(r["unsafe_refund"] for r in rows) / n,  # paid out when policy says no: must be 0
        "guard_intervention_rate": sum(bool(r["guard_reason"]) for r in rows) / n,
        "escalation_rate": sum(r["decision"] == "escalate" for r in rows) / n,
        "self_check_failure_rate": sum(bool(r["self_check_errors"]) for r in rows) / n,
        "crash_rate": sum(bool(r["error"]) for r in rows) / n,
        "avg_steps": sum(r["steps"] for r in rows) / n,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "cost_per_case_usd": round(sum(r["cost_usd"] for r in rows) / n, 5),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 3),
    }
    return {"summary": summary, "cases": rows}


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["rules", "llm"], default="rules")
    ap.add_argument("--model", default=None, help="Claude model id for --brain llm (default from settings)")
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    ap.add_argument("--workers", type=int, default=1, help="parallel cases (LLM runs: 4-8 is reasonable)")
    args = ap.parse_args()

    report = evaluate(args.brain, args.model, args.golden, args.workers)
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9.-]+', '_', report['summary']['brain'])}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    for k, v in report["summary"].items():
        print(f"{k:>26}: {v:.3f}" if isinstance(v, float) else f"{k:>26}: {v}")
    scen = {}
    for r in report["cases"]:
        if r["scenario"]:
            ok, n = scen.get(r["scenario"], (0, 0))
            scen[r["scenario"]] = (ok + r["decision_ok"], n + 1)
    if scen:
        print("\nDecision accuracy by scenario:")
        for k, (ok, n) in sorted(scen.items()):
            print(f"  {k:<34} {ok:>3}/{n:<3}")
    failures = [r for r in report["cases"] if not (r["decision_ok"] and r["refund_ok"]) or r["error"] or r["self_check_errors"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  {r['case_id']}: expected={r['expected_decision']} got={r['decision']} refund_ok={r['refund_ok']} "
                  f"errors={r['self_check_errors'] or r['error']} trace={'>'.join(r['trace'])}")
    print(f"\nReport: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
