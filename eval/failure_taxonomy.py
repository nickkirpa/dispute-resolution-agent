"""Failure taxonomy: why did a case go wrong? One primary category per failed case, checked in a fixed order so the
counts add up. Works on any eval report (eval/results/*.json), offline.

    uv run python eval/failure_taxonomy.py eval/results/<report>.json [more reports...]
    uv run python eval/failure_taxonomy.py --all          # every LLM report on golden_v1 / policy-change sets

Categories (first match wins):
  crash                 the run raised an error
  unsafe_refund         money paid where policy says no  (the one that must stay at 0)
  wrong_type            dispute type misclassified, so everything after is on the wrong track
  wrong_refund_amount   right decision (refund) but the amount differs
  guard_escalation      a code guard sent it to a human (e.g. low confidence, evidence conflict) when policy had an answer
  model_over_escalation the model itself chose escalate when policy had an answer (no guard involved)
  missed_escalation     policy says escalate but the agent decided something else
  evidence_not_found    agent asked the customer for info although the evidence was in the ledger
  wrong_policy_outcome  everything else: right type, wrong decision under policy (e.g. reject vs refund)
Secondary (counted on all cases, not only failures):
  reply_check_failed    final reply still failed the self-check
  citation_miss         an expected clause was not cited
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["crash", "unsafe_refund", "wrong_type", "wrong_refund_amount", "guard_escalation", "model_over_escalation",
         "missed_escalation", "evidence_not_found", "wrong_policy_outcome"]


def classify(c: dict) -> str | None:
    """Primary failure category of one case row, or None if the case is correct."""
    if c.get("error"):
        return "crash"
    if c["decision_ok"] and c["refund_ok"]:
        return None
    if c.get("unsafe_refund"):
        return "unsafe_refund"
    if not c["type_ok"]:
        return "wrong_type"
    if c["decision_ok"]:
        return "wrong_refund_amount"
    if c["decision"] == "escalate":
        return "guard_escalation" if c.get("guard_reason") else "model_over_escalation"
    if c["expected_decision"] == "escalate":
        return "missed_escalation"
    if c["decision"] == "request_info":
        return "evidence_not_found"
    return "wrong_policy_outcome"


def summarize(report: dict) -> dict:
    cases = report["cases"]
    primary = Counter(k for k in (classify(c) for c in cases) if k)
    examples: dict[str, list[str]] = {}
    for c in cases:
        k = classify(c)
        if k:
            examples.setdefault(k, []).append(c["case_id"])
    return {
        "brain": report["summary"]["brain"], "golden": report["summary"].get("golden"), "n": len(cases),
        "failures": sum(primary.values()), "primary": {k: primary[k] for k in ORDER if primary[k]},
        "secondary": {"reply_check_failed": sum(bool(c.get("self_check_errors")) for c in cases),
                      "citation_miss": sum(c["citation_recall"] < 1 for c in cases)},
        "examples": {k: v[:5] for k, v in examples.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="all single-run LLM reports in eval/results")
    args = ap.parse_args()
    paths = args.reports or []
    if args.all:
        paths += [Path(p) for p in sorted(glob.glob(str(ROOT / "eval" / "results" / "2*.json")))
                  if "-x" not in Path(p).stem and "llm" in Path(p).stem]
    total = Counter()
    for p in paths:
        r = json.loads(p.read_text())
        if "cases" not in r or "type_ok" not in r["cases"][0]:
            continue
        s = summarize(r)
        total.update(s["primary"])
        tag = f"{s['brain']} on {s['golden']}"
        print(f"{tag:<75} {s['failures']:>3}/{s['n']} failed  {s['primary'] or ''}")
    if len(paths) > 1:
        print(f"\nAll reports combined ({sum(total.values())} failures):")
        for k in ORDER:
            if total[k]:
                print(f"  {k:<22} {total[k]:>4}")


if __name__ == "__main__":
    main()
