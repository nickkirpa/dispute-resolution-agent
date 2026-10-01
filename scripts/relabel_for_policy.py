"""Re-derive golden labels for a new policy version, deterministically (no LLM, no re-review of narratives).

Golden cases are built from known facts (ledger dates and amounts), so the expected outcome under different policy
parameters can be recomputed. Rules are applied in the same order the agent applies them:
  1. missing evidence -> request_info        (unchanged by parameter changes)
  2. filed after filing_window_days -> reject (POL-GEN-02)
  3. refund above human_review_amount -> escalate (+ POL-GEN-04)

    uv run python scripts/relabel_for_policy.py --kb-version v2
    -> eval/golden_v1_policy_v2.jsonl (+ a summary of which cases changed and why)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.tools import KnowledgeBase  # noqa: E402

NOISE = {"GroceryCo", "FuelStop", "CafeLumen", "PharmaPlus"}  # background merchants added by generate_golden.py


def days_since_case_txn(case: dict) -> int | None:
    txns = [t for t in case["ledger"]["transactions"] if t[2] not in NOISE and t[3] > 0]
    if not txns:
        return None
    return (date.fromisoformat(case["as_of"]) - date.fromisoformat(txns[0][4])).days


def relabel(case: dict, window: float, threshold: float) -> tuple[dict, str | None]:
    out = dict(case)
    d, decision, refund = days_since_case_txn(case), case["expected_decision"], case["expected_refund"]
    if decision == "request_info":
        return out, None
    if d is not None and d > window:
        if decision == "reject" and case["expected_clauses"] == ["POL-GEN-02"]:
            return out, None  # already a late-filing reject
        out.update(expected_decision="reject", expected_refund=0.0, expected_clauses=["POL-GEN-02"])
        return out, f"filed after {d} days > {window:.0f}"
    if decision == "refund" and refund > threshold:
        out.update(expected_decision="escalate", expected_refund=0.0,
                   expected_clauses=sorted(set(case["expected_clauses"]) | {"POL-GEN-04"}))
        return out, f"refund {refund:.2f} > {threshold:.0f} EUR"
    return out, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-version", default="v2")
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden_v1.jsonl")
    args = ap.parse_args()
    kb = KnowledgeBase(Settings().kb_dir, args.kb_version)
    window, threshold = kb.param("filing_window_days"), kb.param("human_review_amount")

    rows = [json.loads(line) for line in args.golden.read_text().splitlines() if line.strip()]
    out_rows, changes = [], Counter()
    for r in rows:
        new, why = relabel(r, window, threshold)
        new["policy_version"] = kb.version
        if why:
            new["relabel"] = {"reason": why, "v1_decision": r["expected_decision"], "v1_refund": r["expected_refund"]}
            changes[(r["scenario"], r["expected_decision"], new["expected_decision"])] += 1
        out_rows.append(new)
    out = args.golden.with_name(f"{args.golden.stem}_policy_{kb.version}.jsonl")
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in out_rows))
    print(f"{kb.version}: filing_window_days={window:.0f}, human_review_amount={threshold:.0f}")
    print(f"{sum(changes.values())}/{len(rows)} labels changed -> {out.resolve().relative_to(ROOT)}")
    for (sc, a, b), n in changes.most_common():
        print(f"  {sc:<38} {a:>8} -> {b:<8} x{n}")


if __name__ == "__main__":
    main()
