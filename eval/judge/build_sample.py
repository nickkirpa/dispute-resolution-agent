"""Build the reply sample for validating the LLM judge: real agent replies + replies with injected defects.

    uv run python eval/judge/build_sample.py --report eval/results/<an LLM run on golden_v1>.json
    -> eval/judge/replies.jsonl  (shuffled; the `defect` field is hidden from the labeller)

Defects are injected deterministically by code so their presence is known (a second ground truth next to the
human labels): wrong_amount, wrong_merchant, contradicts_decision, invented_promise, no_request, rude.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OTHER_MERCHANTS = ["TechMart", "PetPal", "GardenGo", "CityRide", "HomeNest"]


def case_facts(g: dict, row: dict) -> dict:
    txns = [t for t in g["ledger"]["transactions"] if t[2] not in {"GroceryCo", "FuelStop", "CafeLumen", "PharmaPlus"}]
    return {"scenario": g["scenario"], "complaint": g["narrative"], "decision": row["decision"],
            "refund_amount": g["expected_refund"] if row["decision"] == "refund" else 0.0,
            "case_transactions": [{"merchant": t[2], "amount": t[3], "date": t[4]} for t in txns],
            "prior_disputes": len(g["ledger"]["disputes"]), "cited_clauses": row.get("cited_clauses_final") or g["expected_clauses"]}


def inject(kind: str, reply: str, facts: dict, rng: random.Random) -> str | None:
    """Return a defective copy of `reply`, or None if this defect doesn't fit the reply."""
    merchant = facts["case_transactions"][0]["merchant"] if facts["case_transactions"] else None
    if kind == "wrong_amount" and facts["decision"] == "refund":
        amt = f"{facts['refund_amount']:.2f}"
        wrong = f"{facts['refund_amount'] * rng.choice([2, 0.5, 1.5]) + rng.choice([3, 7, 11]):.2f}"
        return reply.replace(amt, wrong) if amt in reply else None
    if kind == "wrong_merchant" and merchant and merchant in reply:
        return reply.replace(merchant, rng.choice([m for m in OTHER_MERCHANTS if m != merchant]))
    if kind == "contradicts_decision" and facts["decision"] in ("reject", "request_info"):
        return reply.rstrip() + " In the meantime, we have issued a provisional credit to your account."
    if kind == "invented_promise" and facts["decision"] in ("escalate", "request_info", "reject"):
        return reply.rstrip() + " We guarantee your money will be back within 24 hours."
    if kind == "no_request" and facts["decision"] == "request_info":
        return "Thank you for contacting Northwind Bank. We have received your dispute and will look into it. Kind regards, Northwind Disputes Team"
    if kind == "rude":
        return "As we have explained many times before, " + reply[0].lower() + reply[1:] + " Please read our policies before contacting us again."
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--real", type=int, default=25)
    ap.add_argument("--defective", type=int, default=15)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = {json.loads(l)["case_id"]: json.loads(l) for l in (ROOT / "eval" / "golden_v1.jsonl").read_text().splitlines() if l.strip()}
    rows = [c for c in json.loads(args.report.read_text())["cases"] if c.get("response_draft") and c["case_id"] in gold]
    by_decision: dict[str, list] = {}
    for c in rows:
        by_decision.setdefault(c["decision"], []).append(c)
    for v in by_decision.values():
        rng.shuffle(v)

    # real replies: spread across decisions
    real, i = [], 0
    while len(real) < args.real and any(by_decision.values()):
        d = sorted(by_decision)[i % len(by_decision)]
        if by_decision[d]:
            real.append(by_decision[d].pop())
        i += 1
    items = [{"facts": case_facts(gold[c["case_id"]], c), "reply": c["response_draft"], "case_id": c["case_id"], "defect": None}
             for c in real]

    # defective replies: cycle through defect kinds over the remaining real replies
    kinds = ["wrong_amount", "wrong_merchant", "contradicts_decision", "invented_promise", "no_request", "rude"]
    pool = [c for v in by_decision.values() for c in v]
    rng.shuffle(pool)
    used: set[str] = set()
    k = 0
    while sum(x["defect"] is not None for x in items) < args.defective and k < 10 * args.defective:
        kind = kinds[k % len(kinds)]  # balanced: each kind looks for a reply it fits
        k += 1
        for c in pool:
            if c["case_id"] in used:
                continue
            facts = case_facts(gold[c["case_id"]], c)
            bad = inject(kind, c["response_draft"], facts, rng)
            if bad:
                used.add(c["case_id"])
                items.append({"facts": facts, "reply": bad, "case_id": c["case_id"], "defect": kind})
                break

    rng.shuffle(items)
    out = ROOT / "eval" / "judge" / "replies.jsonl"
    out.write_text("".join(json.dumps({"id": f"R{n:02d}"} | it) + "\n" for n, it in enumerate(items, 1)))
    from collections import Counter

    print(f"{len(items)} replies -> {out.relative_to(ROOT)} | decisions {dict(Counter(x['facts']['decision'] for x in items))} | "
          f"defects {dict(Counter(x['defect'] for x in items if x['defect']))}")


if __name__ == "__main__":
    main()
