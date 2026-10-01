"""Generate a golden set whose labels are correct by construction.

Each scenario is built in code: ledger rows (customer, transactions, prior disputes) plus the decision, refund
and policy clauses that policy v1 implies for that situation. The LLM is used ONLY to write a natural-sounding
complaint from the structured facts; it never chooses a label. Narratives are checked automatically for
required facts (merchant, amounts) and regenerated if they miss them. A human still reviews faithfulness
(review_status starts as "unreviewed").

    uv run python scripts/generate_golden.py --per-scenario 12 --workers 8
    -> eval/golden_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.llm_brain import make_llm_brain  # noqa: E402

AS_OF = date(2026, 9, 30)

GOODS = [("GadgetHub", "wireless headphones", 40, 300), ("ShoeBox", "running shoes", 45, 180), ("BookNest", "a set of books", 15, 90),
         ("HomeNest", "a floor lamp", 30, 250), ("PetPal", "a pet bed and food", 20, 120), ("GardenGo", "garden tools", 25, 200)]
PRICEY = [("LuxWatch", "a watch", 600, 1500), ("TechMart", "a laptop", 650, 1800)]
SUBS = [("SpotiTunes", "music subscription", 9.99), ("Netflux", "streaming subscription", 15.99),
        ("FitClub", "gym membership", 49.00), ("CloudBox", "cloud storage plan", 11.99)]
TRAVEL = [("AirFly", "a flight", 90, 480), ("HotelLisboa", "a hotel stay", 120, 480), ("CityRide", "a taxi ride", 12, 60)]
NOISE = [("GroceryCo", 8, 120), ("FuelStop", 30, 90), ("CafeLumen", 3, 15), ("PharmaPlus", 5, 60)]

STYLES = ["short and to the point", "polite and detailed", "frustrated, with some irrelevant detail about their week",
          "written by a non-native English speaker with small grammar mistakes", "casual chat-message style, lowercase",
          "formal, like a letter"]


@dataclass
class Case:
    scenario: str
    expected_type: str
    expected_decision: str
    expected_clauses: list[str]
    expected_refund: float
    facts: dict
    txns: list = field(default_factory=list)  # (merchant, amount, date, channel)
    disputes: list = field(default_factory=list)  # (type, opened_on, outcome)
    must_mention: list[float] = field(default_factory=list)


def money(rng, lo, hi) -> float:
    return round(rng.uniform(lo, hi), 2)


def recent(rng, lo=5, hi=100) -> date:
    return AS_OF - timedelta(days=rng.randint(lo, hi))


# ------------------------------------------------------------------ scenarios (policy v1 semantics)

def dup_refund(rng):
    m, item, p = rng.choice(SUBS + [(g[0], g[1], money(rng, g[2], g[3])) for g in GOODS])
    d = recent(rng)
    return Case("duplicate/refund", "duplicate_charge", "refund", ["POL-DUP-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "they were charged twice for the same purchase"},
                [(m, p, d, "online"), (m, p, d + timedelta(days=rng.randint(0, 2)), "online")], must_mention=[p])


def dup_subscription_trap(rng):
    m, item, p = rng.choice(SUBS)
    d = recent(rng, 35, 70)
    return Case("duplicate/reject_recurring", "duplicate_charge", "reject", ["POL-DUP-02"], 0.0,
                {"merchant": m, "item": item, "amount": p, "claim": "they believe they were charged twice, once last month and again this month, but they only signed up once"},
                [(m, p, d, "recurring"), (m, p, d + timedelta(days=30), "recurring")], must_mention=[p])


def dup_high_value(rng):
    m, item, lo, hi = rng.choice(PRICEY)
    p, d = money(rng, lo, hi), recent(rng)
    return Case("duplicate/escalate_high_value", "duplicate_charge", "escalate", ["POL-DUP-01", "POL-GEN-04"], 0.0,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "they were charged twice for one purchase"},
                [(m, p, d, "online"), (m, p, d + timedelta(days=1), "online")], must_mention=[p])


def dup_late(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, hi), recent(rng, 135, 220)
    return Case("duplicate/reject_late", "duplicate_charge", "reject", ["POL-GEN-02"], 0.0,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "they only now noticed they were charged twice for the same order"},
                [(m, p, d, "online"), (m, p, d + timedelta(days=1), "online")], must_mention=[p])


def dup_no_txn(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p = money(rng, lo, hi)
    return Case("duplicate/request_info_no_txn", "duplicate_charge", "request_info", ["POL-GEN-03"], 0.0,
                {"merchant": m, "item": item, "amount": p, "claim": "they were charged twice"}, [], must_mention=[p])


def nr_refund(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, hi), recent(rng, 20, 90)
    return Case("not_received/refund", "not_received", "refund", ["POL-NR-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "the order never arrived",
                 "contacted_merchant": "YES - they already contacted the merchant and got no resolution"},
                [(m, p, d, "online")], must_mention=[p])


def nr_no_contact(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, hi), recent(rng, 20, 90)
    return Case("not_received/request_info_no_contact", "not_received", "request_info", ["POL-NR-02"], 0.0,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "the order never arrived",
                 "contacted_merchant": "NO - do not say or imply they contacted the merchant"},
                [(m, p, d, "online")], must_mention=[p])


def nr_late(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, hi), recent(rng, 135, 220)
    return Case("not_received/reject_late", "not_received", "reject", ["POL-GEN-02"], 0.0,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "the order never arrived, months ago",
                 "contacted_merchant": "YES - they contacted the merchant several times"},
                [(m, p, d, "online")], must_mention=[p])


def nr_high_value(rng):
    m, item, lo, hi = rng.choice(PRICEY)
    p, d = money(rng, lo, hi), recent(rng, 20, 90)
    return Case("not_received/escalate_high_value", "not_received", "escalate", ["POL-NR-01", "POL-GEN-04"], 0.0,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "the order never arrived",
                 "contacted_merchant": "YES - they contacted the merchant and got no answer"},
                [(m, p, d, "online")], must_mention=[p])


def una_refund(rng):
    m, item, lo, hi = rng.choice(GOODS + TRAVEL)
    p, d = money(rng, lo, min(hi, 480)), recent(rng, 2, 40)
    return Case("unauthorized/refund", "unauthorized", "refund", ["POL-UNA-01"], p,
                {"merchant": m, "amount": p, "date": d, "claim": "they do not recognise this payment and did not make it"},
                [(m, p, d, "online")], must_mention=[p])


def una_high_value(rng):
    m, item, lo, hi = rng.choice(PRICEY)
    p, d = money(rng, lo, hi), recent(rng, 2, 40)
    return Case("unauthorized/escalate_high_value", "unauthorized", "escalate", ["POL-UNA-01", "POL-GEN-04"], 0.0,
                {"merchant": m, "amount": p, "date": d, "claim": "they do not recognise this payment; their card may be compromised"},
                [(m, p, d, "online")], must_mention=[p])


def una_repeat(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, hi), recent(rng, 2, 40)
    prior = [("unauthorized", AS_OF - timedelta(days=rng.randint(60, 300)), "refunded") for _ in range(rng.choice([2, 3]))]
    return Case("unauthorized/escalate_repeat_claims", "unauthorized", "escalate", ["POL-UNA-02"], 0.0,
                {"merchant": m, "amount": p, "date": d, "claim": "they did not authorise this payment"},
                [(m, p, d, "online")], disputes=prior, must_mention=[p])


def una_no_txn(rng):
    m, item, lo, hi = rng.choice(GOODS + TRAVEL)
    p = money(rng, lo, min(hi, 480))
    return Case("unauthorized/request_info_no_txn", "unauthorized", "request_info", ["POL-GEN-03"], 0.0,
                {"merchant": m, "amount": p, "claim": "they did not make this payment"}, [], must_mention=[p])


def amt_refund(rng):
    m, item, lo, hi = rng.choice(TRAVEL + GOODS)
    agreed = money(rng, lo, hi)
    charged = round(agreed + money(rng, 5, min(120, agreed)), 2)
    d = recent(rng, 3, 60)
    return Case("wrong_amount/refund", "wrong_amount", "refund", ["POL-AMT-01"], round(charged - agreed, 2),
                {"merchant": m, "item": item, "charged_amount": charged, "agreed_amount": agreed, "date": d,
                 "claim": "they were charged more than the agreed price; state BOTH amounts explicitly, e.g. 'charged X instead of Y'"},
                [(m, charged, d, "card_present")], must_mention=[charged, agreed])


def amt_no_expected(rng):
    m, item, lo, hi = rng.choice(TRAVEL + GOODS)
    charged, d = money(rng, lo, hi), recent(rng, 3, 60)
    return Case("wrong_amount/request_info_no_expected", "wrong_amount", "request_info", ["POL-AMT-01"], 0.0,
                {"merchant": m, "item": item, "charged_amount": charged, "date": d,
                 "claim": "they were overcharged / the amount is wrong, but they do NOT state what the correct price was"},
                [(m, charged, d, "card_present")], must_mention=[charged])


def ref_refund(rng):
    m, item, lo, hi = rng.choice(TRAVEL + GOODS)
    p, d = money(rng, lo, hi), recent(rng, 20, 90)
    return Case("refund_not_processed/refund", "refund_not_processed", "refund", ["POL-REF-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d,
                 "claim": "the merchant promised/confirmed a refund for a cancelled order but it has not arrived"},
                [(m, p, d, "online")], must_mention=[p])


def ref_already_credited(rng):
    m, item, lo, hi = rng.choice(TRAVEL + GOODS)
    p, d = money(rng, lo, hi), recent(rng, 30, 90)
    return Case("refund_not_processed/reject_credited", "refund_not_processed", "reject", ["POL-REF-02"], 0.0,
                {"merchant": m, "item": item, "amount": p, "date": d,
                 "claim": "the merchant promised a refund for a cancelled order and the customer believes it never arrived"},
                [(m, p, d, "online"), (m, -p, d + timedelta(days=rng.randint(5, 20)), "online")], must_mention=[p])


# ---- policy-shift suite: cases inside the bands where policy v1 and v2 disagree, plus unchanged controls.
# Labels below are v1 labels; scripts/relabel_for_policy.py derives the v2 labels.

def shift_nr_window(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, min(hi, 280)), recent(rng, 95, 118)
    return Case("shift/not_received_95_118_days", "not_received", "refund", ["POL-NR-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "the order never arrived",
                 "contacted_merchant": "YES - they contacted the merchant several times without resolution"},
                [(m, p, d, "online")], must_mention=[p])


def shift_dup_window(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, min(hi, 280)), recent(rng, 95, 118)
    return Case("shift/duplicate_95_118_days", "duplicate_charge", "refund", ["POL-DUP-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "they were charged twice for the same order"},
                [(m, p, d, "online"), (m, p, d + timedelta(days=1), "online")], must_mention=[p])


def shift_una_mid_value(rng):
    m, item, lo, hi = rng.choice(TRAVEL[:2])
    p, d = money(rng, 310, 490), recent(rng, 2, 40)
    return Case("shift/unauthorized_310_490_eur", "unauthorized", "refund", ["POL-UNA-01"], p,
                {"merchant": m, "amount": p, "date": d, "claim": "they do not recognise this payment and did not make it"},
                [(m, p, d, "online")], must_mention=[p])


def shift_ref_mid_value(rng):
    m, item, lo, hi = rng.choice(TRAVEL[:2])
    p, d = money(rng, 310, 480), recent(rng, 20, 80)
    return Case("shift/refund_not_processed_310_480_eur", "refund_not_processed", "refund", ["POL-REF-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d,
                 "claim": "the merchant confirmed a refund for a cancelled booking but it has not arrived"},
                [(m, p, d, "online")], must_mention=[p])


def shift_control(rng):
    m, item, lo, hi = rng.choice(GOODS)
    p, d = money(rng, lo, min(hi, 280)), recent(rng, 20, 80)
    return Case("shift/control_unchanged", "not_received", "refund", ["POL-NR-01"], p,
                {"merchant": m, "item": item, "amount": p, "date": d, "claim": "the order never arrived",
                 "contacted_merchant": "YES - they contacted the merchant and got no answer"},
                [(m, p, d, "online")], must_mention=[p])


SUITES = {}  # filled after SCENARIOS is defined

SCENARIOS = [dup_refund, dup_subscription_trap, dup_high_value, dup_late, dup_no_txn, nr_refund, nr_no_contact, nr_late,
             nr_high_value, una_refund, una_high_value, una_repeat, una_no_txn, amt_refund, amt_no_expected, ref_refund,
             ref_already_credited]
SUITES = {"v1": SCENARIOS, "policy_shift": [shift_nr_window, shift_dup_window, shift_una_mid_value, shift_ref_mid_value, shift_control]}


# ------------------------------------------------------------------ narrative writing

class Narrative(BaseModel):
    text: str


PROMPT = """Write one message a bank customer sends to their bank's card-dispute team.

Facts (all must be conveyed; do not add other amounts, merchants or claims):
{facts}

Style: {style}.
Rules:
- Use the merchant name exactly: "{merchant}".
- Write every amount with two decimals and EUR or €, e.g. 89.90 EUR or €89.90.
- {date_rule}
- 2-6 sentences. First person. No subject line, no signature block, no placeholders like [Name]."""


def amount_present(text: str, a: float) -> bool:
    a = abs(a)
    forms = {f"{a:.2f}", f"{a:.2f}".replace(".", ","), f"{a:,.2f}"}
    return any(f in text for f in forms)


def write_case(i: int, case: Case, seed: int, settings: Settings) -> dict | None:
    rng = random.Random(seed * 1000 + i)
    facts = dict(case.facts)
    if "date" in facts:
        mode = rng.choice(["iso", "natural", "natural", "omit"])
        d = facts.pop("date")
        date_rule = {"iso": f"Mention the transaction date as {d.isoformat()}.",
                     "natural": f"Mention the transaction date naturally, e.g. '{d.day} {d:%B}'.",
                     "omit": "Do not mention the exact date."}[mode]
    else:
        date_rule = "Do not mention an exact date."
    brain = make_llm_brain(settings)
    prompt = PROMPT.format(facts=json.dumps(facts, default=str, indent=1), style=rng.choice(STYLES),
                           merchant=case.facts["merchant"], date_rule=date_rule)
    text = None
    for _ in range(3):  # regenerate if required facts are missing
        cand = brain.complete(prompt, Narrative).text.strip()
        if case.facts["merchant"].lower() in cand.lower() and all(amount_present(cand, a) for a in case.must_mention):
            text = cand
            break
    if text is None:
        return None

    cid = f"C{i:04d}"
    noise = [(n, money(rng, lo, hi), recent(rng, 1, 150), "card_present") for n, lo, hi in rng.sample(NOISE, rng.randint(2, 4))]
    txns = [(f"T{i:04d}{k:02d}", cid, m, a, d.isoformat(), ch) for k, (m, a, d, ch) in enumerate(case.txns + noise)]
    disputes = [(f"D{i:04d}{k:02d}", cid, f"OLD{k}", t, o.isoformat(), out) for k, (t, o, out) in enumerate(case.disputes)]
    return {
        "case_id": f"{'V1' if case.scenario.split('/')[0] != 'shift' else 'PS'}-{i:04d}", "scenario": case.scenario, "customer_id": cid, "as_of": AS_OF.isoformat(),
        "narrative": text, "expected_type": case.expected_type, "expected_decision": case.expected_decision,
        "expected_refund": case.expected_refund, "expected_clauses": case.expected_clauses,
        "ledger": {"customers": [(cid, f"Customer {i}", "2021-06-01")], "transactions": txns, "disputes": disputes},
        "generator": {"seed": seed, "narrative_model": settings.model, "date_rule": date_rule},
        "review_status": "unreviewed", "cost_usd": round(brain.usage.cost_usd, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--suite", choices=list(SUITES), default="v1")
    ap.add_argument("--out", type=Path, default=None, help="default: eval/golden_v1.jsonl or eval/golden_<suite>.jsonl")
    args = ap.parse_args()
    args.out = args.out or ROOT / "eval" / ("golden_v1.jsonl" if args.suite == "v1" else f"golden_{args.suite}.jsonl")

    rng = random.Random(args.seed)
    plan = [fn(rng) for fn in SUITES[args.suite] for _ in range(args.per_scenario)]
    settings = Settings()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda ic: write_case(ic[0], ic[1], args.seed, settings), enumerate(plan, 1)))
    kept = [r for r in rows if r]
    with args.out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"{len(kept)}/{len(plan)} cases -> {args.out.resolve().relative_to(ROOT)}  (dropped {len(plan) - len(kept)} with missing facts)")
    print(f"narrative cost: ${sum(r['cost_usd'] for r in kept):.3f} with {settings.model}")


if __name__ == "__main__":
    main()
