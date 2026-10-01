"""Interactive demo: run one case against the fixture ledger, pausing for a human when the graph escalates.

    uv run dispute-agent "I was charged twice by SpotiTunes for 9.99 EUR on 2026-09-02." --customer C001
    uv run dispute-agent "...899.00 EUR charge from LuxWatch I don't recognise..." --customer C006 --as-of 2026-09-30
"""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from .brain import RuleBrain
from .config import ROOT, Settings
from .graph import Deps, build_graph
from .state import CaseState
from .tools import KnowledgeBase


def main() -> None:
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))
    from eval.fixtures import build_fixture_ledger  # demo data; swap for data/ledger.duckdb later

    ap = argparse.ArgumentParser(description="Dispute Resolution Agent demo")
    ap.add_argument("narrative")
    ap.add_argument("--customer", required=True)
    ap.add_argument("--as-of", default="2026-09-30")
    ap.add_argument("--brain", choices=["rules", "llm"], default="rules")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    settings = Settings() if args.model is None else Settings(model=args.model)
    ledger = build_fixture_ledger()
    if args.brain == "rules":
        brain = RuleBrain(known_merchants=ledger.merchants())
    else:
        from .llm_brain import make_llm_brain

        brain = make_llm_brain(settings)
    app = build_graph(Deps(brain=brain, ledger=ledger, kb=KnowledgeBase(settings.kb_dir), settings=settings, human_in_loop=True))

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke(CaseState(case_id="cli", customer_id=args.customer, narrative=args.narrative, as_of=args.as_of), config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== HUMAN REVIEW REQUIRED ===")
        for k, v in payload.items():
            print(f"  {k}: {v}")
        decision = input("\nDecision [refund/request_info/reject/escalate]: ").strip() or "escalate"
        verdict = {"decision": decision, "note": input("Note: ").strip()}
        if decision == "refund":
            verdict["refund_amount"] = float(input(f"Refund amount [{payload['refund_amount']}]: ") or payload["refund_amount"])
        result = app.invoke(Command(resume=verdict), config=config)

    print("\n=== RESULT ===")
    print(f"type:      {result['dispute_type'].value} ({result['type_confidence']:.2f})")
    print(f"decision:  {result['decision'].value}   refund: {result['refund_amount']:.2f} EUR")
    print(f"cited:     {', '.join(result['cited_clauses'])}  (KB {result['kb_version']})")
    print(f"trace:     {' > '.join(result['trace'])}")
    if result["self_check_errors"]:
        print(f"self-check errors: {result['self_check_errors']}")
    print(f"usage:     {result['usage'].llm_calls} LLM calls, ${result['usage'].cost_usd:.4f}")
    print(f"\n{result['response_draft']}")


if __name__ == "__main__":
    main()
