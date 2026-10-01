"""Demo CLI with durable cases: state is checkpointed to SQLite after every step.

    uv run dispute-agent run "I don't recognise a 899.00 EUR charge from LuxWatch on 2026-09-15." --customer C006
        -> pauses for human review (refund > 500 EUR), prints the case id and exits
    uv run dispute-agent pending                          # cases waiting for a human
    uv run dispute-agent show <case_id>                   # current state of a case
    uv run dispute-agent resume <case_id> --decision refund --refund-amount 899 --note "verified by phone"

    add --interactive to `run` to answer the review prompt inline instead of exiting.
    Brain / evidence mode come from .env (DISPUTE_AGENT_PROVIDER, DISPUTE_AGENT_EVIDENCE) or --brain / --evidence.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .brain import RuleBrain
from .config import ROOT, Settings
from .graph import Deps, build_graph
from .state import CHECKPOINT_TYPES, CaseState
from .tools import KnowledgeBase, build_kb

DEFAULT_DB = ROOT / "data" / "cases.sqlite"


def run_config(args, settings: Settings) -> dict:
    """Everything needed to rebuild the same agent later. kb_version is resolved ("latest" -> "v2") on purpose."""
    return {"brain": args.brain, "provider": settings.provider, "model": settings.model,
            "evidence_mode": settings.evidence_mode, "effort_by_step": settings.effort_by_step,
            "kb_version": KnowledgeBase(settings.kb_dir, settings.kb_version).version, "retrieval_mode": settings.retrieval_mode}


def _app(config: dict, conn: sqlite3.Connection):
    sys.path.insert(0, str(ROOT))
    from eval.fixtures import build_fixture_ledger  # demo data; swap for data/ledger.duckdb later

    settings = Settings(provider=config["provider"], model=config["model"], evidence_mode=config["evidence_mode"],
                        effort_by_step=config.get("effort_by_step", {}), kb_version=config["kb_version"],
                        retrieval_mode=config.get("retrieval_mode", "bm25"))
    ledger = build_fixture_ledger()
    if config["brain"] == "rules":
        brain = RuleBrain(known_merchants=ledger.merchants())
    else:
        from .llm_brain import make_llm_brain

        brain = make_llm_brain(settings)
    kb = build_kb(settings)
    deps = Deps(brain=brain, ledger=ledger, kb=kb, settings=settings, human_in_loop=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    return build_graph(deps, checkpointer=SqliteSaver(conn, serde=serde))


def _print_result(result: dict) -> None:
    print("\n=== RESULT ===")
    print(f"type:      {result['dispute_type'].value} ({result['type_confidence']:.2f})")
    print(f"decision:  {result['decision'].value}   refund: {result['refund_amount']:.2f} EUR")
    print(f"cited:     {', '.join(result['cited_clauses'])}  (KB {result['kb_version']})")
    print(f"evidence:  {result['evidence_mode']}, {len(result['tool_calls'])} tool calls"
          + (f", code filled: {result['coverage_fills']}" if result["coverage_fills"] else ""))
    print(f"trace:     {' > '.join(result['trace'])}")
    if result["self_check_errors"]:
        print(f"self-check errors: {result['self_check_errors']}")
    print(f"usage:     {result['usage'].llm_calls} LLM calls, ${result['usage'].cost_usd:.4f}")
    print(f"\n{result['response_draft']}")


def _print_pending(payload: dict) -> None:
    print("\n=== HUMAN REVIEW REQUIRED ===")
    for k, v in payload.items():
        print(f"  {k}: {v}")


def _ask_verdict(payload: dict) -> dict:
    decision = input("\nDecision [refund/request_info/reject/escalate]: ").strip() or "escalate"
    verdict = {"decision": decision, "note": input("Note: ").strip()}
    if decision == "refund":
        verdict["refund_amount"] = float(input(f"Refund amount [{payload['refund_amount']}]: ") or payload["refund_amount"])
    return verdict


def _drive(app, config, result, interactive: bool) -> None:
    case_id = config["configurable"]["thread_id"]
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        _print_pending(payload)
        if not interactive:
            print(f"\nSaved. Resume later with:\n  uv run dispute-agent resume {case_id} --decision <refund|request_info|reject|escalate>"
                  " [--refund-amount X] [--note ...]")
            return
        result = app.invoke(Command(resume=_ask_verdict(payload)), config=config)
    _print_result(result)


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Dispute Resolution Agent demo")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite checkpoint store")
    ap.add_argument("--brain", choices=["rules", "llm"], default="rules")
    ap.add_argument("--evidence", choices=["plan", "agent"], default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a new complaint")
    r.add_argument("narrative")
    r.add_argument("--customer", required=True)
    r.add_argument("--as-of", default="2026-09-30")
    r.add_argument("--interactive", action="store_true", help="answer human review inline")
    rs = sub.add_parser("resume", help="give the human verdict for a paused case")
    rs.add_argument("case_id")
    rs.add_argument("--decision", required=True, choices=["refund", "request_info", "reject", "escalate"])
    rs.add_argument("--refund-amount", type=float)
    rs.add_argument("--note", default="")
    sh = sub.add_parser("show", help="print the latest state of a case")
    sh.add_argument("case_id")
    sub.add_parser("pending", help="list cases waiting for human review")
    args = ap.parse_args()

    settings = Settings(**({"evidence_mode": args.evidence} if args.evidence else {}))
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db, check_same_thread=False)
    cli_config = run_config(args, settings)
    app = _app(cli_config, conn)  # for run / show / pending; resume rebuilds from the case's stored config

    if args.cmd == "run":
        case_id = f"case-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": case_id}}
        print(f"case id: {case_id}  (checkpoints: {args.db})")
        print(f"config:  {cli_config}")
        result = app.invoke(CaseState(case_id=case_id, customer_id=args.customer, narrative=args.narrative, as_of=args.as_of,
                                      run_config=cli_config), config=config)
        _drive(app, config, result, args.interactive)
    elif args.cmd == "resume":
        config = {"configurable": {"thread_id": args.case_id}}
        snap = app.get_state(config)
        if not snap.next:
            sys.exit(f"{args.case_id} is not waiting for review (unknown id or already finished)")
        stored = snap.values.get("run_config") or cli_config  # cases from before run_config existed: use CLI flags
        if stored != cli_config:
            print(f"resuming with the case's original config (ignoring current flags/defaults): {stored}")
        app = _app(stored, conn)
        verdict = {"decision": args.decision, "note": args.note}
        if args.refund_amount is not None:
            verdict["refund_amount"] = args.refund_amount
        _drive(app, config, app.invoke(Command(resume=verdict), config=config), interactive=False)
    elif args.cmd == "show":
        snap = app.get_state({"configurable": {"thread_id": args.case_id}})
        if not snap.values:
            sys.exit(f"unknown case {args.case_id}")
        v = snap.values
        print(f"{args.case_id}: next={list(snap.next) or 'done'}  decision={getattr(v.get('decision'), 'value', None)}  "
              f"refund={v.get('refund_amount')}  trace={' > '.join(v.get('trace', []))}")
    elif args.cmd == "pending":
        ids = [row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
        waiting = [i for i in ids if app.get_state({"configurable": {"thread_id": i}}).next]
        print("\n".join(waiting) if waiting else "no cases waiting for review")


if __name__ == "__main__":
    main()
