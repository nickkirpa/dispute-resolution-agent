from langgraph.types import Command

from dispute_agent.brain import RuleBrain
from dispute_agent.config import Settings
from dispute_agent.graph import Deps, build_graph, run_case
from dispute_agent.state import CaseState, Decision
from dispute_agent.tools import KnowledgeBase
from eval.fixtures import build_fixture_ledger
from eval.run_eval import evaluate


def _app(human_in_loop=False):
    led = build_fixture_ledger()
    deps = Deps(brain=RuleBrain(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir), human_in_loop=human_in_loop)
    return build_graph(deps)


def test_rule_baseline_solves_seed_golden_set():
    """Regression gate: the deterministic baseline must stay at 100% on the seed set."""
    s = evaluate("rules")["summary"]
    assert s["decision_accuracy"] == 1.0
    assert s["refund_amount_accuracy"] == 1.0
    assert s["citation_recall"] == 1.0
    assert s["self_check_failure_rate"] == 0.0


def test_high_value_refund_pauses_for_human_and_resumes():
    app = _app(human_in_loop=True)
    case = CaseState(case_id="hv", customer_id="C006", as_of="2026-09-30",
                     narrative="I don't recognise a 899.00 EUR charge from LuxWatch on 2026-09-15.")
    config = {"configurable": {"thread_id": "hv"}}
    first = app.invoke(case, config=config)
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["refund_amount"] == 899.0

    final = app.invoke(Command(resume={"decision": "refund", "refund_amount": 899.0, "note": "verified with customer"}), config=config)
    assert final["decision"] == Decision.REFUND
    assert "899.00" in final["response_draft"]
    assert final["self_check_errors"] == []


def test_without_human_loop_high_value_is_escalated():
    out = run_case(_app(), CaseState(case_id="hv2", customer_id="C006", as_of="2026-09-30",
                                     narrative="I don't recognise a 899.00 EUR charge from LuxWatch on 2026-09-15."))
    assert out["decision"] == Decision.ESCALATE and out["refund_amount"] == 0.0


def test_step_budget_is_enforced():
    led = build_fixture_ledger()
    deps = Deps(brain=RuleBrain(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir), settings=Settings(max_steps=2))
    out = run_case(build_graph(deps), CaseState(case_id="b", customer_id="C001", as_of="2026-09-30",
                                                narrative="I was charged twice by SpotiTunes for 9.99 EUR on 2026-09-02."))
    assert out["decision"] == Decision.ESCALATE
    assert "human_review" in out["trace"]


def test_refund_contradicting_evidence_goes_to_human():
    """An LLM-style mistake: proposing a refund although the merchant already credited the customer."""
    from dispute_agent.brain import DecisionProposal

    class OverEagerBrain(RuleBrain):
        def decide(self, state):
            return DecisionProposal(decision=Decision.REFUND, confidence=0.95, rationale="refund", cited_clauses=["POL-REF-01"])

    led = build_fixture_ledger()
    deps = Deps(brain=OverEagerBrain(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir))
    out = run_case(build_graph(deps), CaseState(case_id="c", customer_id="C009", as_of="2026-09-30",
                   narrative="AirFly promised me a refund for my cancelled 410.00 EUR flight booked on 2026-08-01, but the refund never arrived."))
    assert out["decision"] == Decision.ESCALATE and out["refund_amount"] == 0.0
    assert "merchant credit already posted" in out["human_reason"]


def test_case_survives_process_restart_via_sqlite(tmp_path):
    """Pause for human review, drop the app (simulated restart), rebuild from the same DB file, resume."""
    import sqlite3

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    from dispute_agent.state import CHECKPOINT_TYPES

    db = tmp_path / "cases.sqlite"

    def app():
        conn = sqlite3.connect(db, check_same_thread=False)
        led = build_fixture_ledger()
        deps = Deps(brain=RuleBrain(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir), human_in_loop=True)
        return build_graph(deps, checkpointer=SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)))

    config = {"configurable": {"thread_id": "restart"}}
    first = app().invoke(CaseState(case_id="restart", customer_id="C006", as_of="2026-09-30",
                                   narrative="I don't recognise a 899.00 EUR charge from LuxWatch on 2026-09-15."), config=config)
    assert "__interrupt__" in first

    fresh = app()  # new graph, new connection: only the DB file carries the case
    assert fresh.get_state(config).next == ("human_review",)
    final = fresh.invoke(Command(resume={"decision": "refund", "refund_amount": 899.0, "note": "ok"}), config=config)
    assert final["decision"] == Decision.REFUND and final["refund_amount"] == 899.0
