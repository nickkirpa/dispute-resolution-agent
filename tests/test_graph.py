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
