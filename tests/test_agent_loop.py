"""Agent-mode evidence loop, driven by a scripted fake brain (offline)."""

from dispute_agent.brain import RuleBrain
from dispute_agent.config import Settings
from dispute_agent.evidence import AgentAction
from dispute_agent.graph import Deps, build_graph, run_case
from dispute_agent.state import CaseState, Decision
from dispute_agent.tools import KnowledgeBase
from eval.fixtures import build_fixture_ledger

CASE = dict(case_id="a", customer_id="C001", as_of="2026-09-30",
            narrative="I was charged twice by SpotiTunes for 9.99 EUR on 2026-09-02. Please refund the duplicate payment.")


class ScriptedAgent(RuleBrain):
    """RuleBrain for language steps; next_action replays a fixed list of actions."""

    def __init__(self, merchants, actions):
        super().__init__(merchants)
        self.actions, self.prompts = list(actions), []

    def next_action(self, prompt):
        self.prompts.append(prompt)
        return self.actions.pop(0) if self.actions else AgentAction(reason="done", tool="finish")


def _run(actions, **settings):
    led = build_fixture_ledger()
    brain = ScriptedAgent(led.merchants(), actions)
    deps = Deps(brain=brain, ledger=led, kb=KnowledgeBase(Settings().kb_dir), settings=Settings(evidence_mode="agent", **settings))
    return run_case(build_graph(deps), CaseState(**CASE)), brain


def test_agent_with_full_coverage_needs_no_fills():
    out, _ = _run([
        AgentAction(reason="find it", tool="find_transactions", merchant="SpotiTunes", amount=9.99),
        AgentAction(reason="dup?", tool="find_duplicates", merchant="SpotiTunes", amount=9.99),
        AgentAction(reason="history", tool="customer_history"),
        AgentAction(reason="policy", tool="search_policy", query="duplicate charge refund"),
        AgentAction(reason="done", tool="finish"),
    ])
    assert out["evidence_mode"] == "agent" and out["coverage_fills"] == []
    assert out["decision"] == Decision.REFUND and out["refund_amount"] == 9.99


def test_skipped_checks_are_filled_by_code_and_recorded():
    out, _ = _run([AgentAction(reason="find", tool="find_transactions", merchant="SpotiTunes"), AgentAction(reason="done", tool="finish")])
    assert set(out["coverage_fills"]) == {"find_duplicates", "customer_history", "search_policy"}
    assert out["decision"] == Decision.REFUND  # the fill ran the duplicate check the agent forgot


def test_duplicate_and_invalid_calls_are_refused_and_fed_back():
    same = AgentAction(reason="x", tool="find_transactions", merchant="SpotiTunes")
    out, brain = _run([same, same, AgentAction(reason="bad", tool="find_duplicates", merchant="SpotiTunes")])
    oks = [c["ok"] for c in out["tool_calls"] if c["reason"] != "coverage" and c["tool"] != "finish"]
    assert oks == [True, False, False]
    assert "Refused: identical call" in brain.prompts[2] and "needs merchant and amount" in brain.prompts[3]


def test_budget_is_enforced():
    actions = [AgentAction(reason=str(i), tool="find_transactions", amount=float(i)) for i in range(20)]
    out, brain = _run(actions, max_tool_calls=3)
    assert len(brain.prompts) == 3


def test_model_failure_falls_back_to_plan():
    class Broken(RuleBrain):
        def next_action(self, prompt):
            raise RuntimeError("provider down")

    led = build_fixture_ledger()
    deps = Deps(brain=Broken(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir), settings=Settings(evidence_mode="agent"))
    out = run_case(build_graph(deps), CaseState(**CASE))
    assert out["evidence_mode"] == "plan_fallback" and "provider down" in out["fallback_error"]
    assert out["decision"] == Decision.REFUND
