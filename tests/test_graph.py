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


def test_reply_must_match_decision():
    from dispute_agent.graph import reply_consistency_errors as check

    assert check(Decision.REFUND, "We have issued a provisional credit of 9.99 EUR.", 9.99) == []
    assert check(Decision.REFUND, "We have issued a provisional credit.", 9.99)  # amount missing
    assert check(Decision.REJECT, "Sorry, we are unable to accept this dispute.", 0) == []
    assert check(Decision.REJECT, "We have refunded the payment.", 0)  # contradicts the decision
    assert check(Decision.REQUEST_INFO, "Could you send the transaction date?", 0) == []
    assert check(Decision.REQUEST_INFO, "Thank you for contacting us.", 0)  # asks for nothing
    assert check(Decision.ESCALATE, "A dispute specialist will contact you.", 0) == []
    assert check(Decision.ESCALATE, "You will receive a refund shortly, a specialist will confirm.", 0)


def test_reply_check_accepts_real_llm_replies_that_were_false_positives():
    """Replies from the 2026-10-02 effort runs that the first version of the check wrongly flagged."""
    from dispute_agent.graph import reply_consistency_errors as check

    ok_request_info = [
        "Before we can review this not-received dispute, please contact GadgetHub first and keep a record of your attempt. "
        "If the issue is still unresolved after that, send us the details of your contact and we can reopen the case.",
        "Please first contact GadgetHub to try to resolve the issue, and keep any proof of that contact. If the order is still "
        "unresolved after 15 days, please reach back out to us and we'll review the case.",
    ]
    for reply in ok_request_info:
        assert check(Decision.REQUEST_INFO, reply, 0) == []
    reject = ("We reviewed your dispute and can't provide a refund or provisional credit for this charge. Our records already "
              "show a matching merchant credit from BookNest posted on 2026-07-15.")
    assert check(Decision.REJECT, reject, 0) == []
    assert check(Decision.REJECT, "We have issued a provisional credit of 59.00 EUR.", 0)  # still caught


def test_failed_self_check_triggers_targeted_redraft():
    class BadFirstDraft(RuleBrain):
        def __init__(self, merchants):
            super().__init__(merchants)
            self.seen_errors = []

        def draft_response(self, state):
            self.seen_errors.append(list(state.self_check_errors))
            if len(self.seen_errors) == 1:
                return "We have refunded you. (Policy reference: POL-DUP-02)"  # wrong for a reject
            return super().draft_response(state)

    led = build_fixture_ledger()
    brain = BadFirstDraft(led.merchants())
    deps = Deps(brain=brain, ledger=led, kb=KnowledgeBase(Settings().kb_dir, "v1"))
    out = run_case(build_graph(deps), CaseState(case_id="d", customer_id="C002", as_of="2026-09-30",
                   narrative="Netflux charged me 15.99 EUR twice, once in July and again in August."))
    assert out["decision"] == Decision.REJECT and out["draft_attempts"] == 2 and out["self_check_errors"] == []
    assert brain.seen_errors[0] == [] and any("must not promise" in e for e in brain.seen_errors[1])
