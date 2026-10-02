"""Reply facts, guidance and fact checks (issues found while labelling replies, 2026-10-02)."""

from dispute_agent.brain import RuleBrain
from dispute_agent.config import Settings
from dispute_agent.graph import Deps, build_graph, run_case
from dispute_agent.replies import fact_errors, reply_facts
from dispute_agent.state import CaseState, Decision
from dispute_agent.tools import KnowledgeBase
from eval.fixtures import build_fixture_ledger

NETFLUX = dict(case_id="nf", customer_id="C002", as_of="2026-09-30",
               narrative="Netflux charged me 15.99 EUR twice, once in July and again in August.")


def _run(case: dict, brain=None) -> dict:
    led = build_fixture_ledger()
    deps = Deps(brain=brain or RuleBrain(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir, "v1"))
    return run_case(build_graph(deps), CaseState(**case))


def test_writer_gets_computed_day_counts():
    out = _run(NETFLUX)
    facts = reply_facts(CaseState(**{k: out[k] for k in ("case_id", "customer_id", "narrative", "as_of", "evidence",
                                                          "decision", "claim", "refund_amount")}))
    assert facts["days_between_charges"] == 31 and facts["days_since_transaction"] == 87


def test_wrong_day_count_in_reply_is_caught():
    """The real reply said '40 days apart' for a 30-day gap; here the gap is 31 days."""
    out = _run(NETFLUX)
    state = CaseState(**{k: out[k] for k in CaseState.model_fields if k in out})
    bad = state.model_copy(update={"response_draft": "The two charges are 40 days apart, so they are recurring."})
    good = state.model_copy(update={"response_draft": "The two charges are 31 days apart, more than the 3 days policy allows."})
    assert any("40 days" in e for e in fact_errors(bad))
    assert fact_errors(good) == []


def test_not_found_reply_must_say_so_and_template_does():
    out = _run(dict(case_id="nf2", customer_id="C012", as_of="2026-09-30", narrative="FitClub charged me twice, 49.00 EUR each time."))
    assert out["decision"] == Decision.REQUEST_INFO and out["self_check_errors"] == []
    assert "could not find a charge" in out["response_draft"]
    state = CaseState(**{k: out[k] for k in CaseState.model_fields if k in out})
    reask = state.model_copy(update={"response_draft": "Please send the transaction date, merchant name and amount."})
    assert any("could not find the charge" in e for e in fact_errors(reask))


def test_rejection_template_gives_next_step():
    out = _run(NETFLUX)
    assert out["decision"] == Decision.REJECT
    assert "cancel the subscription directly with the merchant" in out["response_draft"] and "sorry" in out["response_draft"]


def test_not_found_check_accepts_typographic_apostrophe():
    """Real gpt-5.4-mini replies used 'couldn’t' (U+2019) and were wrongly flagged."""
    from dispute_agent.replies import NOT_FOUND

    assert NOT_FOUND.search("We couldn’t find a charge matching ShoeBox for €174.90 on your account.")
    assert NOT_FOUND.search("We couldn't find it") and NOT_FOUND.search("we could not find any such charge")
    assert not NOT_FOUND.search("Please send the transaction date.")
