"""Publishing a new policy version changes decisions without code changes, and every decision records its version."""

from datetime import date, timedelta

from dispute_agent.brain import RuleBrain
from dispute_agent.config import Settings
from dispute_agent.graph import Deps, build_graph, run_case
from dispute_agent.state import CaseState, Decision
from dispute_agent.tools import KnowledgeBase, Ledger

AS_OF = date(2026, 9, 30)


def _ledger() -> Ledger:
    led = Ledger(":memory:")
    led.load(customers=[("P1", "late filer", "2021-01-01"), ("P2", "mid value", "2021-01-01")],
             transactions=[("P1a", "P1", "ShoeBox", 80.0, AS_OF - timedelta(days=100), "online"),
                           ("P1b", "P1", "ShoeBox", 80.0, AS_OF - timedelta(days=99), "online"),
                           ("P2a", "P2", "AirFly", 350.0, AS_OF - timedelta(days=10), "online")])
    return led


def _run(version: str, customer: str, narrative: str) -> dict:
    led = _ledger()
    deps = Deps(brain=RuleBrain(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir, version))
    return run_case(build_graph(deps), CaseState(case_id=f"{version}-{customer}", customer_id=customer, narrative=narrative, as_of=AS_OF))


LATE_DUP = "ShoeBox charged me twice, 80.00 EUR each time, for one pair of shoes."
MID_UNA = "I don't recognise a 350.00 EUR payment to AirFly. I did not make it."


def test_shorter_filing_window_in_v2_rejects_a_100_day_old_claim():
    v1, v2 = _run("v1", "P1", LATE_DUP), _run("v2", "P1", LATE_DUP)
    assert v1["decision"] == Decision.REFUND and v1["refund_amount"] == 80.0       # 100 days <= 120
    assert v2["decision"] == Decision.REJECT and v2["cited_clauses"] == ["POL-GEN-02"]  # 100 days > 90
    assert v1["policy_params"]["filing_window_days"] == 120 and v2["policy_params"]["filing_window_days"] == 90


def test_lower_review_threshold_in_v2_sends_350_eur_to_a_human():
    v1, v2 = _run("v1", "P2", MID_UNA), _run("v2", "P2", MID_UNA)
    assert v1["decision"] == Decision.REFUND and v1["refund_amount"] == 350.0      # 350 <= 500
    assert v2["decision"] == Decision.ESCALATE and v2["refund_amount"] == 0.0     # 350 > 300
    assert "300 EUR (POL-GEN-04)" in v2["human_reason"]


def test_every_decision_records_policy_version_and_cited_clause_versions():
    out = _run("v2", "P2", MID_UNA)
    assert out["kb_version"] == "v2"
    assert all(c.version == "v2" for c in out["clauses"])
