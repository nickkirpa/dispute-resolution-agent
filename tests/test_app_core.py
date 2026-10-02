"""The Streamlit app's non-UI logic (app/core.py), offline with the rule brain."""

import json

from app import core


def _agent(tmp_path, **cfg):
    conn = core.connect(tmp_path / "cases.sqlite")
    config = core.AppConfig(brain="rules", **cfg)
    return core.build_agent(config, conn), config, conn


def test_case_streams_every_step_and_finishes(tmp_path):
    agent, config, _ = _agent(tmp_path)
    events = list(core.run_events(agent, config, "C001", core.SUGGESTED["C001"]))
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    assert [e["node"] for e in events if e["type"] == "step"][:3] == ["intake", "classify", "gather_evidence"]
    final = events[-1]["state"]
    assert final["decision"] == "refund" and final["refund_amount"] == 9.99
    json.dumps(events)  # everything is JSON-ready (needed for recorded replays)


def test_human_review_queue_and_resume(tmp_path):
    agent, config, conn = _agent(tmp_path)
    events = list(core.run_events(agent, config, "C006", core.SUGGESTED["C006"]))
    assert events[-1]["type"] == "interrupt"
    case_id = events[-1]["case_id"]
    queue = core.pending_cases(agent, conn)
    assert [c["case_id"] for c in queue] == [case_id] and queue[0]["refund_amount"] == 899.0
    done = list(core.resume_events(agent, case_id, {"decision": "refund", "refund_amount": 899.0, "note": "verified"}))
    assert done[-1]["type"] == "done" and done[-1]["state"]["decision"] == "refund"
    assert core.pending_cases(agent, conn) == [] and case_id in core.list_cases(conn)
    assert core.stored_config(agent, case_id)["brain"] == "rules"


def test_boundary_customers_flip_between_policy_versions(tmp_path):
    outcomes = {}
    for version in ("v1", "v2"):
        agent, config, _ = _agent(tmp_path, kb_version=version)
        for cid in ("C013", "C014"):
            last = list(core.run_events(agent, config, cid, core.SUGGESTED[cid]))[-1]
            state = last["state"] if last["type"] == "done" else core.get_case(agent, last["case_id"])
            outcomes[(version, cid)] = (last["type"], state.get("decision"))
    assert outcomes[("v1", "C013")] == ("done", "refund") and outcomes[("v2", "C013")] == ("done", "reject")
    assert outcomes[("v1", "C014")] == ("done", "refund") and outcomes[("v2", "C014")][0] == "interrupt"


def test_record_and_load_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "REPLAYS", tmp_path / "replays")
    agent, config, _ = _agent(tmp_path)
    events = list(core.run_events(agent, config, "C002", core.SUGGESTED["C002"]))
    core.record(events, "Monthly charge trap", "looks like a duplicate, is a subscription")
    replays = core.load_replays()
    assert replays[0]["title"] == "Monthly charge trap" and replays[0]["events"][-1]["state"]["decision"] == "reject"


def test_demo_customers_have_transactions_and_suggestions():
    rows = {c["customer_id"]: c for c in core.demo_customers()}
    assert len(rows) == 14 and rows["C001"]["transactions"] and rows["C007"]["prior_disputes"] == 2
    assert all(c["suggested"] for c in rows.values())


def test_officer_amount_defaults_to_policy_refund_when_model_escalated(tmp_path, monkeypatch):
    """Regression (manual demo): when the LLM itself escalates, the proposed refund is 0 and the officer had to type 899."""
    from dispute_agent.brain import DecisionProposal, RuleBrain
    from dispute_agent.state import Decision

    def decide(self, state):
        return DecisionProposal(decision=Decision.ESCALATE, confidence=0.9, rationale="needs review", cited_clauses=["POL-GEN-04"])

    monkeypatch.setattr(RuleBrain, "decide", decide)
    agent, config, conn = _agent(tmp_path)
    last = list(core.run_events(agent, config, "C006", core.SUGGESTED["C006"]))[-1]
    case = core.get_case(agent, last["case_id"])
    assert case["refund_amount"] == 0.0 and core.suggested_refund(case) == 899.0
    dup = list(core.run_events(agent, config, "C001", core.SUGGESTED["C001"]))[-1]  # duplicate: refund one charge only
    assert core.suggested_refund(core.get_case(agent, dup["case_id"])) == 9.99
