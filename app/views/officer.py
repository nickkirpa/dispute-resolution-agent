"""Dispute officer view: cases the agent sent to a human; decide and resume them."""

import streamlit as st

from app import core, ui


def render() -> None:
    st.title("🧑‍⚖️ Review queue")
    st.caption("Cases a guard sent to a human (large refunds, low confidence, evidence conflicts, repeat fraud claims). "
               "Your verdict resumes the case from its saved checkpoint, with the brain and policy version it started with.")
    ui.sidebar()
    probe = ui.agent_for(core.AppConfig().run_config())  # any agent can read the shared checkpoints
    queue = [q for q in core.pending_cases(probe, ui.conn()) if q["case_id"] in ui.visible_cases([q["case_id"]])]
    if not queue:
        st.success("Nothing waiting for review. Try customer C006 (899 EUR) or C007 (repeat claims) on the first page.")
        return
    labels = {q["case_id"]: f"{q['case_id']} · {q['customer_id']} · {q['dispute_type']} · proposed {q['proposed']}" for q in queue}
    case_id = st.selectbox("Waiting for review", list(labels), format_func=labels.get)
    case = core.get_case(probe, case_id)
    q = next(x for x in queue if x["case_id"] == case_id)

    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("**Customer wrote**")
        st.write(case["narrative"])
        ui.why_human(q["reason"] or "the agent chose to escalate")
        st.markdown("**Evidence the agent gathered**")
        for e in case.get("evidence", []):
            st.markdown(f"- `{e['kind']}`: {e['summary']}")
        st.caption(f"Proposed: {q['proposed']} {q['refund_amount']:.2f} EUR · cited {', '.join(case.get('cited_clauses') or [])} "
                   f"· policy {case.get('kb_version')} · run config {case.get('run_config', {}).get('brain')}")
    with b:
        with st.form("verdict"):
            decision = st.radio("Your decision", ["refund", "reject", "request_info", "escalate"],
                                format_func=lambda d: ui.DECISION_STYLE[d][0])
            amount = st.number_input("Refund amount (EUR)", min_value=0.0, value=float(q["refund_amount"]), step=1.0)
            note = st.text_input("Note for the audit trail", placeholder="e.g. called the customer, confirmed fraud")
            submitted = st.form_submit_button("Submit verdict", type="primary")
    if submitted:
        verdict = {"decision": decision, "note": note} | ({"refund_amount": amount} if decision == "refund" else {})
        stored = core.stored_config(probe, case_id) or core.AppConfig().run_config()
        if stored.get("brain") == "llm" and core.public_mode() and not ui.session_key():
            st.caption("This case was started with an LLM; without your key in this session the reply is written by the rule brain.")
            stored = dict(stored, brain="rules", evidence_mode="plan")
        agent = ui.agent_for(stored)
        last = ui.render_events(core.resume_events(agent, case_id, verdict))
        ui.show_outcome(last)
