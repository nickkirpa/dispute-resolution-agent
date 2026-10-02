"""Case trace: everything the agent did on a case, for audit and debugging."""

import pandas as pd
import streamlit as st

from app import core, ui


def render() -> None:
    st.title("🔍 Case trace")
    st.caption("The full audit trail of a case: steps, tool calls, evidence, policy version, citations, guards and cost.")
    ui.sidebar()
    probe = ui.agent_for(core.AppConfig().run_config())
    ids = core.list_cases(ui.conn())
    if not ids:
        st.info("No cases yet. File a dispute first.")
        return
    case_id = st.selectbox("Case (newest first)", ids)
    c = core.get_case(probe, case_id)
    top = st.columns(4)
    top[0].metric("Decision", c.get("decision") or "pending")
    top[1].metric("Refund", f"{c.get('refund_amount', 0):.2f} EUR")
    top[2].metric("LLM calls", (c.get("usage") or {}).get("llm_calls", 0))
    top[3].metric("Cost", f"${(c.get('usage') or {}).get('cost_usd', 0):.4f}")
    st.caption("Path: " + " → ".join(c.get("trace", [])) + (f" · waiting at {c['next']}" if c.get("next") else ""))

    t1, t2, t3, t4, t5 = st.tabs(["Tool calls", "Evidence", "Policy", "Reply & checks", "Raw state"])
    with t1:
        calls = c.get("tool_calls", [])
        st.caption(f"evidence mode **{c.get('evidence_mode')}**"
                   + (f" · code filled: {', '.join(c['coverage_fills'])}" if c.get("coverage_fills") else "")
                   + (f" · fallback: {c['fallback_error']}" if c.get("fallback_error") else ""))
        if calls:
            st.dataframe(pd.DataFrame([{"tool": x["tool"], "args": {k: v for k, v in x["args"].items() if v is not None and k != "tool"},
                                        "reason": x["reason"], "ok": x["ok"], "observation": x["observation"]} for x in calls]),
                         hide_index=True, width="stretch")
    with t2:
        for e in c.get("evidence", []):
            st.markdown(f"**{e['kind']}** ({e['source']}): {e['summary']}")
    with t3:
        st.markdown(f"Policy **{c.get('kb_version')}** · parameters `{c.get('policy_params')}`")
        st.markdown(f"Cited: **{', '.join(c.get('cited_clauses') or []) or '–'}**"
                    + (f" · dropped as not applicable: {', '.join(c['dropped_citations'])}" if c.get("dropped_citations") else ""))
        if c.get("human_reason"):
            ui.why_human(c["human_reason"])
        st.dataframe(pd.DataFrame([{"clause": x["clause_id"], "version": x["version"], "title": x["title"]} for x in c.get("clauses", [])]),
                     hide_index=True, width="stretch")
    with t4:
        st.info(c.get("response_draft") or "–", icon="✉️")
        st.caption(f"draft attempts {c.get('draft_attempts', 0)} · self-check errors: {c.get('self_check_errors') or 'none'}")
    with t5:
        st.json(c, expanded=False)
