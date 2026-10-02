"""Policy v1 vs v2: the same complaint decided under both policy versions, with the same code."""

import streamlit as st

from app import core, ui

PRESETS = {"C013": "100-day-old double charge (v1 window 120 days, v2 90 days)",
           "C014": "350 EUR unrecognised payment (v1 review threshold 500 EUR, v2 300 EUR)",
           "C001": "control: recent 9.99 duplicate (same under both)"}


def render() -> None:
    st.title("📜 Policy v1 vs v2")
    st.caption("Rule values (filing window, review threshold) live in the policy documents, not in code. "
               "Publishing v2 changes decisions without a code change. Same complaint, same brain, two policy versions:")
    mode, cfg = ui.sidebar()
    if mode == "replay":
        st.info("Switch to Live mode in the sidebar: this page runs the agent (the rule brain works offline).")
        return
    cid = st.radio("Example", list(PRESETS), format_func=lambda c: f"{c}: {PRESETS[c]}")
    narrative = st.text_area("Complaint", value=core.SUGGESTED[cid], key=f"pol-{cid}")
    if st.button("Decide under v1 and v2", type="primary"):
        cols = st.columns(2, gap="large")
        for col, version in zip(cols, ("v1", "v2")):
            with col:
                st.subheader(f"Policy {version}")
                run_cfg = dict(cfg, kb_version=version)
                agent = ui.agent_for(run_cfg, human_in_loop=False)  # escalations finish as "escalate" here
                last = ui.render_events(core.run_events(agent, run_cfg, cid, narrative), keep_open=False)
                s = last.get("state", {})
                ui.decision_badge(s.get("decision"), s.get("refund_amount", 0.0))
                st.caption(f"params {s.get('policy_params')} · cited {', '.join(s.get('cited_clauses') or [])}")
                if s.get("human_reason"):
                    st.caption(f"guard: {s['human_reason']}")
