"""Customer view: pick a demo customer, write a complaint, watch the agent work."""

import pandas as pd
import streamlit as st

from app import core, ui


def render() -> None:
    st.title("🧾 File a dispute")
    st.caption("You are a Northwind Bank customer (fictional). Pick an account, describe the problem, and watch the agent investigate.")
    mode, cfg = ui.sidebar()

    if mode == "replay":
        rep = ui.replay_picker()
        if rep and st.button("▶ Play recorded run", type="primary"):
            last = ui.render_events(rep["events"], delay=0.5)
            ui.show_outcome(last)
        return

    customers = {c["customer_id"]: c for c in core.demo_customers()}
    left, right = st.columns([2, 3], gap="large")
    with left:
        cid = st.selectbox("Customer account", list(customers),
                           format_func=lambda c: f"{c} · {', '.join(sorted({t['merchant'] for t in customers[c]['transactions']}))}")
        c = customers[cid]
        st.markdown("**Recent transactions**")
        st.dataframe(pd.DataFrame(c["transactions"]), hide_index=True, width="stretch")
        if c["prior_disputes"]:
            st.caption(f"{c['prior_disputes']} earlier unauthorized-payment claims in the last 12 months")
    with right:
        narrative = st.text_area("Your message to the bank", value=c["suggested"], height=140, key=f"msg-{cid}")
        go = st.button("Send complaint", type="primary", disabled=not narrative.strip())
    if go:
        agent = ui.agent_for(cfg)
        last = ui.render_events(core.run_events(agent, cfg, cid, narrative))
        ui.show_outcome(last)
