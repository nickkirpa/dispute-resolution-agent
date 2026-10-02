"""Results: the headline eval numbers (snapshot in app/data/metrics.json, built from eval reports by build_metrics.py)."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from app import core, ui

LABELS = {"model_over_escalation": "Model escalated (policy had an answer)", "guard_escalation": "Guard escalated",
          "wrong_policy_outcome": "Wrong outcome under policy", "missed_escalation": "Missed an escalation",
          "evidence_not_found": "Asked for info it could have found", "wrong_type": "Wrong dispute type",
          "wrong_refund_amount": "Wrong refund amount", "unsafe_refund": "Unsafe refund", "crash": "Crash"}


def bar(df: pd.DataFrame, value: str, label: str, fmt: str, title: str, tooltip: list) -> alt.Chart:
    """Single-series horizontal bars: thin, rounded data-ends, value labels, hover tooltip, recessive axes."""
    base = alt.Chart(df).encode(y=alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(color=ui.series_color(), cornerRadiusEnd=4, height=14).encode(
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.25, tickCount=5)),
        tooltip=tooltip)
    text = base.mark_text(align="left", dx=6, color=ui.text_color()).encode(x=f"{value}:Q", text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=alt.Step(30))


def render() -> None:
    st.title("📊 Results")
    ui.sidebar()
    m = json.loads((core.ROOT / "app" / "data" / "metrics.json").read_text())
    cfgs = pd.DataFrame(m["configs"])
    best = cfgs.loc[cfgs["config"] == "gpt-5.4-mini · plan + router"].iloc[0]
    pol = m["policy_change"]

    t = st.columns(4)
    t[0].metric("Decision accuracy", f"{best.decision_accuracy:.1%}", help="gpt-5.4-mini + router, mean of 3 runs on 203 cases", border=True)
    t[1].metric("Unsafe refunds", f"{best.unsafe_refund_rate:.0%}", help="money paid where policy says no", border=True)
    t[2].metric("Cost per case", f"${best.cost_per_case_usd:.4f}", border=True)
    t[3].metric("Policy update", f"{pol['decisions_changed']}/{pol['decisions_changed']} flips",
                help=f"{pol['n']} boundary cases: {pol['v1_correct']}/{pol['n']} correct under v1, {pol['v2_correct']}/{pol['n']} under v2",
                border=True)

    st.subheader("Configurations on golden_v1 (203 human-reviewed cases)")
    tips = ["config", alt.Tooltip("decision_accuracy:Q", format=".1%"), alt.Tooltip("cost_per_case_usd:Q", format="$.4f"),
            alt.Tooltip("latency_s:Q", format=".1f", title="latency (s)"), "runs", "note"]
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Decision accuracy**")
        st.altair_chart(bar(cfgs, "decision_accuracy", "config", ".0%", "decision accuracy", tips), width="stretch")
    with b:
        st.markdown("**Cost per case (USD)**")
        st.altair_chart(bar(cfgs, "cost_per_case_usd", "config", "$.4f", "cost per case", tips), width="stretch")
    st.caption("Unsafe refunds are 0% for every configuration shown. gpt-5.4-nano had 2 (1%) before the repeat-claims guard "
               "was added: the weak model exposed the one policy rule without a code guard.")
    with st.expander("Table"):
        st.dataframe(cfgs.drop(columns=["source"]), hide_index=True, width="stretch")

    st.subheader("Why cases fail (gpt-5.4-mini)")
    tax = pd.DataFrame(m["failure_taxonomy"])
    tax = tax[tax["count"] > 0].assign(cause=lambda d: d["cause"].map(LABELS)).sort_values("count", ascending=False)
    st.altair_chart(bar(tax, "count", "cause", "d", "failed cases (all gpt-5.4-mini runs)", ["cause", "count"]), width="stretch")
    st.caption(f"Scope: {m['failure_taxonomy_scope']}. No unsafe refunds, wrong dispute types or wrong amounts: "
               "when the agent fails, it fails cautious.")

    st.subheader("Fine-tuned router vs zero-shot LLM (Banking77 test, 3,125 messages)")
    r = m["router"]
    st.dataframe(pd.DataFrame([
        {"model": "ModernBERT-base (fine-tuned locally)", "dispute-type accuracy": f"{r['router']['type_accuracy']:.1%}",
         "macro-F1": f"{r['router']['type_macro_f1']:.3f}", "latency": f"{r['router']['latency_ms']:.0f} ms", "cost / 1k": "~$0"},
        {"model": "gpt-5.4-mini (zero-shot)", "dispute-type accuracy": f"{r['llm']['type_accuracy']:.1%}",
         "macro-F1": f"{r['llm']['type_macro_f1']:.3f}", "latency": f"{r['llm']['latency_ms']:.0f} ms",
         "cost / 1k": f"${r['llm']['cost_per_1k_usd']:.2f}"}]), hide_index=True, width="stretch")
    st.caption(f"Banking77 intent accuracy {r['router']['intent_accuracy']:.1%}, trained in {r['router']['train_minutes']:.0f} min on an "
               "Apple M5 Pro. Much of the gap to the LLM is label-convention disagreement, not LLM error.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("LLM judge vs human labels")
        j = m["judge"]
        st.dataframe(pd.DataFrame([{"question": q, "kappa v1": j["v1"][q], "kappa v2": j["v2"][q]}
                                   for q in ("facts", "decision", "promises", "clarity", "send")]), hide_index=True, width="stretch")
        st.caption(f"40 replies (15 with injected defects). 'OK to send' agreement {j['v1']['send_agreement']:.0%} (v1) / "
                   f"{j['v2']['send_agreement']:.0%} (v2). v2 was tuned on the same replies.")
    with c2:
        st.subheader("Policy retrieval (156 queries, 37 clauses)")
        rt = m["retrieval"]
        st.dataframe(pd.DataFrame([{"setting": k.split("/")[0], "mode": k.split("/")[1], "recall@3": round(v["recall@3"], 3),
                                    "MRR": round(v["mrr"], 3)} for k, v in rt.items()]), hide_index=True, width="stretch")
        st.caption("Filtered by dispute type (the agent's default) every mode finds the clause; unfiltered, hybrid wins.")
