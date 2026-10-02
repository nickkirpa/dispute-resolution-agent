"""Shared Streamlit pieces: sidebar settings, cached agents, and the step-by-step case renderer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app import core

DECISION_STYLE = {"refund": ("Refund", "green"), "request_info": ("Ask the customer for info", "blue"),
                  "reject": ("Reject", "red"), "escalate": ("Sent to a dispute officer", "orange")}
ROUTER_DIR = core.ROOT / "models" / "router"


@st.cache_resource
def _conn(db: str):
    return core.connect(Path(db))


@st.cache_resource(show_spinner="Loading the agent…")
def _agent(config_json: str, human_in_loop: bool, db: str):
    return core.build_agent(json.loads(config_json), _conn(db), human_in_loop=human_in_loop)


def agent_for(config: dict, human_in_loop: bool = True):
    return _agent(json.dumps(config, sort_keys=True), human_in_loop, str(core.DB))


def conn():
    return _conn(str(core.DB))


def sidebar() -> tuple[str, dict]:
    """Returns (mode, run_config). Mode: 'live' runs the agent; 'replay' plays recorded LLM runs (no API key)."""
    st.sidebar.header("Settings")
    has_llm = core.llm_available()
    mode = st.sidebar.radio("Mode", ["live", "replay"], horizontal=True,
                            format_func=lambda m: "Live agent" if m == "live" else "Recorded LLM runs",
                            help="Recorded runs are real LLM agent runs saved earlier; they need no API key.")
    cfg = core.AppConfig()
    if mode == "live":
        cfg.brain = st.sidebar.radio("Brain", ["rules", "llm"], horizontal=True, disabled=not has_llm,
                                     format_func=lambda b: "Rules (offline)" if b == "rules" else f"LLM ({cfg.model})",
                                     help=None if has_llm else "No API key configured: the rule brain runs offline.")
        cfg.evidence_mode = st.sidebar.radio("Evidence gathering", ["plan", "agent"], horizontal=True,
                                             disabled=cfg.brain != "llm",
                                             format_func=lambda e: "Scripted plan" if e == "plan" else "LLM agent loop")
        cfg.kb_version = st.sidebar.radio("Policy version", ["v1", "v2"], index=1, horizontal=True,
                                          help="v2: 90-day filing window, 300 EUR human-review threshold")
        if cfg.brain == "llm" and (ROUTER_DIR / "model.safetensors").exists():
            if st.sidebar.toggle("Fine-tuned router first", value=False, help="ModernBERT classifies; LLM below 0.9 confidence"):
                cfg.router_path = str(ROUTER_DIR)
        if cfg.brain == "rules":
            cfg.evidence_mode = "plan"
    st.sidebar.caption("Fictional bank, synthetic data. Nothing here is real customer data.")
    return mode, cfg.run_config()


def why_human(reason: str) -> None:
    """Code guards say 'guard: ...'; anything else is the model's own reason for escalating. Label them differently."""
    if reason.startswith("guard:"):
        st.warning(f"Code guard: {reason.removeprefix('guard: ').replace('; guard: ', '; ')}", icon="🛡️")
    else:
        st.info(f"Agent's reasoning for escalating: {reason}", icon="🤖")


def decision_badge(decision: str | None, refund: float = 0.0) -> None:
    label, color = DECISION_STYLE.get(decision or "", (decision or "–", "gray"))
    st.markdown(f"### :{color}-badge[{label}]" + (f" &nbsp; **{refund:.2f} EUR**" if decision == "refund" else ""))


def _step_detail(node: str, u: dict) -> None:
    if node == "intake" and u.get("claim"):
        c = u["claim"]
        st.write(f"Merchant **{c.get('merchant') or '?'}**, amount **{c.get('amount') or '?'}**, "
                 f"date **{c.get('transaction_date') or '?'}**, contacted merchant: {c.get('contacted_merchant')}")
    elif node == "classify":
        by = u.get("classified_by", "")
        r = u.get("router_prediction") or {}
        st.write(f"**{u.get('dispute_type')}** (confidence {u.get('type_confidence', 0):.2f}, by {by or 'brain'})"
                 + (f" · router said {r.get('type')} at {r.get('confidence')}" if r else ""))
    elif node == "gather_evidence":
        calls = [c for c in u.get("tool_calls", []) if c["tool"] != "finish"]
        st.write(f"{len(calls)} tool calls · mode **{u.get('evidence_mode')}**"
                 + (f" · code filled: {', '.join(u['coverage_fills'])}" if u.get("coverage_fills") else ""))
        if calls:
            st.dataframe(pd.DataFrame([{"tool": c["tool"], "args": {k: v for k, v in c["args"].items() if v is not None and k != "tool"},
                                        "why": c["reason"], "result": c["observation"]} for c in calls]), hide_index=True, width="stretch")
    elif node == "policy_check":
        st.write("Missing evidence: " + (", ".join(u.get("missing_evidence") or []) or "none"))
    elif node == "decide":
        st.write(f"Decision **{u.get('decision')}**, cited {', '.join(u.get('cited_clauses') or [])}"
                 + (f" · dropped as not applicable: {', '.join(u['dropped_citations'])}" if u.get("dropped_citations") else ""))
        if u.get("human_reason"):
            why_human(u["human_reason"])
    elif node == "human_review":
        st.write(f"Officer decision: **{u.get('decision')}**")
    elif node == "self_check":
        errs = u.get("self_check_errors") or []
        st.write("Reply checks passed" if not errs else f"Failed checks: {errs}")


def render_events(events, delay: float = 0.0, keep_open: bool = True) -> dict:
    """Show a case step by step. Returns the last event (done / interrupt). keep_open: leave the trace expanded."""
    last = {}
    with st.status("Agent working…", expanded=True) as status:
        for ev in events:
            last = ev
            if ev["type"] == "start":
                st.caption(f"case **{ev['case_id']}** · brain {ev['config']['brain']} · evidence {ev['config']['evidence_mode']} "
                           f"· policy {ev['config']['kb_version']}")
            elif ev["type"] == "resume":
                st.caption(f"resuming **{ev['case_id']}** with the officer's verdict: {ev['verdict']}")
            elif ev["type"] == "step":
                status.update(label=ev["label"] + "…")
                with st.container(border=True):
                    st.markdown(f"**{ev['label']}**")
                    _step_detail(ev["node"], ev["update"])
            if delay:
                time.sleep(delay)
        status.update(label="Waiting for a dispute officer" if last.get("type") == "interrupt" else "Done: agent steps",
                      state="complete", expanded=keep_open)  # the trace is usually the interesting part
    return last


def show_outcome(last: dict) -> None:
    if last.get("type") == "interrupt":
        p = last["payload"]
        st.info(f"**Sent to a dispute officer.** {p.get('reason')}\n\nCase **{last['case_id']}** is in the Review queue.", icon="🧑‍⚖️")
        return
    s = last.get("state", {})
    decision_badge(s.get("decision"), s.get("refund_amount", 0.0))
    st.markdown("**Reply to the customer**")
    st.info(s.get("response_draft", ""), icon="✉️")
    u = s.get("usage") or {}
    st.caption(f"{u.get('llm_calls', 0)} LLM calls · ${u.get('cost_usd', 0):.4f} · policy {s.get('kb_version')} · "
               f"cited {', '.join(s.get('cited_clauses') or [])}")


def replay_picker() -> dict | None:
    replays = core.load_replays()
    if not replays:
        st.warning("No recorded runs yet. Record them with `uv run python app/record_replays.py`.")
        return None
    titles = {r["title"]: r for r in replays}
    choice = st.selectbox("Recorded LLM run", list(titles), help="Real agent runs, saved with app/record_replays.py")
    st.caption(titles[choice]["description"])
    return titles[choice]


def series_color() -> str:
    """Validated categorical slot 1 (dataviz reference palette), stepped per theme."""
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return "#3987e5" if theme == "dark" else "#2a78d6"


def text_color() -> str:
    """Secondary text ink per theme (value labels wear text tokens, never the series color)."""
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return "#c3c2b7" if theme == "dark" else "#52514e"


def path_rel(p: Path) -> str:
    return str(p.relative_to(core.ROOT))
