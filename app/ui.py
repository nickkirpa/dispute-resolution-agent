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


BYOK_MODELS = {"openai": ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"],
               "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"]}


@st.cache_resource
def _conn(db: str):
    return core.connect(Path(db))


@st.cache_resource(show_spinner="Loading the agent…")
def _shared_agent(config_json: str, human_in_loop: bool, db: str):
    """Agents WITHOUT a visitor key (rule brain, or the server's own env key locally): safe to share across sessions."""
    return core.build_agent(json.loads(config_json), _conn(db), human_in_loop=human_in_loop)


def session_key() -> str | None:
    """The visitor's own API key: kept in this browser session's server-side state only."""
    byok = st.session_state.get("byok") or {}
    return byok.get("key") or None


def agent_for(config: dict, human_in_loop: bool = True):
    key = session_key() if config.get("brain") == "llm" else None
    if not key:
        return _shared_agent(json.dumps(config, sort_keys=True), human_in_loop, str(core.DB))
    # Agents holding a visitor key live in that visitor's session only, never in a process-wide cache.
    agents = st.session_state.setdefault("_session_agents", {})
    cache_key = (json.dumps(config, sort_keys=True), human_in_loop)
    if cache_key not in agents:
        agents[cache_key] = core.build_agent(config, _conn(str(core.DB)), human_in_loop=human_in_loop, api_key=key)
    return agents[cache_key]


def conn():
    return _conn(str(core.DB))


def remember_case(case_id: str) -> None:
    mine = st.session_state.setdefault("my_cases", [])
    if case_id not in mine:
        mine.append(case_id)


def visible_cases(ids: list[str]) -> list[str]:
    """Public demo: each visitor only sees cases they created (complaints are free text and may contain anything)."""
    if not core.public_mode():
        return ids
    mine = set(st.session_state.get("my_cases", []))
    return [i for i in ids if i in mine]


def _key_section(cfg: core.AppConfig) -> bool:
    """'Use your own API key'. Returns True if an LLM is usable in this session."""
    public = core.public_mode()
    env_llm = (not public) and core.llm_available()
    with st.sidebar.expander("Use your own API key", expanded=public and not session_key(), icon="🔑"):
        st.caption("Used only in this browser session to call the provider directly. Never stored, logged or shared. "
                   "Remove it with Clear, or close the tab. Use a key with a spend limit.")
        provider = st.radio("Provider", list(BYOK_MODELS), horizontal=True, key="byok_provider",
                            format_func=lambda p: "OpenAI" if p == "openai" else "Anthropic")
        model = st.selectbox("Model", BYOK_MODELS[provider], key="byok_model")
        key = st.text_input("API key", type="password", key="byok_input", placeholder="sk-…")
        c1, c2 = st.columns(2)
        if c1.button("Use key", disabled=not key.strip(), width="stretch"):
            st.session_state["byok"] = {"provider": provider, "model": model, "key": key.strip()}
            st.session_state.pop("_session_agents", None)
            st.session_state["byok_input"] = ""  # do not keep the raw value in the widget
            st.rerun()
        if c2.button("Clear", width="stretch"):
            st.session_state.pop("byok", None)
            st.session_state.pop("_session_agents", None)
            st.rerun()
        if session_key():
            b = st.session_state["byok"]
            st.success(f"Using your {b['provider']} key ({b['model']}) for this session.", icon="✅")
    if session_key():
        b = st.session_state["byok"]
        cfg.provider, cfg.model = b["provider"], b["model"]
        return True
    return env_llm


def sidebar() -> tuple[str, dict]:
    """Returns (mode, run_config). Mode: 'live' runs the agent; 'replay' plays recorded LLM runs (no API key)."""
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["live", "replay"], horizontal=True,
                            format_func=lambda m: "Live agent" if m == "live" else "Recorded LLM runs",
                            help="Recorded runs are real LLM agent runs saved earlier; they need no API key.")
    cfg = core.AppConfig()
    if core.public_mode():
        cfg.provider, cfg.model = "openai", "gpt-5.4-mini"  # display defaults; server env keys are never used here
    if mode == "live":
        has_llm = _key_section(cfg)
        cfg.brain = st.sidebar.radio("Brain", ["rules", "llm"], horizontal=True, disabled=not has_llm,
                                     index=1 if has_llm and session_key() else 0,
                                     format_func=lambda b: "Rules (offline)" if b == "rules" else f"LLM ({cfg.model})",
                                     help=None if has_llm else "Add your own API key above to run the LLM agent.")
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
        try:
            for ev in events:
                last = ev
                _render_event(ev, status)
                if delay:
                    time.sleep(delay)
        except Exception as e:  # provider errors (bad key, rate limit...) shown without the key
            status.update(label="Stopped: the model call failed", state="error", expanded=True)
            st.error(f"{type(e).__name__}: {core.mask(str(e), session_key())}")
            return {"type": "error"}
        status.update(label="Waiting for a dispute officer" if last.get("type") == "interrupt" else "Done: agent steps",
                      state="complete", expanded=keep_open)  # the trace is usually the interesting part
    return last


def _render_event(ev: dict, status) -> None:
    if ev["type"] == "start":
        remember_case(ev["case_id"])
        st.caption(f"case **{ev['case_id']}** · brain {ev['config']['brain']} · evidence {ev['config']['evidence_mode']} "
                   f"· policy {ev['config']['kb_version']}")
    elif ev["type"] == "resume":
        st.caption(f"resuming **{ev['case_id']}** with the officer's verdict: {ev['verdict']}")
    elif ev["type"] == "step":
        status.update(label=ev["label"] + "…")
        with st.container(border=True):
            st.markdown(f"**{ev['label']}**")
            _step_detail(ev["node"], ev["update"])


def show_outcome(last: dict) -> None:
    if last.get("type") == "error":
        return
    if last.get("type") == "interrupt":
        p = last["payload"]
        st.info(f"**Sent to a dispute officer.** {p.get('reason')}\n\nCase **{last['case_id']}** is in the Review queue.", icon="🧑‍⚖️")
        return
    s = last.get("state", {})
    decision_badge(s.get("decision"), s.get("refund_amount", 0.0))
    st.markdown("**Reply to the customer**")
    st.info(s.get("response_draft", ""), icon="✉️")
    u = s.get("usage") or {}
    cost = f"${u.get('cost_usd', 0):.4f}" if u.get("cost_usd") or not u.get("llm_calls") else "cost not tracked for this provider"
    st.caption(f"{u.get('llm_calls', 0)} LLM calls · {cost} · policy {s.get('kb_version')} · "
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
