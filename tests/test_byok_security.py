"""Bring-your-own-key safety for the public demo: where a visitor's key may and may not go."""

from pathlib import Path

import pytest

from app import core
from dispute_agent.brain import RuleBrain
from dispute_agent.config import Settings
from dispute_agent.llm_brain import make_llm_brain

SECRET = "sk-test-SECRET-7f3a9c21d4"


def test_explicit_key_uses_official_endpoint_not_env_proxy(monkeypatch):
    """A visitor key must never be sent to a proxy configured in the server environment."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://some-internal-proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-key-must-not-be-used")
    brain = make_llm_brain(Settings(provider="openai", model="gpt-5.4-mini"), api_key=SECRET)
    assert brain.client.api_key == SECRET and "api.openai.com" in str(brain.client.base_url)

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://some-internal-proxy.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key-must-not-be-used")
    brain = make_llm_brain(Settings(provider="anthropic", model="claude-opus-5-5"), api_key=SECRET)
    assert brain.client.api_key == SECRET and "api.anthropic.com" in str(brain.client.base_url)


def test_key_never_reaches_case_state_or_checkpoint_files(tmp_path, monkeypatch):
    seen = {}

    class KeyedStub(RuleBrain):  # stands in for the LLM brain: no network, records the key it was given
        name = "llm:stub"

    def fake_make_llm_brain(settings, api_key=None):
        seen["key"] = api_key
        return KeyedStub(known_merchants=core.demo_ledger().merchants())

    monkeypatch.setattr("dispute_agent.llm_brain.make_llm_brain", fake_make_llm_brain)
    db = tmp_path / "cases.sqlite"
    config = core.AppConfig(brain="llm", provider="openai", model="gpt-5.4-mini").run_config()
    agent = core.build_agent(config, core.connect(db), api_key=SECRET)
    events = list(core.run_events(agent, config, "C006", core.SUGGESTED["C006"]))  # pauses: state is checkpointed
    assert seen["key"] == SECRET
    assert SECRET not in repr(events) and SECRET not in repr(core.get_case(agent, events[-1]["case_id"]))
    raw = b"".join(p.read_bytes() for p in Path(tmp_path).glob("cases.sqlite*"))
    assert raw and SECRET.encode() not in raw and b"SECRET" not in raw


def test_public_mode_ignores_server_keys_and_isolates_sessions(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DISPUTE_AGENT_PUBLIC", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-key-must-not-be-used")
    monkeypatch.setenv("DISPUTE_AGENT_PROVIDER", "openai")
    db = str(tmp_path / "c.sqlite")

    def script(view, db):
        import importlib
        import sys
        from pathlib import Path

        sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
        from app import core

        core.DB = Path(db)
        importlib.import_module(f"app.views.{view}").render()

    visitor_a = AppTest.from_function(script, args=("customer", db), default_timeout=120).run()
    brain = next(r for r in visitor_a.radio if r.label == "Brain")
    assert brain.disabled  # no visitor key: the server's env key is not offered
    visitor_a.button[0].click().run()  # visitor A files a case (rule brain)
    assert not visitor_a.exception
    visitor_b = AppTest.from_function(script, args=("trace", db), default_timeout=120).run()  # a different session
    assert any("No cases yet" in i.value for i in visitor_b.info)  # B cannot see A's complaint


def test_mask_removes_key_from_messages():
    assert SECRET not in core.mask(f"401 Incorrect API key provided: {SECRET}", SECRET)
    assert "d4" not in core.mask(f"key ending ...{SECRET[-6:]}", SECRET).split("...")[-1].replace("***", "")


def test_key_entry_flow_in_the_ui(tmp_path, monkeypatch):
    """Regression: 'Use key' raised StreamlitWidgetAlreadyInstantiatedError (widget state set after drawing)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DISPUTE_AGENT_PUBLIC", "1")
    db = str(tmp_path / "c.sqlite")

    def script(view, db):
        import importlib
        import sys
        from pathlib import Path

        sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
        from app import core

        core.DB = Path(db)
        importlib.import_module(f"app.views.{view}").render()

    for view in ("customer", "trace", "officer", "policy", "metrics"):  # the sidebar (and key form) is on every page
        at = AppTest.from_function(script, args=(view, db), default_timeout=120).run()
        button = lambda label: next(b for b in at.button if b.label == label)  # noqa: E731
        button("Use key").click().run()  # empty field: a warning, not a crash
        assert not at.exception and any("Paste a key first" in w.value for w in at.warning)

        at.text_input(key="byok_input").input(SECRET).run()
        button("Use key").click().run()
        assert not at.exception, (view, [e.value for e in at.exception])
        assert at.session_state["byok"]["key"] == SECRET
        assert at.session_state["byok_input"] == ""  # raw value wiped from the widget
        if view != "metrics":  # pages with live settings: the LLM brain is now selectable and selected
            brain = next(r for r in at.radio if r.label == "Brain")
            assert not brain.disabled and brain.value == "llm"
        assert all(SECRET not in m.value for m in at.markdown)  # never rendered

        button("Clear").click().run()
        assert not at.exception and "byok" not in at.session_state
