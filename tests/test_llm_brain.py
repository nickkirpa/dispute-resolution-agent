"""LLMBrain wiring tests with a fake Anthropic client (no network, no API key)."""

from types import SimpleNamespace

import pytest

from dispute_agent.brain import Classification, RuleBrain
from dispute_agent.config import Settings
from dispute_agent.graph import Deps, build_graph, run_case
from dispute_agent.llm_brain import LLMBrain, LLMRefusal
from dispute_agent.state import CaseState, Decision, DisputeType
from dispute_agent.tools import KnowledgeBase
from eval.fixtures import build_fixture_ledger


class FakeMessages:
    def __init__(self, outputs, stop_reason="end_turn"):
        self.outputs, self.stop_reason, self.calls = list(outputs), stop_reason, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        out = self.outputs.pop(0) if self.stop_reason != "refusal" else None
        return SimpleNamespace(parsed_output=out, usage=usage, stop_reason=self.stop_reason, stop_details=SimpleNamespace(category="test"))


def test_cost_tracking_uses_model_price():
    fake = FakeMessages([Classification(dispute_type=DisputeType.UNAUTHORIZED, confidence=0.9)])
    brain = LLMBrain(model="claude-opus-5-5", client=SimpleNamespace(messages=fake))
    from dispute_agent.state import Claim

    assert brain.classify("I did not make this payment", Claim()).dispute_type == DisputeType.UNAUTHORIZED
    assert fake.calls[0]["output_format"] is Classification
    assert brain.usage.cost_usd == pytest.approx((1000 * 4.0 + 200 * 20.0) / 1e6)


def test_refusal_raises():
    brain = LLMBrain(client=SimpleNamespace(messages=FakeMessages([], stop_reason="refusal")))
    from dispute_agent.state import Claim

    with pytest.raises(LLMRefusal):
        brain.classify("x", Claim())


class RefusingDecider(RuleBrain):
    def decide(self, state):
        raise LLMRefusal("test")


def test_refusal_in_graph_escalates_to_human():
    led = build_fixture_ledger()
    deps = Deps(brain=RefusingDecider(led.merchants()), ledger=led, kb=KnowledgeBase(Settings().kb_dir))
    out = run_case(build_graph(deps), CaseState(case_id="r", customer_id="C001", as_of="2026-09-30",
                                                narrative="I was charged twice by SpotiTunes for 9.99 EUR on 2026-09-02."))
    assert out["decision"] == Decision.ESCALATE
    assert "refusal" in out["rationale"]


class FakeRaw:
    def __init__(self, completion, headers):
        self._c, self.headers = completion, headers

    def parse(self):
        return self._c


def test_openai_brain_reads_litellm_cost_header():
    from dispute_agent.llm_brain import OpenAIBrain

    out = Classification(dispute_type=DisputeType.DUPLICATE_CHARGE, confidence=0.8)
    msg = SimpleNamespace(parsed=out, refusal=None)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                 usage=SimpleNamespace(prompt_tokens=300, completion_tokens=40))
    calls = []
    parse = lambda **kw: (calls.append(kw), FakeRaw(completion, {"x-litellm-response-cost": "0.0012"}))[1]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(parse=parse))))
    brain = OpenAIBrain(model="openai/gpt-5.4-mini", client=client)
    from dispute_agent.state import Claim

    assert brain.classify("charged twice", Claim()).dispute_type == DisputeType.DUPLICATE_CHARGE
    assert calls[0]["response_format"] is Classification
    assert brain.usage.cost_usd == pytest.approx(0.0012) and brain.cost_source == "litellm-header"


def test_openai_brain_sends_per_step_effort():
    from dispute_agent.config import _parse_effort
    from dispute_agent.llm_brain import OpenAIBrain
    from dispute_agent.state import Claim

    assert _parse_effort("decide=medium, action=low") == {"decide": "medium", "action": "low"}
    out = Classification(dispute_type=DisputeType.OTHER, confidence=0.5)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=out, refusal=None), finish_reason="stop")],
                                 usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    calls = []
    parse = lambda **kw: (calls.append(kw), FakeRaw(completion, {}))[1]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(parse=parse))))
    brain = OpenAIBrain(client=client, effort_by_step={"classify": "medium"})
    brain.classify("x", Claim())
    brain.complete("x", Classification)  # step "generate": no override, no global -> parameter omitted
    assert calls[0]["reasoning_effort"] == "medium" and "reasoning_effort" not in calls[1]
