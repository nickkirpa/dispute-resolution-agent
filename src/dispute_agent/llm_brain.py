"""LLM brains: shared prompts + typed structured outputs, one subclass per provider.

- LLMBrain:    Anthropic SDK, client.messages.parse(output_format=Model)
- OpenAIBrain: OpenAI SDK (OpenAI or a LiteLLM proxy), client.chat.completions.parse(response_format=Model)

Every call is typed: the model returns a validated Pydantic object or the call fails loudly.
Token usage and cost are tracked per call so the eval can report cost per case.
"""

from __future__ import annotations

import json

import anthropic
from pydantic import BaseModel

from .brain import Classification, DecisionProposal
from .config import MODEL_PRICES
from .state import CaseState, Claim, Usage


class LLMRefusal(RuntimeError):
    """The model declined (stop_reason == 'refusal'). The graph escalates the case to a human."""


class Draft(BaseModel):
    text: str


SYSTEM = (
    "You are a card-dispute analyst at Northwind Bank (a fictional bank). You work inside an automated "
    "dispute pipeline: code gathers evidence and enforces thresholds, and you contribute language understanding "
    "and policy judgement. Be literal about evidence: if the ledger does not show something, do not assume it."
)

EXTRACT = (
    "Extract the facts the customer asserts in this complaint. Leave fields null when not stated. "
    "`amount` is the charged amount they dispute; `claimed_correct_amount` is only for 'charged X instead of Y'. "
    "`contacted_merchant` is true only if they say they already tried the merchant.\n\nComplaint:\n{narrative}"
)

CLASSIFY = (
    "Classify the dispute type of this complaint.\n"
    "- duplicate_charge: same charge posted more than once\n"
    "- not_received: paid for goods/services that never arrived\n"
    "- unauthorized: customer did not make or authorize the payment\n"
    "- wrong_amount: charged more than the agreed price\n"
    "- refund_not_processed: merchant promised/approved a refund that never arrived\n"
    "- other: anything else\n\nExtracted claim: {claim}\n\nComplaint:\n{narrative}"
)

DECIDE = (
    "Decide this dispute strictly under the policy clauses provided.\n"
    "Decisions: refund | request_info | reject | escalate.\n"
    "- Cite only clause ids that appear in the clauses below.\n"
    "- If required evidence is listed as missing, the decision must be request_info.\n"
    "- Do not compute refund amounts; code does that.\n\n"
    "Dispute type: {dtype}\nClaim: {claim}\nMissing evidence: {missing}\n\n"
    "Evidence gathered by tools:\n{evidence}\n\nPolicy clauses (version {version}):\n{clauses}\n\n"
    "Complaint:\n{narrative}"
)

DRAFT = (
    "Write a short, plain-English reply to the customer (under 120 words) communicating this decision. "
    "Mention the refund amount exactly as given if the decision is refund, list what is needed if request_info, "
    "and end with '(Policy reference: <clause ids>)'. Do not promise anything the decision does not say.\n\n"
    "Decision: {decision}\nRefund amount (EUR): {amount:.2f}\nMissing evidence: {missing}\nRationale: {rationale}\n"
    "Cited clauses: {cites}\n\nComplaint:\n{narrative}"
)


class PromptBrain:
    """Prompt construction shared by all providers. Subclasses implement `_parse(prompt, schema)`."""

    name: str
    usage: Usage

    def _parse(self, prompt: str, schema: type[BaseModel]) -> BaseModel:  # pragma: no cover - abstract
        raise NotImplementedError

    def extract_claim(self, narrative: str) -> Claim:
        return self._parse(EXTRACT.format(narrative=narrative), Claim)

    def classify(self, narrative: str, claim: Claim) -> Classification:
        return self._parse(CLASSIFY.format(narrative=narrative, claim=claim.model_dump_json()), Classification)

    def decide(self, state: CaseState) -> DecisionProposal:
        prompt = DECIDE.format(
            dtype=state.dispute_type.value,
            claim=state.claim.model_dump_json() if state.claim else "{}",
            missing=state.missing_evidence or "none",
            evidence="\n".join(f"- [{e.kind}] {e.summary} {json.dumps(e.data, default=str)}" for e in state.evidence) or "none",
            version=state.kb_version,
            clauses="\n\n".join(f"{c.clause_id}: {c.title}\n{c.text}" for c in state.clauses),
            narrative=state.narrative,
        )
        return self._parse(prompt, DecisionProposal)

    def draft_response(self, state: CaseState) -> str:
        prompt = DRAFT.format(
            decision=state.decision.value,
            amount=state.refund_amount,
            missing=state.missing_evidence or "none",
            rationale=state.rationale,
            cites=", ".join(state.cited_clauses),
            narrative=state.narrative,
        )
        return self._parse(prompt, Draft).text


class LLMBrain(PromptBrain):
    """Claude via the Anthropic SDK."""

    def __init__(self, model: str = "claude-opus-5-5", client: anthropic.Anthropic | None = None, max_tokens: int = 8000,
                 price_model: str | None = None):
        self.name = f"llm:{model}"
        self.model = model
        self.price_model = price_model or model  # proxy aliases (LiteLLM) map to a real id for pricing
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.usage = Usage()

    def _parse(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        self._track(response.usage)
        if response.stop_reason == "refusal":
            raise LLMRefusal(getattr(response.stop_details, "category", None) or "refused")
        if response.parsed_output is None:
            raise RuntimeError(f"No parsed output (stop_reason={response.stop_reason})")
        return response.parsed_output

    def _track(self, u) -> None:
        if self.price_model not in MODEL_PRICES:
            raise KeyError(f"No price for {self.price_model!r}; set DISPUTE_AGENT_PRICE_MODEL to a real model id {list(MODEL_PRICES)}")
        price_in, price_out = MODEL_PRICES[self.price_model]
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        cost = (u.input_tokens * price_in + cache_write * price_in * 1.25 + cache_read * price_in * 0.1 + u.output_tokens * price_out) / 1e6
        self.usage = self.usage.add(
            Usage(llm_calls=1, input_tokens=u.input_tokens + cache_read + cache_write, output_tokens=u.output_tokens, cost_usd=cost)
        )


class OpenAIBrain(PromptBrain):
    """OpenAI models via the OpenAI SDK, directly or through a LiteLLM proxy (OPENAI_BASE_URL / OPENAI_API_KEY).

    Cost: read from LiteLLM's `x-litellm-response-cost` header when present (the proxy's own pricing).
    Without the header (OpenAI directly), cost stays 0 unless prices are set in MODEL_PRICES.
    """

    def __init__(self, model: str = "openai/gpt-5.4-mini", client=None, max_completion_tokens: int = 8000,
                 reasoning_effort: str | None = None):
        import openai

        self.name = f"llm:{model}"
        self.model = model
        self.client = client or openai.OpenAI()
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.usage = Usage()
        self.cost_source = "unknown"

    def _parse(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            response_format=schema,
            max_completion_tokens=self.max_completion_tokens,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        raw = self.client.chat.completions.with_raw_response.parse(**kwargs)
        completion = raw.parse()
        self._track(completion.usage, raw.headers.get("x-litellm-response-cost"))
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMRefusal(message.refusal)
        if message.parsed is None:
            raise RuntimeError(f"No parsed output (finish_reason={completion.choices[0].finish_reason})")
        return message.parsed

    def _track(self, u, header_cost: str | None) -> None:
        if header_cost is not None:
            cost, self.cost_source = float(header_cost), "litellm-header"
        elif self.model in MODEL_PRICES:
            price_in, price_out = MODEL_PRICES[self.model]
            cost, self.cost_source = (u.prompt_tokens * price_in + u.completion_tokens * price_out) / 1e6, "price-table"
        else:
            cost, self.cost_source = 0.0, "unpriced"
        self.usage = self.usage.add(
            Usage(llm_calls=1, input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens, cost_usd=cost)
        )


def make_llm_brain(settings) -> PromptBrain:
    """Provider switch used by the CLI, the eval harness and scripts/check_llm.py."""
    if settings.provider == "openai":
        return OpenAIBrain(model=settings.model, reasoning_effort=settings.reasoning_effort)
    return LLMBrain(model=settings.model, price_model=settings.price_model)
