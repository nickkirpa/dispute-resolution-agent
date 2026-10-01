"""Runtime settings. Thresholds live in code, not in prompts, so they are enforced and testable."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

# USD per 1M tokens (input, output). Source: Anthropic pricing, cached 2026-09.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _parse_effort(spec: str) -> dict[str, str]:
    """'decide=medium, action=low' -> {'decide': 'medium', 'action': 'low'}"""
    return {k.strip(): v.strip() for k, _, v in (item.partition("=") for item in spec.split(",")) if k.strip() and v.strip()}


class Settings(BaseModel):
    # env is read when Settings() is created (not at import), so load_dotenv() can run first
    provider: str = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_PROVIDER", "anthropic"))  # anthropic | openai
    model: str = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_MODEL", "claude-opus-5-5"))
    reasoning_effort: str | None = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_REASONING_EFFORT"))  # openai reasoning models only
    # per-step override, e.g. "decide=medium,action=medium" (steps: extract, classify, decide, draft, action)
    effort_by_step: dict[str, str] = Field(default_factory=lambda: _parse_effort(os.getenv("DISPUTE_AGENT_EFFORT", "")))
    # pricing key when `model` is a proxy alias (e.g. LiteLLM); defaults to `model`
    price_model: str | None = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_PRICE_MODEL"))
    kb_dir: Path = ROOT / "kb" / "policies"
    kb_version: str = "latest"
    evidence_mode: str = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_EVIDENCE", "plan"))  # plan | agent
    coverage_fill: bool = True  # agent mode: code runs policy-critical checks the agent skipped (False = ablation)
    max_tool_calls: int = 6  # agent-mode budget per case (invalid and refused calls count too)
    max_steps: int = 12  # hard step budget for one case
    max_draft_attempts: int = 2  # self-check retries before escalation
    min_confidence: float = 0.6  # below this, escalate instead of deciding
    kb_top_k: int = 4
