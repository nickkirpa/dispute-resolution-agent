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


class Settings(BaseModel):
    # env is read when Settings() is created (not at import), so load_dotenv() can run first
    provider: str = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_PROVIDER", "anthropic"))  # anthropic | openai
    model: str = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_MODEL", "claude-opus-5-5"))
    reasoning_effort: str | None = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_REASONING_EFFORT"))  # openai reasoning models only
    # pricing key when `model` is a proxy alias (e.g. LiteLLM); defaults to `model`
    price_model: str | None = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_PRICE_MODEL"))
    kb_dir: Path = ROOT / "kb" / "policies"
    kb_version: str = "latest"
    evidence_mode: str = Field(default_factory=lambda: os.getenv("DISPUTE_AGENT_EVIDENCE", "plan"))  # plan | agent
    coverage_fill: bool = True  # agent mode: code runs policy-critical checks the agent skipped (False = ablation)
    max_tool_calls: int = 6  # agent-mode budget per case (invalid and refused calls count too)
    max_steps: int = 12  # hard step budget for one case
    max_draft_attempts: int = 2  # self-check retries before escalation
    human_review_amount: float = 500.0  # refunds above this always go to a human
    min_confidence: float = 0.6  # below this, escalate instead of deciding
    filing_window_days: int = 120  # mirrors POL-GEN-02 in policy v1
    kb_top_k: int = 4
