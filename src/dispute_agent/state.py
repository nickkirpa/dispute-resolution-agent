"""Typed agent state. Every node reads a CaseState and returns a partial update."""

from __future__ import annotations

import operator
from datetime import date
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DisputeType(str, Enum):
    DUPLICATE_CHARGE = "duplicate_charge"
    NOT_RECEIVED = "not_received"
    UNAUTHORIZED = "unauthorized"
    WRONG_AMOUNT = "wrong_amount"
    REFUND_NOT_PROCESSED = "refund_not_processed"
    OTHER = "other"


class Decision(str, Enum):
    REFUND = "refund"
    REQUEST_INFO = "request_info"
    REJECT = "reject"
    ESCALATE = "escalate"


class Claim(BaseModel):
    """Facts the customer asserts, extracted from the narrative."""

    merchant: str | None = None
    amount: float | None = None
    currency: str = "EUR"
    transaction_date: str | None = Field(None, description="ISO date YYYY-MM-DD if stated")
    claimed_correct_amount: float | None = Field(None, description="For wrong-amount disputes: what they expected to pay")
    contacted_merchant: bool = False
    refund_promised: bool = False


class Evidence(BaseModel):
    source: Literal["ledger", "history", "kb"]
    kind: str  # e.g. matching_transaction, duplicate_transactions, customer_history
    summary: str
    data: dict = Field(default_factory=dict)


class PolicyClause(BaseModel):
    clause_id: str
    version: str
    title: str
    text: str
    applies_to: list[str] = Field(default_factory=list)
    score: float = 0.0


class Usage(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class CaseState(BaseModel):
    # input
    case_id: str
    customer_id: str
    narrative: str
    as_of: date = Field(default_factory=date.today, description="Date the case is processed; fixed in evals")

    # understanding
    claim: Claim | None = None
    dispute_type: DisputeType | None = None
    type_confidence: float = 0.0

    # evidence & policy (evidence uses an additive reducer so nodes can append)
    evidence: Annotated[list[Evidence], operator.add] = Field(default_factory=list)
    clauses: list[PolicyClause] = Field(default_factory=list)
    kb_version: str | None = None
    evidence_mode: str = ""  # plan | agent | plan_fallback
    tool_calls: list[dict] = Field(default_factory=list)  # audit trail of every tool call (args, reason, observation)
    coverage_fills: list[str] = Field(default_factory=list)  # checks code had to run because the agent skipped them
    fallback_error: str | None = None
    missing_evidence: list[str] = Field(default_factory=list)

    # outcome
    decision: Decision | None = None
    decision_confidence: float = 0.0
    refund_amount: float = 0.0
    rationale: str = ""
    cited_clauses: list[str] = Field(default_factory=list)
    response_draft: str = ""

    # control
    needs_human: bool = False
    human_reason: str = ""
    self_check_errors: list[str] = Field(default_factory=list)
    draft_attempts: int = 0
    steps: int = 0
    trace: Annotated[list[str], operator.add] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    def evidence_of(self, kind: str) -> list[Evidence]:
        return [e for e in self.evidence if e.kind == kind]


# Types the checkpointer may rebuild when it loads a saved case. Explicit allow-list: loading a checkpoint must not
# be able to instantiate arbitrary classes.
CHECKPOINT_TYPES = [("dispute_agent.state", name) for name in
                    ("CaseState", "Claim", "Evidence", "PolicyClause", "Usage", "Decision", "DisputeType")]
