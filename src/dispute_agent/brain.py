"""Brains: the parts of the agent that need language understanding or judgement.

The graph (graph.py) owns control flow, tools and guards. A Brain only:
  - extracts the claim from the narrative
  - classifies the dispute type
  - proposes a decision given evidence + policy clauses
  - drafts the customer reply

RuleBrain is a deterministic baseline (regex + policy v1 encoded by hand). It runs offline, powers the
tests, and is the floor every LLM configuration has to beat in the evals.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from .state import CaseState, Claim, Decision, DisputeType, Usage


class DecisionProposal(BaseModel):
    decision: Decision
    confidence: float = Field(description="0..1, how sure the decision is correct under the cited policy")
    rationale: str = Field(description="One or two sentences grounded in the evidence")
    cited_clauses: list[str] = Field(description="Policy clause ids (e.g. POL-DUP-01) that justify the decision")


class Classification(BaseModel):
    dispute_type: DisputeType
    confidence: float = Field(description="0..1")


class Brain(Protocol):
    name: str
    usage: Usage

    def extract_claim(self, narrative: str, as_of: date | None = None) -> Claim: ...
    def classify(self, narrative: str, claim: Claim) -> Classification: ...
    def decide(self, state: CaseState) -> DecisionProposal: ...
    def draft_response(self, state: CaseState) -> str: ...


# ---------------------------------------------------------------- rule baseline

MONEY = r"(\d+(?:[.,]\d{1,2})?)\s*(?:eur|euro|euros|€)"
DATE = r"(\d{4}-\d{2}-\d{2})"

TYPE_PATTERNS: list[tuple[DisputeType, str]] = [
    (DisputeType.DUPLICATE_CHARGE, r"\btwice\b|two times|double[- ]?charged|duplicate|charged again"),
    (DisputeType.REFUND_NOT_PROCESSED, r"refund.{0,40}(never|not|still|hasn't|has not|haven't)|(promised|approved|confirmed).{0,20}refund"),
    (DisputeType.UNAUTHORIZED, r"don'?t recogni[sz]e|do not recogni[sz]e|did ?n[o']t make|never made|not authori[sz]e|unauthori[sz]ed|fraud|stolen"),
    (DisputeType.WRONG_AMOUNT, r"instead of|overcharged|should have been|wrong amount|agreed (price|amount)"),
    (DisputeType.NOT_RECEIVED, r"never (arrived|delivered|received|came)|not (been )?delivered|did ?n[o']t arrive|never got"),
]

DECISION_TEXT = {
    Decision.REFUND: "We have issued a provisional credit of {amount:.2f} EUR while we finalise the dispute.",
    Decision.REQUEST_INFO: "To continue, we need a bit more information from you: {missing}.",
    Decision.REJECT: "After reviewing your account, we are unable to accept this dispute.",
    Decision.ESCALATE: "Your case has been passed to a dispute specialist, who will contact you within 2 business days.",
}

MISSING_TEXT = {
    "transaction_identification": "the transaction date, merchant name and amount",
    "merchant_contact": "confirmation that you contacted the merchant (and their reply, if any)",
    "expected_amount": "the amount you agreed to pay",
}


class RuleBrain:
    name = "rules"

    def __init__(self, known_merchants: list[str] | None = None):
        self.known_merchants = sorted(known_merchants or [], key=len, reverse=True)
        self.usage = Usage()

    def extract_claim(self, narrative: str, as_of: date | None = None) -> Claim:
        text = narrative.lower()
        merchant = next((m for m in self.known_merchants if m.lower() in text), None)
        amounts = [float(a.replace(",", ".")) for a in re.findall(MONEY, text)]
        correct = None
        m = re.search(r"(?:instead of|should have been|agreed (?:price|amount) (?:was|of))\s*" + MONEY, text)
        if m:
            correct = float(m.group(1).replace(",", "."))
        date = re.search(DATE, text)
        return Claim(
            merchant=merchant,
            amount=amounts[0] if amounts else None,
            transaction_date=date.group(1) if date else None,
            claimed_correct_amount=correct,
            contacted_merchant=bool(re.search(r"(contacted|emailed|called|wrote to|reached out to|messaged)\s+(the\s+)?(merchant|seller|shop|store|them|support|\w+ support)", text)),
            refund_promised=bool(re.search(r"(promised|approved|confirmed|agreed to)\s+(me\s+)?(a\s+|the\s+)?refund|would refund", text)),
        )

    def classify(self, narrative: str, claim: Claim) -> Classification:
        text = narrative.lower()
        for dtype, pattern in TYPE_PATTERNS:
            if re.search(pattern, text):
                return Classification(dispute_type=dtype, confidence=0.9)
        return Classification(dispute_type=DisputeType.OTHER, confidence=0.3)

    def decide(self, state: CaseState) -> DecisionProposal:
        t = state.dispute_type
        missing_clause = {
            "transaction_identification": "POL-GEN-03",
            "merchant_contact": "POL-NR-02",
            "expected_amount": "POL-AMT-01",
        }
        if state.missing_evidence:
            cites = [missing_clause[m] for m in state.missing_evidence if m in missing_clause]
            return DecisionProposal(decision=Decision.REQUEST_INFO, confidence=0.9, rationale="Required evidence is missing.", cited_clauses=cites)

        window = state.evidence_of("filing_window")
        limit = state.policy_params["filing_window_days"]
        if window and window[0].data.get("days_since_transaction", 0) > limit:
            return DecisionProposal(decision=Decision.REJECT, confidence=0.9, rationale=f"Filed after the {limit:.0f}-day window.",
                                    cited_clauses=["POL-GEN-02"])

        history = state.evidence_of("customer_history")
        hist = history[0].data if history else {}

        if t == DisputeType.DUPLICATE_CHARGE:
            if state.evidence_of("duplicate_transactions"):
                return DecisionProposal(decision=Decision.REFUND, confidence=0.95, rationale="Identical charges posted within 3 days.", cited_clauses=["POL-DUP-01"])
            return DecisionProposal(decision=Decision.REJECT, confidence=0.8, rationale="Charges are more than 3 days apart: separate payments.", cited_clauses=["POL-DUP-02"])
        if t == DisputeType.NOT_RECEIVED:
            return DecisionProposal(decision=Decision.REFUND, confidence=0.85, rationale="Not delivered and merchant already contacted.", cited_clauses=["POL-NR-01"])
        if t == DisputeType.UNAUTHORIZED:
            if hist.get("unauthorized_disputes_last_12m", 0) >= 2:
                return DecisionProposal(decision=Decision.ESCALATE, confidence=0.9, rationale="Repeated unauthorized claims in 12 months.", cited_clauses=["POL-UNA-02"])
            return DecisionProposal(decision=Decision.REFUND, confidence=0.85, rationale="Customer reports an unauthorized transaction.", cited_clauses=["POL-UNA-01"])
        if t == DisputeType.WRONG_AMOUNT:
            charged = state.evidence_of("matching_transaction")[0].data["amount"]
            expected = state.claim.claimed_correct_amount if state.claim else None
            if expected is not None and expected < charged:
                return DecisionProposal(decision=Decision.REFUND, confidence=0.9, rationale="Charged more than the agreed amount.", cited_clauses=["POL-AMT-01"])
            return DecisionProposal(decision=Decision.REJECT, confidence=0.8, rationale="Charged amount does not exceed the agreed amount.", cited_clauses=["POL-AMT-01"])
        if t == DisputeType.REFUND_NOT_PROCESSED:
            if state.evidence_of("merchant_refunds"):
                return DecisionProposal(decision=Decision.REJECT, confidence=0.9, rationale="Merchant credit already posted.", cited_clauses=["POL-REF-02"])
            return DecisionProposal(decision=Decision.REFUND, confidence=0.85, rationale="Promised refund never credited.", cited_clauses=["POL-REF-01"])
        return DecisionProposal(decision=Decision.ESCALATE, confidence=0.4, rationale="Dispute type not covered by automated policy.", cited_clauses=["POL-GEN-04"])

    def draft_response(self, state: CaseState) -> str:
        missing = ", ".join(MISSING_TEXT.get(m, m) for m in state.missing_evidence) or "details of the transaction"
        body = DECISION_TEXT[state.decision].format(amount=state.refund_amount, missing=missing)
        cites = ", ".join(state.cited_clauses)
        return f"Hello,\n\nThank you for contacting Northwind Bank about your dispute. {body}\n\n(Policy reference: {cites})\n\nKind regards,\nNorthwind Disputes Team"
