"""Deterministic refund calculator. The LLM never computes money."""

from __future__ import annotations

from ..state import Claim, DisputeType


def compute_refund(dispute_type: DisputeType, claim: Claim | None, matching: list[dict], duplicates: list[list[dict]]) -> float:
    if dispute_type == DisputeType.DUPLICATE_CHARGE and duplicates:
        group = duplicates[0]
        return round(sum(t["amount"] for t in group[1:]), 2)  # keep one legitimate charge
    if not matching:
        return 0.0
    charged = matching[0]["amount"]
    if dispute_type == DisputeType.WRONG_AMOUNT:
        expected = claim.claimed_correct_amount if claim else None
        if expected is None or expected >= charged:
            return 0.0
        return round(charged - expected, 2)
    if dispute_type in (DisputeType.NOT_RECEIVED, DisputeType.UNAUTHORIZED, DisputeType.REFUND_NOT_PROCESSED):
        return round(charged, 2)
    return 0.0
