"""Customer-reply support: facts computed by code, decision-specific guidance, and fact checks on the draft.

The writer (LLM or template) gets numbers it would otherwise have to calculate (day counts, the refund amount) and
the details the customer already gave, so it neither miscounts ("40 days" for a 30-day gap) nor re-asks for them.
The self-check then verifies the draft against the same computed facts.
"""

from __future__ import annotations

import re
from datetime import date

from .state import CaseState, Decision

APOS = "['\u2019]"  # LLMs often write the typographic apostrophe: couldn’t
NOT_FOUND = re.compile(rf"(couldn{APOS}?t find|could not find|unable to (find|locate)|didn{APOS}?t find|did not find|"
                       rf"no (matching|such) (charge|transaction|payment)|not able to (find|locate)|can{APOS}?t (find|locate))", re.I)
DAYS = re.compile(r"\b(\d{1,4})[- ]?days?\b", re.I)

# next step for the customer after a rejection, keyed by the clause that decided it (wording from the policy texts)
REJECT_NEXT_STEP = {  # second person: used verbatim by the template and as guidance for the LLM
    "POL-DUP-02": "please cancel the subscription directly with the merchant, as subscriptions can only be cancelled there",
    "POL-GEN-02": "we can no longer dispute this charge, but you can still contact the merchant directly",
    "POL-REF-02": "the merchant's credit is already on your account; you can see it on your statement",
    "POL-AMT-01": "if you have proof of the agreed price, please send it and we will review the case again",
}


def reply_facts(state: CaseState) -> dict:
    """Facts the reply may state, computed from evidence (not by the model)."""
    m = state.evidence_of("matching_transaction")
    facts: dict = {"transaction_found": bool(m), "today": state.as_of.isoformat()}
    if m:
        t = m[0].data
        dates = sorted(date.fromisoformat(str(d)) for d in t.get("all_dates", [t["txn_date"]]))
        facts.update(merchant=t["merchant"], charged_amount=f"{t['amount']:.2f}",
                     transaction_dates=[d.isoformat() for d in dates],
                     days_since_transaction=(state.as_of - dates[0]).days)
        if len(dates) >= 2:
            facts["days_between_charges"] = (dates[1] - dates[0]).days
    if state.decision == Decision.REFUND:
        facts["refund_amount"] = f"{state.refund_amount:.2f}"
    c = state.claim
    if c:
        facts["customer_already_gave"] = {k: v for k, v in {"merchant": c.merchant, "amount": c.amount,
                                                            "date": c.transaction_date}.items() if v is not None}
    return facts


def reply_guidance(state: CaseState) -> list[str]:
    """What this reply must do, given the decision and why it was made."""
    d, g = state.decision, []
    if d == Decision.REFUND:
        g.append(f"Say it is a provisional credit of {state.refund_amount:.2f} EUR while the dispute is finalised.")
    elif d == Decision.REQUEST_INFO:
        if "transaction_identification" in state.missing_evidence:
            gave = reply_facts(state).get("customer_already_gave", {})
            g.append(f"Say clearly that we could not find a charge matching what they described ({gave or 'no details'}) "
                     "on their account. Ask only for what would identify it: the exact transaction date, or a screenshot "
                     "of the statement line. Do not ask again for details they already gave.")
        if "merchant_contact" in state.missing_evidence:
            g.append("Ask them to contact the merchant first; if the merchant does not respond within 15 days, they "
                     "should reply with proof of contact (email, chat, message).")
        if "expected_amount" in state.missing_evidence:
            g.append("Ask for the amount they agreed to pay.")
    elif d == Decision.REJECT:
        step = next((REJECT_NEXT_STEP[c] for c in state.cited_clauses if c in REJECT_NEXT_STEP), None)
        g.append("Briefly acknowledge their frustration. Explain the reason using the computed facts (exact numbers).")
        if step:
            g.append(f"Give this next step: {step}.")
    elif d == Decision.ESCALATE:
        g.append("Say a dispute officer will review the case and contact them with the outcome. Do not promise an "
                 "outcome, an amount or a timeline.")
    g.append("Use only the numbers in the computed facts or the policy; never calculate days or amounts yourself.")
    return g


def fact_errors(state: CaseState) -> list[str]:
    """Deterministic checks of the draft against computed facts."""
    errors, draft, facts = [], state.response_draft, reply_facts(state)
    allowed = {facts.get("days_since_transaction"), facts.get("days_between_charges")}
    for clause in state.clauses:  # day counts the policy itself states (120, 90, 15, 3, ...)
        allowed |= {int(n) for n in DAYS.findall(clause.text)}
    allowed |= {int(v) for k, v in state.policy_params.items() if k.endswith("_days")}
    wrong = sorted({int(n) for n in DAYS.findall(draft)} - allowed)
    if wrong:
        errors.append(f"reply states {', '.join(map(str, wrong))} days; computed/policy values are "
                      f"{sorted(x for x in allowed if x is not None)}")
    if state.decision == Decision.REQUEST_INFO and "transaction_identification" in state.missing_evidence \
            and not NOT_FOUND.search(draft):
        errors.append("transaction not found: the reply must say we could not find the charge on the account")
    return errors
