"""The state machine.

    intake → classify → gather_evidence → policy_check → decide → draft_response → self_check → END
                                                            ↘ human_review ↗            ↺ (retry draft)

Control flow, tool calls and every money/threshold rule live here in code. The Brain is only asked for
language understanding and judgement, and its proposals pass through guards before they take effect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .brain import Brain
from .config import Settings
from .llm_brain import LLMRefusal
from .state import CaseState, Decision, DisputeType, Evidence
from .tools import KnowledgeBase, Ledger, compute_refund


@dataclass
class Deps:
    brain: Brain
    ledger: Ledger
    kb: KnowledgeBase
    settings: Settings = field(default_factory=Settings)
    human_in_loop: bool = False  # True: pause at human_review (interrupt). False: auto-escalate (evals, batch).


def normalize_claim_date(iso: str | None, as_of: date) -> str | None:
    """Customers rarely state the year. If the extracted date is in the future or more than a year old,
    move it to the most recent past occurrence of that month/day. Unparseable dates are dropped."""
    if not iso:
        return None
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return None
    if d > as_of or (as_of - d).days > 366:
        for year in (as_of.year, as_of.year - 1):
            try:
                cand = d.replace(year=year)
            except ValueError:  # 29 Feb
                continue
            if cand <= as_of:
                return cand.isoformat()
    return d.isoformat()


def _step(name: str, state: CaseState, **update) -> dict:
    return {"steps": state.steps + 1, "trace": [name], **update}


def _escalate(name: str, state: CaseState, reason: str, deps: Deps) -> dict:
    return _step(name, state, needs_human=True, human_reason=reason, usage=deps.brain.usage.model_copy())


def build_graph(deps: Deps, checkpointer=None):
    s = deps.settings

    # ------------------------------------------------------------------ nodes
    def intake(state: CaseState) -> dict:
        try:
            claim = deps.brain.extract_claim(state.narrative, state.as_of)
        except LLMRefusal as e:
            return _escalate("intake", state, f"model refusal: {e}", deps)
        claim = claim.model_copy(update={"transaction_date": normalize_claim_date(claim.transaction_date, state.as_of)})
        return _step("intake", state, claim=claim, usage=deps.brain.usage.model_copy())

    def classify(state: CaseState) -> dict:
        try:
            c = deps.brain.classify(state.narrative, state.claim)
        except LLMRefusal as e:
            return _escalate("classify", state, f"model refusal: {e}", deps)
        return _step("classify", state, dispute_type=c.dispute_type, type_confidence=c.confidence, usage=deps.brain.usage.model_copy())

    def gather_evidence(state: CaseState) -> dict:
        """Deterministic tool plan per dispute type (Day 3-5 TODO: bounded LLM tool-use loop, this stays as fallback)."""
        claim, t = state.claim, state.dispute_type
        around = date.fromisoformat(claim.transaction_date) if claim and claim.transaction_date else None
        evidence: list[Evidence] = []

        # widen the search progressively: merchant+amount+date -> merchant+amount -> merchant -> amount
        attempts = [
            dict(merchant=claim.merchant, amount=claim.amount, around=around),
            dict(merchant=claim.merchant, amount=claim.amount),
            dict(merchant=claim.merchant),
            dict(amount=claim.amount),
        ]
        matching: list[dict] = []
        for kw in attempts:
            if not any(v is not None for v in kw.values()):
                continue
            matching = deps.ledger.find_transactions(state.customer_id, **kw)
            if matching:
                break
        if matching:
            first = matching[0]
            dates = ", ".join(str(m["txn_date"]) for m in matching)
            evidence.append(Evidence(source="ledger", kind="matching_transaction",
                                     summary=f"{len(matching)} matching charge(s) of {first['amount']} {first['currency']} at {first['merchant']} on: {dates}",
                                     data=first | {"n_matches": len(matching), "all_dates": [str(m["txn_date"]) for m in matching]}))
            evidence.append(Evidence(source="ledger", kind="filing_window",
                                     summary=f"{(state.as_of - first['txn_date']).days} days since transaction",
                                     data={"days_since_transaction": (state.as_of - first["txn_date"]).days}))

        if t == DisputeType.DUPLICATE_CHARGE and matching:
            dups = deps.ledger.find_duplicates(state.customer_id, matching[0]["merchant"], matching[0]["amount"])
            if dups:
                evidence.append(Evidence(source="ledger", kind="duplicate_transactions",
                                         summary=f"{len(dups[0])} identical charges within 3 days", data={"groups": dups}))
            else:  # negative evidence: say explicitly what the tool did NOT find, so the model can't assume it
                evidence.append(Evidence(source="ledger", kind="no_duplicates",
                                         summary="No identical charges posted within 3 days of each other (duplicate search returned nothing)",
                                         data={"window_days": 3}))
        if t == DisputeType.REFUND_NOT_PROCESSED and matching:
            refunds = deps.ledger.refunds_for(state.customer_id, matching[0]["merchant"])
            if refunds:
                evidence.append(Evidence(source="ledger", kind="merchant_refunds",
                                         summary=f"{len(refunds)} credit(s) from merchant already posted", data={"refunds": refunds}))
            else:
                evidence.append(Evidence(source="ledger", kind="no_merchant_refunds",
                                         summary="No credit from this merchant has been posted to the account", data={}))

        history = deps.ledger.customer_history(state.customer_id, state.as_of)
        evidence.append(Evidence(source="history", kind="customer_history", summary=str(history), data=history))

        # policy retrieval: type-specific clauses by BM25 + all general clauses (small KB; revisit on Day 6-7)
        specific = [c for c in deps.kb.search(f"{t.value} {state.narrative}", k=10, dispute_type=t.value) if t.value in c.applies_to][: s.kb_top_k]
        general = [c for c in deps.kb.clauses if "all" in c.applies_to]
        return _step("gather_evidence", state, evidence=evidence, clauses=specific + general, kb_version=deps.kb.version)

    def policy_check(state: CaseState) -> dict:
        """Which evidence does policy require that we don't have? Deterministic."""
        missing = []
        if not state.evidence_of("matching_transaction"):
            missing.append("transaction_identification")
        elif state.dispute_type == DisputeType.NOT_RECEIVED and not state.claim.contacted_merchant:
            missing.append("merchant_contact")
        elif state.dispute_type == DisputeType.WRONG_AMOUNT and state.claim.claimed_correct_amount is None:
            missing.append("expected_amount")
        return _step("policy_check", state, missing_evidence=missing)

    def decide(state: CaseState) -> dict:
        try:
            p = deps.brain.decide(state)
        except LLMRefusal as e:
            return _escalate("decide", state, f"model refusal: {e}", deps)
        decision, needs_human, reason = p.decision, False, ""

        # guard 1: no refund (or reject) without the evidence policy requires
        if state.missing_evidence and decision in (Decision.REFUND, Decision.REJECT):
            decision, reason = Decision.REQUEST_INFO, "guard: required evidence missing"

        # guard 2: money is computed by code, never by the model
        refund = 0.0
        if decision == Decision.REFUND:
            matching = [e.data for e in state.evidence_of("matching_transaction")]
            dups = state.evidence_of("duplicate_transactions")
            refund = compute_refund(state.dispute_type, state.claim, matching, dups[0].data["groups"] if dups else [])
            if refund <= 0:
                needs_human, reason = True, "guard: refund proposed but amount computes to 0"

        # guard 2b: a refund must not contradict hard evidence -> a human looks at the conflict (no silent override)
        if decision == Decision.REFUND:
            window = state.evidence_of("filing_window")
            conflicts = []
            if state.dispute_type == DisputeType.DUPLICATE_CHARGE and not state.evidence_of("duplicate_transactions"):
                conflicts.append("no duplicate charges found")
            if state.dispute_type == DisputeType.REFUND_NOT_PROCESSED and state.evidence_of("merchant_refunds"):
                conflicts.append("merchant credit already posted")
            if window and window[0].data["days_since_transaction"] > s.filing_window_days:
                conflicts.append(f"filed after {s.filing_window_days} days")
            if conflicts:
                needs_human, reason = True, f"guard: refund contradicts evidence ({'; '.join(conflicts)})"

        # guard 3: thresholds
        if decision == Decision.REFUND and refund > s.human_review_amount:
            needs_human, reason = True, f"guard: refund {refund:.2f} > {s.human_review_amount:.0f} EUR (POL-GEN-04)"
        if min(p.confidence, state.type_confidence) < s.min_confidence:
            needs_human, reason = True, f"guard: low confidence ({min(p.confidence, state.type_confidence):.2f})"
        if decision == Decision.ESCALATE:
            needs_human, reason = True, reason or p.rationale

        return _step("decide", state, decision=decision, decision_confidence=p.confidence, refund_amount=refund,
                     rationale=p.rationale, cited_clauses=p.cited_clauses, needs_human=needs_human,
                     human_reason=reason, usage=deps.brain.usage.model_copy())

    def human_review(state: CaseState) -> dict:
        if deps.human_in_loop:
            verdict = interrupt({
                "case_id": state.case_id, "reason": state.human_reason, "proposed_decision": state.decision,
                "refund_amount": state.refund_amount, "rationale": state.rationale, "cited_clauses": state.cited_clauses,
            })
            decision = Decision(verdict.get("decision", Decision.ESCALATE))
            return _step("human_review", state, decision=decision,
                         refund_amount=float(verdict.get("refund_amount", state.refund_amount if decision == Decision.REFUND else 0.0)),
                         rationale=f"{state.rationale} | human: {verdict.get('note', '')}", needs_human=False,
                         cited_clauses=state.cited_clauses or ["POL-GEN-04"])
        cites = state.cited_clauses if "POL-GEN-04" in state.cited_clauses else [*state.cited_clauses, "POL-GEN-04"]
        clauses = state.clauses
        if not any(c.clause_id == "POL-GEN-04" for c in clauses) and deps.kb.get("POL-GEN-04"):
            clauses = [*clauses, deps.kb.get("POL-GEN-04")]
        return _step("human_review", state, decision=Decision.ESCALATE, refund_amount=0.0, cited_clauses=cites,
                     clauses=clauses, kb_version=deps.kb.version,
                     rationale=f"Escalated to a dispute officer: {state.human_reason}")

    def draft_response(state: CaseState) -> dict:
        try:
            text = deps.brain.draft_response(state)
        except LLMRefusal as e:
            return _escalate("draft_response", state, f"model refusal: {e}", deps)
        return _step("draft_response", state, response_draft=text, draft_attempts=state.draft_attempts + 1,
                     usage=deps.brain.usage.model_copy())

    def self_check(state: CaseState) -> dict:
        """Deterministic verification of the draft against the decision and retrieved policy."""
        errors = []
        retrieved = {c.clause_id for c in state.clauses}
        unknown = [c for c in state.cited_clauses if c not in retrieved]
        if unknown:
            errors.append(f"cited clauses not in retrieved policy: {unknown}")
        if not state.cited_clauses:
            errors.append("no policy clause cited")
        if state.decision == Decision.REFUND and f"{state.refund_amount:.2f}" not in state.response_draft:
            errors.append("draft does not state the computed refund amount")
        mentioned = set(re.findall(r"POL-[A-Z]+-\d+", state.response_draft))
        if mentioned - retrieved:
            errors.append(f"draft mentions unknown clauses: {sorted(mentioned - retrieved)}")
        return _step("self_check", state, self_check_errors=errors)

    # ---------------------------------------------------------------- routing
    def over_budget(state: CaseState) -> bool:
        return state.steps >= s.max_steps

    def after_intake(state: CaseState) -> str:
        return "human_review" if state.needs_human else "classify"

    def after_classify(state: CaseState) -> str:
        return "human_review" if state.needs_human or over_budget(state) else "gather_evidence"

    def after_decide(state: CaseState) -> str:
        return "human_review" if state.needs_human or over_budget(state) else "draft_response"

    def after_self_check(state: CaseState) -> str:
        if not state.self_check_errors:
            return END
        if state.draft_attempts < s.max_draft_attempts and not over_budget(state):
            return "draft_response"
        return END  # errors stay on the state and are reported by the eval

    g = StateGraph(CaseState)
    for name, fn in [("intake", intake), ("classify", classify), ("gather_evidence", gather_evidence),
                     ("policy_check", policy_check), ("decide", decide), ("human_review", human_review),
                     ("draft_response", draft_response), ("self_check", self_check)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", after_intake, ["classify", "human_review"])
    g.add_conditional_edges("classify", after_classify, ["gather_evidence", "human_review"])
    g.add_edge("gather_evidence", "policy_check")
    g.add_edge("policy_check", "decide")
    g.add_conditional_edges("decide", after_decide, ["draft_response", "human_review"])
    g.add_edge("human_review", "draft_response")
    g.add_edge("draft_response", "self_check")
    g.add_conditional_edges("self_check", after_self_check, ["draft_response", END])
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def run_case(app, case: CaseState, thread_id: str | None = None) -> dict:
    """Invoke the graph for one case. Returns the raw result (may contain '__interrupt__')."""
    config = {"configurable": {"thread_id": thread_id or case.case_id}}
    return app.invoke(case, config=config)

