"""Evidence gathering: one tool executor, two drivers.

- plan mode:  a deterministic, scripted sequence of tool calls (the baseline and the fallback)
- agent mode: the LLM chooses the next tool call, one typed action at a time, under a code-enforced budget

Both drivers go through the same `ToolBox.run`, so the two modes are compared on equal terms. The model never
touches the ledger directly: it proposes an action, code validates and executes it, and the model sees a short
observation. After the loop, code checks coverage (did the agent run the checks policy depends on?) and fills
gaps itself, recording each fill so the eval can measure how often the agent missed something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .state import CaseState, DisputeType, Evidence, PolicyClause
from .tools import KnowledgeBase, Ledger

ToolName = Literal["find_transactions", "find_duplicates", "merchant_refunds", "customer_history", "search_policy", "finish"]

TOOL_SPECS = """\
- find_transactions(merchant?, amount?, date?, window_days?): the customer's charges. Filters are optional and combine;
  `date` (YYYY-MM-DD) searches +/- window_days (default 45). Merchant match is a case-insensitive substring.
- find_duplicates(merchant, amount): groups of identical charges posted within 3 days of each other.
- merchant_refunds(merchant): credits (refunds) from that merchant already posted to the account.
- customer_history(): account age and prior disputes in the last 12 months.
- search_policy(query): policy clauses relevant to the query (filtered to this dispute type + general clauses).
- finish(): stop when you have the evidence a dispute officer would need for this dispute type."""


class AgentAction(BaseModel):
    """One step of the agent. Unused arguments must be null."""

    reason: str = Field(description="One short sentence: why this action now")
    tool: ToolName
    merchant: str | None = None
    amount: float | None = None
    date: str | None = Field(None, description="YYYY-MM-DD")
    window_days: int | None = None
    query: str | None = None


@dataclass
class ToolResult:
    observation: str  # what the model sees
    ok: bool = True


@dataclass
class ToolBox:
    """Executes validated tool calls for one case and accumulates the evidence they produce."""

    ledger: Ledger
    kb: KnowledgeBase
    state: CaseState
    calls: list[dict] = field(default_factory=list)
    searches: list[list[dict]] = field(default_factory=list)  # results of each find_transactions call
    dup_checked: bool = False
    refunds_checked: bool = False
    history_checked: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    clauses: dict[str, PolicyClause] = field(default_factory=dict)

    def run(self, a: AgentAction) -> ToolResult:
        sig = a.model_dump(exclude={"reason"})
        if any(c["args"] == sig for c in self.calls):
            result = ToolResult("Refused: identical call already made; use its result or try different arguments.", ok=False)
        else:
            try:
                result = getattr(self, f"_{a.tool}")(a)
            except (ValueError, TypeError) as e:
                result = ToolResult(f"Error: {e}", ok=False)
        self.calls.append({"tool": a.tool, "args": sig, "reason": a.reason, "ok": result.ok, "observation": result.observation[:300]})
        return result

    # ---- tools
    def _find_transactions(self, a: AgentAction) -> ToolResult:
        around = date.fromisoformat(a.date) if a.date else None
        if not any([a.merchant, a.amount is not None, around]):
            raise ValueError("give at least one of merchant, amount, date")
        rows = self.ledger.find_transactions(self.state.customer_id, merchant=a.merchant, amount=a.amount, around=around,
                                             window_days=a.window_days or 45)
        self.searches.append(rows)
        if not rows:
            return ToolResult("0 charges match.")
        listing = "; ".join(f"{r['txn_id']}: {r['amount']:.2f} {r['currency']} {r['merchant']} on {r['txn_date']} ({r['channel']})" for r in rows[:8])
        return ToolResult(f"{len(rows)} charge(s): {listing}")

    def _find_duplicates(self, a: AgentAction) -> ToolResult:
        if not a.merchant or a.amount is None:
            raise ValueError("find_duplicates needs merchant and amount")
        self.dup_checked = True
        groups = self.ledger.find_duplicates(self.state.customer_id, a.merchant, a.amount)
        if groups:
            self.evidence.append(Evidence(source="ledger", kind="duplicate_transactions",
                                          summary=f"{len(groups[0])} identical charges within 3 days", data={"groups": groups}))
            return ToolResult(f"{len(groups[0])} identical charges within 3 days: " + ", ".join(str(t["txn_date"]) for t in groups[0]))
        self.evidence.append(Evidence(source="ledger", kind="no_duplicates",
                                      summary="No identical charges posted within 3 days of each other (duplicate search returned nothing)",
                                      data={"window_days": 3}))
        return ToolResult("No identical charges within 3 days of each other.")

    def _merchant_refunds(self, a: AgentAction) -> ToolResult:
        if not a.merchant:
            raise ValueError("merchant_refunds needs merchant")
        self.refunds_checked = True
        refunds = self.ledger.refunds_for(self.state.customer_id, a.merchant)
        if refunds:
            self.evidence.append(Evidence(source="ledger", kind="merchant_refunds",
                                          summary=f"{len(refunds)} credit(s) from merchant already posted", data={"refunds": refunds}))
            return ToolResult("Credits posted: " + "; ".join(f"{-r['amount']:.2f} on {r['txn_date']}" for r in refunds))
        self.evidence.append(Evidence(source="ledger", kind="no_merchant_refunds",
                                      summary="No credit from this merchant has been posted to the account", data={}))
        return ToolResult("No credits from this merchant.")

    def _customer_history(self, a: AgentAction) -> ToolResult:
        self.history_checked = True
        h = self.ledger.customer_history(self.state.customer_id, self.state.as_of)
        self.evidence.append(Evidence(source="history", kind="customer_history", summary=str(h), data=h))
        return ToolResult(str(h))

    def _search_policy(self, a: AgentAction) -> ToolResult:
        if not a.query:
            raise ValueError("search_policy needs query")
        t = self.state.dispute_type.value
        hits = [c for c in self.kb.search(a.query, k=10, dispute_type=t) if t in c.applies_to][:4]
        for c in hits:
            self.clauses.setdefault(c.clause_id, c)
        return ToolResult("; ".join(f"{c.clause_id}: {c.title}" for c in hits) or "No matching clauses.")

    def _finish(self, a: AgentAction) -> ToolResult:
        return ToolResult("finished")

    # ---- finalisation
    def matching(self) -> list[dict]:
        """The transaction the dispute is about: results of the most recent search that found something."""
        hits = [s for s in self.searches if s]
        return hits[-1] if hits else []


def plan_actions(state: CaseState) -> list[AgentAction]:
    """The scripted baseline: progressively widened search (merchant+amount+date -> ... -> amount)."""
    c = state.claim
    attempts = [dict(merchant=c.merchant, amount=c.amount, date=c.transaction_date), dict(merchant=c.merchant, amount=c.amount),
                dict(merchant=c.merchant), dict(amount=c.amount)]
    return [AgentAction(reason="plan: transaction search", tool="find_transactions", **kw)
            for kw in attempts if any(v is not None for v in kw.values())]


def run_plan(box: ToolBox) -> None:
    for action in plan_actions(box.state):
        box.run(action)
        if box.searches and box.searches[-1]:
            break  # stop widening at the first hit


def fill_coverage(box: ToolBox, ran_policy_search: bool) -> list[str]:
    """Run the checks policy depends on if the driver skipped them. Returns what had to be filled."""
    fills, s, m = [], box.state, box.matching()
    if not box.searches:
        fills.append("transaction_search")
        run_plan(box)
        m = box.matching()
    if s.dispute_type == DisputeType.DUPLICATE_CHARGE and m and not box.dup_checked:
        fills.append("find_duplicates")
        box.run(AgentAction(reason="coverage", tool="find_duplicates", merchant=m[0]["merchant"], amount=m[0]["amount"]))
    if s.dispute_type == DisputeType.REFUND_NOT_PROCESSED and m and not box.refunds_checked:
        fills.append("merchant_refunds")
        box.run(AgentAction(reason="coverage", tool="merchant_refunds", merchant=m[0]["merchant"]))
    if not box.history_checked:
        fills.append("customer_history")
        box.run(AgentAction(reason="coverage", tool="customer_history"))
    if not ran_policy_search:
        fills.append("search_policy")
        box.run(AgentAction(reason="coverage", tool="search_policy", query=f"{s.dispute_type.value} {s.narrative}"))
    return fills


def finalize(box: ToolBox) -> tuple[list[Evidence], list[PolicyClause]]:
    s, m = box.state, box.matching()
    evidence: list[Evidence] = []
    if m:
        first = m[0]
        dates = ", ".join(str(r["txn_date"]) for r in m)
        days = (s.as_of - first["txn_date"]).days
        evidence.append(Evidence(source="ledger", kind="matching_transaction",
                                 summary=f"{len(m)} matching charge(s) of {first['amount']} {first['currency']} at {first['merchant']} on: {dates}",
                                 data=first | {"n_matches": len(m), "all_dates": [str(r["txn_date"]) for r in m]}))
        evidence.append(Evidence(source="ledger", kind="filing_window", summary=f"{days} days since transaction",
                                 data={"days_since_transaction": days}))
    general = [c for c in box.kb.clauses if "all" in c.applies_to]  # guards and escalation always need these
    return evidence + box.evidence, list(box.clauses.values()) + [c for c in general if c.clause_id not in box.clauses]


AGENT_PROMPT = """You are gathering evidence for a card dispute. Choose the NEXT tool call, or finish.

Dispute type: {dtype}
Customer's claim (extracted): {claim}
Complaint: {narrative}
Today: {today}

Tools:
{tools}

Calls so far (tool -> observation):
{history}

Budget: {left} call(s) left. Gather what a dispute officer needs to decide THIS dispute type under policy, then finish.
Do not repeat a call. If a search finds nothing, try a wider search once (e.g. drop the date or the amount)."""


def run_agent(box: ToolBox, brain, budget: int) -> tuple[bool, str | None]:
    """LLM-driven loop. Returns (ran_policy_search, error). On error the caller falls back to the plan."""
    s = box.state
    ran_policy = False
    for step in range(budget):
        history = "\n".join(f"{i + 1}. {c['tool']}({ {k: v for k, v in c['args'].items() if v is not None and k != 'tool'} }) -> {c['observation']}"
                            for i, c in enumerate(box.calls)) or "none yet"
        prompt = AGENT_PROMPT.format(dtype=s.dispute_type.value, claim=s.claim.model_dump_json(), narrative=s.narrative,
                                     today=s.as_of.isoformat(), tools=TOOL_SPECS, history=history, left=budget - step)
        try:
            action = brain.next_action(prompt)
        except Exception as e:  # refusal, parse error, provider error -> deterministic fallback
            return ran_policy, f"{type(e).__name__}: {e}"
        if action.tool == "finish":
            box.calls.append({"tool": "finish", "args": {}, "reason": action.reason, "ok": True, "observation": "finished"})
            break
        result = box.run(action)
        ran_policy |= action.tool == "search_policy" and result.ok
    return ran_policy, None
