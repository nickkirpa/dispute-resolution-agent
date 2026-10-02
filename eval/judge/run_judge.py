"""LLM-as-judge for reply quality, validated against human labels.

The judge answers the same 5 questions as the human labeller (scripts/label_replies.py). We then measure:
  1. judge vs human     agreement + Cohen's kappa per question (the validation)
  2. human vs defects   do injected defects really look broken to a person? (sanity check of the sample)
  3. judge vs defects   which defect kinds the judge catches

    uv run python eval/judge/run_judge.py                          # judge = configured model (.env)
    uv run python eval/judge/run_judge.py --model openai/gpt-5.5   # a stronger judge
    uv run python eval/judge/run_judge.py --prompt v2 --votes 3    # v2 context, majority of 3 runs
    uv run python eval/judge/run_judge.py --prompt v2 --votes 3 --report eval/results/<run>.json
        # screening mode: judge every reply of an eval run (no human labels), e.g. to compare reply prompts

Prompt versions:
  v1  case facts + reply only (the first version)
  v2  + today's date, the policy version's parameters and clause texts, and ledger facts computed by code (days since
      each transaction, days between charges): the context a human reviewer has or can look up. v2 was written after
      seeing v1's disagreements on these same 40 replies, so its scores are optimistic until checked on fresh labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from datetime import date  # noqa: E402

from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.tools import KnowledgeBase  # noqa: E402
from dispute_agent.llm_brain import make_llm_brain  # noqa: E402

QUESTIONS = ["facts", "decision", "promises", "clarity", "send"]


class Verdict(BaseModel):
    facts: bool = Field(description="Merchant, amount and dates in the reply match the case; nothing invented")
    decision: bool = Field(description="Reply matches the decision: refund states the amount; reject explains why; "
                                       "request_info says clearly what is needed; escalate says a person will review")
    promises: bool = Field(description="No invented promises: no deadlines, guarantees or amounts beyond the decision/policy")
    clarity: bool = Field(description="Clear and polite; the customer understands what happens next")
    send: bool = Field(description="OK to send to a real customer as is")
    reason: str = Field(description="One sentence: the main problem, or 'fine'")


PROMPT = """You review a bank's reply to a card-dispute customer before it is sent. Be strict about facts and promises.

Case facts (the truth):
- Decision: {decision}{refund}
- Case transactions: {txns}
- Prior disputes on the account: {prior}
- Policy clauses behind the decision: {clauses}

Customer complaint:
{complaint}

Reply to review:
{reply}

Answer each question true/false. A reply that contradicts the decision, states a wrong amount or merchant, or invents a
promise is not OK to send."""

PROMPT_V2 = """You review a bank's reply to a card-dispute customer before it is sent. Be strict about facts and promises.

Today is {today}. Policy version {version}; its rule values: {params}.

Case facts (the truth):
- Decision: {decision}{refund}
- Case transactions: {txns}
- Computed from the ledger: {derived}
- Prior disputes on the account: {prior}

Policy clauses (exact text; anything the reply says that these clauses support is NOT invented):
{clause_text}

Customer complaint:
{complaint}

Reply to review:
{reply}

Answer each question true/false.
- facts: every merchant, amount, date and number of days in the reply matches the case facts and computed values.
- clarity: the customer understands what happens next AND the tone is polite. Rude, condescending or dismissive phrasing
  (e.g. implying the customer should not contact the bank) makes it false.
A reply that contradicts the decision, states a wrong amount, merchant or number of days, or invents a promise is not
OK to send."""


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    ap.add_argument("--votes", type=int, default=1, help="run the judge N times, majority vote per question")
    ap.add_argument("--report", type=Path, default=None, help="screening mode: judge all replies of this eval report")
    args = ap.parse_args()
    if args.report:
        return screen(args)
    items = [json.loads(l) for l in (ROOT / "eval" / "judge" / "replies.jsonl").read_text().splitlines() if l.strip()]
    labels_path = ROOT / "eval" / "judge" / "labels.jsonl"
    labels = {json.loads(l)["id"]: json.loads(l) for l in labels_path.read_text().splitlines() if l.strip()} if labels_path.exists() else {}
    items = [it for it in items if it["id"] in labels]
    if not items:
        raise SystemExit("no human labels yet: run scripts/label_replies.py first")
    settings = Settings(**({"model": args.model} if args.model else {}))
    prompt_for, judge = make_judge(args, settings)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(judge, items))
    report_rows(args, settings, items, labels, results)


def make_judge(args, settings):
    gold = {json.loads(l)["case_id"]: json.loads(l) for l in (ROOT / "eval" / "golden_v1.jsonl").read_text().splitlines() if l.strip()}
    kb = KnowledgeBase(settings.kb_dir, "v1")  # golden_v1 replies were produced under policy v1

    def prompt_for(it: dict) -> str:
        f = it["facts"]
        common = dict(decision=f["decision"], refund=f" (refund {f['refund_amount']:.2f} EUR)" if f["decision"] == "refund" else "",
                      txns="; ".join(f"{t['merchant']} {t['amount']:.2f} EUR on {t['date']}" for t in f["case_transactions"]) or "none found",
                      prior=f["prior_disputes"], complaint=f["complaint"], reply=it["reply"])
        if args.prompt == "v1":
            return PROMPT.format(clauses=", ".join(f["cited_clauses"]), **common)
        today = date.fromisoformat(gold[it["case_id"]]["as_of"])
        dates = [date.fromisoformat(t["date"]) for t in f["case_transactions"] if t["amount"] > 0]
        derived = [f"{(today - d).days} days between transaction {d} and today" for d in dates]
        if len(dates) >= 2:
            derived.append(f"{abs((dates[1] - dates[0]).days)} days between the two charges")
        if not dates:
            derived.append("no matching transaction found on the account")
        clause_ids = list(dict.fromkeys([*f["cited_clauses"], "POL-GEN-02", "POL-GEN-03", "POL-GEN-04"]))
        clause_text = "\n".join(f"{c.clause_id}: {c.text}" for c in (kb.get(i) for i in clause_ids) if c)
        params = ", ".join(f"{k}={v:g}" for k, v in sorted(kb.params.items()))
        return PROMPT_V2.format(today=today, version=kb.version, params=params, derived="; ".join(derived),
                                clause_text=clause_text, **common)

    def judge(it: dict) -> tuple[Verdict, float]:
        brain = make_llm_brain(settings)
        runs = [brain.complete(prompt_for(it), Verdict) for _ in range(args.votes)]
        votes = {q: sum(getattr(r, q) for r in runs) * 2 > len(runs) for q in QUESTIONS}  # strict majority
        reason = next((r.reason for r in runs if r.send == votes["send"]), runs[0].reason)
        return Verdict(**votes, reason=reason), brain.usage.cost_usd

    return prompt_for, judge


def screen(args) -> None:
    """Judge every reply of an eval report; report the share passing each question, overall and by decision."""
    sys.path.insert(0, str(ROOT / "eval" / "judge"))
    from build_sample import case_facts

    settings = Settings(**({"model": args.model} if args.model else {}))
    report = json.loads(args.report.read_text())
    gold = {json.loads(l)["case_id"]: json.loads(l) for l in (ROOT / "eval" / "golden_v1.jsonl").read_text().splitlines() if l.strip()}
    items = [{"id": c["case_id"], "case_id": c["case_id"], "reply": c["response_draft"], "defect": None,
              "facts": case_facts(gold[c["case_id"]], c)} for c in report["cases"] if c.get("response_draft") and c["case_id"] in gold]
    _, judge = make_judge(args, settings)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(judge, items))
    cost = sum(c for _, c in results)
    print(f"screening {args.report.name}: {len(items)} replies, judge {settings.model} {args.prompt} x{args.votes} (${cost:.3f})")
    by: dict = defaultdict(lambda: defaultdict(int))
    for it, (v, _) in zip(items, results):
        for key in ("all", it["facts"]["decision"]):
            by[key]["n"] += 1
            for q in QUESTIONS:
                by[key][q] += getattr(v, q)
    print(f"{'decision':<13} {'n':>4} " + " ".join(f"{q:>9}" for q in QUESTIONS))
    for key in ["all", "refund", "request_info", "reject", "escalate"]:
        if by[key]["n"]:
            print(f"{key:<13} {by[key]['n']:>4} " + " ".join(f"{by[key][q] / by[key]['n']:>9.2f}" for q in QUESTIONS))
    fails = [(it["id"], it["facts"]["decision"], v.reason) for it, (v, _) in zip(items, results) if not v.send]
    out = ROOT / "eval" / "results" / f"judge-screen-{args.report.stem}-{args.prompt}-x{args.votes}.json"
    out.write_text(json.dumps({"report": args.report.name, "cost_usd": cost, "rates": by, "not_ok": fails}, indent=2, default=str))
    print(f"not OK to send: {len(fails)} | Report: {out.relative_to(ROOT)}")


def report_rows(args, settings, items, labels, results) -> None:
    verdicts = [v for v, _ in results]
    cost = sum(c for _, c in results)

    print(f"judge {settings.model}, prompt {args.prompt}, {args.votes} vote(s), on {len(items)} human-labelled replies (${cost:.3f})\n")
    print(f"{'question':<10} {'agreement':>9} {'kappa':>7} {'human says ok':>14} {'judge says ok':>14}")
    report = {"judge_model": settings.model, "prompt": args.prompt, "votes": args.votes, "n": len(items), "cost_usd": cost,
              "per_question": {}}
    for q in QUESTIONS:
        h = [labels[it["id"]][q] for it in items]
        j = [getattr(v, q) for v in verdicts]
        agree = sum(x == y for x, y in zip(h, j)) / len(h)
        k = kappa(h, j)
        report["per_question"][q] = {"agreement": agree, "kappa": k}
        print(f"{q:<10} {agree:>9.2f} {k:>7.2f} {sum(h):>14} {sum(j):>14}")

    by_kind = defaultdict(lambda: {"n": 0, "human_rejects": 0, "judge_rejects": 0})
    for it, v in zip(items, verdicts):
        kind = it["defect"] or "none (real reply)"
        by_kind[kind]["n"] += 1
        by_kind[kind]["human_rejects"] += not labels[it["id"]]["send"]
        by_kind[kind]["judge_rejects"] += not v.send
    print(f"\n{'reply type':<22} {'n':>3} {'human: not ok':>14} {'judge: not ok':>14}")
    for kind, c in sorted(by_kind.items()):
        print(f"{kind:<22} {c['n']:>3} {c['human_rejects']:>14} {c['judge_rejects']:>14}")
    report["by_defect"] = by_kind

    disagreements = [(it["id"], it["defect"], labels[it["id"]]["send"], v.send, v.reason, labels[it["id"]].get("note", ""))
                     for it, v in zip(items, verdicts) if labels[it["id"]]["send"] != v.send]
    print(f"\nDisagreements on 'OK to send' ({len(disagreements)}):")
    for d in disagreements:
        print(f"  {d[0]} defect={d[1]} human_ok={d[2]} judge_ok={d[3]} | judge: {d[4]} | you: {d[5] or '-'}")
    report["disagreements"] = disagreements
    out = ROOT / "eval" / "results" / f"judge-{settings.model.replace('/', '_')}-{args.prompt}-x{args.votes}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
