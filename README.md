# Dispute Resolution Agent

A **state-driven LLM agent** that resolves card-payment disputes end to end: it classifies the complaint,
gathers evidence with tools, checks policy, and decides **refund / request info / reject / escalate**.
Every decision comes with cited policy clauses and a drafted customer reply.

> Work in progress. See [PLAN.md](PLAN.md) for the roadmap and current status.

## Architecture

```
intake → classify → gather_evidence → policy_check → decide → draft_response → self_check → END
                                                        ↘ human_review ↗             ↺ retry draft
```

| Concern | Where | Notes |
|---|---|---|
| Control flow, guards | `graph.py` (LangGraph) | step budget, retry cap, money/confidence thresholds, human interrupt + resume |
| Typed state | `state.py` (Pydantic) | evidence uses an additive reducer; usage/cost tracked per case |
| Language + judgement | `brain.py`, `llm_brain.py` | `RuleBrain` (offline baseline) and `LLMBrain` (Claude structured outputs) behind one interface |
| Tools | `tools/` | DuckDB ledger, duplicate search, customer history, BM25 policy KB, deterministic refund calculator |
| Policy KB | `kb/policies/v1/` | versioned markdown clauses (`POL-XXX-NN`) for a fictional bank |
| Evaluation | `eval/` | golden set, metrics (decision / refund / citation accuracy, cost, latency), JSON reports |

Design principle: **deterministic first.** Code handles amounts, dates, thresholds and evidence requirements.
The model handles language understanding and policy judgement only, and its proposals pass through guards.

## Quickstart

```bash
uv sync
uv run pytest
uv run python eval/run_eval.py --brain rules            # offline baseline
uv run dispute-agent "I don't recognise a 899.00 EUR charge from LuxWatch on 2026-09-15." --customer C006
#   → pauses for human review (refund > 500 EUR), then resumes from the checkpoint

cp .env.example .env   # add ANTHROPIC_API_KEY
uv run python eval/run_eval.py --brain llm --model claude-opus-5-5
```

## Results

_No published numbers yet. Results are added only from reproducible `eval/` runs._

## Data
Banking77 (public, CC-BY-4.0) for the router, hand-written golden-set complaints, a synthetic ledger and a fictional policy handbook. No real customer or company data.
