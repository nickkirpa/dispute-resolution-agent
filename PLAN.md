# Dispute Resolution Agent: project plan

Portfolio project for NLP/DL and AI-agent Data Scientist roles. Target profile: state-driven AI agents built
from scratch, advanced LLMs, continuously updated knowledge bases, real engineering and math rather than
prompt engineering.

## Goal

Build a **state-driven agent that resolves card-payment disputes end to end**. It takes a customer complaint
("charged twice", "never delivered", "I don't recognise this payment") and reaches one of four decisions:
**REFUND / REQUEST_INFO / REJECT / ESCALATE**. Every decision comes with cited policy clauses, the evidence
gathered through tools, and a drafted customer reply.

The project has to show three things:
1. **Agent engineering:** typed state, guarded transitions, tools, human-in-the-loop, persistence and resume.
2. **Real ML / deep learning:** a fine-tuned transformer router, measured retrieval, and a knowledge base under version control.
3. **Measurement:** a golden set, metrics, comparisons, a breakdown of failure types, and regression checks in CI.

## Architecture

```
intake → classify → gather_evidence ⇄ tools → policy_check → decide
      → draft_response → self_check → END
                             ↘ human_review (high value / low confidence / refusal / step budget)
```

- **State:** `CaseState` (Pydantic) holds the claim, dispute type, evidence, retrieved clauses, missing evidence,
  the decision, the refund amount, a step counter, LLM usage and cost, and a trace.
- **Guards in code, not in the prompt:**
  - no REFUND without the required evidence;
  - the refund amount is computed by a deterministic function, never by the LLM;
  - a maximum step budget;
  - a maximum number of self-check retries;
  - escalation above a money threshold or below a confidence threshold.
- **Tools:** ledger lookup (DuckDB), duplicate search, customer history, policy KB retrieval, refund calculator.
- **Brains:** `RuleBrain` (a deterministic baseline that runs offline) and `LLMBrain` (Claude, structured outputs).
  The same graph runs with either brain, so the evaluation can compare them directly.
- **Knowledge base:** markdown policy clauses (`## POL-XXX-NN: title`) stored in version folders `kb/policies/v1`, `v2`, and so on.

## Data
- **Banking77** (PolyAI, CC-BY-4.0): ~13k real banking support messages with 77 intents, mapped to `DisputeType`
  (`scripts/download_banking77.py` → `data/router_{train,test}.jsonl`). This is the router's training and test data.
  *Note:* the CFPB complaint data was the first choice, but as of 2026 its public exports no longer include narratives (checked 2026-10-01).
  *Gap:* Banking77 has no "goods not received" intent, so `not_received` needs synthetic examples (written or LLM-generated,
  then reviewed by me).
- **Golden-set narratives:** longer multi-fact complaints written by hand or drafted by an LLM and then edited, each paired with ledger rows.
- **Synthetic ledger:** customers, transactions, refunds and prior disputes, generated to match the cases
  (`scripts/generate_ledger.py`).
- **Policy KB:** public card-network reason-code concepts plus a fictional bank's policy handbook
  ("Northwind Bank"). Written by me. No real bank's internal policy.

## Two-week plan (status checkboxes)

### Day 0: scaffold (done by Claude)
- [x] Repo layout, pyproject (uv), state schema, graph skeleton with guards and human interrupt
- [x] Tools: ledger (DuckDB), KB (BM25), refund calculator
- [x] RuleBrain baseline and LLMBrain (Claude structured outputs) with token and cost tracking
- [x] Policy KB v1, fixture ledger, 12-case seed golden set, eval harness, 13 tests, interactive CLI
- [x] Baseline: RuleBrain 100% on the seed set (expected, since the rules encode policy v1 and are the floor, not a result)
- [x] First LLM run, `openai/gpt-5.4-mini` via LiteLLM: 11/12, then 12/12 after the negative-evidence fix ($0.003/case, ~4–6 s/case). Single run on an easy 12-case set; not a headline number

### Days 1–2: data
- [x] Download Banking77 and map intents → `DisputeType` (train 10,003 / test 3,080; ~84% `other`, which is realistic for a router)
- [x] 300 synthetic `not_received` router messages (`data/synthetic/not_received.jsonl`, $0.04). **Review pending (me)**.
      Report router metrics on this class separately (LLM-written, so it's an easier distribution than human text)
- [x] Router splits (`scripts/make_router_splits.py` → `data/router/`): Banking77 train → 90/10 train/val **stratified on the 77
      original intents**; official Banking77 test kept untouched; synthetic not_received 70/15/15; 7 exact-duplicate texts that
      leaked across Banking77's own splits removed from train/val. Result: train 9,204 / val 1,047 / test 3,125
- [x] Review tool `scripts/review.py` (auto-flags + per-scenario sample, verdicts saved into the files; rejected items are
      excluded from evals and router splits). **Review itself pending (me)**
- [x] Golden cases carry their own ledger rows (`ledger` field); the eval builds one DuckDB from them and runs in parallel
- [x] `golden_v1.jsonl`: **204 cases, 17 scenarios × 12, labels correct by construction** (`scripts/generate_golden.py`).
      The LLM writes narratives only, and automatic checks require the merchant and amounts. **Review pending (me)**: skim
      narratives for faithfulness (e.g. "no contact" cases must not imply contact) and set `review_status`

### Days 3–5: the agent core
- [ ] Per-call `effort` (e.g. `low` for extract/classify, higher for decide) through `output_config`, measured by cost vs accuracy
- [ ] Turn `gather_evidence` into a **bounded LLM tool-use loop**: the model picks tools, a tool budget is enforced,
      and invalid calls are handled. Keep the deterministic plan as the fallback and the baseline
- [ ] Add a SQLite checkpointer for persistence; show crash/resume and human-review resume in the CLI
- [ ] Self-check: verify the citations exist in the retrieved clauses and that the reply matches the decision

### Days 6–7: retrieval and a living knowledge base
- [ ] Hybrid retrieval: BM25 plus embeddings plus a reranker. Build a labelled query→clause set and report recall@k and MRR
- [ ] Policy v2 (for example, a changed refund window). Incremental re-indexing; tests show the agent applies v2 and cites v2 clause ids
- [ ] Record which KB version each decision used (audit trail)

### Days 8–9: deep-learning router
- [ ] Fine-tune ModernBERT or DeBERTa-v3 (PyTorch + HF) on Banking77→DisputeType (plus the synthetic not_received set).
      Report macro-F1 on the official Banking77 test split; the classes are imbalanced, so accuracy alone is not enough
- [ ] Compare against zero-shot LLM classification on accuracy, macro-F1, latency and cost per 1k cases
- [ ] Use the router in `classify`, with the LLM as fallback when router confidence is below a threshold

### Days 10–12: evaluation and comparisons
- [ ] Metrics: decision accuracy, citation recall, tool calls per case, cost and latency per case, escalation rate
- [ ] Failure taxonomy (wrong type, missed evidence, wrong policy, a refund that should not have happened)
- [ ] Comparisons: RuleBrain vs LLMBrain; cheap vs strong model; router on vs off; BM25 vs hybrid retrieval
- [ ] If an LLM judge is used for reply quality, **validate it against my own labels** (report agreement)
- [ ] GitHub Actions: tests plus the RuleBrain eval on every push; the LLM eval runs on demand

### Days 13–14: presentation
- [ ] README: architecture diagram, results table, design decisions ("why a state machine and not a ReAct loop")
- [ ] A short demo GIF of the CLI, including the human-review interrupt and resume
- [ ] Resume bullet: **use only measured numbers**

## Findings log
Write down what broke and what fixed it. This is README and interview material.

- **2026-10-01: negative evidence.** G02 (a monthly subscription reported as a "duplicate"): gpt-5.4-mini proposed REFUND
  under POL-DUP-01. The money guard caught it (refund amount computed to 0, so the case escalated instead of paying out). Root cause:
  the evidence listed "2 matching charges" without both dates, and nothing said "no duplicates found". After adding all matching dates
  and explicit `no_duplicates` / `no_merchant_refunds` evidence, the case was decided correctly. Lesson: tools should report what they
  did *not* find. Otherwise the model fills the gap.
- **2026-10-01: dates without a year.** "dated 7 September": the model filled in 2024, decide saw a ledger mismatch and rejected.
  Fix: give extraction today's date, plus deterministic `normalize_claim_date` (future or >1y old moves to the most recent past occurrence).
- **2026-10-01: an unsafe payout the money guard missed.** For refund_not_processed, the model refunded although the ledger showed
  the merchant credit (evidence `merchant_refunds` was present). The refund amount is legitimately non-zero there, so guard 2
  didn't fire. Added guard 2b (a refund that contradicts hard evidence goes to a human) and the eval metric `unsafe_refund_rate`.
  On the next run the guard caught the same pattern on a different case (V1-0196).
- **2026-10-01: rule baseline on free text** drops to 84% (it misses "I contacted BookNest" and "charged X, agreed Y" phrasings).
  A good illustration of why the LLM brain exists.
- **2026-10-01: CFPB narratives are gone** from public exports, so the router uses Banking77 (see Data).

## Rules
- **Honest numbers only.** Every metric in the README and on the resume must come from `eval/` output that can be reproduced.
- **No company data.** Nothing from any employer. Use public and synthetic data only.
- **Deterministic first.** If code can do it reliably (amounts, dates, thresholds), code does it. The LLM handles
  language understanding and judgement only.
- Each step should keep `uv run pytest` green and `uv run python eval/run_eval.py --brain rules` runnable.

## Resume bullet (draft; fill in after measuring)
> **Dispute Resolution Agent: Python · LangGraph · PyTorch · Claude**
> A state-driven agent that resolves card disputes end to end: classify, gather evidence through tools, check policy,
> then decide or escalate. It has guarded transitions, human-in-the-loop checkpoints and retrieval over a versioned policy KB.
> A fine-tuned ModernBERT router cut LLM cost by X% at equal accuracy. A 200-case golden set tracks decision
> accuracy, citation correctness, cost and latency in CI. GitHub
