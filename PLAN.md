# Dispute Resolution Agent: plan, log and lessons

Portfolio project for NLP / AI-agent Data Scientist roles. Built in **3 days (2026-10-01 → 2026-10-03)**, one phase at a
time: each phase had a done-criterion, was reviewed by me, and only then did the next one start.

**Goal:** a state-driven agent that resolves card-payment disputes end to end. A complaint ("charged twice", "never
delivered", "I don't recognise this payment") ends in **REFUND / REQUEST_INFO / REJECT / ESCALATE**, with cited policy
clauses, the evidence gathered through tools and a drafted customer reply. It has to show agent engineering, real ML,
and measurement.

```
intake → classify → gather_evidence ⇄ tools → policy_check → decide → draft_response → self_check → END
                                                                  ↘ human_review (money / confidence / conflict guards)
```

## Rules

- **Honest numbers only:** every metric in the README and resume comes from reproducible `eval/` output.
- **No company data:** public (Banking77) and synthetic data only; personal GitHub and personal API keys only.
- **Deterministic first:** amounts, dates and thresholds are computed by code. The LLM handles language and judgement.
- **Always green:** each phase keeps `uv run pytest` passing and the rule-brain eval runnable.

## Plan

| # | Phase | Done when | Status |
|---|---|---|---|
| 1 | **Scaffold** | Typed `CaseState`, graph with guards and human interrupt, tools (DuckDB ledger, BM25 KB, refund calculator), `RuleBrain` + `LLMBrain`, policy KB v1, eval harness, CLI, tests | ✅ Day 1 |
| 2 | **Data** | Banking77 mapped to dispute types; synthetic `not_received` class; leak-free stratified router splits; 204-case golden set correct by construction; **every generated item reviewed by me** | ✅ Day 1 |
| 3 | **Agent core** | Bounded LLM tool-use loop for evidence; SQLite checkpoints with resume across processes; repeated evals (mean ± std); per-step reasoning effort measured; self-check (citations exist, reply matches decision) | ✅ Day 1–2 |
| 4 | **Living knowledge base** | Thresholds live in the policy text (`Parameters:`); policy v2 published; hybrid BM25 + embeddings (RRF); incremental re-indexing; policy-change eval; full audit trail per case | ✅ Day 2 |
| 5 | **Deep-learning router** | ModernBERT fine-tuned locally; compared with zero-shot LLM; wired into `classify` with a confidence threshold and LLM fallback; end-to-end cost/latency measured | ✅ Day 2 |
| 6 | **Evaluation** | Metrics in every report; failure taxonomy; model / router / retrieval comparisons; LLM judge validated against my labels (kappa); CI with quality gates | ✅ Day 2 |
| 7 | **Presentation** | README with results and architecture; Streamlit app (customer, review queue, trace, policy switch, results); recorded replays; public hosting with bring-your-own-key; demo GIF + MP4 | ✅ Day 3 |
| 8 | **Smoke test** | Live LLM agent on the hosted app with a personal key | ⬜ me |

## Log

### Day 1 (2026-10-01): scaffold, data, agent loop
- Scaffolded the repo (uv, LangGraph, Pydantic state, guards in code, 13 tests). First LLM run: 12/12 on the seed set.
- CFPB exports no longer contain narratives, so switched to **Banking77** (13k real messages, 77 intents → dispute
  types) + 300 synthetic `not_received` messages (reviewed 300/300).
- Router splits stratified on the 77 intents; removed 7 duplicates leaking across Banking77's own splits
  (train 9,204 / val 1,047 / test 3,125).
- **Golden set:** 204 cases, 17 scenarios; labels come from construction, the LLM only writes narratives. Reviewed with
  `scripts/review.py`: 203 approved, 1 rejected.
- Evidence gathering became a **bounded tool-use loop** (6-call budget, repeat calls refused, plan fallback, code
  coverage fills). Durable cases (SQLite) and `--repeats N` evals.

### Day 2 (2026-10-02): knowledge base, router, evaluation
- Self-check: cited clauses must exist, the reply must match the decision; redraft on failure.
- Fixed 3 bugs from manual testing: resume used the wrong brain, usage totals were overwritten, irrelevant citations.
- **Policy v2** (90-day window, 300 EUR review threshold) with no code change: 40/40 correct under v1 and v2,
  exactly the 32 expected decision flips. Hybrid retrieval with an embedding cache (v1 → v2 re-embeds 26, reuses 11).
- **ModernBERT router** (9 min on an M5 Pro): Banking77 intent accuracy 93.2%, dispute type 98.9% at 9 ms vs the LLM's
  90.7% at 1.2 s. In the agent: decides 52% of classifications, cost −7%, accuracy unchanged.
- **Comparisons** on 203 cases: rules 84.2% · gpt-5.4-nano 87.2% · **gpt-5.4-mini 100% ± 0 (3 runs), $0.0024/case with
  router** · gpt-5.5 100% at 9× the cost. 0 unsafe refunds everywhere after the POL-UNA-02 guard.
- **LLM judge** validated on 40 replies against my labels: 85% agreement on "OK to send", kappa 0.69. Used it to measure
  a reply-writer fix (computed facts, decision-specific guidance): "OK to send" 72% → 76%.
- CI: tests + rule-brain evals with accuracy / unsafe-refund gates on every push; LLM eval on demand.

### Day 3 (2026-10-03): app, hosting, demo
- Streamlit app: file a dispute and watch the trace live, review queue that resumes paused cases, case trace,
  policy v1 vs v2 side by side, results charts. 7 recorded LLM runs play without a key.
- Hosted on Streamlit Community Cloud in **public mode**: visitor brings their own key (session only, official
  endpoints, never stored), per-session case isolation, server keys ignored. 5 security tests.
- Fixed a key-entry crash found on the hosted app (Streamlit widget state set after drawing → `on_click` callbacks,
  regression test). Officer's refund amount now pre-filled with what policy would pay.
- Demo recorded by a Playwright script (`scripts/record_demo.py`): 2x resolution, paced scrolling, ≥5 s holds.
- Final state: 60 tests green, CI green, live demo: https://dispute-resolution-agent-yycapp6ubuxabyf8trjcyrv.streamlit.app

## Lessons learnt

1. **Tools must report what they did not find.** Without explicit `no_duplicates` / `no_merchant_refunds` evidence,
   the model filled the gap and proposed a refund for a monthly subscription.
2. **Guards belong in code, and a weak model is how you test them.** gpt-5.4-nano refunded repeat fraud claimants;
   mini and gpt-5.5 never did, which hid that POL-UNA-02 was the only refund-forbidding rule without a code guard.
3. **Let the agent explore, but guarantee the checks policy depends on.** Agent mode hit 100% vs the plan's 99%, at
   2.7× cost and 2.4× latency; without code coverage fills it fell to 94.6% (all failures safe).
4. **More reasoning is not free accuracy.** Adding reasoning effort to gpt-5.4-mini *lowered* accuracy (99.5% → 95.6% at
   worst): it became over-cautious about fraud and overrode the policy, at +86% cost.
5. **Validate the judge like a model, and validate the human too.** The judge caught swapped merchant names I missed;
   it was blind to wrong day counts and lenient on tone. Use it as a relative screen with spot checks.
6. **Checkers need their own eval.** The reply self-check had false positives on real replies, and a regex missed
   "couldn’t" (typographic apostrophe). Regression tests come from real outputs.
7. **A fine-tuned router's lead is partly your own labels.** Much of the 98.9% vs 90.7% gap is label-convention
   disagreement; the calibrated 0.9 threshold is what makes it useful on longer, unfamiliar complaints.
8. **Policy as data makes updates code-free.** Once thresholds moved into the policy documents, the LLM read them from
   the text and the guards from the parameters, so v2 applied correctly with no code change.
9. **Dates and money never go to the LLM.** "7 September" became 2024; refund amounts and day counts are computed.
10. **Platform details bite late.** Apple MPS crashed with 8 threads (one GPU thread fixed it); Streamlit widget state
    and hosted-mode differences only showed up when a real user clicked through.

## Resume bullet

> **Dispute Resolution Agent** (Python, LangGraph, PyTorch, OpenAI/Claude APIs, Streamlit) ·
> [github.com/nickkirpa/dispute-resolution-agent](https://github.com/nickkirpa/dispute-resolution-agent) ·
> [live demo](https://dispute-resolution-agent-yycapp6ubuxabyf8trjcyrv.streamlit.app)
> State-driven LLM agent that resolves card-payment disputes end to end: LLM-chosen tool calls under a budget, a
> versioned policy knowledge base with hybrid retrieval, human-in-the-loop review with durable checkpoints, and
> code-enforced money guards. On 203 human-reviewed cases: 100% decision accuracy over 3 runs, 0% unsafe refunds,
> $0.0024/case. Fine-tuned a ModernBERT router (93.2% on Banking77, 9 ms vs 1.2 s for the LLM) and validated an LLM
> judge against human labels (kappa 0.69). A policy update changed exactly the 32 required decisions with no code change.
