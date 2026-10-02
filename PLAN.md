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
- [x] 300 synthetic `not_received` router messages (`data/synthetic/not_received.jsonl`, $0.04). **Reviewed: 300/300 approved**.
      Report router metrics on this class separately (LLM-written, so it's an easier distribution than human text)
- [x] Router splits (`scripts/make_router_splits.py` → `data/router/`): Banking77 train → 90/10 train/val **stratified on the 77
      original intents**; official Banking77 test kept untouched; synthetic not_received 70/15/15; 7 exact-duplicate texts that
      leaked across Banking77's own splits removed from train/val. Result: train 9,204 / val 1,047 / test 3,125
- [x] Review tool `scripts/review.py` (auto-flags + per-scenario sample, verdicts saved into the files; rejected items are
      excluded from evals and router splits). Review completed 2026-10-02
- [x] Golden cases carry their own ledger rows (`ledger` field); the eval builds one DuckDB from them and runs in parallel
- [x] `golden_v1.jsonl`: **204 cases, 17 scenarios × 12, labels correct by construction** (`scripts/generate_golden.py`).
      The LLM writes narratives only, and automatic checks require the merchant and amounts. **Review pending (me)**: skim
      narratives for faithfulness (e.g. "no contact" cases must not imply contact). **Reviewed: 203 approved, 1 rejected (excluded)**

### Days 3–5: the agent core
- [x] `gather_evidence` is a **bounded LLM tool-use loop** (`evidence.py`): typed actions through structured outputs (works with
      any provider), a budget of 6 calls, repeat calls refused, errors fed back, plan fallback on model failure, and code-run
      coverage fills recorded per case. Plan mode is now the same executor running a scripted call list
- [x] Plan vs agent, 3 runs each on golden_v1 (203 cases). Ablation without coverage fills. Results in README and Findings
- [x] SQLite checkpointer: `dispute-agent run / pending / show / resume` across processes; checkpoint types allow-listed;
      restart test
- [x] `--repeats N` in the eval: mean ± std, flaky cases, always-wrong cases
- [x] Per-step `reasoning_effort` (`DISPUTE_AGENT_EFFORT="decide=medium,action=medium"`, eval `--effort`), measured on
      golden_v1. The model's default turned out to be **no reasoning**. Adding reasoning made results *worse* (see Findings),
      so the default stays: no effort parameter. Claude's `output_config.effort` is not wired (no Claude access to test it)
- [x] Self-check: cited clauses must exist in the retrieved policy, and **the reply must match the decision**
      (refund states the computed amount; reject / request_info / escalate must not promise money; request_info must ask for
      something; escalate must say the case is being reviewed). A failed check triggers a redraft that is told what was wrong.
      The first version had 5 false positives on real LLM replies; fixed, added as tests, and all 731 saved non-refund replies
      re-scored clean

### Bugs found in manual testing (2026-10-02, case-0aec9c90): fixed 2026-10-02
- [x] `resume` used the default brain instead of the case's own. Fix: each case stores `run_config` (brain, model,
      evidence mode, **resolved policy version**) and resume rebuilds from it. A case also can't switch policy version mid-case
- [x] Usage was overwritten across processes. Fix: every graph step is metered and adds only its own usage to the case total
      (verified: the resumed LLM case reports 10 calls / $0.0077 instead of 0 / $0)
- [x] Citation relevance. Fix: `clause_applies()` conditions per clause; inapplicable citations are dropped and recorded
      in `dropped_citations`. Measured on golden_v1 (plan, gpt-5.4-mini): the model over-cites in 76% of cases (mostly
      general clauses); 0 of 278 dropped citations were expected clauses; citation recall 99.6% → 100%

### Days 6–7: retrieval and a living knowledge base
- [x] Hybrid retrieval: BM25 + embeddings (text-embedding-3-small via LiteLLM) fused with Reciprocal Rank Fusion
      (`KnowledgeBase(mode=bm25|dense|hybrid)`, `DISPUTE_AGENT_RETRIEVAL`, eval `--retrieval`). Query set: 156 golden
      complaints → their expected type-specific clauses (`eval/run_retrieval_eval.py`). Reranker not added: with type
      filtering the right clause is already in the top 3 every time (see Findings)
- [x] Policy thresholds moved from code into `Parameters:` lines of the policy itself; policy **v2** published
      (90-day window, 300 EUR review threshold, 24 servicing clauses as distractors, `kb/policies/CHANGELOG.md`)
- [x] Incremental re-indexing: content-hash embedding cache; v1 → v2 re-embeds 26 clauses and reuses 11 (tested)
- [x] Policy-change eval: 40 human-reviewed boundary cases (`golden_policy_shift.jsonl`) + deterministic v2 relabelling
      (`scripts/relabel_for_policy.py`). Same code, agent mode: **40/40 under v1 and 40/40 under v2, 32 decisions changed
      exactly as the policy change requires, 8 controls unchanged**. Tests show the same case flips v1→v2
- [x] Audit trail: every case records `kb_version`, the `policy_params` it used, the version of each clause, and its
      `run_config` (a case keeps its policy version on resume)

### Days 8–9: deep-learning router
- [x] Fine-tuned **ModernBERT-base** (PyTorch + HF Trainer, Apple M5 Pro GPU via MPS, 9.1 min, 4 epochs) on 77 Banking77
      intents + synthetic not_received (`scripts/train_router.py`; best epoch by validation macro-F1). Official test split:
      **intent acc 93.2%, macro-F1 0.932** (in line with published Banking77 results); dispute type acc 98.9%, macro-F1 0.969.
      Synthetic class reported separately (97.8% acc, n=45; it is LLM-written, so an easier distribution)
- [x] Zero-shot LLM on the same test split (`eval/run_router_baseline.py`): dispute type acc 90.7%, macro-F1 0.757,
      1.2 s/msg, $0.31 per 1k. Router: 98.9% / 0.969, 9 ms/msg, ~$0. **Caveat:** most of the gap is label-convention
      disagreement (e.g. "lost or stolen card" is `other` in my mapping; the LLM says `unauthorized`), because the router
      learned my mapping from data and the LLM never saw it. Fairer: report it as "reproduces the labelling scheme"
- [x] Router in `classify` (`router.py`, `--router models/router`, threshold 0.9; LLM fallback below it). On the golden
      complaints (domain shift: long, multi-sentence) the router alone is 96.7%, but at confidence ≥ 0.9 it was right 100% of
      the time on 54% of cases. End to end on golden_v1 (plan mode): router decides 52% of classifications, type accuracy
      stays 100%, LLM calls 4.01 → 3.49/case, cost −7%, latency 5.4 → 4.4 s. The 2 decision misses were known flaky cases
      at the decide step, not routing errors

### Days 10–12: evaluation and comparisons
- [x] Metrics: decision accuracy, citation recall, tool calls per case, cost and latency per case, escalation rate (all in
      every eval report, plus unsafe refunds, guard interventions, coverage fills, dropped citations, router share)
- [x] Failure taxonomy (`eval/failure_taxonomy.py`, printed by every eval): one primary cause per failed case. Over all LLM
      runs so far: **0 unsafe refunds with mini, 0 wrong types, 0 wrong amounts**; 57% of failures are the model
      over-escalating (safe direction)
- [x] Comparisons on golden_v1 (203 cases): rules 84.2% · nano 87.2% (after fix; see Findings) · **mini 100% ± 0 over 3 runs,
      $0.0026, 4.1 s** · mini + router 100% ± 0, $0.0024, 3.8 s · gpt-5.5 100%, $0.023 (9×), 8.5 s · mini agent bm25 100%
      vs hybrid 100% (no difference, as the retrieval eval predicted). `--per-scenario N` probes cost before full runs
- [x] LLM judge for reply quality (`eval/judge/`), **validated against my own labels**: 40 replies (25 real + 15 with defects
      injected by code, blind), 5 yes/no questions, adjudication of disagreements (I re-checked 4, all my misses).
      "OK to send" with a 3-vote majority: judge v1 82% agreement / kappa 0.65; v2 (+ today's date, policy text and values,
      ledger-derived day counts) 85% / 0.69, facts kappa 0.47 → 0.68. Both judges caught 15/15 and 14/15 injected defects;
      on my first pass I caught 11/15 (I missed every wrong merchant). Weak spots: tone (kappa 0.13–0.24), arithmetic
      (missed "40 days" when it was 30), run-to-run variance. v2 was tuned on the same 40 replies, so its scores are optimistic.
      Verdict: an automated screen with human spot checks, not the final word
- [x] GitHub Actions: `ci.yml` (tests + rule-brain evals with `--min-accuracy` / `--max-unsafe` gates on every push, green);
      `llm-eval.yml` on demand only, personal key via secrets (never the employer key)

### Days 13–14: presentation
- [ ] README: architecture diagram, results table, design decisions ("why a state machine and not a ReAct loop")
- [ ] **Streamlit demo app** (`app/`, `uv run streamlit run app/main.py`), reusing the agent code directly:
  - [ ] **Customer view:** pick a demo customer, see their transactions, type a complaint, watch the agent work step by step
        (live node / tool-call progress), see the decision and the drafted reply
  - [ ] **Dispute officer view:** queue of cases waiting for human review (from the SQLite checkpoints); open a case to see
        evidence, cited clauses, proposed decision and guard reasons; approve / reject / ask the customer, then the case resumes
  - [ ] **Agent trace panel:** every tool call with its reason, observation, cost and latency; coverage fills and dropped citations
  - [ ] **Policy switch:** run the same complaint under v1 and v2 side by side (shows the living KB)
  - [ ] **Metrics page:** eval results (accuracy, unsafe refunds, cost/latency, plan vs agent, policy change) from `eval/results`
  - [ ] Settings in the sidebar: brain (rules / LLM), evidence mode (plan / agent), policy version, retrieval mode
  - [ ] **Public hosting** (Streamlit Community Cloud or Hugging Face Spaces). **No employer API key in the hosted app.** Use the
        rule brain plus **recorded replays** of real agent runs (saved traces played back step by step, $0, no key), or a
        personal key with a hard spend limit and rate limiting. Decide before hosting
  - [ ] Tests for the app's non-UI logic (replay loading, queue listing); the demo must work offline with the rule brain
- [ ] A short demo GIF of the **Streamlit app** (customer complaint → human review → resume), embedded in the README
- [ ] Resume bullet: **use only measured numbers**; add the live demo link once hosted

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
- **2026-10-02: validating the LLM judge.** The surprise was on the human side: on a first pass I missed all 3 replies
  with a swapped merchant name, and one invented "24-hour guarantee". The judge caught them. After adjudication we
  agreed on all 15 defects. The judge's own failures were systematic: (1) false alarms when it lacked context (it called
  the policy's "provisional credit" an invented promise); fixed by giving it the policy text. (2) Blind to a wrong day
  count even when given the correct number. (3) Lenient on tone. (4) Different verdicts across runs, so majority-vote it.
  Lesson: validate the judge like a model (labels, kappa, per-defect recall), and validate the human labels too.
- **2026-10-02: reply-quality findings from labelling** (none are caught by the code checks; candidates for the reply
  prompt): request_info replies re-ask for details the customer already gave and don't say the charge wasn't found;
  rejections lack empathy and a next step (e.g. "cancel the subscription with the merchant", POL-DUP-02); a refund
  reply omitted "provisional"; one real reply stated "40 days" for a 30-day gap (fix: give the writer computed facts).
- **2026-10-02: the cheap model found a guard gap.** gpt-5.4-nano refunded 2 "unauthorized" claims from customers with
  2+ prior fraud claims (POL-UNA-02 says escalate). mini and gpt-5.5 never made that mistake, so the gap stayed invisible.
  An audit of every refund-forbidding rule found it was the **only** rule without a code guard. Added the guard, with the
  threshold as a policy parameter (`repeat_unauthorized_claims=2`). nano rerun: 0 unsafe refunds. Lesson: test guards
  with a weaker model too, because a strong model hides missing guards by not needing them.
- **2026-10-02: router, honest framing.** The fine-tuned model is excellent at what it was trained for (93.2% on
  Banking77) and fast (9 ms vs 1.2 s), but (1) a large part of its lead over the LLM is my own label conventions, and
  (2) on longer, unfamiliar complaints it drops to 96.7%. Calibrated confidence makes it useful anyway: a 0.9 threshold
  keeps only the cases it gets right. Savings are modest (−13% LLM calls), because classification is 1 of ~4 calls.
  The bigger lever would be routing the extraction step, or letting the router skip the LLM entirely on simple cases.
- **2026-10-02: Metal (MPS) and threads.** 8 eval threads using the router crashed the process inside Apple's GPU
  driver (`MTLCommandBuffer` assertion), even with a lock around inference. Fix: one dedicated GPU thread owns all
  model loading and inference; other threads submit requests to it.
- **2026-10-02: policy update without code change.** Moving thresholds into the policy documents was the prerequisite:
  before that, publishing "90 days" in the text would have changed nothing, because code still checked 120. After it,
  the agent applied v2 on the policy-change set with 40/40 correct and 32/32 expected flips. The LLM read the new
  numbers from the clause text, and the code guards read them from the clause parameters, so both agree by construction.
- **2026-10-02: retrieval.** Filtered by dispute type (how the agent searches), every mode puts the right clause in the
  top 3 (100%). Recall@1 is only ~66% because sibling clauses compete (refund vs "not a duplicate", refund vs "contact
  the shop first"), and the complaint text can't settle that; the evidence does, which is why code checks
  applicability. Unfiltered (37 clauses compete), embeddings matter: recall@3 bm25 0.66 → dense 0.79 / hybrid 0.80,
  MRR 0.54 → 0.62 / 0.64. A hash "embedder" is far worse than BM25 (0.39), so quality needs real embeddings.
- **2026-10-02: more reasoning made the agent worse here.** gpt-5.4-mini's default is *no* reasoning (0 reasoning tokens;
  `minimal` is rejected by LiteLLM). One run each on golden_v1:
  plan/none 99.5% · plan/decide=medium 98.0% · plan/all=low 95.6% · agent/none 100% · agent/decide+action=medium 99.0%.
  Every extra error is the same pattern: "I don't recognise this payment" (normal amount, policy says refund) escalated
  by the model's own choice. With more reasoning the model becomes over-cautious about fraud and overrides the policy.
  In agent mode, medium effort also made the agent **skip more checks itself** (code coverage fills 25% → 46%) at +86% cost
  ($0.013 vs $0.007/case) and +59% latency (20.5 s vs 12.9 s). All errors were safe (0 unsafe refunds). Lesson: more
  reasoning is not free accuracy. For a policy-following task it can make the model second-guess the rules. Measure it.
- **2026-10-02: the reply check had false positives.** "please contact GadgetHub first… send us the details" was flagged as
  "not asking for info", and "we can't provide a refund or provisional credit" as "promising money". Fixed with positive-
  phrasing patterns plus regression tests from the real replies. Lesson: a checker needs its own eval, or it quietly
  blocks good outputs.
- **2026-10-02: agent vs plan.** Plan mode (scripted tools): 99.0% decisions, stable across 3 runs, with 3 flaky cases.
  Agent mode: 100% in all 3 runs, 0 flaky, but **2.7× the cost** ($0.0074 vs $0.0027) and **2.4× the latency** (11.9 s vs 4.9 s),
  and 24% of cases needed a code coverage fill. Ablation without fills: **94.6%**. The agent skipped the transaction search
  (it went straight to `find_duplicates`) or skipped the duplicate check. **Every** no-fill failure was safe (request_info or
  escalate, 0 unsafe refunds). Lesson: let the agent explore, but code must guarantee the checks that policy depends on.
  Open question: is +1 pp worth 2.7× cost? On this easy synthetic set, the plan wins on cost. Agent mode should pay off on
  messier cases (multiple candidate transactions, ambiguous merchants), which golden v2 should add.
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
