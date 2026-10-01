# CLAUDE.md

Portfolio project: a state-driven dispute-resolution agent. **Read `PLAN.md` first.** It holds the goal, the
two-week plan with status checkboxes, and the rules. Update the checkboxes as work lands.

## Commands
- `uv sync` installs; `uv sync --extra router` adds torch/transformers for the Day 8-9 router
- `uv run pytest`: must stay green
- `uv run python eval/run_eval.py --brain rules`: offline baseline; the seed golden set must stay at 100%
- `uv run python eval/run_eval.py --brain llm`: spends API money. Ask before running it on large golden sets
- `uv run dispute-agent "<narrative>" --customer C001`: interactive demo with human-review interrupts

## Conventions
- Control flow, money, dates and thresholds live in `graph.py` and `tools/`, in code. Never ask the LLM to compute amounts.
- Brains implement the `Brain` protocol in `brain.py`. Any new LLM capability needs a RuleBrain counterpart or
  fallback, so the offline tests keep working.
- LLM brains share prompts in `llm_brain.PromptBrain`. `LLMBrain` uses Claude (`messages.parse(output_format=Model)`, anthropic SDK 1.x);
  `OpenAIBrain` uses `chat.completions.with_raw_response.parse(response_format=Model)`. Pick with `DISPUTE_AGENT_PROVIDER`.
  Current setup: OpenAI models through a LiteLLM proxy (`.env`); cost comes from LiteLLM's `x-litellm-response-cost` header.
- Settings read the environment when `Settings()` is created, so `load_dotenv()` must run before that.
- Smoke-test any endpoint change with `uv run python scripts/check_llm.py` (one call) before running evals.
- Policy clauses are `## POL-XXX-NN: Title` + `Applies to:` in `kb/policies/vN/`. Never edit an old version; add a new one.
- Golden cases are labelled by a human. The LLM may propose labels, but a person confirms them.
- **No invented metrics** in the README or resume text. Numbers must come from `eval/results/*.json`.
- Public and synthetic data only; nothing from the author's employer.
