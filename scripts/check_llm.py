"""One-call smoke test for the LLM endpoint (Anthropic API or a LiteLLM proxy). Costs a fraction of a cent.

    uv run python scripts/check_llm.py

Checks: auth works, the model name resolves, structured output (messages.parse) comes back as a valid
Pydantic object, usage is reported, and pricing resolves for cost tracking.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.llm_brain import make_llm_brain  # noqa: E402
from dispute_agent.state import Claim  # noqa: E402


def main() -> None:
    s = Settings()
    if s.provider == "openai":
        print(f"provider: openai   endpoint: {os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1 (default)'}")
        print(f"auth:     {'OPENAI_API_KEY set' if os.getenv('OPENAI_API_KEY') else 'OPENAI_API_KEY missing'}")
    else:
        print(f"provider: anthropic   endpoint: {os.getenv('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com (default)'}")
        print(f"auth:     {'ANTHROPIC_AUTH_TOKEN (Bearer)' if os.getenv('ANTHROPIC_AUTH_TOKEN') else 'ANTHROPIC_API_KEY (x-api-key)' if os.getenv('ANTHROPIC_API_KEY') else 'none found'}")
    print(f"model:    {s.model}")
    brain = make_llm_brain(s)
    try:
        claim = brain.extract_claim("I was charged twice by SpotiTunes for 9.99 EUR on 2026-09-02. I emailed their support already.")
    except Exception as e:  # provider SDKs raise different classes; show the status + message
        hint = {401: "check the API key", 403: "the key may not be allowed to use this model",
                404: "check the base URL and model name", 400: "the endpoint may not support structured outputs"}
        code = getattr(e, "status_code", None)
        sys.exit(f"FAIL {type(e).__name__} {code or ''}: {e}\nhint: {hint.get(code, 'see the message above')}")
    print(f"parsed:   {claim.model_dump()}")
    ok = claim.merchant and "spoti" in claim.merchant.lower() and claim.amount == 9.99 and claim.contacted_merchant
    print(f"usage:    {brain.usage.model_dump()}  cost source: {getattr(brain, 'cost_source', 'price-table')}")
    print("PASS" if ok else "WARN: call worked but the extraction looks off; inspect the parsed output above")


if __name__ == "__main__":
    main()
