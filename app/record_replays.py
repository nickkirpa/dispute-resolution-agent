"""Record real LLM agent runs for keyless playback in the app (app/replays/*.json, committed).

    uv run python app/record_replays.py        # needs an LLM key in .env; costs a few cents

Each replay is the exact event stream the live app shows. Cases that pause for a human are recorded together with
the officer's verdict and the resumed run, so the replay shows the full pause / review / resume flow.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
load_dotenv(ROOT / ".env")

from app import core  # noqa: E402

SCENARIOS = [
    ("C001", "Duplicate charge refund", "Two identical SpotiTunes charges a day apart: the agent finds the duplicate and refunds one.", None),
    ("C002", "Monthly charge, not a duplicate", "The customer insists on a double charge; the ledger shows charges 31 days apart, so it is a subscription.", None),
    ("C009", "Refund already credited", "The merchant's refund is already on the account; the agent rejects instead of paying twice.", None),
    ("C012", "Charge not found", "No FitClub charge on this account: the agent says so and asks only for what would identify it.", None),
    ("C007", "Repeat fraud claimant", "Third unauthorized claim in 12 months: the POL-UNA-02 guard sends it to the fraud team.",
     {"decision": "escalate", "note": "forwarded to the fraud team"}),
    ("C006", "Large refund needs a human", "899 EUR exceeds the 300 EUR review threshold (policy v2): the case pauses for an officer.",
     {"decision": "refund", "refund_amount": 899.0, "note": "called the customer, confirmed card fraud"}),
    ("C014", "Policy v2 threshold", "350 EUR: auto-refund under v1 (500 EUR limit), human review under v2 (300 EUR).",
     {"decision": "refund", "refund_amount": 350.0, "note": "verified with the customer"}),
]


def main() -> None:
    if not core.llm_available():
        raise SystemExit("no LLM key configured in .env")
    config = core.AppConfig(brain="llm", evidence_mode="agent", kb_version="v2").run_config()
    with tempfile.TemporaryDirectory() as tmp:
        agent = core.build_agent(config, core.connect(Path(tmp) / "rec.sqlite"))
        for cid, title, description, verdict in SCENARIOS:
            events = list(core.run_events(agent, config, cid, core.SUGGESTED[cid]))
            if events[-1]["type"] == "interrupt" and verdict:
                events += list(core.resume_events(agent, events[-1]["case_id"], verdict))
            final = events[-1]
            outcome = final["state"]["decision"] if final["type"] == "done" else "waiting for review"
            path = core.record(events, title, description)
            print(f"{cid} {title:<34} -> {outcome:<14} ({len(events)} events) {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
