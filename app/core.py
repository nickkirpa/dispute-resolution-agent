"""Non-UI logic for the Streamlit demo. Everything here is testable without Streamlit.

- build_agent(config):  the same LangGraph agent as the CLI/evals, on a demo ledger, with SQLite checkpoints
- run_events / resume_events:  stream a case step by step (one event per graph node, then done or interrupt)
- pending_cases / list_cases / get_case:  the dispute officer's queue and case traces, read from the checkpoints
- record / load_replays:  recorded LLM runs played back without an API key (for public hosting)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from dispute_agent.brain import RuleBrain  # noqa: E402
from dispute_agent.config import Settings  # noqa: E402
from dispute_agent.graph import Deps, build_graph  # noqa: E402
from dispute_agent.state import CHECKPOINT_TYPES, CaseState  # noqa: E402
from dispute_agent.tools import Ledger, build_kb  # noqa: E402
from eval.fixtures import CUSTOMERS, DISPUTES, TXNS  # noqa: E402

DB = ROOT / "data" / "app_cases.sqlite"
REPLAYS = ROOT / "app" / "replays"
AS_OF = date(2026, 9, 30)

# Two extra demo customers that sit on the v1/v2 policy boundaries, so the policy switch visibly changes decisions.
EXTRA_CUSTOMERS = [("C013", "Customer 13", date(2022, 1, 1)), ("C014", "Customer 14", date(2022, 1, 1))]
EXTRA_TXNS = [("T130", "C013", "ShoeBox", 80.00, AS_OF - timedelta(days=100), "online"),
              ("T131", "C013", "ShoeBox", 80.00, AS_OF - timedelta(days=99), "online"),
              ("T140", "C014", "AirFly", 350.00, AS_OF - timedelta(days=10), "online")]

SUGGESTED = {
    "C001": "I was charged twice by SpotiTunes, 9.99 EUR each, on 2 September. Please refund the duplicate.",
    "C002": "Netflux charged me 15.99 EUR twice, once in July and again in August. I only subscribed once.",
    "C003": "My GadgetHub headphones for 249.00 EUR from 20 August never arrived. I emailed the seller twice, no answer.",
    "C004": "I paid ShoeBox 89.90 EUR on 1 September for sneakers but they were never delivered.",
    "C005": "There is a 120.00 EUR payment to CryptoMart on 20 September that I don't recognise.",
    "C006": "There's an 899 EUR LuxWatch payment from 15 September that I never made.",
    "C007": "A charge of 34.50 EUR from PizzaNow on 25 September was not authorised by me.",
    "C008": "HotelLisboa charged me 320.00 EUR instead of 280.00 EUR for my stay on 10 September.",
    "C009": "AirFly promised me a refund for my cancelled 410.00 EUR flight booked on 1 August, but it never arrived.",
    "C010": "BookNest confirmed a refund of 59.00 EUR for a cancelled order from 15 August, but I haven't received it.",
    "C011": "I paid OldShop 75.00 EUR on 1 April for a chair that never arrived. I called the shop many times.",
    "C012": "FitClub charged me twice, 49.00 EUR each time, on 12 September.",
    "C013": "ShoeBox charged me twice, 80.00 EUR each time, for one pair of shoes.",
    "C014": "I don't recognise a 350.00 EUR payment to AirFly. I did not make it.",
}

NODE_LABELS = {
    "intake": "Reading the complaint",
    "classify": "Classifying the dispute",
    "gather_evidence": "Investigating the account",
    "policy_check": "Checking required evidence",
    "decide": "Deciding under policy",
    "human_review": "Human review",
    "draft_response": "Writing the reply",
    "self_check": "Checking the reply",
}


@dataclass
class AppConfig:
    brain: str = "rules"  # rules | llm
    evidence_mode: str = "plan"  # plan | agent
    kb_version: str = "v2"
    retrieval_mode: str = "bm25"
    router_path: str | None = None
    provider: str = field(default_factory=lambda: os.getenv("DISPUTE_AGENT_PROVIDER", "anthropic"))
    model: str = field(default_factory=lambda: os.getenv("DISPUTE_AGENT_MODEL", "claude-opus-5-5"))

    def run_config(self) -> dict:
        return asdict(self)


def llm_available(provider: str | None = None) -> bool:
    provider = provider or os.getenv("DISPUTE_AGENT_PROVIDER", "anthropic")
    return bool(os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def demo_ledger() -> Ledger:
    led = Ledger(":memory:")
    return led.load(customers=CUSTOMERS + EXTRA_CUSTOMERS, transactions=TXNS + EXTRA_TXNS, disputes=DISPUTES)


def demo_customers() -> list[dict]:
    rows = []
    for cid, name, opened in CUSTOMERS + EXTRA_CUSTOMERS:
        txns = [{"date": d.isoformat(), "merchant": m, "amount": a, "channel": ch}
                for _, c, m, a, d, ch in sorted(TXNS + EXTRA_TXNS, key=lambda t: t[4]) if c == cid]
        rows.append({"customer_id": cid, "name": name, "transactions": txns,
                     "prior_disputes": sum(1 for d in DISPUTES if d[1] == cid), "suggested": SUGGESTED.get(cid, "")})
    return rows


def connect(db: Path | None = None) -> sqlite3.Connection:
    db = Path(db or DB)  # resolved at call time, so tests (and deployments) can point DB elsewhere
    db.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db, check_same_thread=False)


def build_agent(config: AppConfig | dict, conn: sqlite3.Connection, human_in_loop: bool = True):
    cfg = config if isinstance(config, dict) else config.run_config()
    settings = Settings(provider=cfg["provider"], model=cfg["model"], evidence_mode=cfg["evidence_mode"],
                        kb_version=cfg["kb_version"], retrieval_mode=cfg.get("retrieval_mode", "bm25"),
                        router_path=cfg.get("router_path"))
    ledger = demo_ledger()
    if cfg["brain"] == "rules":
        brain = RuleBrain(known_merchants=ledger.merchants())
    else:
        from dispute_agent.llm_brain import make_llm_brain

        brain = make_llm_brain(settings)
    deps = Deps(brain=brain, ledger=ledger, kb=build_kb(settings), settings=settings, human_in_loop=human_in_loop)
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    return build_graph(deps, checkpointer=SqliteSaver(conn, serde=serde))


def jsonable(x):
    """Plain JSON-ready data from pydantic models, enums, dates (for the UI and for recorded replays)."""
    if isinstance(x, BaseModel):
        return jsonable(x.model_dump(mode="json"))
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, (date, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def _stream(agent, payload, case_id: str) -> Iterator[dict]:
    config = {"configurable": {"thread_id": case_id}}
    for chunk in agent.stream(payload, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                yield {"type": "interrupt", "case_id": case_id, "payload": jsonable(update[0].value)}
                return
            yield {"type": "step", "case_id": case_id, "node": node, "label": NODE_LABELS.get(node, node),
                   "update": jsonable(update)}
    yield {"type": "done", "case_id": case_id, "state": get_case(agent, case_id)}


def run_events(agent, config: AppConfig | dict, customer_id: str, narrative: str, case_id: str | None = None) -> Iterator[dict]:
    """Start a case and stream its events. The case keeps its config (brain, policy version...) for resumes."""
    cfg = config if isinstance(config, dict) else config.run_config()
    case_id = case_id or f"case-{uuid.uuid4().hex[:8]}"
    state = CaseState(case_id=case_id, customer_id=customer_id, narrative=narrative, as_of=AS_OF, run_config=cfg)
    yield {"type": "start", "case_id": case_id, "config": cfg}
    yield from _stream(agent, state, case_id)


def resume_events(agent, case_id: str, verdict: dict) -> Iterator[dict]:
    yield {"type": "resume", "case_id": case_id, "verdict": verdict}
    yield from _stream(agent, Command(resume=verdict), case_id)


def get_case(agent, case_id: str) -> dict:
    snap = agent.get_state({"configurable": {"thread_id": case_id}})
    return {"case_id": case_id, "next": list(snap.next), **jsonable(snap.values)} if snap.values else {}


def list_cases(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute("SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id ORDER BY 2 DESC").fetchall()
    except sqlite3.OperationalError:  # no checkpoints table yet
        return []
    return [r[0] for r in rows]


def stored_config(agent, case_id: str) -> dict:
    return get_case(agent, case_id).get("run_config") or {}


def pending_cases(agent, conn: sqlite3.Connection) -> list[dict]:
    out = []
    for cid in list_cases(conn):
        case = get_case(agent, cid)
        if case.get("next") == ["human_review"]:
            out.append({"case_id": cid, "customer_id": case["customer_id"], "dispute_type": case.get("dispute_type"),
                        "proposed": case.get("decision"), "refund_amount": case.get("refund_amount", 0.0),
                        "reason": case.get("human_reason", ""), "narrative": case["narrative"]})
    return out


# ---------------------------------------------------------------- recorded replays (no API key needed to watch)

def record(events: list[dict], title: str, description: str, path: Path | None = None) -> Path:
    REPLAYS.mkdir(parents=True, exist_ok=True)
    path = path or REPLAYS / f"{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}.json"
    path.write_text(json.dumps({"title": title, "description": description, "events": events}, indent=1))
    return path


def load_replays() -> list[dict]:
    return [json.loads(p.read_text()) | {"file": p.name} for p in sorted(REPLAYS.glob("*.json"))] if REPLAYS.exists() else []
