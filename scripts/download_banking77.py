"""Download Banking77 and map its intents to DisputeType for the router (Days 1-2, 8-9).

Banking77 (PolyAI, CC-BY-4.0): ~13k real banking customer-support messages, 77 intents.
https://github.com/PolyAI-LDN/task-specific-datasets

Why not CFPB: as of 2026 the CFPB bulk export and search API no longer include complaint narratives.

    uv run python scripts/download_banking77.py
Writes data/raw/banking77_{train,test}.csv and data/router_{train,test}.jsonl ({text, intent, dispute_type}).
"""

from __future__ import annotations

import collections
import csv
import json
import ssl
import urllib.request
from pathlib import Path

import certifi

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
DATA = Path(__file__).resolve().parents[1] / "data"

# Banking77 intent -> DisputeType. Everything unmapped becomes "other".
# Known gap: Banking77 has no "goods not received" intent, so not_received needs synthetic examples (see PLAN.md).
INTENT_MAP = {
    "transaction_charged_twice": "duplicate_charge",
    "card_payment_not_recognised": "unauthorized",
    "direct_debit_payment_not_recognised": "unauthorized",
    "cash_withdrawal_not_recognised": "unauthorized",
    "compromised_card": "unauthorized",
    "extra_charge_on_statement": "wrong_amount",
    "card_payment_wrong_exchange_rate": "wrong_amount",
    "wrong_amount_of_cash_received": "wrong_amount",
    "card_payment_fee_charged": "wrong_amount",
    "Refund_not_showing_up": "refund_not_processed",
}


def main() -> None:
    ctx = ssl.create_default_context(cafile=certifi.where())  # python.org macOS builds ship without root certs
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        raw = DATA / "raw" / f"banking77_{split}.csv"
        with urllib.request.urlopen(BASE.format(split=split), timeout=120, context=ctx) as r:
            raw.write_bytes(r.read())
        rows = list(csv.DictReader(raw.open()))
        out = DATA / f"router_{split}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps({"text": row["text"], "intent": row["category"],
                                    "dispute_type": INTENT_MAP.get(row["category"], "other")}) + "\n")
        dist = collections.Counter(INTENT_MAP.get(r["category"], "other") for r in rows)
        print(f"{split}: {len(rows)} rows -> {out.name}  {dict(dist)}")


if __name__ == "__main__":
    main()
