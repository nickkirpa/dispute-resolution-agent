"""Generate a synthetic ledger at data/ledger.duckdb (Day 1-2: extend to match sampled CFPB complaints).

    uv run python scripts/generate_ledger.py --customers 500 --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispute_agent.tools import Ledger  # noqa: E402

MERCHANTS = {
    "online": ["GadgetHub", "ShoeBox", "BookNest", "AirFly", "CryptoMart", "LuxWatch", "PizzaNow"],
    "recurring": ["SpotiTunes", "Netflux", "FitClub", "CloudBox"],
    "card_present": ["GroceryCo", "FuelStop", "HotelLisboa", "CafeLumen"],
}
PRICE = {"online": (10, 600), "recurring": (5, 60), "card_present": (3, 350)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--as-of", default="2026-09-30")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    as_of = date.fromisoformat(args.as_of)

    path = ROOT / "data" / "ledger.duckdb"
    path.unlink(missing_ok=True)
    led = Ledger(path)
    txn_id = 0
    for i in range(1, args.customers + 1):
        cid = f"S{i:05d}"
        led.con.execute("INSERT INTO customers VALUES (?, ?, ?)", [cid, f"Synthetic {i}", as_of - timedelta(days=rng.randint(30, 2500))])
        for channel, merchants in MERCHANTS.items():
            for _ in range(rng.randint(1, 8)):
                txn_id += 1
                lo, hi = PRICE[channel]
                led.con.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, 'EUR', ?, ?)",
                                [f"X{txn_id:07d}", cid, rng.choice(merchants), round(rng.uniform(lo, hi), 2),
                                 as_of - timedelta(days=rng.randint(0, 200)), channel])
    print(f"wrote {args.customers} customers, {txn_id} transactions -> {path}")


if __name__ == "__main__":
    main()
