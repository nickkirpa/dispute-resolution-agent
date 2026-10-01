"""Deterministic fixture ledger that the seed golden set (golden.jsonl) is labelled against.

As-of date for all golden cases: 2026-09-30.
"""

from __future__ import annotations

from datetime import date

from dispute_agent.tools import Ledger

CUSTOMERS = [(f"C{i:03d}", f"Customer {i}", date(2022, 1, 1)) for i in range(1, 13)]

# (txn_id, customer_id, merchant, amount, date, channel)
TXNS = [
    ("T001", "C001", "SpotiTunes", 9.99, date(2026, 9, 2), "online"),
    ("T002", "C001", "SpotiTunes", 9.99, date(2026, 9, 3), "online"),
    ("T003", "C001", "FitClub", 49.00, date(2026, 9, 1), "recurring"),
    ("T004", "C001", "GroceryCo", 63.20, date(2026, 9, 5), "card_present"),
    ("T010", "C002", "Netflux", 15.99, date(2026, 7, 5), "recurring"),
    ("T011", "C002", "Netflux", 15.99, date(2026, 8, 5), "recurring"),
    ("T012", "C002", "GroceryCo", 41.10, date(2026, 8, 9), "card_present"),
    ("T020", "C003", "GadgetHub", 249.00, date(2026, 8, 20), "online"),
    ("T021", "C003", "GroceryCo", 22.75, date(2026, 8, 21), "card_present"),
    ("T030", "C004", "ShoeBox", 89.90, date(2026, 9, 1), "online"),
    ("T040", "C005", "CryptoMart", 120.00, date(2026, 9, 20), "online"),
    ("T041", "C005", "GroceryCo", 17.40, date(2026, 9, 19), "card_present"),
    ("T050", "C006", "LuxWatch", 899.00, date(2026, 9, 15), "online"),
    ("T060", "C007", "PizzaNow", 34.50, date(2026, 9, 25), "online"),
    ("T070", "C008", "HotelLisboa", 320.00, date(2026, 9, 10), "card_present"),
    ("T080", "C009", "AirFly", 410.00, date(2026, 8, 1), "online"),
    ("T081", "C009", "AirFly", -410.00, date(2026, 8, 20), "online"),
    ("T090", "C010", "BookNest", 59.00, date(2026, 8, 15), "online"),
    ("T100", "C011", "OldShop", 75.00, date(2026, 4, 1), "online"),
    ("T110", "C012", "GroceryCo", 55.00, date(2026, 9, 12), "card_present"),
]

# prior disputes: C007 has two unauthorized claims in the last 12 months
DISPUTES = [
    ("D001", "C007", "T999", "unauthorized", date(2026, 2, 10), "refunded"),
    ("D002", "C007", "T998", "unauthorized", date(2026, 6, 3), "refunded"),
]


def build_fixture_ledger() -> Ledger:
    led = Ledger(":memory:")
    led.con.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
    led.con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, 'EUR', ?, ?)", TXNS)
    led.con.executemany("INSERT INTO disputes VALUES (?, ?, ?, ?, ?, ?)", DISPUTES)
    return led
