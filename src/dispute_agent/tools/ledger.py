"""Ledger tool: read-only queries over a DuckDB database of customers, transactions and disputes."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    opened_on DATE
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR,
    merchant VARCHAR,
    amount DOUBLE,           -- positive = charge, negative = refund/credit
    currency VARCHAR,
    txn_date DATE,
    channel VARCHAR          -- card_present | online | recurring
);
CREATE TABLE IF NOT EXISTS disputes (
    dispute_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR,
    txn_id VARCHAR,
    dispute_type VARCHAR,
    opened_on DATE,
    outcome VARCHAR
);
"""


class Ledger:
    def __init__(self, path: str | Path = ":memory:", read_only: bool = False):
        self.con = duckdb.connect(str(path), read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA)

    def fork(self) -> "Ledger":
        """A new cursor on the same database: DuckDB connections are not thread-safe, cursors are cheap."""
        other = Ledger.__new__(Ledger)
        other.con = self.con.cursor()
        return other

    def load(self, customers: list = (), transactions: list = (), disputes: list = ()) -> "Ledger":
        """Bulk insert rows: customers (id, name, opened_on), transactions (id, cust, merchant, amount, date, channel),
        disputes (id, cust, txn_id, type, opened_on, outcome)."""
        if customers:
            self.con.executemany("INSERT INTO customers VALUES (?, ?, ?)", [list(r) for r in customers])
        if transactions:
            self.con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, 'EUR', ?, ?)", [list(r) for r in transactions])
        if disputes:
            self.con.executemany("INSERT INTO disputes VALUES (?, ?, ?, ?, ?, ?)", [list(r) for r in disputes])
        return self

    def _rows(self, sql: str, params: list) -> list[dict]:
        cur = self.con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def find_transactions(
        self,
        customer_id: str,
        merchant: str | None = None,
        amount: float | None = None,
        around: date | None = None,
        window_days: int = 45,
    ) -> list[dict]:
        """Charges (amount > 0) for a customer, optionally filtered by fuzzy merchant, amount and date window."""
        sql = "SELECT * FROM transactions WHERE customer_id = ? AND amount > 0"
        params: list = [customer_id]
        if merchant:
            sql += " AND lower(merchant) LIKE ?"
            params.append(f"%{merchant.lower()}%")
        if amount is not None:
            sql += " AND abs(amount - ?) < 0.01"
            params.append(amount)
        if around is not None:
            sql += " AND txn_date BETWEEN ? AND ?"
            params += [around - timedelta(days=window_days), around + timedelta(days=window_days)]
        return self._rows(sql + " ORDER BY txn_date", params)

    def find_duplicates(self, customer_id: str, merchant: str | None, amount: float | None, within_days: int = 3) -> list[list[dict]]:
        """Groups of >=2 identical charges (same merchant and amount) within `within_days` of each other."""
        txns = self.find_transactions(customer_id, merchant=merchant, amount=amount)
        groups: list[list[dict]] = []
        for t in txns:
            for g in groups:
                head = g[0]
                if (
                    head["merchant"] == t["merchant"]
                    and abs(head["amount"] - t["amount"]) < 0.01
                    and abs((t["txn_date"] - head["txn_date"]).days) <= within_days
                ):
                    g.append(t)
                    break
            else:
                groups.append([t])
        return [g for g in groups if len(g) >= 2]

    def refunds_for(self, customer_id: str, merchant: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM transactions WHERE customer_id = ? AND amount < 0 AND lower(merchant) LIKE ? ORDER BY txn_date",
            [customer_id, f"%{merchant.lower()}%"],
        )

    def customer_history(self, customer_id: str, as_of: date) -> dict:
        cust = self._rows("SELECT * FROM customers WHERE customer_id = ?", [customer_id])
        disputes = self._rows(
            "SELECT dispute_type, opened_on, outcome FROM disputes WHERE customer_id = ? AND opened_on >= ?",
            [customer_id, as_of - timedelta(days=365)],
        )
        return {
            "known_customer": bool(cust),
            "account_age_days": (as_of - cust[0]["opened_on"]).days if cust else None,
            "disputes_last_12m": len(disputes),
            "unauthorized_disputes_last_12m": sum(d["dispute_type"] == "unauthorized" for d in disputes),
        }

    def merchants(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT merchant FROM transactions").fetchall()]
