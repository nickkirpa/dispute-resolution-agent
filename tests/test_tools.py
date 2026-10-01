from dispute_agent.config import Settings
from dispute_agent.state import Claim, DisputeType
from dispute_agent.tools import KnowledgeBase, compute_refund
from eval.fixtures import build_fixture_ledger


def test_duplicates_found_within_3_days():
    led = build_fixture_ledger()
    groups = led.find_duplicates("C001", "SpotiTunes", 9.99)
    assert len(groups) == 1 and len(groups[0]) == 2


def test_monthly_subscription_is_not_duplicate():
    led = build_fixture_ledger()
    assert led.find_duplicates("C002", "Netflux", 15.99) == []


def test_refund_keeps_one_duplicate_charge():
    led = build_fixture_ledger()
    groups = led.find_duplicates("C001", "SpotiTunes", 9.99)
    assert compute_refund(DisputeType.DUPLICATE_CHARGE, None, [], groups) == 9.99


def test_wrong_amount_refunds_difference_only():
    claim = Claim(amount=320.0, claimed_correct_amount=280.0)
    assert compute_refund(DisputeType.WRONG_AMOUNT, claim, [{"amount": 320.0}], []) == 40.0


def test_kb_loads_latest_version_and_filters_by_type():
    kb = KnowledgeBase(Settings().kb_dir)
    assert kb.version == "v1"
    hits = kb.search("charged twice duplicate payment", k=2, dispute_type="duplicate_charge")
    assert hits[0].clause_id == "POL-DUP-01"
    assert all({"duplicate_charge", "all"} & set(h.applies_to) for h in hits)


def test_customer_history_counts_unauthorized_disputes():
    from datetime import date

    hist = build_fixture_ledger().customer_history("C007", date(2026, 9, 30))
    assert hist["unauthorized_disputes_last_12m"] == 2


def test_claim_date_without_year_is_moved_to_recent_past():
    from datetime import date

    from dispute_agent.graph import normalize_claim_date

    as_of = date(2026, 9, 30)
    assert normalize_claim_date("2024-09-07", as_of) == "2026-09-07"  # model guessed the wrong year
    assert normalize_claim_date("2026-12-24", as_of) == "2025-12-24"  # future -> last year
    assert normalize_claim_date("2026-08-01", as_of) == "2026-08-01"  # plausible dates untouched
    assert normalize_claim_date("not a date", as_of) is None
