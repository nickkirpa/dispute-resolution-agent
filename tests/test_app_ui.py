"""Headless UI tests of every Streamlit page (rule brain, offline). Skipped when the app extra is not installed."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _page(view: str, db: str):
    def script(view, db):
        import sys
        from pathlib import Path

        root = Path.cwd()
        sys.path[:0] = [str(root), str(root / "src")]
        from app import core

        core.DB = Path(db)  # isolated checkpoint DB per test
        import importlib

        importlib.import_module(f"app.views.{view}").render()

    return AppTest.from_function(script, args=(view, db), default_timeout=120)


def _errors(at):
    return [e.value for e in at.exception]


def test_customer_files_a_refund_case(tmp_path):
    at = _page("customer", str(tmp_path / "c.sqlite")).run()
    assert not _errors(at)
    at.button[0].click().run()  # suggested complaint for C001 (duplicate 9.99)
    assert not _errors(at)
    assert any("9.99 EUR" in m.value for m in at.markdown)


def test_escalated_case_reaches_the_review_queue_and_resumes(tmp_path):
    db = str(tmp_path / "c.sqlite")
    at = _page("customer", db).run()
    at.selectbox[0].select("C006").run()
    at.button[0].click().run()
    assert any("Review queue" in i.value for i in at.info)
    rq = _page("officer", db).run()
    assert not _errors(rq) and rq.selectbox[0].options
    rq.button[0].click().run()  # submit the default verdict (refund, proposed amount)
    assert not _errors(rq)
    assert any("899.00 EUR" in m.value for m in rq.markdown)


@pytest.mark.parametrize("view", ["trace", "policy", "metrics"])
def test_other_pages_render(tmp_path, view):
    at = _page(view, str(tmp_path / "c.sqlite")).run()
    assert not _errors(at)


def test_policy_page_shows_the_flip(tmp_path):
    at = _page("policy", str(tmp_path / "c.sqlite")).run()
    at.button[0].click().run()  # preset C013: refund under v1, reject under v2
    assert not _errors(at)
    badges = " ".join(m.value for m in at.markdown)
    assert "Refund" in badges and "Reject" in badges
