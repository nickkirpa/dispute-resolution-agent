"""Streamlit demo of the Dispute Resolution Agent.

    uv sync                      (add --extra router for the fine-tuned router)
    uv run streamlit run app/main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Dispute Resolution Agent", page_icon="⚖️", layout="wide")

from app.views import customer, metrics, officer, policy, trace  # noqa: E402

pages = st.navigation([
    st.Page(customer.render, title="File a dispute", icon="🧾", default=True),
    st.Page(officer.render, title="Review queue", icon="🧑‍⚖️", url_path="review"),
    st.Page(trace.render, title="Case trace", icon="🔍", url_path="trace"),
    st.Page(policy.render, title="Policy v1 vs v2", icon="📜", url_path="policy"),
    st.Page(metrics.render, title="Results", icon="📊", url_path="results"),
])
pages.run()
