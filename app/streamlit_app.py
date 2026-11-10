"""AI Real Estate Investment & Property Valuation Intelligence - Streamlit entry point.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Real Estate Intelligence", page_icon="🏘️", layout="wide")

from common import manifest, require_data  # noqa: E402

require_data()

V = Path(__file__).parent / "views"
nav = st.navigation({
    "Market": [
        st.Page(V / "p1_market.py", title="Executive Market Overview", icon="📊", default=True),
        st.Page(V / "p5_geo.py", title="Geospatial Intelligence", icon="🗺️"),
        st.Page(V / "p6_forecast.py", title="Forecasting & Risk", icon="📈"),
    ],
    "Property": [
        st.Page(V / "p2_valuation.py", title="AI Valuation Engine", icon="🏠"),
        st.Page(V / "p3_opportunities.py", title="Opportunity Finder", icon="🔎"),
    ],
    "Investment": [
        st.Page(V / "p4_cashflow.py", title="Rental Yield & Cash Flow", icon="💷"),
        st.Page(V / "p7_scenarios.py", title="Scenario Simulator", icon="⚖️"),
    ],
})

with st.sidebar:
    m = manifest()
    st.markdown("**Real Estate Intelligence**")
    st.caption(f"{m['counts']['sales']:,} real sales · {m['counts']['properties']:,} properties · "
               f"{m['counts']['markets']} English cities · {m['date_range'][0]} to {m['date_range'][1]}")
    st.caption("Decision support only. Not financial or investment advice.")

nav.run()
