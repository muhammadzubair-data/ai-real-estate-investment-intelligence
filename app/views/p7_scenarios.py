"""07 · Investment Scenario Simulator."""
import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, TYPE_ORDER, badge, footer, gbp, load, pct, sales, setup
from reintel.finance.underwriting import SCENARIO_DELTAS, Assumptions, apply_scenario, scenarios, sensitivity

setup("Investment Scenario Simulator",
      "Edit the base case, upside and downside, and compare cash flow, returns and risk side by side.")
badge()

s = sales()
rents = load("rents.parquet")
last = s.sale_date.max()

with st.sidebar:
    st.markdown("### Base case")
    city = st.selectbox("City", sorted(s.city.unique()))
    ptype = st.selectbox("Property type", TYPE_ORDER, index=2)
    w = s[(s.city == city) & (s.property_type == ptype) & (s.sale_date > last - pd.DateOffset(months=12))]
    lr = rents[(rents.city == city) & (rents.property_type == ptype)].sort_values("month")
    d_price = int(round(w.price.median(), -3)) if len(w) else 200_000
    d_rent = int(round(lr.rent.iloc[-1], -1)) if len(lr) else 1_000
    price = st.number_input("Purchase price (£)", 10_000, 5_000_000, d_price, 1_000)
    rent = st.number_input("Monthly rent (£)", 0, 50_000, d_rent, 10)
    ltv = st.slider("Loan-to-value", 0.0, 0.85, 0.75, 0.05)
    io = st.toggle("Interest-only", value=True)
    hold = st.slider("Hold (years)", 1, 25, 5)
    st.caption(f"Pre-filled with the {city} median price (last 12 months) and the ONS rent for this type.")

base = Assumptions(purchase_price=price, monthly_rent=rent, loan_to_value=ltv, interest_only=io, hold_years=hold,
                   service_charge_ground_rent=1_500 if ptype == "Flat/maisonette" else 0)

st.subheader("Scenario assumptions")
st.caption("Edit any cell. Base values are absolute; upside and downside are written as the change from base "
           "and recalculated live.")
fields = [("vacancy_rate", "Vacancy", 100), ("mortgage_rate", "Mortgage rate", 100),
          ("rent_growth", "Rent growth / yr", 100), ("price_growth", "House price growth / yr", 100),
          ("cost_inflation", "Cost inflation / yr", 100), ("selling_costs_pct", "Selling costs at exit", 100)]
grid = pd.DataFrame({
    "Assumption (%)": [f[1] for f in fields],
    "Base": [getattr(base, f[0]) * f[2] for f in fields],
    "Upside (change)": [SCENARIO_DELTAS["Upside"].get(f[0], 0) * f[2] for f in fields],
    "Downside (change)": [SCENARIO_DELTAS["Downside"].get(f[0], 0) * f[2] for f in fields],
})
ed = st.data_editor(grid, hide_index=True, width="stretch", disabled=["Assumption (%)"], key="grid",
                    column_config={c: st.column_config.NumberColumn(format="%.2f", step=0.25)
                                   for c in ["Base", "Upside (change)", "Downside (change)"]})
base = replace(base, **{f[0]: float(v) / f[2] for f, v in zip(fields, ed["Base"])})
deltas = {"Base": {},
          "Upside": {f[0]: float(v) / f[2] for f, v in zip(fields, ed["Upside (change)"])},
          "Downside": {f[0]: float(v) / f[2] for f, v in zip(fields, ed["Downside (change)"])}}
try:
    res = scenarios(base, deltas)
except ValueError as e:
    st.error(f"Check the assumptions: {e}")
    st.stop()

colors = {"Downside": "#e34948", "Base": SERIES[0], "Upside": SERIES[2]}
st.write("")
c = st.columns(3)
for col, name in zip(c, ["Downside", "Base", "Upside"]):
    r = res[name]
    col.markdown(
        f'<div class="kpi"><div class="lbl"><span style="color:{colors[name]}">■</span> {name}</div>'
        f'<div class="val">{pct(r.irr)} IRR</div>'
        f'<div class="sub">Year-one cash flow {gbp(r.cash_flow, False)} · cash-on-cash {pct(r.cash_on_cash)}<br>'
        f'DSCR {"all cash" if np.isinf(r.dscr) else f"{r.dscr:.2f}"} · equity at exit {gbp(r.exit_equity)} · '
        f'{r.equity_multiple:.2f}× money back</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns(2)
with left:
    st.subheader("Annual cash flow")
    fig = go.Figure()
    for name in ["Downside", "Base", "Upside"]:
        y = pd.DataFrame(res[name].yearly)
        fig.add_trace(go.Bar(x=y.year, y=y.cash_flow, name=name, marker_color=colors[name],
                             marker_line=dict(color="#fcfcfb", width=2),
                             hovertemplate=name + " · year %{x}: £%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#898781", width=1))
    fig.update_layout(barmode="group", height=360, xaxis=dict(title="Year", dtick=1), yaxis_title="£ per year (pre-tax)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Operating cash flow only; the sale proceeds at exit are in the IRR and equity figures above.")
with right:
    st.subheader("What moves the base-case IRR most")
    sens = sensitivity(base, "irr")
    sd = pd.DataFrame(sens).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sd.driver, x=sd.low - sd.base, base=sd.base, orientation="h", name="Lower input",
                         marker_color=SERIES[1], hovertemplate="%{y}: IRR %{x:.1%} (lower input)<extra></extra>"))
    fig.add_trace(go.Bar(y=sd.driver, x=sd.high - sd.base, base=sd.base, orientation="h", name="Higher input",
                         marker_color=SERIES[0], hovertemplate="%{y}: IRR %{x:.1%} (higher input)<extra></extra>"))
    fig.add_vline(x=float(sd.base.iloc[0]), line=dict(color="#0b0b0b", width=1))
    fig.update_layout(barmode="overlay", height=360, xaxis=dict(tickformat=".0%", title="IRR"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each bar changes one input at a time from the base case and holds the rest fixed.")

st.subheader("Side by side")
keys = [("sdlt", "Stamp duty", "£"), ("cash_invested", "Cash invested", "£"), ("gross_yield", "Gross yield", "%"),
        ("noi", "NOI (year 1)", "£"), ("debt_service", "Debt service (year 1)", "£"),
        ("cash_flow", "Cash flow (year 1)", "£"), ("cap_rate", "Cap rate", "%"), ("cash_on_cash", "Cash-on-cash", "%"),
        ("dscr", "DSCR", "x"), ("break_even_occupancy", "Break-even occupancy", "%"), ("exit_value", "Sale value", "£"),
        ("exit_equity", "Equity after sale", "£"), ("irr", "IRR", "%"), ("equity_multiple", "Equity multiple", "x")]


def show(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float) and np.isinf(v):
        return "all cash"
    return gbp(v, False) if kind == "£" else pct(v) if kind == "%" else f"{v:.2f}×"


table = pd.DataFrame({"Metric": [k[1] for k in keys],
                      **{n: [show(getattr(res[n], k[0]), k[2]) for k in keys] for n in ["Downside", "Base", "Upside"]}})
st.dataframe(table, hide_index=True, width="stretch")

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "currency": "GBP",
    "market": {"city": city, "property_type": ptype},
    "scenarios": {n: {"assumptions": apply_scenario(base, deltas[n]).to_dict(),
                      "results": {k: v for k, v in res[n].to_dict().items() if k != "yearly"},
                      "yearly": res[n].yearly} for n in res},
    "note": "Pre-tax illustration from user assumptions. Not investment advice.",
}
c1, c2 = st.columns(2)
c1.download_button("Download scenario report (JSON)", json.dumps(report, indent=2, default=float),
                   "scenario_report.json", "application/json")
c2.download_button("Download year-by-year (CSV)",
                   pd.concat([pd.DataFrame(res[n].yearly).assign(scenario=n) for n in res]).to_csv(index=False),
                   "scenario_cash_flows.csv", "text/csv")
footer()
