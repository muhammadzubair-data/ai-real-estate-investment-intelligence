"""03 · Investment Opportunity Finder."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, STATUS, TYPE_COLORS, TYPE_ORDER, badge, footer, kpi, pct, sales, setup
from reintel.finance.underwriting import Assumptions, screen

setup("Investment Opportunity Finder",
      "Rank properties by cash flow, yield, affordability, price against the model's value and risk.")
badge()
st.markdown('<div class="warn">There are no live listings in public data. The candidate pool is the latest six months '
            'of real sales, each treated <b>as if offered at the price it sold for</b>. The model valued every one of '
            'them <b>before</b> its sale was known, so the price-vs-estimate gap is a fair out-of-sample signal. '
            'Rents are area averages for the property type, not property-specific rents.</div>',
            unsafe_allow_html=True)

s = sales()
pool = s[(s.split == "test") & s.area_rent.notna()].copy()

with st.sidebar:
    st.markdown("### Financing & costs")
    ltv = st.slider("Loan-to-value", 0.0, 0.85, 0.75, 0.05)
    rate = st.slider("Mortgage rate", 0.0, 0.10, 0.055, 0.0025, format="%.4f")
    io = st.toggle("Interest-only mortgage", value=True)
    vac = st.slider("Vacancy", 0.0, 0.25, 0.05, 0.01)
    mgmt = st.slider("Letting & management (% of rent collected)", 0.0, 0.20, 0.10, 0.01)
    maint = st.slider("Maintenance (% of gross rent)", 0.0, 0.15, 0.05, 0.01)
    st.caption("Insurance £400/yr, other £200/yr, reserves 2% of rent, legal £2,500, mortgage fee £1,500, SDLT with "
               "the 5% additional-property surcharge.")
a = Assumptions(loan_to_value=ltv, mortgage_rate=rate, interest_only=io, vacancy_rate=vac,
                management_pct=mgmt, maintenance_pct=maint)

f1, f2, f3, f4 = st.columns([1.6, 1.4, 1.2, 1])
cities = f1.multiselect("Cities", sorted(pool.city.unique()), placeholder="All cities")
types = f2.multiselect("Property types", TYPE_ORDER, placeholder="All types")
lo, hi = f3.slider("Price range (£k)", 50, 1000, (75, 350), 25)
max_cash = f4.number_input("Max cash to invest (£)", 10_000, 1_000_000, 80_000, 5_000)

if cities:
    pool = pool[pool.city.isin(cities)]
if types:
    pool = pool[pool.property_type.isin(types)]
pool = pool[(pool.price >= lo * 1000) & (pool.price <= hi * 1000)]
if pool.empty:
    st.info("No properties match these filters.")
    st.stop()

m = screen(pool.price.to_numpy(), pool.area_rent.to_numpy(), a)
for k, v in m.items():
    pool[k] = v
pool["estimate"] = np.exp(pool.log_pred)
pool["est_low"], pool["est_high"] = np.exp(pool.log_lo), np.exp(pool.log_hi)
pool["value_gap"] = pool.estimate / pool.price - 1
pool["range_width"] = pool.est_high / pool.est_low - 1
pool["thin_evidence"] = (pool.sector_type_count < 5) | (pool.comp_max_km.fillna(99) > 2)
pool = pool[pool.cash_invested <= max_cash]
if pool.empty:
    st.info("Nothing fits within that cash budget. Raise the budget or the loan-to-value.")
    st.stop()

# composite score: percentile ranks within the filtered pool
r = lambda x: x.rank(pct=True)
pool["score"] = 100 * (0.35 * r(pool.cash_on_cash) + 0.25 * r(pool.value_gap) + 0.20 * r(pool.dscr.clip(upper=5))
                       + 0.10 * r(-pool.range_width) + 0.10 * r(-pool.cash_invested))
pool.loc[pool.thin_evidence, "score"] -= 10
pool.loc[pool.cash_flow < 0, "score"] -= 20
pool["flags"] = pd.Series(np.where(pool.cash_flow < 0, "negative cash flow; ", ""), index=pool.index) + (
                 np.where(pool.cash_flow < 0, "", "")
                 + np.where(pool.dscr < 1.25, "DSCR under 1.25; ", "")
                 + np.where(pool.thin_evidence, "thin local evidence; ", "")
                 + np.where(pool.range_width > 0.6, "wide value range; ", "")
                 + np.where(pool.new_build == 1, "new build; ", ""))
pool["flags"] = pool["flags"].str.rstrip("; ")

c = st.columns(4)
kpi(c[0], "Candidates", f"{len(pool):,}", "after filters and cash budget")
kpi(c[1], "Positive cash flow", pct((pool.cash_flow > 0).mean(), 0), "year one, pre-tax")
kpi(c[2], "Median cash-on-cash", pct(pool.cash_on_cash.median()))
kpi(c[3], "Median gross yield", pct(pool.gross_yield.median()))

st.write("")
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Cash return vs price against estimate")
    fig = go.Figure()
    for t in TYPE_ORDER:
        g = pool[pool.property_type == t]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g.value_gap, y=g.cash_on_cash, mode="markers", name=t,
            marker=dict(size=8, color=TYPE_COLORS[t], opacity=0.75, line=dict(color="#fcfcfb", width=1)),
            customdata=np.stack([g.postcode, g.price, g.estimate, g.city], axis=1),
            hovertemplate="%{customdata[0]}, %{customdata[3]}<br>price £%{customdata[1]:,.0f} · estimate "
                          "£%{customdata[2]:,.0f}<br>gap %{x:+.0%} · cash-on-cash %{y:.1%}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#898781", width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color="#898781", width=1, dash="dot"))
    fig.update_layout(height=420, xaxis=dict(title="Model estimate vs price (right = priced below estimate)",
                                             tickformat="+.0%"),
                      yaxis=dict(title="Year-one cash-on-cash", tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Cash flow by city")
    cc = pool.groupby("city").agg(n=("price", "size"), cf=("cash_flow", "median"),
                                  coc=("cash_on_cash", "median"), gy=("gross_yield", "median")).reset_index()
    cc = cc.sort_values("cf")
    fig = go.Figure(go.Bar(x=cc.cf, y=cc.city, orientation="h",
                           marker_color=[SERIES[0] if v >= 0 else STATUS["critical"] for v in cc.cf],
                           customdata=np.stack([cc.n, cc.coc, cc.gy], axis=1),
                           hovertemplate="%{y}: £%{x:,.0f}/yr median<br>%{customdata[0]} candidates · CoC "
                                         "%{customdata[1]:.1%} · gross yield %{customdata[2]:.1%}<extra></extra>"))
    fig.update_layout(height=420, xaxis_title="Median year-one cash flow (£ / year)")
    st.plotly_chart(fig, width="stretch")

st.subheader("Ranked shortlist")
st.caption("Score (0–100) blends cash-on-cash 35%, price below estimate 25%, DSCR 20%, a tight value range 10% and "
           "lower cash needed 10%, with penalties for negative cash flow and thin local evidence. Use it to decide "
           "what to look at, not what to buy: size, condition and the real asking rent are unknown here.")
top = pool.sort_values("score", ascending=False).head(100)
st.dataframe(top[["score", "postcode", "city", "property_type", "tenure", "price", "estimate", "value_gap",
                  "area_rent", "gross_yield", "cash_flow", "cash_on_cash", "dscr", "cash_invested", "flags"]],
             hide_index=True, width="stretch", height=420, column_config={
                 "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                 "property_type": "Type", "price": st.column_config.NumberColumn("Price", format="£%,.0f"),
                 "estimate": st.column_config.NumberColumn("Model estimate", format="£%,.0f"),
                 "value_gap": st.column_config.NumberColumn("Est. vs price", format="percent"),
                 "area_rent": st.column_config.NumberColumn("Area rent / mo", format="£%,.0f"),
                 "gross_yield": st.column_config.NumberColumn("Gross yield", format="percent"),
                 "cash_flow": st.column_config.NumberColumn("Cash flow / yr", format="£%,.0f"),
                 "cash_on_cash": st.column_config.NumberColumn("Cash-on-cash", format="percent"),
                 "dscr": st.column_config.NumberColumn("DSCR", format="%.2f"),
                 "cash_invested": st.column_config.NumberColumn("Cash needed", format="£%,.0f"),
                 "flags": "Risk flags"})
st.download_button("Download shortlist (CSV)", top.to_csv(index=False), "shortlist.csv", "text/csv")
footer()
