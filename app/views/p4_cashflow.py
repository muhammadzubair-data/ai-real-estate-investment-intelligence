"""04 · Rental Yield & Cash Flow Analytics."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SEQ, SERIES, STATUS, TYPE_ORDER, badge, footer, gbp, kpi, load, pct, sales, setup
from reintel.finance.underwriting import SDLT_BANDS, Assumptions, analyse

setup("Rental Yield & Cash Flow Analytics",
      "Where gross yields are highest, and what a single purchase earns once costs and the mortgage are paid.")
badge()

s = sales()
rents = load("rents.parquet")
last = s.sale_date.max()
win = s[s.sale_date > last - pd.DateOffset(months=12)]
lr = rents.sort_values("month").groupby(["city", "property_type"]).tail(1)

# --- area yields ----------------------------------------------------------------------------
st.subheader("Gross yield by city and property type")
med = win.groupby(["city", "property_type"]).price.median().rename("price").reset_index()
y = med.merge(lr[["city", "property_type", "rent"]], on=["city", "property_type"], how="left")
y["gross_yield"] = y.rent * 12 / y.price
piv = y.pivot(index="city", columns="property_type", values="gross_yield").reindex(columns=TYPE_ORDER)
piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, xgap=2, ygap=2,
                           colorscale=[[0, SEQ[0]], [0.5, SEQ[3]], [1, SEQ[6]]],
                           text=np.vectorize(lambda v: "–" if pd.isna(v) else f"{v:.1%}")(piv.values),
                           texttemplate="%{text}", colorbar=dict(title="Gross yield", tickformat=".0%"),
                           hovertemplate="%{y} · %{x}: %{z:.2%}<extra></extra>"))
fig.update_layout(height=max(280, 70 + 36 * len(piv)), yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("ONS average rent for the type × 12 ÷ median sale price of that type over the last 12 months. "
           "Area averages: a specific property's rent and price can differ a lot. Gross yield ignores every cost.")

st.divider()
st.subheader("Deal calculator")
c1, c2, c3 = st.columns(3)
city = c1.selectbox("Pre-fill from city", sorted(s.city.unique()))
ptype = c2.selectbox("Property type", TYPE_ORDER, index=2)
row = y[(y.city == city) & (y.property_type == ptype)]
d_price = int(round(row.price.iloc[0], -3)) if len(row) and pd.notna(row.price.iloc[0]) else 200_000
d_rent = int(round(row.rent.iloc[0], -1)) if len(row) and pd.notna(row.rent.iloc[0]) else 1_000
c3.markdown(f"<br><span class='note'>Pre-filled with the {city} median price and ONS rent for this type.</span>",
            unsafe_allow_html=True)

with st.form("deal"):
    a1, a2, a3, a4 = st.columns(4)
    price = a1.number_input("Purchase price (£)", 10_000, 5_000_000, d_price, 1_000)
    rent = a1.number_input("Monthly rent (£)", 0, 50_000, d_rent, 10)
    vac = a1.slider("Vacancy", 0.0, 0.30, 0.05, 0.01)
    add = a1.toggle("Buyer already owns a home (SDLT +5%)", value=True)
    ltv = a2.slider("Loan-to-value", 0.0, 0.85, 0.75, 0.05)
    rate = a2.number_input("Mortgage rate (%)", 0.0, 20.0, 5.5, 0.05) / 100
    io = a2.toggle("Interest-only", value=True)
    term = a2.number_input("Term (years)", 1, 40, 25)
    fee = a2.number_input("Mortgage fee (£)", 0, 20_000, 1_500, 100)
    mgmt = a3.slider("Management (% of rent collected)", 0.0, 0.20, 0.10, 0.01)
    maint = a3.slider("Maintenance (% of gross rent)", 0.0, 0.15, 0.05, 0.01)
    ins = a3.number_input("Insurance (£/yr)", 0, 10_000, 400, 50)
    svc = a3.number_input("Service charge + ground rent (£/yr)", 0, 20_000,
                          1_500 if ptype == "Flat/maisonette" else 0, 100)
    legal = a4.number_input("Legal & survey (£)", 0, 50_000, 2_500, 100)
    refurb = a4.number_input("Refurbishment (£)", 0, 500_000, 0, 500)
    hold = a4.slider("Hold (years)", 1, 25, 5)
    rg = a4.number_input("Rent growth (%/yr)", -10.0, 15.0, 3.0, 0.5) / 100
    pg = a4.number_input("Price growth (%/yr)", -15.0, 15.0, 2.0, 0.5) / 100
    st.form_submit_button("Calculate", type="primary")

a = Assumptions(purchase_price=price, monthly_rent=rent, vacancy_rate=vac, additional_property=add,
                loan_to_value=ltv, mortgage_rate=rate, interest_only=io, mortgage_years=term, mortgage_fees=fee,
                management_pct=mgmt, maintenance_pct=maint, insurance=ins, service_charge_ground_rent=svc,
                legal_and_survey=legal, refurbishment=refurb, hold_years=hold, rent_growth=rg, price_growth=pg)
r = analyse(a)

c = st.columns(6)
kpi(c[0], "Gross yield", pct(r.gross_yield), "rent × 12 ÷ price")
kpi(c[1], "Cap rate", pct(r.cap_rate), "NOI ÷ total acquisition cost")
kpi(c[2], "Annual cash flow", gbp(r.cash_flow, False), "after debt service, before tax")
kpi(c[3], "Cash-on-cash", pct(r.cash_on_cash), f"on {gbp(r.cash_invested, False)} invested")
kpi(c[4], "DSCR", "all cash" if np.isinf(r.dscr) else f"{r.dscr:.2f}", "NOI ÷ debt service; lenders want ≥ 1.25")
kpi(c[5], f"IRR over {hold} years", pct(r.irr), f"equity multiple {r.equity_multiple:.2f}×")

st.write("")
left, right = st.columns([1.2, 1])
with left:
    st.markdown("**Year one: from rent to cash in your pocket**")
    steps = [("Gross rent", r.gross_rent, "absolute"),
             ("Vacancy", r.effective_rent - r.gross_rent, "relative"),
             ("Operating costs", -r.operating_expenses, "relative"),
             ("Reserves", -r.reserves, "relative"),
             ("NOI", r.noi, "total"),
             ("Mortgage", -r.debt_service, "relative"),
             ("Cash flow", r.cash_flow, "total")]
    fig = go.Figure(go.Waterfall(
        x=[x[0] for x in steps], y=[x[1] for x in steps], measure=[x[2] for x in steps],
        text=[gbp(x[1], False) for x in steps], textposition="outside",
        increasing=dict(marker=dict(color=SERIES[0])), decreasing=dict(marker=dict(color=STATUS["critical"])),
        totals=dict(marker=dict(color=SEQ[5])), connector=dict(line=dict(color="#c3c2b7", width=1)),
        hovertemplate="%{x}: £%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=380, yaxis_title="£ per year", showlegend=False)
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("**Money in at purchase**")
    st.dataframe(pd.DataFrame([
        ("Purchase price", price), ("Stamp Duty Land Tax", r.sdlt), ("Legal & survey", legal),
        ("Refurbishment", refurb), ("Total acquisition cost", r.acquisition_cost), ("Less: mortgage", -r.loan),
        ("Mortgage fee", fee if r.loan > 0 else 0), ("Cash you put in", r.cash_invested),
    ], columns=["Item", "£"]), hide_index=True, width="stretch",
        column_config={"£": st.column_config.NumberColumn(format="£%,.0f")})
    st.markdown(f"Monthly mortgage payment: **{gbp(r.monthly_payment, False)}** "
                f"({'interest-only' if io else 'repayment'}). Break-even occupancy: "
                f"**{pct(r.break_even_occupancy, 0)}** of the year let.")
    with st.expander("How SDLT is worked out"):
        st.markdown("England & Northern Ireland residential rates (gov.uk, from 1 April 2025): "
                    + ", ".join(f"{int(rate * 100)}% up to £{upper:,.0f}" if upper != float("inf")
                                else f"{int(rate * 100)}% above" for upper, rate in SDLT_BANDS)
                    + ". Plus 5 percentage points on every band if the buyer already owns a home (purchases of "
                      "£40,000+). Wales and Scotland use different taxes.")

st.markdown("**Year by year**")
yr = pd.DataFrame(r.yearly).rename(columns={
    "year": "Year", "gross_rent": "Gross rent", "effective_rent": "After vacancy", "operating_expenses": "Operating costs",
    "reserves": "Reserves", "noi": "NOI", "debt_service": "Mortgage", "cash_flow": "Cash flow", "sale_value": "Sale value",
    "selling_costs": "Selling costs", "loan_repaid": "Loan repaid", "sale_proceeds": "Net sale proceeds"})
st.dataframe(yr, hide_index=True, width="stretch",
             column_config={c_: st.column_config.NumberColumn(format="£%,.0f") for c_ in yr.columns if c_ != "Year"})
st.caption("Pre-tax. Excludes income tax (including the mortgage-interest restriction for individual landlords), "
           "capital gains tax and void periods longer than the vacancy rate. Not investment advice.")
footer()
