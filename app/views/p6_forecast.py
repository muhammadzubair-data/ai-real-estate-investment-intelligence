"""06 · Market Forecasting & Risk Center."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, STATUS, badge, footer, gbp, kpi, load, pct, setup

setup("Market Forecasting & Risk Center",
      "Price and rent trends, short-range forecasts with honest uncertainty, and how much a downturn would hurt.")
badge()

mkt = load("market_monthly.parquet")
rents = load("rents.parquet")
fc = load("forecasts.parquet")
risk = load("market_risk.csv")

city = st.selectbox("City", sorted(mkt.city.unique()))
pr = risk[(risk.city == city) & (risk.series == "price")]
rr = risk[(risk.city == city) & (risk.series == "rent")]

c = st.columns(4)
if len(pr):
    p = pr.iloc[0]
    kpi(c[0], "House prices, last 12 months", pct(p.yoy_change), "mix-adjusted index")
    kpi(c[1], "Price volatility", pct(p.annualised_volatility), "annualised, from monthly changes")
if len(rr):
    q = rr.iloc[0]
    kpi(c[2], "Rents, last 12 months", pct(q.yoy_change), "ONS average private rent")
    kpi(c[3], "Largest rent fall in the record", pct(q.max_drawdown), "peak to trough, monthly series")


def chart(hist_x, hist_y, f, title, ytitle, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, mode="lines", name="History", line=dict(color=color, width=2),
                             hovertemplate="%{x|%b %Y}: £%{y:,.0f}<extra></extra>"))
    if len(f):
        fig.add_trace(go.Scatter(x=list(f.month) + list(f.month[::-1]), y=list(f.upper) + list(f.lower[::-1]),
                                 fill="toself", fillcolor="rgba(42,120,214,0.14)", line=dict(width=0), mode="lines",
                                 name="80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[hist_x.iloc[-1], *f.month], y=[hist_y.iloc[-1], *f.forecast], mode="lines",
                                 name="Forecast", line=dict(color=color, width=2, dash="dash"),
                                 hovertemplate="%{x|%b %Y}: £%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=340, yaxis_title=ytitle, legend=dict(y=-0.18, yanchor="top"))
    return fig


left, right = st.columns(2)
with left:
    h = mkt[mkt.city == city].dropna(subset=["mix_adjusted_price"]).sort_values("month")
    f = fc[(fc.city == city) & (fc.series == "price")]
    st.subheader("House prices · 6-month forecast")
    st.plotly_chart(chart(h.month, h.mix_adjusted_price, f, "", "Mix-adjusted price (£)",
                          SERIES[0]), width="stretch")
    if len(pr):
        beat = bool(pr.iloc[0].beats_naive)
        st.caption(f"Damped-trend model, tuned by rolling backtest. One-month-ahead backtest error "
                   f"{pct(np.expm1(pr.iloc[0].backtest_mae_1m))} vs {pct(np.expm1(pr.iloc[0].naive_mae_1m))} for "
                   f"'no change'. " + ("It beats 'no change' on this city's history." if beat else
                                       "**It does not beat 'no change' here, so treat the forecast as flat with a range.**"))
with right:
    rh = rents[(rents.city == city) & (rents.property_type == "All")].sort_values("month")
    f = fc[(fc.city == city) & (fc.series == "rent")]
    st.subheader("Rents · 12-month forecast")
    st.plotly_chart(chart(rh.month, rh.rent, f, "", "Average rent (£ per month)", SERIES[1]),
                    width="stretch")
    if len(rr):
        beat = bool(rr.iloc[0].beats_naive)
        st.caption("Rents move smoothly, so short forecasts are steadier than for prices. "
                   + ("The model beats 'no change' on this city's history." if beat else
                      "**The model does not beat 'no change' here.**"))

st.markdown('<div class="warn">The price history is short (about two years of monthly data), so forecasts are a guide '
            'to direction and uncertainty, not a prediction. They do not know about interest-rate decisions, tax '
            'changes or shocks. Ranges come from the method\'s own past errors on this data.</div>',
            unsafe_allow_html=True)

st.divider()
st.subheader("Downside: what a price fall does to your equity")
c1, c2, c3 = st.columns(3)
last_price = float(h.mix_adjusted_price.iloc[-1]) if len(h) else 200_000
price = c1.number_input("Purchase price (£)", 10_000, 5_000_000, int(round(last_price, -3)), 5_000)
ltv = c2.slider("Loan-to-value", 0.0, 0.9, 0.75, 0.05)
costs = c3.slider("Selling costs", 0.0, 0.05, 0.02, 0.005)
falls = np.array([0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30])
loan = price * ltv
deposit = price - loan
equity = price * (1 + falls) * (1 - costs) - loan
fig = go.Figure(go.Bar(x=[f"{int(x * 100)}%" if x else "No change" for x in falls], y=equity,
                       marker_color=[SERIES[0] if e > 0 else STATUS["critical"] for e in equity],
                       text=[f"{e / deposit:+.0%} of deposit" if deposit > 0 else "" for e in equity - deposit],
                       textposition="outside", hovertemplate="Price change %{x}<br>equity £%{y:,.0f}<extra></extra>"))
fig.update_layout(height=340, xaxis=dict(title="House price change", type="category"),
                  yaxis_title="Equity after selling (£)")
st.plotly_chart(fig, width="stretch")
st.caption(f"Deposit {gbp(deposit, False)}, loan {gbp(loan, False)}. Interest-only loan, so the balance does not fall. "
           "Labels show the gain or loss against the deposit put in.")

st.subheader("Risk across markets")
t = risk[risk.series == "price"][["city", "latest", "yoy_change", "annualised_volatility", "max_drawdown",
                                  "worst_3m_change", "beats_naive"]].merge(
    risk[risk.series == "rent"][["city", "yoy_change"]].rename(columns={"yoy_change": "rent_yoy"}), on="city", how="left")
st.dataframe(t.sort_values("annualised_volatility"), hide_index=True, width="stretch", column_config={
    "city": "City",
    "latest": st.column_config.NumberColumn("Index level", format="£%,.0f"),
    "yoy_change": st.column_config.NumberColumn("Price yoy", format="percent"),
    "annualised_volatility": st.column_config.NumberColumn("Volatility (ann.)", format="percent"),
    "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
    "worst_3m_change": st.column_config.NumberColumn("Worst 3 months", format="percent"),
    "beats_naive": st.column_config.CheckboxColumn("Forecast beats 'no change'"),
    "rent_yoy": st.column_config.NumberColumn("Rent yoy", format="percent")})
footer()
