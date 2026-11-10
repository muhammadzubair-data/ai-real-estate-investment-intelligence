"""02 · AI Property Valuation Engine."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, STATUS, TYPE_ORDER, badge, footer, gbp, kpi, load_json, pct, sales, setup, valuation_service

setup("AI Property Valuation Engine",
      "Estimate today's value from location, type and recent local sales, with comparables, an 80% range and the reasons behind it.")
badge()

svc = valuation_service()
s = sales()
met = load_json("metrics.json")

mode = st.radio("Value", ["A recently sold property", "Any postcode in the covered cities"], horizontal=True)
if mode == "A recently sold property":
    test = s[s.split == "test"].sort_values("sale_date", ascending=False)
    c1, c2 = st.columns([1, 2])
    city = c1.selectbox("City", sorted(test.city.unique()))
    sub = test[test.city == city].head(400)
    lab = {r.transaction_id: f"{r.postcode} · {r.property_type} · {r.tenure} · sold {r.sale_date:%b %Y} for {gbp(r.price, False)}"
           for r in sub.itertuples()}
    tid = c2.selectbox("Property (latest 400 sales)", sub.transaction_id, format_func=lab.get)
    r = sub.set_index("transaction_id").loc[tid]
    postcode, ptype, tenure, newb, pid = r.postcode, r.property_type, r.tenure, bool(r.new_build), r.property_id
    actual = (r.price, r.sale_date, np.exp(r.log_pred), np.exp(r.log_lo), np.exp(r.log_hi))
else:
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 0.8])
    postcode = c1.text_input("Postcode", value=s.postcode.iloc[-1]).strip().upper()
    ptype = c2.selectbox("Property type", TYPE_ORDER, index=2)
    tenure = c3.selectbox("Tenure", ["Freehold", "Leasehold"], index=1 if ptype == "Flat/maisonette" else 0)
    newb = c4.checkbox("New build")
    pid, actual = None, None

try:
    v = svc.value(postcode, ptype, tenure, newb, property_id=pid)
except KeyError as e:
    st.error(str(e).strip("'") + ". Try a postcode from the list on the Geospatial page.")
    st.stop()

st.divider()
c = st.columns(4)
kpi(c[0], f"Estimated value ({v['as_of'] + pd.offsets.MonthBegin(1):%b %Y})", gbp(v["estimate"], False),
    f"{ptype}, {tenure.lower()}{', new build' if newb else ''}")
kpi(c[1], "80% range", f"{gbp(v['low'])} – {gbp(v['high'])}",
    f"tested on unseen months: {pct(met['temporal_holdout']['LightGBM (deployed)']['interval_coverage'], 0)} of prices fell inside")
f = v["features"]
kpi(c[2], "Comparable evidence", f"{int(f.sector_type_count)} sector sales",
    f"10 nearest same-type sales within {f.comp_max_km:.1f} km" if pd.notna(f.comp_max_km) else "no nearby comparables")
city_slice = {d["city"]: d for d in met["slices"]["city"]}.get(f.city, {})
kpi(c[3], f"Typical error in {f.city}", pct(city_slice.get("median_ape")),
    "median % error on the latest months (out of sample)")
if actual:
    st.markdown(f'<span class="pill">Actual sale: {gbp(actual[0], False)} in {actual[1]:%b %Y}</span>'
                f'<span class="pill">Model estimate at the time (before the sale was known): {gbp(actual[2], False)} '
                f'({pct(actual[2] / actual[0] - 1)} vs price) · range {gbp(actual[3])}–{gbp(actual[4])}</span>',
                unsafe_allow_html=True)

for w in v["warnings"]:
    st.markdown(f'<div class="warn">{w}</div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Why this estimate")
    d = v["drivers"].head(10).iloc[::-1]

    def fmt(r):
        if r.feature.startswith("log_") and pd.notna(r.value):
            return f"{r.label}: {gbp(np.exp(r.value))}"
        if pd.isna(r.value):
            return f"{r.label}: none"
        return f"{r.label}: {r.value:,.2f}" if isinstance(r.value, float) else f"{r.label}: {r.value}"
    pct_effect = np.exp(d.shap) - 1
    fig = go.Figure(go.Bar(x=pct_effect, y=[fmt(r) for r in d.itertuples()], orientation="h",
                           marker_color=[SERIES[0] if x > 0 else STATUS["critical"] for x in pct_effect],
                           hovertemplate="%{y}<br>%{x:+.1%} on the estimate<extra></extra>"))
    fig.update_layout(height=420, xaxis=dict(tickformat="+.0%", title="Effect on the estimate vs the average property"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Starting point: the model's average prediction ({gbp(v['base_value'])}). Each bar shows how one "
               "input moved the estimate up (blue) or down (red). These describe the model, not causes of value.")

with right:
    st.subheader("Comparable sales")
    comps = v["comparables"]
    if comps.empty:
        st.info("No same-type sales nearby in the last 12 months.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scattermap(lat=comps.latitude, lon=comps.longitude, mode="markers",
                                    marker=dict(size=11, color=SERIES[1]), name="Comparable",
                                    text=[f"{r.postcode} · {gbp(r.price, False)} ({r.sale_date:%b %Y})<br>"
                                          f"today ≈ {gbp(r.adjusted_price, False)} · {r.distance_km:.2f} km"
                                          for r in comps.itertuples()], hoverinfo="text"))
        fig.add_trace(go.Scattermap(lat=[f.latitude], lon=[f.longitude], mode="markers",
                                    marker=dict(size=16, color=SERIES[0]), name="This property",
                                    hovertext=[postcode], hoverinfo="text"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                          map=dict(style="carto-positron", zoom=12.5,
                                   center=dict(lat=float(f.latitude), lon=float(f.longitude))),
                          legend=dict(y=0.02, x=0.02, bgcolor="rgba(255,255,255,0.85)"))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(comps[["rank", "postcode", "sale_date", "price", "adjusted_price", "distance_km", "tenure"]],
                     hide_index=True, width="stretch", height=250,
                     column_config={"rank": "#", "sale_date": st.column_config.DateColumn("Sold", format="MMM YYYY"),
                                    "price": st.column_config.NumberColumn("Price", format="£%,.0f"),
                                    "adjusted_price": st.column_config.NumberColumn("Market-adjusted", format="£%,.0f"),
                                    "distance_km": st.column_config.NumberColumn("km", format="%.2f")})
        st.caption("Market-adjusted = price brought forward to today with the city index for this property type. "
                   f"Median of the 10: {gbp(comps.adjusted_price.median(), False)}.")
footer()
