"""05 · Geospatial Property Intelligence."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import DIVERGING, SEQ, badge, footer, gbp, kpi, load, pct, sales, setup

setup("Geospatial Property Intelligence",
      "Where prices, growth, yields and valuation accuracy differ, neighbourhood by neighbourhood.")
badge()

s = sales()
rents = load("rents.parquet")
last = s.sale_date.max()

c1, c2, c3 = st.columns([1.2, 1.6, 1])
city = c1.selectbox("City", sorted(s.city.unique()))
metric = c2.radio("Colour neighbourhoods by", ["Median price", "Price change (yoy)", "Gross yield", "Valuation error"],
                  horizontal=True)
min_n = c3.slider("Minimum sales per neighbourhood", 5, 60, 20, 5,
                  help="Neighbourhoods with fewer sales are shown hollow: their figures are too noisy to trust.")

cs = s[s.city == city]
cur = cs[cs.sale_date > last - pd.DateOffset(months=12)]
prv = cs[(cs.sale_date <= last - pd.DateOffset(months=12)) & (cs.sale_date > last - pd.DateOffset(months=24))]
lr = rents.sort_values("month").groupby(["city", "property_type"]).tail(1)
rent_t = lr[lr.city == city].set_index("property_type").rent

g = cur.groupby("postcode_sector").agg(n=("price", "size"), median_price=("price", "median"),
                                      lat=("latitude", "mean"), lon=("longitude", "mean"))
g["prev_median"] = prv.groupby("postcode_sector").price.median()
g["yoy"] = g.median_price / g.prev_median - 1
cur = cur.assign(ty=cur.property_type.map(rent_t) * 12 / cur.price)
g["gross_yield"] = cur.groupby("postcode_sector").ty.median()
te = cs[(cs.split == "test")]
g["valuation_error"] = te.assign(e=np.abs(np.exp(te.log_pred) / te.price - 1)).groupby("postcode_sector").e.median()
g["reliable"] = g.n >= min_n
g = g.reset_index()

col, fmt, scale = {
    "Median price": ("median_price", lambda v: gbp(v), [[0, SEQ[0]], [0.5, SEQ[3]], [1, SEQ[6]]]),
    "Price change (yoy)": ("yoy", lambda v: pct(v), [[i / 4, c] for i, c in enumerate(DIVERGING)]),
    "Gross yield": ("gross_yield", lambda v: pct(v), [[0, SEQ[0]], [0.5, SEQ[3]], [1, SEQ[6]]]),
    "Valuation error": ("valuation_error", lambda v: pct(v), [[0, SEQ[0]], [0.5, SEQ[3]], [1, SEQ[6]]]),
}[metric]

c = st.columns(4)
kpi(c[0], "Neighbourhoods (postcode sectors)", f"{len(g)}", f"{g.reliable.sum()} with {min_n}+ sales in 12 months")
kpi(c[1], "Median price", gbp(cur.price.median()), "last 12 months")
kpi(c[2], "Cheapest reliable sector", gbp(g[g.reliable].median_price.min()))
kpi(c[3], "Dearest reliable sector", gbp(g[g.reliable].median_price.max()))

left, right = st.columns([1.6, 1])
with left:
    ok = g[g.reliable & g[col].notna()]
    weak = g[~g.reliable | g[col].isna()]
    vals = ok[col]
    if metric == "Price change (yoy)":
        lim = max(abs(vals.quantile(0.05)), abs(vals.quantile(0.95)), 0.02)
        cmin, cmax = -lim, lim
    else:
        cmin, cmax = vals.quantile(0.05), vals.quantile(0.95)
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=weak.lat, lon=weak.lon, mode="markers", name=f"Fewer than {min_n} sales",
        marker=dict(size=9, color="#c3c2b7", opacity=0.6),
        text=[f"{r.postcode_sector}: {r.n} sales (too few to trust)" for r in weak.itertuples()], hoverinfo="text"))
    fig.add_trace(go.Scattermap(
        lat=ok.lat, lon=ok.lon, mode="markers", name=metric,
        marker=dict(size=np.clip(6 + np.sqrt(ok.n) * 1.6, 8, 30), color=vals, colorscale=scale, cmin=cmin, cmax=cmax,
                    opacity=0.9, colorbar=dict(title=metric, thickness=10,
                                               tickformat=".0%" if metric != "Median price" else ",.0f")),
        text=[f"<b>{r.postcode_sector}</b><br>{r.n} sales · median {gbp(r.median_price)}<br>"
              f"yoy {pct(r.yoy)} · gross yield {pct(r.gross_yield)}<br>valuation error {pct(r.valuation_error)}"
              for r in ok.itertuples()], hoverinfo="text"))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                      map=dict(style="carto-positron", zoom=10.5,
                               center=dict(lat=float(cs.latitude.median()), lon=float(cs.longitude.median()))),
                      legend=dict(y=0.02, x=0.02, bgcolor="rgba(255,255,255,0.85)"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each circle is a postcode sector (e.g. LS6 1), placed at the average location of its sales and "
               "sized by sales count. Valuation error = median absolute % error on the latest six months, out of sample.")
with right:
    st.subheader("Neighbourhood table")
    st.dataframe(g.sort_values("n", ascending=False)[["postcode_sector", "n", "median_price", "yoy", "gross_yield",
                                                      "valuation_error", "reliable"]],
                 hide_index=True, width="stretch", height=520, column_config={
                     "postcode_sector": "Sector", "n": st.column_config.NumberColumn("Sales", format="%d"),
                     "median_price": st.column_config.NumberColumn("Median", format="£%,.0f"),
                     "yoy": st.column_config.NumberColumn("YoY", format="percent"),
                     "gross_yield": st.column_config.NumberColumn("Gross yield", format="percent"),
                     "valuation_error": st.column_config.NumberColumn("Val. error", format="percent"),
                     "reliable": st.column_config.CheckboxColumn(f"{min_n}+ sales")})

st.markdown('<div class="warn"><b>Quality caveats.</b> Median prices mix property types and sizes, so a sector that sold '
            'more flats this year can look cheaper without any real price change. Gross yields use the city rent for '
            'each property type, so they vary only with price, not with local rents. Locations are postcode '
            f'centres (accurate to roughly 100 m); {pct((cs.geo_precision != "postcode").mean())} of {city} sales use '
            'the sector centre because their postcode has been retired.</div>', unsafe_allow_html=True)
footer()
