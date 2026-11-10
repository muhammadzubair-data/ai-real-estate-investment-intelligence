"""01 · Executive Market Overview."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, badge, city_list, footer, gbp, kpi, load, load_json, pct, sales, setup

setup("Executive Market Overview",
      "Prices, rents, yields and sales volumes across the covered English cities.")
badge()

mkt = load("market_monthly.parquet")
rents = load("rents.parquet")
s = sales()
q = load_json("quality_report.json")
cities = city_list()

# --- latest snapshot per city ---------------------------------------------------------------
last_month = s.sale_date.max().to_period("M").to_timestamp()
win = s[s.sale_date >= last_month - pd.DateOffset(months=11)]
prev = s[(s.sale_date >= last_month - pd.DateOffset(months=23)) & (s.sale_date < last_month - pd.DateOffset(months=11))]
snap = win.groupby("city").agg(median_price=("price", "median"), sales=("price", "size")).reset_index()
snap["prev_median"] = snap.city.map(prev.groupby("city").price.median())
snap["yoy"] = snap.median_price / snap.prev_median - 1
rent_all = rents[rents.property_type == "All"].sort_values("month")
latest_rent = rent_all.groupby("city").tail(1).set_index("city")
snap["rent"] = snap.city.map(latest_rent.rent)
snap["rent_yoy"] = snap.city.map(
    rent_all.groupby("city").apply(lambda g: g.rent.iloc[-1] / g.rent.iloc[-13] - 1 if len(g) >= 13 else np.nan,
                                   include_groups=False))
snap["gross_yield"] = snap.rent * 12 / snap.median_price

c = st.columns(5)
kpi(c[0], "Sales analysed", f"{q['sales_final']:,}", f"{q['properties_final']:,} distinct properties")
kpi(c[1], "Markets", f"{len(cities)}", "English cities (local authorities)")
kpi(c[2], "Median price, last 12 months", gbp(win.price.median()),
    f"{pct(win.price.median() / prev.price.median() - 1)} vs previous 12 months" if len(prev) else "")
kpi(c[3], "Average monthly rent", gbp(snap.rent.mean(), short=False), "simple average of city rents (ONS)")
kpi(c[4], "Median gross yield", pct(snap.gross_yield.median()), "city rent × 12 ÷ city median price")

st.write("")
st.subheader("City comparison")
st.caption("Last 12 months of completed sales against the 12 months before. Rents are ONS local-authority "
           "averages (all property types). Gross yield ignores costs and vacancy.")
tbl = snap.sort_values("gross_yield", ascending=False)[
    ["city", "median_price", "yoy", "rent", "rent_yoy", "gross_yield", "sales"]]
st.dataframe(tbl, hide_index=True, width="stretch", column_config={
    "city": "City", "median_price": st.column_config.NumberColumn("Median price", format="£%,.0f"),
    "yoy": st.column_config.NumberColumn("Price change (yoy)", format="percent"),
    "rent": st.column_config.NumberColumn("Avg rent / month", format="£%,.0f"),
    "rent_yoy": st.column_config.NumberColumn("Rent change (yoy)", format="percent"),
    "gross_yield": st.column_config.ProgressColumn("Gross yield", min_value=0, max_value=0.10, format="percent"),
    "sales": st.column_config.NumberColumn("Sales (12m)", format="%,d"),
})

st.divider()
pick = st.multiselect("Cities to chart", cities, default=cities[:5], max_selections=8)
mode = st.radio("Show", ["£ price", "Indexed (first month = 100)"], horizontal=True)
left, right = st.columns(2)
with left:
    st.subheader("Mix-adjusted price")
    fig = go.Figure()
    for i, city in enumerate(pick):
        g = mkt[(mkt.city == city)].dropna(subset=["mix_adjusted_price"]).sort_values("month")
        y = g.mix_adjusted_price if mode == "£ price" else 100 * g.mix_adjusted_price / g.mix_adjusted_price.iloc[0]
        fig.add_trace(go.Scatter(x=g.month, y=y, name=city, mode="lines", line=dict(color=SERIES[i], width=2),
                                 hovertemplate=city + "<br>%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
    fig.update_layout(height=340, yaxis_title="£" if mode == "£ price" else "Index")
    st.plotly_chart(fig, width="stretch")
    st.caption("Mix-adjusted: per-type medians weighted by each city's usual mix, so a month with more flats "
               "sold does not look like a price fall.")
with right:
    st.subheader("Average monthly rent (ONS)")
    fig = go.Figure()
    for i, city in enumerate(pick):
        g = rent_all[rent_all.city == city]
        y = g.rent if mode == "£ price" else 100 * g.rent / g.rent.iloc[0]
        fig.add_trace(go.Scatter(x=g.month, y=y, name=city, mode="lines", line=dict(color=SERIES[i], width=2),
                                 hovertemplate=city + "<br>%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
    fig.update_layout(height=340, yaxis_title="£ per month" if mode == "£ price" else "Index")
    st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Sales volume")
    vol = mkt[mkt.city.isin(pick)].pivot_table(index="month", columns="city", values="transaction_count").fillna(0)
    fig = go.Figure()
    for i, city in enumerate(pick):
        if city in vol:
            fig.add_trace(go.Bar(x=vol.index, y=vol[city], name=city, marker_color=SERIES[i],
                                 marker_line=dict(color="#fcfcfb", width=1),
                                 hovertemplate=city + "<br>%{x|%b %Y}: %{y:,} sales<extra></extra>"))
    fig.update_layout(barmode="stack", height=320, yaxis_title="Completed sales per month")
    st.plotly_chart(fig, width="stretch")
    st.caption("The most recent months are lower partly because registrations arrive with a delay of weeks to "
               "months. Listings and stock for sale are not in the public data, so inventory is not shown.")
with right:
    st.subheader("What sells: property type mix (last 12 months)")
    mix = win[win.city.isin(pick)].groupby(["city", "property_type"]).size().unstack(fill_value=0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    fig = go.Figure()
    for i, t in enumerate(["Detached", "Semi-detached", "Terraced", "Flat/maisonette"]):
        if t in mix:
            fig.add_trace(go.Bar(y=mix.index, x=mix[t], name=t, orientation="h", marker_color=SERIES[i],
                                 marker_line=dict(color="#fcfcfb", width=2),
                                 hovertemplate="%{y}: %{x:.0%} " + t.lower() + "<extra></extra>"))
    fig.update_layout(barmode="stack", height=320, xaxis=dict(tickformat=".0%", range=[0, 1]))
    st.plotly_chart(fig, width="stretch")

with st.expander("Data quality: what was removed and why"):
    labels = {
        "rows_read_national": "Rows read (all of England & Wales)", "rows_in_markets": "Rows in covered cities",
        "removed_superseded_or_deleted": "Removed: superseded or deleted records",
        "removed_category_b": "Removed: category B (repossessions, company/buy-to-let transfers, non-market)",
        "removed_property_type_other": "Removed: 'other' property type (commercial, land, mixed)",
        "removed_price_or_date_out_of_range": "Removed: price under £10k or over £10m, or bad date",
        "removed_missing_postcode": "Removed: no postcode", "removed_duplicate_sales": "Removed: duplicate sales",
        "removed_incomplete_last_month": "Removed: latest month (registrations still incomplete)",
        "sales_geocoded_to_sector_centroid": "Placed at postcode-sector centre (postcode retired)",
        "removed_not_geocoded": "Removed: could not be located", "sales_final": "Sales kept",
    }
    st.dataframe(pd.DataFrame([(v, q[k]) for k, v in labels.items() if k in q], columns=["Step", "Rows"]),
                 hide_index=True, width="stretch", column_config={"Rows": st.column_config.NumberColumn(format="%,d")})
    st.caption("Source files: " + ", ".join(q.get("source_files", [])))
footer()
