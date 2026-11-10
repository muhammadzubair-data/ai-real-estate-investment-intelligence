"""Leakage-safe valuation features.

The rule: a sale completed in month *m* may only use information that a valuer
could have had at the start of month *m*. Land Registry registrations arrive
with a delay, so we add one month of publication lag: features for month *m*
use sales completed in months *m − 13* to *m − 2* (a trailing 12-month window).

Nothing from the sale itself, from later sales, or from the target enters the
features. The sale price only becomes a comparable for sales at least two
months later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

LAG = 2              # months between a sale and its first use as market evidence
WINDOW = 12          # trailing months of evidence
K_COMPS = 10
EARTH_KM = 6371.0

CAT_FEATURES = ["city", "property_type", "tenure"]
NUM_FEATURES = [
    "new_build", "latitude", "longitude", "month_of_year",
    "log_sector_type_median", "log_sector_median", "log_district_type_median", "log_city_type_median",
    "sector_count", "sector_type_count", "log_comp_median", "comp_mean_km", "comp_max_km",
    "log_prev_price_indexed", "months_since_prev", "has_prev_sale", "city_momentum_3m",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

FEATURE_LABELS = {
    "city": "Market (city)", "property_type": "Property type", "tenure": "Tenure", "new_build": "New build",
    "latitude": "Latitude", "longitude": "Longitude", "month_of_year": "Month of year",
    "log_sector_type_median": "Postcode-sector median, same type (12m)",
    "log_sector_median": "Postcode-sector median, all types (12m)",
    "log_district_type_median": "Postcode-district median, same type (12m)",
    "log_city_type_median": "City median, same type (12m)",
    "sector_count": "Sector sales in window", "sector_type_count": "Sector sales of same type",
    "log_comp_median": "Median of 10 nearest comparable sales",
    "comp_mean_km": "Average distance to comparables (km)", "comp_max_km": "Farthest comparable (km)",
    "log_prev_price_indexed": "Previous sale of this property, market-adjusted",
    "months_since_prev": "Months since previous sale", "has_prev_sale": "Has a previous sale",
    "city_momentum_3m": "City price momentum (3m vs 12m)",
}


def month_index(d: pd.Series, start: str = "2024-01") -> pd.Series:
    s = pd.Period(start, "M")
    return d.dt.to_period("M").astype("int64") - s.ordinal


def city_type_index(sales: pd.DataFrame) -> pd.DataFrame:
    """Monthly log-median price by city and type, smoothed over 3 months (for time adjustment)."""
    g = (sales.groupby(["city", "property_type", "m"]).price.median().apply(np.log)
         .rename("log_med").reset_index())
    g = g.sort_values("m")
    g["log_idx"] = g.groupby(["city", "property_type"]).log_med.transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    return g[["city", "property_type", "m", "log_idx"]]


def _window_medians(hist: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    g = hist.groupby(keys).agg(**{f"log_{name}": ("log_price", "median"), f"{name}_n": ("log_price", "size")})
    return g.reset_index()


def prepare(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Add month index, log price and radians; return (sales, city/type log index)."""
    s = sales.copy()
    s["m"] = month_index(s.sale_date)
    s["log_price"] = np.log(s.price)
    s["month_of_year"] = s.sale_date.dt.month
    rad = np.radians(s[["latitude", "longitude"]].to_numpy())
    s["_lat_r"], s["_lon_r"] = rad[:, 0], rad[:, 1]
    idx = city_type_index(s).set_index(["city", "property_type", "m"]).log_idx
    return s, idx


def _index_adjust(h: pd.DataFrame, ref_m: int, idx_map: pd.Series) -> np.ndarray:
    """Bring historical log prices to the reference month with the city/type index."""
    now = idx_map.reindex(list(zip(h.city, h.property_type, [ref_m] * len(h)))).to_numpy()
    then = idx_map.reindex(list(zip(h.city, h.property_type, h.m))).to_numpy()
    d = np.where(np.isnan(now) | np.isnan(then), 0.0, now - then)
    return h.log_price.to_numpy() + d


def month_features(cur: pd.DataFrame, s: pd.DataFrame, m: int, idx_map: pd.Series,
                   k: int = K_COMPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for the rows in ``cur`` valued in month ``m``, using only evidence from ``s``
    completed in months m−LAG−WINDOW+1 … m−LAG. ``cur`` needs: transaction_id, property_id,
    city, property_type, postcode_sector, postcode_district, _lat_r, _lon_r."""
    ref_m = m - LAG
    hist = s[(s.m <= ref_m) & (s.m > ref_m - WINDOW)]
    f = cur[["transaction_id", "property_id", "city", "property_type", "postcode_sector",
             "postcode_district"]].copy()
    f["m"] = m
    comps_all = []
    if hist.empty:
        return f, pd.DataFrame()
    hist = hist.assign(log_price=_index_adjust(hist, ref_m, idx_map))
    for keys, name in ((["postcode_sector", "property_type"], "sector_type_median"),
                       (["postcode_sector"], "sector_median"),
                       (["postcode_district", "property_type"], "district_type_median"),
                       (["city", "property_type"], "city_type_median")):
        f = f.merge(_window_medians(hist, keys, name), on=keys, how="left")
    f = f.rename(columns={"sector_median_n": "sector_count", "sector_type_median_n": "sector_type_count"})
    f = f.drop(columns=[c for c in f.columns if c.endswith("_n")])
    f["sector_count"] = f.sector_count.fillna(0)
    f["sector_type_count"] = f.sector_type_count.fillna(0)

    raw = s[(s.m <= ref_m) & (s.m > ref_m - WINDOW)]
    r3 = raw[raw.m > ref_m - 3].groupby("city").log_price.median()
    r12 = raw.groupby("city").log_price.median()
    f["city_momentum_3m"] = f.city.map(r3 - r12)

    pos = {t: i for i, t in enumerate(f.transaction_id)}
    comp_med = np.full(len(f), np.nan)
    comp_mean = np.full(len(f), np.nan)
    comp_max = np.full(len(f), np.nan)
    for ptype, cq in cur.groupby("property_type"):
        hp = hist[hist.property_type == ptype]
        if len(hp) < k + 3:
            continue
        # Comparables are other properties: the property's own earlier sale is already
        # used through the previous-sale feature.
        tree = BallTree(hp[["_lat_r", "_lon_r"]].to_numpy(), metric="haversine")
        extra = 3
        dist, ind = tree.query(cq[["_lat_r", "_lon_r"]].to_numpy(), k=k + extra)
        same = hp.property_id.to_numpy()[ind] == cq.property_id.to_numpy()[:, None]
        order = np.argsort(same, axis=1, kind="stable")[:, :k]       # non-matching first, nearest first
        dist = np.take_along_axis(dist, order, axis=1)
        ind = np.take_along_axis(ind, order, axis=1)
        dist_km = dist * EARTH_KM
        lp = hp.log_price.to_numpy()[ind]
        rows = [pos[t] for t in cq.transaction_id]
        comp_med[rows] = np.median(lp, axis=1)
        comp_mean[rows] = dist_km.mean(axis=1)
        comp_max[rows] = dist_km.max(axis=1)
        ids = hp.transaction_id.to_numpy()
        n = len(cq)
        comps_all.append(pd.DataFrame({
            "transaction_id": np.repeat(cq.transaction_id.to_numpy(), k),
            "rank": np.tile(np.arange(1, k + 1), n),
            "comp_transaction_id": ids[ind].ravel(),
            "distance_km": dist_km.ravel().round(3),
            "adjusted_price": np.exp(lp).ravel().round(0)}))
    f["log_comp_median"], f["comp_mean_km"], f["comp_max_km"] = comp_med, comp_mean, comp_max

    prev = (s[(s.m <= ref_m) & s.property_id.isin(cur.property_id)]
            .sort_values("sale_date").groupby("property_id").tail(1))
    if len(prev):
        pv = pd.DataFrame({"property_id": prev.property_id,
                           "log_prev_price_indexed": _index_adjust(prev, ref_m, idx_map), "prev_m": prev.m})
        f = f.merge(pv, on="property_id", how="left")
        f["months_since_prev"] = f.m - f.prev_m
        f = f.drop(columns="prev_m")
    else:
        f["log_prev_price_indexed"] = np.nan
        f["months_since_prev"] = np.nan
    f["has_prev_sale"] = f.log_prev_price_indexed.notna().astype(int)
    comps = pd.concat(comps_all, ignore_index=True) if comps_all else pd.DataFrame()
    return f, comps


def build_features(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for every sale (valued in its own month) plus its comparables."""
    s, idx_map = prepare(sales)
    feats, comps = [], []
    for m in sorted(s.m.unique()):
        f, c = month_features(s[s.m == m], s, m, idx_map)
        feats.append(f)
        comps.append(c)
    F = pd.concat(feats, ignore_index=True)
    F["has_prev_sale"] = F.log_prev_price_indexed.notna().astype(int) if "log_prev_price_indexed" in F else 0
    base = s[["transaction_id", "sale_date", "price", "log_price", "tenure", "new_build", "latitude",
              "longitude", "month_of_year", "geo_precision"]]
    F = base.merge(F, on="transaction_id")
    return F, pd.concat(comps, ignore_index=True)
