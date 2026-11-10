"""End-to-end build: raw public files -> validated data -> models -> app artifacts.

    python -m reintel.build --raw data/raw --out data/demo

Raw inputs (see docs/data_provenance.md for download links):
    data/raw/pp-*.csv                 HM Land Registry Price Paid Data
    data/raw/codepo_gb.zip            OS Code-Point Open
    data/raw/*private*rents*.xlsx     ONS Price Index of Private Rents (or rents_normalised.csv)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from reintel import __version__
from reintel.config import ATTRIBUTION, CURRENCY, DEMO_DIR, MARKETS, RAW_DIR
from reintel.data.ingest import attach_coordinates, load_codepoint, load_price_paid
from reintel.data.rents import load_rents
from reintel.market.forecast import forecast_city, mix_adjusted_index, risk_metrics
from reintel.valuation import models as M
from reintel.valuation.features import FEATURE_LABELS, FEATURES, LAG, build_features, month_index

WARMUP_MONTHS = LAG + 3      # need at least 3 months of evidence before a sale is modelled
VALID_MONTHS = 3
TEST_MONTHS = 6
GEO_HOLDOUT_SHARE = 0.20


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def find_sources(raw: Path) -> dict:
    pp = sorted(raw.glob("pp-*.csv"))
    cp = sorted(raw.glob("codepo*.zip"))
    rents = sorted(raw.glob("*rents*.xlsx")) or sorted(raw.glob("rents_normalised.csv"))
    missing = [n for n, v in (("Land Registry pp-*.csv", pp), ("Code-Point Open zip", cp),
                              ("ONS rents workbook", rents)) if not v]
    if missing:
        raise FileNotFoundError(f"Missing in {raw}: {', '.join(missing)}. See docs/data_provenance.md.")
    return {"price_paid": pp, "codepoint": cp[0], "rents": rents[0]}


def geo_holdout_districts(districts: pd.Series, share: float) -> set[str]:
    """Deterministic ~20% of postcode districts (hash-based, independent of row order)."""
    u = sorted(districts.unique())
    h = {d: int(hashlib.md5(d.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for d in u}
    return {d for d in u if h[d] < share}


def build(raw: Path = RAW_DIR, out: Path = DEMO_DIR, sources: dict | None = None,
          data_label: str = "Real HM Land Registry sales") -> dict:
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = sources or find_sources(Path(raw))
    qlog: dict = {}

    # 1. ingest ---------------------------------------------------------------------------
    sales = load_price_paid(src["price_paid"], qlog)
    cp = load_codepoint(src["codepoint"], set(sales.postcode), qlog)
    sales = attach_coordinates(sales, cp, qlog)
    rents = load_rents(src["rents"])
    rents = rents[rents.city.isin(set(MARKETS.values()))]
    qlog["sales_final"] = int(len(sales))
    qlog["properties_final"] = int(sales.property_id.nunique())
    qlog["markets"] = int(sales.city.nunique())
    qlog["date_range"] = [str(sales.sale_date.min().date()), str(sales.sale_date.max().date())]
    log(f"{len(sales):,} validated sales, {sales.property_id.nunique():,} properties, {sales.city.nunique()} markets", t0)

    # Drop a trailing month that is clearly incomplete (registrations still arriving)
    sales["m"] = month_index(sales.sale_date)
    per_m = sales.groupby("m").size()
    last = per_m.index.max()
    if per_m[last] < 0.5 * per_m.loc[last - 3:last - 1].mean():
        qlog["removed_incomplete_last_month"] = int(per_m[last])
        sales = sales[sales.m < last]
        last -= 1

    # 2. features --------------------------------------------------------------------------
    F, comps = build_features(sales.drop(columns="m"))
    F = F.merge(sales[["transaction_id", "property_id", "postcode", "postcode_sector", "postcode_district",
                       "city", "property_type"]], on="transaction_id", suffixes=("", "_s"))
    F = F.drop(columns=[c for c in F.columns if c.endswith("_s")])
    log("features and comparables built", t0)

    # 3. splits -----------------------------------------------------------------------------
    test_start = last - TEST_MONTHS + 1
    valid_start = test_start - VALID_MONTHS
    F["split"] = np.select([F.m < WARMUP_MONTHS, F.m < valid_start, F.m < test_start],
                           ["warmup", "train", "valid"], default="test")
    tr, va, te = (F[F.split == s] for s in ("train", "valid", "test"))

    # 4. models -----------------------------------------------------------------------------
    first = M.fit_valuer(tr, va)                                  # early stopping + calibration
    final = M.fit_valuer(pd.concat([tr, va]), va, n_trees=first.n_trees)
    final.conformal_q = first.conformal_q                          # calibration from held-out months
    ridge = M.make_ridge().fit(pd.concat([tr, va])[FEATURES], pd.concat([tr, va]).log_price)

    te = te.copy()
    pr = final.predict(te)
    te[["log_pred", "log_lo", "log_hi"]] = pr.to_numpy()
    te["log_nbhd"] = M.neighbourhood_median(te)
    te["log_comps"] = M.comparable_sales(te)
    te["log_ridge"] = ridge.predict(te[FEATURES])
    seen = set(pd.concat([tr, va]).property_id)
    te["seen_property"] = te.property_id.isin(seen)
    log(f"models trained ({first.n_trees} trees, conformal widening {first.conformal_q:.3f})", t0)

    y = te.log_price.to_numpy()
    results = {
        "Neighbourhood median": M.metrics(y, te.log_nbhd.to_numpy()),
        "Comparable sales (10 nearest)": M.metrics(y, te.log_comps.to_numpy()),
        "Ridge regression": M.metrics(y, te.log_ridge.to_numpy()),
        "LightGBM (deployed)": M.metrics(y, te.log_pred.to_numpy(), te.log_lo.to_numpy(), te.log_hi.to_numpy()),
    }
    unseen = te[~te.seen_property]
    results_unseen = M.metrics(unseen.log_price.to_numpy(), unseen.log_pred.to_numpy(),
                               unseen.log_lo.to_numpy(), unseen.log_hi.to_numpy())

    # 5. geographic holdouts ---------------------------------------------------------------
    hold = geo_holdout_districts(F.postcode_district, GEO_HOLDOUT_SHARE)
    trva = pd.concat([tr, va])
    g_model = M.fit_valuer(trva[~trva.postcode_district.isin(hold)], va[~va.postcode_district.isin(hold)],
                           n_trees=first.n_trees)
    g_model.conformal_q = first.conformal_q
    gt = te[te.postcode_district.isin(hold)]
    gp = g_model.predict(gt)
    geo = {
        "held_out_districts": len(hold), "districts_total": int(F.postcode_district.nunique()),
        "held_out_test_sales": int(len(gt)),
        "held_out": M.metrics(gt.log_price.to_numpy(), gp.log_pred.to_numpy(), gp.log_lo.to_numpy(),
                              gp.log_hi.to_numpy()),
        "same_sales_full_model": M.metrics(gt.log_price.to_numpy(), gt.log_pred.to_numpy(),
                                           gt.log_lo.to_numpy(), gt.log_hi.to_numpy()),
    }
    loco = []
    for city in sorted(F.city.unique()):
        trn = trva[trva.city != city]
        ct = te[te.city == city]
        if len(ct) < 30:
            continue
        m_ = M.fit_valuer(trn, va[va.city != city], n_trees=first.n_trees, intervals=False)
        p_ = m_.predict(ct)
        loco.append({"city": city, **M.metrics(ct.log_price.to_numpy(), p_.log_pred.to_numpy()),
                     "in_sample_city_mae": M.metrics(ct.log_price.to_numpy(), ct.log_pred.to_numpy())["mae"]})
    log(f"geographic holdouts done ({len(hold)} districts held out, {len(loco)} cities left out in turn)", t0)

    # 6. slices -----------------------------------------------------------------------------
    te["price_tier"] = M.price_tier(te.price)
    te["comps_quality"] = np.where(te.comp_max_km.isna(), "no comparables",
                                   np.where(te.comp_max_km > 2, "comparables > 2 km away", "comparables within 2 km"))
    slices = {by: M.slice_metrics(te, "log_pred", by).to_dict(orient="records")
              for by in ("city", "property_type", "price_tier", "new_build", "comps_quality", "geo_precision")}

    # 7. explainability ---------------------------------------------------------------------
    Xs, _ = M.to_lgb(te.sample(min(4000, len(te)), random_state=0), final.categories)
    sv = shap.TreeExplainer(final.point).shap_values(Xs)
    gimp = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_LABELS[f] for f in FEATURES],
                         "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)

    # 8. market tables and forecasts --------------------------------------------------------
    mkt = mix_adjusted_index(sales)
    rent_all = rents[rents.property_type == "All"].rename(columns={"rent": "median_rent"})
    mkt = mkt.merge(rent_all[["city", "month", "median_rent"]], on=["city", "month"], how="left")
    mkt["gross_yield"] = mkt.median_rent * 12 / mkt.mix_adjusted_price
    fc, risk = [], []
    for city, g in mkt.groupby("city"):
        ser = g.set_index("month").mix_adjusted_price
        try:
            f = forecast_city(ser, h=6)
            fc.append(f["forecast"].assign(city=city, series="price",
                                           beats_naive=f["beats_naive"]))
            risk.append({"city": city, "series": "price", **risk_metrics(ser),
                         "backtest_mae_1m": f["backtest_mae_log_model"][0],
                         "naive_mae_1m": f["backtest_mae_log_naive"][0], "beats_naive": f["beats_naive"],
                         **{f"param_{k}": v for k, v in f["params"].items()}})
        except ValueError:
            pass
        rs = rent_all[rent_all.city == city].set_index("month").median_rent.sort_index()
        if len(rs) >= 15:
            f = forecast_city(rs, h=12)
            fc.append(f["forecast"].assign(city=city, series="rent", beats_naive=f["beats_naive"]))
            risk.append({"city": city, "series": "rent", **risk_metrics(rs),
                         "backtest_mae_1m": f["backtest_mae_log_model"][0],
                         "naive_mae_1m": f["backtest_mae_log_naive"][0], "beats_naive": f["beats_naive"]})
    fc = pd.concat(fc, ignore_index=True)
    risk = pd.DataFrame(risk)
    log("market indices and forecasts built", t0)

    # 9. property universe and app tables --------------------------------------------------
    latest_rent = (rents.sort_values("month").groupby(["city", "property_type"]).tail(1)
                   .set_index(["city", "property_type"]).rent)
    all_pred = final.predict(F[F.split != "warmup"])
    F.loc[all_pred.index, ["log_pred", "log_lo", "log_hi"]] = all_pred.to_numpy()
    F["area_rent"] = [latest_rent.get((c, t), latest_rent.get((c, "All"), np.nan))
                      for c, t in zip(F.city, F.property_type)]
    sales_out = F[["transaction_id", "property_id", "sale_date", "price", "postcode", "postcode_sector",
                   "postcode_district", "city", "property_type", "tenure", "new_build", "latitude", "longitude",
                   "geo_precision", "split", "m", "area_rent", "log_pred", "log_lo", "log_hi",
                   *[c for c in FEATURES if c not in ("city", "property_type", "tenure", "new_build",
                                                      "latitude", "longitude")]]]
    sales_out = sales_out.sort_values(["sale_date", "transaction_id"]).reset_index(drop=True)
    comps = comps[comps.transaction_id.isin(set(F.loc[F.split == "test", "transaction_id"]))]

    # schema-compliant exports (columns from schemas/*.csv in the client package)
    props = (sales.sort_values("sale_date").groupby("property_id").tail(1)
             .assign(neighborhood=lambda d: d.postcode_sector, area_sqft=np.nan, bedrooms=np.nan,
                     bathrooms=np.nan, year_built=np.nan, listing_date=pd.NaT, asking_price=np.nan,
                     currency=CURRENCY, source_id="HMLR-PPD;OS-CPO;ONS-PIPR"))
    props["expected_monthly_rent"] = [latest_rent.get((c, t), np.nan) for c, t in zip(props.city, props.property_type)]
    props = props[["property_id", "city", "neighborhood", "latitude", "longitude", "property_type", "area_sqft",
                   "bedrooms", "bathrooms", "year_built", "listing_date", "asking_price", "expected_monthly_rent",
                   "currency", "source_id"]]
    txs = sales[["transaction_id", "property_id", "sale_date", "price"]].rename(columns={"price": "sale_price"})
    txs = txs.assign(currency=CURRENCY, source_id="HMLR-PPD")
    mm = mkt.rename(columns={"median_price": "median_sale_price", "sales": "transaction_count"})
    mm = mm.assign(mortgage_rate_annual=np.nan, currency=CURRENCY, source_id="HMLR-PPD;ONS-PIPR")
    mm = mm[["city", "month", "median_sale_price", "median_rent", "transaction_count", "mortgage_rate_annual",
             "currency", "source_id", "mix_adjusted_price", "gross_yield"]]

    # 10. write ----------------------------------------------------------------------------
    sales_out.to_parquet(out / "sales.parquet", index=False, compression="zstd")
    comps.to_parquet(out / "comparables.parquet", index=False, compression="zstd")
    props.to_parquet(out / "properties.parquet", index=False, compression="zstd")
    txs.to_parquet(out / "transactions.parquet", index=False, compression="zstd")
    mm.to_parquet(out / "market_monthly.parquet", index=False)
    rents.to_parquet(out / "rents.parquet", index=False)
    fc.to_parquet(out / "forecasts.parquet", index=False)
    risk.to_csv(out / "market_risk.csv", index=False)
    gimp.to_csv(out / "shap_global.csv", index=False)
    final.save(out)
    metrics = {
        "currency": CURRENCY,
        "periods": {
            "train": [str(tr.sale_date.min().date()), str(tr.sale_date.max().date())],
            "valid": [str(va.sale_date.min().date()), str(va.sale_date.max().date())],
            "test": [str(te.sale_date.min().date()), str(te.sale_date.max().date())],
        },
        "rows": {"train": int(len(tr)), "valid": int(len(va)), "test": int(len(te)),
                 "test_unseen_properties": int(len(unseen))},
        "temporal_holdout": results, "unseen_properties": results_unseen,
        "geographic_holdout": geo, "leave_one_city_out": loco, "slices": slices,
        "interval_level": M.INTERVAL, "conformal_widening_log": first.conformal_q, "n_trees": first.n_trees,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "quality_report.json").write_text(json.dumps(qlog, indent=2, default=str))
    manifest = {
        "project": "AI Real Estate Investment & Property Valuation Intelligence",
        "package_version": __version__, "built_on": str(date.today()), "python": platform.python_version(),
        "markets": sorted(sales.city.unique().tolist()),
        "counts": {"sales": int(len(sales)), "properties": int(sales.property_id.nunique()),
                   "markets": int(sales.city.nunique()), "postcode_sectors": int(sales.postcode_sector.nunique())},
        "date_range": qlog["date_range"], "currency": CURRENCY, "attribution": ATTRIBUTION,
        "data_label": data_label,
        "build_seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"artifacts written to {out}", t0)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(DEMO_DIR))
    a = ap.parse_args()
    print(json.dumps(build(Path(a.raw), Path(a.out))["counts"], indent=2))


if __name__ == "__main__":
    main()
