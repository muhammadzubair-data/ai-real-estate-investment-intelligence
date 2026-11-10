import json

import numpy as np
import pandas as pd
import pytest

from reintel.data.ingest import postcode_parts
from reintel.valuation.features import FEATURES, LAG, WINDOW, build_features, month_index
from reintel.valuation.models import Valuer, metrics


def test_postcode_parts():
    p = postcode_parts(pd.Series(["LS6 1AB", "b11aa", "SW1A 1AA"]))
    assert p.postcode_district.tolist() == ["LS6", "B1", "SW1A"]
    assert p.postcode_sector.tolist() == ["LS6 1", "B1 1", "SW1A 1"]


def test_validation_removes_bad_rows(fixture_build):
    _, out, _ = fixture_build
    q = json.loads((out / "quality_report.json").read_text())
    assert q["removed_category_b"] == 1
    assert q["removed_property_type_other"] == 1
    assert q["removed_price_or_date_out_of_range"] == 1
    assert q["removed_duplicate_sales"] == 1
    assert q["rows_in_markets"] == q["rows_read_national"] - 1          # out-of-market row dropped


def test_addresses_are_not_published(fixture_build):
    _, out, _ = fixture_build
    s = pd.read_parquet(out / "sales.parquet")
    assert not {"paon", "saon", "street"} & set(s.columns)
    assert s.property_id.str.match(r"^P[0-9a-f]{11}$").all()


def test_features_use_only_past_evidence(fixture_build):
    """Recompute features after deleting every sale from the valuation month onward:
    the features of that month must not change (no look-ahead)."""
    _, out, _ = fixture_build
    s = pd.read_parquet(out / "sales.parquet")
    base = s[["transaction_id", "property_id", "sale_date", "price", "postcode", "postcode_sector",
              "postcode_district", "city", "property_type", "tenure", "new_build", "latitude", "longitude",
              "geo_precision"]]
    m = month_index(base.sale_date)
    target_m = int(m.max()) - 1
    F_all, _ = build_features(base)
    # keep the target month itself (its rows are being valued) but drop everything after it,
    # and corrupt the target month's own prices: features must not move
    cut = base[m <= target_m].copy()
    cut["price"] = cut.price.astype(float)
    cut.loc[month_index(cut.sale_date) == target_m, "price"] *= 7.3
    F_cut, _ = build_features(cut)
    num = [c for c in FEATURES if c not in ("city", "property_type", "tenure")]
    a = F_all[F_all.m == target_m].set_index("transaction_id")[num].sort_index()
    b = F_cut[F_cut.m == target_m].set_index("transaction_id")[num].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_lag_and_window_constants():
    assert LAG >= 1 and WINDOW >= 6


def test_splits_are_time_ordered(fixture_build):
    _, out, _ = fixture_build
    s = pd.read_parquet(out / "sales.parquet")
    last = {k: s[s.split == k].sale_date.max() for k in ("warmup", "train", "valid")}
    first = {k: s[s.split == k].sale_date.min() for k in ("train", "valid", "test")}
    assert last["warmup"] < first["train"] and last["train"] < first["valid"] and last["valid"] < first["test"]


def test_metrics_report_every_baseline_and_holdout(fixture_build):
    _, out, _ = fixture_build
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["temporal_holdout"]) == {"Neighbourhood median", "Comparable sales (10 nearest)",
                                          "Ridge regression", "LightGBM (deployed)"}
    lgb_ = m["temporal_holdout"]["LightGBM (deployed)"]
    assert 0 < lgb_["median_ape"] < 1 and 0.5 < lgb_["interval_coverage"] <= 1
    assert m["geographic_holdout"]["held_out_test_sales"] > 0
    assert {r["city"] for r in m["leave_one_city_out"]} == {"Leeds", "Bristol"}
    assert m["rows"]["test_unseen_properties"] <= m["rows"]["test"]


def test_metrics_function_known_values():
    y = np.log(np.array([100.0, 200.0]))
    p = np.log(np.array([110.0, 180.0]))
    r = metrics(y, p, p - 0.2, p + 0.2)
    assert r["mae"] == pytest.approx(15)
    assert r["median_ape"] == pytest.approx(0.10)
    assert r["within_10pct"] == 1.0 and r["interval_coverage"] == 1.0


def test_saved_model_round_trip_and_valuation_service(fixture_build):
    _, out, _ = fixture_build
    from reintel.valuation.service import ValuationService
    s = pd.read_parquet(out / "sales.parquet")
    v = Valuer.load(out)
    svc = ValuationService(s, v)
    pc = s.postcode.iloc[-1]
    r = svc.value(pc, "Terraced", "Freehold", False)
    assert r["low"] <= r["estimate"] <= r["high"]
    assert len(r["comparables"]) == 10 and r["warnings"]
    # comparables never include the property being valued
    own = s[s.postcode == pc].property_id.iloc[0]
    r2 = svc.value(pc, s[s.postcode == pc].property_type.iloc[0], "Freehold", False, property_id=own)
    comp_props = set(s.set_index("transaction_id").loc[r2["comparables"].comp_transaction_id, "property_id"])
    assert own not in comp_props
    with pytest.raises(KeyError):
        svc.value("ZZ99 9ZZ", "Terraced", "Freehold", False)


def test_forecast_and_market_tables(fixture_build):
    _, out, _ = fixture_build
    fc = pd.read_parquet(out / "forecasts.parquet")
    assert set(fc.series) == {"price", "rent"}
    assert (fc.lower <= fc.forecast).all() and (fc.forecast <= fc.upper).all()
    mm = pd.read_parquet(out / "market_monthly.parquet")
    schema = ["city", "month", "median_sale_price", "median_rent", "transaction_count", "mortgage_rate_annual",
              "currency", "source_id"]
    assert list(mm.columns[:8]) == schema and (mm.currency == "GBP").all()


def test_schema_exports_match_client_schemas(fixture_build):
    _, out, _ = fixture_build
    props = pd.read_parquet(out / "properties.parquet")
    assert list(props.columns) == ["property_id", "city", "neighborhood", "latitude", "longitude", "property_type",
                                   "area_sqft", "bedrooms", "bathrooms", "year_built", "listing_date",
                                   "asking_price", "expected_monthly_rent", "currency", "source_id"]
    tx = pd.read_parquet(out / "transactions.parquet")
    assert list(tx.columns) == ["transaction_id", "property_id", "sale_date", "sale_price", "currency", "source_id"]
