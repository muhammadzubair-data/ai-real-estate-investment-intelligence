"""Value any property today, with comparables, interval, explanation and warnings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from reintel.valuation.features import FEATURE_LABELS, FEATURES, LAG, month_features, prepare
from reintel.valuation.models import Valuer, to_lgb


class ValuationService:
    def __init__(self, sales: pd.DataFrame, valuer: Valuer):
        cols = ["transaction_id", "property_id", "sale_date", "price", "postcode", "postcode_sector",
                "postcode_district", "city", "property_type", "tenure", "new_build", "latitude", "longitude",
                "geo_precision"]
        self.s, self.idx = prepare(sales[cols])
        self.valuer = valuer
        self.last_m = int(self.s.m.max())
        self.explainer = shap.TreeExplainer(valuer.point)
        self.postcodes = (self.s.groupby("postcode")
                          .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                               postcode_sector=("postcode_sector", "first"),
                               postcode_district=("postcode_district", "first"), city=("city", "first")))

    @property
    def as_of(self) -> pd.Timestamp:
        """Latest month of evidence used for a valuation 'today'."""
        return self.s.sale_date.max().to_period("M").to_timestamp()

    def value(self, postcode: str, property_type: str, tenure: str, new_build: bool,
              property_id: str | None = None) -> dict:
        pc = postcode.upper().replace(" ", "")
        if pc not in self.postcodes.index:
            raise KeyError(f"Postcode {postcode} has no recorded sales in the covered markets")
        loc = self.postcodes.loc[pc]
        m = self.last_m + LAG                 # value now: all evidence up to the latest month
        q = pd.DataFrame([{
            "transaction_id": "QUERY", "property_id": property_id or "QUERY", "city": loc.city,
            "property_type": property_type, "postcode_sector": loc.postcode_sector,
            "postcode_district": loc.postcode_district,
            "_lat_r": np.radians(loc.latitude), "_lon_r": np.radians(loc.longitude)}])
        f, comps = month_features(q, self.s, m, self.idx)
        today = self.as_of + pd.offsets.MonthBegin(1)
        f = f.assign(tenure=tenure, new_build=int(new_build), latitude=loc.latitude, longitude=loc.longitude,
                     month_of_year=today.month)
        pr = self.valuer.predict(f).iloc[0]
        Xl, _ = to_lgb(f, self.valuer.categories)
        sv = self.explainer.shap_values(Xl)[0]
        base = float(np.ravel(self.explainer.expected_value)[0])
        drivers = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_LABELS[c] for c in FEATURES],
                                "value": [f.iloc[0][c] for c in FEATURES], "shap": sv})
        drivers = drivers.reindex(drivers.shap.abs().sort_values(ascending=False).index)

        if not comps.empty:
            comps = comps.merge(self.s[["transaction_id", "sale_date", "price", "postcode", "property_type",
                                        "tenure", "new_build", "latitude", "longitude"]],
                                left_on="comp_transaction_id", right_on="transaction_id", suffixes=("", "_c"))
            comps = comps.drop(columns=["transaction_id_c"]).sort_values("rank")

        row = f.iloc[0]
        warnings = []
        if comps.empty:
            warnings.append("No comparable sales of this type in the last 12 months: the estimate leans on wider area data.")
        elif row.comp_max_km > 2:
            warnings.append(f"Comparables are spread out (farthest {row.comp_max_km:.1f} km away), so local detail is thin.")
        if row.sector_type_count < 5:
            warnings.append(f"Only {int(row.sector_type_count)} sales of this type in the postcode sector in the last 12 months.")
        if new_build:
            warnings.append("New builds usually sell at a premium that fades on resale; treat the estimate with care.")
        width = np.exp(pr.log_hi) / np.exp(pr.log_lo) - 1
        if width > 0.6:
            warnings.append("The 80% range is wide: the model is uncertain about this property.")
        warnings.append("The model knows location, type, tenure and new-build status but not size, bedrooms or "
                        "condition, which drive a large part of any individual price.")
        return {
            "estimate": float(np.exp(pr.log_pred)), "low": float(np.exp(pr.log_lo)), "high": float(np.exp(pr.log_hi)),
            "features": row, "drivers": drivers, "base_value": float(np.exp(base)), "comparables": comps,
            "warnings": warnings, "as_of": self.as_of,
        }
