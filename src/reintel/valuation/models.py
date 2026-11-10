"""Valuation models and honest evaluation.

Models (all predict log price; reported in £):

1. Neighbourhood median   - trailing 12-month median of the same postcode sector
                            and type, falling back to district, then city.
2. Comparable sales       - median of the 10 nearest prior same-type sales,
                            adjusted to today's market with the city/type index.
3. Ridge regression       - regularised linear model on the same features.
4. LightGBM               - gradient boosting (the deployed model), plus
                            quantile models for an 80% prediction interval that
                            is calibrated on validation months (conformalised
                            quantile regression).

Validation:

* temporal holdout - fit on earlier months, tune on the next months, test on the
  latest months nobody has seen;
* geographic holdout - whole postcode districts withheld from training;
* leave-one-city-out - each city valued by a model trained only on the others;
* property grouping - results also reported for properties never seen in
  training, so repeat sales cannot flatter the score.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reintel.valuation.features import CAT_FEATURES, FEATURES, NUM_FEATURES

LGB_PARAMS = {
    "objective": "regression", "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 30,
    "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": 42, "deterministic": True, "force_row_wise": True, "num_threads": 2,
}
INTERVAL = 0.80
PRICE_TIERS = [0, 150_000, 250_000, 400_000, np.inf]
TIER_LABELS = ["< £150k", "£150k–250k", "£250k–400k", "£400k+"]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def neighbourhood_median(X: pd.DataFrame) -> np.ndarray:
    out = X.log_sector_type_median.where(X.sector_type_count >= 3)
    out = out.fillna(X.log_district_type_median).fillna(X.log_city_type_median)
    return out.to_numpy()


def comparable_sales(X: pd.DataFrame) -> np.ndarray:
    return X.log_comp_median.fillna(pd.Series(neighbourhood_median(X), index=X.index)).to_numpy()


def make_ridge():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         NUM_FEATURES),
    ])
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 12)))


# ---------------------------------------------------------------------------
# LightGBM with calibrated intervals
# ---------------------------------------------------------------------------
def to_lgb(X: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    X = X[FEATURES].copy()
    categories = categories or {c: sorted(X[c].dropna().unique()) for c in CAT_FEATURES}
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c].where(X[c].isin(categories[c])), categories=categories[c])
    return X, categories


@dataclass
class Valuer:
    point: lgb.Booster
    lower: lgb.Booster
    upper: lgb.Booster
    categories: dict
    conformal_q: float          # widening (log units) that calibrates the interval on validation
    n_trees: int

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xl, _ = to_lgb(X, self.categories)
        p = self.point.predict(Xl)
        lo = self.lower.predict(Xl) - self.conformal_q
        hi = self.upper.predict(Xl) + self.conformal_q
        lo, hi = np.minimum(lo, p), np.maximum(hi, p)
        return pd.DataFrame({"log_pred": p, "log_lo": lo, "log_hi": hi}, index=X.index)

    def save(self, folder) -> None:
        import json
        from pathlib import Path
        folder = Path(folder)
        self.point.save_model(str(folder / "valuer_point.txt"))
        self.lower.save_model(str(folder / "valuer_lower.txt"))
        self.upper.save_model(str(folder / "valuer_upper.txt"))
        (folder / "valuer_meta.json").write_text(json.dumps(
            {"categories": self.categories, "conformal_q": self.conformal_q, "n_trees": self.n_trees}, indent=2))

    @classmethod
    def load(cls, folder) -> "Valuer":
        import json
        from pathlib import Path
        folder = Path(folder)
        meta = json.loads((folder / "valuer_meta.json").read_text())
        return cls(lgb.Booster(model_file=str(folder / "valuer_point.txt")),
                   lgb.Booster(model_file=str(folder / "valuer_lower.txt")),
                   lgb.Booster(model_file=str(folder / "valuer_upper.txt")),
                   meta["categories"], meta["conformal_q"], meta["n_trees"])


def fit_valuer(train: pd.DataFrame, valid: pd.DataFrame, calibrate: pd.DataFrame | None = None,
               n_trees: int | None = None, intervals: bool = True) -> Valuer:
    """Fit point + quantile models. With ``n_trees`` given, skip early stopping.
    ``intervals=False`` fits the point model only (used for the many leave-one-city-out fits)."""
    Xt, cats = to_lgb(train)
    Xv, _ = to_lgb(valid, cats)
    yt, yv = train.log_price, valid.log_price
    if n_trees is None:
        b = lgb.train(LGB_PARAMS, lgb.Dataset(Xt, yt), 3000, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        n_trees = b.best_iteration or 500
        point = b
    else:
        point = lgb.train(LGB_PARAMS, lgb.Dataset(Xt, yt), n_trees)
    if not intervals:
        return Valuer(point, point, point, cats, 0.0, n_trees)
    a = (1 - INTERVAL) / 2
    q_lo = lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": a}, lgb.Dataset(Xt, yt), n_trees)
    q_hi = lgb.train({**LGB_PARAMS, "objective": "quantile", "alpha": 1 - a}, lgb.Dataset(Xt, yt), n_trees)
    v = Valuer(point, q_lo, q_hi, cats, 0.0, n_trees)
    cal = valid if calibrate is None else calibrate
    pr = v.predict(cal)
    # conformalised quantile regression: widen by the 80th percentile of the miss
    score = np.maximum(pr.log_lo - cal.log_price, cal.log_price - pr.log_hi)
    n = len(score)
    level = min(1.0, np.ceil((n + 1) * INTERVAL) / n)
    v.conformal_q = float(np.quantile(score, level))
    return v


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y_log: np.ndarray, pred_log: np.ndarray, lo_log=None, hi_log=None) -> dict:
    y, p = np.exp(y_log), np.exp(pred_log)
    err = p - y
    ape = np.abs(err) / y
    out = {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_error": float(np.median(np.abs(err))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "bias_median_pct": float(np.median(err / y)),
    }
    if lo_log is not None:
        out["interval_coverage"] = float(np.mean((y_log >= lo_log) & (y_log <= hi_log)))
        out["interval_median_width_pct"] = float(np.median(np.exp(hi_log) / np.exp(lo_log) - 1))
    return out


def slice_metrics(df: pd.DataFrame, pred_col: str, by: str) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(by, observed=True):
        m = metrics(g.log_price.to_numpy(), g[pred_col].to_numpy(),
                    g.get("log_lo", pd.Series(dtype=float)).to_numpy() if "log_lo" in g else None,
                    g.get("log_hi", pd.Series(dtype=float)).to_numpy() if "log_hi" in g else None)
        rows.append({by: k, **m})
    return pd.DataFrame(rows)


def price_tier(price: pd.Series) -> pd.Series:
    return pd.cut(price, PRICE_TIERS, labels=TIER_LABELS, right=False)
