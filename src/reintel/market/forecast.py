"""Market indices, rolling forecasts and downside risk.

House-price index: a **mix-adjusted** monthly median. Raw medians jump when the
mix of flats and detached houses sold changes from month to month, so each
city's index is the weighted average of per-type medians, with weights fixed at
the city's overall type mix.

Forecasts: damped-trend exponential smoothing (Holt) on the log index, with
parameters chosen by rolling-origin backtest. Prediction intervals come from
the backtest's own errors at each horizon, so they reflect how wrong the method
actually was on this data, not an assumed distribution. A naive "no change"
forecast is always reported next to it: if the model cannot beat it, the
dashboard says so.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def mix_adjusted_index(sales: pd.DataFrame, min_sales: int = 15) -> pd.DataFrame:
    s = sales.assign(month=sales.sale_date.dt.to_period("M").dt.to_timestamp())
    w = s.groupby(["city", "property_type"]).size()
    w = (w / w.groupby(level=0).transform("sum")).rename("w").reset_index()
    med = s.groupby(["city", "property_type", "month"]).agg(med=("price", "median"), n=("price", "size")).reset_index()
    med = med[med.n >= min_sales].merge(w, on=["city", "property_type"])
    med["lw"] = med.w * np.log(med.med)
    idx = med.groupby(["city", "month"]).agg(lw=("lw", "sum"), w=("w", "sum"), n=("n", "sum")).reset_index()
    idx = idx[idx.w > 0.6]                         # need most of the mix observed that month
    idx["mix_adjusted_price"] = np.exp(idx.lw / idx.w)
    raw = s.groupby(["city", "month"]).price.agg(median_price="median", sales="size").reset_index()
    out = raw.merge(idx[["city", "month", "mix_adjusted_price"]], on=["city", "month"], how="left")
    return out.sort_values(["city", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def holt_damped(y: np.ndarray, alpha: float, beta: float, phi: float, h: int) -> np.ndarray:
    level, trend = y[0], (y[min(3, len(y) - 1)] - y[0]) / max(min(3, len(y) - 1), 1)
    for v in y[1:]:
        prev = level
        level = alpha * v + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1 - beta) * phi * trend
    steps = np.cumsum(phi ** np.arange(1, h + 1))
    return level + steps * trend


GRID = list(itertools.product([0.2, 0.4, 0.6, 0.8], [0.05, 0.15, 0.3], [0.8, 0.9, 0.98]))


def backtest(y: np.ndarray, params, h: int, min_train: int = 12) -> np.ndarray:
    """Errors (actual − forecast) by horizon from rolling origins. Shape (origins, h)."""
    errs = []
    for t in range(min_train, len(y) - 1):
        f = holt_damped(y[:t], *params, h) if params != "naive" else np.repeat(y[t - 1], h)
        a = y[t:t + h]
        e = np.full(h, np.nan)
        e[:len(a)] = a - f[:len(a)]
        errs.append(e)
    return np.array(errs) if errs else np.full((0, h), np.nan)


def forecast_city(series: pd.Series, h: int = 6, level: float = 0.80) -> dict:
    """Forecast a monthly price series ``h`` months ahead with backtested intervals."""
    s = series.dropna()
    y = np.log(s.to_numpy())
    if len(y) < 15:
        raise ValueError("Need at least 15 months of history to forecast")
    best, best_mae = None, np.inf
    for p in GRID:
        e = backtest(y, p, 3)
        mae = np.nanmean(np.abs(e)) if e.size else np.inf
        if mae < best_mae:
            best, best_mae = p, mae
    e_model = backtest(y, best, h)
    e_naive = backtest(y, "naive", h)
    f = holt_damped(y, *best, h)
    a = (1 - level) / 2
    lo_q = np.array([np.nanquantile(e_model[:, i], a) if np.isfinite(e_model[:, i]).sum() >= 4 else np.nan
                     for i in range(h)])
    hi_q = np.array([np.nanquantile(e_model[:, i], 1 - a) if np.isfinite(e_model[:, i]).sum() >= 4 else np.nan
                     for i in range(h)])
    # Where long horizons have too few backtest points, grow the 1-step band with √h
    base = np.nanmax(np.abs([lo_q[0], hi_q[0]]))
    for i in range(h):
        if np.isnan(lo_q[i]):
            lo_q[i], hi_q[i] = -base * np.sqrt(i + 1), base * np.sqrt(i + 1)
    # the band must contain the point forecast even when past errors were one-sided
    lo_q, hi_q = np.minimum(lo_q, 0.0), np.maximum(hi_q, 0.0)
    months = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    mae_by_h = lambda e: [float(np.nanmean(np.abs(e[:, i]))) if np.isfinite(e[:, i]).any() else None
                          for i in range(h)]
    return {
        "params": {"alpha": best[0], "beta": best[1], "phi": best[2]},
        "forecast": pd.DataFrame({"month": months, "forecast": np.exp(f), "lower": np.exp(f + lo_q),
                                  "upper": np.exp(f + hi_q)}),
        "backtest_mae_log_model": mae_by_h(e_model),
        "backtest_mae_log_naive": mae_by_h(e_naive),
        "beats_naive": bool(np.nanmean(np.abs(e_model)) < np.nanmean(np.abs(e_naive))),
    }


def risk_metrics(series: pd.Series) -> dict:
    s = series.dropna()
    r = np.log(s).diff().dropna()
    peak = s.cummax()
    dd = s / peak - 1
    yoy = s.iloc[-1] / s.iloc[-13] - 1 if len(s) >= 13 else np.nan
    return {
        "latest": float(s.iloc[-1]),
        "yoy_change": float(yoy),
        "monthly_volatility": float(r.std()),
        "annualised_volatility": float(r.std() * np.sqrt(12)),
        "max_drawdown": float(dd.min()),
        "worst_3m_change": float((s / s.shift(3) - 1).min()),
    }
