import numpy as np
import pandas as pd
import pytest

from reintel.market.forecast import forecast_city, holt_damped, mix_adjusted_index, risk_metrics


def test_holt_on_straight_line_extrapolates():
    y = np.log(100 * 1.01 ** np.arange(30))
    f = holt_damped(y, 0.8, 0.3, 0.999, 3)
    assert np.allclose(np.exp(f), 100 * 1.01 ** np.arange(30, 33), rtol=0.01)


def test_forecast_intervals_and_naive_comparison():
    idx = pd.date_range("2023-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(1)
    s = pd.Series(200_000 * np.exp(np.cumsum(rng.normal(0.004, 0.01, 30))), index=idx)
    r = forecast_city(s, h=6)
    f = r["forecast"]
    assert len(f) == 6 and (f.lower < f.forecast).all() and (f.forecast < f.upper).all()
    assert f.month.iloc[0] == pd.Timestamp("2025-07-01")
    assert isinstance(r["beats_naive"], bool)
    with pytest.raises(ValueError):
        forecast_city(s.iloc[:10])


def test_mix_adjustment_ignores_change_in_mix():
    rows = []
    for m in pd.date_range("2024-01-01", periods=6, freq="MS"):
        n_flat = 40 if m.month % 2 else 10          # flats share swings month to month
        rows += [("X", "Flat/maisonette", m, 100_000)] * n_flat + [("X", "Detached", m, 400_000)] * 30
    df = pd.DataFrame(rows, columns=["city", "property_type", "sale_date", "price"])
    idx = mix_adjusted_index(df)
    assert idx.median_price.nunique() > 1            # raw median jumps with the mix
    assert idx.mix_adjusted_price.round(6).nunique() == 1


def test_risk_metrics():
    s = pd.Series([100, 110, 99, 105, 120, 108, 100, 101, 103, 104, 106, 107, 110.0])
    r = risk_metrics(s)
    assert r["max_drawdown"] == pytest.approx(100 / 120 - 1)
    assert r["yoy_change"] == pytest.approx(0.10)
